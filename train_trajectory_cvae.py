from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import FeatureExtractor, TrajectoryCVAE, get_resnet18_backbone, trajectory_cvae_loss

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "cifar10n": {
        "num_classes": 10,
        "default_label_key": "aggre_label",
        "default_artifacts_dir": "./artifacts",
        "softmax_file": "softmax_history_cifar10n.npy",
        "backbone_file": "resnet18_backbone.pth",
        "trajectory_cvae_file": "trajectory_cvae.pth",
        "features_file": "X_features_cifar10n.npy",
        "targets_file": "trajectory_targets_cifar10n_softmax.npy",
        "setup_fn": setup_cifar10n,
    },
    "cifar100n": {
        "num_classes": 100,
        "default_label_key": "noisy_label",
        "default_artifacts_dir": "./artifacts_cifar100n",
        "softmax_file": "softmax_history_cifar100n.npy",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "trajectory_cvae_file": "trajectory_cvae.pth",
        "features_file": "X_features_cifar100n.npy",
        "targets_file": "trajectory_targets_cifar100n_softmax.npy",
        "setup_fn": setup_cifar100n,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conditional VAE to generate full prediction trajectories from frozen embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10n", "cifar100n"),
        default="cifar100n",
        help="Dataset/noise benchmark to use.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for the trajectory CVAE.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for CVAE training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset directory.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Optional path to local cifar-10-100n repository.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory containing stage-1 outputs and receiving CVAE outputs.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="Optional label key from noise metadata (auto-selected by dataset if omitted).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension used by the CVAE.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension used by the CVAE encoder/decoder.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate inside the CVAE.",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1.0,
        help="KL regularization weight.",
    )
    parser.add_argument(
        "--reconstruction-loss",
        type=str,
        choices=("smooth_l1", "mse"),
        default="smooth_l1",
        help="Reconstruction loss for trajectory regression.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="How often to log detailed generation statistics.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trajectory_targets(softmax_history: np.ndarray) -> np.ndarray:
    if softmax_history.ndim != 3:
        raise ValueError("softmax_history must have shape [T, N, C]")
    return np.transpose(softmax_history, (1, 0, 2)).astype(np.float32)


def extract_features(trainset, backbone_path: Path, num_classes: int, device: torch.device) -> np.ndarray:
    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device)
    backbone.eval()

    extractor = FeatureExtractor(backbone).to(device)
    extractor.eval()

    loader = DataLoader(trainset, batch_size=256, shuffle=False)
    features_list = []

    with torch.no_grad():
        for images, _ in loader:
            embeddings = extractor(images.to(device))
            features_list.append(embeddings.cpu().numpy())

    return np.concatenate(features_list, axis=0).astype(np.float32)


def log_generation_snapshot(
    generator: TrajectoryCVAE,
    features: torch.Tensor,
    epoch: int,
    device: torch.device,
) -> None:
    generator.eval()
    with torch.no_grad():
        generated = generator.generate(features.to(device)).cpu().numpy()
    logging.info(
        "Generation snapshot | epoch=%d | shape=%s | mean=%.6f | std=%.6f | min=%.6f | max=%.6f",
        epoch,
        generated.shape,
        float(generated.mean()),
        float(generated.std()),
        float(generated.min()),
        float(generated.max()),
    )


def train_trajectory_cvae(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    config = DATASET_CONFIG[args.dataset]
    setup_fn = config["setup_fn"]
    artifacts_dir = Path(args.artifacts_dir or config["default_artifacts_dir"])
    label_key = args.label_key or config["default_label_key"]

    softmax_path = artifacts_dir / str(config["softmax_file"])
    backbone_path = artifacts_dir / str(config["backbone_file"])
    features_path = artifacts_dir / str(config["features_file"])

    if not softmax_path.exists() or not backbone_path.exists():
        raise FileNotFoundError(
            "Required artifacts are missing. Expected files: "
            f"{softmax_path}, {backbone_path}"
        )

    trainset, noise_data = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    if label_key not in noise_data:
        raise KeyError(f"Label key '{label_key}' not found. Available keys: {list(noise_data.keys())}")

    noisy_labels = np.asarray(noise_data[label_key], dtype=np.int64)
    softmax_history = np.load(softmax_path)
    trajectory_targets = load_trajectory_targets(softmax_history)

    if trajectory_targets.shape[0] != noisy_labels.shape[0]:
        raise ValueError(
            f"Target count {trajectory_targets.shape[0]} does not match label count {noisy_labels.shape[0]}."
        )

    feature_matrix = extract_features(trainset, backbone_path, config["num_classes"], device)
    if feature_matrix.shape[0] != trajectory_targets.shape[0]:
        raise ValueError(
            f"Feature count {feature_matrix.shape[0]} does not match target count {trajectory_targets.shape[0]}."
        )

    dataset = TensorDataset(torch.from_numpy(feature_matrix), torch.from_numpy(trajectory_targets))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = TrajectoryCVAE(
        input_dim=feature_matrix.shape[1],
        sequence_length=trajectory_targets.shape[1],
        num_classes=trajectory_targets.shape[2],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(generator.parameters(), lr=args.lr)

    logging.info(
        "Trajectory CVAE started: dataset=%s, epochs=%d, samples=%d, sequence_length=%d, num_classes=%d, label_key=%s",
        args.dataset,
        args.epochs,
        trajectory_targets.shape[0],
        trajectory_targets.shape[1],
        trajectory_targets.shape[2],
        label_key,
    )

    probe_features = torch.from_numpy(feature_matrix[: min(8, feature_matrix.shape[0])])
    for epoch in range(args.epochs):
        generator.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = generator(batch_features, batch_targets)
            loss_output = trajectory_cvae_loss(
                output.generated_trajectory,
                batch_targets,
                output.mu,
                output.logvar,
                reconstruction_loss=args.reconstruction_loss,
                kl_weight=args.kl_weight,
            )
            loss_output.total_loss.backward()
            optimizer.step()

            total_loss += float(loss_output.total_loss.item())
            total_recon += float(loss_output.reconstruction_loss.item())
            total_kl += float(loss_output.kl_divergence.item())

        num_batches = max(1, len(loader))
        logging.info(
            "Epoch %d/%d | total=%.6f | recon=%.6f | kl=%.6f",
            epoch + 1,
            args.epochs,
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
        )

        if (epoch + 1) % args.log_interval == 0:
            log_generation_snapshot(generator, probe_features, epoch + 1, device)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cvae_path = artifacts_dir / str(config["trajectory_cvae_file"])
    generated_preview_path = artifacts_dir / f"generated_trajectory_preview_{args.dataset}.npy"
    targets_dump_path = artifacts_dir / str(config["targets_file"])

    torch.save(generator.state_dict(), cvae_path)
    np.save(features_path, feature_matrix)
    np.save(targets_dump_path, trajectory_targets)

    generator.eval()
    with torch.no_grad():
        preview = generator.generate(probe_features.to(device)).cpu().numpy()
    np.save(generated_preview_path, preview)

    logging.info("Trajectory CVAE saved to %s", cvae_path)
    logging.info("Extracted embeddings saved to %s", features_path)
    logging.info("Trajectory preview saved to %s", generated_preview_path)
    logging.info("Trajectory targets saved to %s", targets_dump_path)

    return cvae_path, generated_preview_path, targets_dump_path


def main() -> None:
    configure_logging()
    args = parse_args()
    train_trajectory_cvae(args)


if __name__ == "__main__":
    main()

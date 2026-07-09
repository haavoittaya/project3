from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import AdvancedMLP, FeatureExtractor, get_resnet18_backbone

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "cifar10n": {
        "num_classes": 10,
        "default_label_key": "aggre_label",
        "default_artifacts_dir": "./artifacts",
        "margin_file": "margin_history_cifar10n.npy",
        "softmax_file": "softmax_history_cifar10n.npy",
        "backbone_file": "resnet18_backbone.pth",
        "mlp_file": "mlp_4d_cifar10n.pth",
        "features_file": "X_features.npy",
        "targets_file": "Y_targets_4d.npy",
        "setup_fn": setup_cifar10n,
    },
    "cifar100n": {
        "num_classes": 100,
        "default_label_key": "noisy_label",
        "default_artifacts_dir": "./artifacts_cifar100n",
        "margin_file": "margin_history_cifar100n.npy",
        "softmax_file": "softmax_history_cifar100n.npy",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "mlp_file": "mlp_4d_cifar100n.pth",
        "features_file": "X_features_cifar100n.npy",
        "targets_file": "Y_targets_4d_cifar100n.npy",
        "setup_fn": setup_cifar100n,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: distill training dynamics into a descriptor MLP for CIFAR-10N/CIFAR-100N."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10n", "cifar100n"),
        default="cifar10n",
        help="Dataset/noise benchmark to use.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Stage-2 batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of MLP training epochs.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for MLP optimizer.")
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
        help="Directory containing stage-1 outputs and receiving stage-2 outputs (auto-selected if omitted).",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="Optional label key from noise metadata (auto-selected by dataset if omitted).",
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


def compute_descriptors(
    margin_history: np.ndarray,
    softmax_history: np.ndarray,
    noisy_labels: np.ndarray,
) -> np.ndarray:
    num_samples = softmax_history.shape[1]
    aum_arr = np.mean(margin_history, axis=0)
    mean_conf_arr = np.zeros(num_samples, dtype=np.float32)
    var_arr = np.zeros(num_samples, dtype=np.float32)
    forget_arr = np.zeros(num_samples, dtype=np.float32)

    logging.info("Computing descriptor targets for %d samples", num_samples)
    for i in range(num_samples):
        true_conf = softmax_history[:, i, noisy_labels[i]]
        mean_conf_arr[i] = np.mean(true_conf)
        var_arr[i] = np.std(true_conf)
        predictions = true_conf > 0.5
        forget_arr[i] = np.sum(predictions[:-1] & (~predictions[1:]))

    return np.column_stack((aum_arr, mean_conf_arr, var_arr, forget_arr)).astype(np.float32)


def extract_features(
    trainset,
    backbone_path: Path,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device)
    backbone.eval()

    extractor = FeatureExtractor(backbone).to(device)
    extractor.eval()

    extract_loader = DataLoader(trainset, batch_size=256, shuffle=False)
    features_list = []

    with torch.no_grad():
        for images, _ in extract_loader:
            features = extractor(images.to(device))
            features_list.append(features.cpu().numpy())

    return np.concatenate(features_list, axis=0).astype(np.float32)


def train_stage2(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    config = DATASET_CONFIG[args.dataset]
    artifacts_dir = Path(args.artifacts_dir or config["default_artifacts_dir"])
    label_key = args.label_key or config["default_label_key"]
    setup_fn = config["setup_fn"]

    margin_path = artifacts_dir / str(config["margin_file"])
    softmax_path = artifacts_dir / str(config["softmax_file"])
    backbone_path = artifacts_dir / str(config["backbone_file"])

    if not margin_path.exists() or not softmax_path.exists() or not backbone_path.exists():
        raise FileNotFoundError(
            "Stage-1 artifacts are missing. Expected files: "
            f"{margin_path}, {softmax_path}, {backbone_path}"
        )

    margin_history = np.load(margin_path)
    softmax_history = np.load(softmax_path)

    trainset, noise_data = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    if label_key not in noise_data:
        raise KeyError(f"Label key '{label_key}' not found. Available keys: {list(noise_data.keys())}")
    noisy_labels = np.array(noise_data[label_key]).astype(np.int64)

    descriptor_targets = compute_descriptors(margin_history, softmax_history, noisy_labels)
    feature_matrix = extract_features(trainset, backbone_path, int(config["num_classes"]), device)

    mlp_dataset = TensorDataset(
        torch.from_numpy(feature_matrix),
        torch.from_numpy(descriptor_targets),
    )
    mlp_loader = DataLoader(mlp_dataset, batch_size=args.batch_size, shuffle=True)

    mlp = AdvancedMLP().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

    logging.info("Stage 2 started: dataset=%s, epochs=%d, label_key=%s", args.dataset, args.epochs, label_key)
    for epoch in range(args.epochs):
        mlp.train()
        epoch_loss = 0.0

        for batch_x, batch_y in mlp_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = mlp(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(mlp_loader))
        logging.info("Epoch %d/%d | loss=%.6f", epoch + 1, args.epochs, avg_loss)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlp_path = artifacts_dir / str(config["mlp_file"])
    features_path = artifacts_dir / str(config["features_file"])
    targets_path = artifacts_dir / str(config["targets_file"])

    torch.save(mlp.state_dict(), mlp_path)
    np.save(features_path, feature_matrix)
    np.save(targets_path, descriptor_targets)

    logging.info("Stage 2 artifacts saved to %s", artifacts_dir.resolve())
    return mlp_path, features_path, targets_path


def main() -> None:
    configure_logging()
    args = parse_args()
    train_stage2(args)


if __name__ == "__main__":
    main()

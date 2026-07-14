"""
Stage 2: Distill full training trajectories using a Trajectory Generator.
Corrects data augmentation leaks during feature extraction and standardizes
trajectory matrix alignment.
"""

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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import FeatureExtractor, TrajectoryGenerator, get_resnet18_backbone

# Global dataset specifications matching Stage 1 and evaluation modules[cite: 6]
DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "cifar10n": {
        "num_classes": 10,
        "default_label_key": "aggre_label",
        "default_artifacts_dir": "./artifacts",
        "margin_file": "margin_history_cifar10n.npy",
        "softmax_file": "softmax_history_cifar10n.npy",
        "backbone_file": "resnet18_backbone.pth",
        "trajectory_file": "trajectory_generator.pth",
        "features_file": "X_features_cifar10n.npy",
        "targets_file": "trajectory_targets_cifar10n.npy",
        "setup_fn": setup_cifar10n,
    },
    "cifar100n": {
        "num_classes": 100,
        "default_label_key": "noisy_label",
        "default_artifacts_dir": "./artifacts_cifar100n",
        "margin_file": "margin_history_cifar100n.npy",
        "softmax_file": "softmax_history_cifar100n.npy",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "trajectory_file": "trajectory_generator.pth",
        "features_file": "X_features_cifar100n.npy",
        "targets_file": "trajectory_targets_cifar100n.npy",
        "setup_fn": setup_cifar100n,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Distill full training trajectories with a TrajectoryGenerator."
    )
    parser.add_argument(
        "--dataset", type=str, choices=("cifar10n", "cifar100n"), default="cifar10n"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Stage-2 batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of trajectory distillation epochs.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for trajectory generator.")
    parser.add_argument(
        "--loss", type=str, choices=("smooth_l1", "mse"), default="smooth_l1",
        help="Loss function for trajectory regression."
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset directory.")
    parser.add_argument("--repo-root", type=str, default=None, help="Optional path to local cifar repository.")
    parser.add_argument(
        "--artifacts-dir", type=str, default=None,
        help="Directory containing stage-1 outputs and receiving stage-2 outputs."
    )
    parser.add_argument(
        "--label-key", type=str, default=None,
        help="Optional label key from noise metadata."
    )
    parser.add_argument(
        "--target-type", type=str, choices=("margin", "softmax"), default="margin",
        help="Full trajectory target to distill."
    )
    parser.add_argument(
        "--sequence-length", type=int, default=50,
        help="Expected trajectory length / number of training epochs."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden size used inside the trajectory generator."
    )
    parser.add_argument(
        "--temporal-dim", type=int, default=128,
        help="Internal temporal width used by the trajectory generator."
    )
    parser.add_argument(
        "--num-residual-blocks", type=int, default=3,
        help="Number of residual MLP blocks inside the trajectory generator."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate used inside the trajectory generator."
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


def load_trajectory_targets(
    target_type: str,
    margin_history: np.ndarray,
    softmax_history: np.ndarray,
    noisy_labels: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """
    Load a full trajectory target in sample-major format [N, T].
    Uses explicit metadata dimensions to perform transpositions robustly.
    """
    num_samples = noisy_labels.shape[0]

    if target_type == "margin":
        # Stage 1 saves margins as [T, N]. We verify and transpose strictly[cite: 6, 7]
        if margin_history.shape == (sequence_length, num_samples):
            targets = margin_history.T
        elif margin_history.shape == (num_samples, sequence_length):
            targets = margin_history
        else:
            raise ValueError(
                f"Unexpected margin_history dimensions: {margin_history.shape}. "
                f"Expected ({sequence_length}, {num_samples}) or vice versa."
            )
        return targets.astype(np.float32)

    if target_type == "softmax":
        # Stage 1 saves softmax history as [T, N, C][cite: 6, 7]
        if softmax_history.ndim != 3:
            raise ValueError(f"softmax_history must be 3D [T, N, C], got shape {softmax_history.shape}")
        if softmax_history.shape[1] != num_samples:
            raise ValueError("softmax_history sample dimension and noisy_labels length must match")
            
        sample_indices = np.arange(num_samples)
        # Slicing trajectories of target noisy labels across all epochs: shape [T, N]
        trajectory = softmax_history[:sequence_length, sample_indices, noisy_labels]
        # Transpose to sample-major format [N, T]
        return trajectory.T.astype(np.float32)

    raise ValueError(f"Unsupported target_type: {target_type}")


def extract_features(
    trainset: Any,
    backbone_path: Path,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """
    Extracts deep features from the backbone network.
    METHODOLOGICAL FIX: Temporarily overrides dataset transforms to guarantee 
    deterministic, non-augmented feature extraction.
    """
    # 1. Initialize evaluation-grade backbone[cite: 7]
    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device)
    backbone.eval()

    extractor = FeatureExtractor(backbone).to(device)
    extractor.eval()

    # 2. METHODOLOGICAL FIX: Clean deterministic transform override
    original_transform = getattr(trainset, "transform", None)
    trainset.transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalizations here if they were used during training
    ])

    extract_loader = DataLoader(
        trainset, batch_size=256, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda")
    )
    features_list = []

    logging.info("Extracting deterministic features from training images...")
    with torch.no_grad():
        for images, _ in extract_loader:
            features = extractor(images.to(device))
            features_list.append(features.cpu().numpy())

    # Restore original augmentations for dataset integrity
    if original_transform is not None:
        trainset.transform = original_transform

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

    # Load history artifacts generated in Stage 1[cite: 7]
    margin_history = np.load(margin_path)
    softmax_history = np.load(softmax_path)

    trainset, noise_data = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    if label_key not in noise_data:
        raise KeyError(f"Label key '{label_key}' not found. Available keys: {list(noise_data.keys())}")
    noisy_labels = np.array(noise_data[label_key]).astype(np.int64)

    # METHODOLOGICAL FIX: Safely parse and align trajectory targets[cite: 7]
    trajectory_targets = load_trajectory_targets(
        args.target_type, margin_history, softmax_history, noisy_labels, args.sequence_length
    )
    
    if trajectory_targets.shape[1] != args.sequence_length:
        raise ValueError(
            f"Expected trajectory length {args.sequence_length}, but got {trajectory_targets.shape[1]}. "
            "Check stage-1 epoch count or sequence_length config."
        )

    # METHODOLOGICAL FIX: Extract features without random crop/flip noise[cite: 7]
    feature_matrix = extract_features(trainset, backbone_path, int(config["num_classes"]), device)
    if feature_matrix.shape[0] != trajectory_targets.shape[0]:
        raise ValueError(
            f"Feature count {feature_matrix.shape[0]} does not match trajectory target count {trajectory_targets.shape[0]}."
        )

    # Setup the regression dataset[cite: 7]
    trajectory_dataset = TensorDataset(
        torch.from_numpy(feature_matrix),
        torch.from_numpy(trajectory_targets),
    )
    trajectory_loader = DataLoader(
        trajectory_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    # Initialize the generator[cite: 7]
    generator = TrajectoryGenerator(
        input_dim=feature_matrix.shape[1],
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        temporal_dim=args.temporal_dim,
        num_residual_blocks=args.num_residual_blocks,
        dropout=args.dropout,
        output_channels=1,
        use_logit_norm=False,
    ).to(device)

    criterion: nn.Module = nn.MSELoss() if args.loss == "mse" else nn.SmoothL1Loss()
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)

    logging.info(
        "Stage 2 started: dataset=%s, epochs=%d, target_type=%s, label_key=%s, sequence_length=%d",
        args.dataset, args.epochs, args.target_type, label_key, args.sequence_length,
    )
    
    # Start regression distillation training[cite: 7]
    for epoch in range(args.epochs):
        generator.train()
        epoch_loss = 0.0

        for batch_x, batch_y in trajectory_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = generator(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(trajectory_loader))
        logging.info("Epoch %d/%d | loss=%.6f", epoch + 1, args.epochs, avg_loss)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Define output file structures matching stage-1 setups[cite: 7]
    trajectory_path = artifacts_dir / str(config["trajectory_file"])
    features_path = artifacts_dir / str(config["features_file"])
    targets_name = f"{Path(str(config['targets_file'])).stem}_{args.target_type}.npy"
    targets_path = artifacts_dir / targets_name

    # Export refined models and matrices[cite: 7]
    torch.save(generator.state_dict(), trajectory_path)
    np.save(features_path, feature_matrix)
    np.save(targets_path, trajectory_targets)

    logging.info("Stage 2 artifacts successfully saved to %s", artifacts_dir.resolve())
    return trajectory_path, features_path, targets_path


def main() -> None:
    configure_logging()
    args = parse_args()
    train_stage2(args)


if __name__ == "__main__":
    main()
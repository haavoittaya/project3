"""
Stage 2 (Method B): Distill 4D uncertainty descriptors using a lightweight Parametric MLP.
Corrects stochastic leakage during feature extraction and standardizes metadata alignment.
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
from src.models import AdvancedMLP, FeatureExtractor, get_resnet18_backbone

# Global dataset specifications matching the reproducible pipeline architecture[cite: 8]
DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "cifar10n": {
        "num_classes": 10,
        "default_label_key": "aggre_label",
        "default_artifacts_dir": "./artifacts",
        "margin_file": "margin_history_cifar10n.npy",
        "softmax_file": "softmax_history_cifar10n.npy",
        "backbone_file": "resnet18_backbone.pth",
        "mlp_file": "mlp_descriptor_head.pth",
        "features_file": "X_features_cifar10n.npy",
        "targets_file": "descriptors_4d_targets_cifar10n.npy",
        "setup_fn": setup_cifar10n,
    },
    "cifar100n": {
        "num_classes": 100,
        "default_label_key": "noisy_label",
        "default_artifacts_dir": "./artifacts_cifar100n",
        "margin_file": "margin_history_cifar100n.npy",
        "softmax_file": "softmax_history_cifar100n.npy",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "mlp_file": "mlp_descriptor_head.pth",
        "features_file": "X_features_cifar100n.npy",
        "targets_file": "descriptors_4d_targets_cifar100n.npy",
        "setup_fn": setup_cifar100n,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Distill 4D uncertainty descriptors with a lightweight MLP."
    )
    parser.add_argument(
        "--dataset", type=str, choices=("cifar10n", "cifar100n"), default="cifar10n"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of regression epochs.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset directory.")
    parser.add_argument("--repo-root", type=str, default=None, help="Optional local repo path.")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Path for storing outputs.")
    parser.add_argument("--label-key", type=str, default=None, help="Noise metadata label key.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_4d_descriptors(
    softmax_history: np.ndarray, 
    margin_history: np.ndarray, 
    noisy_labels: np.ndarray
) -> np.ndarray:
    """
    Transforms tracking artifacts into 4D descriptor targets [N, 4].
    Enforces strict dimension checks before mathematical aggregation.
    """
    if softmax_history.ndim != 3 or margin_history.ndim != 2:
        raise ValueError("Invalid history dimensions. Expected Softmax: [T, N, C], Margin: [T, N].")
        
    T, N, C = softmax_history.shape
    
    if margin_history.shape[1] != N or noisy_labels.shape[0] != N:
        raise ValueError("Sample dimension mismatch between artifacts and dataset labels.")
    
    # 1. AUM (Area Under the Margin)[cite: 8]
    aum = np.mean(margin_history, axis=0)
    
    # 2. Mean Confidence (over the assigned noisy labels)[cite: 8]
    sample_indices = np.arange(N)
    true_probs = softmax_history[:, sample_indices, noisy_labels]
    mean_conf = np.mean(true_probs, axis=0)
    
    # 3. Variability (Standard deviation of confidences)[cite: 8]
    variability = np.std(true_probs, axis=0)
    
    # 4. Forgetting Count (Margin transitions from > 0 to <= 0)[cite: 8]
    learned = margin_history > 0
    forgetting_count = np.sum((learned[:-1, :] == True) & (learned[1:, :] == False), axis=0)

    # Stack descriptors into shape [N, 4][cite: 8]
    descriptors = np.stack([aum, mean_conf, variability, forgetting_count], axis=-1)
    return descriptors.astype(np.float32)


def extract_features(
    trainset: Any, 
    backbone_path: Path, 
    num_classes: int, 
    device: torch.device
) -> np.ndarray:
    """
    Extracts deep features from the backbone network.
    METHODOLOGICAL FIX: Temporarily overrides dataset transforms to guarantee 
    deterministic, non-augmented feature extraction.
    """
    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device)
    backbone.eval()

    extractor = FeatureExtractor(backbone).to(device)
    extractor.eval()

    # METHODOLOGICAL FIX: Clean deterministic transform override
    original_transform = getattr(trainset, "transform", None)
    trainset.transform = transforms.Compose([
        transforms.ToTensor(),
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

    # Restore original augmentations
    if original_transform is not None:
        trainset.transform = original_transform

    return np.concatenate(features_list, axis=0).astype(np.float32)


def train_stage2_mlp(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
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
            f"Missing Stage-1 artifacts: {margin_path}, {softmax_path}, {backbone_path}"
        )

    # Load Stage-1 dynamic artifacts[cite: 8]
    margin_history = np.load(margin_path)
    softmax_history = np.load(softmax_path)

    trainset, noise_data = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    noisy_labels = np.array(noise_data[label_key]).astype(np.int64)

    # 1. Compute 4D regression targets strictly[cite: 8]
    descriptors = compute_4d_descriptors(softmax_history, margin_history, noisy_labels)
    
    # 2. Extract aligned deterministic features[cite: 8]
    feature_matrix = extract_features(trainset, backbone_path, int(config["num_classes"]), device)
    
    if feature_matrix.shape[0] != descriptors.shape[0]:
        raise ValueError("Mismatch between number of extracted features and descriptor targets.")
    
    dataset = TensorDataset(torch.from_numpy(feature_matrix), torch.from_numpy(descriptors))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 3. Model Initialization (4D output) & Loss Setup[cite: 8]
    model = AdvancedMLP(input_dim=feature_matrix.shape[1], output_dim=4).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logging.info("Stage 2 MLP started: dataset=%s, epochs=%d, target_dim=4", args.dataset, args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(loader))
        logging.info("Epoch %d/%d | loss=%.6f", epoch + 1, args.epochs, avg_loss)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    mlp_path = artifacts_dir / str(config["mlp_file"])
    features_path = artifacts_dir / str(config["features_file"])
    targets_path = artifacts_dir / str(config["targets_file"])

    torch.save(model.state_dict(), mlp_path)
    np.save(features_path, feature_matrix)
    np.save(targets_path, descriptors)

    logging.info("Stage 2 MLP artifacts saved to %s", artifacts_dir.resolve())
    return mlp_path, features_path, targets_path


def main() -> None:
    configure_logging()
    args = parse_args()
    train_stage2_mlp(args)


if __name__ == "__main__":
    main()
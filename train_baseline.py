import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import AdvancedMLP, FeatureExtractor, get_resnet18_backbone

DATASET_CONFIG = {
    "cifar10n": {
        "num_classes": 10,
        "train_noise_key": "aggre_label",
        "clean_label_key": "clean_label",
        "backbone_file": "resnet18_backbone.pth",
        "setup_fn": setup_cifar10n,
        "default_artifacts_dir": "./artifacts",
    },
    "cifar100n": {
        "num_classes": 100,
        "train_noise_key": "noisy_label",
        "clean_label_key": "clean_label",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "setup_fn": setup_cifar100n,
        "default_artifacts_dir": "./artifacts_cifar100n",
    },
}

# ------------------------------------------------------------
# Normalization statistics (must match Stage 1)
# ------------------------------------------------------------
CIFAR_STATS = {
    "cifar10n": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100n": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


def get_train_transform(dataset_name: str) -> transforms.Compose:
    """Return training transform with augmentation and normalization."""
    stats = CIFAR_STATS.get(dataset_name, CIFAR_STATS["cifar10n"])
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline direct image-only label correctness predictor.")
    parser.add_argument("--dataset", type=str, choices=("cifar10n", "cifar100n"), default="cifar10n")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = DATASET_CONFIG[args.dataset]
    artifacts_dir = Path(args.artifacts_dir or config["default_artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Preparing dataset and ground-truth binary targets...")
    # Create training transform with augmentation and normalization
    train_transform = get_train_transform(args.dataset)
    trainset, noise_data = config["setup_fn"](
        data_root=args.data_root,
        repo_root=args.repo_root,
        transform=train_transform
    )
    
    clean_labels = np.asarray(noise_data[config["clean_label_key"]], dtype=np.int64)
    noisy_labels = np.asarray(noise_data[config["train_noise_key"]], dtype=np.int64)
    
    binary_targets = (clean_labels == noisy_labels).astype(np.float32)
    trainset.targets = binary_targets.tolist()
    
    # ---------------------------------------------------------
    # FIXED: Rigorous Meta-Train split to prevent data leak
    # ---------------------------------------------------------
    rng = np.random.default_rng(42)
    indices = np.arange(len(trainset))
    rng.shuffle(indices)
    split_point = int(len(indices) * 0.5)
    meta_train_indices = indices[:split_point]
    
    train_subset = Subset(trainset, meta_train_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    logging.info("Loading pre-trained backbone...")
    backbone_path = artifacts_dir / config["backbone_file"]
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone not found at {backbone_path}.")
        
    backbone = get_resnet18_backbone(num_classes=config["num_classes"])
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
        
    feature_extractor = FeatureExtractor(backbone).to(device)
    feature_extractor.eval()

    logging.info("Initializing baseline MLP predictor...")
    mlp = AdvancedMLP(input_dim=512, output_dim=1, hidden_dim=256, dropout=0.2).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

    logging.info("Starting training on Meta-Train subset...")
    for epoch in range(args.epochs):
        mlp.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            with torch.no_grad():
                features = feature_extractor(images)
                
            optimizer.zero_grad(set_to_none=True)
            logits = mlp(features)
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_preds += (preds == targets).sum().item()
            total_samples += images.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = correct_preds / total_samples
        logging.info("Epoch %2d/%d | Loss: %.4f | Accuracy: %.4f", epoch + 1, args.epochs, epoch_loss, epoch_acc)

    output_path = artifacts_dir / "baseline_binary_predictor.pth"
    torch.save(mlp.state_dict(), output_path)
    logging.info("Baseline predictor saved to %s", output_path)


if __name__ == "__main__":
    main()
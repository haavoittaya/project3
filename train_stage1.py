from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import get_resnet18_backbone

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "cifar10n": {
        "num_classes": 10,
        "default_label_key": "aggre_label",
        "default_output_dir": "./artifacts",
        "softmax_file": "softmax_history_cifar10n.npy",
        "margin_file": "margin_history_cifar10n.npy",
        "backbone_file": "resnet18_backbone.pth",
        "setup_fn": setup_cifar10n,
    },
    "cifar100n": {
        "num_classes": 100,
        "default_label_key": "noisy_label",
        "default_output_dir": "./artifacts_cifar100n",
        "softmax_file": "softmax_history_cifar100n.npy",
        "margin_file": "margin_history_cifar100n.npy",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "setup_fn": setup_cifar100n,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: train backbone and track per-sample dynamics on CIFAR-10N/CIFAR-100N."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10n", "cifar100n"),
        default="cifar10n",
        help="Dataset/noise benchmark to use.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
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
        "--label-key",
        type=str,
        default=None,
        help="Optional label key from noise metadata (auto-selected by dataset if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store stage-1 outputs (auto-selected by dataset if omitted).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def compute_margin_batch(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    batch_idx = np.arange(logits_np.shape[0])

    true_logits = logits_np[batch_idx, labels_np]
    non_true_logits = logits_np.copy()
    non_true_logits[batch_idx, labels_np] = -np.inf

    return true_logits - non_true_logits.max(axis=1)


def train_stage1(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    config = DATASET_CONFIG[args.dataset]
    label_key = args.label_key or config["default_label_key"]
    setup_fn = config["setup_fn"]

    trainset, noise_data = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    if label_key not in noise_data:
        raise KeyError(f"Label key '{label_key}' not found. Available keys: {list(noise_data.keys())}")
    trainset.targets = np.array(noise_data[label_key]).tolist()

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    num_samples = len(trainset)
    num_classes = int(config["num_classes"])

    model = get_resnet18_backbone(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    softmax_history = np.zeros((args.epochs, num_samples, num_classes), dtype=np.float32)
    margin_history = np.zeros((args.epochs, num_samples), dtype=np.float32)

    logging.info(
        "Stage 1 started: dataset=%s, epochs=%d, samples=%d, label_key=%s",
        args.dataset,
        args.epochs,
        num_samples,
        label_key,
    )
    for epoch in range(args.epochs):
        model.train()
        sample_idx = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                margins = compute_margin_batch(logits, labels)
                batch_size = images.size(0)
                end_idx = sample_idx + batch_size

                softmax_history[epoch, sample_idx:end_idx, :] = probs
                margin_history[epoch, sample_idx:end_idx] = margins
                sample_idx = end_idx

        logging.info("Epoch %d/%d completed", epoch + 1, args.epochs)

    output_dir = Path(args.output_dir or config["default_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    softmax_path = output_dir / str(config["softmax_file"])
    margin_path = output_dir / str(config["margin_file"])
    backbone_path = output_dir / str(config["backbone_file"])

    np.save(softmax_path, softmax_history)
    np.save(margin_path, margin_history)
    torch.save(model.state_dict(), backbone_path)

    logging.info("Stage 1 artifacts saved to %s", output_dir.resolve())
    return softmax_path, margin_path, backbone_path


def main() -> None:
    configure_logging()
    args = parse_args()
    train_stage1(args)


if __name__ == "__main__":
    main()

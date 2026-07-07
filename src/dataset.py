from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

CIFAR10N_REPO_URL = "https://github.com/UCSC-REAL/cifar-10-100n.git"
CIFAR10N_REPO_DIR = "cifar-10-100n"
CIFAR10N_LABELS_FILE = Path("data") / "CIFAR-10_human.pt"


def _ensure_cifar10n_repo(repo_path: Path) -> None:
    """Clone the CIFAR-10N metadata repository if it is not available locally."""
    if repo_path.exists():
        return

    print("Cloning UCSC-REAL/cifar-10-100n repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", CIFAR10N_REPO_URL, str(repo_path)],
        check=True,
    )


def _default_transform() -> transforms.Compose:
    """Return the default preprocessing pipeline for CIFAR-10 samples."""
    return transforms.Compose([transforms.ToTensor()])


def setup_cifar10n(
    data_root: str = "./data",
    repo_root: Optional[str] = None,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Prepare CIFAR-10 training set and CIFAR-10N noise annotations.

    Args:
        data_root: Local path for torchvision CIFAR-10 dataset storage.
        repo_root: Local path of CIFAR-10N repository clone. If omitted,
            ``./cifar-10-100n`` is used.
        transform: Optional torchvision transform for CIFAR-10 samples.

    Returns:
        Tuple consisting of:
        1) CIFAR-10 training dataset.
        2) Dictionary with CIFAR-10N human/noise labels and metadata.
    """
    repo_path = Path(repo_root) if repo_root else Path(CIFAR10N_REPO_DIR)
    _ensure_cifar10n_repo(repo_path)

    labels_path = repo_path / CIFAR10N_LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"CIFAR-10N labels file not found: {labels_path}")

    noise_data: Dict[str, Any] = torch.load(labels_path, weights_only=False)
    dataset_transform = transform or _default_transform()
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=dataset_transform,
    )

    return trainset, noise_data
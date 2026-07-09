from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

CIFAR10N_LABELS_FILE = Path("CIFAR-10_human.pt")
CIFAR10N_LABELS_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/UCSC-REAL/cifar-10-100n@master/data/CIFAR-10_human.pt",
    "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/CIFAR-10_human.pt",
)
CIFAR100N_LABELS_FILE = Path("CIFAR-100_human.pt")
CIFAR100N_LABELS_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/UCSC-REAL/cifar-10-100n@master/data/CIFAR-100_human.pt",
    "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/CIFAR-100_human.pt",
)


def _default_transform() -> transforms.Compose:
    """Return the default preprocessing pipeline for CIFAR-10 samples."""
    return transforms.Compose([transforms.ToTensor()])


def _ensure_cifar10n_labels(data_root: Path, repo_root: Optional[Path]) -> Path:
    """Ensure that the CIFAR-10N label file is available locally.

    The function first checks a user-provided repository path, then the data root,
    and finally downloads only the required `CIFAR-10_human.pt` artifact from a
    faster mirror instead of cloning the full repository.
    """
    candidate_paths = []
    if repo_root is not None:
        candidate_paths.append(repo_root / "data" / CIFAR10N_LABELS_FILE.name)
    candidate_paths.append(data_root / CIFAR10N_LABELS_FILE.name)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    data_root.mkdir(parents=True, exist_ok=True)
    target_path = data_root / CIFAR10N_LABELS_FILE.name

    last_error: Optional[Exception] = None
    for url in CIFAR10N_LABELS_MIRRORS:
        try:
            print(f"Downloading CIFAR-10N labels from {url}...")
            urlretrieve(url, target_path)
            return target_path
        except (URLError, OSError, ValueError) as exc:
            last_error = exc

    raise FileNotFoundError(
        "Unable to obtain CIFAR-10N labels file from available mirrors. "
        f"Last error: {last_error}"
    )


def _ensure_cifar100n_labels(data_root: Path, repo_root: Optional[Path]) -> Path:
    """Ensure that the CIFAR-100N label file is available locally."""
    candidate_paths = []
    if repo_root is not None:
        candidate_paths.append(repo_root / "data" / CIFAR100N_LABELS_FILE.name)
    candidate_paths.append(data_root / CIFAR100N_LABELS_FILE.name)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    data_root.mkdir(parents=True, exist_ok=True)
    target_path = data_root / CIFAR100N_LABELS_FILE.name

    last_error: Optional[Exception] = None
    for url in CIFAR100N_LABELS_MIRRORS:
        try:
            print(f"Downloading CIFAR-100N labels from {url}...")
            urlretrieve(url, target_path)
            return target_path
        except (URLError, OSError, ValueError) as exc:
            last_error = exc

    raise FileNotFoundError(
        "Unable to obtain CIFAR-100N labels file from available mirrors. "
        f"Last error: {last_error}"
    )


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
    data_path = Path(data_root)
    repo_path = Path(repo_root) if repo_root else None
    labels_path = _ensure_cifar10n_labels(data_path, repo_path)

    noise_data: Dict[str, Any] = torch.load(labels_path, weights_only=False)
    dataset_transform = transform or _default_transform()
    trainset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=dataset_transform,
    )

    return trainset, noise_data


def setup_cifar100n(
    data_root: str = "./data",
    repo_root: Optional[str] = None,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Prepare CIFAR-100 training set and CIFAR-100N noise annotations."""
    data_path = Path(data_root)
    repo_path = Path(repo_root) if repo_root else None
    labels_path = _ensure_cifar100n_labels(data_path, repo_path)

    noise_data: Dict[str, Any] = torch.load(labels_path, weights_only=False)
    dataset_transform = transform or _default_transform()
    trainset = torchvision.datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=dataset_transform,
    )

    return trainset, noise_data
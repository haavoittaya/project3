from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import AdvancedMLP, FeatureExtractor, get_resnet18_backbone

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "cifar10n": {
        "num_classes": 10,
        "default_artifacts_dir": "./artifacts",
        "default_reports_dir": "./artifacts/reports",
        "default_train_label_key": "aggre_label",
        "default_eval_label_key": "worse_label",
        "clean_label_key": "clean_label",
        "backbone_file": "resnet18_backbone.pth",
        "mlp_file": "mlp_4d_cifar10n.pth",
        "features_file": "X_features.npy",
        "noisy_report_prefix": "worse_split",
        "ood_report_prefix": "ood_cifar10_vs_svhn",
        "ood_title": "OOD detection (CIFAR-10 vs SVHN)",
        "noisy_title": "Noisy label detection (Worse split)",
        "setup_fn": setup_cifar10n,
    },
    "cifar100n": {
        "num_classes": 100,
        "default_artifacts_dir": "./artifacts_cifar100n",
        "default_reports_dir": "./artifacts_cifar100n/reports",
        "default_train_label_key": "noisy_label",
        "default_eval_label_key": "noisy_label",
        "clean_label_key": "clean_label",
        "backbone_file": "resnet18_backbone_cifar100n.pth",
        "mlp_file": "mlp_4d_cifar100n.pth",
        "features_file": "X_features_cifar100n.npy",
        "noisy_report_prefix": "cifar100n_noisy_labels",
        "ood_report_prefix": "ood_cifar100_vs_svhn",
        "ood_title": "OOD detection (CIFAR-100 vs SVHN)",
        "noisy_title": "Noisy label detection (CIFAR-100N)",
        "setup_fn": setup_cifar100n,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate distilled descriptors on noisy labels, OOD data, and synthetic corruptions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10n", "cifar100n"),
        default="cifar10n",
        help="Dataset/noise benchmark to evaluate.",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset directory.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Optional path to local cifar-10-100n repository.",
    )
    parser.add_argument(
        "--svhn-root",
        type=str,
        default="./data_svhn",
        help="SVHN test split directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory with stage-1 and stage-2 artifacts (auto-selected if omitted).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Directory for evaluation reports (CSV/PNG) (auto-selected if omitted).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Feature extraction batch size.")
    parser.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="Optional training-label key used for descriptor computation semantics.",
    )
    parser.add_argument(
        "--eval-label-key",
        type=str,
        default=None,
        help="Optional label key used for noisy-label evaluation target.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def compute_risk_coverage(
    error_labels: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute risk-coverage curve and AURC.

    Lower uncertainty is treated as more confident prediction and retained first.
    """
    if error_labels.ndim != 1 or uncertainty_scores.ndim != 1:
        raise ValueError("error_labels and uncertainty_scores must be 1D arrays")
    if error_labels.shape[0] != uncertainty_scores.shape[0]:
        raise ValueError("error_labels and uncertainty_scores must have the same length")

    order = np.argsort(uncertainty_scores)  # keep most confident samples first
    sorted_errors = error_labels[order].astype(np.float64)

    cumulative_errors = np.cumsum(sorted_errors)
    ranks = np.arange(1, sorted_errors.shape[0] + 1, dtype=np.float64)

    coverage = ranks / ranks[-1]
    risk = cumulative_errors / ranks
    aurc = float(np.mean(risk))

    return coverage, risk, aurc


def save_risk_coverage_report(
    coverage: np.ndarray,
    risk: np.ndarray,
    aurc: float,
    report_prefix: str,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"{report_prefix}_risk_coverage.csv"
    png_path = reports_dir / f"{report_prefix}_risk_coverage.png"

    report_data = np.column_stack((coverage, risk))
    np.savetxt(
        csv_path,
        report_data,
        delimiter=",",
        header="coverage,risk",
        comments="",
    )

    plt.figure(figsize=(7, 5))
    plt.plot(coverage, risk, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title(f"Risk-Coverage Curve ({report_prefix}, AURC={aurc:.4f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    logging.info("Saved risk-coverage report: %s", csv_path)
    logging.info("Saved risk-coverage plot: %s", png_path)


def apply_corruption(images_tensor: torch.Tensor, corruption_type: str, severity: int) -> torch.Tensor:
    np_imgs = images_tensor.permute(0, 2, 3, 1).cpu().numpy()
    corrupted_list: List[np.ndarray] = []

    for img in np_imgs:
        if corruption_type == "blur":
            kernel_size = severity * 2 + 1
            corrupted = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif corruption_type == "noise":
            noise = np.random.normal(0, 0.05 * severity, img.shape)
            corrupted = np.clip(img + noise, 0, 1)
        else:
            raise ValueError(f"Unsupported corruption type: {corruption_type}")

        corrupted_list.append(corrupted)

    return torch.tensor(np.array(corrupted_list), dtype=torch.float32).permute(0, 3, 1, 2)


def build_models(
    artifacts_dir: Path,
    backbone_file: str,
    mlp_file: str,
    num_classes: int,
    device: torch.device,
) -> Tuple[FeatureExtractor, AdvancedMLP]:
    backbone_path = artifacts_dir / backbone_file
    mlp_path = artifacts_dir / mlp_file

    if not backbone_path.exists() or not mlp_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts in {artifacts_dir}. Expected {backbone_path} and {mlp_path}."
        )

    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device)
    backbone.eval()

    extractor = FeatureExtractor(backbone).to(device)
    extractor.eval()

    mlp = AdvancedMLP().to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.eval()

    return extractor, mlp


def extract_features(dataset, extractor: FeatureExtractor, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_batches: List[np.ndarray] = []

    with torch.no_grad():
        for images, _ in loader:
            feats = extractor(images.to(device)).cpu().numpy()
            feature_batches.append(feats)

    return np.concatenate(feature_batches, axis=0).astype(np.float32)


def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DATASET_CONFIG[args.dataset]
    artifacts_dir = Path(args.artifacts_dir or config["default_artifacts_dir"])
    reports_dir = Path(args.reports_dir or config["default_reports_dir"])
    train_label_key = args.label_key or config["default_train_label_key"]
    eval_label_key = args.eval_label_key or config["default_eval_label_key"]
    clean_label_key = str(config["clean_label_key"])
    setup_fn = config["setup_fn"]

    extractor, mlp = build_models(
        artifacts_dir,
        str(config["backbone_file"]),
        str(config["mlp_file"]),
        int(config["num_classes"]),
        device,
    )

    features_path = artifacts_dir / str(config["features_file"])
    if features_path.exists():
        x_features = np.load(features_path)
        logging.info("Loaded cached ID feature matrix: %s", features_path)
    else:
        trainset, _ = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
        x_features = extract_features(trainset, extractor, args.batch_size, device)
        logging.info("Cached feature matrix not found; extracted features on-the-fly.")

    _, noise_data = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    if eval_label_key not in noise_data or clean_label_key not in noise_data:
        raise KeyError(
            "Missing required keys for noisy-label evaluation. "
            f"Required: {eval_label_key}, {clean_label_key}; available: {list(noise_data.keys())}"
        )

    with torch.no_grad():
        id_preds = mlp(torch.from_numpy(x_features).to(device)).cpu().numpy()

    pred_aum = id_preds[:, 0]
    eval_labels = np.array(noise_data[eval_label_key])
    clean_labels = np.array(noise_data[clean_label_key])
    is_noisy_error = (eval_labels != clean_labels).astype(np.int32)

    noisy_auroc = roc_auc_score(is_noisy_error, -pred_aum)
    noisy_auprc = average_precision_score(is_noisy_error, -pred_aum)
    noisy_coverage, noisy_risk, noisy_aurc = compute_risk_coverage(is_noisy_error, -pred_aum)

    logging.info(
        "%s | AUROC: %.4f | AUPRC: %.4f | AURC: %.4f",
        str(config["noisy_title"]),
        noisy_auroc,
        noisy_auprc,
        noisy_aurc,
    )
    save_risk_coverage_report(
        noisy_coverage,
        noisy_risk,
        noisy_aurc,
        str(config["noisy_report_prefix"]),
        reports_dir,
    )

    svhn = torchvision.datasets.SVHN(
        root=args.svhn_root,
        split="test",
        download=True,
        transform=transforms.ToTensor(),
    )
    x_svhn = extract_features(svhn, extractor, args.batch_size, device)

    with torch.no_grad():
        svhn_preds = mlp(torch.from_numpy(x_svhn).to(device)).cpu().numpy()

    id_scores = pred_aum
    ood_scores = svhn_preds[:, 0]

    ood_labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)]).astype(np.int32)
    ood_confidence = np.concatenate([-id_scores, -ood_scores])

    ood_auroc = roc_auc_score(ood_labels, ood_confidence)
    ood_auprc = average_precision_score(ood_labels, ood_confidence)
    ood_coverage, ood_risk, ood_aurc = compute_risk_coverage(ood_labels, ood_confidence)

    logging.info(
        "%s | AUROC: %.4f | AUPRC: %.4f | AURC: %.4f",
        str(config["ood_title"]),
        ood_auroc,
        ood_auprc,
        ood_aurc,
    )
    save_risk_coverage_report(
        ood_coverage,
        ood_risk,
        ood_aurc,
        str(config["ood_report_prefix"]),
        reports_dir,
    )
    logging.info("Mean predicted AUM | ID: %.4f | OOD: %.4f", np.mean(id_scores), np.mean(ood_scores))

    trainset, noise_data_for_train = setup_fn(data_root=args.data_root, repo_root=args.repo_root)
    if train_label_key not in noise_data_for_train:
        raise KeyError(
            f"Training label key '{train_label_key}' not found. "
            f"Available keys: {list(noise_data_for_train.keys())}"
        )
    clean_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    clean_batch, _ = next(iter(clean_loader))

    for corruption_type in ["blur", "noise"]:
        logging.info("Corruption type: %s", corruption_type)
        for severity in range(1, 6):
            corrupted = apply_corruption(clean_batch, corruption_type, severity).to(device)
            with torch.no_grad():
                c_preds = mlp(extractor(corrupted)).cpu().numpy()

            logging.info("Severity %d | Mean predicted AUM: %.4f", severity, np.mean(c_preds[:, 0]))


def main() -> None:
    configure_logging()
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

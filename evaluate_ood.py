from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import setup_cifar100n, setup_cifar10n
from src.models import LogitNorm, TrajectoryGenerator, get_resnet18_backbone


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    backbone_file: str
    trajectory_file: str
    features_file: str
    artifacts_dir: str
    reports_dir: str
    train_setup_fn: Any
    train_noise_key: str
    clean_label_key: str
    clean_test_ctor: Any
    near_ood_ctor: Any
    opposite_name: str


DATASET_CONFIG: Dict[str, DatasetSpec] = {
    "cifar10n": DatasetSpec(
        name="cifar10n",
        num_classes=10,
        backbone_file="resnet18_backbone.pth",
        trajectory_file="trajectory_generator.pth",
        features_file="X_features_cifar10n.npy",
        artifacts_dir="./artifacts",
        reports_dir="./artifacts/reports",
        train_setup_fn=setup_cifar10n,
        train_noise_key="aggre_label",
        clean_label_key="clean_label",
        clean_test_ctor=torchvision.datasets.CIFAR10,
        near_ood_ctor=torchvision.datasets.CIFAR100,
        opposite_name="CIFAR-100",
    ),
    "cifar100n": DatasetSpec(
        name="cifar100n",
        num_classes=100,
        backbone_file="resnet18_backbone_cifar100n.pth",
        trajectory_file="trajectory_generator.pth",
        features_file="X_features_cifar100n.npy",
        artifacts_dir="./artifacts_cifar100n",
        reports_dir="./artifacts_cifar100n/reports",
        train_setup_fn=setup_cifar100n,
        train_noise_key="noisy_label",
        clean_label_key="clean_label",
        clean_test_ctor=torchvision.datasets.CIFAR100,
        near_ood_ctor=torchvision.datasets.CIFAR10,
        opposite_name="CIFAR-10",
    ),
}


@dataclass
class GroupOutputs:
    name: str
    features: np.ndarray
    logits: np.ndarray
    size: int


@dataclass
class MetricResult:
    method: str
    task: str
    auroc: float
    auprc: float
    aurc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rigorous OOD-centric evaluation for clean ID, noisy ID, near-OOD, and far-OOD groups."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10n", "cifar100n"),
        default="cifar10n",
        help="ID/noise benchmark to evaluate.",
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
        help="Directory for summary reports (auto-selected if omitted).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument(
        "--logitnorm-tau",
        type=float,
        default=1.0,
        help="Temperature parameter used by the LogitNorm baseline.",
    )
    parser.add_argument(
        "--logitnorm-score",
        type=str,
        choices=("msp", "energy"),
        default="msp",
        help="Score type computed on normalized logits.",
    )
    parser.add_argument(
        "--mahalanobis-reg",
        type=float,
        default=1e-3,
        help="Diagonal regularization added to the shared covariance matrix.",
    )
    parser.add_argument(
        "--trajectory-sequence-length",
        type=int,
        default=50,
        help="Trajectory length used by the trained generator.",
    )
    parser.add_argument(
        "--trajectory-hidden-dim",
        type=int,
        default=256,
        help="Hidden size used by the trained trajectory generator.",
    )
    parser.add_argument(
        "--trajectory-temporal-dim",
        type=int,
        default=128,
        help="Temporal width used by the trained trajectory generator.",
    )
    parser.add_argument(
        "--trajectory-num-residual-blocks",
        type=int,
        default=3,
        help="Residual block count used by the trained trajectory generator.",
    )
    parser.add_argument(
        "--trajectory-dropout",
        type=float,
        default=0.1,
        help="Dropout used by the trained trajectory generator.",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=256,
        help="Batch size used for score computation on large groups.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def compute_risk_coverage(
    error_labels: np.ndarray,
    certainty_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if error_labels.ndim != 1 or certainty_scores.ndim != 1:
        raise ValueError("error_labels and certainty_scores must be 1D arrays")
    if error_labels.shape[0] != certainty_scores.shape[0]:
        raise ValueError("error_labels and certainty_scores must have the same length")

    order = np.argsort(-certainty_scores)
    sorted_errors = error_labels[order].astype(np.float64)
    ranks = np.arange(1, sorted_errors.shape[0] + 1, dtype=np.float64)

    coverage = ranks / ranks[-1]
    risk = np.cumsum(sorted_errors) / ranks
    aurc = float(np.mean(risk))

    return coverage, risk, aurc


def save_summary_csv(results: List[MetricResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "task", "auroc", "auprc", "aurc"])
        for row in results:
            writer.writerow([row.method, row.task, f"{row.auroc:.6f}", f"{row.auprc:.6f}", f"{row.aurc:.6f}"])


def build_standard_dataset(name: str, root: str, train: bool) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def build_svhn_dataset(root: str) -> Dataset:
    return torchvision.datasets.SVHN(root=root, split="test", download=True, transform=transforms.ToTensor())


def stratified_split_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_value in np.unique(labels[indices]):
        class_indices = indices[labels[indices] == class_value]
        if class_indices.size == 0:
            continue
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(round(shuffled.size * val_fraction)))
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())

    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def forward_with_features(backbone: torch.nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = backbone.conv1(images)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)

    x = backbone.avgpool(x)
    features = torch.flatten(x, 1)
    logits = backbone.fc(features)
    return features, logits


def load_backbone(artifacts_dir: Path, backbone_file: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    backbone_path = artifacts_dir / backbone_file
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing backbone checkpoint: {backbone_path}")

    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device)
    backbone.eval()
    return backbone


def load_trajectory_generator(
    artifacts_dir: Path,
    trajectory_file: str,
    device: torch.device,
    sequence_length: int,
    hidden_dim: int,
    temporal_dim: int,
    num_residual_blocks: int,
    dropout: float,
) -> TrajectoryGenerator:
    checkpoint_path = artifacts_dir / trajectory_file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trajectory generator checkpoint: {checkpoint_path}")

    generator = TrajectoryGenerator(
        input_dim=512,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        temporal_dim=temporal_dim,
        num_residual_blocks=num_residual_blocks,
        dropout=dropout,
        output_channels=1,
        use_logit_norm=False,
    )
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.to(device)
    generator.eval()
    return generator


def collect_outputs(dataset: Dataset, backbone: torch.nn.Module, batch_size: int, device: torch.device) -> GroupOutputs:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_batches: List[np.ndarray] = []
    logit_batches: List[np.ndarray] = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            features, logits = forward_with_features(backbone, images)
            feature_batches.append(features.cpu().numpy().astype(np.float32))
            logit_batches.append(logits.cpu().numpy().astype(np.float32))

    features_np = np.concatenate(feature_batches, axis=0)
    logits_np = np.concatenate(logit_batches, axis=0)
    return GroupOutputs(name="", features=features_np, logits=logits_np, size=features_np.shape[0])


def fit_mahalanobis(train_features: np.ndarray, train_labels: np.ndarray, num_classes: int, reg: float) -> Tuple[np.ndarray, np.ndarray]:
    train_features = train_features.astype(np.float64)
    train_labels = train_labels.astype(np.int64)
    feature_dim = train_features.shape[1]

    class_means = np.zeros((num_classes, feature_dim), dtype=np.float64)
    for class_index in range(num_classes):
        class_mask = train_labels == class_index
        if not np.any(class_mask):
            raise ValueError(f"No training samples found for class {class_index}")
        class_means[class_index] = train_features[class_mask].mean(axis=0)

    centered = train_features - class_means[train_labels]
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    covariance += reg * np.eye(feature_dim, dtype=np.float64)
    precision = np.linalg.pinv(covariance)
    return class_means, precision


def score_mahalanobis(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    scores: List[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].astype(np.float64)
        diffs = batch[:, None, :] - class_means[None, :, :]
        transformed = np.einsum("bcd,df->bcf", diffs, precision)
        distances = np.einsum("bcf,bcf->bc", transformed, diffs)
        scores.append((-np.min(distances, axis=1)).astype(np.float32))
    return np.concatenate(scores, axis=0)


def score_msp(logits: np.ndarray) -> np.ndarray:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)
    return probabilities.max(dim=-1).values.cpu().numpy().astype(np.float32)


def score_logitnorm(logits: np.ndarray, tau: float, mode: str) -> np.ndarray:
    logits_tensor = torch.from_numpy(logits)
    logit_norm = LogitNorm(tau=tau)
    normalized = logit_norm(logits_tensor)

    if mode == "msp":
        certainty = torch.softmax(normalized, dim=-1).max(dim=-1).values
    else:
        certainty = -torch.logsumexp(normalized, dim=-1)

    return certainty.cpu().numpy().astype(np.float32)


def score_trajectory(
    features: np.ndarray,
    generator: TrajectoryGenerator,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> np.ndarray:
    certainty_batches: List[np.ndarray] = []

    for start in range(0, features.shape[0], batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        with torch.no_grad():
            predicted_trajectory = generator(batch).cpu().numpy()
        if sequence_length > 1:
            certainty = np.trapz(predicted_trajectory, dx=1.0, axis=1) / float(sequence_length - 1)
        else:
            certainty = predicted_trajectory[:, 0]
        certainty_batches.append(certainty.astype(np.float32))

    return np.concatenate(certainty_batches, axis=0)


def evaluate_binary_task(
    clean_certainty: np.ndarray,
    target_certainty: np.ndarray,
) -> Tuple[float, float, float]:
    labels = np.concatenate([
        np.zeros(clean_certainty.shape[0], dtype=np.int32),
        np.ones(target_certainty.shape[0], dtype=np.int32),
    ])
    certainty = np.concatenate([clean_certainty, target_certainty])

    auroc = roc_auc_score(labels, -certainty)
    auprc = average_precision_score(labels, -certainty)
    _, _, aurc = compute_risk_coverage(labels, certainty)
    return float(auroc), float(auprc), float(aurc)


def build_groups(
    spec: DatasetSpec,
    data_root: str,
    repo_root: str | None,
    svhn_root: str,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, np.ndarray]:
    trainset_clean, noise_data = spec.train_setup_fn(data_root=data_root, repo_root=repo_root)
    clean_labels = np.asarray(trainset_clean.targets, dtype=np.int64)
    noisy_labels = np.asarray(noise_data[spec.train_noise_key], dtype=np.int64)
    noise_mask = noisy_labels != np.asarray(noise_data[spec.clean_label_key], dtype=np.int64)
    noisy_indices = np.flatnonzero(noise_mask)
    clean_indices = np.flatnonzero(~noise_mask)
    clean_fit_indices, clean_val_indices = stratified_split_indices(
        clean_indices,
        clean_labels,
        val_fraction=0.2,
        seed=42,
    )

    base_name = "cifar10" if spec.name == "cifar10n" else "cifar100"
    clean_id = Subset(trainset_clean, clean_val_indices.tolist())
    near_ood_name = "cifar100" if base_name == "cifar10" else "cifar10"
    near_ood = build_standard_dataset(near_ood_name, root=data_root, train=False)
    far_ood = build_svhn_dataset(svhn_root)
    noisy_id = Subset(trainset_clean, noisy_indices.tolist())
    clean_fit = Subset(trainset_clean, clean_fit_indices.tolist())

    logging.info(
        "Group sizes | Clean ID(val): %d | Clean-fit subset: %d | Noisy ID(train subset): %d | Near-OOD(%s): %d | Far-OOD(SVHN): %d",
        len(clean_id),
        len(clean_fit),
        len(noisy_id),
        spec.opposite_name,
        len(near_ood),
        len(far_ood),
    )
    logging.info(
        "Noisy subset rate in ID training split: %.2f%% (%d / %d)",
        100.0 * float(noise_mask.mean()),
        int(noise_mask.sum()),
        noise_mask.shape[0],
    )

    return clean_fit, noisy_id, clean_id, near_ood, far_ood, clean_labels[clean_fit_indices]


def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASET_CONFIG[args.dataset]
    artifacts_dir = Path(args.artifacts_dir or spec.artifacts_dir)
    reports_dir = Path(args.reports_dir or spec.reports_dir)

    backbone = load_backbone(artifacts_dir, spec.backbone_file, spec.num_classes, device)
    trajectory_generator = load_trajectory_generator(
        artifacts_dir=artifacts_dir,
        trajectory_file=spec.trajectory_file,
        device=device,
        sequence_length=args.trajectory_sequence_length,
        hidden_dim=args.trajectory_hidden_dim,
        temporal_dim=args.trajectory_temporal_dim,
        num_residual_blocks=args.trajectory_num_residual_blocks,
        dropout=args.trajectory_dropout,
    )

    clean_fit, noisy_id, clean_id, near_ood, far_ood, train_labels = build_groups(
        spec=spec,
        data_root=args.data_root,
        repo_root=args.repo_root,
        svhn_root=args.svhn_root,
    )

    logging.info("Collecting backbone outputs for all groups...")
    clean_outputs = collect_outputs(clean_id, backbone, args.batch_size, device)
    noisy_outputs = collect_outputs(noisy_id, backbone, args.batch_size, device)
    near_outputs = collect_outputs(near_ood, backbone, args.batch_size, device)
    far_outputs = collect_outputs(far_ood, backbone, args.batch_size, device)

    logging.info("Fitting Mahalanobis baseline on clean ID training features...")
    train_outputs = collect_outputs(clean_fit, backbone, args.score_batch_size, device)
    class_means, precision = fit_mahalanobis(train_outputs.features, train_labels, spec.num_classes, args.mahalanobis_reg)

    logging.info("Computing certainty scores for baseline methods...")
    msp_scores = {
        "clean_id": score_msp(clean_outputs.logits),
        "noisy_id": score_msp(noisy_outputs.logits),
        "near_ood": score_msp(near_outputs.logits),
        "far_ood": score_msp(far_outputs.logits),
    }
    logitnorm_scores = {
        "clean_id": score_logitnorm(clean_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "noisy_id": score_logitnorm(noisy_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "near_ood": score_logitnorm(near_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "far_ood": score_logitnorm(far_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
    }
    mahalanobis_scores = {
        "clean_id": score_mahalanobis(clean_outputs.features, class_means, precision, args.score_batch_size),
        "noisy_id": score_mahalanobis(noisy_outputs.features, class_means, precision, args.score_batch_size),
        "near_ood": score_mahalanobis(near_outputs.features, class_means, precision, args.score_batch_size),
        "far_ood": score_mahalanobis(far_outputs.features, class_means, precision, args.score_batch_size),
    }
    trajectory_scores = {
        "clean_id": score_trajectory(
            clean_outputs.features,
            trajectory_generator,
            args.score_batch_size,
            args.trajectory_sequence_length,
            device,
        ),
        "noisy_id": score_trajectory(
            noisy_outputs.features,
            trajectory_generator,
            args.score_batch_size,
            args.trajectory_sequence_length,
            device,
        ),
        "near_ood": score_trajectory(
            near_outputs.features,
            trajectory_generator,
            args.score_batch_size,
            args.trajectory_sequence_length,
            device,
        ),
        "far_ood": score_trajectory(
            far_outputs.features,
            trajectory_generator,
            args.score_batch_size,
            args.trajectory_sequence_length,
            device,
        ),
    }

    task_pairs = [
        ("clean_id", "noisy_id", "Clean ID vs Noisy ID"),
        ("clean_id", "near_ood", f"Clean ID vs Near-OOD ({spec.opposite_name})"),
        ("clean_id", "far_ood", "Clean ID vs Far-OOD (SVHN)"),
    ]
    method_scores = {
        "MSP": msp_scores,
        f"LogitNorm-{args.logitnorm_score.upper()}": logitnorm_scores,
        "Mahalanobis": mahalanobis_scores,
        "Trajectory": trajectory_scores,
    }

    results: List[MetricResult] = []
    for method_name, scores_by_group in method_scores.items():
        for clean_group, target_group, task_name in task_pairs:
            auroc, auprc, aurc = evaluate_binary_task(
                scores_by_group[clean_group],
                scores_by_group[target_group],
            )
            results.append(MetricResult(method=method_name, task=task_name, auroc=auroc, auprc=auprc, aurc=aurc))

    summary_path = reports_dir / f"ood_summary_{spec.name}.csv"
    save_summary_csv(results, summary_path)

    logging.info(
        "Evaluation protocol: clean ID=test split with correct labels; noisy ID=train subset with human-flipped labels; near-OOD=%s test split; far-OOD=SVHN.",
        spec.opposite_name,
    )
    logging.info(
        "Scores are reported for MSP, LogitNorm-%s, Mahalanobis, and Trajectory prediction; the comparison is descriptive and should be read as evidence of complementary behavior rather than unconditional dominance.",
        args.logitnorm_score.upper(),
    )
    logging.info("Summary CSV saved to %s", summary_path)

    for row in results:
        logging.info(
            "%s | %s | AUROC: %.4f | AUPRC: %.4f | AURC: %.4f",
            row.method,
            row.task,
            row.auroc,
            row.auprc,
            row.aurc,
        )


def main() -> None:
    configure_logging()
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

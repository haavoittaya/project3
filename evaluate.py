from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from scipy.integrate import trapezoid
except ImportError:
    try:
        from numpy import trapezoid
    except ImportError:
        from numpy import trapz as trapezoid

# =======================================================
# NORMALIZATION STATS & TRANSFORMS
# =======================================================
CIFAR_STATS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "cifar10n": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100n": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


def get_eval_transform(dataset_name: str) -> transforms.Compose:
    stats = CIFAR_STATS.get(dataset_name, CIFAR_STATS["cifar10n"])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    backbone_file: str
    predictor_file: str
    trajectory_file: str
    features_file: str
    artifacts_dir: str
    reports_dir: str
    train_setup_fn_name: str
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
        predictor_file="baseline_binary_predictor.pth",
        trajectory_file="trajectory_generator.pth",
        features_file="X_features_cifar10n.npy",
        artifacts_dir="./artifacts",
        reports_dir="./artifacts/reports",
        train_setup_fn_name="setup_cifar10n",
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
        predictor_file="baseline_binary_predictor.pth",
        trajectory_file="trajectory_generator.pth",
        features_file="X_features_cifar100n.npy",
        artifacts_dir="./artifacts_cifar100n",
        reports_dir="./artifacts_cifar100n/reports",
        train_setup_fn_name="setup_cifar100n",
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
        description="Unified Evaluation Protocol comparing Baselines, Trajectory Distillation, and Generative CVAE Models."
    )
    parser.add_argument("--dataset", type=str, choices=("cifar10n", "cifar100n"), default="cifar10n")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--svhn-root", type=str, default="./data_svhn")
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logitnorm-tau", type=float, default=1.0)
    parser.add_argument("--logitnorm-score", type=str, choices=("msp", "energy"), default="msp")
    parser.add_argument("--mahalanobis-reg", type=float, default=1e-3)
    parser.add_argument("--trajectory-sequence-length", type=int, default=50)
    parser.add_argument("--trajectory-hidden-dim", type=int, default=256)
    parser.add_argument("--trajectory-temporal-dim", type=int, default=128)
    parser.add_argument("--trajectory-num-residual-blocks", type=int, default=3)
    parser.add_argument("--trajectory-dropout", type=float, default=0.1)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--cvae-samples", type=int, default=30)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def compute_risk_coverage(error_labels: np.ndarray, certainty_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if error_labels.ndim != 1 or certainty_scores.ndim != 1:
        raise ValueError("error_labels and certainty_scores must be 1D arrays")
    if error_labels.shape[0] != certainty_scores.shape[0]:
        raise ValueError("error_labels and certainty_scores must have the same length")

    order = np.argsort(-certainty_scores)
    sorted_errors = error_labels[order].astype(np.float64)
    ranks = np.arange(1, sorted_errors.shape[0] + 1, dtype=np.float64)

    coverage = ranks / ranks[-1]
    risk = np.cumsum(sorted_errors) / ranks

    aurc = float(trapezoid(risk, coverage))
    return coverage, risk, aurc


def save_summary_csv(results: List[MetricResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "method", "auroc", "auprc", "aurc"])
        for row in results:
            writer.writerow([row.task, row.method, f"{row.auroc:.6f}", f"{row.auprc:.6f}", f"{row.aurc:.6f}"])


def build_standard_dataset(name: str, root: str, train: bool, transform: transforms.Compose) -> Dataset:
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def build_svhn_dataset(root: str, transform: transforms.Compose) -> Dataset:
    return torchvision.datasets.SVHN(root=root, split="test", download=True, transform=transform)


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


# =======================================================
# MODEL LOADERS (WITH SAFE FALLBACKS)
# =======================================================

def load_backbone(artifacts_dir: Path, backbone_file: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    from src.models import get_resnet18_backbone

    backbone_path = artifacts_dir / backbone_file
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing backbone checkpoint: {backbone_path}")

    backbone = get_resnet18_backbone(num_classes=num_classes)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
    backbone.to(device).eval()
    return backbone


def load_baseline_predictor(artifacts_dir: Path, predictor_file: str, device: torch.device) -> Optional[torch.nn.Module]:
    from src.models import AdvancedMLP

    predictor_path = artifacts_dir / predictor_file
    if not predictor_path.exists():
        logging.warning("Missing Image-Only Predictor checkpoint: %s. Skipping this baseline.", predictor_path)
        return None

    try:
        predictor = AdvancedMLP(input_dim=512, output_dim=1, hidden_dim=256, dropout=0.2)
    except TypeError:
        predictor = AdvancedMLP()

    predictor.load_state_dict(torch.load(predictor_path, map_location=device, weights_only=True))
    predictor.to(device).eval()
    return predictor


def load_trajectory_generator(
    artifacts_dir: Path, trajectory_file: str, device: torch.device,
    sequence_length: int, hidden_dim: int, temporal_dim: int, num_residual_blocks: int, dropout: float,
) -> Optional[torch.nn.Module]:
    from src.models import TrajectoryGenerator

    checkpoint_path = artifacts_dir / trajectory_file
    if not checkpoint_path.exists():
        logging.warning("Missing trajectory generator checkpoint: %s. Skipping.", checkpoint_path)
        return None

    generator = TrajectoryGenerator(
        input_dim=512, sequence_length=sequence_length, hidden_dim=hidden_dim,
        temporal_dim=temporal_dim, num_residual_blocks=num_residual_blocks,
        dropout=dropout, output_channels=1, use_logit_norm=False,
    )
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.to(device).eval()
    return generator


def load_mlp_head(artifacts_dir: Path, device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, np.ndarray]]]:
    from src.models import AdvancedMLP

    mlp_path = artifacts_dir / "mlp_descriptor_head.pth"
    stats_path = artifacts_dir / "descriptors_stats.npz"

    if not mlp_path.exists():
        logging.warning("Missing AdvancedMLP checkpoint: %s. Skipping.", mlp_path)
        return None, None

    try:
        mlp = AdvancedMLP(input_dim=512, output_dim=4, hidden_dim=256, dropout=0.2)
    except TypeError:
        try:
            mlp = AdvancedMLP(input_dim=512)
        except TypeError:
            mlp = AdvancedMLP()

    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.to(device).eval()

    stats = None
    if stats_path.exists():
        data = np.load(stats_path)
        stats = {"mean": data["mean"], "std": data["std"]}
        logging.info("Loaded descriptor normalization stats: mean=%s, std=%s", data["mean"], data["std"])
    else:
        logging.warning("Missing descriptors normalization stats: %s. MLP scores may be in normalized scale.", stats_path)

    return mlp, stats


def load_cvae_model(artifacts_dir: Path, sequence_length: int, device: torch.device) -> Optional[torch.nn.Module]:
    from src.models import TrajectoryCVAE

    cvae_path = artifacts_dir / "trajectory_cvae.pth"
    if not cvae_path.exists():
        logging.warning("Missing CVAE checkpoint: %s. Skipping.", cvae_path)
        return None

    cvae = TrajectoryCVAE(
        input_dim=512, sequence_length=sequence_length, num_classes=1, latent_dim=128, hidden_dim=512, output_activation="none"
    )
    cvae.load_state_dict(torch.load(cvae_path, map_location=device, weights_only=True))
    cvae.to(device).eval()
    return cvae


def collect_outputs(dataset: Dataset, backbone: torch.nn.Module, batch_size: int, device: torch.device) -> GroupOutputs:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )
    feature_batches, logit_batches = [], []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            features, logits = forward_with_features(backbone, images)
            feature_batches.append(features.cpu().numpy().astype(np.float32))
            logit_batches.append(logits.cpu().numpy().astype(np.float32))

    features_np = np.concatenate(feature_batches, axis=0)
    logits_np = np.concatenate(logit_batches, axis=0)
    return GroupOutputs(name="", features=features_np, logits=logits_np, size=features_np.shape[0])


def fit_mahalanobis(train_features: np.ndarray, train_labels: np.ndarray, num_classes: int, reg: float) -> Tuple[np.ndarray, np.ndarray]:
    train_features, train_labels = train_features.astype(np.float64), train_labels.astype(np.int64)
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
    return class_means, np.linalg.pinv(covariance)


# =======================================================
# SCORING FUNCTIONS
# =======================================================

def score_msp(logits: np.ndarray) -> np.ndarray:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)
    return probabilities.max(dim=-1).values.cpu().numpy().astype(np.float32).ravel()


def score_logitnorm(logits: np.ndarray, tau: float, mode: str) -> np.ndarray:
    from src.models import LogitNorm

    logits_tensor = torch.from_numpy(logits)
    logit_norm = LogitNorm(tau=tau)
    normalized = logit_norm(logits_tensor)

    if mode == "msp":
        certainty = torch.softmax(normalized, dim=-1).max(dim=-1).values
    else:
        certainty = torch.logsumexp(normalized, dim=-1)

    return certainty.cpu().numpy().astype(np.float32).ravel()


def score_mahalanobis(features: np.ndarray, class_means: np.ndarray, precision: np.ndarray, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].astype(np.float64)
        diffs = batch[:, None, :] - class_means[None, :, :]
        transformed = np.einsum("bcd,df->bcf", diffs, precision)
        distances = np.einsum("bcf,bcf->bc", transformed, diffs)
        scores.append((-np.min(distances, axis=1)).astype(np.float32))
    return np.concatenate(scores, axis=0).ravel()


def score_baseline_predictor(features: np.ndarray, predictor: torch.nn.Module, batch_size: int, device: torch.device) -> np.ndarray:
    certainty_batches = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = predictor(batch)
            probabilities = torch.sigmoid(logits).view(-1)
            certainty_batches.append(probabilities.cpu().numpy().astype(np.float32))
    return np.concatenate(certainty_batches, axis=0).ravel()


def score_trajectory(features: np.ndarray, generator: torch.nn.Module, batch_size: int, sequence_length: int, device: torch.device) -> np.ndarray:
    certainty_batches = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        with torch.no_grad():
            predicted_trajectory = generator(batch)
            if predicted_trajectory.ndim == 3:
                predicted_trajectory = predicted_trajectory.squeeze(-1)
            if sequence_length > 1:
                certainty = (predicted_trajectory[:, :-1] + predicted_trajectory[:, 1:]).mean(dim=1) * 0.5
            else:
                certainty = predicted_trajectory[:, 0]
            certainty_batches.append(certainty.cpu().numpy().astype(np.float32).ravel())
    return np.concatenate(certainty_batches, axis=0).ravel()


def score_mlp(features: np.ndarray, mlp: torch.nn.Module, stats: Optional[Dict[str, np.ndarray]],
              batch_size: int, device: torch.device) -> np.ndarray:
    certainty_batches = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        with torch.no_grad():
            out = mlp(batch)
            if out.shape[-1] == 4:
                pred_aum = out[:, 0]
                if stats is not None:
                    pred_aum = pred_aum * stats["std"][0] + stats["mean"][0]
                certainty_batches.append(pred_aum.cpu().numpy().astype(np.float32))
            elif out.shape[-1] == 2:
                prob_ood = torch.softmax(out, dim=-1)[:, 1]
                certainty_batches.append((-prob_ood).cpu().numpy().astype(np.float32))
            else:
                prob_ood = torch.sigmoid(out).view(-1)
                certainty_batches.append((-prob_ood).cpu().numpy().astype(np.float32))
    return np.concatenate(certainty_batches, axis=0).ravel()


def score_cvae(features: np.ndarray, cvae: torch.nn.Module, batch_size: int, num_samples: int,
               sequence_length: int, device: torch.device, mode: str = "mc_margin") -> np.ndarray:
    certainty_batches = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        curr_batch_size = batch.size(0)

        with torch.no_grad():
            if hasattr(cvae, "generate"):
                batch_repeated = batch.repeat_interleave(num_samples, dim=0)
                sample = cvae.generate(batch_repeated)
            else:
                batch_repeated = batch.repeat_interleave(num_samples, dim=0)
                sample = cvae(batch_repeated)[0]

            sample = sample.squeeze(-1).view(curr_batch_size, num_samples, sequence_length)

            if mode == "mc_margin":
                certainty = sample.mean(dim=-1).mean(dim=1)
            else:  # "uncertainty" (-std)
                uncertainty = sample.std(dim=1).mean(dim=1)
                certainty = -uncertainty

            certainty_batches.append(certainty.cpu().numpy().astype(np.float32))

    return np.concatenate(certainty_batches, axis=0).ravel()


# =======================================================
# MAIN EVALUATION PIPELINE
# =======================================================

def evaluate_binary_task(clean_certainty: np.ndarray, target_certainty: np.ndarray) -> Tuple[float, float, float]:
    labels = np.concatenate([np.zeros(clean_certainty.shape[0], dtype=np.int32), np.ones(target_certainty.shape[0], dtype=np.int32)])
    certainty = np.concatenate([clean_certainty, target_certainty]).ravel()
    return float(roc_auc_score(labels, -certainty)), float(average_precision_score(labels, -certainty)), compute_risk_coverage(labels, certainty)[2]


def build_groups(spec: DatasetSpec, data_root: str, repo_root: str | None, svhn_root: str) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset, np.ndarray]:
    import src.dataset as dataset_module

    eval_transform = get_eval_transform(spec.name)
    train_setup_fn = getattr(dataset_module, spec.train_setup_fn_name)

    trainset_clean, noise_data = train_setup_fn(data_root=data_root, repo_root=repo_root, transform=eval_transform)
    clean_labels = np.asarray(trainset_clean.targets, dtype=np.int64)
    noisy_labels = np.asarray(noise_data[spec.train_noise_key], dtype=np.int64)
    noise_mask = noisy_labels != np.asarray(noise_data[spec.clean_label_key], dtype=np.int64)

    # === Reproducible 50/50 split ===
    rng = np.random.default_rng(42)
    indices = np.arange(len(trainset_clean))
    rng.shuffle(indices)
    split_point = int(len(indices) * 0.5)

    meta_train_indices = indices[:split_point]
    meta_test_indices = indices[split_point:]

    clean_mask = ~noise_mask
    clean_fit_indices = np.intersect1d(meta_train_indices, np.flatnonzero(clean_mask))
    clean_id_eval_indices = np.intersect1d(meta_test_indices, np.flatnonzero(clean_mask))
    noisy_id_indices = np.intersect1d(meta_test_indices, np.flatnonzero(noise_mask))

    clean_id_test = spec.clean_test_ctor(
        root=data_root,
        train=False,
        download=True,
        transform=eval_transform
    )

    base_name = "cifar10" if spec.name == "cifar10n" else "cifar100"

    return (
        Subset(trainset_clean, clean_fit_indices.tolist()),       # clean_fit
        Subset(trainset_clean, noisy_id_indices.tolist()),         # noisy_id
        Subset(trainset_clean, clean_id_eval_indices.tolist()),    # clean_id_eval (Meta-Test)
        clean_id_test,                                             # clean_id_test
        build_standard_dataset("cifar100" if base_name == "cifar10" else "cifar10", root=data_root, train=False, transform=eval_transform),
        build_svhn_dataset(svhn_root, transform=eval_transform),
        clean_labels[clean_fit_indices]
    )


def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASET_CONFIG[args.dataset]
    artifacts_dir = Path(args.artifacts_dir or spec.artifacts_dir)
    reports_dir = Path(args.reports_dir or spec.reports_dir)

    logging.info("Loading backbone model...")
    backbone = load_backbone(artifacts_dir, spec.backbone_file, spec.num_classes, device)

    logging.info("Loading baseline, trajectory, and generative models...")
    baseline_predictor = load_baseline_predictor(artifacts_dir, spec.predictor_file, device)
    trajectory_generator = load_trajectory_generator(
        artifacts_dir, spec.trajectory_file, device, args.trajectory_sequence_length,
        args.trajectory_hidden_dim, args.trajectory_temporal_dim, args.trajectory_num_residual_blocks, args.trajectory_dropout
    )
    mlp_head, mlp_stats = load_mlp_head(artifacts_dir, device)
    cvae_model = load_cvae_model(artifacts_dir, args.trajectory_sequence_length, device)

    clean_fit, noisy_id, clean_id_eval, clean_id_test, near_ood, far_ood, train_labels = build_groups(
        spec, args.data_root, args.repo_root, args.svhn_root
    )

    logging.info("Collecting backbone outputs for all groups...")
    clean_eval_outputs = collect_outputs(clean_id_eval, backbone, args.batch_size, device)
    clean_test_outputs = collect_outputs(clean_id_test, backbone, args.batch_size, device)
    noisy_outputs       = collect_outputs(noisy_id, backbone, args.batch_size, device)
    near_outputs        = collect_outputs(near_ood, backbone, args.batch_size, device)
    far_outputs         = collect_outputs(far_ood, backbone, args.batch_size, device)

    logging.info("Fitting Mahalanobis baseline on clean ID training features...")
    class_means, precision = fit_mahalanobis(
        collect_outputs(clean_fit, backbone, args.score_batch_size, device).features,
        train_labels,
        spec.num_classes,
        args.mahalanobis_reg
    )

    logging.info("Computing certainty scores across ALL evaluation methods...")

    method_scores: Dict[str, Dict[str, np.ndarray]] = {}

    # 1. Baseline Logit / Softmax
    method_scores["Maximum Softmax Prob (MSP)"] = {
        "clean_id_eval": score_msp(clean_eval_outputs.logits),
        "clean_id_test":  score_msp(clean_test_outputs.logits),
        "noisy_id":       score_msp(noisy_outputs.logits),
        "near_ood":       score_msp(near_outputs.logits),
        "far_ood":        score_msp(far_outputs.logits)
    }

    method_scores["Energy Score (LogitNorm)"] = {
        "clean_id_eval": score_logitnorm(clean_eval_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "clean_id_test":  score_logitnorm(clean_test_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "noisy_id":       score_logitnorm(noisy_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "near_ood":       score_logitnorm(near_outputs.logits, args.logitnorm_tau, args.logitnorm_score),
        "far_ood":        score_logitnorm(far_outputs.logits, args.logitnorm_tau, args.logitnorm_score)
    }

    # 2. Representation Distance Baseline
    method_scores["Mahalanobis Distance"] = {
        "clean_id_eval": score_mahalanobis(clean_eval_outputs.features, class_means, precision, args.score_batch_size),
        "clean_id_test":  score_mahalanobis(clean_test_outputs.features, class_means, precision, args.score_batch_size),
        "noisy_id":       score_mahalanobis(noisy_outputs.features, class_means, precision, args.score_batch_size),
        "near_ood":       score_mahalanobis(near_outputs.features, class_means, precision, args.score_batch_size),
        "far_ood":        score_mahalanobis(far_outputs.features, class_means, precision, args.score_batch_size)
    }

    # 3. Image-Only Binary Predictor
    if baseline_predictor is not None:
        method_scores["Image-Only Binary Predictor"] = {
            "clean_id_eval": score_baseline_predictor(clean_eval_outputs.features, baseline_predictor, args.score_batch_size, device),
            "clean_id_test":  score_baseline_predictor(clean_test_outputs.features, baseline_predictor, args.score_batch_size, device),
            "noisy_id":       score_baseline_predictor(noisy_outputs.features, baseline_predictor, args.score_batch_size, device),
            "near_ood":       score_baseline_predictor(near_outputs.features, baseline_predictor, args.score_batch_size, device),
            "far_ood":        score_baseline_predictor(far_outputs.features, baseline_predictor, args.score_batch_size, device)
        }

    # 4. Trajectory Generator
    if trajectory_generator is not None:
        method_scores["Deterministic Trajectory Gen"] = {
            "clean_id_eval": score_trajectory(clean_eval_outputs.features, trajectory_generator, args.score_batch_size, args.trajectory_sequence_length, device),
            "clean_id_test":  score_trajectory(clean_test_outputs.features, trajectory_generator, args.score_batch_size, args.trajectory_sequence_length, device),
            "noisy_id":       score_trajectory(noisy_outputs.features, trajectory_generator, args.score_batch_size, args.trajectory_sequence_length, device),
            "near_ood":       score_trajectory(near_outputs.features, trajectory_generator, args.score_batch_size, args.trajectory_sequence_length, device),
            "far_ood":        score_trajectory(far_outputs.features, trajectory_generator, args.score_batch_size, args.trajectory_sequence_length, device)
        }

    # 5. Distilled / Parametric 4D MLP Head
    if mlp_head is not None:
        method_scores["Distilled 4D MLP Head (AUM)"] = {
            "clean_id_eval": score_mlp(clean_eval_outputs.features, mlp_head, mlp_stats, args.score_batch_size, device),
            "clean_id_test":  score_mlp(clean_test_outputs.features, mlp_head, mlp_stats, args.score_batch_size, device),
            "noisy_id":       score_mlp(noisy_outputs.features, mlp_head, mlp_stats, args.score_batch_size, device),
            "near_ood":       score_mlp(near_outputs.features, mlp_head, mlp_stats, args.score_batch_size, device),
            "far_ood":        score_mlp(far_outputs.features, mlp_head, mlp_stats, args.score_batch_size, device)
        }

    # 6. Trajectory CVAE Generative Model
    if cvae_model is not None:
        method_scores["Trajectory CVAE (MC Mean Margin)"] = {
            "clean_id_eval": score_cvae(clean_eval_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="mc_margin"),
            "clean_id_test":  score_cvae(clean_test_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="mc_margin"),
            "noisy_id":       score_cvae(noisy_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="mc_margin"),
            "near_ood":       score_cvae(near_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="mc_margin"),
            "far_ood":        score_cvae(far_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="mc_margin")
        }
        method_scores["Trajectory CVAE (MC Uncertainty)"] = {
            "clean_id_eval": score_cvae(clean_eval_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="uncertainty"),
            "clean_id_test":  score_cvae(clean_test_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="uncertainty"),
            "noisy_id":       score_cvae(noisy_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="uncertainty"),
            "near_ood":       score_cvae(near_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="uncertainty"),
            "far_ood":        score_cvae(far_outputs.features, cvae_model, args.score_batch_size, args.cvae_samples, args.trajectory_sequence_length, device, mode="uncertainty")
        }

    task_pairs = [
        ("clean_id_eval", "noisy_id", "Task 1: Clean ID (Meta-Test) vs Noisy ID"),
        ("clean_id_test",  "near_ood", f"Task 2: Clean ID (Test) vs Near-OOD ({spec.opposite_name})"),
        ("clean_id_test",  "far_ood",  "Task 3: Clean ID (Test) vs Far-OOD (SVHN)"),
    ]

    ordered_results: List[MetricResult] = []

    print("\n" + "="*95)
    print(" UNIFIED COMPREHENSIVE BENCHMARK SUMMARY ".center(95, "="))
    print("="*95)

    for clean_group, target_group, task_name in task_pairs:
        print(f"\n▶ {task_name.upper()}")
        print("-" * 95)
        print(f"{'Method Name':<38} | {'AUROC ↑':<10} | {'AUPRC ↑':<10} | {'AURC ↓':<10}")
        print("-" * 95)

        for method_name, scores_by_group in method_scores.items():
            auroc, auprc, aurc = evaluate_binary_task(scores_by_group[clean_group], scores_by_group[target_group])
            res = MetricResult(method=method_name, task=task_name, auroc=auroc, auprc=auprc, aurc=aurc)
            ordered_results.append(res)
            print(f"{res.method:<38} | {res.auroc:<10.4f} | {res.auprc:<10.4f} | {res.aurc:<10.4f}")
        print("-" * 95)

    summary_path = reports_dir / f"unified_summary_{spec.name}.csv"
    save_summary_csv(ordered_results, summary_path)
    logging.info("Unified Summary CSV saved to %s", summary_path)


def main() -> None:
    configure_logging()
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
# Distilling Training Dynamics into Inference-Time Uncertainty Descriptors

This repository implements a two-stage pipeline for distilling training dynamics into lightweight inference-time uncertainty descriptors.

## Project Structure

- `src/models.py`: network components (ResNet-18 backbone, feature extractor, MLP head)
- `src/dataset.py`: CIFAR-10/CIFAR-10N data setup and repository bootstrap
- `train_stage1.py`: trajectory tracking and backbone training
- `train_stage2.py`: descriptor target construction and MLP distillation
- `evaluate_ood.py`: noisy-label, OOD, and robustness evaluation

## Requirements

Install dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Data and Artifacts

- CIFAR-10 is downloaded automatically via torchvision.
- CIFAR-10N metadata is cloned automatically from `UCSC-REAL/cifar-10-100n`.
- Stage outputs are saved to `./artifacts` by default.

Expected artifacts after full run:

- `artifacts/softmax_history_cifar10n.npy`
- `artifacts/margin_history_cifar10n.npy`
- `artifacts/resnet18_backbone.pth`
- `artifacts/mlp_4d_cifar10n.pth`
- `artifacts/X_features.npy`
- `artifacts/Y_targets_4d.npy`

## Stage 1: Trajectory Tracking

Train the backbone on CIFAR-10N aggregate labels and store trajectory statistics:

```powershell
python train_stage1.py --epochs 50 --batch-size 256 --lr 0.001 --seed 42
```

## Stage 2: Descriptor Distillation

Compute 4D descriptor targets (AUM, mean confidence, confidence variability, forgetting count) and train the MLP head:

```powershell
python train_stage2.py --epochs 40 --batch-size 512 --lr 0.005 --seed 42
```

## Stage 3: Evaluation

Run evaluation on noisy-label detection, OOD detection (CIFAR-10 vs SVHN), and synthetic corruption robustness:

```powershell
python evaluate_ood.py --batch-size 256
```

Reported metrics include:

- AUROC and AUPRC for noisy-label detection
- AUROC and AUPRC for OOD detection
- Risk-Coverage based AURC for selective classification behavior

Evaluation also exports Risk-Coverage reports to `./artifacts/reports` by default:

- `worse_split_risk_coverage.csv`
- `worse_split_risk_coverage.png`
- `ood_cifar10_vs_svhn_risk_coverage.csv`
- `ood_cifar10_vs_svhn_risk_coverage.png`

## Optional Paths

You can override default paths in all scripts:

- `--data-root` for CIFAR-10 data
- `--repo-root` for local CIFAR-10N repository path
- `--artifacts-dir` for saved artifacts
- `--svhn-root` in `evaluate_ood.py`
- `--reports-dir` in `evaluate_ood.py`

## Reproducibility Notes

- All scripts support `--seed` for deterministic random initialization.
- Device selection is automatic: CUDA if available, otherwise CPU.

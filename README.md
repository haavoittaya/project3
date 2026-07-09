# Distilling Training Dynamics into Inference-Time Uncertainty Descriptors

This repository implements a two-stage pipeline for distilling training dynamics into lightweight inference-time uncertainty descriptors.

## Project Structure

- `src/models.py`: network components (ResNet-18 backbone, feature extractor, MLP head)
- `src/dataset.py`: CIFAR-10/CIFAR-10N data setup and repository bootstrap
- `train_stage1.py`: trajectory tracking and backbone training
- `train_stage2.py`: full trajectory distillation with TrajectoryGenerator
- `evaluate_ood.py`: OOD-centric evaluation across clean ID, noisy ID, near-OOD, and far-OOD groups
- `train_stage1_cifar100n.py`: CIFAR-100N trajectory tracking and backbone training
- `train_stage2_cifar100n.py`: CIFAR-100N descriptor distillation
- `evaluate_ood_cifar100n.py`: CIFAR-100N noisy-label, OOD, and robustness evaluation

## Requirements

Install dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Data and Artifacts

- CIFAR-10 is downloaded automatically via torchvision.
- CIFAR-10N metadata is downloaded automatically as a single `CIFAR-10_human.pt` file from a fast mirror, instead of cloning the full repository.
- CIFAR-100N metadata is downloaded automatically as a single `CIFAR-100_human.pt` file from a fast mirror.
- Stage outputs are saved to `./artifacts` by default.

Expected artifacts after full run:

- `artifacts/softmax_history_cifar10n.npy`
- `artifacts/margin_history_cifar10n.npy`
- `artifacts/resnet18_backbone.pth`
- `artifacts/trajectory_generator.pth`
- `artifacts/X_features_cifar10n.npy`
- `artifacts/trajectory_targets_cifar10n_margin.npy`

Expected artifacts for CIFAR-100N pipeline (default `./artifacts_cifar100n`):

- `artifacts_cifar100n/softmax_history_cifar100n.npy`
- `artifacts_cifar100n/margin_history_cifar100n.npy`
- `artifacts_cifar100n/resnet18_backbone_cifar100n.pth`
- `artifacts_cifar100n/trajectory_generator.pth`
- `artifacts_cifar100n/X_features_cifar100n.npy`
- `artifacts_cifar100n/trajectory_targets_cifar100n_margin.npy`

## Unified Pipeline (Single Scripts)

All main scripts now support both datasets via `--dataset`:

- `cifar10n`
- `cifar100n`

Stage 1 (CIFAR-10N):

```powershell
python train_stage1.py --dataset cifar10n --epochs 50 --batch-size 256 --lr 0.001 --seed 42
```

Stage 1 (CIFAR-100N):

```powershell
python train_stage1.py --dataset cifar100n --epochs 50 --batch-size 256 --lr 0.001 --seed 42
```

Stage 2 (CIFAR-10N):

```powershell
python train_stage2.py --dataset cifar10n --epochs 40 --batch-size 512 --lr 0.005 --seed 42
```

Stage 2 (CIFAR-100N):

```powershell
python train_stage2.py --dataset cifar100n --epochs 40 --batch-size 512 --lr 0.005 --seed 42
```

Evaluation (CIFAR-10N):

```powershell
python evaluate_ood.py --dataset cifar10n --batch-size 256
```

Evaluation (CIFAR-100N):

```powershell
python evaluate_ood.py --dataset cifar100n --batch-size 256
```

## Stage 1: Trajectory Tracking

Train the backbone on CIFAR-10N aggregate labels and store trajectory statistics:

```powershell
python train_stage1.py --epochs 50 --batch-size 256 --lr 0.001 --seed 42
```

## Stage 2: Trajectory Distillation

Train a trajectory generator to regress the full epoch-wise margin sequence:

```powershell
python train_stage2.py --epochs 40 --batch-size 512 --lr 0.005 --seed 42
```

## CIFAR-100N Pipeline

Stage 1 (CIFAR-100N):

```powershell
python train_stage1_cifar100n.py --epochs 50 --batch-size 256 --lr 0.001 --seed 42
```

Stage 2 (CIFAR-100N):

```powershell
python train_stage2_cifar100n.py --epochs 40 --batch-size 512 --lr 0.005 --seed 42
```

Evaluation (CIFAR-100N):

```powershell
python evaluate_ood_cifar100n.py --batch-size 256
```

## OOD-Centric Evaluation

Run the OOD-centric evaluation with:

```powershell
python evaluate_ood.py --dataset cifar10n --batch-size 256
python evaluate_ood.py --dataset cifar100n --batch-size 256
```

The main evaluation script now uses four groups for the selected ID dataset:

- Clean ID: test split with correct labels
- Noisy ID: noisy training subset with human-flipped labels
- Near-OOD: CIFAR-100 when ID is CIFAR-10, and CIFAR-10 when ID is CIFAR-100
- Far-OOD: SVHN test split

It compares four scoring methods:

- Maximum Softmax Probability (MSP)
- LogitNorm + MSP or Energy
- Mahalanobis distance
- Trajectory prediction score from `TrajectoryGenerator`

Evaluation reports AUROC, AUPRC, and AURC for these binary tasks:

- Clean ID vs Noisy ID
- Clean ID vs Near-OOD
- Clean ID vs Far-OOD

The script also writes a summary CSV to the reports directory, for example:

- `artifacts/reports/ood_summary_cifar10n.csv`
- `artifacts_cifar100n/reports/ood_summary_cifar100n.csv`

## Optional Paths

You can override default paths in all scripts:

- `--data-root` for CIFAR-10 data
- `--repo-root` for local CIFAR-10N repository path
- `--artifacts-dir` for saved artifacts
- `--svhn-root` in `evaluate_ood.py`
- `--reports-dir` in `evaluate_ood.py`

## Evaluation Note

The comparison is intended to assess whether trajectory prediction contributes complementary information relative to standard confidence- and distance-based baselines. It does not assume unconditional superiority of any single method.

## Reproducibility Notes

- All scripts support `--seed` for deterministic random initialization.
- Device selection is automatic: CUDA if available, otherwise CPU.

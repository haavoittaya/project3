"""
Reproducible Experiment Pipeline Orchestrator for Robust OOD and Label Noise Evaluation.
Supports sequential pipeline runs (Stage 1 -> Stage 2 [MLP, Trajectory, CVAE] -> Evaluation) 
with strict directory isolation, parameter customization, and seed incrementing.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academic-grade pipeline runner for deterministic and stochastic OOD benchmarking."
    )

    # ==========================================
    # GLOBAL & REPLICATION SETTINGS
    # ==========================================
    global_grp = parser.add_argument_group("Global Settings")
    global_grp.add_argument(
        "--start-seed", type=int, default=42, 
        help="Initial random seed for the replication sequence."
    )
    global_grp.add_argument(
        "--num-runs", type=int, default=3, 
        help="Number of sequential replication pipeline runs."
    )
    global_grp.add_argument(
        "--dataset", type=str, choices=("cifar10n", "cifar100n"), default="cifar100n",
        help="Target benchmark dataset to train and evaluate on."
    )
    global_grp.add_argument(
        "--output-root", type=str, default="./experiments",
        help="Root directory where all seed-isolated subfolders will be generated."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Trajectory sequence length (accepted for orchestrator alignment)."
    )

    # ==========================================
    # STAGE 1: BACKBONE TRAINING SETTINGS
    # ==========================================
    stage1_grp = parser.add_argument_group("Stage 1 (Backbone) Hyperparameters")
    stage1_grp.add_argument(
        "--stage1-script", type=str, default="train_stage1.py",
        help="Filename/path of the backbone model training script."
    )
    stage1_grp.add_argument(
        "--lr1", type=float, default=0.1, help="Learning rate for backbone training."
    )
    stage1_grp.add_argument(
        "--epochs1", type=int, default=120, help="Number of epochs for backbone training."
    )

    # ==========================================
    # STAGE 2: METHOD A - DETERMINISTIC TRAJECTORY
    # ==========================================
    traj_grp = parser.add_argument_group("Stage 2 (Method A): Deterministic Trajectory Generator")
    traj_grp.add_argument(
        "--traj-script", type=str, default="train_stage2.py",
        help="Script path to train the deterministic trajectory regression model."
    )
    traj_grp.add_argument(
        "--lr-traj", type=float, default=1e-3, help="Learning rate for trajectory generator."
    )
    traj_grp.add_argument(
        "--epochs-traj", type=int, default=80, help="Epochs for training trajectory generator."
    )
    traj_grp.add_argument(
        "--seq-len", type=int, default=50, help="Trajectory sequence length (e.g., 50 dimensions)."
    )

    # ==========================================
    # STAGE 2: METHOD B - PARAMETRIC MLP HEAD
    # ==========================================
    mlp_grp = parser.add_argument_group("Stage 2 (Method B): Parametric MLP Head")
    mlp_grp.add_argument(
        "--mlp-script", type=str, default="train_stage2_mlp.py",
        help="Script path to train the parametric MLP OOD classifier."
    )
    mlp_grp.add_argument(
        "--lr-mlp", type=float, default=1e-3, help="Learning rate for MLP classifier."
    )
    mlp_grp.add_argument(
        "--epochs-mlp", type=int, default=40, help="Epochs for training the MLP classifier."
    )

    # ==========================================
    # STAGE 2: METHOD C - STOCHASTIC CVAE MODEL
    # ==========================================
    cvae_grp = parser.add_argument_group("Stage 2 (Method C): Stochastic Trajectory CVAE")
    cvae_grp.add_argument(
        "--cvae-script", type=str, default="train_trajectory_cvae.py",
        help="Script path to train the variational autoencoder trajectory estimator."
    )
    cvae_grp.add_argument(
        "--lr-cvae", type=float, default=5e-4, help="Learning rate for Trajectory CVAE."
    )
    cvae_grp.add_argument(
        "--epochs-cvae", type=int, default=100, help="Epochs for training Trajectory CVAE."
    )
    cvae_grp.add_argument(
        "--latent-dim", type=int, default=128, help="Latent space dimension for CVAE."
    )

    # ==========================================
    # STAGE 3: EVALUATION PROTOCOL
    # ==========================================
    eval_grp = parser.add_argument_group("Evaluation Protocol Settings")
    eval_grp.add_argument(
        "--eval-script", type=str, default="evaluate.py",
        help="Filename/path of the scientific evaluation/OOD metrics script."
    )
    eval_grp.add_argument(
        "--eval-batch-size", type=int, default=256, help="Inference batch size during metrics computation."
    )

    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def execute_subprocess(cmd: List[str], env: dict[str, str], stage_name: str) -> None:
    """Executes a training or evaluation command with stdout streaming and error handling."""
    logging.info("Executing %s...", stage_name)
    logging.info("Command: %s", " ".join(cmd))
    
    start_time = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        logging.error("❌ %s failed with exit code %d.", stage_name, result.returncode)
        sys.exit(result.returncode)
        
    logging.info("✅ %s completed successfully in %.2f seconds.\n", stage_name, elapsed_time)


def main() -> None:
    configure_logging()
    args = parse_args()

    output_root = Path(args.output_root)
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Синхронизация параметров длины последовательности
    final_seq_len = args.seq_len
    if args.sequence_length != 50 and args.seq_len == 50:
        final_seq_len = args.sequence_length
    elif args.seq_len != 50:
        final_seq_len = args.seq_len

    logging.info("=========================================================================")
    logging.info("       EXPERIMENTAL PIPELINE ORCHESTRATOR FOR BENCHMARK REPLICABILITY     ")
    logging.info("=========================================================================")
    logging.info("Target Dataset:  %s", args.dataset)
    logging.info("Initial Seed:    %d", args.start_seed)
    logging.info("Number of Runs:  %d", args.num_runs)
    logging.info("Seq Length:      %d", final_seq_len)
    logging.info("Output Directory: %s", output_root.resolve())
    logging.info("=========================================================================\n")

    for run_idx in range(args.num_runs):
        current_seed = args.start_seed + run_idx
        logging.info(">>> RUN [%d/%d] | INITIALIZING EXPERIMENT WITH SEED = %d <<<", run_idx + 1, args.num_runs, current_seed)

        run_dir = output_root / f"seed_{current_seed}"
        artifacts_dir = run_dir / "artifacts"
        reports_dir = run_dir / "reports"

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # STAGE 1: BACKBONE TRAINING
        # --------------------------------------------------
        cmd_stage1 = [
            sys.executable, args.stage1_script,
            "--dataset", args.dataset,
            "--seed", str(current_seed),
            "--output-dir", str(artifacts_dir),
            "--lr", str(args.lr1),
            "--epochs", str(args.epochs1)
        ]
        execute_subprocess(cmd_stage1, env, f"Stage 1 [Backbone Training - Seed {current_seed}]")

        # --------------------------------------------------
        # STAGE 2 (METHOD A): DETERMINISTIC TRAJECTORY GENERATOR
        # --------------------------------------------------
        cmd_traj = [
            sys.executable, args.traj_script,
            "--dataset", args.dataset,
            "--seed", str(current_seed),
            "--artifacts-dir", str(artifacts_dir),
            "--lr", str(args.lr_traj),
            "--epochs", str(args.epochs_traj),
            "--sequence-length", str(final_seq_len)
        ]
        execute_subprocess(cmd_traj, env, f"Stage 2-A [Deterministic Trajectory - Seed {current_seed}]")

        # --------------------------------------------------
        # STAGE 2 (METHOD B): PARAMETRIC MLP HEAD
        # --------------------------------------------------
        cmd_mlp = [
            sys.executable, args.mlp_script,
            "--dataset", args.dataset,
            "--seed", str(current_seed),
            "--artifacts-dir", str(artifacts_dir),
            "--lr", str(args.lr_mlp),
            "--epochs", str(args.epochs_mlp)
        ]
        execute_subprocess(cmd_mlp, env, f"Stage 2-B [Parametric MLP Head - Seed {current_seed}]")

        # --------------------------------------------------
        # STAGE 2 (METHOD C): STOCHASTIC TRAJECTORY CVAE
        # --------------------------------------------------
        cmd_cvae = [
            sys.executable, args.cvae_script,
            "--dataset", args.dataset,
            "--seed", str(current_seed),
            "--artifacts-dir", str(artifacts_dir),
            "--lr", str(args.lr_cvae),
            "--epochs", str(args.epochs_cvae),
            "--latent-dim", str(args.latent_dim),
            "--sequence-length", str(final_seq_len)
        ]
        execute_subprocess(cmd_cvae, env, f"Stage 2-C [Stochastic Trajectory CVAE - Seed {current_seed}]")

        # --------------------------------------------------
        # STAGE 3: EVALUATION & METRIC CALCULATION
        # --------------------------------------------------
        cmd_eval = [
            sys.executable, args.eval_script,
            "--dataset", args.dataset,
            "--artifacts-dir", str(artifacts_dir),
            "--reports-dir", str(reports_dir),
            "--batch-size", str(args.eval_batch_size),
            "--trajectory-sequence-length", str(final_seq_len)
        ]
        execute_subprocess(cmd_eval, env, f"Evaluation Protocol [Seed {current_seed}]")

        logging.info("Run %d completed. Outputs exported to: %s\n", run_idx + 1, run_dir.resolve())

    logging.info("=========================================================================")
    logging.info("🎉 SUCCESS: All planned benchmark runs completed successfully!")
    logging.info("=========================================================================")


if __name__ == "__main__":
    main()
CIFAR-N Training Dynamics & Trajectory Distillation
================================================================================

OVERVIEW
--------------------------------------------------------------------------------
This repository provides a comprehensive framework for tracking, distilling, and
evaluating neural network training dynamics on datasets with noisy labels. By
analyzing how a model learns over time (training trajectories), the pipeline
extracts uncertainty descriptors and distills them into lightweight generative
and parametric models. These distilled models are then used to evaluate sample
correctness and detect Out-of-Distribution (OOD) data.

The project relies on human-annotated noise benchmarks, specifically utilizing
CIFAR-10N and CIFAR-100N.


PROJECT ARCHITECTURE
--------------------------------------------------------------------------------
The pipeline is divided into distinct stages, moving from initial dynamic
tracking to feature distillation and comprehensive evaluation.

1. Stage 1: Dynamics Tracking
   - Script: stage1.py
   - Functionality: Trains a ResNet-18 backbone on noisy datasets.
   - Artifacts: Tracks and saves per-sample softmax probabilities
     (softmax_history) and confidence margins (margin_history) iteratively over
     the specified training epochs.

2. Stage 2: Distillation Methods
   The project provides three distinct methods to distill the extracted Stage 1
   trajectories into lightweight predictors from frozen deterministic embeddings.

   - Method A: Trajectory CVAE (stage2_cvae.py)
     Trains a Conditional Variational Autoencoder (CVAE) that generates full
     prediction trajectories conditioned on static feature embeddings to capture
     uncertainty.

   - Method B: 4D MLP Descriptors (stage2_mlp.py)
     Extracts 4D uncertainty descriptors based on training history: Area Under
     the Margin (AUM), Mean Confidence, Variability, and Forgetting Count.
     Trains an Advanced MLP to regress these normalized descriptors directly
     from image features.

   - Method C: Trajectory Generator (stage2_trajectory.py)
     Utilizes a temporal generator with residual MLP blocks to regress the entire
     continuous trajectory matrix (margin or softmax).

3. Baseline Predictors
   - Script: train_baseline_predictor.py
   - Functionality: Trains a direct, image-only binary predictor using a Binary
     Cross-Entropy (BCE) loss to classify labels as clean or noisy.

4. Unified Evaluation Protocol
   - Script: eval.py
   - Tasks Evaluated:
     * Task 1: Clean ID (Meta-Test) vs. Noisy ID.
     * Task 2: Clean ID vs. Near-OOD (using CIFAR-10 as OOD for CIFAR-100, and vice versa).
     * Task 3: Clean ID vs. Far-OOD (using the SVHN dataset).
   - Baselines Compared: Maximum Softmax Probability (MSP), LogitNorm
     (Energy Score), Mahalanobis Distance, and the Image-Only Binary Predictor.
   - Metrics: Computes Risk Coverage (AURC), AUROC, and AUPRC. Results are
     exported to a unified CSV summary.


MODELS & COMPONENTS
--------------------------------------------------------------------------------
The framework implements several custom neural network architectures:
- ResNet-18 Backbone: Modified for configurable output classes (10 or 100).
- FeatureExtractor: Strips the final classification layer from the backbone to
  yield dense latent embeddings.
- AdvancedMLP: A customizable Multi-Layer Perceptron used for 4D descriptor
  regression and baseline binary prediction.
- TrajectoryGenerator: Projects features into a temporal sequence using 1D
  convolutions and residual blocks.
- TrajectoryCVAE: An encoder-decoder architecture that uses reparameterization
  to generate sequences from a latent space.
- LogitNorm: A module for normalizing logits by their L2 norm and a
  temperature factor (tau).


USAGE INSTRUCTIONS
--------------------------------------------------------------------------------
Prerequisites & Data Setup:
  The datasets (CIFAR-10N and CIFAR-100N) will automatically download their base
  images via torchvision and retrieve the required human-annotated noise labels
  (CIFAR-10_human.pt or CIFAR-100_human.pt) from specified repository mirrors.

Running the Pipeline:
  Execute the stages sequentially to ensure artifacts are properly generated
  and passed down the pipeline.

  1. Run Stage 1 (Backbone & Tracking)
     python stage1.py --dataset cifar10n --epochs 50 --batch-size 256

  2. Run Stage 2 (Distillation)
     - To train the 4D Descriptor MLP:
       python stage2_mlp.py --dataset cifar10n --epochs 40 --batch-size 512

     - To train the Trajectory CVAE:
       python stage2_cvae.py --dataset cifar10n --epochs 50 --batch-size 256

     - To train the Trajectory Generator:
       python stage2_trajectory.py --dataset cifar10n --epochs 40 --target-type margin

  3. Train the Baseline Predictor (Optional)
     python train_baseline_predictor.py --dataset cifar10n --epochs 30

  4. Run Unified Evaluation
     python eval.py --dataset cifar10n --batch-size 256 --score-batch-size 256

  This will compile the predictive scores across all methods, compare them
  against OOD datasets, and generate a final benchmarking report in your
  artifacts directory.
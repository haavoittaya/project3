\# AI Agent Context: Uncertainty Descriptors Distillation



\## Project Overview

\* \[cite\_start]\*\*Title:\*\* Distilling Deep Learning Training Dynamics into Inference-Time Uncertainty Descriptors\[cite: 1, 8, 11].

\* \[cite\_start]\*\*Primary Authors:\*\* Stanislav Kozlov, Matvey Druzhinin\[cite: 3, 13].

\* \[cite\_start]\*\*Institution:\*\* Faculty of Mathematical Foundations of AI, Innopolis University\[cite: 4].

\* \[cite\_start]\*\*Objective:\*\* Bridge the gap between offline dataset pruning and real-time inference by distilling training dynamics into a lightweight inference model\[cite: 23, 28].



\## Core Methodology

\* \[cite\_start]\*\*Stage 1 (Trajectory Tracking):\*\* Train a ResNet-18 backbone on the CIFAR-10N aggregate noise split for 50 epochs\[cite: 18, 24]. \[cite\_start]Log epoch-wise softmax probabilities and logit margins\[cite: 25].

\* \[cite\_start]\*\*Descriptor Vector ($s\_i$):\*\* Calculate the ground-truth dynamic vector for each sample: AUM (Area Under the Margin), mean confidence, variability, and forgetting count\[cite: 26, 27].

\* \*\*Stage 2 (Descriptor Distillation):\*\* Freeze the trained ResNet-18 backbone and use a lightweight MLP head ($g\_\\phi$) trained with Huber Loss to predict the descriptors ($\\hat{S}\_i$) from static pre-classification embeddings\[cite: 28].



\## Datasets \& Benchmarks

\* \[cite\_start]\*\*CIFAR-10N:\*\* Primary dataset used for training (aggregate noise split) and generalization testing (worse noise split)\[cite: 18].

\* \[cite\_start]\*\*SVHN:\*\* Used exclusively for Out-of-Distribution (OOD) detection benchmarking\[cite: 19].

\* \*\*CIFAR-10-C:\*\* Used to test robustness and severity degradation tracking against 15 algorithmic corruptions\[cite: 20].



\## Repository Architecture Constraints

\* `src/models.py`: Must contain the PyTorch network architectures (ResNet-18 backbone, FeatureExtractor, AdvancedMLP).

\* `src/dataset.py`: Handles data loading and UCSC-REAL/cifar-10-100n repository cloning logic.

\* Scripts (`train\_stage1.py`, `train\_stage2.py`, `evaluate\_ood.py`) must import from the `src` module.



\## AI Assistant Rules

1\. \*\*Device Agnostic:\*\* Always ensure tensors and models are sent to `device = torch.device("cuda" if torch.cuda.is\_available() else "cpu")`.

2\. \[cite\_start]\*\*Metrics Focus:\*\* When writing evaluation code, prioritize AUROC and AUPRC for noisy label detection\[cite: 32]. \[cite\_start]Use Risk-Coverage curves and AURC for selective classification tasks\[cite: 33].

3\. \[cite\_start]\*\*Robustness:\*\* Ensure OOD testing explicitly compares in-distribution (CIFAR-10) metrics against out-of-distribution (SVHN) metrics\[cite: 34].


# Structural Damage Image Classification (Custom ResNet-18)

Binary image classifier for detecting structural damage in post-earthquake building imagery using a **custom ResNet-18 architecture trained from scratch** on the PEER Center dataset.

## Problem Statement

Classify 224×224 RGB building images into two structural classes:

| Label | Class       | Typical Proportion |
|-------|-------------|--------------------|
| 0     | `undamaged` | ~54%               |
| 1     | `damaged`   | ~46%               |

Reference baseline from the PEER paper: **InceptionV3 transfer learning → 83% validation accuracy**. 

This project replaces transfer learning with a **custom, manually-implemented ResNet-18 architecture** trained from scratch, accompanied by a **70/15/15 stratified data split** (train/val/test), class-balanced sampling (`WeightedRandomSampler`), controlled data augmentations, and preparation for systematic post-hoc interpretability analysis via **Grad-CAM**.

## Architecture & Design

- **Backbone**: Custom ResNet-18 (`BasicBlock` residual units, $64 \to 128 \to 256 \to 512$ channels across 4 stages with identity/shortcut downsampling projections).
- **Global Pooling**: Global average pooling (`AdaptiveAvgPool2d((1, 1))`).
- **Classifier Head**: Two-layer MLP head with `BatchNorm1d`, `ReLU`, and `Dropout` ($512 \to 512 \to 256 \to 2$).
- **Interpretability Readiness**: Exposes `get_target_layer()` pointing to the final residual block (`layer4[-1]`) for Grad-CAM activation and gradient extraction.

## Dataset Partitioning (70 / 15 / 15)

The dataset is partitioned into three disjoint splits per class:
- **Training (70%)**: Used for model parameter optimization with controlled augmentations (Horizontal Flip, Small Rotation, Translation/Scale, Mild ColorJitter).
- **Validation (15%)**: Used for hyperparameter tuning, learning rate scheduling, and early stopping.
- **Held-Out Test (15%)**: Kept completely untouched during training and model selection; evaluated only once using `evaluate.py`.

## Repository Structure

```
.
├── config.py              # Hyperparameters, paths, and augmentation knobs
├── dataset.py             # Transforms, Dataset class, WeightedRandomSampler
├── model.py               # Custom ResNet-18 architecture & classifier head
├── train.py               # Training loop, optimizer, scheduler, early stopping, checkpointing
├── evaluate.py            # Held-out evaluation (Confusion matrix, ROC/PR curves, classification report)
├── predict.py             # Single-image / directory inference with plot outputs
├── prepare_data.py        # Stratified dataset split into train/val/test (70/15/15)
├── requirements.txt
├── data/
│   ├── train/   (undamaged/, damaged/)
│   ├── val/     (undamaged/, damaged/)
│   └── test/    (undamaged/, damaged/)   # Held-out test set
├── checkpoints/
│   ├── best_model.pt
│   └── last_model.pt
├── logs/                  # TensorBoard event files
└── outputs/               # Evaluation plots + metrics
```

## Setup

```bash
# 1. Create a virtual env (recommended)
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify PyTorch setup
python3 -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

## Prepare Data

Split a raw PEER-style source directory containing `undamaged/` and `damaged/` subdirectories into 70% train, 15% validation, and 15% test:

```bash
python prepare_data.py --source ./PEER_images --val-ratio 0.15 --test-ratio 0.15 --copy
```

Directory structure produced:

```
data/train/undamaged/*.jpg
data/train/damaged/*.jpg
data/val/undamaged/*.jpg
data/val/damaged/*.jpg
data/test/undamaged/*.jpg
data/test/damaged/*.jpg
```

## Train

Train the custom ResNet-18 from scratch:

```bash
python train.py
```

Highlights:
- **Sampling**: `WeightedRandomSampler` balances training batch class distribution.
- **Augmentation**: Controlled geometric and color transforms (RandomErasing disabled by default to preserve structural damage cues).
- **Optimization**: Adam optimizer with `StepLR` scheduler and early stopping based on validation accuracy.

Outputs produced:
- `checkpoints/best_model.pt` — Best model checkpoint based on validation accuracy.
- `checkpoints/last_model.pt` — Final epoch checkpoint.
- `logs/run_<timestamp>/` — TensorBoard event logs (`tensorboard --logdir logs`).

## Evaluate on Held-Out Test Set

After training completes, evaluate the best checkpoint on the untouched held-out test split:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

Outputs produced in `outputs/`:
- `test_confusion_matrix.png` — Confusion matrix heatmap (TP, TN, FP, FN).
- `test_roc_curve.png` — ROC curve with AUC score.
- `test_pr_curve.png` — Precision-Recall curve with AP score.
- `test_metrics.txt` — Full classification report & metrics per class.

## Predict

Run inference on single images or image directories:

```bash
python predict.py \
  --checkpoint checkpoints/best_model.pt \
  --images data/test/damaged/IMG_1234.jpg data/test/undamaged/ \
  --save-plot outputs/sample_predictions.png
```

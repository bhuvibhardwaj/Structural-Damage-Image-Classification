# Structural Damage Image Classification (Custom ResNet-18)

Binary image classifier for detecting structural damage in post-earthquake building imagery using a **custom ResNet-18 architecture trained from scratch** on the PEER Hub ImageNet dataset.

## Problem Statement

The objective is to classify 224×224 building images into two structural classes:

| Label | Class | Approx. Proportion |
|---|---|---:|
| 0 | `undamaged` | 53.2% |
| 1 | `damaged` | 46.8% |

The reference study reported approximately **83% validation accuracy using ImageNet-pretrained InceptionV3**.

This project instead uses a **custom, manually implemented ResNet-18 trained from scratch**, with controlled data augmentation and a separate held-out test set.

## Dataset

The project uses the **PEER Hub ImageNet Task 2 (damage state)** dataset.

The supplied NumPy dataset contains:

- **11,811 training images**
- **1,460 held-out test images**
- Image size: **224×224×3**
- Two classes: `undamaged` and `damaged`
- One-hot encoded labels

The provided training set is split into:

- **85% training:** 10,040 images
- **15% validation:** 1,771 images
- **Held-out test:** 1,460 images

The test set is kept separate from training and validation and is used for final evaluation after model selection.

The images are stored using **BGR channel ordering with mean subtraction**. During loading, this preprocessing is reversed and images are restored to RGB before the PyTorch transformation pipeline.

## Architecture & Design

- **Backbone:** Custom ResNet-18 with `BasicBlock` residual units and four residual stages.
- **Channels:** `64 → 128 → 256 → 512`.
- **Downsampling:** Identity/shortcut projections are used when dimensions change.
- **Global Pooling:** `AdaptiveAvgPool2d((1, 1))`.
- **Classifier Head:** Two-layer MLP with `BatchNorm1d`, `ReLU`, and `Dropout`.
- **Output:** Two-class classification (`undamaged`, `damaged`).
- **Training:** The network is trained from scratch rather than using pretrained ImageNet weights.

## Data Augmentation

Training images use conservative augmentations designed to preserve structural damage patterns:

- Random horizontal flip
- Random rotation up to ±10°
- Random translation up to 5%
- Random scaling between 0.95× and 1.05×
- Mild brightness and contrast jitter

Validation and test images receive no augmentation.

The augmentation strategy is deliberately conservative because aggressive geometric transformations can distort cracks and other structural features.

## Repository Structure

```text
.
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── prepare_data.py
├── requirements.txt
├── structural_damage_training.ipynb
├── results/
│   ├── results.md
│   └── evaluation_results.json
└── data/                         # Local dataset files, not committed
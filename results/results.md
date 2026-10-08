# Experiment Results

## Dataset

The experiment used the PEER structural-damage image dataset provided as NumPy arrays.

- Training images: 11,811
- Test images: 1,460
- Image size: 224 × 224 × 3
- Class 0: Undamaged
- Class 1: Damaged
- Validation split: 15% of the provided training set
- Stratified train/validation split
- Held-out test set used only for final evaluation

## Model

- Architecture: Custom ResNet-18
- Training: From scratch
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Epochs: 20
- Batch size: 16
- Loss: CrossEntropyLoss
- Best checkpoint selected using validation accuracy

## Results

| Metric | Result |
|---|---:|
| Best validation accuracy | 81.71% |
| Test accuracy | 81.37% |
| Test loss | 0.4431 |
| Damaged precision | 79.42% |
| Damaged recall | 83.64% |
| Damaged F1 | 81.47% |
| Undamaged precision | 83.45% |
| Undamaged recall | 79.19% |
| Undamaged F1 | 81.27% |

## Confusion Matrix

Rows represent true labels and columns represent predicted labels.

```text
                 Predicted
              Undamaged  Damaged

Actual
Undamaged        590       155
Damaged          117       598
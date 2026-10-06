import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from config import Config
from dataset import build_dataloaders, set_seed, CLASS_NAMES
from model import load_checkpoint
from train import resolve_device, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ResNet damage classifier")
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join("checkpoints", "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for plots (defaults to config.OUTPUT_DIR)",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds),
    )


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], save_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Damaged Class")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str
) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}", color="navy", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Damaged Class")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = Config()
    if args.device != "auto":
        config.DEVICE = args.device
    set_seed(config.SEED)

    device = resolve_device(config)
    output_dir = args.output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    train_loader, val_loader, test_loader = build_dataloaders(config)
    loader = val_loader if args.split == "val" else test_loader
    if loader is None:
        raise RuntimeError(f"No {args.split} split available. "
                           f"Place images in data/{args.split}/<class_name>/")

    print(f"[INFO] Evaluating on {args.split} split ({len(loader.dataset)} samples)...")
    print(f"       Checkpoint: {args.checkpoint}")
    print(f"       Device: {device}")

    model = load_checkpoint(args.checkpoint, config, device)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device, split=args.split)

    print("\n[RESULTS]")
    for k, v in metrics.items():
        if k.startswith("acc_") or k in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            print(f"  {k:>18s}: {100*v:6.2f}%" if not k == "loss" else f"  {k:>18s}: {v:.4f}")
        else:
            print(f"  {k:>18s}: {v:.4f}")

    y_true, y_prob, y_pred = collect_predictions(model, loader, device)

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("[CONFUSION MATRIX]")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    cm_path = os.path.join(output_dir, f"{args.split}_confusion_matrix.png")
    roc_path = os.path.join(output_dir, f"{args.split}_roc_curve.png")
    pr_path = os.path.join(output_dir, f"{args.split}_pr_curve.png")

    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    if len(CLASS_NAMES) == 2:
        plot_roc_curve(y_true, y_prob, roc_path)
        plot_pr_curve(y_true, y_prob, pr_path)

    report_path = os.path.join(output_dir, f"{args.split}_metrics.txt")
    with open(report_path, "w") as f:
        f.write(f"Split: {args.split}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"TN={cm[0,0]} FP={cm[0,1]}\n")
        f.write(f"FN={cm[1,0]} TP={cm[1,1]}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    print(f"\n[INFO] Plots saved to: {output_dir}")
    print(f"       Confusion matrix: {cm_path}")
    if len(CLASS_NAMES) == 2:
        print(f"       ROC curve:        {roc_path}")
        print(f"       PR curve:         {pr_path}")
    print(f"       Metrics:          {report_path}")


if __name__ == "__main__":
    main()

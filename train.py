import os
import time
import shutil
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    _LRScheduler,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import build_dataloaders, set_seed, CLASS_NAMES
from model import build_model, ResNetDamageClassifier


def resolve_device(config: Config) -> torch.device:
    if config.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.DEVICE)


def build_optimizer(model: ResNetDamageClassifier, config: Config) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = config.OPTIMIZER.lower()
    if name == "adam":
        return Adam(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif name == "sgd":
        return SGD(
            params,
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=True,
        )
    elif name == "adamw":
        from torch.optim import AdamW
        return AdamW(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: Optimizer, config: Config
) -> Tuple[Optional[_LRScheduler], bool]:
    name = config.SCHEDULER.lower()
    if name == "none":
        return None, False
    elif name == "steplr":
        return StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA), False
    elif name == "plateau":
        return (
            ReduceLROnPlateau(
                optimizer, mode="max", factor=config.GAMMA, patience=config.PATIENCE
            ),
            True,
        )
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS), False
    else:
        raise ValueError(f"Unknown scheduler: {name}")


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: ResNetDamageClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.1f}%")

    return {
        "loss": total_loss / total if total > 0 else float("nan"),
        "accuracy": correct / total if total > 0 else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: ResNetDamageClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc=f"          [{split}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    acc = correct / total if total > 0 else 0.0
    loss_val = total_loss / total if total > 0 else float("nan")

    from sklearn.metrics import f1_score, precision_score, recall_score
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    metrics = {
        "loss": loss_val,
        "accuracy": acc,
        "f1_macro": float(f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0)),
    }

    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels_np == i
        if mask.sum() > 0:
            metrics[f"acc_{name}"] = float((all_preds_np[mask] == i).mean())

    return metrics


def save_checkpoint(
    path: str,
    model: ResNetDamageClassifier,
    optimizer: Optimizer,
    epoch: int,
    best_acc: float,
    config: Config,
    scheduler: Optional[_LRScheduler] = None,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_accuracy": best_acc,
        "config": {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, tuple, list, dict))
        },
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def train(config: Optional[Config] = None) -> ResNetDamageClassifier:
    config = config or Config()
    set_seed(config.SEED)

    device = resolve_device(config)
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _ = build_dataloaders(config)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"[INFO] Dataset — train: {n_train}, val: {n_val}")

    counts = train_loader.dataset.get_class_counts()
    for i, name in enumerate(CLASS_NAMES):
        print(f"       {name}: train={counts[i]}")
    val_counts = val_loader.dataset.get_class_counts()
    for i, name in enumerate(CLASS_NAMES):
        print(f"       {name}: val={val_counts[i]}")

    model = build_model(config, device)
    summary = model.trainable_params_summary()
    print(f"[INFO] Model: Custom ResNet-18")
    print(f"       Total params: {summary['total_parameters']:,}")
    print(f"       Trainable:    {summary['trainable_parameters']:,} "
          f"({summary['trainable_ratio_pct']:.2f}%)")
    print(f"       Frozen:       {summary['frozen_parameters']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_needs_metric = build_scheduler(optimizer, config)

    early = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode="max")

    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, f"run_{int(time.time())}"))

    best_acc = 0.0
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    last_path = os.path.join(config.CHECKPOINT_DIR, "last_model.pt")

    print("\n[INFO] Starting training...\n")
    header = f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | {'Val Loss':>8} {'Val Acc':>7} {'F1':>6} | {'LR':>9} | {'Time':>5}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_clip=config.GRADIENT_CLIPPING,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, split="val")

        if scheduler is not None:
            if scheduler_needs_metric:
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()

        lr_current = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1_macro"], epoch)
        writer.add_scalar("LR", lr_current, epoch)

        improved = val_metrics["accuracy"] > best_acc
        if improved:
            best_acc = val_metrics["accuracy"]
            save_checkpoint(best_path, model, optimizer, epoch, best_acc, config, scheduler)

        save_checkpoint(last_path, model, optimizer, epoch, best_acc, config, scheduler)

        print(
            f"{epoch:>5} | "
            f"{train_metrics['loss']:>10.4f} {100*train_metrics['accuracy']:>8.2f}% | "
            f"{val_metrics['loss']:>8.4f} {100*val_metrics['accuracy']:>6.2f}% "
            f"{100*val_metrics['f1_macro']:>5.2f}% | "
            f"{lr_current:>9.2e} | "
            f"{elapsed:>4.1f}s"
            f"{'  [BEST]' if improved else ''}"
        )

        if early(val_metrics["accuracy"]):
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}. "
                  f"Best val accuracy: {100*best_acc:.2f}%")
            break

    writer.close()
    print(f"\n[INFO] Training complete. Best model @ {best_path} "
          f"(val_acc={100*best_acc:.2f}%)")
    return model


if __name__ == "__main__":
    train()

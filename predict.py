import os
import argparse
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from dataset import get_val_transforms, CLASS_NAMES, denormalize
from model import load_checkpoint
from train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict damage from images")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join("checkpoints", "best_model.pt"))
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Image file paths or a directory")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save predictions plot to this path")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def expand_image_paths(paths: List[str]) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    results = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                if fname.lower().endswith(exts):
                    results.append(os.path.join(p, fname))
        elif os.path.isfile(p):
            results.append(p)
    return results


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image_path: str,
    tfms,
    device: torch.device,
    config: Config,
) -> Tuple[Image.Image, np.ndarray, int]:
    img_pil = Image.open(image_path).convert("RGB")
    tensor = tfms(img_pil).unsqueeze(0).to(device, non_blocking=True)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    return img_pil, probs, pred


def plot_predictions(
    samples: List[Tuple[str, Image.Image, np.ndarray, int]],
    save_path: str,
    class_names: List[str] = CLASS_NAMES,
) -> None:
    n = len(samples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (path, pil_img, probs, pred) in zip(axes, samples):
        ax.imshow(pil_img)
        ax.axis("off")
        label = os.path.basename(path)
        if len(label) > 24:
            label = label[:21] + "..."
        prob_str = "\n".join(f"{class_names[i]}: {100*probs[i]:.1f}%" for i in range(len(class_names)))
        ax.set_title(f"{label}\nPred: {class_names[pred]}\n{prob_str}", fontsize=8)

    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = Config()
    if args.device != "auto":
        config.DEVICE = args.device

    device = resolve_device(config)
    tfms = get_val_transforms(config)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = load_checkpoint(args.checkpoint, config, device)
    image_paths = expand_image_paths(args.images)
    if not image_paths:
        raise RuntimeError("No images found from input paths.")

    print(f"[INFO] Predicting {len(image_paths)} image(s)...\n")
    header = f"{'Image':<50s} | {'Prediction':<12s} | " + " | ".join(
        f"P({c})" for c in CLASS_NAMES
    )
    print(header)
    print("-" * len(header))

    samples = []
    for path in image_paths:
        pil_img, probs, pred = predict_single(model, path, tfms, device, config)
        probs_str = " | ".join(f"{100*probs[i]:>6.2f}%" for i in range(len(CLASS_NAMES)))
        name = os.path.basename(path)
        if len(name) > 48:
            name = name[:45] + "..."
        print(f"{name:<50s} | {CLASS_NAMES[pred]:<12s} | {probs_str}")
        samples.append((path, pil_img, probs, pred))

    if args.save_plot:
        plot_predictions(samples, args.save_plot)
        print(f"\n[INFO] Plot saved: {args.save_plot}")


if __name__ == "__main__":
    main()

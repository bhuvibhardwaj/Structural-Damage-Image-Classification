import os
import argparse
import random
import shutil
from typing import List, Tuple

import numpy as np

from config import Config
from dataset import CLASS_NAMES, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PEER-style dataset into train/val/test splits"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Source directory containing class subdirs (e.g. PEER_images/)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of images for validation (default: 0.15 = 15%)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Fraction for held-out test set (default: 0.15 = 15%)",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of moving them",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(root: str, class_name: str) -> List[str]:
    d = os.path.join(root, class_name)
    if not os.path.isdir(d):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([
        f for f in os.listdir(d)
        if f.lower().endswith(exts)
    ])


def split_indices(
    n: int, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def transfer_files(
    src_dir: str,
    dst_dir: str,
    class_name: str,
    filenames: List[str],
    copy: bool,
) -> None:
    dst_class = os.path.join(dst_dir, class_name)
    os.makedirs(dst_class, exist_ok=True)
    for fn in filenames:
        s = os.path.join(src_dir, class_name, fn)
        d = os.path.join(dst_class, fn)
        if os.path.exists(d):
            continue
        if copy:
            shutil.copy2(s, d)
        else:
            shutil.move(s, d)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = Config()

    if not os.path.isdir(args.source):
        raise NotADirectoryError(f"Source dir not found: {args.source}")

    print(f"[INFO] Source:       {args.source}")
    print(f"[INFO] Train dir:    {config.TRAIN_DIR}")
    print(f"[INFO] Val dir:      {config.VAL_DIR}")
    if args.test_ratio > 0:
        print(f"[INFO] Test dir:     {config.TEST_DIR}")
    print(f"[INFO] Val ratio:    {args.val_ratio:.2%}")
    print(f"[INFO] Test ratio:   {args.test_ratio:.2%}")
    print(f"[INFO] Mode:         {'copy' if args.copy else 'move'}")
    print()

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASS_NAMES:
        files = list_images(args.source, class_name)
        n = len(files)
        if n == 0:
            print(f"[SKIP] {class_name}: 0 images")
            continue

        tr_idx, va_idx, te_idx = split_indices(
            n, args.val_ratio, args.test_ratio, args.seed
        )

        tr_files = [files[i] for i in tr_idx]
        va_files = [files[i] for i in va_idx]
        te_files = [files[i] for i in te_idx]

        transfer_files(args.source, config.TRAIN_DIR, class_name, tr_files, args.copy)
        transfer_files(args.source, config.VAL_DIR, class_name, va_files, args.copy)
        if te_files:
            transfer_files(args.source, config.TEST_DIR, class_name, te_files, args.copy)

        total_train += len(tr_files)
        total_val += len(va_files)
        total_test += len(te_files)

        print(f"{class_name:<12s}: total={n:4d}  train={len(tr_files):4d}  "
              f"val={len(va_files):4d}  test={len(te_files):4d}")

    print(f"\n[DONE] train={total_train}  val={total_val}  test={total_test}")


if __name__ == "__main__":
    main()

import os
import random
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import Config


CLASS_NAMES = ["undamaged", "damaged"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms(config: Config) -> transforms.Compose:
    aug = config.AUGMENTATION
    tf_list = [
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.RandomAffine(
            degrees=0,
            translate=aug["affine_translate"],
            scale=aug["affine_scale"],
        ),
        transforms.ColorJitter(
            brightness=aug["brightness"],
            contrast=aug["contrast"],
            saturation=aug["saturation"],
            hue=aug["hue"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ]
    p_erase = aug.get("random_erasing_p", 0.0)
    if p_erase > 0.0:
        tf_list.append(transforms.RandomErasing(p=p_erase, scale=(0.02, 0.15)))
    return transforms.Compose(tf_list)


def get_val_transforms(config: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


class StructuralDamageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_names: List[str] = CLASS_NAMES,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    self.samples.append((fpath, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_class_counts(self) -> np.ndarray:
        counts = np.zeros(len(self.class_names), dtype=np.int64)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def get_sample_weights(self) -> torch.Tensor:
        counts = self.get_class_counts()
        weights = 1.0 / counts
        sample_weights = [weights[label] for _, label in self.samples]
        return torch.DoubleTensor(sample_weights)


def build_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    set_seed(config.SEED)

    train_tf = get_train_transforms(config)
    val_tf = get_val_transforms(config)

    train_ds = StructuralDamageDataset(config.TRAIN_DIR, transform=train_tf)
    val_ds = StructuralDamageDataset(config.VAL_DIR, transform=val_tf)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}. "
                           f"Expected subdirectories: {CLASS_NAMES}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation images found in {config.VAL_DIR}. "
                           f"Expected subdirectories: {CLASS_NAMES}")

    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = None
    if os.path.isdir(config.TEST_DIR) and any(
        os.path.isdir(os.path.join(config.TEST_DIR, c)) for c in CLASS_NAMES
    ):
        test_ds = StructuralDamageDataset(config.TEST_DIR, transform=val_tf)
        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=False,
            )

    return train_loader, val_loader, test_loader


def denormalize(tensor: torch.Tensor, config: Config) -> torch.Tensor:
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    return tensor * std + mean

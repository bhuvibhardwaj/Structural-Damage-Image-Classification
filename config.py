import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    SEED: int = 42

    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
    TRAIN_DIR: str = os.path.join(DATA_DIR, "train")
    VAL_DIR: str = os.path.join(DATA_DIR, "val")
    TEST_DIR: str = os.path.join(DATA_DIR, "test")

    CHECKPOINT_DIR: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs")
    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "outputs")

    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    DROPOUT_RATE: float = 0.3
    HIDDEN_DIM: int = 512
    NUM_CLASSES: int = 2

    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    OPTIMIZER: str = "adam"
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9

    SCHEDULER: str = "steplr"
    STEP_SIZE: int = 10
    GAMMA: float = 0.5
    PATIENCE: int = 5

    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 12
    GRADIENT_CLIPPING: float = 1.0

    AUGMENTATION: dict = None

    DEVICE: str = "auto"

    def __post_init__(self):
        if self.AUGMENTATION is None:
            self.AUGMENTATION = {
                "horizontal_flip_p": 0.5,
                "vertical_flip_p": 0.0,
                "rotation_degrees": 15,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.1,
                "hue": 0.05,
                "random_erasing_p": 0.0,
                "affine_scale": (0.9, 1.1),
                "affine_translate": (0.1, 0.1),
            }

        for d in [self.CHECKPOINT_DIR, self.LOG_DIR, self.OUTPUT_DIR,
                  self.TRAIN_DIR, self.VAL_DIR, self.TEST_DIR]:
            os.makedirs(d, exist_ok=True)

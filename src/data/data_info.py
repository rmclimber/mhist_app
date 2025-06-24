from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class MHISTDataConfig:
    train_path: str | Path
    val_path: str | Path
    test_path: str | Path
    num_workers: int = 4
    batch_size: int = 32

    def __post_init__(self):
        self.train_path = Path(self.train_path)
        self.val_path = Path(self.val_path)
        self.test_path = Path(self.test_path)
        self.num_workers = min(os.cpu_count(), self.num_workers)


@dataclass
class MHISTDataInfo:
    label_map: dict
    num_classes: int = field(init=False)
    mean: list = field(init=False)
    std: list = field(init=False)
    stats: dict

    def __post_init__(self):
        label_map = {int(k): v for k, v in self.label_map.items()}
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.mean = self.stats["mean"]
        self.std = self.stats["std"]
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class MHISTDataConfig:
    data_path: str | Path
    info_path: str | Path
    num_workers: int = 4
    batch_size: int = 32

    def __post_init__(self):
        # ensure consistent formatting by making passed paths Path objects
        self.data_path = Path(self.data_path)
        self.info_path = Path(self.info_path)

        # generate individual paths
        for mode in ["train", "val", "test"]:
            label_attr_name = f"{mode}_label_path"
            images_attr_name = f"{mode}_image_path"
            label_suffix = f"mhist_{mode}_labels.npy"
            image_suffix = f"mhist_{mode}_images.npy"
            setattr(self, label_attr_name, self.data_path / label_suffix)
            setattr(self, images_attr_name, self.data_path / image_suffix)
        self.num_workers = min(os.cpu_count(), self.num_workers)


@dataclass
class MHISTDataInfo:
    info_dict: dict
    label_map: dict = field(init=False)
    num_classes: int = field(init=False)
    mean: list = field(init=False)
    std: list = field(init=False)

    def __post_init__(self):
        label_map = self.info_dict["label_map"]
        label_map = {int(k): v for k, v in label_map.items()}
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.mean = self.info_dict["stats"]["mean"]
        self.std = self.info_dict["stats"]["std"]
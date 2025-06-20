from dataclasses import dataclass, field

class MHISTDataInfo:
    label_map: dict = field(init=False)
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
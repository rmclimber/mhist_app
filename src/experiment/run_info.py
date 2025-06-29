from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunInfo:
    checkpoint_dir: str | Path = ""
    max_epochs: int = 50
    lr: float = 1e-3

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
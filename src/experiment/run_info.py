from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunInfo:
    checkpoint_dir: str | Path = ""
    max_epochs: int = 50
    lr: float = 1e-3
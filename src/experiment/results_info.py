from dataclasses import dataclass
import numpy as np

@dataclass
class ResultsInfo:
    y: np.ndarray
    logits: np.ndarray
    y_hat: np.ndarray = None

    def __post_init__(self):
        if self.y_hat is None:
            self.y_hat = np.argmax(self.logits, axis=1)
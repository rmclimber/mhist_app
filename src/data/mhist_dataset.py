import torch
from torch.utils.data import Dataset
import numpy as np

class MHISTDataset(Dataset):
    def __init__(self, 
                 img_filename: str, 
                 label_filename: str,
                 device: str = 'cpu'):
        super().__init__()
        self.img_filename = img_filename
        self.label_filename = label_filename
        self.device = device
        self._setup()

    def _setup(self):
        self.images = torch.from_numpy(np.load(self.img_filename))
        self.labels = torch.from_numpy(np.load(self.label_filename))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx].to(self.device), self.labels[idx].to(self.device)


        
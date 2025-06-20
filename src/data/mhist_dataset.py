import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np

from .data_info import *

class MHISTDataset(Dataset):
    def __init__(self, 
                 img_filename: str, 
                 label_filename: str,
                 data_info: MHISTDataInfo,
                 device: str = 'cpu'):
        super().__init__()
        self.img_filename = img_filename
        self.label_filename = label_filename
        self.data_info = data_info
        self.device = device
        self.normalize = Normalize(mean=self.data_info.mean, 
                                   std=self.data_info.std)
        self._setup()

    def _setup(self):
        self.images = torch.from_numpy(np.load(self.img_filename))
        self.labels = torch.from_numpy(np.load(self.label_filename))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.normalize(self.images[idx]), self.labels[idx]
        return x.to(self.device), y.to(self.device)


        
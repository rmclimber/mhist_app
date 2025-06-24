import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Normalize, 
    RandomRotation, 
    RandomHorizontalFlip, 
    RandomVerticalFlip, 
    RandomResizedCrop,
    Compose,
    ToTensor,
    ColorJitter)
from PIL import Image
import numpy as np

from .data_info import *

class MHISTDataset(Dataset):
    def __init__(self, 
                 img_filename: str, 
                 label_filename: str,
                 data_info: MHISTDataInfo,
                 mode: str = "train"):
        super().__init__()
        self.img_filename = img_filename
        self.label_filename = label_filename
        self.data_info = data_info
        self.mode = mode
        self._setup()
        self.apply_transforms = self._assemble_transforms()

    def _setup(self):
        self.images = np.load(self.img_filename)
        self.labels = torch.from_numpy(np.load(self.label_filename))

    def _assemble_transforms(self):
        # NOTE: this must run after _setup()
        if self.mode == "train":
            return Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomRotation(10),
                RandomResizedCrop(self.images[0].shape[-2], scale=(.9, 1.0)),
                ColorJitter(brightness=.1, contrast=.1),
                ToTensor(),
                Normalize(mean=self.data_info.mean / 255., 
                          std=self.data_info.std / 255.)
            ])
        return Compose([ToTensor(), Normalize(mean=self.data_info.mean / 255., 
                                              std=self.data_info.std) / 255.])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # necessary transpositions and casts for torchvision transforms
        x_chw = self.images[idx]
        x_hwc = np.moveaxis(x_chw, (1, 2, 0))
        x = Image.fromarray(x_hwc.astype(np.uint8))

        # apply the transforms and get the transformed x back
        x = self.apply_transforms(x)
        
        return x, self.labels[idx]


        
from yaml import safe_load
from lightning import LightningDataModule
from torch.utils.data import DataLoader

# my code
from .mhist_dataset import *
from .data_info import *

class MHISTDataModule(LightningDataModule):
    def __init__(self, 
                 datasets: dict[str, MHISTDataset],
                 data_config: MHISTDataConfig,
                 data_info: MHISTDataInfo):
        super().__init__()
        self.datasets = datasets
        self.data_config = data_config
        self.data_info = data_info
        self.setup()

    def setup(self, stage=None):
        # save references to the initialized datasets
        self.train_dataset = self.datasets['train']
        self.val_dataset = self.datasets['val']
        self.test_dataset = self.datasets['test']
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.data_config.batch_size,  
                          num_workers=self.data_config.num_workers, 
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.data_config.batch_size,  
                          num_workers=self.data_config.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.data_config.batch_size,  
                          num_workers=self.data_config.num_workers)
    
    def predict_dataloader(self, mode: str):
        if mode == "train":
            return self.train_dataloader()
        elif mode == "val":
            return self.val_dataloader()
        elif mode == "test":
            return self.test_dataloader()
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'train', 'val', or 'test'.")
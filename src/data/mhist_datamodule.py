from yaml import safe_load
from lightning import LightningDataModule
from torch.utils.data import DataLoader

# my code
from .mhist_dataset import *
from .data_info import *

class MHISTDataModule(LightningDataModule):
    def __init__(self, 
                 datasets: dict[str, MHISTDataset],
                 data_config: MHISTDataConfig):
        super().__init__()
        self.datasets = datasets
        self.data_config = data_config
        self.data_info = None
        self.setup()

    def setup(self, stage=None):
        # collect basic information about the datasets themselves
        self.data_info = MHISTDataInfo(
            info_dict=safe_load(open(self.data_config.info_path, 'r')))
        
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
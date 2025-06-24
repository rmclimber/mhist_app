from lightning.pytorch.loggers import WandbLogger
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# my imports
from ..model.lightning_vit import *
from ..data.mhist_datamodule import *



class MHISTTraining:
    def __init__(self):
        self.datamodule = self._build_datamodule()
        self.logger = self._build_logger()
        self.model = self._build_model()
        self.trainer = self._build_trainer()

    def _build_datamodule(self):
        self.datasets = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset
        }
        datamodule = MHISTDataModule(datasets=self.datasets,
                                     data_config=self.data_config)
        return datamodule

    def _assemble_callbacks(self):
        model_checkpoint = ModelCheckpoint()
        early_stopping = EarlyStopping()
        return [model_checkpoint, early_stopping]

    def _build_logger(self):
        logger = WandbLogger()
        return logger

    def _build_model(self):
        model = LightningViT()
        return model


    def _build_trainer(self):
        pass

    def run(self):
        print("Not implemented yet")

if __name__ == '__main__':
    mhist_training = MHISTTraining()
    mhist_training.run()
from lightning.pytorch.loggers import WandbLogger
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# my imports
from ..model.lightning_vit import *
from ..data.mhist_datamodule import *
from ..data.data_info import *
from .run_info import *




class MHISTTraining:
    def __init__(self, run_info: dict, data_config: dict):
        self.run_info = RunInfo(**run_info)
        self.data_config = MHISTDataConfig(**data_config)
        self.datamodule = self._build_datamodule(run_info, self.data_config)
        self.logger = self._build_logger()
        self.model = self._build_model()
        self.trainer = self._build_trainer()

    def _build_datamodule(self, 
                          data_config: MHISTDataConfig = None) -> LightningDataModule:
        if data_config is None:
            data_config = self.data_config

        self.datasets = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset
        }
        datamodule = MHISTDataModule(datasets=self.datasets,
                                     data_config=data_config)
        return datamodule

    def _assemble_callbacks(self) -> list:
        """
        These callbacks will be passed to the Trainer.
        """
        # specifying model checkpointing behavior
        model_checkpoint = ModelCheckpoint(
            dirpath=self.run_info.checkpoint_dir,
            filename="{epoch}-{step}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True)
        
        # ensure that loss is non-inf and non-NaN, and keeps improving
        early_stopping = early_stopping = EarlyStopping(
            monitor="train_loss", patience=100, check_finite=True
        )
        return [model_checkpoint, early_stopping]

    def _build_logger(self) -> Logger:
        logger = WandbLogger(project="mhist-mlops")
        logger.watch(self.model, log="all", log_freq=100)
        return logger

    def _build_model(self) -> LightningModule:
        model = LightningViT()
        return model


    def _build_trainer(self) -> Trainer:
        """
        Simple wrapper allowing for more complex Trainer construction later.
        """
        callbacks = self._assemble_callbacks()
        trainer = Trainer(max_epochs=self.run_info.max_epochs, 
                          logger=self.logger, 
                          callbacks=callbacks)
        return trainer

    def run(self):
        self.trainer.fit(self.model, self.datamodule)

        # TODO: what sort of outputs are needed here?
        pass

if __name__ == '__main__':
    # load in the configs from YML files
    run_config = None
    data_config = None

    mhist_training = MHISTTraining(run_config, data_config)
    mhist_training.run()
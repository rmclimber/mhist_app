# base python imports
from yaml import safe_load
from argparse import ArgumentParser
import os
from google.cloud import storage


# lightning imports
from lightning.pytorch.loggers import WandbLogger
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# other packages
import wandb

# my imports
from ..model.lightning_vit import *
from ..data.mhist_datamodule import *
from ..data.data_info import *
from .run_info import *
from ..gcp.gcp_info import *

class MHISTTraining:
    def __init__(self, config: dict):
        # construct config objects
        self.run_info = RunInfo(**config["run"])
        self.data_config = MHISTDataConfig(**config["data"])
        self.gcp_config = GCPInfo(**config["gcp"])
        
        # prepare client for storage bucket interaction
        self.storage_client = storage.Client()

        # collect basic information about the datasets themselves
        self.data_info = MHISTDataInfo(
            info_dict=safe_load(open(self.data_config.info_path, 'r')))
        
        # assemble all pieces for the training run
        self.datamodule = self._build_datamodule(self.run_info, self.data_config)
        self.logger = self._build_logger()
        self.model = self._build_model()
        self.trainer = self._build_trainer()

    def _download_data(self,
                       client: storage.Client,
                       data_config: MHISTDataConfig,
                       gcp_config: GCPInfo):
        
        # set up for download
        data_bucket = client.get_bucket(gcp_config.data_bucket)
        os.makedirs(data_config.data_path, exist_ok=True)

        for blob in data_bucket.list_blobs():
            blob.download_to_filename(str(data_config.data_path / blob.name))
            print(f"Downloaded {blob.name} to {data_config.data_path}")

    def _build_datamodule(self, 
                          data_config: MHISTDataConfig = None, 
                          data_info: MHISTDataInfo = None) -> LightningDataModule:
        """
        All data-relevant objects are assembled here, other than the configs.
        """
        if data_config is None:
            data_config = self.data_config
        if data_info is None:
            data_info = self.data_info

        if not hasattr(self, "client"):
            self.client = storage.Client()

        self._download_data(data_config)

        # initialize all the datasets
        self.datasets = {}
        for mode in ["train", "val", "test"]:
            self.datasets[mode] = MHISTDataset(
                img_filename=getattr(data_config, f"{mode}_image_path"),
                label_filename=getattr(data_config, f"{mode}_label_path"),
                data_info=data_info,
                mode=mode)
        
        # assemble the datamodule
        datamodule = MHISTDataModule(datasets=self.datasets,
                                     data_config=data_config,
                                     data_info=data_info)
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
        """
        Specify the logger here.
        """
        wandb.login()
        logger = WandbLogger(project="mhist-mlops")
        logger.watch(self.model, log="all", log_freq=100)
        return logger

    def _build_model(self, 
                     data_info: MHISTDataInfo = None,
                     run_info: RunInfo = None) -> LightningModule:
        """
        Simple wrapper for more complex model construction later.
        """
        if data_info is None:
            data_info = self.data_info
        if run_info is None:
            run_info = self.run_info

        model = LightningViT(num_classes=data_info.num_classes,
                             lr=run_info.lr)
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
    
    def upload_checkpoints(self):
        for checkpoint_filename in self.run_info.checkpoint_dir.glob("*.ckpt"):
            self.s3.upload_file()

    def generate_outputs(self):
        pass

    def run(self):
        self.trainer.fit(self.model, self.datamodule)
        # TODO: upload checkpoints to registry?
        self.upload_checkpoints()

        # TODO: what sort of outputs are needed here? prob just stats
        self.generate_outputs()
        pass

if __name__ == '__main__':
    # get filename location from the command line (passed in entrypoint)
    parser = ArgumentParser(prog="MHIST MLOps project")
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    # load in the config from YML file
    config = safe_load(open(args.config, 'r'))

    # do the training
    mhist_training = MHISTTraining(config)
    mhist_training.run()
# base python imports
from yaml import safe_load
from argparse import ArgumentParser
import os
from google.cloud import storage
import json
from datetime import datetime
from pathlib import Path

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
from .output_manager import *

class MHISTTraining:
    def __init__(self, config: dict):
        # construct config objects
        self.run_info = RunInfo(**config["run"])
        self.data_config = MHISTDataConfig(**config["data"])
        self.gcp_config = GCPInfo(**config["gcp"])
        
        # prepare client for storage bucket interaction
        self._bucket_setup()

        # collect basic information about the datasets themselves
        self._data_info_setup()
        
        # assemble all pieces for the training run
        self.datamodule = self._build_datamodule(
            data_config=self.data_config, data_info=self.data_info)
        self.model = self._build_model()
        self.logger = self._build_logger()
        self.trainer = self._build_trainer()

        # prepare relative filename for bucket
        self.version = self.logger.version
        self.output_path = Path(self.version)

    def _bucket_setup(self):
        # prepare client
        if not hasattr(self, "client") or self.client is None:
            self.client = storage.Client()
        
        # prepare data storage bucket
        if not hasattr(self, "data_bucket") or self.data_bucket is None:
            self.data_bucket = self.client.get_bucket(
                self.gcp_config.data_bucket)
        
        # prepare bucket for run outputs
        if not hasattr(self, "output_bucket") or self.output_bucket is None:
            self.output_bucket = self.client.get_bucket(
                self.gcp_config.output_bucket)

    def _data_info_setup(self):
        # ensure buckets are ready to go
        self._bucket_setup()

        # set up for download
        os.makedirs(self.data_config.info_path.parent, exist_ok=True)

        blob = self.data_bucket.blob(str(self.data_config.info_path.name))
        blob.download_to_filename(str(self.data_config.info_path))
        
        self.data_info = MHISTDataInfo(
            info_dict=safe_load(open(self.data_config.info_path, 'r')))

    def _download_data(self,
                       data_config: MHISTDataConfig):
        
        # ensure buckets are ready to go
        self._bucket_setup()

        # set up for download
        os.makedirs(data_config.data_path, exist_ok=True)

        for blob in self.data_bucket.list_blobs():
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

        # ensure client exists; then download data
        self._bucket_setup()
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
        early_stopping = EarlyStopping(
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
        self._bucket_setup()
        for checkpoint_filename in self.run_info.checkpoint_dir.glob("*.ckpt"):
            checkpt_blob = self.output_bucket.blob(
                str(self.output_path / checkpoint_filename))
            checkpt_blob.upload_from_filename(checkpoint_filename)


    def generate_outputs(self,
                         model: LightningModule,
                         trainer: Trainer,
                         datamodule: LightningDataModule) -> dict:
        """
        Used to collect all necessary outputs.
        """
        manager = OutputManager(model=model,
                                trainer=trainer,
                                datamodule=datamodule)
        return manager.generate_outputs()
    
    def upload_outputs(self, outputs: dict[str, dict]):
        """
        Uploads all the output stats into the GCP bucket at the specified path.
        """
        # prelimaries
        self._bucket_setup()

        # set up base filename
        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = "_".join([
            self.version, datetime_str, "{}", "stats.json"])
        
        # upload each output
        for key, val in outputs.items():
            json_str = json.dumps(val)
            blob = self.output_bucket.blob(
                self.output_path / output_filename.format(key))
            blob.upload_from_string(json_str)

    def run(self):
        self.trainer.fit(self.model, self.datamodule)
        self.upload_checkpoints()

        # collect and save outputs as necessary
        outputs = self.generate_outputs(
            model=self.model, trainer=self.trainer, datamodule=self.datamodule
        )
        self.upload_outputs(outputs)

        val_loss = self.trainer.validate(self.model, self.datamodule)[0]["val_loss"]
        return val_loss

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
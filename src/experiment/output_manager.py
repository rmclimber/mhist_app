import numpy as np
import sklearn as skl
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score)

from lightning import LightningModule, LightningDataModule, Trainer

# my code
from .results_info import *

class OutputManager:
    metrics = {metric.__name__: metric for metric in [
        accuracy_score, balanced_accuracy_score, confusion_matrix, 
        f1_score, precision_score, recall_score]}
    
    def __init__(self, 
                 model: LightningModule, 
                 trainer: Trainer,
                 datamodule: LightningDataModule):
        self.model = model
        self.trainer = trainer
        self.datamodule = datamodule
    
    def _consolidate_results_info(self, results_list: list[ResultsInfo]):
        """
        Trainer.predict() returns a list of batch results. In our case, that
        will be a list of ResultsInfo objects. They need to be consolidated for
        use gathering stats.
        """
        y = np.array([y for results_info in results_list for y in results_info.y])
        y_hat = np.array([y_hat for results_info in results_list for y_hat in results_info.y_hat])
        logits = np.array([logits for results_info in results_list for logits in results_info.logits])
        return ResultsInfo(y=y, logits=logits, y_hat=y_hat)

    def _collect_predictions(self,
                             model: LightningModule = None,
                             trainer: Trainer = None,
                             datamodule: LightningDataModule = None):
        """
        Uses Trainer.predict() to get predictions from the model across train,
        val, and test sets.
        """
        if model is None:
            model = self.model
        if trainer is None:
            trainer = self.trainer
        if datamodule is None:
            datamodule = self.datamodule
        
        results_dict = {}

        for mode in ["train", "val", "test"]:
            results_list = trainer.predict(
                model,
                dataloaders=datamodule.predict_dataloader(mode))
            results_dict[mode] = self._consolidate_results_info(results_list)
        
        return results_dict
    
    def _collect_output_stats(self, results_dict: dict[str, ResultsInfo]):
        """
        Score all predictions from the classifier. 
        """
        output_dict = {}
        for mode in ["train", "val", "test"]:
            output_dict[mode] = {}
            results_info = results_dict[mode]
            for metric_name, metric in self.metrics.items():
                score = metric(results_info.y, results_info.y_hat)
                print(f"{mode} {metric_name}: {score}")
                if isinstance(score, np.ndarray):
                    score = score.tolist()
                elif isinstance(score, np.float64 | np.float32):
                    score = float(score)
                output_dict[mode][metric_name] = score
        return output_dict
    
    def generate_outputs(self, 
                         model: LightningModule = None, 
                         trainer: Trainer = None, 
                         datamodule: LightningDataModule = None):
        """
        The only method the client should touch. 
        """
        if model is None:
            model = self.model
        if trainer is None:
            trainer = self.trainer
        if datamodule is None:
            datamodule = self.datamodule
        
        # get all the predictions and evaluate the prediction quality
        results_dict = self._collect_predictions(model, trainer, datamodule)
        output_stats = self._collect_output_stats(results_dict)
        return output_stats
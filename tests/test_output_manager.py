import pytest
import numpy as np
import sys
import torch
from pathlib import Path

# Adjust the path to allow importing from the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiment.output_manager import OutputManager
from src.experiment.results_info import ResultsInfo
from lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

class DummyTrainer:
    @staticmethod
    def predict(model, dataloaders=None, datamodule=None, mode=None, return_predictions=True):
        # Simulate batches of ResultsInfo objects
        y = np.array([0, 1, 1, 0])
        logits = np.array([[2.0, 1.0], [1.0, 2.0], [0.5, 1.5], [1.2, 0.8]])
        y_hat = np.argmax(logits, axis=1)
        # Return a list of ResultsInfo objects (simulate 2 batches)
        return [ResultsInfo(y=y[:2], logits=logits[:2], y_hat=y_hat[:2]),
                ResultsInfo(y=y[2:], logits=logits[2:], y_hat=y_hat[2:])]

class DummyModel(LightningModule):
    """Dummy model for testing that mimics the LightningViT interface."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        # Simple linear layer for dummy predictions
        self.classifier = torch.nn.Linear(10, num_classes)  # 10 input features, 2 classes
    
    def forward(self, x):
        """Forward pass that returns logits."""
        return self.classifier(x)
    
    def predict_step(self, batch, batch_idx):
        """Predict step that returns ResultsInfo objects like the real LightningViT."""
        x, y = batch
        logits = self(x)
        y_hat = torch.argmax(logits, dim=1)
        return ResultsInfo(y=y.cpu().numpy(), logits=logits.cpu().numpy(), y_hat=y_hat.cpu().numpy())


class DummyDataModule(LightningDataModule):
    """Dummy datamodule for testing that mimics the MHISTDataModule interface."""
    
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        # Create dummy datasets with consistent data
        self._create_dummy_datasets()
    
    def _create_dummy_datasets(self):
        """Create dummy datasets for train, val, and test."""
        # Create dummy data: 4 samples with 10 features each
        x = torch.randn(4, 10)  # 4 samples, 10 features
        y = torch.tensor([0, 1, 1, 0])  # Binary labels
        
        # Create datasets for each split
        self.train_dataset = TensorDataset(x, y)
        self.val_dataset = TensorDataset(x, y)
        self.test_dataset = TensorDataset(x, y)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self, mode: str):
        """Predict dataloader that accepts mode parameter like MHISTDataModule."""
        if mode == "train":
            return self.train_dataloader()
        elif mode == "val":
            return self.val_dataloader()
        elif mode == "test":
            return self.test_dataloader()
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'train', 'val', or 'test'.")

def test_consolidate_results_info():
    y1 = np.array([0, 1])
    y2 = np.array([1, 0])
    logits1 = np.array([[2.0, 1.0], [1.0, 2.0]])
    logits2 = np.array([[0.5, 1.5], [1.2, 0.8]])
    y_hat1 = np.argmax(logits1, axis=1)
    y_hat2 = np.argmax(logits2, axis=1)
    results_list = [ResultsInfo(y=y1, logits=logits1, y_hat=y_hat1),
                    ResultsInfo(y=y2, logits=logits2, y_hat=y_hat2)]
    manager = OutputManager(DummyModel(), DummyTrainer(), DummyDataModule())
    consolidated = manager._consolidate_results_info(results_list)
    assert np.array_equal(consolidated.y, np.array([0, 1, 1, 0]))
    assert consolidated.logits.shape == (4, 2)
    assert consolidated.y_hat.shape == (4,)


def test_collect_predictions():
    manager = OutputManager(DummyModel(), DummyTrainer(), DummyDataModule())
    results = manager._collect_predictions()
    assert set(results.keys()) == {"train", "val", "test"}
    for mode in ["train", "val", "test"]:
        assert isinstance(results[mode], ResultsInfo)
        assert results[mode].y.shape[0] == 4
        assert results[mode].logits.shape == (4, 2)
        assert results[mode].y_hat.shape == (4,)


def test_collect_output_stats():
    manager = OutputManager(DummyModel(), DummyTrainer(), DummyDataModule())
    # Simulate results_dict
    y = np.array([0, 1, 1, 0])
    y_hat = np.array([0, 1, 1, 0])
    logits = np.array([[2.0, 1.0], [1.0, 2.0], [0.5, 1.5], [1.2, 0.8]])
    results_dict = {mode: ResultsInfo(y=y, logits=logits, y_hat=y_hat) for mode in ["train", "val", "test"]}
    output_stats = manager._collect_output_stats(results_dict)
    for mode in ["train", "val", "test"]:
        assert "accuracy_score" in output_stats[mode]
        assert "f1_score" in output_stats[mode]
        assert "confusion_matrix" in output_stats[mode]
        assert output_stats[mode]["accuracy_score"] == 1.0


def test_generate_outputs():
    manager = OutputManager(DummyModel(), DummyTrainer(), DummyDataModule())
    output_stats = manager.generate_outputs()
    assert set(output_stats.keys()) == {"train", "val", "test"}
    for mode in ["train", "val", "test"]:
        assert "accuracy_score" in output_stats[mode]
        assert "f1_score" in output_stats[mode]
        assert "confusion_matrix" in output_stats[mode]
        assert isinstance(output_stats[mode]["accuracy_score"], float)
        assert isinstance(output_stats[mode]["f1_score"], float)
        assert isinstance(output_stats[mode]["confusion_matrix"], list)

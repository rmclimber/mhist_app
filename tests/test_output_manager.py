import pytest
import numpy as np
import sys
from pathlib import Path

# Adjust the path to allow importing from the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiment.output_manager import OutputManager
from src.experiment.results_info import ResultsInfo

class DummyTrainer:
    @staticmethod
    def predict(model, datamodule, mode=None, return_predictions=True):
        # Simulate batches of ResultsInfo objects
        y = np.array([0, 1, 1, 0])
        logits = np.array([[2.0, 1.0], [1.0, 2.0], [0.5, 1.5], [1.2, 0.8]])
        y_hat = np.argmax(logits, axis=1)
        # Return a list of ResultsInfo objects (simulate 2 batches)
        return [ResultsInfo(y=y[:2], logits=logits[:2], y_hat=y_hat[:2]),
                ResultsInfo(y=y[2:], logits=logits[2:], y_hat=y_hat[2:])]

class DummyModel:
    pass

class DummyDataModule:
    pass

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

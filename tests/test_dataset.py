import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Adjust the path to allow importing from the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.mhist_dataset import MHISTDataset
from src.data.data_info import MHISTDataInfo

@pytest.fixture
def sample_data_info():
    """
    Fixture to create a sample MHISTDataInfo object.
    """
    info_dict = {
        "label_map": {"0": "HP", "1": "SSA"},
        "stats": {"mean": [128.0, 128.0, 128.0], "std": [64.0, 64.0, 64.0]}
    }
    return MHISTDataInfo(info_dict=info_dict)

@pytest.fixture
def dummy_image_data():
    """
    Fixture to create dummy image data for testing.
    """
    return np.random.randint(0, 256, size=(10, 3, 224, 224), dtype=np.uint8)

@pytest.fixture
def dummy_label_data():
    """
    Fixture to create dummy label data for testing.
    """
    return np.random.randint(0, 2, size=(10,), dtype=np.int64)

@pytest.fixture
def dummy_data_files(tmp_path, dummy_image_data, dummy_label_data):
    """
    Fixture to create dummy .npy files for images and labels.
    """
    img_file = tmp_path / "dummy_images.npy"
    label_file = tmp_path / "dummy_labels.npy"
    np.save(img_file, dummy_image_data)
    np.save(label_file, dummy_label_data)
    return img_file, label_file

@pytest.fixture
def sample_mhist_dataset(dummy_data_files, sample_data_info):
    """
    Fixture to create a sample MHISTDataset for testing.
    """
    img_file, label_file = dummy_data_files
    dataset = MHISTDataset(
        img_filename=str(img_file),
        label_filename=str(label_file),
        data_info=sample_data_info,
        mode="train"
    )
    return dataset


def test_mhist_dataset_length(sample_mhist_dataset, dummy_image_data):
    """
    Test that the length of the dataset matches the number of images.
    """
    dataset = sample_mhist_dataset
    assert len(dataset) == dummy_image_data.shape[0]


def test_mhist_dataset_getitem_shape(sample_mhist_dataset):
    """
    Test that __getitem__ returns a tuple of (image, label) with correct shapes/types.
    """
    img, label = sample_mhist_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape[0] == 3  # channels
    assert img.shape[1] == 224 and img.shape[2] == 224
    assert label.dim() == 0 or label.shape == torch.Size([])


def test_mhist_dataset_label_values(sample_mhist_dataset):
    """
    Test that labels are within the expected range (0 or 1).
    """
    labels = [sample_mhist_dataset[i][1].item() for i in range(len(sample_mhist_dataset))]
    assert all(l in [0, 1] for l in labels)


def test_mhist_dataset_transform_train_mode(sample_mhist_dataset):
    """
    Test that transforms are applied in train mode and output is normalized.
    """
    img, _ = sample_mhist_dataset[0]
    # After normalization, values should be roughly in a standard range
    assert img.dtype == torch.float32
    assert img.min() < 1.0 and img.max() > -1.0


def test_mhist_dataset_eval_mode(dummy_data_files, sample_data_info):
    """
    Test that transforms in eval mode do not apply augmentation and normalization is correct.
    """
    img_file, label_file = dummy_data_files
    dataset = MHISTDataset(
        img_filename=str(img_file),
        label_filename=str(label_file),
        data_info=sample_data_info,
        mode="val"
    )
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3
    assert img.dtype == torch.float32
    # Check normalization: mean should be close to 0 after normalization
    assert abs(float(img.mean())) < 2.0


def test_mhist_dataset_out_of_bounds(sample_mhist_dataset):
    """
    Test that IndexError is raised when accessing out-of-bounds index.
    """
    with pytest.raises(IndexError):
        _ = sample_mhist_dataset[100]


def test_mhist_dataset_label_type(sample_mhist_dataset):
    """
    Test that label is a scalar tensor of type long/int64.
    """
    _, label = sample_mhist_dataset[0]
    assert label.dtype in (torch.int64, torch.long)
    assert label.dim() == 0 or label.shape == torch.Size([])


def test_mhist_dataset_multiple_items(sample_mhist_dataset):
    """
    Test that multiple items can be accessed and are consistent.
    """
    for i in range(5):
        img, label = sample_mhist_dataset[i]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert img.shape[0] == 3
        assert img.shape[1] == 224 and img.shape[2] == 224
        assert label.item() in [0, 1]
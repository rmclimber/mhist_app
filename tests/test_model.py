import pytest
import torch
import sys
from pathlib import Path

# to make the last import work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model.lightning_vit import LightningViT

@pytest.fixture
def sample_lightning_vit():
    """
    Fixture to create a sample LightningViT model for testing.
    """
    num_classes = 2
    model = LightningViT(num_classes=num_classes)
    return model

def test_lightning_vit_init(sample_lightning_vit):
    """
    Test the initialization of the LightningViT model.
    """
    model = sample_lightning_vit
    assert model.num_classes == 2
    assert model.lr == 1e-4
    assert hasattr(model, 'vit')
    assert hasattr(model, 'classifier')
    assert isinstance(model.loss_fn, torch.nn.CrossEntropyLoss)

def test_lightning_vit_forward(sample_lightning_vit):
    """
    Test the forward pass of the LightningViT model.
    """
    model = sample_lightning_vit
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == torch.Size([1, model.num_classes])

def test_lightning_vit_step(sample_lightning_vit):
    """
    Test the step method (used by training_step, validation_step, test_step).
    """
    model = sample_lightning_vit
    # Create dummy batch
    pixel_values = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, model.num_classes, (2,))
    batch = (pixel_values, labels)

    loss = model.step(batch, "train")
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_lightning_vit_training_step(sample_lightning_vit):
    """
    Test the training_step method.
    """
    model = sample_lightning_vit
    pixel_values = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, model.num_classes, (2,))
    batch = (pixel_values, labels)
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_lightning_vit_validation_step(sample_lightning_vit):
    """
    Test the validation_step method.
    """
    model = sample_lightning_vit
    pixel_values = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, model.num_classes, (2,))
    batch = (pixel_values, labels)
    loss = model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_lightning_vit_test_step(sample_lightning_vit):
    """
    Test the test_step method.
    """
    model = sample_lightning_vit
    pixel_values = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, model.num_classes, (2,))
    batch = (pixel_values, labels)
    loss = model.test_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_lightning_vit_predict_step(sample_lightning_vit):
    """
    Test the predict_step method.
    """
    model = sample_lightning_vit
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, model.num_classes, (2,))
    batch = (x, y)
    results_info = model.predict_step(batch, 0)
    assert hasattr(results_info, 'y')
    assert hasattr(results_info, 'logits')
    assert hasattr(results_info, 'y_hat')
    assert results_info.y.shape == y.shape
    assert results_info.y_hat.shape == y.shape
    assert results_info.logits.shape == torch.Size([2, model.num_classes])
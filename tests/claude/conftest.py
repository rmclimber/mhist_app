import pytest
from fastapi.testclient import TestClient
import httpx
from unittest.mock import AsyncMock
from io import BytesIO
from PIL import Image
import os
import tempfile


@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient for testing"""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def test_client(mock_http_client, monkeypatch):
    """Create test client with mocked dependencies"""
    # Mock the global http_client
    monkeypatch.setattr("main.http_client", mock_http_client)
    
    # Set test environment variables
    test_env = {
        "CLASSIFICATION_SERVICE_URL": "http://test-classification:8001",
        "VALIDATION_SERVICE_URL": "http://test-validation:8002",
        "ENABLE_VALIDATION": "false",
        "GITHUB_REPO_URL": "https://github.com/test/repo",
        "WANDB_PROJECT_URL": "https://wandb.ai/test/project"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Import app after setting environment variables
    from main import app
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple 100x100 RGB image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_image_file(sample_image):
    """Create a temporary image file for upload testing"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(sample_image.getvalue())
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_classification_response():
    """Mock response from classification service"""
    return {
        "prediction": "cat",
        "confidence": 0.95,
        "class_probabilities": {
            "cat": 0.95,
            "dog": 0.03,
            "bird": 0.02
        }
    }


@pytest.fixture
def mock_validation_response():
    """Mock response from validation service"""
    return {"is_valid": True}


@pytest.fixture
def mock_model_stats():
    """Mock model statistics response"""
    return {
        "accuracy": 0.92,
        "total_parameters": 1000000,
        "model_type": "ResNet50",
        "training_date": "2024-01-15",
        "dataset_size": 50000
    }
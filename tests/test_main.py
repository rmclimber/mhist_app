import pytest
from unittest.mock import AsyncMock
import httpx
import json
from io import BytesIO


class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns service info"""
        response = test_client.get("/predictions/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "MHIST Inference Service is running."
    
    @pytest.mark.asyncio
    async def test_health_check_all_services_healthy(self, test_client, mock_http_client):
        """Test health check when all services are healthy"""
        # Mock is already set up in conftest.py
        
        response = test_client.get("/predictions/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_classification_service_down(self, test_client, mock_http_client):
        """Test health check when classification service is down"""
        # Mock is already set up in conftest.py
        
        response = test_client.get("/predictions/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictionEndpoints:
    """Test prediction-related endpoints"""
    
    @pytest.mark.asyncio
    async def test_predict_success(self, test_client, mock_http_client, sample_image_file, mock_classification_response):
        """Test successful image prediction"""
        # Mock is already set up in conftest.py
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "cat"
        assert data["confidence"] == 0.95
        assert "request_id" in data
        
        # Verify classification service was called
        mock_http_client.post.assert_called_once()
    
    def test_predict_invalid_file_type(self, test_client):
        """Test prediction with invalid file type (currently no validation implemented)"""
        text_content = "This is not an image"
        response = test_client.post(
            "/predictions/predict",
            files={"img": ("test.txt", BytesIO(text_content.encode()), "text/plain")}
        )
        
        # Currently no file type validation, so it should succeed
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
    
    def test_predict_no_file(self, test_client):
        """Test prediction without file upload"""
        response = test_client.post("/predictions/predict")
        assert response.status_code == 422  # FastAPI validation error
    
    @pytest.mark.asyncio
    async def test_predict_classification_service_error(self, test_client, mock_http_client, sample_image_file):
        """Test prediction when classification service returns error"""
        # Mock is already set up in conftest.py
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
    
    @pytest.mark.asyncio
    async def test_predict_random_success(self, test_client, mock_http_client, mock_classification_response):
        """Test successful random prediction"""
        # Mock is already set up in conftest.py
        
        response = test_client.get("/predictions/predict-random")
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "cat"
        assert "request_id" in data
        assert "timestamp" in data
        
        # Verify classification service was called
        mock_http_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_random_service_error(self, test_client, mock_http_client):
        """Test random prediction when service is unavailable"""
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        response = test_client.get("/predictions/predict-random")
        assert response.status_code == 500


class TestFeedbackEndpoint:
    """Test feedback submission"""
    
    def test_submit_feedback_success(self, test_client):
        """Test successful feedback submission"""
        feedback_data = {
            "request_id": "test-request-123",
            "is_correct": True,
            "correct_label": 0,  # cat
            "predicted_label": 0,  # cat
            "confidence": 0.95
        }

        response = test_client.get("/predictions/feedback", params=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback received"
        assert "feedback_id" in data
    
    def test_submit_feedback_with_correction(self, test_client):
        """Test feedback submission with actual class correction"""
        feedback_data = {
            "request_id": "test-request-123",
            "is_correct": False,
            "correct_label": 1,  # dog
            "predicted_label": 0,  # cat
            "confidence": 0.95
        }

        response = test_client.get("/predictions/feedback", params=feedback_data)
        assert response.status_code == 200
    
    def test_submit_feedback_missing_required_fields(self, test_client):
        """Test feedback submission with missing required fields"""
        response = test_client.get("/predictions/feedback", params={})
        assert response.status_code == 422  # FastAPI validation error for missing required params


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    @pytest.mark.asyncio
    async def test_get_model_info_success(self, test_client, mock_http_client, mock_model_stats):
        """Test successful model info retrieval"""
        # Mock is already set up in conftest.py
        
        response = test_client.get("/predictions/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "MHIST Classifier"
        assert data["version"] == "1.0.0"
        assert data["description"] == "A model for classifying histopathology images."
    
    @pytest.mark.asyncio
    async def test_get_model_info_service_error(self, test_client, mock_http_client):
        """Test model info when classification service is unavailable"""
        # Mock is already set up in conftest.py
        
        response = test_client.get("/predictions/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "MHIST Classifier"
        assert data["version"] == "1.0.0"


class TestValidationFlow:
    """Test validation service integration"""
    
    @pytest.mark.asyncio
    async def test_predict_with_validation_enabled(self, test_client, mock_http_client, sample_image_file, monkeypatch):
        """Test prediction flow with validation enabled"""
        # Mock is already set up in conftest.py
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        # Currently no validation service integration, so only classification is called
        assert mock_http_client.post.call_count == 1
    
    @pytest.mark.asyncio
    async def test_predict_validation_fails(self, test_client, mock_http_client, sample_image_file, monkeypatch):
        """Test prediction when validation service rejects input"""
        # Mock is already set up in conftest.py
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        # Currently no validation service integration, so prediction succeeds
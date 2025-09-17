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
        # Mock successful responses from downstream services
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_http_client.get.return_value = mock_response
        
        response = test_client.get("/predictions/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["classification"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_classification_service_down(self, test_client, mock_http_client):
        """Test health check when classification service is down"""
        # Mock failed response from classification service
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        response = test_client.get("/predictions/health")
        assert response.status_code == 200
        data = response.json()
        assert data["services"]["classification"]["status"] == "unhealthy"
        assert "error" in data["services"]["classification"]


class TestPredictionEndpoints:
    """Test prediction-related endpoints"""
    
    @pytest.mark.asyncio
    async def test_predict_success(self, test_client, mock_http_client, sample_image_file, mock_classification_response):
        """Test successful image prediction"""
        # Mock classification service response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_classification_response
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
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
        assert "timestamp" in data
        
        # Verify classification service was called
        mock_http_client.post.assert_called_once()
    
    def test_predict_invalid_file_type(self, test_client):
        """Test prediction with invalid file type"""
        text_content = "This is not an image"
        response = test_client.post(
            "/predictions/predict",
            files={"img": ("test.txt", BytesIO(text_content.encode()), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]
    
    def test_predict_no_file(self, test_client):
        """Test prediction without file upload"""
        response = test_client.post("/predictions/predict")
        assert response.status_code == 422  # FastAPI validation error
    
    @pytest.mark.asyncio
    async def test_predict_classification_service_error(self, test_client, mock_http_client, sample_image_file):
        """Test prediction when classification service returns error"""
        # Mock classification service error
        mock_http_client.post.side_effect = httpx.HTTPStatusError(
            "Service error", 
            request=AsyncMock(), 
            response=AsyncMock(status_code=500)
        )
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 500
        assert "Classification service unavailable" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_predict_random_success(self, test_client, mock_http_client, mock_classification_response):
        """Test successful random prediction"""
        # Add random image data to mock response
        random_response = {
            **mock_classification_response,
            "test_image_path": "test_images/cat_001.jpg",
            "image_url": "data:image/jpeg;base64,/9j/4AAQ..."
        }
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = random_response
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        response = test_client.get("/predictions/predict-random")
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "cat"
        assert "test_image_path" in data
        assert "request_id" in data
        
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
            "predicted_class": "cat",
            "confidence": 0.95
        }
        
        response = test_client.post("/predictions/feedback", params=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback recorded"
        assert "feedback_id" in data
    
    def test_submit_feedback_with_correction(self, test_client):
        """Test feedback submission with actual class correction"""
        feedback_data = {
            "request_id": "test-request-123",
            "is_correct": False,
            "predicted_class": "cat",
            "actual_class": "dog",
            "confidence": 0.95
        }
        
        response = test_client.post("/predictions/feedback", params=feedback_data)
        assert response.status_code == 200
    
    def test_submit_feedback_missing_required_fields(self, test_client):
        """Test feedback submission with missing required fields"""
        response = test_client.post("/predictions/feedback", params={})
        assert response.status_code == 422  # FastAPI validation error


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    @pytest.mark.asyncio
    async def test_get_model_info_success(self, test_client, mock_http_client, mock_model_stats):
        """Test successful model info retrieval"""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_model_stats
        mock_http_client.get.return_value = mock_response
        
        response = test_client.get("/predictions/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_stats" in data
        assert "project_info" in data
        assert "pipeline_config" in data
        assert data["model_stats"]["accuracy"] == 0.92
        assert data["project_info"]["github_url"] == "https://github.com/test/repo"
    
    @pytest.mark.asyncio
    async def test_get_model_info_service_error(self, test_client, mock_http_client):
        """Test model info when classification service is unavailable"""
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        response = test_client.get("/predictions/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_stats"] == {}  # Empty due to service error
        assert "error" in data
        assert data["project_info"]["github_url"] == "https://github.com/test/repo"


class TestValidationFlow:
    """Test validation service integration"""
    
    @pytest.mark.asyncio
    async def test_predict_with_validation_enabled(self, test_client, mock_http_client, sample_image_file, monkeypatch):
        """Test prediction flow with validation enabled"""
        # Enable validation
        monkeypatch.setenv("ENABLE_VALIDATION", "true")
        
        # Mock validation service response
        validation_response = AsyncMock()
        validation_response.json.return_value = {"is_valid": True}
        validation_response.raise_for_status.return_value = None
        
        # Mock classification service response
        classification_response = AsyncMock()
        classification_response.status_code = 200
        classification_response.json.return_value = {"prediction": "cat", "confidence": 0.95}
        classification_response.raise_for_status.return_value = None
        
        # Set up mock to return different responses for different URLs
        def mock_post(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if 'validate' in url:
                return validation_response
            else:
                return classification_response
        
        mock_http_client.post.side_effect = mock_post
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        # Should have called both validation and classification services
        assert mock_http_client.post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_predict_validation_fails(self, test_client, mock_http_client, sample_image_file, monkeypatch):
        """Test prediction when validation service rejects input"""
        # Enable validation
        monkeypatch.setenv("ENABLE_VALIDATION", "true")
        
        # Mock validation service response - invalid input
        validation_response = AsyncMock()
        validation_response.json.return_value = {"is_valid": False}
        validation_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = validation_response
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "validation failed" in data["error"]
        
        # Should only call validation service, not classification
        assert mock_http_client.post.call_count == 1
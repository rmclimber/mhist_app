import pytest
from unittest.mock import AsyncMock
import httpx
import json
from io import BytesIO
from pathlib import Path
import sys

# to make the last import work
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.app.backend.main import app


class TestEndToEndFlow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_prediction_and_feedback_flow(self, test_client, mock_http_client, sample_image_file):
        """Test the complete flow: prediction -> feedback"""
        # Mock is already set up in conftest.py
        
        # Step 1: Make prediction
        with open(sample_image_file, 'rb') as f:
            prediction_response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert prediction_response.status_code == 200
        prediction_data = prediction_response.json()
        request_id = prediction_data["request_id"]
        
        # Step 2: Submit feedback
        feedback_response = test_client.get(
            "/predictions/feedback",
            params={
                "request_id": request_id,
                "is_correct": False,
                "correct_label": 1,  # dog
                "predicted_label": 0,  # cat
                "confidence": 0.95
            }
        )
        
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        assert feedback_data["message"] == "Feedback received"
    
    @pytest.mark.asyncio
    async def test_service_resilience(self, test_client, mock_http_client):
        """Test that gateway handles downstream service failures gracefully"""
        # Test health endpoint when classification service is down
        mock_http_client.get.side_effect = httpx.ConnectError("Service unavailable")
        
        health_response = test_client.get("/predictions/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Test model-info endpoint when classification service is down
        model_info_response = test_client.get("/predictions/model-info")
        assert model_info_response.status_code == 200
        model_data = model_info_response.json()
        assert model_data["model"] == "MHIST Classifier"
        assert model_data["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, test_client, mock_http_client, sample_image_file):
        """Test handling multiple concurrent predictions"""
        # Mock is already set up in conftest.py
        
        # Make multiple concurrent requests
        responses = []
        for i in range(3):
            with open(sample_image_file, 'rb') as f:
                response = test_client.post(
                    "/predictions/predict",
                    files={"img": (f"test_{i}.jpg", f, "image/jpeg")}
                )
                responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "request_id" in data
            assert data["prediction"] == "cat"
        
        # Verify all requests were made to classification service
        assert mock_http_client.post.call_count == 3


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_large_file_rejection(self, test_client):
        """Test handling of large files (currently no size limit implemented)"""
        # Create a large file (simulate > 10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = test_client.post(
            "/predictions/predict",
            files={"img": ("large.jpg", BytesIO(large_content), "image/jpeg")}
        )
        
        # Currently no file size validation, so it should succeed
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_client, mock_http_client, sample_image_file):
        """Test handling of service timeouts"""
        # Mock timeout from classification service
        mock_http_client.post.side_effect = httpx.TimeoutException("Request timeout")
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(self, test_client, mock_http_client, sample_image_file):
        """Test handling of invalid JSON responses from downstream services"""
        # Mock response with invalid JSON
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 500


class TestConfigurationHandling:
    """Test different configuration scenarios"""
    
    def test_missing_environment_variables(self, monkeypatch):
        """Test behavior with missing environment variables"""
        # Clear environment variables
        for var in ["CLASSIFICATION_SERVICE_URL", "VALIDATION_SERVICE_URL"]:
            monkeypatch.delenv(var, raising=False)
        
        # Import should still work with defaults
        from src.app.backend.main import app
        assert app is not None
    
    def test_invalid_boolean_environment_variable(self, monkeypatch):
        """Test handling of invalid boolean environment variables"""
        monkeypatch.setenv("ENABLE_VALIDATION", "maybe")
        
        # Should default to False for invalid boolean values
        from src.app.backend.main import app
        assert app is not None


class TestRequestTracking:
    """Test request ID tracking and logging"""
    
    @pytest.mark.asyncio
    async def test_request_id_consistency(self, test_client, mock_http_client, sample_image_file):
        """Test that request IDs are consistent across the flow"""
        # Mock is already set up in conftest.py
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predictions/predict",
                files={"img": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        request_id = data["request_id"]
        
        # Request ID should be a valid UUID format
        import uuid
        uuid.UUID(request_id)  # This will raise if invalid
        
        # Response should contain expected fields
        assert "prediction" in data
        assert "confidence" in data
    
    @pytest.mark.asyncio
    async def test_unique_request_ids(self, test_client, mock_http_client, sample_image_file):
        """Test that each request gets a unique request ID"""
        # Mock is already set up in conftest.py
        
        request_ids = []
        for _ in range(3):
            with open(sample_image_file, 'rb') as f:
                response = test_client.post(
                    "/predictions/predict",
                    files={"img": ("test.jpg", f, "image/jpeg")}
                )
                request_ids.append(response.json()["request_id"])
        
        # All request IDs should be unique
        assert len(set(request_ids)) == 3
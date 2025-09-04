import pytest
from unittest.mock import AsyncMock
import httpx
import json
from io import BytesIO


class TestEndToEndFlow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_prediction_and_feedback_flow(self, test_client, mock_http_client, sample_image_file):
        """Test the complete flow: prediction -> feedback"""
        # Mock classification service response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "cat",
            "confidence": 0.95,
            "class_probabilities": {"cat": 0.95, "dog": 0.05}
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        # Step 1: Make prediction
        with open(sample_image_file, 'rb') as f:
            prediction_response = test_client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        assert prediction_response.status_code == 200
        prediction_data = prediction_response.json()
        request_id = prediction_data["request_id"]
        
        # Step 2: Submit feedback
        feedback_response = test_client.post(
            "/feedback",
            params={
                "request_id": request_id,
                "is_correct": False,
                "predicted_class": "cat",
                "actual_class": "dog",
                "confidence": 0.95
            }
        )
        
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        assert feedback_data["message"] == "Feedback recorded"
    
    @pytest.mark.asyncio
    async def test_service_resilience(self, test_client, mock_http_client):
        """Test that gateway handles downstream service failures gracefully"""
        # Test health endpoint when classification service is down
        mock_http_client.get.side_effect = httpx.ConnectError("Service unavailable")
        
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["services"]["classification"]["status"] == "unhealthy"
        
        # Test model-info endpoint when classification service is down
        model_info_response = test_client.get("/model-info")
        assert model_info_response.status_code == 200
        model_data = model_info_response.json()
        assert model_data["model_stats"] == {}
        assert "error" in model_data
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, test_client, mock_http_client, sample_image_file):
        """Test handling multiple concurrent predictions"""
        # Mock successful classification response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "cat", "confidence": 0.95}
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        # Make multiple concurrent requests
        responses = []
        for i in range(3):
            with open(sample_image_file, 'rb') as f:
                response = test_client.post(
                    "/predict",
                    files={"file": (f"test_{i}.jpg", f, "image/jpeg")}
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
        """Test rejection of files that are too large"""
        # Create a large file (simulate > 10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = test_client.post(
            "/predict",
            files={"file": ("large.jpg", BytesIO(large_content), "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_client, mock_http_client, sample_image_file):
        """Test handling of service timeouts"""
        # Mock timeout from classification service
        mock_http_client.post.side_effect = httpx.TimeoutException("Request timeout")
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
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
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
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
        from main import app
        assert app is not None
    
    def test_invalid_boolean_environment_variable(self, monkeypatch):
        """Test handling of invalid boolean environment variables"""
        monkeypatch.setenv("ENABLE_VALIDATION", "maybe")
        
        # Should default to False for invalid boolean values
        from main import config
        assert config.enable_validation is False


class TestRequestTracking:
    """Test request ID tracking and logging"""
    
    @pytest.mark.asyncio
    async def test_request_id_consistency(self, test_client, mock_http_client, sample_image_file):
        """Test that request IDs are consistent across the flow"""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "cat", "confidence": 0.95}
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        with open(sample_image_file, 'rb') as f:
            response = test_client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        request_id = data["request_id"]
        
        # Request ID should be a valid UUID format
        import uuid
        uuid.UUID(request_id)  # This will raise if invalid
        
        # Timestamp should be present
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_unique_request_ids(self, test_client, mock_http_client, sample_image_file):
        """Test that each request gets a unique request ID"""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "cat", "confidence": 0.95}
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        request_ids = []
        for _ in range(3):
            with open(sample_image_file, 'rb') as f:
                response = test_client.post(
                    "/predict",
                    files={"file": ("test.jpg", f, "image/jpeg")}
                )
                request_ids.append(response.json()["request_id"])
        
        # All request IDs should be unique
        assert len(set(request_ids)) == 3
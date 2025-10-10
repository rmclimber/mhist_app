import httpx
import asyncio
import uuid
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

# fastapi imports
from fastapi import FastAPI, UploadFile, HTTPException

# custom imports
from ..service_info import ServiceInfo

class InferenceGateway:
    def __init__(self, 
                 client: httpx.AsyncClient = None, 
                 logger: logging.Logger = None,
                 service_info: ServiceInfo = None):
        self.client = client or httpx.AsyncClient(timeout=60.0)
        self.logger = logger or self._get_logger()
        self.service_info = service_info

    def _get_logger(self):
        """
        Set up a logger for the inference pipeline.
        """
        logger = logging.getLogger("inference_pipeline")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    async def predict(self, img: UploadFile) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        self.logger.info(f"Received prediction request: {request_id}")
        try:
            data = await img.read()
            await img.seek(0)  # Reset file pointer after reading

            # TODO: implement validation later
            
            # Prediction logic here
            self.logger.info(f"Classifying image for request: {request_id}")
            result = await self._classify_image(data, img.filename)
            result["request_id"] = request_id
            return result
        except Exception as e:
            self.logger.error(f"Error in prediction request {request_id}: {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Prediction failed: {str(e)}")
    
    async def predict_random(self) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        self.logger.info(f"Received random prediction request: {request_id}")

        try:
            # Prediction logic here
            self.logger.info(f"Classifying random image for request: {request_id}")
            response = await self.client.get(
                f"{self.service_info.class_service_url}/predict_random")
            
            response.raise_for_status()
            result = response.json()
            result["request_id"] = request_id
            result["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
            return result
        
        except HTTPException as e:
            self.logger.error(f"Classification service error: {e.detail}")
            raise HTTPException(status_code=e.status_code, 
                                detail="Classification service error")
        
        except Exception as e:
            self.logger.error(f"Error in random prediction request {request_id}: {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Random prediction failed: {str(e)}")
    
    async def _classify_image(self, 
                              image_data: bytes, 
                              filename: str) -> Dict[str, Any]:
        self.logger.info(f"Classifying image: {filename}")
        # package request
        files = {'file': (filename, image_data, 'application/octet-stream')}

        #sent request
        try:
            response = await self.client.post(
                f"{self.service_info.class_service_url}/predict", files=files)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Received classification result for image: {filename}")
            return result
        except HTTPException as e:
            self.logger.error(f"Classification service error: {e.detail}")
            raise HTTPException(status_code=e.status_code, 
                                detail="Classification service error")
        except Exception as e:
            self.logger.error(f"Error during classification of image {filename}: {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Classification failed: {str(e)}")

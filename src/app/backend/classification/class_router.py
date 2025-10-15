import uuid
import logging
from fastapi import APIRouter, UploadFile

from .model_handler import ModelHandler

class ClassRouter:
    def __init__(self, logger: logging.Logger, model_handler: ModelHandler):
        self.router = APIRouter(prefix="/classify", tags=["classification"])
        self.logger = logger
        self.model_handler = model_handler
    
    def _register_routes(self):
        """
        Register routes with the router. Allows for clear namespacing. 
        """
        self.router.get("/health")(self.health_check)
        self.router.get("/")(self.root)
        self.router.get("/model-info")(self.model_info)
        self.router.post("/predict")(self.predict)
        self.router.get("/predict-random")(self.predict_random)
    
    async def health_check(self):
        return {"status": "healthy"}
    
    async def root(self):
        return {"message": "MHIST Classification Service is running."}
    
    async def model_info(self):
        return {
            "model": "MHIST Classifier",
            "version": "1.0.0",
            "description": "A model for classifying histopathology images.",
        }
    
    async def predict(self, img: UploadFile):
        return await self.model_handler.predict(img)
    
    async def predict_random(self):
        return await self.model_handler.predict_random()
    
    def register_routes(self):
        self._register_routes()
        return self.router
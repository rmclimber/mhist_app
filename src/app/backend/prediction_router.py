import uuid
import logging
from fastapi import APIRouter, UploadFile
from .inference_pipeline import InferencePipeline

class PredictionRouter:
    def __init__(self, logger: logging.Logger, inference_pipeline: InferencePipeline):
        self.router = APIRouter(prefix="/predictions", tags=["predictions"])
        self.logger = logger
        self.inference_pipeline = inference_pipeline

    def _register_routes(self):
        """
        Register routes with the router. Allows for clear namespacing. 
        """
        self.router.get("/health")(self.health_check)
        self.router.get("/")(self.root)
        self.router.get("/model-info")(self.model_info)
        self.router.post("/predict")(self.predict)
        self.router.get("/predict-random")(self.predict_random)
        self.router.get("/feedback")(self.submit_feedback)

    async def health_check(self):
        return {"status": "healthy"}
    
    async def root(self):
        return {"message": "MHIST Inference Service is running."}
    
    async def model_info(self):
        return {
            "model": "MHIST Classifier",
            "version": "1.0.0",
            "description": "A model for classifying histopathology images.",
        }

    async def predict(self, img: UploadFile):
        return await self.inference_pipeline.predict(img)

    async def predict_random(self):
        return await self.inference_pipeline.predict_random()
    
    async def submit_feedback(self,
                              request_id: str,
                              is_correct: bool,
                              correct_label: int = None,
                              predicted_label: int = None,
                              confidence: float = None):
        feedback = {
            "request_id": request_id,
            "is_correct": is_correct,
            "correct_label": correct_label,
            "predicted_label": predicted_label,
            "confidence": confidence
        }

        self.logger.info(f"Received feedback: {feedback}")
        return {"message": "Feedback received", "feedback_id": str(uuid.uuid4())}

    def register_routes(self):
        self._register_routes()
        return self.router
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from pathlib import Path
import random

# project code
from .model_handler import *
from .class_router import *

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
model_handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_handler

    # startup
    logger.info("Starting up the MHIST classification service...")

    try:
        # load model
        model_handler = ModelHandler(ServiceInfo())
        await model_handler.load_model()
        await model_handler.load_test_data()
        logger.info("Model and test data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or test data: {e}")
        raise

    yield
    # shutdown
    logger.info("Shutting down the MHIST classification service...")

app = FastAPI(title="MHIST Classification Service",
              version="1.0.0",
              lifespan=lifespan)

router = ClassRouter(model_handler=model_handler).register_routes()
app.include_router(router)
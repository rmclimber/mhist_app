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
from .model_handler import *

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
model_handler = None
labels = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_handler, model, labels, test_path

    # startup
    logger.info("Starting up the MHIST classification service...")

    try:
        # load model
        model_handler = ModelHandler(ServiceInfo())
        await model_handler.load_model()
        await model_handler.load_labels()
        logger.info("Model and labels loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or labels: {e}")
        raise

    yield
    # shutdown
    logger.info("Shutting down the MHIST classification service...")

app = FastAPI(title="MHIST Classification Service",
              version="1.0.0",
              lifespan=lifespan)
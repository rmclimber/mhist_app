from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import httpx
import logging
import uuid
from typing import Dict, Any

from .inference_gateway import InferenceGateway
from ..service_info import ServiceInfo
from .gateway_router import GatewayRouter


### PRELIMINARIES FOR THE APP
# set up client and service info
http_client = httpx.AsyncClient(timeout=30.0)
service_info = ServiceInfo()

# make sure cleanup works properly
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await http_client.aclose()

### SET UP THE APP
# create the top-level FastAPI app for uvicorn to detect and run
app = FastAPI(
    title="MHIST Gateway Service",
    description="A FastAPI service for MHIST image classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# set up the prediction pipeline
inference_gateway = InferenceGateway(
    client=http_client,
    logger=logging.getLogger("inference_gateway"),
    service_info=service_info
)

# set up router
router = GatewayRouter(
    logger=logging.getLogger("gateway_router"),
    inference_gateway=inference_gateway
).register_routes()
app.include_router(router)


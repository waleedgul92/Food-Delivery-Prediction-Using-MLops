from fastapi import APIRouter, HTTPException
from .schema import PredictionInput, FeedbackInput  # Change from ..app.schema
from .model_handler import make_prediction, get_model_info, get_health_status, retrain_with_feedback, prune_model_registry
from .feedback import save_feedback_to_csv, get_feedback_stats
import asyncio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
def root():
    return {
        "status": "API running",
        "message": "Delivery Time Predictor API"
    }

@router.get("/health")
def health():
    return get_health_status()

@router.get("/model-info")
def model_info():
    return get_model_info()

@router.post("/predict")
async def predict(data: PredictionInput):
    return await make_prediction(data)

@router.post("/feedback")
async def feedback(data: FeedbackInput):
    return await save_feedback(data)

@router.get("/feedback-stats")
def stats():
    return get_feedback_stats()

@router.post("/retrain")
async def retrain():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, retrain_with_feedback, logger)
    prune_model_registry("DeliveryTimePredictor", logger)
    return result

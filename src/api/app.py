from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from src.core.kaagapai_agent import KaagapAIAgent

app = FastAPI(
    title="KaagapAI Demo API",
    description="Autonomous AI Agent for Financial Decision Making",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

kaagapai_agent = None
logger = logging.getLogger(__name__)

class CustomerInput(BaseModel):
    customer_id: str
    age: int
    gender: str
    monthly_income: float
    current_savings: float
    goal_amount: float
    goal_timeline_months: int
    risk_tolerance: str
    @validator('age')
    def validate_age(cls, v):
        if not 18 <= v <= 65:
            raise ValueError('Age must be between 18 and 65')
        return v

class PredictionResponse(BaseModel):
    success_probability: float
    confidence: float
    processing_time_ms: float
    recommendations: List[Dict]
    explanations: Dict
    timestamp: str

@app.on_event("startup")
async def startup_event():
    global kaagapai_agent
    try:
        logger.info("Initializing KaagapAI system...")
        config = {"model_path": "./models/trained"}
        kaagapai_agent = KaagapAIAgent(config)
        logger.info("KaagapAI system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_goal_achievement(customer_input: CustomerInput, background_tasks: BackgroundTasks):
    if kaagapai_agent is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    try:
        start_time = asyncio.get_event_loop().time()
        customer_context = customer_input.dict()
        result = await kaagapai_agent.autonomous_decision_making(customer_context)
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        background_tasks.add_task(log_prediction_request, customer_input.customer_id, result, processing_time)
        response = PredictionResponse(
            success_probability=result['predictions']['goal_achievement_probability']['success_probability'],
            confidence=result['confidence_score'],
            processing_time_ms=processing_time,
            recommendations=result['recommendations'].get('primary_recommendations', []),
            explanations=result['explanations'],
            timestamp=result['timestamp']
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "kaagapai_status": "ready" if kaagapai_agent else "not_ready"
    }

async def log_prediction_request(customer_id: str, result: Dict, processing_time: float):
    try:
        success_prob = result['predictions']['goal_achievement_probability']['success_probability']
        logger.info(f"Prediction - Customer: {customer_id}, Time: {processing_time:.2f}ms, Success: {success_prob:.2%}")
    except Exception as e:
        logger.error(f"Logging error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

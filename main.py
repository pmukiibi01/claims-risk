"""
Claims-Based Risk Adjustment and Cost Drivers System
Main FastAPI application for estimating member risk and explaining spend variation.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import logging

from models.risk_adjustment import RiskAdjustmentModel
from models.cost_drivers import CostDriverAnalysis
from data.sample_data import generate_sample_data
from utils.data_processing import ClaimsProcessor
from utils.hcc_mapping import HCCMapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Claims-Based Risk Adjustment System",
    description="Estimate member risk and explain spend variation using HCC adaptation, GLM/GAM, and two-part models",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize models
risk_model = RiskAdjustmentModel()
cost_driver_model = CostDriverAnalysis()
claims_processor = ClaimsProcessor()
hcc_mapper = HCCMapper()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload-claims")
async def upload_claims(file: UploadFile = File(...)):
    """Upload and process claims data"""
    try:
        # Read the uploaded file
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Process the claims data
        processed_data = claims_processor.process_claims(df)
        
        # Store processed data
        os.makedirs("data", exist_ok=True)
        processed_data.to_csv("data/processed_claims.csv", index=False)
        
        return {
            "message": "Claims data uploaded and processed successfully",
            "records": len(processed_data),
            "columns": list(processed_data.columns)
        }
        
    except Exception as e:
        logger.error(f"Error processing claims data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing claims: {str(e)}")

@app.post("/generate-sample-data")
async def generate_sample():
    """Generate sample claims data for testing"""
    try:
        sample_data = generate_sample_data(n_members=1000, n_claims=5000)
        
        # Save sample data
        os.makedirs("data", exist_ok=True)
        sample_data.to_csv("data/sample_claims.csv", index=False)
        
        return {
            "message": "Sample data generated successfully",
            "records": len(sample_data),
            "file_path": "data/sample_claims.csv"
        }
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

@app.get("/download-sample")
async def download_sample():
    """Download sample data template"""
    try:
        if not os.path.exists("data/sample_claims.csv"):
            # Generate sample data if it doesn't exist
            sample_data = generate_sample_data(n_members=1000, n_claims=5000)
            os.makedirs("data", exist_ok=True)
            sample_data.to_csv("data/sample_claims.csv", index=False)
        
        return FileResponse(
            "data/sample_claims.csv",
            media_type="text/csv",
            filename="sample_claims.csv"
        )
        
    except Exception as e:
        logger.error(f"Error downloading sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading sample data: {str(e)}")

@app.post("/train-risk-model")
async def train_risk_model():
    """Train the risk adjustment model"""
    try:
        # Load processed claims data
        if not os.path.exists("data/processed_claims.csv"):
            raise HTTPException(status_code=404, detail="No processed claims data found. Please upload data first.")
        
        df = pd.read_csv("data/processed_claims.csv")
        
        # Train the risk adjustment model
        model_results = risk_model.train(df)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        risk_model.save_model("models/risk_adjustment_model.pkl")
        
        return {
            "message": "Risk adjustment model trained successfully",
            "model_performance": model_results,
            "model_saved": "models/risk_adjustment_model.pkl"
        }
        
    except Exception as e:
        logger.error(f"Error training risk model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training risk model: {str(e)}")

@app.post("/analyze-cost-drivers")
async def analyze_cost_drivers():
    """Analyze cost drivers and generate insights"""
    try:
        # Load processed claims data
        if not os.path.exists("data/processed_claims.csv"):
            raise HTTPException(status_code=404, detail="No processed claims data found. Please upload data first.")
        
        df = pd.read_csv("data/processed_claims.csv")
        
        # Analyze cost drivers
        cost_analysis = cost_driver_model.analyze(df)
        
        return {
            "message": "Cost driver analysis completed",
            "analysis_results": cost_analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing cost drivers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing cost drivers: {str(e)}")

@app.get("/predict-risk/{member_id}")
async def predict_member_risk(member_id: str):
    """Predict risk score for a specific member"""
    try:
        # Load model if it exists
        if not os.path.exists("models/risk_adjustment_model.pkl"):
            raise HTTPException(status_code=404, detail="Risk model not found. Please train the model first.")
        
        risk_model.load_model("models/risk_adjustment_model.pkl")
        
        # Load member data
        if not os.path.exists("data/processed_claims.csv"):
            raise HTTPException(status_code=404, detail="No processed claims data found.")
        
        df = pd.read_csv("data/processed_claims.csv")
        member_data = df[df['member_id'] == member_id]
        
        if member_data.empty:
            raise HTTPException(status_code=404, detail=f"Member {member_id} not found.")
        
        # Predict risk score
        risk_score = risk_model.predict_member_risk(member_data)
        
        return {
            "member_id": member_id,
            "risk_score": float(risk_score),
            "risk_level": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
        }
        
    except Exception as e:
        logger.error(f"Error predicting member risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting member risk: {str(e)}")

@app.get("/model-performance")
async def get_model_performance():
    """Get current model performance metrics"""
    try:
        if not os.path.exists("models/risk_adjustment_model.pkl"):
            raise HTTPException(status_code=404, detail="Risk model not found. Please train the model first.")
        
        performance = risk_model.get_performance_metrics()
        return performance
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
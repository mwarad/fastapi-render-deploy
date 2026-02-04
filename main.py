"""
Fraud Detection API - Production Ready for Render.com
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Fraud Detection API",
    description="Ensemble fraud detection using XGBoost + Random Forest Pipelines",
    version="1.0.0"
)

# ============================================================
# LOAD PIPELINES & METADATA
# ============================================================
MODEL_DIR = Path("models")

try:
    # Load complete pipelines (preprocessor + model)
    xgb_pipeline = joblib.load(MODEL_DIR / "xgboost_pipeline.pkl")
    rf_pipeline = joblib.load(MODEL_DIR / "random_forest_pipeline.pkl")
    
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    
    with open(MODEL_DIR / "feature_stats.json") as f:
        feature_stats = json.load(f)
    
    # Extract feature info
    raw_features = metadata.get("feature_columns", metadata.get("features", []))
    FEATURE_COLS = ['category' if f == 'category_encoded' else f for f in raw_features]
    CATEGORY_CLASSES = metadata.get("category_classes", [])
    NUMERIC_COLS = [f for f in FEATURE_COLS if f != "category"]
    
    if not FEATURE_COLS:
        FEATURE_COLS = ['category', 'amount', 'age_at_transaction', 'days_until_card_expires',
                        'loc_delta', 'trans_volume_mavg', 'trans_volume_mstd', 'trans_freq', 'loc_delta_mavg']
        NUMERIC_COLS = FEATURE_COLS[1:]
    
    print(f"Models loaded! Features: {FEATURE_COLS}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class TransactionInput(BaseModel):
    """
    Pydantic model for transaction input with validation.
    
    Demonstrates:
    - Field() with constraints (ge, le, description)
    - field_validator for custom validation
    - model_validator for cross-field validation
    """
    
    # Field with description and constraints
    category: str = Field(
        ...,  # Required field
        description="Transaction category (e.g., Grocery, Electronics)",
        examples=["Grocery", "Electronics", "Clothing"]
    )
    
    amount: float = Field(
        ...,
        ge=0.01,  # Greater than or equal to 0.01
        le=50000,  # Less than or equal to 50000
        description="Transaction amount in dollars"
    )
    
    age_at_transaction: float = Field(
        ...,
        ge=18,
        le=100,
        description="Customer age at time of transaction"
    )
    
    days_until_card_expires: float = Field(
        ...,
        ge=0,
        le=3650,  # Max 10 years
        description="Days until card expiration"
    )
    
    loc_delta: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Haversine distance from previous transaction (normalized 0-1)"
    )
    
    trans_volume_mavg: float = Field(
        ...,
        ge=0,
        description="4-hour moving average of transaction amounts"
    )
    
    trans_volume_mstd: float = Field(
        default=0.0,
        ge=0,
        description="4-hour standard deviation of transaction amounts"
    )
    
    trans_freq: float = Field(
        default=1.0,
        ge=1,
        description="Transaction frequency in 4-hour window"
    )
    
    loc_delta_mavg: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="4-hour moving average of location changes"
    )
    
    # Custom validator for category
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        """Validate category is not empty and properly formatted."""
        if not v or not v.strip():
            raise ValueError('Category cannot be empty')
        return v.strip()
    
    # Cross-field validation
    @model_validator(mode='after')
    def validate_transaction(self):
        """Cross-field validation: high amount + high location delta = suspicious."""
        if self.amount > 5000 and self.loc_delta > 0.5:
            # Just a warning - we still process but log it
            print(f"WARNING: High-risk pattern detected: amount=${self.amount}, loc_delta={self.loc_delta}")
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "Grocery",
                "amount": 150.50,
                "age_at_transaction": 35.5,
                "days_until_card_expires": 365.0,
                "loc_delta": 0.05,
                "trans_volume_mavg": 120.0,
                "trans_volume_mstd": 45.0,
                "trans_freq": 3.0,
                "loc_delta_mavg": 0.03
            }
        }


class PredictionResponse(BaseModel):
    transaction_id: str
    timestamp: str
    xgboost_probability: float
    random_forest_probability: float
    ensemble_probability: float
    ensemble_prediction: int
    verdict: str
    drift_warnings: List[str]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def check_drift(features: dict) -> list:
    """Check if numeric features are outside training distribution."""
    warnings = []
    for feature, value in features.items():
        if feature in feature_stats:
            stats = feature_stats[feature]
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            if std > 0 and abs(value - mean) > 3 * std:
                warnings.append(f"{feature}: {value:.2f} is >3 std from mean ({mean:.2f})")
    return warnings


# Global counter
request_counter = 0

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def root():
    """Health check."""
    return {
        "status": "healthy",
        "message": "Fraud Detection API is running",
        "models": ["XGBoost", "Random Forest"],
        "categories": CATEGORY_CLASSES
    }


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": True,
        "total_predictions": request_counter
    }


@app.get("/model-info")
def model_info():
    """Get model metadata."""
    return {
        "feature_columns": FEATURE_COLS,
        "categorical_columns": metadata.get("categorical_columns", ["category"]),
        "numeric_columns": metadata.get("numeric_columns", FEATURE_COLS[1:]),
        "category_classes": CATEGORY_CLASSES,
        "training_samples": metadata.get("training_samples", 145457),
        "test_samples": metadata.get("test_samples", 36365),
        "pipeline_steps": metadata.get("pipeline_steps", ["OrdinalEncoder", "Classifier"]),
        "metrics": metadata.get("metrics", {})
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    """Predict fraud probability for a transaction."""
    global request_counter
    request_counter += 1
    request_id = f"txn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_counter:06d}"
    
    try:
        # Create DataFrame with raw features
        input_data = {
            'category': [transaction.category],
            'amount': [transaction.amount],
            'age_at_transaction': [transaction.age_at_transaction],
            'days_until_card_expires': [transaction.days_until_card_expires],
            'loc_delta': [transaction.loc_delta],
            'trans_volume_mavg': [transaction.trans_volume_mavg],
            'trans_volume_mstd': [transaction.trans_volume_mstd],
            'trans_freq': [transaction.trans_freq],
            'loc_delta_mavg': [transaction.loc_delta_mavg]
        }
        X = pd.DataFrame(input_data)[FEATURE_COLS]
        
        # Check for drift
        numeric_features = {k: v for k, v in transaction.model_dump().items() if k != 'category'}
        drift_warnings = check_drift(numeric_features)
        
        # Get predictions from pipelines
        xgb_prob = float(xgb_pipeline.predict_proba(X)[0, 1])
        rf_prob = float(rf_pipeline.predict_proba(X)[0, 1])
        ensemble_prob = (xgb_prob + rf_prob) / 2
        ensemble_pred = int(ensemble_prob >= 0.5)
        
        # Determine verdict
        if ensemble_prob >= 0.7:
            verdict = "HIGH RISK - Likely Fraud"
        elif ensemble_prob >= 0.5:
            verdict = "SUSPICIOUS - Review Recommended"
        elif ensemble_prob >= 0.3:
            verdict = "LOW RISK - Monitor"
        else:
            verdict = "LEGITIMATE - Approved"
        
        return PredictionResponse(
            transaction_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            xgboost_probability=round(xgb_prob, 4),
            random_forest_probability=round(rf_prob, 4),
            ensemble_probability=round(ensemble_prob, 4),
            ensemble_prediction=ensemble_pred,
            verdict=verdict,
            drift_warnings=drift_warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

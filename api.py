"""
================================================================================
  ParkWatch AI — FastAPI Backend
================================================================================
  Exposes the LightGBM model as a REST API.

  SETUP
  -----
  pip install fastapi uvicorn pydantic

  RUN
  ---
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

  ENDPOINTS
  ---------
  POST /predict          → single violation prediction
  POST /predict/batch    → bulk predictions from a list
  GET  /hotspots         → top-50 hotspot zones (for map render)
  GET  /health           → API + model status check
  GET  /metadata         → model metadata (features, thresholds, classes)
================================================================================
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing   import Optional, List

from fastapi             import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses   import JSONResponse
from pydantic            import BaseModel, Field

# Import inference functions from your pipeline file
from parking_intelligence_complete import (
    predict_congestion_impact,
    predict_batch,
    _load_inference_artifacts,
    ARTIFACTS_DIR,
)

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ParkWatch AI — Parking Violation Intelligence API",
    description = "Predicts congestion impact of parking violations using LightGBM",
    version     = "1.0.0",
)

# Allow any origin so the HTML file can call the API whether it's
# opened locally (file://) or served from another domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST", "OPTIONS"],
    allow_headers  = ["*"],
)

# Pre-load the model at startup so the first request isn't slow
@app.on_event("startup")
def load_model():
    print("[INFO] Loading ML model artifacts...")
    _load_inference_artifacts()
    print("[OK] Model ready.")


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────
class ViolationInput(BaseModel):
    latitude:             float  = Field(...,  example=12.9716,  description="GPS latitude of the violation")
    longitude:            float  = Field(...,  example=77.5946,  description="GPS longitude of the violation")
    created_datetime:     str    = Field(...,  example="2024-03-15T08:30:00+00:00", description="ISO8601 timestamp")
    violation_type:       str    = Field(...,  example="PARKING IN A MAIN ROAD",    description="Raw violation type string")
    vehicle_type:         str    = Field(...,  example="LGV",    description="Vehicle category")
    police_station:       str    = Field(...,  example="Upparpet",description="Police station name")
    junction_name:        str    = Field("",   example="BTP051 - Safina Plaza Junction", description="Junction name (optional)")
    grid_violation_count: Optional[int] = Field(None, example=420, description="Pre-computed cell density; omit to use median")


class BatchInput(BaseModel):
    violations: List[ViolationInput] = Field(..., description="List of violation records to score")


class PredictionResponse(BaseModel):
    high_congestion_impact: int
    congestion_probability: float
    risk_level:             str
    key_risk_factors:       List[str]


class BatchPredictionResponse(BaseModel):
    count:   int
    results: List[PredictionResponse]


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check that the API and model are running."""
    try:
        _load_inference_artifacts()
        return {
            "status":     "ok",
            "model":      "LightGBM",
            "model_file": str((ARTIFACTS_DIR / "best_model.pkl").resolve()),
            "api":        "ParkWatch AI v1.0",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {e}")


@app.get("/metadata", tags=["System"])
def get_metadata():
    """Return model metadata: feature list, thresholds, label classes."""
    meta_path = ARTIFACTS_DIR / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found")
    with open(meta_path) as f:
        return json.load(f)


@app.get("/hotspots", tags=["Data"])
def get_hotspots(limit: int = 50):
    """
    Return the top predicted hotspot grid cells.
    Used by the frontend to render congestion circles on the map.
    """
    hotspot_path = ARTIFACTS_DIR / "hotspot_report.csv"
    if not hotspot_path.exists():
        raise HTTPException(status_code=404, detail="hotspot_report.csv not found. Run the training pipeline first.")
    df = pd.read_csv(hotspot_path).head(limit)
    return {
        "count":    len(df),
        "hotspots": df.to_dict(orient="records"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: ViolationInput):
    """
    Predict the congestion impact of a single parking violation.

    Returns:
    - high_congestion_impact: 0 or 1
    - congestion_probability: float between 0 and 1
    - risk_level: LOW | MEDIUM | HIGH | CRITICAL
    - key_risk_factors: list of human-readable factors that drove the score
    """
    try:
        result = predict_congestion_impact(
            latitude             = data.latitude,
            longitude            = data.longitude,
            created_datetime     = data.created_datetime,
            violation_type       = data.violation_type,
            vehicle_type         = data.vehicle_type,
            police_station       = data.police_station,
            junction_name        = data.junction_name,
            grid_violation_count = data.grid_violation_count,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch_endpoint(data: BatchInput):
    """
    Predict congestion impact for a batch of violations.
    Accepts a list of up to 1000 records.
    """
    if len(data.violations) > 1000:
        raise HTTPException(status_code=400, detail="Batch limit is 1000 records per request.")

    try:
        df = pd.DataFrame([v.dict() for v in data.violations])
        scored = predict_batch(df)
        results = []
        for _, row in scored.iterrows():
            results.append({
                "high_congestion_impact": int(row["predicted_impact"]),
                "congestion_probability": float(row["congestion_probability"]),
                "risk_level":             row["risk_level"],
                "key_risk_factors":       [],   # factors omitted in batch for speed
            })
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
from model import RainNowcaster, prepare_era5_data

app = FastAPI(
    title="BrollyCast☂️: Rain Nowcasting API",
    description="A production-ready API for short-term precipitation forecasting.",
    version="1.0.0"
)

# Global variables for the model and scaler
MODEL_PATH = "rain_nowcaster.pth"
SCALER_PATH = "scaler.joblib"
model = None
scaler = None

# Define input data schema using Pydantic
class HourlyWeatherReading(BaseModel):
    temperature: float = Field(..., description="Temperature at 2 meters in Kelvin or Celsius")
    dewpoint: float = Field(..., description="Dewpoint temperature at 2 meters")
    pressure: float = Field(..., description="Mean sea level pressure in Pascals")
    cloud_cover: float = Field(..., description="Cloud cover fraction between 0.0 and 1.0", ge=0.0, le=1.0)

class NowcastRequest(BaseModel):
    # Input weather data from the past 3 hours
    past_readings: List[HourlyWeatherReading] = Field(..., min_items=3, max_items=3)

# Define the output data schema
class NowcastResponse(BaseModel):
    rain_predicted: bool
    probability: float

# Load model weights on startup
@app.on_event("startup")
def load_assets():
    global model, scaler
    model = RainNowcaster(input_dim=12)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # Load the scaler
    scaler = joblib.load(SCALER_PATH)
    print("Successfully loaded model and scaler.")

@app.post("/predict", response_model=NowcastResponse)
def predict_rain(request: NowcastRequest):
    # Extract raw features
    raw_features = []
    for reading in request.past_readings:
        raw_features.append([
            reading.temperature,
            reading.dewpoint,
            reading.pressure,
            reading.cloud_cover
        ])
        
    # Scale the features
    scaled_features = scaler.transform(raw_features)
    
    # Flatten into the 12-dimensional vector expected by the model
    input_array = scaled_features.flatten().reshape(1, -1)
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    
    # Run Inference
    with torch.no_grad():
        raw_logit = model(input_tensor)
        probability = torch.sigmoid(raw_logit).item()
    
    return NowcastResponse(
        rain_predicted=probability >= 0.5,
        probability=round(probability, 4)
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}



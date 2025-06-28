from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Define the input schema
class SizeRequest(BaseModel):
    height: float
    weight: float
    chest: float
    waist: float
    hips: float
    brand: str

# Load the model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    model = None

app = FastAPI(title="Clothing Size Comparator API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Clothing Size Comparator API!"}

@app.post("/predict")
def predict(req: SizeRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    features = np.array([[req.height, req.weight, req.chest, req.waist, req.hips]])
    prediction = model.predict(features)[0]

    return {
        "brand": req.brand,
        "recommended_size": prediction
    }

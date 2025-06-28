from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import json
from typing import Optional, Dict, Any
import os

# Import our NLP processor (make sure nlp_processor.py is in the same directory)
try:
    from nlp_processor import NaturalLanguageProcessor
except ImportError:
    print("Warning: nlp_processor.py not found. Natural language processing will be limited.")
    NaturalLanguageProcessor = None

app = FastAPI(title="Sonic SizeBot - Enhanced Clothing Size Predictor")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schemas
class SizeRequest(BaseModel):
    height: float
    weight: float
    chest: float
    waist: float
    hips: float
    brand: str

class NaturalLanguageRequest(BaseModel):
    message: str

# Initialize NLP processor
nlp_processor = NaturalLanguageProcessor() if NaturalLanguageProcessor else None

# Load the enhanced model and encoders
try:
    model = joblib.load("enhanced_model.pkl")
    brand_encoder = joblib.load("brand_encoder.pkl")
    with open("brand_mapping.json", "r") as f:
        brand_mapping = json.load(f)
    print("Enhanced model loaded successfully!")
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    print("Please run train_model.py first to create the model files.")
    model = None
    brand_encoder = None
    brand_mapping = {}

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main page"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {"message": "Sonic SizeBot API is running! Please add static/index.html for the web interface."}

@app.get("/brands")
async def get_supported_brands():
    """Get list of supported brands"""
    if brand_mapping:
        return {"brands": list(brand_mapping.keys())}
    else:
        return {"brands": ["Zara", "H&M", "Uniqlo", "Gap", "Old Navy", "Shein"]}

@app.post("/predict")
def predict_size(req: SizeRequest):
    """Predict clothing size using structured input"""
    if model is None or brand_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    # Validate brand
    brand_name = req.brand.title()
    if brand_name not in brand_mapping:
        # Try to find closest match
        available_brands = list(brand_mapping.keys())
        brand_name = available_brands[0]  # Default to first available brand
    
    try:
        # Encode brand
        brand_encoded = brand_encoder.transform([brand_name])[0]
        
        # Prepare features
        features = np.array([[req.height, req.weight, req.chest, req.waist, req.hips, brand_encoded]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get size probabilities
        size_probabilities = {}
        for size, prob in zip(model.classes_, probabilities):
            size_probabilities[size] = round(float(prob), 3)
        
        return {
            "brand": brand_name,
            "recommended_size": prediction,
            "confidence": round(float(max(probabilities)), 3),
            "size_probabilities": size_probabilities,
            "measurements": {
                "height": req.height,
                "weight": req.weight,
                "chest": req.chest,
                "waist": req.waist,
                "hips": req.hips
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-natural")
def predict_size_natural(req: NaturalLanguageRequest):
    """Predict clothing size using natural language input"""
    if model is None or brand_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    if nlp_processor is None:
        raise HTTPException(status_code=500, detail="Natural language processor not available.")
    
    try:
        # Parse natural language input
        parsed_data = nlp_processor.parse_input(req.message)
        
        # Validate input
        is_valid, errors = nlp_processor.validate_input(parsed_data)
        if not is_valid:
            return {
                "success": False,
                "errors": errors,
                "suggestion": "Please provide your height and weight. For example: 'I'm 5'8 and 150 lbs, looking for clothes at Zara'"
            }
        
        # Validate brand
        brand_name = parsed_data['brand']
        if brand_name not in brand_mapping:
            # Try to find closest match or use default
            available_brands = list(brand_mapping.keys())
            brand_name = available_brands[0]
        
        # Encode brand
        brand_encoded = brand_encoder.transform([brand_name])[0]
        
        # Prepare features
        features = np.array([[
            parsed_data['height'],
            parsed_data['weight'],
            parsed_data['chest'],
            parsed_data['waist'],
            parsed_data['hips'],
            brand_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get size probabilities
        size_probabilities = {}
        for size, prob in zip(model.classes_, probabilities):
            size_probabilities[size] = round(float(prob), 3)
        
        return {
            "success": True,
            "brand": brand_name,
            "recommended_size": prediction,
            "confidence": round(float(max(probabilities)), 3),
            "size_probabilities": size_probabilities,
            "parsed_measurements": {
                "height": f"{parsed_data['height']} cm",
                "weight": f"{parsed_data['weight']} kg",
                "chest": f"{parsed_data['chest']} cm",
                "waist": f"{parsed_data['waist']} cm",
                "hips": f"{parsed_data['hips']} cm"
            },
            "original_input": parsed_data['original_input']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "nlp_available": nlp_processor is not None,
        "supported_brands": list(brand_mapping.keys()) if brand_mapping else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
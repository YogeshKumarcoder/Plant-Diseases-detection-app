from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI()

# CORS — React frontend allow karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model load karo
model = tf.keras.models.load_model('plant_disease_model.keras')

with open('class_names.json') as f:
    class_names = json.load(f)
with open('treatment_map.json') as f:
    treatment_map = json.load(f)

@app.get("/")
def root():
    return {"status": "CropGuard AI Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Image read karo
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions) * 100)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Parse
    parts = predicted_class.replace('___', '|').split('|')
    crop = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else predicted_class
    treatment = treatment_map.get(predicted_class, 'Treatment not available')
    is_healthy = 'healthy' in predicted_class.lower()
    
    # Top 3
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3 = [
        {
            "class": class_names[i].replace('___', ' → ').replace('_', ' '),
            "confidence": round(float(predictions[0][i] * 100), 1)
        }
        for i in top3_idx
    ]
    
    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 1),
        "is_healthy": is_healthy,
        "treatment": treatment,
        "top3": top3
    }
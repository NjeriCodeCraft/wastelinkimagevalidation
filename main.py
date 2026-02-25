import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

# 1. Handle the Keras 3 Lambda issue explicitly
@keras.utils.register_keras_serializable()
def preprocess_input(x):
    return x

# Global variable for the model
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(
            'wastelink_v1_model.keras',
            custom_objects={'preprocess_input': preprocess_input},
            compile=False
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server"}
    
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
    
    preds = model(img_array, training=False)
    # Your logic for labels goes here...
    return {"status": "success", "raw_prediction": preds.numpy().tolist()}

import os
import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File

# We must set this before importing tensorflow/keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras  # This defines 'keras' for your decorator

app = FastAPI()

# 1. Handle the Keras 3 Lambda issue explicitly
# This now works because 'keras' was imported above
@keras.utils.register_keras_serializable()
def preprocess_input(x):
    return x

# Global variable for the model
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # This tells Keras 3 how to handle the old 'batch_shape' key
        from keras.src.layers import InputLayer
        
        # We manually patch the InputLayer to accept the old argument
        original_init = InputLayer.__init__
        def patched_init(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['shape'] = kwargs.pop('batch_shape')
            original_init(self, *args, **kwargs)
        InputLayer.__init__ = patched_init

        model = keras.models.load_model(
            'wastelink_v1_model.keras',
            custom_objects={'preprocess_input': preprocess_input},
            compile=False
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")

@app.get("/")
async def root():
    return {"message": "WasteLink API is running", "model_loaded": model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server"}
    
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    
    # Convert image to array using keras utilities
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run prediction
    preds = model.predict(img_array)
    
    return {
        "status": "success", 
        "raw_prediction": preds.tolist()
    }

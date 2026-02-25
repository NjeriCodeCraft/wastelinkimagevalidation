import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File

# Force Legacy Keras environment variables
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tf_keras as keras

# --- THE MONKEY PATCH FOR 'batch_shape' ---
# This intercepts the error and renames the variable so the model loads
from tf_keras.layers import InputLayer
original_init = InputLayer.__init__
def patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['shape'] = kwargs.pop('batch_shape')
    original_init(self, *args, **kwargs)
InputLayer.__init__ = patched_init
# ------------------------------------------

app = FastAPI()

@keras.utils.register_keras_serializable()
def preprocess_input(x):
    return x

model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = keras.models.load_model(
            'wastelink_v1_model.keras',
            custom_objects={'preprocess_input': preprocess_input},
            compile=False
        )
        print("✅ SUCCESS: Model is loaded and ready!")
    except Exception as e:
        print(f"❌ STILL FAILING: {e}")

@app.get("/")
async def root():
    return {"status": "Online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    return {"predictions": preds.tolist()}

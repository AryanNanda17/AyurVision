#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = FastAPI()

MODEL = tf.keras.models.load_model("../Saved_Models/1") 

CLASS_NAMES = ['Aloevera','Amla', 'Amruta_Balli','Arali','Ashoka','Ashwagandha','Avacado','Bamboo','Basale','Betel','Betel_Nut',
 'Brahmi','Castor','Curry_Leaf','Doddapatre','Ekka','Ganike','Gauva','Geranium','Henna','Hibiscus','Honge','Insulin','Jasmine',
 'Lemon','Lemon_grass','Mango','Mint','Nagadali','Neem','Nithyapushpa','Nooni','Pappaya','Pepper','Pomegranate','Raktachandini','Rose',
 'Sapota','Tulasi','Wood_sorel']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return "Hello! I am alive"


@app.post("/predict")
async def predict( file: UploadFile = File(...) ):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
 
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
from array import array
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from fastapi import FastAPI, Body
from pydantic import BaseModel
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from typing import List
from tensorflow.keras.models import load_model
from fastapi.encoders import jsonable_encoder

app = FastAPI()

class Input(BaseModel):
    SepalLength: float
    SepalWidth: float
    PetalLength: float
    PetalWidth: float

class item(BaseModel):
    items: List[Input]

@app.get("/")
async def root():
    return {"message": "Such ML much wow"}

def get_model():
    return load_model("mymodel.h5")

@app.post("/predictions")
async def prediction(input: dict = Body(...)):

    body = input['items']

    X = np.array([list(input.values()) for input in body])

    model = get_model()

    y_pred = np.argmax(model.predict(X), axis=1)

    encoder = pickle.load(open('enc.pkl', 'rb'))

    labels1 = list(encoder.inverse_transform(y_pred)) 

    return labels1

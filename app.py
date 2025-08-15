# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, make_prediction

app = FastAPI()
model = load_model("models/decision_tree.pkl")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisData):
    prediction = make_prediction(model, [
        data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    ])
    return {"prediction": prediction.tolist()}

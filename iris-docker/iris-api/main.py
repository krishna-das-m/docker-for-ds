from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Iris Classifier API", version="1.0.0")

model = joblib.load("model.joblib") # loaded once at startup

labels = ["setosa", "versicolor", "virginica"]

class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, lt=20, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, lt=20, description="Sepal width in cm")
    petal_length: float = Field(gt=0, lt=20, description="Petal length in cm")
    petal_width: float = Field(gt=0, lt=20, description="Petal width in cm")

class PredictionResponse(BaseModel):
    species:    str
    species_id: int
    confidence: float

@app.post("/predict", response_model=PredictionResponse,
        summary="Predict Iris species",tags=["prediction"])
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width,
    ]])
    prediction   = model.predict(X)[0]
    probs        = model.predict_proba(X)[0]
    confidence   = float(probs[prediction])

    return PredictionResponse(
        species    = labels[prediction],
        species_id = int(prediction),
        confidence = round(confidence, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "iris-classifier"}
# from fastapi import FastAPI

# app = FastAPI(title="My API", version="1.0.0")

# @app.get("/")
# def root():
#     return {"message": "Hello, World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str | None = None):
#     # item_id is a path param (int, auto-validated)
#     # q is an optional query param: /items/5?q=search
#     return {"item_id": item_id, "q": q}
"""
SAMPLE CODE TO INTERFACE A ML MODEL
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import joblib
import numpy as np


class PredictRequest(BaseModel):
    """
    Model to validate API body
    """
    data: list  # list of feature vectors or a single feature vector


app = FastAPI(title="Iris RF Demo")

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# For demo, load a model saved earlier if present; otherwise create a dummy model
try:
    model = joblib.load("artifacts/rf_iris.joblib")
except FileNotFoundError:
    # fallback: a simple scikit-learn model trained quickly
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.utils import Bunch
    # pylint: disable=no-member
    iris: Bunch = load_iris(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_train, y_train)


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoind handler for interfacing model
    """
    data = np.array(req.data)
    # ensure 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    preds = model.predict(data).tolist()
    return {"predictions": preds}

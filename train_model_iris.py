# Basic MLflow experiment: train an sklearn model on Iris and log params/metrics/artifacts
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os

# Prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Configure MLflow tracking URI (file store in workspace)
mlflow.set_tracking_uri("file:///tmp/mlruns")  # safe local file store in Colab
mlflow.set_experiment("day3_mlflow_demo")

with mlflow.start_run(run_name="rf_demo"):
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", float(acc))

    # save a model artifact (joblib) and log it
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/rf_iris.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    # log sklearn model with MLflow's model registry format (local model save)
    mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

    print("Test accuracy:", acc)
    print("Classification report:\n", classification_report(y_test, preds))

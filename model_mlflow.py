import mlflow
import mlflow.sklearn
import pandas as pd 
import joblib

# Load the pre-trained model
model = joblib.load("vodafone_churn_model.joblib")

accuracy = 0.99

# Start MLflow tracking
with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("source", "Pre-trained")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model artifact
    mlflow.sklearn.log_model(model, "logistic_model")

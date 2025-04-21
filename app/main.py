from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
import os

# Ajusta o tracking URI para funcionar dentro do container
os.environ['MLFLOW_TRACKING_URI'] = 'file:/app/mlruns'

app = FastAPI()

class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# Carrega o último modelo treinado via caminho direto (evita erro de path absoluto no Docker)
client = MlflowClient()
experiment = client.get_experiment_by_name("Churn Prediction")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
latest_run = runs[0]
model_path = f"file:/app/mlruns/{experiment.experiment_id}/{latest_run.info.run_id}/artifacts/model"
model = mlflow.pyfunc.load_model(model_path)

@app.get("/")
def read_root():
    return {"message": "API de previsão de churn está ativa!"}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
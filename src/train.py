# Treinamento e log do modelo com MLflow
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DATA_PATH = "data/processed/processed_churn.csv"

def train_model():
    # Carregar os dados processados
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separar features e target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Iniciar experimento no MLflow
    mlflow.set_experiment("Churn Prediction")

    best_model = None
    best_f1 = 0
    best_run_id = None

    # Treinar e logar Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression") as run:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc_lr)
        mlflow.log_metric("f1_score", f1_lr)
        mlflow.sklearn.log_model(lr, "model")

        print(f"LogisticRegression - Accuracy: {acc_lr:.4f}, F1 Score: {f1_lr:.4f}")

        if f1_lr > best_f1:
            best_model = "LogisticRegression"
            best_f1 = f1_lr
            best_run_id = run.info.run_id

    # Treinar e logar Random Forest
    with mlflow.start_run(run_name="RandomForestClassifier") as run:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc_rf)
        mlflow.log_metric("f1_score", f1_rf)
        mlflow.sklearn.log_model(rf, "model")

        print(f"RandomForestClassifier - Accuracy: {acc_rf:.4f}, F1 Score: {f1_rf:.4f}")

        if f1_rf > best_f1:
            best_model = "RandomForestClassifier"
            best_f1 = f1_rf
            best_run_id = run.info.run_id

    # Registrar o melhor modelo no Model Registry
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        result = mlflow.register_model(model_uri, "ChurnModel")
        print(f"Melhor modelo registrado: {best_model} com F1 Score: {best_f1:.4f}")
        print(f"Model URI: {model_uri}")

if __name__ == "__main__":
    train_model()
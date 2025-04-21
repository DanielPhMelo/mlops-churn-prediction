# Realiza o pré-processamento dos dados
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/processed_churn.csv"

def preprocess_data():
    # Carregar os dados
    df = pd.read_csv(RAW_DATA_PATH)

    # Remover colunas desnecessárias
    df.drop(columns=["customerID"], inplace=True)

    # Substituir espaços em branco por NaN
    df.replace(" ", pd.NA, inplace=True)

    # Preencher valores ausentes
    df.fillna(method="ffill", inplace=True)

    # Codificar variáveis categóricas
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Salvar o arquivo pré-processado
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Dados processados salvos em {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()
# Gera uma amostra de dados atuais com alterações simuladas para teste de drift
import pandas as pd
import numpy as np
import os

SOURCE_PATH = "data/processed/processed_churn.csv"
OUTPUT_PATH = "data/current/current_churn_sample.csv"

# Gera amostra dos dados atuais
def generate_current_sample():
    df = pd.read_csv(SOURCE_PATH)
    sample = df.sample(n=100, random_state=42).copy()

    # Simular mudanças no padrão de uso (ex: aumento nas mensalidades)
    sample["MonthlyCharges"] = sample["MonthlyCharges"] * np.random.uniform(1.05, 1.25, size=sample.shape[0])
    sample["TotalCharges"] = sample["TotalCharges"] * np.random.uniform(1.05, 1.25, size=sample.shape[0])

    # Garante que os diretórios existam
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Salva o arquivo
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Amostra atual gerada em: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_current_sample()
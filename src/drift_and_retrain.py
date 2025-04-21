# Executa monitoramento e re-treinamento automático caso seja detectado drift
import pandas as pd
import os
import subprocess
from evidently.test_suite import TestSuite
from evidently.tests import DataDriftTestPreset

REFERENCE_DATA_PATH = "data/processed/processed_churn.csv"
CURRENT_DATA_PATH = "data/current/current_churn_sample.csv"

def detect_drift():
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)

    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(reference_data=reference, current_data=current)
    results = suite.as_dict()

    drift_detected = results['summary']['all_passed'] is False
    return drift_detected

def retrain_model():
    print("Drift detectado! Re-treinando modelo...")
    subprocess.run(["python", "src/train.py"])

if __name__ == "__main__":
    if detect_drift():
        retrain_model()
    else:
        print("Nenhum drift detectado. Nenhuma ação necessária.")
# Gera relatórios de drift com Evidently (versão 0.6.7)
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

REFERENCE_DATA_PATH = "data/processed/processed_churn.csv"
CURRENT_DATA_PATH = "data/current/current_churn_sample.csv"
REPORT_OUTPUT_PATH = "data/monitoring/drift_report.html"

def monitor_data():
    # Carrega os dados
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)

    # Gera o relatório
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Garante que o diretório exista
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)

    # Salva o relatório em HTML
    report.save_html(REPORT_OUTPUT_PATH)
    print(f"Relatório de monitoramento salvo em: {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    monitor_data()
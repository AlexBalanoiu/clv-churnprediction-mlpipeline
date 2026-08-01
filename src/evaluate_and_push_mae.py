"""Evaluează modelul și trimite metrici (MAE, RMSE, R2) către Prometheus
Pushgateway — rulat ca job periodic, nu ca parte a API-ului live.

DE CE E SEPARAT DE API: MAE-ul real necesită ground truth (cheltuiala
efectivă a clientului), disponibilă abia după ce trece fereastra de
predicție (90 zile la noi). Nu poți calcula MAE la fiecare request — abia
îl afli mult mai târziu, când poți compara predicția veche cu ce s-a
întâmplat de fapt.

ÎN PRODUCȚIE REALĂ, acest script ar:
  1. citi predicțiile logate în trecut (customer_id, predicted_clv, data
     predicției) dintr-un store persistent
  2. lua tranzacțiile NOI (ultimele 90 de zile) pentru acei clienți
  3. calcula actual_clv și compara cu predicted_clv -> MAE real, "proaspăt"
  4. rula periodic (CronJob în k8s, sau un DAG Airflow) — nu la fiecare request

ÎN ACEST PROIECT, dataset-ul e istoric (2010-2011), deci nu există trafic
nou din care să "aflăm" adevărul peste 90 de zile. Scriptul de mai jos
folosește MAE-ul deja calculat pe test set la antrenare
(models/metadata.json) — demonstrează mecanismul de push complet
funcțional, cu mențiunea onestă că sursa datelor e alta față de producție.

Usage:
    python -m src.evaluate_and_push_mae
"""
import json
import os

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

MODELS_DIR = "models"
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "pushgateway:9091")
JOB_NAME = "clv_model_evaluation"


def main():
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    reg_metrics = metadata["metrics"]["regression"]
    best_clf = metadata["metrics"]["classification"][0]  # deja sortat după roc_auc

    registry = CollectorRegistry()

    mae_gauge = Gauge("clv_model_mae", "Mean Absolute Error al modelului de regresie CLV", registry=registry)
    rmse_gauge = Gauge("clv_model_rmse", "Root Mean Squared Error al modelului de regresie CLV", registry=registry)
    r2_gauge = Gauge("clv_model_r2", "R-squared al modelului de regresie CLV", registry=registry)
    auc_gauge = Gauge("clv_churn_model_roc_auc", "ROC-AUC al modelului de clasificare churn", registry=registry)

    mae_gauge.set(reg_metrics["mae"])
    rmse_gauge.set(reg_metrics["rmse"])
    r2_gauge.set(reg_metrics["r2"])
    auc_gauge.set(best_clf["roc_auc"])

    push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)

    print(f"Trimise către Pushgateway ({PUSHGATEWAY_URL}):")
    print(f"  MAE:  {reg_metrics['mae']:.2f}")
    print(f"  RMSE: {reg_metrics['rmse']:.2f}")
    print(f"  R2:   {reg_metrics['r2']:.3f}")
    print(f"  Churn ROC-AUC: {best_clf['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
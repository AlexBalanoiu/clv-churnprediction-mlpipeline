"""Entry point: run the full CLV pipeline and save model artifacts.

Usage:
    python -m src.train
"""
import json
import os

import joblib

from .data_cleaning import load_and_clean
from .features import build_features, FEATURE_COLS
from .models import split_data, train_regressor, evaluate_regressor, train_best_classifier

DATA_PATH = os.path.join("data", "online_retail.csv")
MODELS_DIR = "models"
HORIZON_DAYS = 90


def main():
    print("Loading and cleaning data...")
    data = load_and_clean(DATA_PATH)
    print(f"  {len(data)} rows, {data['CustomerID'].nunique()} customers")

    print("Building features...")
    features, calibration_end, observation_end = build_features(data, holdout_days=HORIZON_DAYS)
    print(f"  {len(features)} customers usable for modeling")

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(features)

    print("Tuning + training XGBoost regressor (this can take a minute)...")
    reg_model, best_params = train_regressor(X_train, y_reg_train)
    reg_metrics = evaluate_regressor(reg_model, X_test, y_reg_test, y_reg_train)
    print(f"  MAE={reg_metrics['mae']:.2f}  R2={reg_metrics['r2']:.3f}  "
          f"(baseline MAE={reg_metrics['baseline_mae']:.2f})")

    print("Training churn classifiers...")
    clf_model, best_clf_name, clf_results = train_best_classifier(
        X_train, y_clf_train, X_test, y_clf_test
    )
    print(f"  Best: {best_clf_name} (ROC-AUC={clf_results[0]['roc_auc']:.3f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    reg_model.save_model(os.path.join(MODELS_DIR, "xgb_clv.json"))
    joblib.dump(clf_model, os.path.join(MODELS_DIR, "churn_model.pkl"))

    metadata = {
        "feature_cols": FEATURE_COLS,
        "horizon_days": HORIZON_DAYS,
        "best_churn_model": best_clf_name,
        "regression_best_params": best_params,
        "metrics": {"regression": reg_metrics, "classification": clf_results},
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to ./{MODELS_DIR}/")


if __name__ == "__main__":
    main()

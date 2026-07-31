"""Pure inference function — used directly by the FastAPI wrapper, no refactor needed."""
import json
import os

import joblib
import pandas as pd
import xgboost as xgb

MODELS_DIR = "models"

_reg_model = None
_clf_model = None
_metadata = None


def _load_artifacts():
    """Lazy-load models once per process (cheap on repeated calls)."""
    global _reg_model, _clf_model, _metadata
    if _reg_model is None:
        _reg_model = xgb.XGBRegressor()
        _reg_model.load_model(os.path.join(MODELS_DIR, "xgb_clv.json"))
        _clf_model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
        with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
            _metadata = json.load(f)
    return _reg_model, _clf_model, _metadata


def predict_customer(customer_row: dict) -> dict:
    """
    customer_row must contain: frequency_cal, recency_cal, T_cal,
    monetary_value_cal, avg_basket_value, n_unique_products, avg_quantity, tenure_days
    """
    reg_model, clf_model, metadata = _load_artifacts()

    x = pd.DataFrame([customer_row])[metadata["feature_cols"]]

    clv_pred = float(reg_model.predict(x)[0])
    churn_proba = float(clf_model.predict_proba(x)[:, 1][0])
    prob_active = 1 - churn_proba
    expected_value = clv_pred * prob_active

    return {
        "predicted_clv": round(clv_pred, 2),
        "churn_probability": round(churn_proba, 3),
        "prob_active": round(prob_active, 3),
        "expected_value": round(expected_value, 2),
    }

import os
import pytest

from src.inference import predict_customer

MODELS_EXIST = os.path.exists(os.path.join("models", "metadata.json"))

SAMPLE_CUSTOMER = {
    "frequency_cal": 5,
    "recency_cal": 120,
    "T_cal": 200,
    "monetary_value_cal": 45.5,
    "avg_basket_value": 45.5,
    "n_unique_products": 12,
    "avg_quantity": 8.3,
    "tenure_days": 200,
}


@pytest.mark.skipif(not MODELS_EXIST, reason="run `python -m src.train` first to generate models/")
def test_predict_customer_returns_expected_keys():
    result = predict_customer(SAMPLE_CUSTOMER)

    assert set(result.keys()) == {
        "predicted_clv", "churn_probability", "prob_active", "expected_value",
    }
    assert result["predicted_clv"] >= 0
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["prob_active"] <= 1
    assert abs(result["prob_active"] - (1 - result["churn_probability"])) < 1e-6
    assert abs(result["expected_value"] - result["predicted_clv"] * result["prob_active"]) < 0.05

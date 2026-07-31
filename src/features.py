"""Feature engineering for the CLV pipeline: calibration/holdout split + RFM features."""
import pandas as pd
from lifetimes.utils import calibration_and_holdout_data

FEATURE_COLS = [
    "frequency_cal", "recency_cal", "T_cal", "monetary_value_cal",
    "avg_basket_value", "n_unique_products", "avg_quantity", "tenure_days",
]


def build_features(data: pd.DataFrame, holdout_days: int = 90):
    """Build the calibration/holdout feature set and both targets.

    Returns
    -------
    features : pd.DataFrame
        One row per customer, with FEATURE_COLS + 'actual_clv_holdout' + 'churned'
    calibration_end : pd.Timestamp
    observation_end : pd.Timestamp
    """
    observation_end = data["InvoiceDate"].max()
    calibration_end = observation_end - pd.Timedelta(days=holdout_days)

    cal_hold = calibration_and_holdout_data(
        transactions=data,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        calibration_period_end=calibration_end,
        observation_period_end=observation_end,
        freq="D",
    )
    cal_hold = cal_hold[cal_hold["frequency_cal"] > 0]

    cal_data = data[data["InvoiceDate"] <= calibration_end]
    extra_feats = cal_data.groupby("CustomerID").agg(
        avg_basket_value=("TotalPrice", "mean"),
        n_unique_products=("StockCode", "nunique"),
        avg_quantity=("Quantity", "mean"),
        tenure_days=("InvoiceDate", lambda x: (calibration_end - x.min()).days),
    )

    features = cal_hold.join(extra_feats, how="left").fillna(0)
    features["actual_clv_holdout"] = (
        features["monetary_value_holdout"] * features["frequency_holdout"]
    )
    features["churned"] = (features["frequency_holdout"] == 0).astype(int)

    return features, calibration_end, observation_end


def build_customer_features_for_scoring(data: pd.DataFrame, as_of: pd.Timestamp = None) -> pd.DataFrame:
    """Build FEATURE_COLS for every customer as of a given date, for scoring
    (production) rather than training. No holdout/target — used by inference.
    """
    if as_of is None:
        as_of = data["InvoiceDate"].max()

    data = data[data["InvoiceDate"] <= as_of]

    grouped = data.groupby("CustomerID")
    last_purchase = grouped["InvoiceDate"].max()
    first_purchase = grouped["InvoiceDate"].min()

    out = pd.DataFrame({
        "frequency_cal": grouped["InvoiceDate"].nunique() - 1,
        "recency_cal": (last_purchase - first_purchase).dt.days,
        "T_cal": (as_of - first_purchase).dt.days,
        "monetary_value_cal": grouped["TotalPrice"].mean(),
        "avg_basket_value": grouped["TotalPrice"].mean(),
        "n_unique_products": grouped["StockCode"].nunique(),
        "avg_quantity": grouped["Quantity"].mean(),
        "tenure_days": (as_of - first_purchase).dt.days,
    })
    return out[FEATURE_COLS]

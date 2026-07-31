"""Data loading and cleaning for the CLV pipeline."""
import pandas as pd


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw Online Retail transactions CSV and apply cleaning rules.

    Rules:
      - drop rows without CustomerID
      - drop cancellations (InvoiceNo starting with 'C')
      - drop non-positive Quantity / UnitPrice
      - compute TotalPrice = Quantity * UnitPrice
      - drop extreme outliers (> 99.9th percentile of TotalPrice)
    """
    df = pd.read_csv(csv_path, encoding="latin1")

    data = df.copy()
    data["CustomerID"] = data["CustomerID"].astype("Int64")
    data = data.dropna(subset=["CustomerID"])

    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data = data[~data["InvoiceNo"].str.startswith("C")]

    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    cap = data["TotalPrice"].quantile(0.999)
    data = data[data["TotalPrice"] <= cap]

    return data.reset_index(drop=True)

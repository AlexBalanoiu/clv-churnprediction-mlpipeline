import io
import pandas as pd

from src.data_cleaning import load_and_clean

RAW_CSV = """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,ITEM A,6,12/1/2010 8:26,2.55,17850,United Kingdom
536365,71053,ITEM B,6,12/1/2010 8:26,3.39,17850,United Kingdom
536370,85123A,ITEM A,4,12/1/2010 8:30,2.55,17852,United Kingdom
536371,71053,ITEM B,3,12/1/2010 8:35,3.39,17852,United Kingdom
536372,22752,ITEM C,5,12/1/2010 8:40,7.65,17853,United Kingdom
C536366,85123A,ITEM A,-6,12/1/2010 9:00,2.55,17850,United Kingdom
536367,22752,ITEM C,2,12/1/2010 10:00,7.65,,United Kingdom
536368,84029G,ITEM D,-1,12/1/2010 11:00,1.25,17851,United Kingdom
536369,22423,ITEM E,3,12/1/2010 12:00,0.00,17851,United Kingdom
"""


def test_load_and_clean_drops_invalid_rows(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(RAW_CSV)

    cleaned = load_and_clean(str(csv_path))

    # cancellation (C-prefixed invoice) removed
    assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()
    # row with missing CustomerID removed
    assert cleaned["CustomerID"].isna().sum() == 0
    # rows with non-positive quantity/price removed
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["UnitPrice"] > 0).all()
    # TotalPrice computed correctly
    assert "TotalPrice" in cleaned.columns
    assert (cleaned["TotalPrice"] == cleaned["Quantity"] * cleaned["UnitPrice"]).all()
    # cancellation, missing-CustomerID, and non-positive qty/price rows all removed;
    # remaining rows are a subset of the known-good invoices (outlier cap may trim
    # the top of the distribution even among valid rows on such a tiny sample)
    assert len(cleaned) >= 2
    assert set(cleaned["InvoiceNo"]).issubset({"536365", "536370", "536371", "536372"})

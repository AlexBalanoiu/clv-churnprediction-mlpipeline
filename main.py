"""FastAPI wrapper over the CLV pipeline.

Run locally:
    uvicorn main:app --reload --port 8000

Then POST to http://localhost:8000/predict, or open http://localhost:8000/docs
for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from src.inference import predict_customer

app = FastAPI(
    title="Customer Lifetime Value API",
    description="Predicts 90-day CLV, churn probability, and expected value for a customer.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

class CustomerFeatures(BaseModel):
    frequency_cal: float = Field(..., description="Number of repeat purchases in calibration period")
    recency_cal: float = Field(..., description="Days between first and last purchase in calibration")
    T_cal: float = Field(..., description="Customer age in days at end of calibration period")
    monetary_value_cal: float = Field(..., description="Average transaction value in calibration")
    avg_basket_value: float = Field(..., description="Average order value")
    n_unique_products: float = Field(..., description="Number of distinct products purchased")
    avg_quantity: float = Field(..., description="Average quantity per order line")
    tenure_days: float = Field(..., description="Days since first purchase")

    class Config:
        json_schema_extra = {
            "example": {
                "frequency_cal": 5,
                "recency_cal": 120,
                "T_cal": 200,
                "monetary_value_cal": 45.5,
                "avg_basket_value": 45.5,
                "n_unique_products": 12,
                "avg_quantity": 8.3,
                "tenure_days": 200,
            }
        }


class CLVPrediction(BaseModel):
    predicted_clv: float
    churn_probability: float
    prob_active: float
    expected_value: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=CLVPrediction)
def predict(customer: CustomerFeatures):
    try:
        result = predict_customer(customer.model_dump())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run `python -m src.train` first.",
        )
    return result

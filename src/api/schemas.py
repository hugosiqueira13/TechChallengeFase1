"""Schemas Pydantic para a API de churn (Telco Customer Churn)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Features de entrada de um cliente Telco para predição de churn."""

    # Numéricos
    tenure: int = Field(..., ge=0, le=72, description="Meses como cliente")
    MonthlyCharges: float = Field(..., gt=0, description="Cobrança mensal (R$)")
    TotalCharges: float | None = Field(None, ge=0, description="Total acumulado (R$)")

    # Demográficos
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # Telefonia
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    # Internet
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # Contrato e pagamento
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        }
    }


class PredictResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_predicted: bool
    threshold_used: float
    risk_tier: Literal["low", "medium", "high"]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

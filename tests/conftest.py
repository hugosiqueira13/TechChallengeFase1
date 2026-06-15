"""Fixtures compartilhadas entre todos os módulos de teste."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.mlp import MLP

# ── Cliente de exemplo (valores reais do dataset Telco) ───────────────────────
SAMPLE_CUSTOMER: dict = {
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


@pytest.fixture()
def sample_customer() -> dict:
    return SAMPLE_CUSTOMER.copy()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame com 5 cópias do cliente de exemplo."""
    return pd.DataFrame([SAMPLE_CUSTOMER] * 5)


@pytest.fixture()
def tiny_mlp() -> MLP:
    """MLP pequeno para testes rápidos (input_dim=10)."""
    return MLP(input_dim=10, output_dim=1, hidden_dims=[8, 4], dropout=0.0)


@pytest.fixture()
def batch_input() -> torch.Tensor:
    return torch.randn(4, 10)

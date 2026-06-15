"""Schema Pandera para validação do dataset Telco Customer Churn."""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

TELCO_SCHEMA = DataFrameSchema(
    columns={
        # ── Numéricos ──────────────────────────────────────────────────────────
        "tenure": Column(int, Check.in_range(0, 72), nullable=False),
        "MonthlyCharges": Column(float, Check.greater_than(0), nullable=False),
        # TotalCharges é null para clientes recém-chegados
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        # ── Demográficos ───────────────────────────────────────────────────────
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        # ── Telefonia ─────────────────────────────────────────────────────────
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),
        # ── Internet ──────────────────────────────────────────────────────────
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        # ── Contrato e pagamento ───────────────────────────────────────────────
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
        ),
        # ── Target (opcional — ausente em chamadas de inferência) ─────────────
        "Churn": Column(str, Check.isin(["Yes", "No"]), nullable=True, required=False),
    },
    coerce=True,
    strict=False,
)

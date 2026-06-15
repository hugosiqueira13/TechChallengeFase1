"""Testes de validação de schema com Pandera (Telco Customer Churn)."""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from src.schemas.telco import TELCO_SCHEMA
from tests.conftest import SAMPLE_CUSTOMER


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    return pd.DataFrame([SAMPLE_CUSTOMER])


class TestTelcoSchemaValid:
    def test_valid_row_passes(self, valid_df: pd.DataFrame) -> None:
        TELCO_SCHEMA.validate(valid_df)

    def test_multiple_valid_rows_pass(self) -> None:
        df = pd.DataFrame([SAMPLE_CUSTOMER] * 10)
        TELCO_SCHEMA.validate(df)

    def test_null_total_charges_allowed(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["TotalCharges"] = None
        TELCO_SCHEMA.validate(df)

    def test_churn_column_optional(self, valid_df: pd.DataFrame) -> None:
        """Coluna Churn ausente deve passar (inferência não a fornece)."""
        assert "Churn" not in valid_df.columns
        TELCO_SCHEMA.validate(valid_df)

    def test_churn_column_present_valid(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["Churn"] = "Yes"
        TELCO_SCHEMA.validate(df)


class TestTelcoSchemaInvalid:
    def test_invalid_gender_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["gender"] = "Unknown"
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

    def test_negative_tenure_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["tenure"] = -1
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

    def test_tenure_above_72_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["tenure"] = 73
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

    def test_zero_monthly_charges_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["MonthlyCharges"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

    def test_invalid_contract_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["Contract"] = "Weekly"
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

    def test_invalid_payment_method_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["PaymentMethod"] = "Crypto"
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

    def test_invalid_senior_citizen_fails(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["SeniorCitizen"] = 2
        with pytest.raises(pa.errors.SchemaError):
            TELCO_SCHEMA.validate(df)

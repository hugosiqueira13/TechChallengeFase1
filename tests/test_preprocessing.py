"""Testes para o pipeline de pré-processamento (features Telco Churn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import build_preprocessor, encode_labels
from src.utils.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from tests.conftest import SAMPLE_CUSTOMER


@pytest.fixture()
def preprocessor(sample_df: pd.DataFrame):
    pp = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    pp.fit(sample_df)
    return pp


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, sample_df: pd.DataFrame) -> None:
        from sklearn.compose import ColumnTransformer

        pp = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        pp.fit(sample_df)
        assert isinstance(pp, ColumnTransformer)

    def test_output_is_numpy(self, preprocessor, sample_df: pd.DataFrame) -> None:
        X = preprocessor.transform(sample_df)
        assert isinstance(X, np.ndarray)

    def test_output_row_count_matches(self, preprocessor, sample_df: pd.DataFrame) -> None:
        X = preprocessor.transform(sample_df)
        assert X.shape[0] == len(sample_df)

    def test_output_has_no_nan(self, preprocessor, sample_df: pd.DataFrame) -> None:
        X = preprocessor.transform(sample_df)
        assert not np.isnan(X).any()

    def test_output_has_no_inf(self, preprocessor, sample_df: pd.DataFrame) -> None:
        X = preprocessor.transform(sample_df)
        assert not np.isinf(X).any()

    def test_single_row_transform(self, preprocessor) -> None:
        row = pd.DataFrame([SAMPLE_CUSTOMER])
        X = preprocessor.transform(row)
        assert X.shape[0] == 1

    def test_output_dim_greater_than_input_dim(self) -> None:
        """OHE expande a dimensão quando há múltiplos valores por categoria."""
        rows = [
            {**SAMPLE_CUSTOMER, "gender": "Male", "Contract": "One year"},
            {**SAMPLE_CUSTOMER, "gender": "Female", "Contract": "Two year"},
            {**SAMPLE_CUSTOMER, "gender": "Male", "Contract": "Month-to-month"},
        ]
        df = pd.DataFrame(rows)
        pp = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        X = pp.fit_transform(df)
        n_raw = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X.shape[1] > n_raw

    def test_null_total_charges_imputed(self, preprocessor) -> None:
        row = pd.DataFrame([{**SAMPLE_CUSTOMER, "TotalCharges": None}])
        X = preprocessor.transform(row)
        assert not np.isnan(X).any()


class TestEncodeLabels:
    def test_returns_array_and_encoder(self) -> None:
        y = pd.Series(["Yes", "No", "Yes", "No"])
        encoded, le = encode_labels(y)
        assert len(encoded) == 4

    def test_binary_classes(self) -> None:
        y = pd.Series(["Yes", "No", "Yes"])
        _, le = encode_labels(y)
        assert len(le.classes_) == 2

    def test_encoded_values_are_0_or_1(self) -> None:
        y = pd.Series(["Yes", "No", "Yes", "No"])
        encoded, _ = encode_labels(y)
        assert set(encoded).issubset({0, 1})

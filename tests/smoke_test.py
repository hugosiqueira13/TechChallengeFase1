"""Smoke tests end-to-end: valida caminho crítico sem pesos reais."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.models.mlp import MLP
from src.preprocessing.pipeline import build_preprocessor
from src.schemas.telco import TELCO_SCHEMA
from src.utils.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from tests.conftest import SAMPLE_CUSTOMER

# ── Fixtures com escopo de módulo (criadas uma vez por sessão) ────────────────

@pytest.fixture(scope="module")
def fitted_preprocessor():
    df = pd.DataFrame([SAMPLE_CUSTOMER] * 50)
    pp = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    pp.fit(df)
    return pp


@pytest.fixture(scope="module")
def untrained_model(fitted_preprocessor):
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    input_dim = fitted_preprocessor.transform(df).shape[1]
    return MLP(input_dim=input_dim, output_dim=1, hidden_dims=[16, 8], dropout=0.0)


# ── Smoke tests ───────────────────────────────────────────────────────────────

class TestSchemaToPipelineToModel:
    def test_full_pipeline_single_customer(
        self, fitted_preprocessor, untrained_model: MLP
    ) -> None:
        df = pd.DataFrame([SAMPLE_CUSTOMER])
        TELCO_SCHEMA.validate(df)
        X = fitted_preprocessor.transform(df)
        tensor = torch.tensor(X, dtype=torch.float32)
        untrained_model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(untrained_model(tensor)).item()
        assert 0.0 <= prob <= 1.0

    def test_probability_always_in_range(
        self, fitted_preprocessor, untrained_model: MLP
    ) -> None:
        """10 chamadas independentes — probabilidade sempre em [0, 1]."""
        untrained_model.eval()
        for _ in range(10):
            df = pd.DataFrame([SAMPLE_CUSTOMER])
            X = fitted_preprocessor.transform(df)
            tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                prob = torch.sigmoid(untrained_model(tensor)).item()
            assert 0.0 <= prob <= 1.0

    def test_batch_inference_shape(
        self, fitted_preprocessor, untrained_model: MLP
    ) -> None:
        batch_size = 16
        df = pd.DataFrame([SAMPLE_CUSTOMER] * batch_size)
        X = fitted_preprocessor.transform(df)
        tensor = torch.tensor(X, dtype=torch.float32)
        untrained_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(untrained_model(tensor))
        assert probs.shape == (batch_size, 1)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestSchemaValidationSmoke:
    def test_valid_customer_passes_schema(self) -> None:
        df = pd.DataFrame([SAMPLE_CUSTOMER])
        TELCO_SCHEMA.validate(df)

    def test_pipeline_output_has_no_nan(self, fitted_preprocessor) -> None:
        import numpy as np
        df = pd.DataFrame([SAMPLE_CUSTOMER])
        X = fitted_preprocessor.transform(df)
        assert not np.isnan(X).any()

    def test_pipeline_output_has_no_inf(self, fitted_preprocessor) -> None:
        import numpy as np
        df = pd.DataFrame([SAMPLE_CUSTOMER])
        X = fitted_preprocessor.transform(df)
        assert not np.isinf(X).any()

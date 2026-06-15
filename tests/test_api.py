"""Testes de integração para a API FastAPI (Telco Churn)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_CUSTOMER


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_predictor() -> MagicMock:
    mock = MagicMock()
    mock.is_ready = True
    mock.predict.return_value = {
        "churn_probability": 0.82,
        "churn_predicted": True,
        "threshold_used": 0.5,
        "risk_tier": "high",
    }
    return mock


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert {"status", "model_loaded", "version"} <= data.keys()

    def test_status_is_ok(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_latency_header_present(self, client: TestClient) -> None:
        assert "x-latency-ms" in client.get("/health").headers

    def test_latency_is_non_negative(self, client: TestClient) -> None:
        val = float(client.get("/health").headers["x-latency-ms"])
        assert val >= 0


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_payload_returns_200(
        self, client: TestClient, mock_predictor: MagicMock
    ) -> None:
        with patch("src.api.main._predictor", mock_predictor):
            resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 200

    def test_response_schema(
        self, client: TestClient, mock_predictor: MagicMock
    ) -> None:
        with patch("src.api.main._predictor", mock_predictor):
            data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert {"churn_probability", "churn_predicted", "threshold_used", "risk_tier"} <= data.keys()

    def test_probability_in_range(
        self, client: TestClient, mock_predictor: MagicMock
    ) -> None:
        with patch("src.api.main._predictor", mock_predictor):
            prob = client.post("/predict", json=SAMPLE_CUSTOMER).json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_latency_header_present(
        self, client: TestClient, mock_predictor: MagicMock
    ) -> None:
        with patch("src.api.main._predictor", mock_predictor):
            resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert "x-latency-ms" in resp.headers

    def test_model_not_ready_returns_503(self, client: TestClient) -> None:
        not_ready = MagicMock()
        not_ready.is_ready = False
        with patch("src.api.main._predictor", not_ready):
            resp = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 503


# ── Validação Pydantic ─────────────────────────────────────────────────────────

class TestPredictValidation:
    def test_invalid_gender_returns_422(self, client: TestClient) -> None:
        payload = {**SAMPLE_CUSTOMER, "gender": "X"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_tenure_returns_422(self, client: TestClient) -> None:
        payload = {**SAMPLE_CUSTOMER, "tenure": -1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_contract_returns_422(self, client: TestClient) -> None:
        payload = {**SAMPLE_CUSTOMER, "Contract": "Daily"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_null_total_charges_accepted(
        self, client: TestClient, mock_predictor: MagicMock
    ) -> None:
        payload = {**SAMPLE_CUSTOMER, "TotalCharges": None}
        with patch("src.api.main._predictor", mock_predictor):
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

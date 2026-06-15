"""Carregamento do modelo Telco Churn e lógica de inferência."""

from __future__ import annotations

import logging

import pandas as pd
import torch

from src.models.mlp import MLP
from src.preprocessing.pipeline import load_preprocessor
from src.utils.config import CATEGORICAL_FEATURES, DEFAULT_HPARAMS, MODELS_DIR, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

_THRESHOLD = 0.5
_RISK_THRESHOLDS = [(0.3, "low"), (0.6, "medium"), (1.0, "high")]
_WEIGHTS_PATH = MODELS_DIR / "saved" / "mlp_model.pt"


def _risk_tier(prob: float) -> str:
    for ceiling, label in _RISK_THRESHOLDS:
        if prob <= ceiling:
            return label
    return "high"


class Predictor:
    """Encapsula carregamento lazy e inferência do MLP treinado para churn."""

    def __init__(self) -> None:
        self._preprocessor = None
        self._model: MLP | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._threshold = _THRESHOLD

    def load(self) -> None:
        self._preprocessor = load_preprocessor()

        # Infere input_dim diretamente dos pesos salvos (evita dados dummy)
        state_dict = torch.load(_WEIGHTS_PATH, map_location=self._device, weights_only=True)
        input_dim: int = state_dict["network.0.weight"].shape[1]

        self._model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=DEFAULT_HPARAMS["hidden_dims"],
            dropout=0.0,
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device).eval()
        logger.info("Modelo carregado: input_dim=%d, device=%s", input_dim, self._device)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._preprocessor is not None

    def predict(self, features: dict) -> dict:
        df = pd.DataFrame([features])
        X = self._preprocessor.transform(df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
        tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            logit = self._model(tensor)  # type: ignore[misc]
        prob = float(torch.sigmoid(logit).cpu().item())
        predicted = prob >= self._threshold

        logger.info(
            "Predição: prob=%.4f, predicted=%s, threshold=%.2f",
            prob,
            predicted,
            self._threshold,
        )
        return {
            "churn_probability": round(prob, 4),
            "churn_predicted": predicted,
            "threshold_used": self._threshold,
            "risk_tier": _risk_tier(prob),
        }

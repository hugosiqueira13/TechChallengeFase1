"""Carregamento do modelo Telco Churn e lógica de inferência."""

from __future__ import annotations

import logging

import pandas as pd
from types import SimpleNamespace
import torch

from src.models.mlp import MLP
from src.preprocessing.pipeline import load_preprocessor
from src.utils.config import CATEGORICAL_FEATURES, DEFAULT_HPARAMS, MODELS_DIR, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

_THRESHOLD = 0.5
_RISK_THRESHOLDS = [(0.3, "low"), (0.6, "medium"), (1.0, "high")]
_WEIGHTS_PATH = MODELS_DIR / "mlp_model.pt"


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
        pre = load_preprocessor()

        # Se o preprocessor salvo for um tuple (columns, scaler), cria um wrapper
        if isinstance(pre, (list, tuple)) and len(pre) == 2:
            columns, scaler = pre

            def _transform(df: pd.DataFrame):
                X = pd.get_dummies(df, drop_first=True)
                X = X.reindex(columns=columns, fill_value=0)
                return scaler.transform(X)

            self._preprocessor = SimpleNamespace(transform=_transform)
        else:
            self._preprocessor = pre

        # Carrega o modelo salvo (MLflow salva o modelo inteiro, não apenas state_dict)
        loaded = torch.load(_WEIGHTS_PATH, map_location=self._device, weights_only=False)
        
        # Se é um dicionário, é state_dict; senão, é a classe MLP
        if isinstance(loaded, dict):
            state_dict = loaded
        else:
            # É uma instância de MLP, extrai o state_dict
            state_dict = loaded.state_dict()
        
        # Infere input_dim a partir do primeiro layer da rede
        first_weight_key = next((k for k in state_dict.keys() if 'weight' in k and '0.' in k), None)
        if not first_weight_key:
            raise ValueError("Não foi possível inferir input_dim do modelo salvo")
        
        input_dim: int = state_dict[first_weight_key].shape[1]

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

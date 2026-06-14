"""Pipeline de pré-processamento com Scikit-Learn."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils.config import PREPROCESSOR_PATH


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Constrói o ColumnTransformer com pipelines para numéricas e categóricas."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_and_save_preprocessor(
    X: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Treina o preprocessador e salva como artefato."""
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    preprocessor.fit(X)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Carrega o preprocessador salvo em disco."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessador não encontrado em {PREPROCESSOR_PATH}")
    return joblib.load(PREPROCESSOR_PATH)


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Codifica os rótulos e retorna array + encoder."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

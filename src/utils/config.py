"""Configurações centralizadas do projeto."""

from pathlib import Path

# Raízes
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "saved"

# Dataset principal
DATASET_PATH = ROOT_DIR / "dataset" / "Dataset Telco-Customer-Churn.csv"

# MLflow — SQLite (compatível com MLflow 3.x)
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "Telco-Churn-MLP-Comparison"

# Dados processados
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artefatos salvos
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "mlp_model.pt"

# Features do dataset Telco-Customer-Churn
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
DROP_FEATURES = ["customerID"]
TARGET_COLUMN = "Churn"

# Modelo de custo (churn business impact)
MONTHS_RETAINED = 12
RETENTION_RATE = 0.40
CAMPAIGN_COST_RATIO = 0.30  # custo da campanha = 30% ARPU

# Hiperparâmetros padrão do MLP
DEFAULT_HPARAMS: dict = {
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 200,
    "patience": 15,
    "random_state": 42,
    "test_size": 0.2,
}

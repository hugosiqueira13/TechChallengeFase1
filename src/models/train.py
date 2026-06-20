"""Script de treinamento: MLP PyTorch + baselines Scikit-Learn + MLflow tracking."""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from src.models.dataset import TabularDataset
from src.models.mlp import MLP
from src.preprocessing.pipeline import fit_and_save_preprocessor
from src.utils.config import (
    CAMPAIGN_COST_RATIO,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    DEFAULT_HPARAMS,
    DROP_FEATURES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MONTHS_RETAINED,
    NUMERICAL_FEATURES,
    RETENTION_RATE,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Tipos de dados ───────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    cost_fn: float
    cost_fp: float
    net_value: float
    best_threshold: float = 0.5


# ── Carregamento de dados ────────────────────────────────────────────────────

def load_telco_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega e limpa o dataset Telco-Customer-Churn."""
    df = pd.read_csv(DATASET_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop(columns=DROP_FEATURES)

    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


# ── Métricas ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
    }


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    arpu: float,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost_fn = fn * arpu * MONTHS_RETAINED
    cost_fp = fp * arpu * CAMPAIGN_COST_RATIO
    savings = tp * arpu * MONTHS_RETAINED * RETENTION_RATE
    return {
        "biz_cost_fn": round(cost_fn, 2),
        "biz_cost_fp": round(cost_fp, 2),
        "biz_savings": round(savings, 2),
        "biz_net_value": round(savings - cost_fp, 2),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    arpu: float,
    n_steps: int = 81,
) -> tuple[float, float]:
    """Varre thresholds e retorna (threshold_ótimo, melhor_valor_líquido)."""
    best_thr, best_val = 0.5, float("-inf")
    for thr in np.linspace(0.1, 0.9, n_steps):
        preds = (y_prob >= thr).astype(int)
        bm = compute_business_metrics(y_true, preds, arpu)
        if bm["biz_net_value"] > best_val:
            best_val = bm["biz_net_value"]
            best_thr = float(thr)
    return round(best_thr, 2), round(best_val, 2)


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> bool:
        """Retorna True quando deve parar."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Baseline Scikit-Learn ────────────────────────────────────────────────────

def train_sklearn_model(
    name: str,
    model_obj,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    arpu: float,
    base_tags: dict,
) -> ModelResult:
    """Treina um modelo sklearn, loga no MLflow e retorna ModelResult."""
    model_obj.fit(X_train, y_train)
    y_pred = model_obj.predict(X_test)
    y_prob = model_obj.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    biz = compute_business_metrics(y_test, y_pred, arpu)
    opt_thr, opt_val = find_optimal_threshold(y_test, y_prob, arpu)

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tags({**base_tags, "model_type": name})
        params = {k: v for k, v in model_obj.get_params().items()
                  if not isinstance(v, dict) and not callable(v)}
        mlflow.log_params({"model": name, **{k: str(v) for k, v in params.items()}})
        mlflow.log_metrics({**metrics, **biz, "optimal_threshold": opt_thr, "optimal_net_value": opt_val})
        mlflow.sklearn.log_model(model_obj, artifact_path="model")

    log.info("%-22s | AUC: %.4f | F1: %.4f | Recall: %.4f | Net: R$%.0f",
             name, metrics["roc_auc"], metrics["f1"], metrics["recall"], opt_val)
    return ModelResult(
        name=name, **metrics,
        cost_fn=biz["biz_cost_fn"], cost_fp=biz["biz_cost_fp"],
        net_value=opt_val, best_threshold=opt_thr,
    )


# ── Treinamento MLP ──────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hparams: dict,
    arpu: float,
    base_tags: dict,
) -> ModelResult:
    """Treina MLP com early stopping e loga artefatos no MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Dispositivo: %s", device)

    pos_weight_val = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    train_ds = TabularDataset(X_train, y_train.astype(float))
    test_ds = TabularDataset(X_test, y_test.astype(float))
    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=hparams["batch_size"])

    # MLP com saída única (BCEWithLogitsLoss para classificação binária)
    model = MLP(
        input_dim=X_train.shape[1],
        output_dim=1,
        hidden_dims=hparams["hidden_dims"],
        dropout=hparams["dropout"],
    ).to(device)

    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    early_stop = EarlyStopping(patience=hparams["patience"])

    best_roc_auc = 0.0
    best_epoch = 0
    best_state: dict = {}

    with mlflow.start_run(run_name="MLP_PyTorch", nested=True):
        mlflow.set_tags({**base_tags, "model_type": "MLP", "framework": "pytorch"})
        mlflow.log_params({
            "input_dim": X_train.shape[1],
            "num_parameters": model.num_parameters,
            "pos_weight": round(pos_weight_val, 4),
            **{k: str(v) for k, v in hparams.items()},
        })

        for epoch in range(1, hparams["epochs"] + 1):
            # Treino
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(1), yb.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)

            # Validação
            model.eval()
            probs_list, labels_list = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb.to(device)).squeeze(1)
                    probs_list.extend(torch.sigmoid(logits).cpu().numpy())
                    labels_list.extend(yb.numpy())

            val_probs = np.array(probs_list)
            val_labels = np.array(labels_list)
            val_preds = (val_probs >= 0.5).astype(int)
            val_roc_auc = float(roc_auc_score(val_labels, val_probs))
            val_f1 = float(f1_score(val_labels, val_preds, zero_division=0))

            scheduler.step(val_roc_auc)
            mlflow.log_metrics(
                {"train_loss": epoch_loss, "val_roc_auc": val_roc_auc, "val_f1": val_f1},
                step=epoch,
            )

            if val_roc_auc > best_roc_auc:
                best_roc_auc = val_roc_auc
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())

            if early_stop.step(val_roc_auc):
                log.info("Early stopping na epoch %d (melhor: epoch %d, ROC-AUC=%.4f)",
                         epoch, best_epoch, best_roc_auc)
                break

            if epoch % 20 == 0:
                log.info("Epoch %3d | loss: %.4f | roc_auc: %.4f | f1: %.4f",
                         epoch, epoch_loss, val_roc_auc, val_f1)

        # Carrega o melhor estado
        model.load_state_dict(best_state)
        model.eval()

        # Avaliação final
        final_probs, final_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(device)).squeeze(1)
                final_probs.extend(torch.sigmoid(logits).cpu().numpy())
                final_labels.extend(yb.numpy())

        final_probs_arr = np.array(final_probs)
        final_labels_arr = np.array(final_labels)
        final_preds = (final_probs_arr >= 0.5).astype(int)

        metrics = compute_metrics(final_labels_arr, final_preds, final_probs_arr)
        biz = compute_business_metrics(final_labels_arr, final_preds, arpu)
        opt_thr, opt_val = find_optimal_threshold(final_labels_arr, final_probs_arr, arpu)

        mlflow.log_metrics({
            **{f"best_{k}": v for k, v in metrics.items()},
            **biz,
            "best_epoch": best_epoch,
            "optimal_threshold": opt_thr,
            "optimal_net_value": opt_val,
        })

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="mlp_weights")
        mlflow.pytorch.log_model(model, artifact_path="mlp_model")

    log.info("MLP_PyTorch        | AUC: %.4f | F1: %.4f | Recall: %.4f | Net: R$%.0f",
             metrics["roc_auc"], metrics["f1"], metrics["recall"], opt_val)
    return ModelResult(
        name="MLP_PyTorch", **metrics,
        cost_fn=biz["biz_cost_fn"], cost_fp=biz["biz_cost_fp"],
        net_value=opt_val, best_threshold=opt_thr,
    )


# ── Tabela comparativa ───────────────────────────────────────────────────────

def print_comparison_table(results: list[ModelResult]) -> None:
    rows = []
    for r in results:
        rows.append({
            "Modelo": r.name,
            "Accuracy": r.accuracy,
            "Precision": r.precision,
            "Recall": r.recall,
            "F1": r.f1,
            "ROC-AUC": r.roc_auc,
            "PR-AUC": r.pr_auc,
            "Net Value (R$)": r.net_value,
            "Thr*": r.best_threshold,
        })
    df = pd.DataFrame(rows).set_index("Modelo")

    sep = "=" * 105
    log.info("\n%s\n  TABELA COMPARATIVA DE MODELOS — TELCO CHURN\n%s", sep, sep)
    log.info("\n%s", df.to_string(float_format="%.4f"))

    best_auc = df["ROC-AUC"].idxmax()
    best_recall = df["Recall"].idxmax()
    best_net = df["Net Value (R$)"].idxmax()
    log.info("\n  Melhor ROC-AUC    : %s (%.4f)", best_auc, df.loc[best_auc, "ROC-AUC"])
    log.info("  Melhor Recall     : %s (%.4f)", best_recall, df.loc[best_recall, "Recall"])
    log.info("  Melhor Net Value  : %s (R$ %.2f)", best_net, df.loc[best_net, "Net Value (R$)"])


# ── Entry-point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina MLP + baselines com MLflow tracking")
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_HPARAMS["epochs"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_HPARAMS["lr"])
    parser.add_argument("--batch-size",   type=int,   default=DEFAULT_HPARAMS["batch_size"])
    parser.add_argument("--dropout",      type=float, default=DEFAULT_HPARAMS["dropout"])
    parser.add_argument("--patience",     type=int,   default=DEFAULT_HPARAMS["patience"])
    parser.add_argument("--random-state", type=int,   default=DEFAULT_HPARAMS["random_state"])
    args = parser.parse_args()

    hparams = {
        **DEFAULT_HPARAMS,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "patience": args.patience,
        "random_state": args.random_state,
    }

    # 1. Dados
    X_df, y_series = load_telco_data()
    y = y_series.values
    arpu = float(pd.read_csv(DATASET_PATH)["MonthlyCharges"].median())
    log.info("Dataset: %d amostras | %d features | churn rate=%.2f%%",
             len(y), X_df.shape[1], y.mean() * 100)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=hparams["test_size"],
        random_state=hparams["random_state"],
        stratify=y,
    )

    # 2. Pré-processamento com pipeline (StandardScaler + OHE)
    preprocessor = fit_and_save_preprocessor(
        X_train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )
    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    base_tags = {
        "project": "telco-churn",
        "split_strategy": "stratified",
    }

    # 3. MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results: list[ModelResult] = []

    with mlflow.start_run(run_name="full-comparison"):
        # Baselines
        baselines = [
            ("LogisticRegression", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000, C=1.0)),
            ])),
            ("DecisionTree", DecisionTreeClassifier(
                max_depth=5, random_state=42, class_weight="balanced"
            )),
            ("RandomForest", RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42,
                class_weight="balanced", n_jobs=-1,
            )),
            ("GradientBoosting", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ]

        for name, model_obj in baselines:
            r = train_sklearn_model(
                name, model_obj, X_train, y_train, X_test, y_test, arpu, base_tags
            )
            results.append(r)

        # MLP
        mlp_result = train_mlp(X_train, y_train, X_test, y_test, hparams, arpu, base_tags)
        results.append(mlp_result)

    print_comparison_table(results)
    log.info("Artefatos no MLflow: %s", MLFLOW_TRACKING_URI)
    log.info("Execute: mlflow ui --backend-store-uri \"%s\"", MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    main()

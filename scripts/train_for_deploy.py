#!/usr/bin/env python3
"""Train the Telco churn model during deploy/build time.

This script is intended to be used by Render, Vercel (Docker build), or GitHub Actions
when the app needs to generate model artifacts at deploy time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from subprocess import run

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "Dataset Telco-Customer-Churn.csv"
MODEL_DIR = ROOT_DIR / "models" / "saved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina o modelo Telco Churn e salva os artefatos em models/saved"
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("DEPLOY_EPOCHS", 20)))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("DEPLOY_BATCH_SIZE", 64)))
    parser.add_argument("--lr", type=float, default=float(os.getenv("DEPLOY_LR", 1e-3)))
    parser.add_argument("--dropout", type=float, default=float(os.getenv("DEPLOY_DROPOUT", 0.3)))
    parser.add_argument("--patience", type=int, default=int(os.getenv("DEPLOY_PATIENCE", 10)))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("DEPLOY_RANDOM_STATE", 42)))
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing artifacts and retrain")
    return parser.parse_args()


def validate_environment() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATASET_PATH}. Verifique se o arquivo está no repositório."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    os.chdir(ROOT_DIR)
    validate_environment()

    print("[deploy train] Root directory:", ROOT_DIR)
    print("[deploy train] Dataset:", DATASET_PATH)
    print("[deploy train] Saving artifacts to:", MODEL_DIR)
    print("[deploy train] Hyperparameters: epochs=%d batch_size=%d lr=%s dropout=%s patience=%d"
          % (args.epochs, args.batch_size, args.lr, args.dropout, args.patience))

    model_file = MODEL_DIR / "mlp_model.pt"
    preprocessor_file = MODEL_DIR / "preprocessor.pkl"
    if not args.no_cache and model_file.exists() and preprocessor_file.exists():
        print("[deploy train] Artefatos já existem. Use --no-cache para forçar o retrain.")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "src.models.train",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--dropout",
        str(args.dropout),
        "--patience",
        str(args.patience),
        "--random-state",
        str(args.random_state),
    ]

    result = run(cmd, check=False)
    if result.returncode != 0:
        print("[deploy train] Treinamento falhou. Veja os logs acima.")
        return result.returncode

    print("[deploy train] Treinamento concluído e artefatos salvos em models/saved/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

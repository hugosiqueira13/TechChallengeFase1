"""Runner de tarefas cross-platform (substituto do make para Windows)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _run(*cmd: str) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def install() -> None:
    _run(sys.executable, "-m", "pip", "install", "-e", ".[dev]")


def lint() -> None:
    _run(sys.executable, "-m", "ruff", "check", "src/", "tests/")


def format_code() -> None:
    _run(sys.executable, "-m", "ruff", "format", "src/", "tests/")


def test() -> None:
    _run(
        sys.executable, "-m", "pytest", "tests/",
        "-v", "--cov=src", "--cov-report=term-missing",
        "--ignore=tests/smoke_test.py",
    )


def test_smoke() -> None:
    _run(sys.executable, "-m", "pytest", "tests/smoke_test.py", "-v")


def run() -> None:
    _run(
        sys.executable, "-m", "uvicorn",
        "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000",
    )


def clean() -> None:
    for name in (".pytest_cache", "htmlcov", ".mypy_cache"):
        shutil.rmtree(name, ignore_errors=True)
    Path(".coverage").unlink(missing_ok=True)
    for p in Path(".").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    print("Limpeza concluída.")


_TASKS = {
    "install": install,
    "lint": lint,
    "format": format_code,
    "test": test,
    "test-smoke": test_smoke,
    "run": run,
    "clean": clean,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _TASKS:
        print("Uso: python tasks.py <tarefa>")
        print("Tarefas disponíveis:", ", ".join(_TASKS))
        sys.exit(1)
    _TASKS[sys.argv[1]]()

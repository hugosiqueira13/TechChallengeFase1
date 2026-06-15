.DEFAULT_GOAL := help

.PHONY: help install lint format test test-smoke run clean

help:
	@echo ""
	@echo "  install     Instala pacote com dependencias de dev"
	@echo "  lint        Ruff linter"
	@echo "  format      Ruff formatter"
	@echo "  test        Suite completa (cobertura)"
	@echo "  test-smoke  Smoke tests end-to-end"
	@echo "  run         Inicia servidor FastAPI"
	@echo "  clean       Remove artefatos gerados"
	@echo ""

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --ignore=tests/smoke_test.py

test-smoke:
	pytest tests/smoke_test.py -v

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, True) for p in ['.pytest_cache', 'htmlcov', '.mypy_cache']]; [p.unlink(missing_ok=True) for p in [pathlib.Path('.coverage')]]; [shutil.rmtree(p, True) for p in pathlib.Path('.').rglob('__pycache__')]"

# ML Project — MLP Classifier

Projeto de Machine Learning com **PyTorch** (MLP), **Scikit-Learn** (pré-processamento e baseline), **MLflow** (tracking de experimentos) e **FastAPI** (API de inferência).

O dataset de exemplo é o clássico **Iris**, mas a arquitetura foi projetada para ser facilmente substituída por qualquer dataset tabular.

---

## Estrutura do Repositório

```
ml-project/
├── src/
│   ├── api/
│   │   ├── main.py          # App FastAPI (endpoints /health e /predict)
│   │   ├── predictor.py     # Carregamento do modelo e lógica de inferência
│   │   └── schemas.py       # Schemas Pydantic (request/response)
│   ├── models/
│   │   ├── mlp.py           # Arquitetura MLP (PyTorch)
│   │   ├── dataset.py       # TabularDataset (torch.utils.data.Dataset)
│   │   └── train.py         # Loop de treinamento + MLflow tracking
│   ├── preprocessing/
│   │   └── pipeline.py      # ColumnTransformer Scikit-Learn
│   └── utils/
│       └── config.py        # Configurações centralizadas (paths, hparams)
├── data/
│   ├── raw/                 # Dados brutos (ignorados pelo git)
│   └── processed/           # Dados processados (ignorados pelo git)
├── models/
│   └── saved/               # Pesos do modelo e artefatos (ignorados pelo git)
├── tests/
│   ├── test_mlp.py          # Testes unitários do modelo PyTorch
│   ├── test_preprocessing.py
│   └── test_api.py          # Testes de integração FastAPI
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── docs/
│   └── architecture.md
├── pyproject.toml           # Single source of truth: deps + lint + pytest
├── .gitignore
└── .env.example
```

---

## Pré-requisitos

- Python **3.10+**
- `pip` atualizado (`pip install --upgrade pip`)

---

## Setup

### 1. Clone e entre no projeto

```bash
git clone https://github.com/<seu-usuario>/ml-project.git
cd ml-project
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# ou
.venv\Scripts\activate           # Windows
```

### 3. Instale as dependências

```bash
# Produção
pip install -e .

# Desenvolvimento (inclui pytest, ruff, mypy)
pip install -e ".[dev]"

# Notebooks (inclui jupyter, matplotlib, seaborn)
pip install -e ".[notebook]"
```

---

## Treinamento

```bash
python -m src.models.train
```

Opções disponíveis:

```bash
python -m src.models.train --epochs 50 --lr 0.001 --batch-size 32 --dropout 0.2
```

Após o treinamento:
- O preprocessador Scikit-Learn é salvo em `models/saved/preprocessor.pkl`
- Os pesos do MLP são salvos em `models/saved/mlp_model.pt`
- Métricas e artefatos são registrados no MLflow em `mlruns/`

### Visualizar experimentos no MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
# Abra http://localhost:5000 no navegador
```

---

## API de Inferência

### Iniciar o servidor

```bash
# Via entry-point
python -m src.api.main

# Ou diretamente com uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status da API e do modelo |
| `POST` | `/predict` | Inferência |
| `GET` | `/docs` | Swagger UI (FastAPI automático) |
| `GET` | `/redoc` | ReDoc |

### Exemplo de predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2
  }'
```

Resposta esperada:

```json
{
  "predicted_class": "setosa",
  "class_index": 0,
  "probabilities": {
    "setosa": 0.9821,
    "versicolor": 0.0124,
    "virginica": 0.0055
  }
}
```

---

## Testes

```bash
pytest
```

Com relatório de cobertura:

```bash
pytest --cov=src --cov-report=html
# Abra htmlcov/index.html
```

---

## Linting e Formatação

```bash
# Verificar
ruff check .

# Corrigir automaticamente
ruff check . --fix

# Formatar
ruff format .
```

---

## Bibliotecas Utilizadas

| Biblioteca | Versão mín. | Uso |
|-----------|-------------|-----|
| **PyTorch** | 2.2 | Arquitetura e treinamento do MLP |
| **Scikit-Learn** | 1.4 | Pipeline de pré-processamento e baseline RandomForest |
| **MLflow** | 2.11 | Tracking de experimentos, parâmetros, métricas e artefatos |
| **FastAPI** | 0.110 | API REST de inferência |
| **Uvicorn** | 0.29 | Servidor ASGI para FastAPI |
| **Pydantic** | 2.6 | Validação de schemas (request/response) |

---

## Histórico de Commits Sugerido

```
feat: scaffold inicial com estrutura de diretórios
feat: pipeline de pré-processamento com Scikit-Learn
feat: arquitetura MLP com PyTorch (BatchNorm + Dropout)
feat: loop de treinamento com MLflow tracking
feat: baseline RandomForest com nested run no MLflow
feat: API FastAPI com endpoints /health e /predict
test: testes unitários para MLP, preprocessing e API
docs: README, arquitetura e docstrings
chore: pyproject.toml, .gitignore e .env.example
```

---

## Licença

MIT
# TechChallengeFase1

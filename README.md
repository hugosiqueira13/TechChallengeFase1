# Telco Customer Churn Predictor

Projeto de Machine Learning para predição de churn de clientes de telecomunicações. Utiliza uma rede neural MLP implementada em **PyTorch**, com pré-processamento via **Scikit-Learn**, rastreamento de experimentos com **MLflow** e API de inferência em tempo real com **FastAPI**.

---

## Resultado Principal

| Métrica | MLP (modelo final) |
|---------|-------------------|
| ROC-AUC | **0.8396** |
| Recall | **0.7834** (threshold 0.5) / **0.9893** (threshold 0.1) |
| Valor líquido estimado | **R$ 109.813** |

O modelo foi selecionado após comparação com Regressão Logística, Árvore de Decisão, Random Forest e Gradient Boosting. Mais detalhes em [docs/model_card.md](docs/model_card.md).

---

## Estrutura do Repositório

```
ml-project/
├── src/
│   ├── api/
│   │   ├── main.py           # App FastAPI (lifespan, /health, /predict)
│   │   ├── middleware.py     # LatencyMiddleware (log JSON + header X-Latency-Ms)
│   │   ├── predictor.py      # Carregamento de artefatos e inferência
│   │   └── schemas.py        # Schemas Pydantic (request / response)
│   ├── models/
│   │   ├── mlp.py            # Arquitetura MLP PyTorch (BatchNorm + Dropout)
│   │   ├── dataset.py        # TabularDataset para DataLoader
│   │   └── train.py          # Pipeline de treino + MLflow tracking
│   ├── preprocessing/
│   │   └── pipeline.py       # ColumnTransformer (Imputer + Scaler + OHE)
│   ├── schemas/
│   │   └── telco.py          # Schema Pandera para validação de DataFrame
│   └── utils/
│       ├── config.py         # Configurações centralizadas (paths, hparams)
│       └── logging_config.py # Logging estruturado JSON
├── tests/
│   ├── conftest.py           # Fixtures compartilhadas
│   ├── test_mlp.py           # Testes unitários do modelo PyTorch
│   ├── test_preprocessing.py # Testes do pipeline Scikit-Learn
│   ├── test_api.py           # Testes de integração FastAPI
│   ├── test_schemas.py       # Testes de validação Pandera
│   └── smoke_test.py         # Teste end-to-end (treino → inferência)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # EDA completo
│   └── 02_MLP_Modeling.ipynb          # Modelagem e comparação de baselines
├── dataset/
│   └── Dataset Telco-Customer-Churn.csv  # Dataset original (7.043 clientes)
├── docs/
│   ├── model_card.md         # Model Card completo (performance, limitações, vieses)
│   ├── deploy_architecture.md # Arquitetura de deploy + justificativa
│   ├── monitoring_plan.md    # Plano de monitoramento e playbook de incidentes
│   └── architecture.md       # Visão geral de componentes
├── models/
│   └── saved/                # Artefatos do modelo (ignorados pelo git)
│       ├── mlp_model.pt
│       └── preprocessor.pkl
├── data/
│   ├── raw/                  # Dados brutos (ignorados pelo git)
│   └── processed/            # Dados processados (ignorados pelo git)
├── Makefile                  # Automação de tarefas (Unix)
├── tasks.py                  # Automação de tarefas (Windows / cross-platform)
├── pyproject.toml            # Dependências, linting e configuração de testes
└── .env.example              # Variáveis de ambiente de exemplo
```

---

## Pré-requisitos

- Python **3.10+**
- `pip` atualizado: `pip install --upgrade pip`

---

## Setup

### 1. Clone o repositório

```bash
git clone https://github.com/hugosiqueira13/ml-project.git
cd ml-project
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
# Apenas produção
pip install -e .

# Com ferramentas de desenvolvimento (pytest, ruff, mypy)
pip install -e ".[dev]"

# Com Jupyter e bibliotecas de visualização
pip install -e ".[notebook]"
```

---

## Treinamento

### Executar com configuração padrão

```bash
python -m src.models.train
```

### Executar com parâmetros customizados

```bash
python -m src.models.train \
  --epochs 200 \
  --lr 0.001 \
  --batch-size 64 \
  --dropout 0.3
```

### Parâmetros disponíveis

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--epochs` | 200 | Número máximo de épocas |
| `--lr` | 0.001 | Taxa de aprendizado |
| `--batch-size` | 64 | Tamanho do mini-batch |
| `--dropout` | 0.3 | Taxa de dropout |
| `--patience` | 15 | Paciência do early stopping |

### Artefatos gerados

Após o treinamento:

- `models/saved/preprocessor.pkl` — ColumnTransformer serializado (joblib)
- `models/saved/mlp_model.pt` — Pesos do modelo PyTorch
- `mlruns/` — Experimentos MLflow (todos os baselines + MLP)

### Visualizar experimentos no MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Abra http://localhost:5000
```

---

## API de Inferência

### Iniciar o servidor

```bash
# Via Makefile (Unix)
make run

# Via tasks.py (Windows)
python tasks.py run

# Diretamente com Uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints disponíveis

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status da API e do modelo |
| `POST` | `/predict` | Predição de churn |
| `GET` | `/docs` | Swagger UI (documentação interativa) |
| `GET` | `/redoc` | ReDoc |

### Verificar saúde da API

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Exemplo de predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 75.50,
    "TotalCharges": 906.0,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

**Resposta esperada:**

```json
{
  "churn_probability": 0.7312,
  "churn_predicted": true,
  "threshold_used": 0.5,
  "risk_tier": "high"
}
```

### Tiers de risco

| Tier | Probabilidade | Ação recomendada |
|------|--------------|-----------------|
| `low` | ≤ 0.30 | Monitoramento padrão |
| `medium` | 0.30 – 0.60 | Oferta preventiva |
| `high` | > 0.60 | Contato proativo de retenção |

---

## Testes

### Executar todos os testes com cobertura

```bash
# Via Makefile (Unix)
make test

# Via tasks.py (Windows)
python tasks.py test

# Diretamente
pytest --cov=src --cov-report=term-missing
```

### Executar smoke test (end-to-end)

```bash
make test-smoke
# ou
python tasks.py test-smoke
```

### Gerar relatório HTML de cobertura

```bash
pytest --cov=src --cov-report=html
# Abra htmlcov/index.html no navegador
```

### Suíte de testes

| Arquivo | Escopo |
|---------|--------|
| `test_mlp.py` | Shape de saída, NaN/Inf, contagem de parâmetros, determinismo em eval mode |
| `test_preprocessing.py` | ColumnTransformer, imputation, encoding, shape de saída |
| `test_api.py` | Endpoints `/health` e `/predict`, validação Pydantic |
| `test_schemas.py` | Validação Pandera do DataFrame de entrada |
| `smoke_test.py` | Pipeline completo treino → inferência |

---

## Linting e Formatação

```bash
# Verificar erros de lint
ruff check src/ tests/

# Corrigir automaticamente
ruff check src/ tests/ --fix

# Formatar código
ruff format src/ tests/
```

---

## Automação de Tarefas

### Unix (Makefile)

```bash
make install     # Instalar dependências de desenvolvimento
make lint        # Verificar lint
make format      # Formatar código
make test        # Executar testes com cobertura
make test-smoke  # Executar smoke test
make run         # Iniciar API em modo desenvolvimento
make clean       # Remover artefatos de cache e build
```

### Windows (tasks.py)

```powershell
python tasks.py install
python tasks.py lint
python tasks.py format
python tasks.py test
python tasks.py test-smoke
python tasks.py run
python tasks.py clean
```

---

## Variáveis de Ambiente

Copie o arquivo de exemplo e ajuste conforme necessário:

```bash
cp .env.example .env
```

| Variável | Default | Descrição |
|----------|---------|-----------|
| `MODEL_PATH` | `models/saved/mlp_model.pt` | Caminho para os pesos do modelo |
| `PREPROCESSOR_PATH` | `models/saved/preprocessor.pkl` | Caminho para o pré-processador |
| `HOST` | `0.0.0.0` | Host do servidor Uvicorn |
| `PORT` | `8000` | Porta do servidor |
| `PREDICT_THRESHOLD` | `0.5` | Threshold de classificação (use `0.1` para máximo recall) |

---

## Arquitetura

```
Dataset CSV (7.043 clientes)
         │
         ▼
  src/models/train.py
    ├── ColumnTransformer (Imputer + Scaler + OHE)
    ├── Baselines (LogReg, Tree, RF, GBM) → MLflow
    └── MLP PyTorch (128→64→32→1) → MLflow
         │
         ├── models/saved/preprocessor.pkl
         └── models/saved/mlp_model.pt
                    │
                    ▼
           src/api/main.py (FastAPI)
            ├── LatencyMiddleware → log JSON
            ├── GET /health → model_loaded status
            └── POST /predict
                  ├── Pydantic validation
                  ├── preprocessor.transform()
                  ├── model.forward() → sigmoid
                  └── PredictResponse (probability + tier)
```

Documentação detalhada em:
- [docs/deploy_architecture.md](docs/deploy_architecture.md) — decisão batch vs. real-time, topologia de deploy
- [docs/monitoring_plan.md](docs/monitoring_plan.md) — métricas, alertas, playbook de incidentes
- [docs/model_card.md](docs/model_card.md) — performance, limitações, vieses, cenários de falha

---

## Dependências Principais

| Biblioteca | Versão mínima | Uso |
|-----------|--------------|-----|
| **PyTorch** | 2.2 | Arquitetura e treinamento do MLP |
| **Scikit-Learn** | 1.4 | Pipeline de pré-processamento e baselines |
| **MLflow** | 2.11 | Tracking de experimentos e artefatos |
| **FastAPI** | 0.110 | API REST de inferência |
| **Uvicorn** | 0.29 | Servidor ASGI |
| **Pydantic** | 2.6 | Validação de schemas |
| **Pandera** | 0.19 | Validação de DataFrames |
| **Joblib** | 1.3 | Serialização do pré-processador |

---

## Licença

MIT

---

**Notas importantes**

- Os artefatos esperados pela API ficam em `models/saved/`.
- O servidor carrega automaticamente o pré-processador salvo e espera que a transformação produza o mesmo número e ordem de colunas usados no treinamento.
- Os testes unitários e de integração passaram localmente após as correções.

---

**Como executar `/predict` localmente (passo a passo)**

1. Ative seu ambiente virtual e instale dependências (veja seção Setup acima).

2. Verifique a presença dos artefatos em `models/saved/`:

   - `models/saved/mlp_model.pt` (pesos do modelo)
   - `models/saved/preprocessor.pkl` (pré-processador)

   Se `preprocessor.pkl` não existir ou você quiser reconstruí-lo, rode um dos scripts:

```bash
python save_safe_preprocessor.py
# ou, para o preprocessor legado compatível com get_dummies
python build_legacy_preprocessor.py
```

3. Inicie a API (qualquer uma das opções):

```bash
# Unix
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Windows (PowerShell)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

4. Verifique o endpoint de saúde:

```bash
curl http://127.0.0.1:8000/health
```

5. Faça uma chamada de predição com `curl` (bash):

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }' | jq
```

6. Chamada equivalente no PowerShell (evita prompt de parsing):

```powershell
$body = '{"tenure":12,"MonthlyCharges":65.5,"TotalCharges":786.0,"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check" }'
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8000/predict -Method POST -Headers @{'Content-Type'='application/json'} -Body $body | Select-Object -ExpandProperty Content
```

7. Resposta esperada (exemplo):

```json
{"churn_probability":0.4976,"churn_predicted":false,"threshold_used":0.5,"risk_tier":"medium"}
```

8. Problemas comuns e soluções rápidas:

- Erro `Modelo não disponível` ou 503: verifique `models/saved/preprocessor.pkl` e `models/saved/mlp_model.pt` e reinicie o servidor.
- Erro de shapes em multiplicação matricial: indica que as colunas produzidas pelo pré-processador não batem com o `input_dim` do modelo; reconstrua o pré-processador com `save_safe_preprocessor.py` ou use `build_legacy_preprocessor.py` para reproduzir o pipeline usado em treino.

---

Se quiser, eu posso também commitar e abrir um PR com essa versão do `README.md`, ou adaptar o texto em inglês. Deseja que eu faça o commit e crie o PR agora?

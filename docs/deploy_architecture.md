# Arquitetura de Deploy — Telco Churn Predictor

## Decisão: Real-Time Inference via API REST

### Abordagem Escolhida

O sistema foi implantado como uma **API REST de inferência em tempo real**, exposta via FastAPI com servidor Uvicorn. Cada requisição recebe um único cliente e retorna a probabilidade de churn em menos de 100ms.

### Justificativa

A escolha por **real-time** (em oposição a batch) foi baseada nos seguintes critérios:

| Critério | Real-Time (escolhido) | Batch |
|----------|-----------------------|-------|
| **Latência** | < 100ms por predição | Horas (agendado) |
| **Integração com CRM** | Nativa via HTTP | Requer ETL + agendamento |
| **Custo computacional** | Baixo (modelo ~15k params) | Baixo mas com overhead de orquestração |
| **Frescor do dado** | Score reflete estado atual do cliente | Score pode ter horas/dias de atraso |
| **Complexidade operacional** | Baixa (1 serviço, 1 endpoint) | Alta (scheduler, fila, storage) |
| **Casos de uso primários** | CRM, alertas, decisões pontuais | Relatórios consolidados mensais |

> O modelo MLP com ~14.785 parâmetros é extremamente leve (< 1ms de inferência pura). Não há justificativa técnica para batch processing neste estágio — o custo de overhead superaria o benefício de throughput.

---

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENTE / CRM                           │
│                    (HTTP POST /predict)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ JSON payload
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI + Uvicorn                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LatencyMiddleware → log JSON + header X-Latency-Ms      │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌────────────────────┐   ┌────────────────────────────────┐   │
│  │  /health (GET)     │   │  /predict (POST)               │   │
│  │  → model_loaded    │   │  → Pydantic validation         │   │
│  │  → version         │   │  → Predictor.predict()         │   │
│  └────────────────────┘   └───────────────┬────────────────┘   │
└──────────────────────────────────────────-┼────────────────────┘
                                            │
                             ┌──────────────▼──────────────┐
                             │         Predictor           │
                             │  ┌────────────────────────┐ │
                             │  │  preprocessor.pkl      │ │
                             │  │  (ColumnTransformer)   │ │
                             │  └──────────┬─────────────┘ │
                             │             │ array (30,)    │
                             │  ┌──────────▼─────────────┐ │
                             │  │  mlp_model.pt          │ │
                             │  │  (PyTorch MLP)         │ │
                             │  └──────────┬─────────────┘ │
                             │             │ logit          │
                             │         Sigmoid             │
                             └──────────────┬──────────────┘
                                            │
                             ┌──────────────▼──────────────┐
                             │       PredictResponse       │
                             │  churn_probability: 0.73    │
                             │  churn_predicted: true      │
                             │  risk_tier: "high"          │
                             │  threshold_used: 0.5        │
                             └─────────────────────────────┘
```

---

## Componentes Detalhados

### 1. FastAPI + Uvicorn

**Arquivo:** [src/api/main.py](../src/api/main.py)

- Framework ASGI assíncrono com startup/shutdown via lifespan context manager
- Carregamento do `Predictor` no startup (lazy loading de artefatos)
- Documentação automática: `/docs` (Swagger UI) e `/redoc`
- Retorna `503 Service Unavailable` se modelo não carregado

### 2. LatencyMiddleware

**Arquivo:** [src/api/middleware.py](../src/api/middleware.py)

- Mede latência de ponta a ponta (request → response)
- Adiciona header `X-Latency-Ms` na resposta
- Emite log estruturado JSON com: `method`, `path`, `status_code`, `latency_ms`

### 3. Predictor (Inferência)

**Arquivo:** [src/api/predictor.py](../src/api/predictor.py)

- Desserializa `preprocessor.pkl` via joblib
- Carrega pesos do modelo `mlp_model.pt` via `torch.load`
- Executa `model.eval()` + `torch.no_grad()` para inferência determinística
- Aplica sigmoid ao logit de saída
- Mapeia probabilidade para `risk_tier`: `low` (≤0.3), `medium` (≤0.6), `high` (>0.6)

### 4. Validação de Schema

**Arquivo:** [src/api/schemas.py](../src/api/schemas.py)

- Pydantic v2 valida campos e tipos antes da inferência
- Retorna `422 Unprocessable Entity` com detalhes do erro para entradas inválidas
- Garante que apenas dados estruturalmente corretos chegam ao modelo

---

## Fluxo de Dados: Treinamento → Produção

```
dataset/
└── Dataset Telco-Customer-Churn.csv
         │
         ▼
  src/models/train.py
         │
         ├─── MLflow (experimentos, métricas, artefatos)
         │
         ├─── models/saved/preprocessor.pkl  ──┐
         │                                      │
         └─── models/saved/mlp_model.pt  ───────┤
                                                │
                                     src/api/predictor.py
                                                │
                                         POST /predict
```

---

## Configuração de Ambiente

### Variáveis de Ambiente

```env
# Caminhos dos artefatos (sobrescrevem defaults em src/utils/config.py)
MODEL_PATH=models/saved/mlp_model.pt
PREPROCESSOR_PATH=models/saved/preprocessor.pkl

# Servidor
HOST=0.0.0.0
PORT=8000

# Threshold de predição (default=0.5; use 0.1 para maximizar recall)
PREDICT_THRESHOLD=0.5
```

### Inicialização

```bash
# Desenvolvimento
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Produção
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Deploy no Render e consumo da API

### URL pública

- Base URL: https://techchallengefase1-qmaw.onrender.com
- Healthcheck: https://techchallengefase1-qmaw.onrender.com/health
- Endpoint de inferência: https://techchallengefase1-qmaw.onrender.com/predict

### Fluxo de build e deploy

1. O build do Render instala as dependências com `pip install -e .`.
2. Em seguida, executa `python scripts/train_for_deploy.py --no-cache` para gerar os artefatos `models/saved/preprocessor.pkl` e `models/saved/mlp_model.pt`.
3. O serviço sobe com Uvicorn e a aplicação carrega o modelo automaticamente no startup.
4. O endpoint `/health` confirma se o modelo foi carregado corretamente.

### Exemplo de consumo via cURL

```bash
curl -X POST https://techchallengefase1-qmaw.onrender.com/predict \
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
  }'
```

### Exemplo de consumo em Python

```python
import requests

payload = {
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
    "PaymentMethod": "Electronic check",
}

response = requests.post(
    "https://techchallengefase1-qmaw.onrender.com/predict",
    json=payload,
    timeout=30,
)
print(response.status_code)
print(response.json())
```

### Exemplo de resposta

```json
{
  "churn_probability": 0.7312,
  "churn_predicted": true,
  "threshold_used": 0.5,
  "risk_tier": "high"
}
```

## Topologia de Deploy Recomendada

### Estágio Atual (MVP / Pós-graduação)

```
[Laptop/VM única]
    └── uvicorn (1 worker, porta 8000)
```

### Próximo Estágio (Produção leve)

```
[Load Balancer / Nginx]
    ├── uvicorn worker 1 (porta 8001)
    ├── uvicorn worker 2 (porta 8002)
    └── uvicorn worker N (porta 800N)
```

### Estágio de Escala (Alta disponibilidade)

```
[Kubernetes Cluster]
    └── Deployment: churn-predictor
         ├── ReplicaSet: 3 pods mínimos
         ├── HPA: escala até 10 pods se CPU > 70%
         ├── ConfigMap: variáveis de ambiente
         └── PersistentVolume: artefatos do modelo
```

---

## Opção Alternativa: Batch Scoring

Embora não implementado neste projeto, o batch scoring seria preferível nos seguintes cenários:

| Quando usar Batch | Justificativa |
|-------------------|---------------|
| Base > 1 milhão de clientes diários | Overhead de HTTP por cliente se torna proibitivo |
| Relatórios consolidados noturnos | Não há requisito de tempo real |
| Retreinamento incremental | Processar novos dados em lote é mais eficiente |
| Integração com Data Warehouse | Pipelines Spark/dbt já existem na empresa |

**Implementação hipotética de batch:**

```python
# Exemplo: score diário via Spark ou pandas + joblib
df = pd.read_parquet("s3://telco/customers/2026-06-15.parquet")
X = preprocessor.transform(df[FEATURE_COLS])
scores = model(torch.tensor(X, dtype=torch.float32)).sigmoid().detach().numpy()
df["churn_probability"] = scores
df.to_parquet("s3://telco/scores/2026-06-15.parquet")
```

---

## SLOs (Service Level Objectives)

| Métrica | Target | Crítico |
|---------|--------|---------|
| **Latência p50** | < 50ms | > 200ms |
| **Latência p99** | < 200ms | > 500ms |
| **Disponibilidade** | 99.5% | < 99.0% |
| **Taxa de erro (5xx)** | < 0.1% | > 1.0% |
| **Taxa de validação inválida (422)** | < 5% | > 20% |

---

## Considerações de Segurança

- **Autenticação:** não implementada nesta versão (MVP). Em produção, adicionar API key via header ou OAuth2 Bearer Token.
- **Rate limiting:** recomendado em produção para evitar abuso (ex: slowapi + Redis).
- **Input sanitization:** validado via Pydantic — campos com tipos incorretos são rejeitados com 422.
- **Dados sensíveis:** `CustomerID` não é processado pelo modelo. Nunca logar dados pessoais em plaintext.
- **CORS:** configurar `allow_origins` explicitamente para domínios autorizados em produção.

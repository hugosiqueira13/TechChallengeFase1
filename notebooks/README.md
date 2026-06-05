# Notebooks — Guia de Uso

Este diretório contém os notebooks exploratórios do projeto **Telco Churn Prediction**. Cada notebook cobre uma etapa distinta da análise.

---

## Índice

| Notebook | Propósito | Dataset |
|---|---|---|
| [01_exploratory_analysis.ipynb](#01_exploratory_analysisipynb) | Análise exploratória introdutória e validação do pipeline de pré-processamento | Iris (sklearn) |
| [01_EDA_and_Baselines.ipynb](#01_eda_and_baselinesipynb) | EDA completa + modelos baseline com rastreamento MLflow | Telco Customer Churn |

---

## Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow
```

Execute os notebooks a partir da pasta `notebooks/`, ou certifique-se de que o caminho `../dataset/` seja resolvível (o notebook adiciona `..` ao `sys.path` automaticamente).

---

## 01_exploratory_analysis.ipynb

**Objetivo:** notebook curto de aquecimento que explora o dataset Iris e valida o pipeline de pré-processamento do projeto.

### O que o notebook faz

| Célula | Descrição |
|---|---|
| Carregamento | Carrega o dataset Iris via `sklearn.datasets.load_iris` e adiciona a coluna `target` com os nomes das espécies |
| Estatísticas descritivas | Exibe `df.describe()` e contagem de valores nulos — confirma que o dataset não possui missings |
| Pairplot | Gera um pairplot com `seaborn` colorido por espécie para visualizar a separabilidade das classes |
| Histogramas | Plota histogramas sobrepostos de cada feature (`sepal length`, `sepal width`, `petal length`, `petal width`) segmentados por espécie |

### Como executar

```bash
jupyter notebook 01_exploratory_analysis.ipynb
```

Não requer nenhum arquivo externo — o dataset é baixado via sklearn na primeira execução.

### Saídas esperadas

- Tabela com as 5 primeiras linhas do DataFrame
- Estatísticas descritivas (count, mean, std, min, max, quartis)
- Contagem de valores nulos (todos zero)
- Pairplot colorido por espécie (`setosa`, `versicolor`, `virginica`)
- 4 histogramas com distribuições por espécie

---

## 01_EDA_and_Baselines.ipynb

**Objetivo:** notebook principal da fase 1 do projeto. Realiza a análise exploratória completa do dataset Telco Customer Churn, define a estratégia de métricas com base nos dados, e treina/avalia modelos baseline rastreados com MLflow.

### Dataset

`dataset/Dataset Telco-Customer-Churn.csv` — ~7.043 clientes, 21 colunas, incluindo dados de perfil, tipo de contrato, serviços contratados e a variável-alvo `Churn`.

### Estrutura do notebook

#### Seção 1 — Introdução e Contexto de Negócio

Apresenta o problema: uma operadora de telecomunicações precisa prever quais clientes têm maior probabilidade de cancelar o serviço. O custo de aquisição de novos clientes é 5–7× maior que o custo de retenção.

#### Seção 2 — ML Canvas

Tabela de alinhamento técnico-negócio que define:
- **Decisão:** identificar clientes com alta probabilidade de churn nos próximos 30 dias
- **Ação:** campanhas personalizadas de retenção (descontos, upgrades, contato proativo)
- **SLOs propostos:** Recall ≥ 75%, ROC-AUC ≥ 0.80, latência de inferência ≤ 200 ms

#### Seção 3 — Carregamento dos Dados

Carrega o CSV e exibe shape, tipos de colunas e amostra.

```python
DATA_PATH = os.path.join('..', 'dataset', 'Dataset Telco-Customer-Churn.csv')
df_raw = pd.read_csv(DATA_PATH)
```

#### Seção 4 — Data Readiness Assessment

Avalia qualidade dos dados antes da modelagem:
- **Volume:** 7.043 linhas × 21 colunas
- **Duplicados:** nenhum
- **Missing values:** `TotalCharges` contém espaços em branco para clientes com `tenure=0` (tratado como `object`, não `NaN`) — convertida para `float64` com imputação pela mediana
- **Tabela resumo** com dtype, valores únicos, missings e exemplo de cada coluna

#### Seção 5 — Análise Exploratória de Dados (EDA)

**5.1 Variável Alvo**
- ~26,5% de churn → dataset moderadamente desbalanceado (razão ~2,7:1)
- Gráfico de barras + pizza com distribuição e percentuais

**5.2 Variáveis Numéricas** (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Estatísticas descritivas
- Histogramas sobrepostos por classe de Churn
- Boxplots por classe de Churn
- Detecção de outliers via IQR

**5.3 Variáveis Categóricas**
- Frequência e percentual de cada categoria
- Gráficos de barras horizontais para as 13 colunas categóricas

**5.4 Relação das Variáveis com Churn**
- Taxa de churn por categoria (vermelho = acima da média global)
- Variáveis destacadas: `Contract`, `InternetService`, `TechSupport`, `OnlineSecurity`, `PaymentMethod`, `PaperlessBilling`
- Violin plots das variáveis numéricas por classe de Churn
- Taxa de churn: clientes senior (~42%) vs não-senior (~24%)

**5.5 Matriz de Correlação**
- Heatmap triangular inferior entre variáveis numéricas + `SeniorCitizen` + `Churn_bin`
- `TotalCharges` altamente correlacionada com `tenure` (multicolinearidade esperada)

#### Seção 6 — Conclusões da EDA

Principais drivers de churn identificados:

| Driver | Evidência |
|---|---|
| Contrato mensal | Taxa de churn ~42% |
| Internet Fiber Optic | Taxa de churn ~42% |
| Ausência de TechSupport / OnlineSecurity | Forte correlação com churn |
| Tenure baixo (< 12 meses) | Período crítico de decisão |
| MonthlyCharges elevado | Sensibilidade ao preço |
| Pagamento por boleto eletrônico | Taxa de churn ~45% |

#### Seção 6.1 — Framework de Métricas

Justifica a escolha das métricas com base nos dados:
- **Accuracy descartada** — com 73/27% de split, um modelo que sempre prediz "No Churn" já atingiria ~73% de accuracy
- **Métricas prioritárias:** ROC-AUC (≥ 0.80) e PR-AUC (≥ 0.65)
- **Recall** recebe peso elevado pois o custo de um Falso Negativo (churner não identificado) é muito maior que o custo de um Falso Positivo

#### Seção 6.2 — Modelo de Custo de Negócio

Define a métrica financeira `biz_net_value`:

```
ARPU            = mediana de MonthlyCharges (derivada do dataset)
COST_FN         = ARPU × 12              (receita perdida por churner não detectado)
COST_FP         = ARPU × 0.30           (custo da campanha de retenção)
Receita Salva   = TP × 0.40 × ARPU × 12 (40% de sucesso na retenção)
Valor Líquido   = Receita Salva − Gasto Campanha
```

#### Seção 7 — Preparação para Modelagem

- Remove `customerID` (sem valor preditivo)
- Converte `Churn` para binário (`1` = Yes, `0` = No)
- Aplica `pd.get_dummies(drop_first=True)` nas variáveis categóricas
- Split treino/teste 80/20 com `stratify=y` para preservar a proporção de churn

#### Seção 8 — Configuração do MLflow

```python
MLFLOW_TRACKING_URI = f'sqlite:///{MLFLOW_DB_PATH}'   # backend SQLite (MLflow 3.x)
EXPERIMENT_NAME     = 'Telco-Churn-Baselines'
```

- Rastreia parâmetros, métricas técnicas e métricas de negócio por run
- Versiona o dataset via hash MD5
- Salva artefatos: confusion matrix, classification report, ROC curve, PR curve, feature importance, curva custo-benefício

#### Seção 9 — Baseline 1: DummyClassifier

- Estratégia `most_frequent` (sempre prediz "No Churn")
- Estabelece o **piso de desempenho**: Recall = 0, PR-AUC ≈ 0
- Run registrado no MLflow com todas as métricas técnicas e de negócio

#### Seção 10 — Baseline 2: Logistic Regression

- Pipeline `StandardScaler → LogisticRegression` (L2, C=1.0, solver=lbfgs)
- Análise de threshold ótimo: varre thresholds de 0.1 a 0.9 e identifica o que maximiza `biz_net_value`
- Curva custo-benefício por threshold + curva Precision-Recall por threshold
- Run registrado no MLflow com métricas para threshold 0.5 e threshold ótimo

#### Seção 11 — Comparação dos Baselines

Tabela e gráfico comparativo com todas as métricas e linhas de SLO:

| Modelo | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| DummyClassifier | ~0.50 | ~0.27 | 0.00 | 0.00 |
| Logistic Regression | ≥ 0.80 | ≥ 0.65 | — | — |

#### Seção 12 — Próximos Passos

Roteiro para os notebooks seguintes: Feature Engineering, MLP PyTorch, Hyperparameter Tuning, Deploy via FastAPI e Monitoramento com Evidently AI.

### Como executar

```bash
# 1. Certifique-se de que o dataset existe
ls ../dataset/Dataset\ Telco-Customer-Churn.csv

# 2. Abra o notebook
jupyter notebook 01_EDA_and_Baselines.ipynb

# 3. Execute todas as células em ordem (Kernel > Restart & Run All)
```

### Saídas geradas

| Artefato | Localização |
|---|---|
| Banco MLflow | `mlflow.db` (raiz do projeto) |
| Runs e modelos | `notebooks/mlruns/` |
| Confusion matrix | `mlruns/artifacts/*/confusion_matrix.png` |
| Classification report | `mlruns/artifacts/*/classification_report.txt` |
| ROC curve | `mlruns/artifacts/logistic_regression/roc_curve.png` |
| PR curve | `mlruns/artifacts/logistic_regression/pr_curve.png` |
| Feature importance | `mlruns/artifacts/logistic_regression/feature_importance.png` |
| Curva custo-benefício | `mlruns/artifacts/logistic_regression/cost_benefit_threshold.png` |

### Visualizar experimentos no MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acesse: http://localhost:5000
```

---

## Relação com o restante do projeto

```
ml-project/
├── dataset/                  # CSV do Telco Customer Churn
├── notebooks/                # ← você está aqui
│   ├── 01_exploratory_analysis.ipynb
│   └── 01_EDA_and_Baselines.ipynb
├── src/
│   ├── preprocessing/        # Pipeline de pré-processamento reutilizável
│   ├── models/               # Definição e treinamento do MLP (PyTorch)
│   └── api/                  # API FastAPI para servir predições
└── tests/                    # Testes unitários
```

O notebook `01_EDA_and_Baselines.ipynb` é o ponto de entrada analítico do projeto. As decisões tomadas aqui — escolha de métricas, tratamento de `TotalCharges`, encoding, split estratificado e modelo de custo de negócio — propagam-se para os módulos em `src/`.

# Model Card — Telco Customer Churn MLP Classifier

## Informações Gerais

| Campo | Valor |
|-------|-------|
| **Nome do Modelo** | Telco Churn MLP v1.0 |
| **Tipo de Tarefa** | Classificação binária (churn / não-churn) |
| **Framework** | PyTorch 2.2 |
| **Data de Treino** | Junho 2026 |
| **Versão do Dataset** | Telco Customer Churn (Kaggle / IBM Sample Data) |
| **Linguagem** | Python 3.10+ |

---

## Descrição do Modelo

Rede neural do tipo Multi-Layer Perceptron (MLP) treinada para prever a probabilidade de cancelamento (churn) de clientes de uma operadora de telecomunicações. A saída é uma probabilidade contínua em [0, 1], convertida em decisão binária via threshold configurável.

O modelo foi selecionado após comparação com quatro baselines (Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting) com base em ROC-AUC e valor de negócio líquido estimado.

---

## Arquitetura

```
Entrada (30 features após OneHotEncoding)
    │
    ├─ Linear(30 → 128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
    ├─ Linear(128 → 64) → BatchNorm1d(64)  → ReLU → Dropout(0.3)
    ├─ Linear(64 → 32)  → BatchNorm1d(32)  → ReLU → Dropout(0.3)
    └─ Linear(32 → 1)   → [logit]
                              │
                          Sigmoid → P(churn)
```

**Parâmetros treináveis:** ~14.785  
**Função de perda:** `BCEWithLogitsLoss` com `pos_weight` para compensar desbalanceamento de classes  
**Otimizador:** Adam (lr=1e-3, weight_decay=1e-4)  
**Scheduler:** ReduceLROnPlateau (fator=0.5, patience=5 épocas)  
**Early stopping:** patience=15 épocas monitorando val_roc_auc

---

## Dataset de Treinamento

| Atributo | Valor |
|----------|-------|
| **Registros** | 7.043 clientes |
| **Features** | 18 colunas brutas → 30 após encoding |
| **Taxa de churn** | 26,5% (positivo = churn) |
| **Split treino/teste** | 80% / 20% estratificado |
| **Período** | Dataset histórico sem data de corte explícita |

### Features de Entrada

**Numéricas (3):**
- `tenure` — meses como cliente (0–72)
- `MonthlyCharges` — cobrança mensal em R$ (>0)
- `TotalCharges` — cobrança total acumulada em R$ (≥0)

**Categóricas (15):**
- Dados demográficos: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Serviços de telefone: `PhoneService`, `MultipleLines`
- Serviços de internet: `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Conta: `Contract`, `PaperlessBilling`, `PaymentMethod`

### Pré-processamento

```
Numéricas → SimpleImputer(median) → StandardScaler
Categóricas → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
```

---

## Performance no Conjunto de Teste

### Métricas Quantitativas (threshold = 0.5)

| Métrica | MLP | Gradient Boosting | Random Forest | Logistic Reg. |
|---------|-----|-------------------|---------------|---------------|
| **ROC-AUC** | **0.8396** | 0.8433 | 0.8304 | 0.8432 |
| **PR-AUC** | 0.6267 | **0.6560** | 0.6322 | 0.6381 |
| **Recall** | 0.7834 | 0.5390 | 0.5411 | 0.5532 |
| **Precision** | 0.5279 | 0.6471 | 0.6399 | 0.6507 |
| **F1** | 0.6308 | 0.5880 | 0.5863 | 0.5973 |
| **Accuracy** | 0.7566 | 0.8006 | 0.7991 | 0.8048 |

### Métricas de Negócio (threshold otimizado = 0.1)

| Métrica | MLP |
|---------|-----|
| **Valor Líquido (R$)** | **R$ 109.813** |
| **Recall** | 98.93% |
| **ARPU utilizado** | R$ 70,35/mês |
| **Taxa de Retenção** | 40% |
| **Custo da campanha** | 30% do ARPU |
| **Meses retidos estimados** | 12 meses |

> O MLP foi selecionado como modelo final por maximizar o valor de negócio líquido, mesmo que o Gradient Boosting apresente ROC-AUC e PR-AUC ligeiramente superiores. Com threshold=0.1, o modelo captura ~99% dos churners, priorizando Recall para minimizar o custo de falsos negativos (custo de FN ≈ 40× maior que FP).

---

## Uso Pretendido

### Casos de Uso Suportados

- **Campanhas proativas de retenção:** identificar clientes de alto risco para abordagem antes do cancelamento
- **Segmentação por risco:** classificar clientes em tiers (`low`, `medium`, `high`) para priorização de recursos
- **Análise exploratória:** suporte a decisões estratégicas de produto e pricing
- **Score mensal em lote:** processar base completa de clientes para relatórios periódicos

### Usuários Finais Esperados

- Times de retenção e CRM
- Analistas de marketing e produto
- Sistemas automatizados de alertas via API

---

## Limitações

### Limitações Técnicas

1. **Distribuição de treino:** o modelo foi treinado em dados históricos de uma única operadora sem data de corte explícita. Mudanças de mercado, planos, regulamentações ou comportamento pós-COVID não estão representadas.

2. **Features ausentes no momento da inferência:** `TotalCharges` pode ser nulo para clientes novos (tenure=0). O imputer usa mediana como fallback, o que pode introduzir viés para clientes em início de contrato.

3. **Threshold fixo em produção:** a API usa threshold=0.5 por padrão. O threshold ótimo de negócio (0.1) deve ser configurado explicitamente. Deployments sem ajuste subestimam severamente o recall.

4. **Ausência de features temporais:** o modelo não captura tendências de comportamento ao longo do tempo (ex: aumento recente de chamadas ao suporte). Apenas o estado atual do cliente é considerado.

5. **Sem retraining automático:** o modelo não se atualiza com novos dados. É necessário retreinar manualmente ao detectar data drift.

### Limitações de Escopo

6. **Escopo geográfico desconhecido:** o dataset não especifica país/região. Padrões de churn podem variar significativamente por mercado.

7. **Clientes B2B excluídos:** o dataset representa clientes residenciais. O modelo não deve ser aplicado a segmentos corporativos sem revalidação.

---

## Potenciais Vieses

### Viés Demográfico

- **`gender`** está presente como feature. Embora o modelo possa usar essa informação para melhorar a acurácia preditiva, deve-se auditar regularmente se gênero está sendo usado de forma desproporcional ou discriminatória nas decisões de campanha.

- **`SeniorCitizen`** (0/1) está incluído. Ações de retenção podem ser mais ou menos agressivas para clientes idosos — isso deve ser revisado com equipes de ética e compliance.

### Viés de Recall vs. Precision

- Com threshold=0.1, o modelo tem recall alto (~99%) mas baixa precisão (~40%). Clientes sem intenção real de churn receberão ofertas de retenção desnecessárias (falsos positivos). Isso pode degradar a satisfação de clientes não-churners e elevar custos operacionais.

### Viés de Classe

- A taxa de churn de 26,5% cria desbalanceamento moderado. O `pos_weight` na função de perda compensa parcialmente, mas o modelo ainda pode subperformar em subpopulações com taxas de churn muito diferentes da média (ex: novos clientes com tenure < 3 meses).

---

## Cenários de Falha

### Cenário 1: Data Drift Silencioso

**Descrição:** A operadora lança novos planos ou muda a estrutura de preços. Valores de `MonthlyCharges` sobem 30% em média.  
**Impacto:** O StandardScaler aplica a média/desvio do treino. Features numéricas ficam fora da distribuição de treinamento.  
**Sintoma:** ROC-AUC cai gradualmente nas semanas seguintes. Falsos negativos aumentam.  
**Mitigação:** Monitorar PSI (Population Stability Index) mensalmente. Retreinar ao detectar PSI > 0.2.

### Cenário 2: Mudança de Produto/Política

**Descrição:** A empresa remove a opção de contrato "Month-to-month" ou adiciona um novo tipo de contrato.  
**Impacto:** O OneHotEncoder com `handle_unknown="ignore"` silenciosamente descarta a nova categoria. O modelo perde uma feature discriminante importante.  
**Sintoma:** Performance estável nas métricas (sem erro explícito) mas qualidade de predição degradada.  
**Mitigação:** Validar via Pandera schema (`src/schemas/telco.py`) toda entrada antes da inferência. Alertar quando categorias desconhecidas aparecerem.

### Cenário 3: Falha de Carregamento do Modelo

**Descrição:** O arquivo `models/saved/mlp_model.pt` ou `preprocessor.pkl` não existe, está corrompido, ou incompatível com a versão do PyTorch instalada.  
**Impacto:** A API retorna `503 Service Unavailable` para todas as predições.  
**Sintoma:** `/health` retorna `model_loaded: false`.  
**Mitigação:** Health check no startup, alertas automáticos ao detectar `model_loaded: false`, pipeline CI/CD que valida o modelo após cada treino.

### Cenário 4: Entradas Extremas / Out-of-Distribution

**Descrição:** Um cliente com `tenure=0` e `TotalCharges=null` (cliente recém-ativado) é submetido à API.  
**Impacto:** O imputer usa a mediana do treino (~29 meses e R$ 1.397). O score resultante é enganoso.  
**Sintoma:** Probabilidades de churn muito baixas ou muito altas para clientes novos sem histórico real.  
**Mitigação:** Adicionar flag `is_new_customer` no response quando `tenure < 3`. Documentar que o modelo é menos confiável para clientes com menos de 3 meses.

### Cenário 5: Alta Latência / Timeout

**Descrição:** Pico de carga com múltiplas requisições simultâneas em hardware com CPU apenas.  
**Impacto:** Latência p99 ultrapassa SLA de 500ms.  
**Sintoma:** Timeout errors no cliente. `X-Latency-Ms` header com valores acima de 500.  
**Mitigação:** Implementar cache de resultados por `customerID`, escalar horizontalmente via Docker/Kubernetes, ou migrar para batch scoring para grandes volumes.

### Cenário 6: Threshold Desatualizado

**Descrição:** O modelo foi retreinado mas o threshold de 0.5 (padrão da API) não foi reavaliado.  
**Impacto:** O threshold ótimo de negócio pode ter mudado. Campanhas de retenção perdem efetividade ou tornam-se excessivamente custosas.  
**Sintoma:** Queda no valor de negócio líquido observado após retreino.  
**Mitigação:** Automatizar o cálculo de threshold ótimo como parte do pipeline de treino e armazenar no MLflow. Injetar na configuração da API via variável de ambiente.

---

## Rastreabilidade e Reprodutibilidade

- Todos os experimentos são logados no MLflow com parâmetros, métricas, artefatos e hash de dataset
- Seed fixo: `random_state=42` para splits e inicialização do modelo
- Preprocessador e pesos do modelo são versionados juntos
- Dataset original em `dataset/Dataset Telco-Customer-Churn.csv`
- Notebooks de EDA e modelagem em `notebooks/`

---

## Informações de Contato

| Campo | Valor |
|-------|-------|
| **Autor** | hugosiqueira13 |
| **E-mail** | hugo.feagle15@gmail.com |
| **Repositório** | ml-project |
| **Versão** | 1.0.0 |

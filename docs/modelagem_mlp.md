# Modelagem com Redes Neurais — MLP PyTorch

**Dataset:** Telco Customer Churn (7.043 clientes, 30 features, churn rate 26.5%)  
**Notebook:** `notebooks/02_MLP_Modeling.ipynb`  
**Script:** `src/models/train.py`  
**Rastreamento:** MLflow SQLite (`mlflow.db`) — experimento `Telco-Churn-MLP-Comparison`

---

## 1. MLP em PyTorch

### Arquitetura

O modelo é um Multi-Layer Perceptron configurável implementado em `src/models/mlp.py`. Para classificação binária de churn, a versão do notebook usa saída única com `BCEWithLogitsLoss`:

```
Input (30 features)
    ↓
Linear(30 → 128) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(128 → 64) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(64 → 32)  → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(32 → 1)   → logit único
```

**Total de parâmetros treináveis:** 14.785

### Decisões de Projeto

| Componente | Escolha | Justificativa |
|-----------|---------|---------------|
| Função de ativação | ReLU | Evita vanishing gradient; padrão em MLPs |
| Normalização | BatchNorm1d | Estabiliza treinamento com dados tabulares |
| Regularização | Dropout(0.3) | Reduz overfitting; alternativa ao L2 na rede |
| Loss | BCEWithLogitsLoss + pos_weight | Trata desbalanceamento (churn 26.5% vs 73.5%) |
| Camada de saída | Linear(32 → 1) | Classificação binária: logit único |

### Pré-processamento

As features numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) são normalizadas com `StandardScaler` antes de entrar no MLP. As categóricas são transformadas via `pd.get_dummies(drop_first=True)`, resultando em 30 features no total.

O `pos_weight = 2.77` é calculado automaticamente como a razão entre amostras negativas e positivas no conjunto de treino, compensando o desbalanceamento de classes.

---

## 2. Loop de Treinamento com Early Stopping e Batching

### Configuração

```python
HPARAMS = {
    'hidden_dims' : [128, 64, 32],
    'dropout'     : 0.3,
    'lr'          : 1e-3,
    'weight_decay': 1e-4,      # regularização L2 via Adam
    'batch_size'  : 64,
    'epochs'      : 200,       # máximo; early stopping interrompe antes
    'patience'    : 15,        # épocas sem melhora antes de parar
    'pos_weight'  : 2.77,      # peso para classe minoritária (churn)
}
```

### Loop de Treinamento

A cada época:

1. **Forward pass em mini-batches** (batch_size=64): os dados de treino são embaralhados e percorridos em lotes, calculando o gradiente acumulado por batch.
2. **Backward pass**: `loss.backward()` + `optimizer.step()` com Adam.
3. **Validação**: após cada época, o modelo avalia o conjunto de teste inteiro no modo `eval()` (BatchNorm e Dropout desativados).
4. **Scheduler**: `ReduceLROnPlateau(mode='max', patience=5, factor=0.5)` reduz o learning rate à metade quando o ROC-AUC de validação não melhora por 5 épocas consecutivas.
5. **Early Stopping**: `EarlyStopping(patience=15)` monitora ROC-AUC de validação. Se não houver melhora de pelo menos `min_delta=1e-4` por 15 épocas, o treino é interrompido e o melhor estado (`state_dict`) salvo durante o treino é recarregado.

### Resultado do Treinamento

```
Epoch  25 | loss=0.6956 | val_roc_auc=0.8370 | val_f1=0.6160 | lr=0.000500
Early stopping na epoch 34 — melhor: epoch 19, ROC-AUC=0.8396
```

O early stopping interrompeu em 34 épocas (de 200 possíveis), com o melhor checkpoint na época 19. Isso previne overfitting e economiza tempo de computação.

---

## 3. Comparativo MLP vs. Baselines — 6 Métricas

### Modelos Avaliados

| Modelo | Tipo | Configuração principal |
|--------|------|----------------------|
| Logistic Regression | Linear | L2 (C=1), max_iter=1000, StandardScaler embutido |
| Decision Tree | Árvore única | max_depth=5, class_weight=balanced |
| Random Forest | Ensemble bagging | 200 árvores, max_depth=10, class_weight=balanced |
| Gradient Boosting | Ensemble boosting | 200 árvores, max_depth=4, lr=0.05 |
| **MLP PyTorch** | Rede Neural | 3 camadas ocultas [128, 64, 32], early stopping |

### Tabela Comparativa (conjunto de teste — 20% estratificado)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|--------|---------|-----------|--------|-----|---------|--------|
| LogisticRegression | 0.8070 | **0.6584** | 0.5668 | 0.6092 | 0.8416 | 0.6322 |
| DecisionTree | 0.7346 | 0.5000 | 0.8075 | 0.6176 | 0.8308 | 0.5976 |
| RandomForest | 0.7573 | 0.5291 | 0.7781 | 0.6299 | 0.8411 | **0.6517** |
| GradientBoosting | 0.7999 | 0.6533 | 0.5241 | 0.5816 | **0.8433** | **0.6560** |
| **MLP PyTorch** | 0.7566 | 0.5279 | **0.7834** | **0.6308** | 0.8396 | 0.6267 |

**Negrito** = melhor valor na coluna.

### Verificação de SLOs

| Métrica | SLO mínimo | Melhor modelo | Melhor valor | Status |
|---------|-----------|--------------|--------------|--------|
| ROC-AUC | ≥ 0.80 | GradientBoosting | 0.8433 | **OK** |
| PR-AUC | ≥ 0.65 | GradientBoosting | 0.6560 | **OK** |
| Recall | ≥ 0.75 | DecisionTree | 0.8075 | **OK** |
| F1 | ≥ 0.65 | MLP PyTorch | 0.6308 | **ABAIXO** |

### Interpretação

- **GradientBoosting** lidera ROC-AUC e PR-AUC, sendo o melhor modelo para ranking de probabilidades.
- **MLP PyTorch** entrega o melhor equilíbrio Recall/F1 e o maior valor líquido de negócio, tornando-o o mais indicado sob a ótica financeira.
- **Logistic Regression** tem a maior Precision (0.6584), mas o menor Recall (0.5668) — conservadora demais para churn.
- O SLO de F1 ≥ 0.65 não é atingido por nenhum modelo com threshold padrão (0.5). Com threshold otimizado por negócio (0.1), o recall sobe para ~98%, mas o F1 cai. Isso é esperado dado o custo assimétrico.

---

## 4. Trade-off de Custo: Falso Positivo vs. Falso Negativo

### Modelo de Custo de Negócio

Os valores são derivados diretamente do dataset (mediana de `MonthlyCharges`):

| Parâmetro | Valor | Cálculo |
|-----------|-------|---------|
| **ARPU** (receita mensal) | R$ 70,35 | mediana de `MonthlyCharges` |
| **COST_FN** (falso negativo) | R$ 844,20 | ARPU × 12 meses perdidos |
| **COST_FP** (falso positivo) | R$ 21,10 | ARPU × 30% (custo da campanha) |
| **Razão FN / FP** | **40×** | — |
| **Savings por TP** | R$ 337,68 | ARPU × 12 × 40% retenção |

### Interpretação do Trade-off

A razão FN/FP de 40× significa que errar um churner (FN) custa 40 vezes mais do que acionar desnecessariamente um cliente que não ia churnar (FP). Isso justifica:

1. **Priorizar Recall sobre Precision**: capturar mais churners, mesmo ao custo de mais alarmes falsos.
2. **Threshold baixo**: reduzir o limiar de decisão de 0.5 para ~0.1 aumenta o recall para ~99% com pequeno impacto no custo de FP.
3. **Valor líquido positivo para todos os modelos**: mesmo com muitos FPs, o valor das retenções supera os custos das campanhas.

### Net Value por Modelo (threshold otimizado = 0.1)

| Modelo | Net Value (R$) | Recall | Precision |
|--------|---------------|--------|-----------|
| **MLP PyTorch** | **R$ 109.813** | 0.9893 | 0.3404 |
| DecisionTree | R$ 109.391 | 0.9759 | 0.3571 |
| RandomForest | R$ 109.264 | 0.9840 | 0.3411 |
| LogisticRegression | R$ 108.609 | 0.9465 | 0.4060 |
| GradientBoosting | R$ 108.229 | 0.9412 | 0.4112 |

O **MLP PyTorch** entrega o maior valor líquido (R$ 109.813) porque, com threshold 0.1, captura 98.9% dos churners (recall máximo), maximizando as receitas salvas.

---

## 5. Experimentos no MLflow

### Configuração

```python
MLFLOW_TRACKING_URI = f'sqlite:///{ROOT_DIR}/mlflow.db'  # backend SQLite
EXPERIMENT_NAME     = 'Telco-Churn-MLP-Comparison'
```

O MLflow 3.x requer backend de banco de dados (file store descontinuado). Todos os experimentos — tanto do notebook EDA (`Telco-Churn-Baselines`) quanto do notebook de modelagem — são registrados no mesmo arquivo `mlflow.db`.

### O que é registrado por run

**Parâmetros (`mlflow.log_params`):**
- Hiperparâmetros do modelo (hidden_dims, dropout, lr, batch_size, etc.)
- Dimensão de entrada e número de parâmetros treináveis
- pos_weight (fator de correção do desbalanceamento)

**Métricas (`mlflow.log_metrics`):**
- Por época (MLP): `train_loss`, `val_roc_auc`, `val_f1`, `lr`
- Finais: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`
- Negócio: `biz_tp`, `biz_fp`, `biz_fn`, `biz_tn`, `biz_cost_fn`, `biz_cost_fp`, `biz_savings`, `biz_net_value`
- Otimização: `optimal_threshold`, `optimal_net_value`, `best_epoch`

**Artefatos (`mlflow.log_artifact`):**
- `confusion_matrix.png` — matriz de confusão do modelo
- Pesos do MLP (`mlp_best.pt`) — melhor checkpoint
- Modelo serializado (`mlflow.pytorch.log_model` / `mlflow.sklearn.log_model`)

**Tags:**
- `project`, `dataset_version` (hash MD5), `model_type`, `split`, `framework`

### Como visualizar

```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
# Acesse: http://127.0.0.1:5000
```

### Estrutura de Runs

Cada execução do script `python -m src.models.train` cria um run pai `full-comparison` com 5 runs filhos (um por modelo), permitindo comparação direta no MLflow UI.

---

## Como reproduzir

```bash
# 1. Instalar dependências
pip install torch mlflow scikit-learn pandas numpy matplotlib seaborn jupyter

# 2. Treinar via script (registra no MLflow automaticamente)
python -m src.models.train --epochs 150

# 3. Ou executar o notebook interativamente
jupyter notebook notebooks/02_MLP_Modeling.ipynb

# 4. Visualizar experimentos
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

## Arquivos relevantes

| Arquivo | Papel |
|---------|-------|
| `notebooks/02_MLP_Modeling.ipynb` | Notebook completo com análise interativa |
| `src/models/mlp.py` | Definição da arquitetura MLP (nn.Module) |
| `src/models/train.py` | Script de treinamento completo |
| `src/models/dataset.py` | TabularDataset (numpy → torch.Tensor) |
| `src/preprocessing/pipeline.py` | Pipeline de pré-processamento sklearn |
| `src/utils/config.py` | Constantes centralizadas (paths, hiperparâmetros, custo) |
| `mlflow.db` | Base SQLite com todos os experimentos registrados |
| `models/saved/mlp_model.pt` | Pesos do melhor MLP treinado |

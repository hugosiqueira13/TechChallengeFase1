# Arquitetura do Projeto

## Visão Geral

```
Request → FastAPI → Preprocessor (Scikit-Learn) → MLP (PyTorch) → Response
                                                       ↑
                                              Treinamento via MLflow
```

## Componentes

| Camada | Tecnologia | Responsabilidade |
|--------|-----------|-----------------|
| Pré-processamento | Scikit-Learn | Imputer, Scaler, OneHotEncoder |
| Modelo | PyTorch | MLP com BatchNorm + Dropout |
| Tracking | MLflow | Parâmetros, métricas, artefatos |
| API | FastAPI | Endpoint `/predict` + `/health` |

## Fluxo de Treinamento

1. `load_demo_data()` — carrega o dataset Iris
2. `fit_and_save_preprocessor()` — treina e serializa o ColumnTransformer
3. `train_baseline()` — RandomForest logado no MLflow (nested run)
4. `train_mlp()` — loop de treino PyTorch com logging step-a-step no MLflow
5. Melhor modelo salvo em `models/saved/mlp_model.pt`

## Fluxo de Inferência

1. `Predictor._load()` — carrega preprocessador (joblib) e pesos do modelo (torch)
2. `POST /predict` — recebe JSON, transforma, roda forward pass, retorna classe + probabilidades

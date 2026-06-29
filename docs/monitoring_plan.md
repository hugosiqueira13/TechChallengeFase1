# Plano de Monitoramento — Telco Churn Predictor

## Objetivos

Este documento define as métricas a monitorar, thresholds de alerta e o playbook de resposta a incidentes para o modelo de predição de churn em produção. O objetivo é detectar degradação de performance (modelo e sistema) antes que cause impacto de negócio.

---

## 1. Categorias de Métricas

O monitoramento é dividido em três camadas:

```
┌─────────────────────────────────────────────────────────────┐
│  Camada 1: Saúde do Sistema (infraestrutura + API)          │
├─────────────────────────────────────────────────────────────┤
│  Camada 2: Qualidade dos Dados (distribuição de entrada)    │
├─────────────────────────────────────────────────────────────┤
│  Camada 3: Performance do Modelo (predições + resultados)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Métricas por Camada

### Camada 1 — Saúde do Sistema

| Métrica | Fonte | Frequência | Alerta (Warning) | Alerta (Crítico) |
|---------|-------|-----------|-----------------|-----------------|
| **Latência p50** | `X-Latency-Ms` header / logs | Por requisição | > 100ms | > 300ms |
| **Latência p99** | `X-Latency-Ms` header / logs | Por requisição | > 300ms | > 500ms |
| **Taxa de erros 5xx** | Logs da API | Por minuto | > 0.5% | > 2% |
| **Taxa de erros 422** | Logs da API | Por minuto | > 10% | > 25% |
| **model_loaded** | `GET /health` | A cada 30s | `false` por > 1min | `false` por > 5min |
| **Throughput (RPS)** | Logs da API | Por minuto | Queda > 50% vs. média | Queda > 80% |
| **Uso de CPU** | Sistema operacional | A cada 60s | > 70% | > 90% |
| **Uso de memória** | Sistema operacional | A cada 60s | > 80% | > 95% |

### Camada 2 — Qualidade dos Dados (Data Drift)

Monitorar mensalmente via **PSI (Population Stability Index)** e KL-Divergence entre a distribuição de treino e as entradas de produção.

| Feature | Tipo | Métrica de Drift | Threshold Warning | Threshold Crítico |
|---------|------|-----------------|-------------------|-------------------|
| `tenure` | Numérica | PSI | > 0.10 | > 0.20 |
| `MonthlyCharges` | Numérica | PSI | > 0.10 | > 0.20 |
| `TotalCharges` | Numérica | PSI | > 0.10 | > 0.20 |
| `Contract` | Categórica | Chi-quadrado / proporção | Δ > 5pp | Δ > 15pp |
| `PaymentMethod` | Categórica | Chi-quadrado / proporção | Δ > 5pp | Δ > 15pp |
| `InternetService` | Categórica | Chi-quadrado / proporção | Δ > 5pp | Δ > 15pp |
| **Categorias desconhecidas** | - | Proporção | > 1% | > 5% |
| **Taxa de nulos em TotalCharges** | - | Proporção | > 5% | > 15% |

> **PSI Reference:** PSI < 0.10 = estável; 0.10–0.20 = monitorar; > 0.20 = retreinar.

### Camada 3 — Performance do Modelo

| Métrica | Frequência | Alerta (Warning) | Alerta (Crítico) |
|---------|-----------|-----------------|-----------------|
| **Distribuição de churn_probability** | Semanal | Média Δ > 5pp vs. treino | Média Δ > 15pp |
| **Taxa de predição positiva** | Semanal | Δ > 5pp vs. linha de base | Δ > 15pp |
| **ROC-AUC** (quando labels disponíveis) | Mensal | < 0.80 | < 0.75 |
| **Recall** (quando labels disponíveis) | Mensal | < 0.75 | < 0.65 |
| **F1-Score** (quando labels disponíveis) | Mensal | < 0.55 | < 0.45 |
| **Valor líquido realizado** | Mensal | Queda > 15% vs. esperado | Queda > 30% |

> **Nota sobre labels de produção:** A taxa de churn real só é conhecida após 30+ dias. Implementar pipeline de feedback que vincula predições ao status real do cliente (cancelled=1, active=0) para cálculo de métricas retrospectivas.

---

## 3. Coleta de Métricas

### Logs Estruturados (Já Implementado)

O `LatencyMiddleware` já emite logs JSON por requisição. Exemplo:

```json
{
  "timestamp": "2026-06-15T14:32:01.123Z",
  "level": "INFO",
  "method": "POST",
  "path": "/predict",
  "status_code": 200,
  "latency_ms": 43.7
}
```

### Extensão Recomendada: Métricas Estruturadas de Predição

Adicionar log de predição para análise posterior:

```json
{
  "timestamp": "2026-06-15T14:32:01.123Z",
  "event": "prediction",
  "churn_probability": 0.73,
  "churn_predicted": true,
  "risk_tier": "high",
  "threshold_used": 0.5,
  "tenure": 12,
  "contract": "Month-to-month",
  "monthly_charges": 75.5
}
```

### Dashboard Recomendado

Ferramentas compatíveis (em ordem de complexidade crescente):

| Ferramenta | Caso de Uso | Complexidade |
|------------|-------------|-------------|
| **MLflow** | Métricas de treino, comparação de experimentos | Já integrado |
| **Grafana + Prometheus** | Latência, RPS, erros em tempo real | Média |
| **Evidently AI** | Data drift, model drift com relatórios automáticos | Média |
| **WhyLabs / Arize** | Monitoramento ML end-to-end gerenciado | Alta / SaaS |

---

## 4. Playbook de Resposta a Incidentes

### INC-01: Modelo Não Carregado (`model_loaded: false`)

**Detecção:** `GET /health` retorna `model_loaded: false` por mais de 1 minuto.  
**Impacto:** 100% das requisições `/predict` retornam `503 Service Unavailable`.

**Passos de Investigação:**
1. Verificar existência dos artefatos:
   ```bash
   ls -lh models/saved/mlp_model.pt models/saved/preprocessor.pkl
   ```
2. Checar logs de startup do Uvicorn por exceções de carregamento.
3. Validar compatibilidade de versão do PyTorch:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

**Resolução:**
- Se artefatos ausentes: restaurar do último backup ou retreinar (`python -m src.models.train`).
- Se artefatos corrompidos: restaurar backup. Implementar checksum (MD5/SHA256) no pipeline de treino.
- Se incompatibilidade de versão: fixar versão no `pyproject.toml` e reconstruir ambiente.

**Prevenção:** Health check automatizado a cada 30s com alerta imediato.

---

### INC-02: Alta Latência (p99 > 500ms)

**Detecção:** Header `X-Latency-Ms` > 500ms em mais de 1% das requisições nos últimos 5 minutos.  
**Impacto:** Degradação de experiência no CRM; possível timeout de cliente.

**Passos de Investigação:**
1. Verificar CPU e memória do servidor.
2. Identificar padrão no log: endpoint, horário, payload específico.
3. Verificar se há pico de throughput incomum.
4. Checar se o modelo está sendo carregado a cada request (bug de lazy loading).

**Resolução:**
- Pico de CPU: escalar horizontalmente (adicionar workers ou pods).
- Carregamento repetido: garantir que `Predictor` é singleton (instanciado no lifespan, não por request).
- Payload anômalo: investigar se há features com alto custo de encoding.

**Prevenção:** Load test com locust antes de cada deploy. SLO de p99 < 200ms.

---

### INC-03: Aumento de Erros de Validação (422 > 20%)

**Detecção:** Taxa de respostas 422 ultrapassa 20% nos últimos 10 minutos.  
**Impacto:** Cliente recebendo erros em vez de predições. Possível bug no sistema chamador.

**Passos de Investigação:**
1. Analisar logs de erros 422 para identificar padrão no campo inválido.
2. Verificar se houve deploy recente no sistema chamador (CRM) que mudou o payload.
3. Verificar se houve mudança de schema no lado da API.

**Resolução:**
- Se bug no chamador: notificar time responsável com exemplo de payload válido.
- Se mudança de schema na API: versionar a API (`/v1/predict`, `/v2/predict`) e manter compatibilidade retroativa.
- Se novo valor categórico: atualizar enum no Pydantic schema e retreinar se necessário.

**Prevenção:** Testes de contrato (contract testing) entre API e sistemas chamadores. Versionamento semântico.

---

### INC-04: Data Drift Detectado (PSI > 0.20)

**Detecção:** Job mensal de análise de drift reporta PSI > 0.20 em features numéricas ou mudança > 15pp em categóricas.  
**Impacto:** Predições potencialmente desatualizadas. Performance do modelo degradada silenciosamente.

**Passos de Investigação:**
1. Identificar quais features estão em drift.
2. Investigar causa raiz: mudança de produto, sazonalidade, nova regulamentação.
3. Comparar distribuição de `churn_probability` atual vs. linha de base.
4. Se labels disponíveis: calcular ROC-AUC no período recente.

**Resolução:**
- Drift leve (PSI 0.10–0.20): monitorar semanalmente, sem ação imediata.
- Drift moderado (PSI 0.20–0.30): agendar retreino nas próximas 2 semanas.
- Drift severo (PSI > 0.30): retreino imediato com dados recentes:
  ```bash
  python -m src.models.train --epochs 200
  ```
- Após retreino: revalidar métricas e recalcular threshold ótimo.

**Prevenção:** Pipeline mensal automatizado de análise de drift. Alerta via e-mail/Slack ao time de ML.

---

### INC-05: Queda de Performance do Modelo (ROC-AUC < 0.75)

**Detecção:** Avaliação mensal com labels reais retorna ROC-AUC < 0.75.  
**Impacto:** Campanhas de retenção ineficientes. Custo de campanha sem retorno proporcional.

**Passos de Investigação:**
1. Verificar se o drift de dados foi corrigido antes da avaliação.
2. Analisar se há segmentos específicos com performance muito baixa (ex: clientes novos, contrato anual).
3. Revisar se o threshold de produção está calibrado para o modelo atual.
4. Comparar com performance dos baselines para ver se o problema é sistêmico.

**Resolução:**
- Retreinar com dados mais recentes (últimos 12 meses).
- Avaliar feature engineering adicional (ex: delta de `MonthlyCharges` vs. média do segmento).
- Considerar modelos alternativos (XGBoost, LightGBM) se MLP não convergir adequadamente.
- Recalcular threshold ótimo após retreino.

**Prevenção:** Feedback loop com dados reais de churn. Revisão trimestral de performance com stakeholders.

---

## 5. Calendário de Revisões

| Frequência | Atividade | Responsável |
|------------|-----------|-------------|
| **Diária** | Revisar logs de latência e erros | On-call |
| **Semanal** | Analisar distribuição de scores e taxa de positivos | Time de ML |
| **Mensal** | Calcular PSI para todas as features | Time de ML |
| **Mensal** | Calcular métricas com labels reais (se disponíveis) | Time de ML |
| **Trimestral** | Revisão de performance com time de negócio | ML + Produto |
| **Semestral** | Revisão completa de bias e fairness | ML + Ética/Compliance |

---

## 6. Artefatos de Monitoramento

| Artefato | Localização | Descrição |
|----------|-------------|-----------|
| Logs da API | `stdout` (JSON estruturado) | Latência, status code, path por request |
| Experimentos MLflow | `mlruns/` ou `mlflow.db` | Métricas de treino e baselines |
| Relatório de drift | A implementar: `reports/drift_YYYY-MM.html` | PSI por feature (Evidently AI) |
| Score history | A implementar: `data/predictions/YYYY-MM-DD.parquet` | Histórico de predições para análise |
| Model registry | MLflow Model Registry | Versões de modelo com stage (Staging/Production) |

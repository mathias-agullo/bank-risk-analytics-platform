# Bank Risk Analytics Platform

Plataforma de análisis de riesgo crediticio y fraude bancario con datos macroeconómicos reales de Chile. Construida para simular el flujo de decisiones de crédito de un banco retail, desde la ingesta de datos hasta la generación de reportes regulatorios con IA.

---

## Problema de negocio

Los bancos necesitan decidir si otorgar o rechazar créditos en segundos, considerando no solo el perfil del cliente sino también el contexto macroeconómico actual. Un error hacia arriba (aprobar a quien no paga) genera pérdidas directas. Un error hacia abajo (rechazar a quien sí pagaría) pierde negocio.

Este proyecto modela ese problema de forma completa: desde los datos hasta la decisión final, incluyendo explicabilidad regulatoria y detección de fraude.

---

## Arquitectura

```
Datos de clientes (simulados)
        │
        ├── Macro real ──── BCCh API (TPM, UF, USD/CLP, desempleo)
        │                   yfinance (S&P500, Cobre, USD/CLP)
        │
        ├── Forecasting ─── ARIMA(1,1,0) → proyección 6 meses
        │
        ├── Preprocessing ─ Feature engineering + interacciones macro×cliente
        │
        ├── Modelo crédito ─ Logistic Regression + Gradient Boosting (ensemble)
        │                    CalibratedClassifierCV + umbral óptimo F2-score
        │                    Walk-forward temporal validation
        │
        ├── Explainability ─ SHAP TreeExplainer → factores por cliente
        │
        ├── Fraude ──────── Isolation Forest + reglas de negocio
        │
        └── Reportes IA ─── Groq (llama-3.3-70b) → narrativa regulatoria
                            Dashboard Streamlit
```

---

## Metodología

### Modelo de riesgo crediticio

Se entrena un ensemble de dos modelos con pesos distintos según su fortaleza:

| Modelo | Peso | Por qué |
|--------|------|---------|
| Logistic Regression | 40% | Interpretable, exigido por reguladores |
| Gradient Boosting (300 árboles) | 60% | Mayor poder predictivo |

Ambos modelos se calibran con `CalibratedClassifierCV(cv=5, method="isotonic")` para garantizar que las probabilidades de default (PD) sean reales y no solo rankings.

**Umbral de decisión:** Se optimiza con F2-score en lugar de accuracy, penalizando más los falsos negativos (no detectar un default es más costoso que rechazar un buen cliente).

**Validación temporal:** Walk-forward con `TimeSeriesSplit(n_splits=5, gap=30)` para simular validación en producción real, respetando el orden cronológico de los datos.

### Features de interacción macro×cliente

Las variables macroeconómicas por sí solas son constantes para todos los clientes, por lo que su importancia en el modelo sería cero. Se crean 8 features de interacción que capturan cómo el contexto macro afecta de forma diferente a cada cliente:

```python
unemp_x_employment   # desempleo golpea más a quienes ya están sin trabajo
rate_x_debt          # alza de tasas duele más a quienes tienen más deuda
usdclp_x_income      # depreciación del peso afecta más a ingresos bajos
stress_x_missed      # stress macro amplifica historial de pagos atrasados
```

### Explainabilidad regulatoria (SHAP)

Para cada cliente se calculan los top factores que aumentan o reducen su probabilidad de default usando SHAP (Shapley Additive Explanations). Esto cumple con requerimientos regulatorios de transparencia: un banco no puede rechazar un crédito sin documentar la razón.

### Detección de fraude

Isolation Forest (algoritmo de anomalías no supervisado) sobre features de transacciones: monto, hora, canal de pago, país, categoría de comercio. Las alertas se clasifican en cuatro niveles: normal, sospechoso, alto riesgo, crítico.

---

## Resultados

| Métrica | Valor |
|---------|-------|
| AUC-ROC ensemble | **0.756** |
| AUC-PR | 0.604 |
| Walk-forward AUC (5 folds) | **0.727 ± 0.045** |
| Tiempo de ejecución pipeline | ~15 segundos |

**Distribución de riesgo del portafolio:**
- LOW: 56% de clientes
- MEDIUM: 28.5%
- HIGH: 15.5%

**Datos macroeconómicos reales (al momento del último run):**
- TPM Banco Central: 4.50%
- USD/CLP: $895.28
- Cobre: $5.76/lb (+37.9% YTD)
- S&P 500: +24.3% YTD

---

## Estructura del proyecto

```
bank-risk-analytics-platform/
├── main.py                          # Orquestador del pipeline
├── config/
│   └── config.yaml                  # Escenarios y parámetros
├── data/
│   └── simulated/generate_data.py   # Generación de cartera sintética
├── src/
│   ├── ingestion/
│   │   ├── bcch_data.py             # API Banco Central de Chile
│   │   └── market_data.py           # yfinance / Stooq
│   ├── macro/
│   │   └── analyzer.py              # Monitor de riesgo de mercado
│   ├── forecasting/
│   │   └── forecaster.py            # ARIMA forecasting macroeconómico
│   ├── preprocessing/
│   │   └── cleaner.py               # Feature engineering
│   ├── credit_risk/
│   │   ├── model.py                 # Ensemble LR + GBM
│   │   └── explainer.py             # SHAP explainability
│   ├── fraud/
│   │   └── detector.py              # Isolation Forest
│   └── ai/
│       └── reporter.py              # Reportes narrativos con Groq
├── dashboards/
│   └── app.py                       # Dashboard Streamlit
└── outputs/
    ├── clients_scored.csv           # Resultado por cliente
    ├── transactions_scored.csv      # Resultado por transacción
    ├── feature_importance.csv       # Importancia de variables
    └── summary.json                 # Resumen ejecutivo
```

---

## Cómo correr el proyecto

### Requisitos

```bash
pip install -r requirements.txt
```

Requiere Python 3.10+. Las dependencias principales son: `scikit-learn`, `shap`, `yfinance`, `bcchapi`, `groq`, `streamlit`, `statsmodels`.

### Pipeline completo (terminal)

```bash
# Escenario normal
python main.py

# Escenario de crisis económica
python main.py --scenario crisis

# Alza de tasas
python main.py --scenario rate_hike

# Reentrenar modelos desde cero
python main.py --retrain
```

### Dashboard interactivo

```bash
streamlit run dashboards/app.py
```

### Outputs generados

Después de correr el pipeline, se generan en `outputs/`:

- `clients_scored.csv` — cada cliente con su PD, credit score, nivel de riesgo y decisión
- `transactions_scored.csv` — transacciones con fraud score y categoría de alerta
- `feature_importance.csv` — importancia de cada variable en el modelo
- `summary.json` — resumen ejecutivo del portafolio

---

## Escenarios económicos

El sistema permite simular tres contextos macroeconómicos distintos. En cada escenario, los shocks se aplican de forma estocástica a nivel de cliente individual — no todos reciben el mismo impacto.

| Escenario | Desempleo | Tasas | Shocks por cliente |
|-----------|-----------|-------|-------------------|
| Normal | 8.5% | 4.5% | — |
| Crisis | +2.5pp | +1.5pp | caída de ingresos, stress de pagos, drenaje de ahorros |
| Alza de tasas | +0.5pp | +2.0pp | aumento ratio de deuda, stress de pagos |

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Modelamiento | scikit-learn (LR, GBM, Isolation Forest) |
| Calibración | CalibratedClassifierCV |
| Explainability | SHAP (TreeExplainer) |
| Forecasting | statsmodels (ARIMA) |
| Datos macro Chile | bcchapi (API oficial BCCh) |
| Datos mercado | yfinance, pandas-datareader |
| IA generativa | Groq API — llama-3.3-70b-versatile |
| Dashboard | Streamlit + Plotly |
| Lenguaje | Python 3.10+ |

---

## Decisiones de diseño relevantes

**¿Por qué ensemble y no un solo modelo?**
Logistic Regression es requerimiento regulatorio en banca (interpretable, auditado). GBM captura relaciones no lineales que LR no puede. El ensemble captura lo mejor de ambos.

**¿Por qué F2-score para el umbral?**
En crédito, un falso negativo (no detectar un default) es más costoso que un falso positivo (rechazar un cliente bueno). F2 penaliza más los falsos negativos que F1.

**¿Por qué walk-forward y no cross-validation estándar?**
Los datos financieros tienen dependencia temporal. Cross-validation estándar filtra información del futuro al pasado. Walk-forward respeta el orden cronológico, simulando validación en producción real.

**¿Por qué modelos de crédito y fraude separados?**
En banca real, riesgo crediticio y fraude son unidades con gobernanza distinta, métricas distintas y regulación distinta. Mezclar los modelos contamina ambas métricas. Se mantienen separados y se integran solo en la capa de decisión final.

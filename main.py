"""
Bank Risk Analytics Platform — Pipeline Principal
==================================================
Flujo: datos → macro → forecast → credit risk → fraud → AI reports

Uso:
    python main.py                              # escenario normal, llama3.2:3b
    python main.py --scenario crisis            # crisis economica
    python main.py --scenario rate_hike         # alza de tasas
    python main.py --model mistral:7b           # cambiar modelo Ollama
    python main.py --client-id 42               # analisis de cliente especifico

Requiere Ollama corriendo: https://ollama.com
    ollama pull llama3.2:3b     # rapido (~2GB)
    ollama pull mistral:7b      # mejor calidad (~4GB)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # carga variables desde .env

import numpy as np

# Windows UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import yaml

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from data.simulated.generate_data import run as generate_data
from src.ingestion.market_data import fetch_all
from src.macro.analyzer import analyze as analyze_macro
from src.forecasting.forecaster import forecast_macro
from src.ingestion.bcch_data import fetch_all_bcch, extract_macro_values
from src.ingestion.mindicador_data import fetch_mindicador, extract_macro_values_mindicador
from src.preprocessing.cleaner import clean_clients, add_macro_features, get_feature_columns
from src.credit_risk.model import CreditRiskModel
from src.credit_risk.explainer import CreditRiskExplainer, format_shap_for_prompt
from src.fraud.detector import FraudDetector
from src.ai.reporter import batch_generate_reports

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_or_generate_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients_path = PROCESSED_DIR / "clients.csv"
    txns_path = PROCESSED_DIR / "transactions.csv"

    if clients_path.exists() and txns_path.exists():
        print("📂 Cargando datos existentes...")
        clients = pd.read_csv(clients_path)
        transactions = pd.read_csv(txns_path)
    else:
        print("🔧 Generando datos simulados...")
        data_cfg = config["data"]
        clients, transactions = generate_data(
            n_clients=data_cfg["n_clients"],
            n_transactions=data_cfg["n_transactions"],
            random_state=data_cfg["random_state"],
        )
    return clients, transactions


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_pipeline(scenario: str = "normal", target_client_id: int | None = None, model: str | None = None, retrain: bool = False) -> dict:
    start = time.time()
    config = load_config()
    ollama_model = model or config["ai"]["model"]

    print_section("🏦 BANK RISK ANALYTICS PLATFORM")
    print(f"  Escenario: {config['scenarios'][scenario]['label']}")
    print(f"  Modelo IA: {ollama_model} (Ollama)")

    # ── 1. DATOS ──────────────────────────────────────────────────────────────
    print_section("1️⃣  DATA LAYER")
    clients, transactions = load_or_generate_data(config)
    print(f"  Clientes: {len(clients):,} | Transacciones: {len(transactions):,}")
    print(f"  Default rate base: {clients['default'].mean():.1%}")

    # ── 2. MACRO MONITOR ─────────────────────────────────────────────────────
    print_section("2️⃣  MACRO MONITOR")
    print("  Descargando datos de mercado...")
    market_data = fetch_all(config)
    macro_analysis = analyze_macro(market_data)
    labels = {"ipsa": "IPSA    ", "sp500": "S&P500  ", "usdclp": "USD/CLP ", "copper": "Cobre   "}
    print(f"  Riesgo de mercado: {macro_analysis['market_risk'].upper()}")
    for name, data in macro_analysis.items():
        if isinstance(data, dict) and "return_ytd" in data:
            label = labels.get(name, name.upper())
            print(f"  {label}: precio={data['last_price']:>10} | retorno={data['return_ytd']:>+7.1%} | tendencia={data['trend']:<7} | alerta={data['alert']}")

    # ── 3. FORECASTING MACROECONÓMICO ────────────────────────────────────────
    print_section("3️⃣  FORECASTING MACROECONÓMICO")

    # 1º — Banco Central de Chile (si hay credenciales)
    print("  Intentando API Banco Central de Chile...")
    bcch_raw = fetch_all_bcch()
    bcch_values = extract_macro_values(bcch_raw) if bcch_raw else None

    if bcch_values:
        print("  BCCh conectado — fuente: REAL")
        if "tpm_current" in bcch_values:
            print(f"  TPM actual (BCCh):   {bcch_values['tpm_current']:.2f}%")
        if "uf_value" in bcch_values:
            print(f"  UF actual (BCCh):    ${bcch_values['uf_value']:,.2f}")
        if "usdclp_official" in bcch_values:
            print(f"  USD/CLP obs (BCCh):  ${bcch_values['usdclp_official']:,.2f}")
    else:
        # 2º — mindicador.cl (gratuito, sin credenciales)
        print("  BCCh no disponible — intentando mindicador.cl...")
        mini_raw = fetch_mindicador()
        if mini_raw:
            bcch_values = extract_macro_values_mindicador(mini_raw)
            print("  mindicador.cl conectado — fuente: PÚBLICO")
            if "tpm_current" in bcch_values:
                print(f"  TPM actual:          {bcch_values['tpm_current']:.2f}%")
            if "uf_value" in bcch_values:
                print(f"  UF actual:           ${bcch_values['uf_value']:,.2f}")
            if "usdclp_official" in bcch_values:
                print(f"  USD/CLP:             ${bcch_values['usdclp_official']:,.2f}")
        else:
            print("  mindicador.cl no disponible — usando historial aproximado")

    scenario_cfg = config["scenarios"][scenario]
    scenario_deltas = {
        "unemployment_delta": scenario_cfg.get("unemployment_delta", 0),
        "rate_delta": scenario_cfg.get("rate_delta", 0),
    }
    macro_forecast = forecast_macro(
        horizon=config["forecasting"]["horizon_months"],
        scenario_deltas=scenario_deltas,
        bcch_values=bcch_values,
    )
    print(f"  Desempleo proyectado: {macro_forecast['unemployment_forecast']:.1f}%")
    print(f"  Tasa proyectada:      {macro_forecast['rate_forecast']:.1f}%")
    print(f"  Inflacion proyectada: {macro_forecast['inflation_forecast']:.1f}%")
    print(f"  Escenario de tasas:   {macro_forecast['rate_scenario'].upper()}")
    print(f"  Stress macro:         {macro_forecast['macro_stress'].upper()}")

    # ── 4. PREPROCESSING ─────────────────────────────────────────────────────
    print_section("4️⃣  PREPROCESSING")
    clients_clean = clean_clients(clients)
    clients_clean = add_macro_features(clients_clean, macro_forecast, macro_analysis)

    # ── Shocks de escenario sobre variables individuales ─────────────────────
    # Afectan las features con mayor importancia en el modelo:
    # income_log (24.9%), credit_maturity (24.8%), payment_history_score (14.2%),
    # debt_ratio (8.6%), loan_to_income (7.1%), savings_ratio (5.1%)
    income_drop       = scenario_cfg.get("income_drop", 0.0)
    debt_increase     = scenario_cfg.get("debt_ratio_increase", 0.0)
    layoff_prob       = scenario_cfg.get("layoff_prob", 0.0)
    payment_stress    = scenario_cfg.get("payment_stress", 0.0)
    savings_drain     = scenario_cfg.get("savings_drain", 0.0)

    rng_scenario = np.random.default_rng(config["data"]["random_state"] + 1)
    n = len(clients_clean)

    if income_drop > 0:
        # Caída de ingreso real distribuida heterogéneamente:
        # los de menor ingreso sufren más (regresivo, como en crisis reales)
        drop_factor = rng_scenario.uniform(income_drop * 0.5, income_drop * 1.5, n)
        clients_clean["income_log"] = (clients_clean["income_log"] * (1 - drop_factor)).clip(0)
        clients_clean["loan_to_income"] = (clients_clean["loan_amount"] / np.expm1(clients_clean["income_log"]).clip(1)).clip(0, 20)
        print(f"  Shock ingreso real: -{income_drop:.0%} promedio")

    if debt_increase > 0:
        # Alza de tasas encarece deudas variables (hipotecas, consumo)
        clients_clean["debt_ratio"] = (clients_clean["debt_ratio"] + debt_increase * rng_scenario.uniform(0.5, 1.5, n)).clip(0, 0.98)
        clients_clean["financial_burden"] = clients_clean["debt_ratio"] * clients_clean["loan_to_income"]
        print(f"  Shock debt ratio: +{debt_increase:.0%} promedio")

    if layoff_prob > 0:
        # Fracción de empleados pierde trabajo → employment_code = unemployed (3)
        employed_mask = clients_clean["employment_status"] == "employed"
        layoffs = rng_scenario.random(n) < layoff_prob
        clients_clean.loc[employed_mask & layoffs, "employment_code"] = 3
        n_laid_off = int((employed_mask & layoffs).sum())
        print(f"  Shock desempleo: {n_laid_off} empleados despedidos ({n_laid_off/n:.1%})")

    if payment_stress > 0:
        # Atrasos adicionales estocásticos — peor para quienes ya tienen historial corto
        extra_missed = rng_scenario.binomial(3, payment_stress, n)
        clients_clean["missed_payments"] = (clients_clean["missed_payments"] + extra_missed).clip(0, 12)
        clients_clean["payment_history_score"] = 1 - (clients_clean["missed_payments"] / 12).clip(0, 1)
        print(f"  Shock pagos: +{extra_missed.mean():.1f} atrasos promedio por cliente")

    if savings_drain > 0:
        clients_clean["savings_ratio"] = (clients_clean["savings_ratio"] * (1 - savings_drain)).clip(0, 1)
        print(f"  Shock ahorro: -{savings_drain:.0%} sobre savings_ratio")

    # Recalcular rule_flag con los nuevos valores
    clients_clean["rule_flag"] = (
        (clients_clean["debt_ratio"] > 0.70)
        | (clients_clean["missed_payments"] >= 3)
        | (clients_clean["employment_code"] == 3)
    ).astype(int)

    feature_cols = get_feature_columns()
    print(f"  Features: {len(feature_cols)} | Registros: {len(clients_clean):,}")

    # ── 5. CREDIT RISK MODEL ─────────────────────────────────────────────────
    print_section("5️⃣  CREDIT RISK MODEL")
    credit_model = CreditRiskModel(config)

    if retrain or not credit_model.is_trained():
        if not retrain:
            print("  Modelos no encontrados — entrenando por primera vez...")
        else:
            print("  Reentrenando modelos...")
        metrics = credit_model.train(clients_clean, feature_cols)

        lr_m   = metrics["logistic_regression"]
        gbm_m  = metrics["gradient_boosting"]
        ens_m  = metrics["ensemble"]
        wf_m   = metrics["walk_forward"]
        print(f"  Logistic Regression → AUC-ROC: {lr_m['auc_roc']:.4f} | AUC-PR: {lr_m['auc_pr']:.4f}")
        print(f"  Gradient Boosting   → AUC-ROC: {gbm_m['auc_roc']:.4f} | AUC-PR: {gbm_m['auc_pr']:.4f}")
        print(f"  Ensemble            → AUC-ROC: {ens_m['auc_roc']:.4f} | Umbral optimo: {ens_m['optimal_threshold']:.3f}")
        if wf_m["auc_mean"]:
            print(f"  Walk-forward (5 folds): {wf_m['auc_mean']:.4f} +/- {wf_m['auc_std']:.4f}")

        # SHAP — solo al entrenar
        print("  Calculando SHAP explainer...")
        try:
            explainer = CreditRiskExplainer()
            X_background = clients_clean[feature_cols].fillna(0).sample(200, random_state=42)
            explainer.fit(credit_model.pipeline_gbm, X_background, feature_cols)
            print("  SHAP explainer guardado")
        except Exception as e:
            print(f"  SHAP no disponible: {e} (instala: pip install shap)")
    else:
        print("  Cargando modelos existentes (usa --retrain para reentrenar)")
        credit_model.load()
        meta = credit_model.get_metadata()
        print(f"  Entrenado: {meta.get('trained_at', 'N/A')[:10]}")
        print(f"  AUC-ROC ensemble: {meta.get('auc_roc_ensemble', 'N/A')}")
        wf_mean = meta.get('walk_forward_mean')
        if wf_mean:
            print(f"  Walk-forward:     {wf_mean:.4f} +/- {meta.get('walk_forward_std', 0):.4f}")

    clients_scored = credit_model.predict(clients_clean)

    risk_dist = clients_scored["risk_level"].value_counts()
    print(f"\n  Distribucion de riesgo:")
    for level in ["low", "medium", "high"]:
        count = risk_dist.get(level, 0)
        pct   = count / len(clients_scored)
        print(f"    {level.upper():8s}: {count:5,} clientes ({pct:.1%})")

    # ── 6. FRAUD DETECTION ───────────────────────────────────────────────────
    print_section("6️⃣  FRAUD DETECTION")
    fraud_detector = FraudDetector(config)
    if retrain or not fraud_detector.is_trained():
        if not retrain:
            print("  Modelo no encontrado — entrenando por primera vez...")
        else:
            print("  Reentrenando detector de fraude...")
        fraud_metrics = fraud_detector.train(transactions)
    else:
        print("  Cargando detector de fraude existente (usa --retrain para reentrenar)")
        fraud_detector.load()
        fraud_metrics = None  # se calcula después de predict

    transactions_scored = fraud_detector.predict(transactions)

    if fraud_metrics is None:
        fraud_metrics = {
            "trained_on": len(transactions),
            "contamination": config["fraud"]["contamination"],
            "detected_fraud_rate": round(float(transactions_scored["fraud_flag"].mean()), 4),
        }

    fraud_dist = transactions_scored["fraud_category"].value_counts()
    print(f"  Fraude detectado: {fraud_metrics['detected_fraud_rate']:.1%} del portafolio")
    print(f"  Crítico: {fraud_dist.get('critical', 0):,} | Alto riesgo: {fraud_dist.get('high_risk', 0):,} | Sospechoso: {fraud_dist.get('suspicious', 0):,}")

    # ── 7. AI REPORTS ────────────────────────────────────────────────────────
    print_section("7️⃣  AI REPORTS")
    n_reports = 1 if target_client_id else 5

    if target_client_id:
        client_row = clients_scored[clients_scored["client_id"] == target_client_id]
        if client_row.empty:
            print(f"  Cliente {target_client_id} no encontrado.")
            clients_for_report = clients_scored
        else:
            clients_for_report = client_row
    else:
        clients_for_report = clients_scored

    # SHAP por cliente para enriquecer reportes IA
    shap_explanations: dict = {}
    try:
        exp = CreditRiskExplainer.load()
        for _, row in clients_for_report.nlargest(n_reports, "pd").iterrows():
            cid = row["client_id"]
            shap_explanations[cid] = exp.explain_client(row)
    except FileNotFoundError:
        pass   # SHAP no entrenado aún — esperado
    except Exception as e:
        print(f"  [SHAP] Fallo inesperado al cargar explainer: {e}")

    reports = batch_generate_reports(
        clients_df=clients_for_report,
        macro_forecast=macro_forecast,
        macro_analysis=macro_analysis,
        fraud_df=transactions_scored,
        scenario_name=scenario_cfg["label"],
        n_reports=n_reports,
        model=ollama_model,
        shap_explanations=shap_explanations,
    )

    for r in reports:
        print(f"\n  Cliente #{r['client_id']} | Riesgo: {str(r['risk_level']).upper()} | PD: {r['pd']:.1%} | Decisión: {r['decision']}")
        print("  " + "─" * 56)
        for line in r["report"].split("\n"):
            print(f"  {line}")

    # ── GUARDAR RESULTADOS ───────────────────────────────────────────────────
    print_section("💾 GUARDANDO RESULTADOS")
    clients_scored.to_csv(OUTPUTS_DIR / "clients_scored.csv", index=False)
    transactions_scored.to_csv(OUTPUTS_DIR / "transactions_scored.csv", index=False)

    # Feature importance
    importance = credit_model.get_feature_importance()
    importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    # Resumen JSON
    summary = {
        "scenario": scenario,
        "scenario_label": scenario_cfg["label"],
        "portfolio": {
            "total_clients": len(clients_scored),
            "default_rate": round(float(clients_scored["default"].mean()), 4),
            "risk_distribution": {
                str(k): int(v) for k, v in risk_dist.items()
            },
            "avg_pd": round(float(clients_scored["pd"].mean()), 4),
            "avg_credit_score": round(float(clients_scored["credit_score"].mean()), 1),
        },
        "model_metrics": credit_model.get_metadata(),
        "macro": {
            "market_risk": macro_analysis["market_risk"],
            "macro_stress": macro_forecast["macro_stress"],
            "unemployment_forecast": macro_forecast["unemployment_forecast"],
            "rate_forecast": macro_forecast["rate_forecast"],
        },
        "fraud": {
            "fraud_rate": fraud_metrics["detected_fraud_rate"],
            "critical_transactions": int(fraud_dist.get("critical", 0)),
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }

    with open(OUTPUTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  clients_scored.csv       → {len(clients_scored):,} registros")
    print(f"  transactions_scored.csv  → {len(transactions_scored):,} registros")
    print(f"  feature_importance.csv   → {len(importance)} features")
    print(f"  summary.json             → resumen ejecutivo")

    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"  ✅ Pipeline completado en {elapsed:.1f}s")
    print(f"{'═' * 60}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank Risk Analytics Platform")
    parser.add_argument(
        "--scenario",
        choices=["normal", "crisis", "rate_hike"],
        default="normal",
        help="Escenario económico a simular",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="ID de cliente especifico para analisis",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo Ollama a usar (ej: llama3.2:3b, mistral:7b, llama3.1:8b)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Forzar reentrenamiento del modelo (por defecto carga modelo guardado)",
    )
    args = parser.parse_args()
    run_pipeline(scenario=args.scenario, target_client_id=args.client_id, model=args.model, retrain=args.retrain)

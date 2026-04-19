"""
Preprocessing: limpieza, feature engineering e interacciones macro-cliente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EMPLOYMENT_MAP = {
    "employed":     0,
    "self_employed": 1,
    "retired":      2,
    "unemployed":   3,
}

REGION_RISK_MAP = {
    "RM":           0.00,
    "Valparaíso":   0.05,
    "Antofagasta":  0.02,
    "Biobío":       0.10,
    "Maule":        0.12,
    "Araucanía":    0.18,
}


def clean_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y encoda el dataset de clientes."""
    df = df.copy()

    df["employment_code"] = df["employment_status"].map(EMPLOYMENT_MAP).fillna(1)
    df["region_risk"]     = df["region"].map(REGION_RISK_MAP).fillna(0.10)

    df["income_log"]            = np.log1p(df["income"])
    df["loan_to_income"]        = (df["loan_amount"] / df["income"].clip(1)).clip(0, 20)
    df["financial_burden"]      = df["debt_ratio"] * df["loan_to_income"]
    df["payment_history_score"] = 1 - (df["missed_payments"] / 12).clip(0, 1)
    df["credit_maturity"]       = np.log1p(df["credit_history_months"])

    df["rule_flag"] = (
        (df["debt_ratio"] > 0.70)
        | (df["missed_payments"] >= 3)
        | (df["employment_status"] == "unemployed")
    ).astype(int)

    return df


def add_macro_features(df: pd.DataFrame, macro_forecast: dict, macro_analysis: dict) -> pd.DataFrame:
    """
    Crea features de interacción macro × cliente.
    Las variables macro puras son constantes por cliente y tienen importancia=0
    en modelos de árbol. Las interacciones varían por individuo y son captadas.
    """
    df = df.copy()

    # Valores macro escalares (usados para interacciones, no directamente)
    unemp  = macro_forecast.get("unemployment_forecast", 8.5)
    rate   = macro_forecast.get("rate_forecast", 5.0)
    stress = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(
        macro_forecast.get("macro_stress", "low"), 0.0
    )
    risk_map = {"low": 0.0, "medium": 0.33, "high": 0.66, "extreme": 1.0, "unknown": 0.33}
    mkt_risk = risk_map.get(macro_analysis.get("market_risk", "low"), 0.33)

    usdclp_data = macro_analysis.get("usdclp", {})
    usdclp_ret  = usdclp_data.get("return_ytd", 0.0) if isinstance(usdclp_data, dict) else 0.0
    usdclp_pres = max(0.0, usdclp_ret)

    copper_data = macro_analysis.get("copper", {})
    copper_ret  = copper_data.get("return_ytd", 0.0) if isinstance(copper_data, dict) else 0.0
    copper_risk = max(0.0, -copper_ret)

    # ── Interacciones macro × cliente (varían por individuo) ─────────────────

    # Desempleo alto golpea más a desempleados y auto-empleados
    df["unemp_x_employment"] = (unemp / 10.0) * df["employment_code"].map({0: 0.1, 1: 0.5, 2: 0.2, 3: 1.0}).fillna(0.3)

    # Desempleo alto + deuda alta = riesgo amplificado
    df["unemp_x_debt"]       = (unemp / 10.0) * df["debt_ratio"]

    # Tasas altas encarecen deudas — impacto proporcional a la deuda del cliente
    df["rate_x_debt"]        = (rate / 10.0) * df["debt_ratio"]

    # Tasas altas + alta carga deuda/ingreso = peor capacidad de pago
    df["rate_x_loan_income"] = (rate / 10.0) * df["loan_to_income"].clip(0, 10) / 10.0

    # Stress macro amplifica el historial malo
    df["stress_x_missed"]    = stress * df["missed_payments"]

    # Presión cambiaria golpea más a quien tiene menos ingreso real
    df["usdclp_x_income"]    = usdclp_pres * (1.0 / df["income_log"].clip(1))

    # Riesgo cobre golpea más a regiones mineras
    df["copper_x_region"]    = copper_risk * df["region_risk"]

    # Riesgo de mercado amplifica la carga financiera general
    df["mkt_x_burden"]       = mkt_risk * df["financial_burden"].clip(0, 5)

    return df


def get_feature_columns() -> list[str]:
    """Features para el modelo — incluye interacciones macro×cliente."""
    return [
        # Features individuales del cliente
        "age",
        "income_log",
        "debt_ratio",
        "employment_code",
        "credit_maturity",
        "num_loans",
        "loan_to_income",
        "missed_payments",
        "savings_ratio",
        "financial_burden",
        "payment_history_score",
        "region_risk",
        "rule_flag",
        # Interacciones macro × cliente (varían por individuo)
        "unemp_x_employment",
        "unemp_x_debt",
        "rate_x_debt",
        "rate_x_loan_income",
        "stress_x_missed",
        "usdclp_x_income",
        "copper_x_region",
        "mkt_x_burden",
    ]

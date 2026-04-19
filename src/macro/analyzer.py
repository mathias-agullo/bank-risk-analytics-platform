"""
Macro Monitor: analiza retornos, volatilidad, drawdown y tendencia de mercado.
Indicadores: IPSA, S&P500, USD/CLP, Cobre.
Output: dict con señales de riesgo de mercado y alertas por indicador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    return df[price_col].pct_change().dropna()


def compute_volatility(returns: pd.Series, window: int = 21) -> float:
    """Volatilidad anualizada (ventana rolling de 21 días ~ 1 mes)."""
    return float(returns.rolling(window).std().iloc[-1] * np.sqrt(252))


def compute_drawdown(df: pd.DataFrame, price_col: str = "Close") -> float:
    """Drawdown desde el último pico."""
    prices = df[price_col]
    peak = prices.cummax()
    return float(((prices - peak) / peak).iloc[-1])


def compute_trend(returns: pd.Series, short: int = 20, long: int = 60) -> str:
    if len(returns) < long:
        return "neutral"
    short_ma = returns.iloc[-short:].mean()
    long_ma = returns.iloc[-long:].mean()
    if short_ma > long_ma * 1.001:
        return "up"
    elif short_ma < long_ma * 0.999:
        return "down"
    return "neutral"


# Umbrales de alerta por indicador
_ALERTS = {
    "usdclp": {
        # USD/CLP sube → peso se deprecia → mayor riesgo (deuda en USD, importaciones)
        "high_threshold": 0.10,    # +10% en el año = alerta
        "extreme_threshold": 0.20, # +20% = alerta extrema
        "direction": "up_is_bad",
    },
    "copper": {
        # Cobre baja → economía chilena se contrae → mayor riesgo
        "high_threshold": -0.15,
        "extreme_threshold": -0.25,
        "direction": "down_is_bad",
    },
    "ipsa": {
        "high_threshold": -0.10,
        "extreme_threshold": -0.20,
        "direction": "down_is_bad",
    },
    "sp500": {
        "high_threshold": -0.10,
        "extreme_threshold": -0.20,
        "direction": "down_is_bad",
    },
}


def _indicator_alert(name: str, return_ytd: float, trend: str) -> str:
    """Genera alerta textual para un indicador específico."""
    cfg = _ALERTS.get(name, {})
    direction = cfg.get("direction", "down_is_bad")
    extreme = cfg.get("extreme_threshold", -0.20)
    high = cfg.get("high_threshold", -0.10)

    if direction == "up_is_bad":
        if return_ytd >= abs(extreme):
            return "EXTREMO"
        elif return_ytd >= abs(high):
            return "ALTO"
        return "NORMAL"
    else:
        if return_ytd <= extreme:
            return "EXTREMO"
        elif return_ytd <= high:
            return "ALTO"
        return "NORMAL"


def assess_market_risk(results: dict) -> str:
    """
    Clasifica riesgo global considerando los 4 indicadores chilenos.
    IPSA + cobre = economía local. USD/CLP = presión cambiaria. S&P500 = contagio global.
    """
    score = 0

    for name, data in results.items():
        if not isinstance(data, dict):
            continue
        alert = data.get("alert", "NORMAL")
        vol = data.get("volatility_annualized", 0)
        trend = data.get("trend", "neutral")

        if alert == "EXTREMO":
            score += 3
        elif alert == "ALTO":
            score += 2

        if vol > 0.30:
            score += 1
        if trend == "down" and name in ("ipsa", "copper"):
            score += 1
        if trend == "up" and name == "usdclp":
            score += 1

    if score >= 8:
        return "extreme"
    elif score >= 5:
        return "high"
    elif score >= 2:
        return "medium"
    return "low"


def analyze(market_data: dict[str, pd.DataFrame]) -> dict:
    """
    Analiza IPSA, S&P500, USD/CLP y Cobre.

    Returns dict con métricas por indicador + market_risk global + alertas.
    """
    results: dict = {}

    for name, df in market_data.items():
        if df.empty or "Close" not in df.columns:
            continue

        returns = compute_returns(df)
        vol = compute_volatility(returns)
        dd = compute_drawdown(df)
        trend = compute_trend(returns)
        ytd_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1)
        last_price = round(float(df["Close"].iloc[-1]), 4)
        alert = _indicator_alert(name, ytd_return, trend)

        results[name] = {
            "return_ytd": round(ytd_return, 4),
            "volatility_annualized": round(vol, 4),
            "drawdown_current": round(dd, 4),
            "trend": trend,
            "last_price": last_price,
            "alert": alert,
        }

    market_risk = assess_market_risk(results)
    results["market_risk"] = market_risk
    results["summary"] = _build_summary(results, market_risk)
    return results


def _build_summary(results: dict, risk: str) -> str:
    labels = {
        "ipsa": "IPSA",
        "sp500": "S&P500",
        "usdclp": "USD/CLP",
        "copper": "Cobre",
    }
    lines = [f"Riesgo de mercado: {risk.upper()}"]
    for name, data in results.items():
        if not isinstance(data, dict):
            continue
        label = labels.get(name, name.upper())
        lines.append(
            f"  {label}: precio={data['last_price']} | "
            f"retorno={data['return_ytd']:.1%} | "
            f"tendencia={data['trend']} | "
            f"alerta={data['alert']}"
        )
    return " | ".join(lines)

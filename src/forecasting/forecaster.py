"""
Forecasting: proyecta variables macroeconómicas (desempleo, tasa, inflación).
- Si hay credenciales BCCh → usa series reales como punto de partida
- Fallback a datos históricos aproximados de Chile 2020-2024
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Datos históricos base (Chile, valores aproximados 2020-2024)
# Usados como fallback si BCCh no está disponible
_MACRO_HISTORY = {
    "unemployment": [10.8, 9.8, 8.4, 8.7, 8.5, 8.3, 8.7, 8.1, 7.9, 8.0, 8.2, 8.4],
    "rate":         [0.5, 1.5, 4.0, 7.5, 11.25, 11.25, 10.0, 8.5, 6.5, 5.5, 5.0, 5.0],
    "inflation":    [3.0, 7.2, 12.8, 12.3, 8.7, 6.0, 4.5, 4.2, 4.0, 3.8, 3.5, 3.5],
}


def _arima_forecast(series: list[float], horizon: int) -> list[float]:
    """Intenta ARIMA(1,1,0). Fallback a proyección lineal si falla."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(series, order=(1, 1, 0)).fit()
        return [round(float(v), 2) for v in fit.forecast(steps=horizon)]
    except Exception:
        last = series[-1]
        slope = (series[-1] - series[-3]) / 2 if len(series) >= 3 else 0
        return [round(last + slope * (i + 1) * 0.1, 2) for i in range(horizon)]


def _build_history_from_bcch(bcch_values: dict) -> dict:
    """
    Reemplaza el historial base con datos reales del BCCh donde estén disponibles.
    bcch_values viene de bcch_data.extract_macro_values().
    """
    history = {k: list(v) for k, v in _MACRO_HISTORY.items()}

    # Desempleo real → reemplaza 'unemployment'
    if "unemployment_history" in bcch_values and len(bcch_values["unemployment_history"]) >= 6:
        history["unemployment"] = bcch_values["unemployment_history"]
        print(f"  [BCCh] Desempleo real: ultimo valor = {history['unemployment'][-1]:.2f}%")

    # TPM real → reemplaza 'rate'
    if "tpm_history" in bcch_values and len(bcch_values["tpm_history"]) >= 6:
        history["rate"] = bcch_values["tpm_history"]
        print(f"  [BCCh] TPM real: ultimo valor = {history['rate'][-1]:.2f}%")

    # IPC real → reemplaza 'inflation'
    if "ipc_history" in bcch_values and len(bcch_values["ipc_history"]) >= 6:
        # Convertir variaciones mensuales a nivel acumulado anual (suma móvil)
        monthly = bcch_values["ipc_history"]
        # Construir serie de inflación anual como suma de últimos 12 meses rolling
        inflation_series = [
            round(sum(monthly[max(0, i-11):i+1]), 2)
            for i in range(len(monthly))
        ]
        history["inflation"] = inflation_series[-12:]
        print(f"  [BCCh] IPC real: inflacion anual = {history['inflation'][-1]:.2f}%")

    return history


def forecast_macro(
    horizon: int = 6,
    scenario_deltas: dict | None = None,
    bcch_values: dict | None = None,
) -> dict:
    """
    Proyecta variables macro para `horizon` meses.

    Args:
        horizon: meses hacia adelante
        scenario_deltas: ajustes por escenario, e.g. {"unemployment_delta": 2.5}
        bcch_values: dict de extract_macro_values() — usa datos BCCh reales si están

    Returns dict con forecasts, último valor real, y metadata de stress.
    """
    deltas = scenario_deltas or {}

    # Seleccionar historial: BCCh real o fallback
    if bcch_values:
        history = _build_history_from_bcch(bcch_values)
        data_source = "BCCh (real)"
    else:
        history = {k: list(v) for k, v in _MACRO_HISTORY.items()}
        data_source = "historico aproximado"

    results: dict = {"data_source": data_source}

    for var, hist in history.items():
        base_forecast = _arima_forecast(hist, horizon)
        delta = deltas.get(f"{var}_delta", 0.0)
        adjusted = [round(v + delta * ((i + 1) / horizon), 2) for i, v in enumerate(base_forecast)]

        last_value = hist[-1]
        final_value = adjusted[-1]
        direction = (
            "rising" if final_value > last_value * 1.05
            else "falling" if final_value < last_value * 0.95
            else "stable"
        )

        results[var] = {
            "history": hist,
            "forecast": adjusted,
            "last": last_value,
            "forecast_end": final_value,
            "direction": direction,
        }

    # Valores finales de convenencia
    unemp_end = results["unemployment"]["forecast_end"]
    rate_end   = results["rate"]["forecast_end"]
    infl_end   = results["inflation"]["forecast_end"]

    results["rate_scenario"] = results["rate"]["direction"]
    results["unemployment_forecast"] = unemp_end
    results["rate_forecast"]         = rate_end
    results["inflation_forecast"]    = infl_end

    # UF y USD/CLP oficial si vienen del BCCh
    if bcch_values:
        if "uf_value" in bcch_values:
            results["uf_value"] = bcch_values["uf_value"]
        if "usdclp_official" in bcch_values:
            results["usdclp_official"] = bcch_values["usdclp_official"]
        if "tpm_current" in bcch_values:
            results["tpm_current"] = bcch_values["tpm_current"]

    # Stress score
    score = 0
    if unemp_end > 10:   score += 2
    elif unemp_end > 8.5: score += 1
    if rate_end > 9:     score += 2
    elif rate_end > 6:   score += 1
    if infl_end > 8:     score += 1

    results["macro_stress"] = "high" if score >= 3 else "medium" if score >= 1 else "low"

    return results

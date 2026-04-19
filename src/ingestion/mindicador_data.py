"""
mindicador.cl — datos macro chilenos gratuitos, sin credenciales.
API pública: https://mindicador.cl

Indicadores usados:
  uf    → Valor UF diario (pesos)
  ipc   → IPC variación mensual (porcentaje)
  tpm   → Tasa de Política Monetaria mensual (porcentaje)
  dolar → Dólar observado diario (pesos)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR  = Path(__file__).parent.parent.parent / "data" / "raw" / "mindicador"
BASE_URL   = "https://mindicador.cl/api"
CACHE_HOURS = 6
TIMEOUT     = 10  # segundos


def _fetch_serie(indicador: str) -> list[dict] | None:
    """Descarga la serie completa de un indicador."""
    try:
        r = requests.get(f"{BASE_URL}/{indicador}", timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data.get("serie", [])
    except Exception as e:
        logging.warning("[mindicador] Error al descargar '%s': %s", indicador, e)
        return None


def _serie_to_df(serie: list[dict]) -> pd.DataFrame:
    """Convierte lista de {fecha, valor} a DataFrame con índice de fechas (tz-naive)."""
    df = pd.DataFrame(serie)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True).dt.tz_localize(None)
    df = df.set_index("fecha").rename(columns={"valor": "value"})
    df = df[["value"]].sort_index()
    return df


def fetch_mindicador(use_cache: bool = True) -> dict[str, pd.DataFrame] | None:
    """
    Descarga indicadores macro desde mindicador.cl.
    Retorna dict {nombre: DataFrame con columna 'value'} o None si falla todo.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "mindicador_macro.csv"

    # Cache de 6 horas
    if use_cache and cache_path.exists():
        age_h = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).seconds / 3600
        if age_h < CACHE_HOURS:
            try:
                df_cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                print(f"  [mindicador] cache OK ({len(df_cache)} obs)")
                return _split(df_cache)
            except Exception:
                pass

    indicadores = {"uf": "uf", "ipc_anual": "ipc", "tpm": "tpm", "usdclp": "dolar"}
    dfs: dict[str, pd.DataFrame] = {}

    for nombre, codigo in indicadores.items():
        serie = _fetch_serie(codigo)
        if serie:
            dfs[nombre] = _serie_to_df(serie)
            print(f"  [mindicador] {codigo}: {len(dfs[nombre])} obs")

    if not dfs:
        return None

    # Guardar caché unificada
    combined = pd.concat(
        {k: v["value"] for k, v in dfs.items()}, axis=1
    )
    combined.to_csv(cache_path)

    return dfs


def _split(df_cache: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for col in df_cache.columns:
        sub = df_cache[[col]].dropna().rename(columns={col: "value"})
        result[col] = sub
    return result


def extract_macro_values_mindicador(data: dict[str, pd.DataFrame]) -> dict:
    """
    Extrae valores actuales del dict de DataFrames de mindicador.cl.
    Retorna el mismo formato que extract_macro_values() de bcch_data.py.
    """
    out: dict = {"source": "mindicador.cl (público)"}
    today = pd.Timestamp(datetime.today().date())

    # TPM
    if "tpm" in data and not data["tpm"].empty:
        hist = data["tpm"]["value"].dropna()
        hist = hist[hist.index <= today]
        if not hist.empty:
            out["tpm_current"] = round(float(hist.iloc[-1]), 2)
            out["tpm_history"] = [round(v, 2) for v in hist.tail(24).tolist()]

    # IPC — mindicador da variación mensual, acumulamos últimos 12 meses como proxy anual
    if "ipc_anual" in data and not data["ipc_anual"].empty:
        hist = data["ipc_anual"]["value"].dropna()
        hist = hist[hist.index <= today]
        if not hist.empty:
            # IPC anual ≈ suma de últimos 12 meses (variación mensual)
            ipc_anual = round(float(hist.tail(12).sum()), 2)
            out["ipc_anual"]   = ipc_anual
            out["ipc_history"] = [round(v, 2) for v in hist.tail(24).tolist()]

    # UF
    if "uf" in data and not data["uf"].empty:
        hist = data["uf"]["value"].dropna()
        hist = hist[hist.index <= today]
        if not hist.empty:
            out["uf_value"] = round(float(hist.iloc[-1]), 2)

    # USD/CLP
    if "usdclp" in data and not data["usdclp"].empty:
        hist = data["usdclp"]["value"].dropna()
        hist = hist[hist.index <= today]
        if not hist.empty:
            out["usdclp_official"] = round(float(hist.iloc[-1]), 2)

    return out

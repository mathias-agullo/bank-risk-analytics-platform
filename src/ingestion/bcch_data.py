"""
Banco Central de Chile — datos reales via bcchapi.
Requiere: pip install bcchapi
Credenciales via variables de entorno: BCCH_USER / BCCH_PASS
  o archivo config/bcch_credentials.txt (linea 1: user, linea 2: pass)

Series confirmadas:
  F022.TPM.TIN.D001.NO.Z.M  → TPM mensual (porcentaje)
  G073.IPC.V12.2018.M       → IPC variacion anual (porcentaje)
  F073.UFF.PRE.Z.D          → Valor UF diario (pesos)
  F073.TCO.PRE.Z.D          → Dolar observado diario (pesos)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "bcch"
CRED_FILE  = Path(__file__).parent.parent.parent / "config" / "bcch_credentials.txt"

# Series y sus nombres amigables
SERIES = {
    "tpm":        "F022.TPM.TIN.D001.NO.Z.M",
    "ipc_anual":  "G073.IPC.V12.2018.M",
    "uf":         "F073.UFF.PRE.Z.D",
    "usdclp":     "F073.TCO.PRE.Z.D",
    "unemployment": "F049.DES.TAS.INE1.10.M",   # Tasa desocupación mensual INE
}


def _get_siete():
    """Inicializa bcchapi.Siete con credenciales de variables de entorno."""
    try:
        import bcchapi
    except ImportError:
        raise ImportError("Instala bcchapi: pip install bcchapi")

    user = os.environ.get("BCCH_USER", "")
    pwd  = os.environ.get("BCCH_PASS", "")

    if user and pwd:
        return bcchapi.Siete(user, pwd)

    # CN-009: eliminado fallback a archivo de credenciales en texto plano.
    # Si las variables de entorno no están configuradas, retornar None sin
    # intentar leer ningún archivo (evita uso accidental de credenciales en disco).
    return None


def fetch_all_bcch(days: int = 400) -> dict[str, pd.DataFrame] | None:
    """
    Descarga series macro del BCCh.
    Retorna dict {nombre: DataFrame con columna 'value'} o None si no hay credenciales.
    """
    siete = _get_siete()
    if siete is None:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hasta  = datetime.today().strftime("%Y-%m-%d")
    desde  = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    series_ids = list(SERIES.values())
    nombres    = list(SERIES.keys())

    # Caché de 6 horas
    cache_path = CACHE_DIR / "bcch_macro.csv"
    if cache_path.exists():
        age_h = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).seconds / 3600
        if age_h < 6:
            df_cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"  [BCCh] cache OK ({len(df_cache)} obs)")
            return _split_dataframe(df_cache, nombres)

    try:
        df = siete.cuadro(
            series=series_ids,
            nombres=nombres,
            desde=desde,
            hasta=hasta,
        )
        df.to_csv(cache_path)
        print(f"  [BCCh] {len(df)} observaciones descargadas")
        return _split_dataframe(df, nombres)
    except Exception as e:
        print(f"  [BCCh] error: {e}")
        # Intentar desde caché aunque sea vieja
        if cache_path.exists():
            df_cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"  [BCCh] usando cache anterior ({len(df_cache)} obs)")
            return _split_dataframe(df_cache, nombres)
        return None


def _split_dataframe(df: pd.DataFrame, nombres: list[str]) -> dict[str, pd.DataFrame]:
    """Convierte el cuadro unificado en dict de DataFrames individuales."""
    result = {}
    for name in nombres:
        if name in df.columns:
            sub = df[[name]].dropna().rename(columns={name: "value"})
            result[name] = sub
    return result


def extract_macro_values(bcch_data: dict[str, pd.DataFrame]) -> dict:
    """
    Extrae valores recientes y construye historiales para el forecaster.

    Retorna dict con:
        tpm_current   → ultimo valor TPM (%)
        tpm_history   → lista de ultimos 24 meses
        ipc_anual     → ultimo IPC variacion anual (%)
        ipc_history   → lista de ultimos 24 meses
        uf_value      → valor UF hoy
        usdclp_official → tipo de cambio observado hoy
    """
    out: dict = {"source": "BCCh (real)"}

    # TPM
    if "tpm" in bcch_data and not bcch_data["tpm"].empty:
        series = bcch_data["tpm"]["value"].dropna()
        # Filtrar solo valores historicos reales (descartar proyecciones futuras > hoy)
        today = pd.Timestamp(datetime.today().date())
        hist  = series[series.index <= today]
        if not hist.empty:
            out["tpm_current"] = round(float(hist.iloc[-1]), 2)
            out["tpm_history"] = [round(v, 2) for v in hist.tail(24).tolist()]

    # IPC variacion anual
    if "ipc_anual" in bcch_data and not bcch_data["ipc_anual"].empty:
        series = bcch_data["ipc_anual"]["value"].dropna()
        today  = pd.Timestamp(datetime.today().date())
        hist   = series[series.index <= today]
        if not hist.empty:
            out["ipc_anual"]   = round(float(hist.iloc[-1]), 2)
            out["ipc_history"] = [round(v, 2) for v in hist.tail(24).tolist()]

    # UF
    if "uf" in bcch_data and not bcch_data["uf"].empty:
        out["uf_value"] = round(float(bcch_data["uf"]["value"].iloc[-1]), 2)

    # USD/CLP observado
    if "usdclp" in bcch_data and not bcch_data["usdclp"].empty:
        series = bcch_data["usdclp"]["value"].dropna()
        if not series.empty:
            out["usdclp_official"] = round(float(series.iloc[-1]), 2)

    # Desempleo — serie mensual INE
    if "unemployment" in bcch_data and not bcch_data["unemployment"].empty:
        series = bcch_data["unemployment"]["value"].dropna()
        today  = pd.Timestamp(datetime.today().date())
        hist   = series[series.index <= today]
        if not hist.empty:
            out["unemployment_current"] = round(float(hist.iloc[-1]), 2)
            # Interpolar a mensual si viene trimestral
            if len(hist) >= 4:
                hist_monthly = hist.resample("MS").interpolate("linear")
                out["unemployment_history"] = [round(v, 2) for v in hist_monthly.tail(24).tolist()]
            else:
                out["unemployment_history"] = [round(v, 2) for v in hist.tolist()]

    return out

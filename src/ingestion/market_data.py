"""
Descarga datos de mercado: IPSA (Stooq), S&P500, USD/CLP y Cobre (yfinance).
Cadena de fallback por ticker:
  IPSA  → Stooq → simulado
  resto → yfinance → simulado
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# Tickers que usamos Stooq en vez de yfinance
_STOOQ_TICKERS = {"IPSA.SN", "^IPSA"}


def _simulate_market(ticker: str, days: int, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=days)
    mu = 0.0003
    sigma = 0.012 if "GSPC" in ticker else 0.015
    log_returns = rng.normal(mu, sigma, len(dates))
    price = 100 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"Date": dates, "Close": price, "Volume": rng.integers(1e6, 5e6, len(dates))})


def _fetch_stooq(ticker: str, days: int) -> pd.DataFrame:
    """Descarga datos desde Stooq vía pandas_datareader."""
    from pandas_datareader import data as pdr

    end = datetime.today()
    start = end - timedelta(days=days)
    # Stooq usa ticker en minúsculas sin ^
    stooq_ticker = ticker.lower().replace("^", "").replace(".sn", "")
    df = pdr.get_data_stooq(f"^{stooq_ticker}", start=start, end=end)
    if df.empty:
        raise ValueError("Stooq retornó datos vacíos")
    df = df.reset_index()
    df.columns = [c if c != "index" else "Date" for c in df.columns]
    # Stooq retorna columnas: Date, Open, High, Low, Close, Volume
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close", "Volume"]].sort_values("Date").reset_index(drop=True)
    return df


def _fetch_yfinance(ticker: str, days: int) -> pd.DataFrame:
    import yfinance as yf

    end = datetime.today()
    start = end - timedelta(days=days)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if raw.empty:
        raise ValueError("yfinance retornó datos vacíos")
    raw = raw.reset_index()[["Date", "Close", "Volume"]].copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw


def fetch_ticker(ticker: str, days: int = 365, use_cache: bool = True) -> pd.DataFrame:
    """
    Descarga datos de mercado. Estrategia por ticker:
      IPSA  → Stooq primero, luego simulado
      resto → yfinance primero, luego simulado
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{ticker.replace('^', '').replace('=', '_')}.csv"

    if use_cache and cache_path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).seconds / 3600
        if age_hours < 24:
            df = pd.read_csv(cache_path, parse_dates=["Date"])
            print(f"  [{ticker}] cache OK ({len(df)} dias)")
            return df

    use_stooq = ticker in _STOOQ_TICKERS

    # Intento principal
    try:
        if use_stooq:
            df = _fetch_stooq(ticker, days)
            print(f"  [{ticker}] Stooq OK: {len(df)} dias")
        else:
            df = _fetch_yfinance(ticker, days)
            print(f"  [{ticker}] yfinance OK: {len(df)} dias")
        df.to_csv(cache_path, index=False)
        return df

    except Exception as e1:
        source = "Stooq" if use_stooq else "yfinance"
        print(f"  [{ticker}] {source} fallo ({e1})", end="")

        # Si Stooq falló, intentar yfinance como segundo intento
        if use_stooq:
            try:
                df = _fetch_yfinance(ticker, days)
                df.to_csv(cache_path, index=False)
                print(f", yfinance OK: {len(df)} dias")
                return df
            except Exception as e2:
                print(f", yfinance fallo ({e2})", end="")

        print(", datos simulados")
        df = _simulate_market(ticker, days)
        df.to_csv(cache_path, index=False)
        return df


def fetch_all(config: dict) -> dict[str, pd.DataFrame]:
    """Descarga todos los tickers definidos en config (en paralelo)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tickers = config["market"]["tickers"]
    days    = config["market"]["lookback_days"]

    result: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        futures = {
            executor.submit(fetch_ticker, symbol, days): name
            for name, symbol in tickers.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result[name] = future.result()
            except Exception as e:
                print(f"  [{name}] error inesperado: {e} — usando datos simulados")
                result[name] = _simulate_market(name, days)
    return result

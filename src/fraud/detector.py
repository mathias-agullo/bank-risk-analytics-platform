"""
Fraud Detection: Isolation Forest + reglas de negocio.
Input: dataset de transacciones.
Output: fraud_score, flag, risk_category por transacción.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


MERCHANT_RISK = {
    "luxury": 0.8,
    "travel": 0.5,
    "online": 0.4,
    "atm": 0.3,
    "restaurant": 0.1,
    "retail": 0.1,
    "utility": 0.0,
}

CHANNEL_RISK = {"web": 0.3, "mobile": 0.2, "pos": 0.1, "atm": 0.25}


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features numéricas para Isolation Forest."""
    feat = pd.DataFrame()
    feat["amount_log"] = np.log1p(df["amount"])
    feat["is_foreign"] = df["is_foreign"].astype(float)
    feat["is_weekend"] = df["is_weekend"].astype(float)
    feat["merchant_risk"] = df["merchant_category"].map(MERCHANT_RISK).fillna(0.2)
    feat["channel_risk"] = df["channel"].map(CHANNEL_RISK).fillna(0.2)
    feat["night_flag"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(float)
    feat["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return feat


def _business_rules(row: pd.Series) -> float:
    """Score adicional por reglas de negocio (0–1)."""
    score = 0.0
    if row["amount"] > 2_000_000:
        score += 0.3
    if row["is_foreign"] and row["amount"] > 500_000:
        score += 0.25
    if row["hour_of_day"] < 4 and row["amount"] > 200_000:
        score += 0.2
    if row["merchant_category"] == "luxury" and row["is_foreign"]:
        score += 0.2
    return min(score, 1.0)


class FraudDetector:
    def __init__(self, config: dict):
        self.contamination = config["fraud"]["contamination"]
        self.random_state = config["fraud"]["random_state"]
        self.pipeline: Optional[Pipeline] = None

    # ── Integridad de archivos (CN-004) ──────────────────────────────────────

    @staticmethod
    def _save_hash(path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".sha256").write_text(digest, encoding="utf-8")

    @staticmethod
    def _verify_hash(path: Path) -> None:
        hash_path = path.with_suffix(".sha256")
        if not hash_path.exists():
            raise FileNotFoundError(
                f"Archivo de integridad no encontrado para {path.name}. "
                "Re-entrena el modelo con --retrain."
            )
        expected = hash_path.read_text(encoding="utf-8").strip()
        actual   = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"¡Verificación de integridad fallida para {path.name}! "
                "El archivo puede haber sido modificado o corrompido."
            )

    def train(self, df: pd.DataFrame) -> dict:
        features = _build_features(df)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iso", IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )),
        ])
        self.pipeline.fit(features)
        fraud_path = MODELS_DIR / "fraud_iso.pkl"
        joblib.dump(self.pipeline, fraud_path)
        self._save_hash(fraud_path)

        preds = self.predict(df)
        fraud_rate = float(preds["fraud_flag"].mean())
        return {
            "trained_on": len(df),
            "contamination": self.contamination,
            "detected_fraud_rate": round(fraud_rate, 4),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Modelo no entrenado.")

        features = _build_features(df)
        # Isolation Forest: -1 anomaly, +1 normal → invertimos a score 0-1
        raw_scores = self.pipeline.named_steps["iso"].score_samples(
            self.pipeline.named_steps["scaler"].transform(features)
        )
        # Normalizar a [0,1] donde 1 = más anómalo
        iso_score = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

        # Combinar con reglas
        rule_scores = df.apply(_business_rules, axis=1).values
        fraud_score = (0.65 * iso_score + 0.35 * rule_scores).clip(0, 1)

        result = df.copy()
        result["fraud_score"] = fraud_score.round(4)
        result["fraud_flag"] = (fraud_score > 0.55).astype(int)
        result["fraud_category"] = pd.cut(
            result["fraud_score"],
            bins=[-0.001, 0.30, 0.55, 0.75, 1.001],
            labels=["normal", "suspicious", "high_risk", "critical"],
        )
        return result

    def is_trained(self) -> bool:
        """Retorna True si el modelo ya fue entrenado y guardado en disco."""
        return (MODELS_DIR / "fraud_iso.pkl").exists()

    def load(self):
        fraud_path = MODELS_DIR / "fraud_iso.pkl"
        self._verify_hash(fraud_path)
        self.pipeline = joblib.load(fraud_path)

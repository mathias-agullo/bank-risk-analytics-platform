"""
SHAP Explainer: explicabilidad regulatoria por cliente.
Permite justificar cada decisión de crédito con las variables más influyentes.
Requerimiento regulatorio en banca: toda negativa debe ser explicable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"


# ── Helpers de integridad (CN-004) ────────────────────────────────────────────

def _save_hash(path: Path) -> None:
    """Guarda el SHA256 del .pkl junto al archivo."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest, encoding="utf-8")


def _verify_hash(path: Path) -> None:
    """Verifica SHA256 antes de deserializar. Lanza ValueError si hay discrepancia."""
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


class CreditRiskExplainer:
    def __init__(self):
        self.explainer_gbm = None
        self.feature_cols: list[str] = []
        self.expected_value: float = 0.0

    def fit(self, pipeline_gbm, X_background: pd.DataFrame, feature_cols: list[str]):
        """Inicializa el SHAP TreeExplainer sobre el GBM calibrado."""
        try:
            import shap
        except ImportError:
            raise ImportError("Instala shap: pip install shap")

        self.feature_cols = feature_cols
        X_bg = X_background[feature_cols].fillna(0)

        # Extraer el GBM base desde el CalibratedClassifierCV
        clf = pipeline_gbm.named_steps["clf"]
        if hasattr(clf, "calibrated_classifiers_"):
            gbm_base = clf.calibrated_classifiers_[0].estimator
        else:
            gbm_base = clf

        self.explainer_gbm = shap.TreeExplainer(
            gbm_base,
            data=shap.sample(X_bg, 100),
            feature_perturbation="interventional",
        )
        self.expected_value = float(self.explainer_gbm.expected_value)
        shap_path = MODELS_DIR / "shap_explainer.pkl"
        joblib.dump(self, shap_path)
        _save_hash(shap_path)

    def explain_client(self, client_row: pd.Series, top_n: int = 5) -> dict:
        """
        Retorna las top_n variables que más influyen en la decisión para un cliente.
        Positivo = aumenta riesgo, negativo = reduce riesgo.
        """
        if self.explainer_gbm is None:
            return {}

        X = pd.DataFrame([client_row[self.feature_cols].fillna(0)])
        shap_vals = self.explainer_gbm.shap_values(X)[0]

        contributions = pd.Series(shap_vals, index=self.feature_cols)
        top_positive = contributions.nlargest(top_n)
        top_negative = contributions.nsmallest(top_n)

        return {
            "expected_value": round(self.expected_value, 4),
            "top_risk_factors": [
                {"feature": k, "shap": round(v, 4)} for k, v in top_positive.items()
            ],
            "top_protective_factors": [
                {"feature": k, "shap": round(v, 4)} for k, v in top_negative.items()
            ],
            "all_shap": {k: round(v, 4) for k, v in contributions.items()},
        }

    def explain_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula SHAP para todo el portafolio. Retorna DataFrame con columnas por feature."""
        if self.explainer_gbm is None:
            return pd.DataFrame()

        X = df[self.feature_cols].fillna(0)
        shap_vals = self.explainer_gbm.shap_values(X)
        return pd.DataFrame(shap_vals, columns=self.feature_cols, index=df.index)

    @classmethod
    def load(cls) -> "CreditRiskExplainer":
        path = MODELS_DIR / "shap_explainer.pkl"
        if not path.exists():
            raise FileNotFoundError("No se encontró shap_explainer.pkl — ejecuta con --retrain")
        _verify_hash(path)
        return joblib.load(path)

    def is_available(self) -> bool:
        return (MODELS_DIR / "shap_explainer.pkl").exists()


def format_shap_for_prompt(explanation: dict) -> str:
    """Formatea el análisis SHAP para incluirlo en el prompt del reporte IA."""
    if not explanation:
        return ""

    lines = ["\n== EXPLICABILIDAD (SHAP) =="]
    lines.append("Factores que AUMENTAN el riesgo:")
    for f in explanation.get("top_risk_factors", [])[:3]:
        lines.append(f"  + {f['feature']}: {f['shap']:+.3f}")

    lines.append("Factores que REDUCEN el riesgo:")
    for f in explanation.get("top_protective_factors", [])[:3]:
        lines.append(f"  - {f['feature']}: {f['shap']:+.3f}")

    return "\n".join(lines)

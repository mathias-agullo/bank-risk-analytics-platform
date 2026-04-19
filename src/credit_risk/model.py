"""
Credit Risk Model: Probabilidad de Default (PD) y scoring.
Modelos:
  - Logistic Regression con interacciones polinomiales (interpretable, regulatorio)
  - Gradient Boosting (mejor AUC, captura no-linealidades)
  - Ensemble ponderado LR + GBM

Separación train / inference:
  - python main.py --retrain   → entrena y guarda modelo
  - python main.py             → carga modelo existente y predice
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class CreditRiskModel:
    def __init__(self, config: dict):
        self.cfg            = config["credit_risk"]
        self.threshold_high: float   = self.cfg["threshold_high"]
        self.threshold_medium: float = self.cfg["threshold_medium"]
        self.random_state: int       = self.cfg["random_state"]
        self.feature_cols: list[str] = []
        self.pipeline_lr: Optional[Pipeline] = None
        self.pipeline_gbm: Optional[Pipeline] = None
        self.optimal_threshold: float = 0.5
        self.metrics: dict = {}

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, feature_cols: list[str]) -> dict:
        self.feature_cols = feature_cols
        X = df[feature_cols].fillna(0)
        y = df["default"]

        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg["test_size"],
            random_state=self.random_state,
            stratify=y,
        )

        # ── Logistic Regression calibrada ─────────────────────────────────────
        lr_base = LogisticRegression(
            max_iter=1000, C=0.3,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.pipeline_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(lr_base, cv=5, method="isotonic")),
        ])
        self.pipeline_lr.fit(X_train, y_train)

        # ── Gradient Boosting calibrado ────────────────────────────────────────
        gbm_base = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=self.random_state,
        )
        self.pipeline_gbm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(gbm_base, cv=5, method="isotonic")),
        ])
        self.pipeline_gbm.fit(X_train, y_train)

        # ── Métricas ──────────────────────────────────────────────────────────
        prob_lr  = self.pipeline_lr.predict_proba(X_test)[:, 1]
        prob_gbm = self.pipeline_gbm.predict_proba(X_test)[:, 1]
        prob_ens = 0.4 * prob_lr + 0.6 * prob_gbm

        # Umbral óptimo por F2-score (penaliza falsos negativos — lógica bancaria)
        self.optimal_threshold = self._find_optimal_threshold(y_test, prob_ens)

        # Walk-forward validation
        wf_aucs = self._walk_forward_validate(df, feature_cols)

        self.metrics = {
            "logistic_regression": {
                "auc_roc": round(roc_auc_score(y_test, prob_lr), 4),
                "auc_pr":  round(average_precision_score(y_test, prob_lr), 4),
            },
            "gradient_boosting": {
                "auc_roc": round(roc_auc_score(y_test, prob_gbm), 4),
                "auc_pr":  round(average_precision_score(y_test, prob_gbm), 4),
            },
            "ensemble": {
                "auc_roc":           round(roc_auc_score(y_test, prob_ens), 4),
                "auc_pr":            round(average_precision_score(y_test, prob_ens), 4),
                "optimal_threshold": round(self.optimal_threshold, 4),
            },
            "walk_forward": {
                "auc_mean": round(float(np.mean(wf_aucs)), 4) if wf_aucs else None,
                "auc_std":  round(float(np.std(wf_aucs)), 4)  if wf_aucs else None,
                "folds":    [round(v, 4) for v in wf_aucs],
            },
            "train_size":   len(X_train),
            "test_size":    len(X_test),
            "default_rate": round(float(y.mean()), 4),
        }

        self._save(df)
        return self.metrics

    def _find_optimal_threshold(self, y_true, y_prob) -> float:
        """Umbral que maximiza F2-score (más peso en recall — evitar falsos negativos)."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        beta = 2.0
        f2 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + 1e-9)
        best_idx = np.argmax(f2[:-1])
        return float(thresholds[best_idx])

    def _walk_forward_validate(self, df: pd.DataFrame, feature_cols: list[str]) -> list[float]:
        """Validación temporal: entrena en pasado, valida en futuro."""
        if "origination_date" not in df.columns:
            return []

        df_sorted = df.sort_values("origination_date").reset_index(drop=True)
        X = df_sorted[feature_cols].fillna(0)
        y = df_sorted["default"]

        tscv = TimeSeriesSplit(n_splits=5, gap=30)
        aucs = []
        for train_idx, test_idx in tscv.split(X):
            if y.iloc[test_idx].sum() < 5:   # muy pocos positivos en test
                continue
            from sklearn.ensemble import GradientBoostingClassifier as GBC
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GBC(n_estimators=100, max_depth=3, learning_rate=0.1,
                            random_state=self.random_state)),
            ])
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[test_idx], prob))
        return aucs

    # ── Predicción ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_lr is None or self.pipeline_gbm is None:
            raise RuntimeError("Modelo no cargado. Usa train() o load().")

        X = df[self.feature_cols].fillna(0)
        prob_lr  = self.pipeline_lr.predict_proba(X)[:, 1]
        prob_gbm = self.pipeline_gbm.predict_proba(X)[:, 1]
        pd_ensemble = (0.4 * prob_lr + 0.6 * prob_gbm)

        result = df.copy()
        result["pd_lr"]  = prob_lr.round(4)
        result["pd_gbm"] = prob_gbm.round(4)
        result["pd"]     = pd_ensemble.round(4)
        result["risk_level"] = pd.cut(
            result["pd"],
            bins=[-0.001, self.threshold_medium, self.threshold_high, 1.001],
            labels=["low", "medium", "high"],
        )
        result["decision"] = result["risk_level"].map({
            "low":    "APROBAR",
            "medium": "REVISAR",
            "high":   "RECHAZAR",
        })
        result["credit_score"] = ((1 - result["pd"]) * 700 + 150).round(0).astype(int).clip(300, 850)
        return result

    # ── Persistencia ──────────────────────────────────────────────────────────

    # ── Integridad de archivos (CN-004) ──────────────────────────────────────

    @staticmethod
    def _save_hash(path: Path) -> None:
        """Guarda el SHA256 del .pkl junto al archivo para verificación posterior."""
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".sha256").write_text(digest, encoding="utf-8")

    @staticmethod
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

    def _save(self, df: pd.DataFrame):
        lr_path  = MODELS_DIR / "credit_risk_lr.pkl"
        gbm_path = MODELS_DIR / "credit_risk_gbm.pkl"
        joblib.dump(self.pipeline_lr,  lr_path)
        joblib.dump(self.pipeline_gbm, gbm_path)
        self._save_hash(lr_path)
        self._save_hash(gbm_path)

        # Guardar feature_cols
        with open(MODELS_DIR / "feature_cols.json", "w") as f:
            json.dump(self.feature_cols, f)

        # Metadata de audit trail
        data_hash = hashlib.sha256(df[self.feature_cols].fillna(0).values.tobytes()).hexdigest()[:16]
        metadata  = {
            "trained_at":        datetime.now().isoformat(),
            "train_size":        self.metrics.get("train_size"),
            "test_size":         self.metrics.get("test_size"),
            "default_rate":      self.metrics.get("default_rate"),
            "auc_roc_ensemble":  self.metrics["ensemble"]["auc_roc"],
            "auc_roc_lr":        self.metrics["logistic_regression"]["auc_roc"],
            "auc_roc_gbm":       self.metrics["gradient_boosting"]["auc_roc"],
            "walk_forward_mean": self.metrics["walk_forward"]["auc_mean"],
            "walk_forward_std":  self.metrics["walk_forward"]["auc_std"],
            "optimal_threshold": self.optimal_threshold,
            "feature_cols":      self.feature_cols,
            "data_hash":         data_hash,
        }
        with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load(self):
        lr_path  = MODELS_DIR / "credit_risk_lr.pkl"
        gbm_path = MODELS_DIR / "credit_risk_gbm.pkl"
        self._verify_hash(lr_path)
        self._verify_hash(gbm_path)
        self.pipeline_lr  = joblib.load(lr_path)
        self.pipeline_gbm = joblib.load(gbm_path)
        with open(MODELS_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(MODELS_DIR / "model_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.optimal_threshold = meta.get("optimal_threshold", 0.5)

    def is_trained(self) -> bool:
        return (
            (MODELS_DIR / "credit_risk_lr.pkl").exists()
            and (MODELS_DIR / "credit_risk_gbm.pkl").exists()
            and (MODELS_DIR / "feature_cols.json").exists()
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Importancia del GBM (más confiable que el árbol simple)."""
        if self.pipeline_gbm is None:
            return pd.DataFrame()
        clf = self.pipeline_gbm.named_steps["clf"]
        # CalibratedClassifierCV envuelve el estimador
        if hasattr(clf, "estimators_"):
            importances = np.mean([e.feature_importances_ for e in clf.estimators_], axis=0)
        elif hasattr(clf, "calibrated_classifiers_"):
            importances = np.mean(
                [c.estimator.feature_importances_ for c in clf.calibrated_classifiers_], axis=0
            )
        else:
            return pd.DataFrame()
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_metadata(self) -> dict:
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

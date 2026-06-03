"""Pluggable ML backends: XGBoost, LightGBM, CatBoost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from ranking_pipeline.config import ModelBackend


class RankingModelPair(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_excess(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class TrainedRankingModels:
    classifier: Any
    regressor: Any
    feature_columns: list[str]
    backend: ModelBackend


def _build_classifier(backend: ModelBackend, n_features: int) -> Any:
    if backend == ModelBackend.XGBOOST:
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    if backend == ModelBackend.LIGHTGBM:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if backend == ModelBackend.CATBOOST:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
        )
    raise ValueError(f"No classifier for backend {backend}")


def _build_regressor(backend: ModelBackend) -> Any:
    if backend == ModelBackend.XGBOOST:
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    if backend == ModelBackend.LIGHTGBM:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if backend == ModelBackend.CATBOOST:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
        )
    raise ValueError(f"No regressor for backend {backend}")


def train_models(
    X: pd.DataFrame,
    y_class: pd.Series,
    y_reg: pd.Series,
    backend: ModelBackend,
) -> TrainedRankingModels:
    clf = _build_classifier(backend, X.shape[1])
    reg = _build_regressor(backend)
    clf.fit(X, y_class)
    reg.fit(X, y_reg)
    return TrainedRankingModels(
        classifier=clf,
        regressor=reg,
        feature_columns=list(X.columns),
        backend=backend,
    )


def predict(models: TrainedRankingModels, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = models.feature_columns
    X_aligned = X.reindex(columns=cols).fillna(0.0)
    proba = models.classifier.predict_proba(X_aligned)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        p_outperform = proba[:, 1]
    else:
        p_outperform = proba.ravel()
    expected = models.regressor.predict(X_aligned)
    return p_outperform, expected


def artifact_dir(base: Path, backend: ModelBackend) -> Path:
    return base / backend.value


def save_models(models: TrainedRankingModels, base: Path) -> Path:
    out_dir = artifact_dir(base, models.backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.classifier, out_dir / "classifier.joblib")
    joblib.dump(models.regressor, out_dir / "regressor.joblib")
    joblib.dump(models.feature_columns, out_dir / "feature_columns.joblib")
    return out_dir


def load_models(base: Path, backend: ModelBackend) -> TrainedRankingModels | None:
    out_dir = artifact_dir(base, backend)
    clf_path = out_dir / "classifier.joblib"
    if not clf_path.exists():
        return None
    return TrainedRankingModels(
        classifier=joblib.load(out_dir / "classifier.joblib"),
        regressor=joblib.load(out_dir / "regressor.joblib"),
        feature_columns=joblib.load(out_dir / "feature_columns.joblib"),
        backend=backend,
    )

"""Train ranking ML models on universe feature panels."""

from __future__ import annotations

import logging

from ranking_pipeline.config import ModelBackend, RankingPipelineConfig, default_config
from ranking_pipeline.ml.dataset import build_panel_from_symbols, feature_columns_for_ml, xy_from_panel
from ranking_pipeline.ml.registry import save_models, train_models
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)


def train_ranking_models(
    backend: ModelBackend | None = None,
    *,
    train_end: str = "2024-12-31",
    config: RankingPipelineConfig | None = None,
) -> dict:
    cfg = config or default_config()
    backend = backend or cfg.model_backend
    if backend == ModelBackend.COMPOSITE_ONLY:
        raise ValueError("Choose xgboost, lightgbm, or catboost for training")

    store = open_store(cfg)
    symbols = store.load_universe_symbols()
    if not symbols:
        raise RuntimeError("No universe symbols — refresh universe first")

    panel = build_panel_from_symbols(
        symbols,
        classification_target=cfg.classification_target,
    )
    if panel.empty:
        raise RuntimeError("Empty training panel")

    cols = feature_columns_for_ml(panel)
    X, y_cls, y_reg = xy_from_panel(
        panel,
        cols,
        classification_target=cfg.classification_target,
    )
    dates = X.index.get_level_values("date")
    cutoff_mask = dates <= train_end
    X_train = X.loc[cutoff_mask]
    y_cls_train = y_cls.loc[cutoff_mask]
    y_reg_train = y_reg.loc[cutoff_mask]

    models = train_models(X_train, y_cls_train, y_reg_train, backend)
    out_dir = save_models(models, cfg.artifacts_dir)
    logger.info("Saved %s models to %s (%d rows)", backend.value, out_dir, len(X_train))
    return {
        "backend": backend.value,
        "artifact_dir": str(out_dir),
        "train_rows": len(X_train),
        "feature_count": len(cols),
        "classification_target": cfg.classification_target.value,
        "regression_target": "excess_ret_5d",
    }

#!/usr/bin/env python3
"""Train ranking ML classifier + regressor for active universe."""

from __future__ import annotations

import argparse
import logging
import sys

from ranking_pipeline.config import ModelBackend
from ranking_pipeline.ml.train import train_ranking_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ranking ML models.")
    parser.add_argument(
        "--backend",
        choices=[b.value for b in ModelBackend if b != ModelBackend.COMPOSITE_ONLY],
        default="xgboost",
    )
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument(
        "--target",
        choices=["outperform_spy", "top_quintile"],
        default=None,
        help="Classification target (default: env or outperform_spy)",
    )
    args = parser.parse_args(argv)
    from ranking_pipeline.config import default_config
    from ranking_pipeline.ml.labels import ClassificationTarget

    cfg = default_config()
    if args.target:
        cfg.classification_target = ClassificationTarget(args.target)
    result = train_ranking_models(
        ModelBackend(args.backend),
        train_end=args.train_end,
        config=cfg,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

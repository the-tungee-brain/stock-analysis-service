"""Batch progress logging."""

from __future__ import annotations

from ranking_pipeline.pipeline.progress_log import progress_log_interval


def test_progress_log_interval_scales_with_total() -> None:
    assert progress_log_interval(6525) == 65
    assert progress_log_interval(100) == 1
    assert progress_log_interval(0) == 1

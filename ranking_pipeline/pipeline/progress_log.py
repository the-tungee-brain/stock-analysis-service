"""INFO-level progress lines for long parallel batch jobs."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def progress_log_interval(total: int, *, min_step: int = 1, percent_step: float = 1.0) -> int:
    """Log about every ``percent_step``% of ``total`` (at least ``min_step``)."""
    if total <= 0:
        return 1
    return max(min_step, int(total * percent_step / 100.0))


def log_batch_progress(
    label: str,
    done: int,
    total: int,
    *,
    detail: str | None = None,
    step: int | None = None,
) -> None:
    """Emit ``label: done / total`` at start, end, and every ``step`` completions."""
    if total <= 0:
        return
    interval = step if step is not None else progress_log_interval(total)
    if done not in (1, total) and done % interval != 0:
        return
    msg = f"{label}: {done} / {total}"
    if detail:
        msg = f"{msg} ({detail})"
    logger.info(msg)

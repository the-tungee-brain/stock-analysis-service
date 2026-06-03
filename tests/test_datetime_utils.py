"""Timezone-safe as-of comparisons."""

from __future__ import annotations

import pandas as pd

from ranking_pipeline.datetime_utils import to_naive_utc_index, to_naive_utc_timestamp


def test_naive_and_aware_timestamps_compare_with_feature_index() -> None:
    idx = to_naive_utc_index(pd.date_range("2024-01-01", periods=3, freq="B"))
    as_of = to_naive_utc_timestamp(pd.Timestamp("2024-01-03", tz="UTC"))
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    assert not df[df.index <= as_of].empty

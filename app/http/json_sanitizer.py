from __future__ import annotations

import math
from numbers import Real
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is available in normal backend runtime
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is available in normal backend runtime
    pd = None  # type: ignore[assignment]


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]

    if value is None or isinstance(value, (str, bool)):
        return value

    if pd is not None and (value is pd.NA or value is pd.NaT):
        return None

    if np is not None and isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, Real):
        return value if math.isfinite(float(value)) else None

    return value

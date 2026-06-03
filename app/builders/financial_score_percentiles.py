"""Model-relative rank labels from category scores (0–100).

These are deterministic peer-style bands until a live cross-sectional universe
percentile service is available.
"""

from __future__ import annotations


def rank_label_for_score(score: int) -> str:
    value = max(0, min(100, int(score)))
    if value >= 99:
        return "Top 1%"
    if value >= 95:
        return "Top 5%"
    if value >= 90:
        return "Top 10%"
    if value >= 80:
        return "Top 20%"
    if value >= 65:
        return "Top 35%"
    if value >= 50:
        return "Middle"
    if value >= 35:
        return "Bottom 35%"
    if value >= 20:
        return "Bottom 20%"
    if value >= 10:
        return "Bottom 10%"
    if value >= 5:
        return "Bottom 5%"
    return "Bottom 1%"

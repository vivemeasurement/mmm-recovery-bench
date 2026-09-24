"""Score a fitted model against the simulated truth, one row per channel."""
from __future__ import annotations

import numpy as np


def hdi(draws: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    x = np.sort(np.asarray(draws).ravel())
    n = len(x)
    k = int(np.floor(prob * n))
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def score(roas_draws: dict[str, np.ndarray], truth: dict) -> list[dict]:
    rows = []
    for ch, draws in roas_draws.items():
        true = truth["roas"][ch]
        est = float(np.mean(draws))
        lo, hi = hdi(draws)
        rows.append({
            "channel": ch,
            "true_roas": true,
            "est_roas": est,
            "hdi_low": lo,
            "hdi_high": hi,
            "abs_pct_error": abs(est - true) / true,
            "covered": bool(lo <= true <= hi),
            "rel_interval_width": (hi - lo) / true,
        })
    return rows

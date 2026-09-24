"""Simulate weekly marketing data where the true channel effects are known.

Every scenario uses the same data-generating process (geometric adstock, then
logistic saturation, plus trend, yearly seasonality, a promo control and noise).
Scenarios change only the *spend pattern* and history length, because that is
what decides whether a model can tell the channels apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ["tv", "social", "search"]
CONTROLS = ["promo", "trend"]  # given to every tool
L_MAX = 8

# Ground truth shared by every scenario
TRUE_ALPHA = {"tv": 0.60, "social": 0.30, "search": 0.10}   # adstock decay
TRUE_LAM = {"tv": 2.0, "social": 4.0, "search": 3.0}        # saturation speed
TRUE_BETA = {"tv": 4500.0, "social": 3000.0, "search": 2500.0}  # max weekly sales added
SPEND_SCALE = {"tv": 3000.0, "social": 1500.0, "search": 1200.0}  # peak weekly spend


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    weeks: int = 156
    correlation: float = 0.0      # shared driver across channels, 0..1
    variation: float = 1.0        # 1 = normal week-to-week variation, small = flat
    endogenous: bool = False      # search spend follows demand (seasonality)
    noise_sd: float = 350.0
    tags: tuple = field(default_factory=tuple)


SCENARIOS = {
    s.name: s
    for s in [
        Scenario("clean", "Independent channels, healthy spend variation, 3 years of weekly data."),
        Scenario("correlated", "Channels are planned together: 80% of spend movement is shared.", correlation=0.8),
        Scenario("low_variation", "Always-on budgets with little week-to-week change.", variation=0.25),
        Scenario("short_history", "Only one year of weekly data.", weeks=52),
        Scenario("endogenous", "Search spend chases demand, so it rises when sales would rise anyway.", endogenous=True),
    ]
}


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = L_MAX) -> np.ndarray:
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    return np.convolve(x, w)[: len(x)]


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def _spend_patterns(sc: Scenario, rng: np.random.Generator, season: np.ndarray) -> dict[str, np.ndarray]:
    n = sc.weeks
    # TV: flighted, four to six weeks on, then dark
    tv = np.zeros(n)
    for start in range(0, n, 13):
        tv[start : start + rng.integers(4, 7)] = rng.uniform(0.5, 1.0)
    # Social: always on with occasional bursts
    social = rng.uniform(0.2, 0.6, n)
    social[rng.random(n) > 0.85] += 0.5
    # Search: always on, moderate noise
    search = rng.uniform(0.3, 0.8, n)
    raw = {"tv": tv, "social": social, "search": search}

    if sc.correlation > 0:
        shared = rng.uniform(0.2, 1.0, n)
        shared = np.convolve(shared, np.ones(4) / 4, mode="same")  # smooth planning cycles
        for ch in CHANNELS:
            base = raw[ch]
            on = base > 0 if ch == "tv" else np.ones(n, bool)
            raw[ch] = np.where(on, (1 - sc.correlation) * base + sc.correlation * shared, 0.0)

    if sc.endogenous:
        demand = (season - season.min()) / (season.max() - season.min())
        raw["search"] = 0.25 * raw["search"] + 0.75 * demand + 0.05

    for ch in CHANNELS:
        x = raw[ch]
        if sc.variation != 1.0:
            on = x > 0
            mean = x[on].mean()
            x = np.where(on, mean + sc.variation * (x - mean), 0.0)
        raw[ch] = np.clip(x, 0, None) / max(x.max(), 1e-9)
    return raw


def simulate(sc: Scenario, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Return (weekly data frame, truth dict) for one scenario and seed."""
    rng = np.random.default_rng(seed)
    n = sc.weeks
    t = np.arange(n)
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    season = 900 * np.sin(2 * np.pi * t / 52.18) + 400 * np.cos(2 * np.pi * t / 52.18)

    pattern = _spend_patterns(sc, rng, season)
    spend = {ch: pattern[ch] * SPEND_SCALE[ch] for ch in CHANNELS}

    contrib = {
        ch: TRUE_BETA[ch]
        * logistic_saturation(geometric_adstock(spend[ch] / spend[ch].max(), TRUE_ALPHA[ch]), TRUE_LAM[ch])
        for ch in CHANNELS
    }

    promo = np.zeros(n)
    promo[rng.choice(n, size=max(1, n // 50), replace=False)] = 1
    trend = 4.0 * t
    sales = 10_000 + trend + season + 2_500 * promo + sum(contrib.values()) + rng.normal(0, sc.noise_sd, n)

    df = pd.DataFrame({"date": dates, **spend, "promo": promo, "trend": t / n, "sales": sales})
    truth = {
        "contribution": {ch: float(contrib[ch].sum()) for ch in CHANNELS},
        "spend": {ch: float(spend[ch].sum()) for ch in CHANNELS},
        "roas": {ch: float(contrib[ch].sum() / spend[ch].sum()) for ch in CHANNELS},
        "adstock_alpha": dict(TRUE_ALPHA),
        "saturation_lam": dict(TRUE_LAM),
    }
    return df, truth

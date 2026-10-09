"""Shared helpers for the proof notebooks.

A proof notebook backs one public post with something a reader can rerun. These helpers
reuse the benchmark's data-generating process (bench/simulate.py) so the claim is tested
on the same ground truth as the rest of this repo, and add two things the benchmark does
not need: a "jitter" knob (random week-to-week changes in spend) and a fast, exact fit.

The fit is ordinary least squares with the TRUE adstock and saturation curves supplied, so
only the channel effects are estimated. That isolates one question, "can the data tell the
channels apart?", and is the easiest case a real MMM could face. Real models also have to
learn the curves, so recovery in practice is no better than what you see here.
"""
from __future__ import annotations

import numpy as np

from bench.simulate import (
    CHANNELS,
    SPEND_SCALE,
    TRUE_ALPHA,
    TRUE_BETA,
    TRUE_LAM,
    Scenario,
    _spend_patterns,
    geometric_adstock,
    logistic_saturation,
)

PERIOD = 52.18  # weeks per year, as in bench/simulate.py


def lockstep(weeks: int = 156, shared: float = 0.95, noise_sd: float = 350.0) -> Scenario:
    """Channels planned off one demand forecast: `shared` of the spend movement is common."""
    return Scenario(
        "lockstep",
        f"{shared:.0%} of spend movement follows one shared demand forecast.",
        weeks=weeks,
        correlation=shared,
        noise_sd=noise_sd,
    )


def simulate_jittered(sc: Scenario, seed: int, jitter: float = 0.0) -> dict:
    """One simulated brand. `jitter` multiplies each channel's weekly spend by 1 + U(-jitter, jitter).

    The base spend plan, promo weeks and sales noise depend only on `seed`, so every jitter
    level for the same seed sees the same plan and the same noise. Only the jitter differs.
    Dark weeks (zero spend) stay dark.
    """
    rng = np.random.default_rng(seed)
    n = sc.weeks
    t = np.arange(n)
    season = 900 * np.sin(2 * np.pi * t / PERIOD) + 400 * np.cos(2 * np.pi * t / PERIOD)

    pattern = _spend_patterns(sc, rng, season)
    spend = {ch: pattern[ch] * SPEND_SCALE[ch] for ch in CHANNELS}
    if jitter > 0:
        jrng = np.random.default_rng([seed, 1])  # separate stream: never disturbs the base plan
        for ch in CHANNELS:
            spend[ch] = spend[ch] * (1 + jitter * jrng.uniform(-1, 1, n))

    media = {
        ch: logistic_saturation(geometric_adstock(spend[ch] / spend[ch].max(), TRUE_ALPHA[ch]), TRUE_LAM[ch])
        for ch in CHANNELS
    }
    promo = np.zeros(n)
    promo[rng.choice(n, size=max(1, n // 50), replace=False)] = 1
    noise = rng.normal(0, sc.noise_sd, n)
    sales = 10_000 + 4.0 * t + season + 2_500 * promo + sum(TRUE_BETA[ch] * media[ch] for ch in CHANNELS) + noise
    return {"t": t, "spend": spend, "media": media, "promo": promo, "sales": sales}


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(1 - resid.var() / y.var())


def fit_known_transforms(d: dict) -> dict:
    """OLS of sales on controls plus the true-transformed media. Returns errors, R2 and VIFs."""
    n = len(d["t"])
    ang = 2 * np.pi * d["t"] / PERIOD
    controls = np.column_stack([np.ones(n), d["t"] / n, d["promo"], np.sin(ang), np.cos(ang)])
    media = np.column_stack([d["media"][ch] for ch in CHANNELS])
    X = np.column_stack([controls, media])

    coef, *_ = np.linalg.lstsq(X, d["sales"], rcond=None)
    beta = dict(zip(CHANNELS, coef[controls.shape[1]:]))
    resid = d["sales"] - X @ coef

    # Effect error and ROAS error are the same number: ROAS = beta x (summed media) / spend.
    err = {ch: abs(beta[ch] - TRUE_BETA[ch]) / TRUE_BETA[ch] for ch in CHANNELS}

    vif = {}
    for j, ch in enumerate(CHANNELS):
        k = controls.shape[1] + j
        others = np.delete(X, k, axis=1)
        vif[ch] = float(1 / max(1 - _r2(X[:, k], others), 1e-12))

    return {
        "beta": {ch: float(v) for ch, v in beta.items()},
        "err": {ch: float(v) for ch, v in err.items()},
        "mean_err": float(np.mean(list(err.values()))),
        "r2": float(1 - resid.var() / d["sales"].var()),
        "vif": vif,
        "contribution_est": {ch: float(beta[ch] * d["media"][ch].sum()) for ch in CHANNELS},
        "contribution_true": {ch: float(TRUE_BETA[ch] * d["media"][ch].sum()) for ch in CHANNELS},
    }

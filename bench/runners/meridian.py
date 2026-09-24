"""Google Meridian runner: national (single-geo) model, library default priors.

Meridian's defaults are geometric adstock + Hill saturation with an ROI prior on
each channel. Spend is passed as the media variable, so ROI here is directly
comparable with the other tools' ROAS (incremental sales / spend).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
NAME = "Meridian"


def version() -> str:
    import meridian

    return meridian.__version__


def fit(df: pd.DataFrame, channels: list[str], controls: list[str], seed: int, draws: int, tune: int, chains: int) -> dict:
    import arviz as az
    from meridian.analysis import analyzer
    from meridian.data import data_frame_input_data_builder as dfb
    from meridian.model import model, spec

    warnings.filterwarnings("ignore")
    d = df.copy()
    d["time"] = d["date"].dt.strftime("%Y-%m-%d")
    for ch in channels:
        d[f"{ch}_impr"] = d[ch]

    data = (
        dfb.DataFrameInputDataBuilder(kpi_type="revenue")
        .with_kpi(d, kpi_col="sales")
        .with_controls(d, control_cols=controls)
        .with_media(d, media_cols=[f"{c}_impr" for c in channels], media_spend_cols=channels, media_channels=channels)
        .build()
    )
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(max_lag=8))
    mmm.sample_posterior(n_chains=chains, n_adapt=tune // 2, n_burnin=tune // 2, n_keep=draws, seed=seed)

    roi = np.asarray(analyzer.Analyzer(mmm).roi())  # (chain, draw, channel)
    roas = {ch: roi[..., i].ravel() for i, ch in enumerate(channels)}

    post = mmm.inference_data.posterior
    summ = az.summary(post[["roi_m", "alpha_m", "ec_m"]], kind="diagnostics")
    div = 0
    if "sample_stats" in mmm.inference_data.groups() and "diverging" in mmm.inference_data.sample_stats:
        div = int(mmm.inference_data.sample_stats["diverging"].sum())
    return {
        "roas_draws": roas,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess": float(summ["ess_bulk"].min()),
        "divergences": div,
    }

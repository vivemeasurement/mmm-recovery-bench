"""PyMC-Marketing runner: geometric adstock + logistic saturation, library default priors."""
from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd

NAME = "PyMC-Marketing"


def version() -> str:
    import pymc_marketing

    return pymc_marketing.__version__


def fit(df: pd.DataFrame, channels: list[str], controls: list[str], seed: int, draws: int, tune: int, chains: int) -> dict:
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    warnings.filterwarnings("ignore")
    mmm = MMM(
        date_column="date",
        channel_columns=channels,
        control_columns=controls,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )
    X, y = df.drop(columns="sales"), df["sales"]
    mmm.fit(X, y, chains=chains, cores=min(chains, 4), draws=draws, tune=tune,
            target_accept=0.95, random_seed=seed, progressbar=False)

    contrib = mmm.compute_channel_contribution_original_scale().sum("date")  # (chain, draw, channel)
    roas = {ch: contrib.sel(channel=ch).values.ravel() / df[ch].sum() for ch in channels}

    summ = az.summary(mmm.idata, var_names=["saturation_beta", "adstock_alpha", "saturation_lam"], kind="diagnostics")
    return {
        "roas_draws": roas,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess": float(summ["ess_bulk"].min()),
        "divergences": int(mmm.idata.sample_stats["diverging"].sum()),
    }

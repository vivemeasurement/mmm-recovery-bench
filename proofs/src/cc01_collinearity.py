# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Claim Check #1: does a 1% random change in spend break collinearity?
#
# **The claim.** When every channel follows the same demand forecast, a marketing mix model
# cannot tell the channels apart and gives wrong channel effects. As relayed in MMM Hub issue 70
# (a Monzo case by Ryan O'Sullivan), the repair is not a better prior but a small random
# variation in budgets, as little as 1%.
#
# **What this notebook tests.** The claim itself, on simulated data where the true channel
# effects are known. It does not reproduce the original authors' setup.
#
# **Pre-registered rule (written before the first run, not changed after it).**
# Per seed, the error is the mean absolute percentage error of the three channel effects.
# Take the median across seeds.
#
# | Verdict | Condition |
# |---|---|
# | Premise not reproduced | Lockstep spend with no jitter gives median error below 15%: there was nothing to fix |
# | Confirmed | Median error at 1% jitter is at most half the no-jitter error |
# | Partly confirmed | 1% fails that bar, but a larger jitter (2% or 5%) meets it |
# | Not reproduced | No tested jitter level meets it |
#
# Rerun it: `python proofs/run_proof.py cc01_collinearity --seeds 200`, or open this notebook in Colab.

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
for p in [root, *root.parents]:
    if (p / "bench" / "simulate.py").exists():
        root = p
        break
else:  # e.g. Colab: fetch the repo
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/vivemeasurement/mmm-recovery-bench", "mmm-recovery-bench"], check=True)
    root = Path("mmm-recovery-bench").resolve()
sys.path.insert(0, str(root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proofs.common import fit_known_transforms, lockstep, simulate_jittered

OUT = Path(os.environ.get("PROOF_OUT", root / "proofs" / "cc01_collinearity"))
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Setup (fixed in advance)

# %%
CONFIG = {
    "seeds": int(os.environ.get("PROOF_SEEDS", 200)),
    "weeks": 156,
    "shared": 0.95,                      # share of spend movement that follows one demand forecast
    "jitter_levels": [0.0, 0.01, 0.02, 0.05],
    "noise_sd": 350.0,
    "premise_min_error": 0.15,
    "confirm_ratio": 0.5,
}
print(CONFIG)

# %% [markdown]
# ## Run
# Same seed means the same spend plan, promo weeks and sales noise at every jitter level.
# Only the jitter changes. Jitter multiplies each channel's weekly spend by `1 + U(-j, j)`.

# %%
sc = lockstep(CONFIG["weeks"], CONFIG["shared"], CONFIG["noise_sd"])
rows = []
for seed in range(CONFIG["seeds"]):
    for j in CONFIG["jitter_levels"]:
        fit = fit_known_transforms(simulate_jittered(sc, seed, j))
        rows.append({"seed": seed, "jitter": j, "mean_err": fit["mean_err"], "r2": fit["r2"],
                     "max_vif": max(fit["vif"].values())})
df = pd.DataFrame(rows)

summary = (
    df.groupby("jitter")
    .agg(median_error=("mean_err", "median"),
         p25=("mean_err", lambda s: s.quantile(0.25)),
         p75=("mean_err", lambda s: s.quantile(0.75)),
         share_within_20pct=("mean_err", lambda s: (s <= 0.20).mean()),
         median_max_vif=("max_vif", "median"))
    .reset_index()
)
summary.round(3)

# %% [markdown]
# ## Chart

# %%
levels = CONFIG["jitter_levels"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.boxplot([df[df.jitter == j].mean_err * 100 for j in levels], showfliers=False,
           tick_labels=[f"{j:.0%}" for j in levels], medianprops={"color": "#c0392b", "linewidth": 2})
a1.set_xlabel("Random change in weekly spend (jitter)")
a1.set_ylabel("Typical channel-effect error (%)")
a1.set_title("Effect error by jitter, per seed")
a2.bar([f"{j:.0%}" for j in levels], summary.median_max_vif, color="#34495e")
a2.set_yscale("log")
a2.set_xlabel("Random change in weekly spend (jitter)")
a2.set_ylabel("Median of the worst VIF (log scale)")
a2.set_title("Collinearity (VIF) by jitter")
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Lockstep spend, {CONFIG['seeds']} seeds, true curves supplied; simulated data", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "chart.png", dpi=160)

# %% [markdown]
# ## Verdict (applies the rule written above)

# %%
med = dict(zip(summary.jitter, summary.median_error))
base = med[0.0]
bar = CONFIG["confirm_ratio"] * base
if base < CONFIG["premise_min_error"]:
    verdict, detail = "premise not reproduced", f"lockstep spend alone gave only {base:.0%} median error"
elif med[0.01] <= bar:
    verdict, detail = "confirmed", f"median error fell from {base:.0%} to {med[0.01]:.0%} at 1% jitter"
else:
    ok = [j for j in levels if j > 0.01 and med[j] <= bar]
    if ok:
        verdict, detail = "partly confirmed", f"1% was not enough ({med[0.01]:.0%} vs {base:.0%} with none); {min(ok):.0%} was ({med[min(ok)]:.0%})"
    else:
        verdict, detail = "not reproduced", f"median error stayed at {med[max(levels)]:.0%} even at {max(levels):.0%} jitter (none: {base:.0%})"

result = {
    "id": "cc01_collinearity",
    "title": "Claim Check #1: does a 1% random change in spend break collinearity?",
    "kind": "claim_check",
    "verdict": verdict,
    "detail": detail,
    "seeds": CONFIG["seeds"],
    "config": CONFIG,
    "median_error": {f"{k:.0%}": round(float(v), 4) for k, v in med.items()},
    "median_max_vif": {f"{k:.0%}": round(float(v), 1) for k, v in zip(summary.jitter, summary.median_max_vif)},
}
(OUT / "results.json").write_text(json.dumps(result, indent=2))
print(f"VERDICT: {verdict}. {detail}.")

# %% [markdown]
# ## Limits (read before quoting this)
# - **Simulated data.** True effects are known only because we planted them. Real brands differ.
# - **Best case for the fit.** The true adstock and saturation curves are supplied, so only the
#   channel effects are estimated. A real model must also learn the curves, so it recovers less.
# - **One kind of collinearity.** Spend follows one shared driver with weight 0.95. Other patterns can behave differently.
# - **Jitter is applied to spend and sales respond to it.** In practice that means actually moving the
#   budget, which costs something. This notebook measures the benefit, not the cost.
# - **OLS, not Bayesian.** Priors can help or hurt here; they are not tested.

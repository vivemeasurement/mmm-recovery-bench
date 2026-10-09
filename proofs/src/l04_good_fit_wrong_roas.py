# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lesson 4: a great fit is not a correct channel split
#
# **The idea.** A marketing mix model can track sales almost perfectly and still hand the
# credit to the wrong channels. Fit quality (R-squared, MAPE) says how well the model follows
# the total. It says nothing about how the total is divided. Only a test where the answer is
# known, or a lift test, can check the split.
#
# **What this notebook shows.** On simulated data we plant the true effect of three channels,
# let their spend move together (as it does when budgets are planned off one forecast), fit a
# model, and compare two numbers per seed: how well it fits, and how wrong the split is.
# This is a demonstration of a mechanism, not a test of a published claim.
#
# Rerun it: `python proofs/run_proof.py l04_good_fit_wrong_roas --seeds 200`, or open it in Colab.

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

from bench.simulate import CHANNELS
from proofs.common import fit_known_transforms, lockstep, simulate_jittered

OUT = Path(os.environ.get("PROOF_OUT", root / "proofs" / "l04_good_fit_wrong_roas"))
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {"seeds": int(os.environ.get("PROOF_SEEDS", 200)), "weeks": 156, "shared": 0.95, "noise_sd": 350.0,
          "good_fit_r2": 0.95, "wrong_split_error": 0.25}
print(CONFIG)

# %% [markdown]
# ## Fit many simulated brands
# Spend is planned off one forecast (95% of its movement is shared). The true adstock and
# saturation curves are supplied, so only the channel effects are estimated.

# %%
sc = lockstep(CONFIG["weeks"], CONFIG["shared"], CONFIG["noise_sd"])
rows, example = [], None
for seed in range(CONFIG["seeds"]):
    fit = fit_known_transforms(simulate_jittered(sc, seed, 0.0))
    rows.append({"seed": seed, "r2": fit["r2"], "mean_err": fit["mean_err"]})
    if seed == 0:
        example = fit
df = pd.DataFrame(rows)

good = df.r2 >= CONFIG["good_fit_r2"]
wrong = df.mean_err >= CONFIG["wrong_split_error"]
print(f"Fits with R-squared >= {CONFIG['good_fit_r2']}: {good.mean():.0%} of seeds")
print(f"Median R-squared: {df.r2.median():.3f}")
print(f"Median channel-effect error: {df.mean_err.median():.0%}")
print(f"Good fit AND wrong split (error >= {CONFIG['wrong_split_error']:.0%}): {(good & wrong).mean():.0%} of seeds")

# %% [markdown]
# ## One brand, side by side (seed 0)
# Same fit, two very different stories about which channel did the work.

# %%
tbl = pd.DataFrame({
    "true contribution": example["contribution_true"],
    "model's contribution": example["contribution_est"],
}).round(0)
tbl["error"] = ((tbl["model's contribution"] - tbl["true contribution"]) / tbl["true contribution"]).map("{:+.0%}".format)
print(f"R-squared for this brand: {example['r2']:.3f}")
tbl

# %% [markdown]
# ## Chart

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.scatter(df.r2, df.mean_err * 100, s=14, color="#34495e", alpha=0.7)
a1.axvline(CONFIG["good_fit_r2"], color="#999", lw=1, ls="--")
a1.axhline(CONFIG["wrong_split_error"] * 100, color="#999", lw=1, ls="--")
a1.set_xlabel("Fit quality (R-squared), each dot is one simulated brand")
a1.set_ylabel("Typical channel-effect error (%)")
a1.set_title("Good fit, wrong split")
x = np.arange(len(CHANNELS))
a2.bar(x - 0.2, [example["contribution_true"][c] for c in CHANNELS], 0.4, label="True", color="#34495e")
a2.bar(x + 0.2, [example["contribution_est"][c] for c in CHANNELS], 0.4, label="Model", color="#c0392b")
a2.set_xticks(x, CHANNELS)
a2.set_ylabel("Sales credited to channel")
a2.set_title(f"One brand (seed 0), R-squared {example['r2']:.3f}")
a2.legend(frameon=False)
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Lockstep spend, {CONFIG['seeds']} seeds; simulated data", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "chart.png", dpi=160)

# %%
result = {
    "id": "l04_good_fit_wrong_roas",
    "title": "Lesson 4: a great fit is not a correct channel split",
    "kind": "lesson",
    "verdict": "n/a (lesson)",
    "detail": f"median R-squared {df.r2.median():.2f} with median channel-effect error {df.mean_err.median():.0%}; {(good & wrong).mean():.0%} of seeds crossed R-squared >= {CONFIG['good_fit_r2']} and error >= {CONFIG['wrong_split_error']:.0%}",
    "seeds": CONFIG["seeds"],
    "config": CONFIG,
    "median_r2": round(float(df.r2.median()), 4),
    "median_error": round(float(df.mean_err.median()), 4),
    "share_good_fit_wrong_split": round(float((good & wrong).mean()), 4),
}
(OUT / "results.json").write_text(json.dumps(result, indent=2))
print(result["detail"])
# Read this honestly: the strict "both thresholds" share is small. The defensible claim is the median pair above.

# %% [markdown]
# ## Limits
# - **Simulated data.** The truth is known only because we planted it.
# - **Best case for the fit.** The true curves are supplied. A real model learns them too, so the split is harder, not easier.
# - **In-sample R-squared.** The fit statistic here is measured on the data the model was fitted to.
# - **What would catch it.** Parameter recovery on simulated data, or a lift test on one channel. Fit statistics cannot.

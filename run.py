"""Run the benchmark: every tool x scenario x seed, appended to results/results.csv.

    python run.py                                  # all tools, all scenarios, seed 0
    python run.py --tools pymc --scenarios clean --seeds 0 1 2
    python run.py --quick                          # fewer draws, for a smoke test
"""
from __future__ import annotations

import argparse
import importlib
import platform
import time
from pathlib import Path

import pandas as pd

from bench.score import score
from bench.simulate import CHANNELS, CONTROLS, SCENARIOS, simulate

TOOLS = {"pymc": "bench.runners.pymc_marketing", "meridian": "bench.runners.meridian"}
OUT = Path(__file__).parent / "results" / "results.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tools", nargs="+", default=list(TOOLS), choices=list(TOOLS))
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--quick", action="store_true", help="short chains for a smoke test")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--fresh", action="store_true", help="overwrite results.csv instead of appending")
    a = p.parse_args()

    draws, tune, chains = (200, 200, 2) if a.quick else (a.draws, a.tune, a.chains)
    OUT.parent.mkdir(exist_ok=True)
    done = pd.read_csv(OUT) if OUT.exists() and not a.fresh else pd.DataFrame()

    for tool in a.tools:
        runner = importlib.import_module(TOOLS[tool])
        for sc_name in a.scenarios:
            for seed in a.seeds:
                if not done.empty and ((done.tool == runner.NAME) & (done.scenario == sc_name) & (done.seed == seed)
                                        & (done.draws == draws) & (done.chains == chains)).any():
                    print(f"skip {runner.NAME} / {sc_name} / seed {seed} (already in results)")
                    continue
                df, truth = simulate(SCENARIOS[sc_name], seed)
                t0 = time.time()
                res = runner.fit(df, CHANNELS, CONTROLS, seed=seed, draws=draws, tune=tune, chains=chains)
                secs = time.time() - t0
                rows = score(res["roas_draws"], truth)
                for r in rows:
                    r.update(tool=runner.NAME, tool_version=runner.version(), scenario=sc_name, seed=seed,
                             seconds=round(secs, 1), max_rhat=res.get("max_rhat"), min_ess=res.get("min_ess"),
                             divergences=res.get("divergences"), draws=draws, chains=chains,
                             python=platform.python_version())
                new = pd.DataFrame(rows)
                done = pd.concat([done, new], ignore_index=True)
                done.to_csv(OUT, index=False)
                mape = new.abs_pct_error.mean()
                print(f"{runner.NAME:15s} {sc_name:14s} seed {seed}: ROAS error {mape:6.1%}, "
                      f"covered {new.covered.sum()}/{len(new)}, {secs:5.0f}s")


if __name__ == "__main__":
    main()

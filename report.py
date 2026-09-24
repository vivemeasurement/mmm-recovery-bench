"""Turn results/results.csv into the headline chart, a summary table and the README block."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bench.simulate import SCENARIOS

ROOT = Path(__file__).parent
RES = ROOT / "results"
TOOL_STYLE = {  # fixed order and colour per tool; shape is the secondary encoding
    "PyMC-Marketing": {"color": "#12A4A0", "marker": "o"},
    "Meridian": {"color": "#D9642B", "marker": "D"},
    "Robyn": {"color": "#6B5CC4", "marker": "s"},
}
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e6e5df"
LABEL = {"clean": "Clean", "correlated": "Correlated channels", "low_variation": "Low spend variation",
         "short_history": "Short history (1 yr)", "endogenous": "Spend chases demand"}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(flagged=(df.max_rhat > 1.03) | (df.divergences / (df.draws * df.chains) > 0.01))
    g = df.groupby(["scenario", "tool"])
    runs = df.drop_duplicates(["tool", "scenario", "seed"]).groupby(["scenario", "tool"])
    out = pd.DataFrame({
        "roas_error": g.abs_pct_error.median(),
        "flagged_runs": runs.flagged.sum(),
        "coverage": g.covered.mean(),
        "interval_width": g.rel_interval_width.median(),
        "max_rhat": g.max_rhat.max(),
        "seconds": g.seconds.mean(),
        "runs": g.seed.nunique(),
    }).reset_index()
    order = [s for s in SCENARIOS if s in set(out.scenario)]
    out["scenario"] = pd.Categorical(out.scenario, order, ordered=True)
    return out.sort_values(["scenario", "tool"])


def chart(s: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#9a998f", "xtick.color": MUTED,
                         "ytick.color": INK, "axes.labelcolor": MUTED})
    scen = list(s.scenario.cat.categories)
    tools = [t for t in TOOL_STYLE if t in set(s.tool)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.62 * len(scen) + 1.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.4, 1]}, dpi=120)
    off = {t: ((len(tools) - 1) / 2 - i) * 0.22 for i, t in enumerate(tools)}
    for ax, col, title, fmt in [(axes[0], "roas_error", "Typical ROAS error, median (lower is better)", "{:.0%}"),
                                (axes[1], "coverage", "Truth inside 94% interval (dashed = 94% target)", "{:.0%}")]:
        for t in tools:
            sub = s[s.tool == t].set_index("scenario").reindex(scen)
            y = [len(scen) - 1 - i + off[t] for i in range(len(scen))]
            ax.hlines(y, 0, sub[col], color=GRID, lw=2, zorder=1)
            ok = (sub.flagged_runs == 0).values
            c = TOOL_STYLE[t]["color"]
            ax.scatter(sub[col][ok], [v for v, k in zip(y, ok) if k], s=70, color=c, marker=TOOL_STYLE[t]["marker"],
                       edgecolor="white", linewidth=1.5, zorder=3, label=t)
            ax.scatter(sub[col][~ok], [v for v, k in zip(y, ok) if not k], s=62, facecolor="white", edgecolor=c,
                       marker=TOOL_STYLE[t]["marker"], linewidth=2, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlim(-0.03, 1.08)
    axes[1].axvline(0.94, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
    axes[0].set_xlim(0, max(0.1, s.roas_error.max() * 1.15))
    axes[0].set_yticks(range(len(scen)), [LABEL.get(x, x) for x in reversed(scen)])
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, handletextpad=0.3)
    fig.suptitle("Can the model find the ROAS we planted?", x=0.01, ha="left", fontsize=14, color=INK, weight="bold")
    fig.text(0.01, 0.01, "Hollow marker: sampler warnings in at least one run (R-hat > 1.03 or > 1% divergences).\n"
             "Simulated weekly data · 3 channels · 3 seeds · library-default priors · vivemeasurement/mmm-recovery-bench", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def table(s: pd.DataFrame) -> str:
    lines = ["| Scenario | Tool | Median ROAS error | Truth inside 94% interval | Interval width | Runs with sampler warnings |",
             "|---|---|---:|---:|---:|---:|"]
    for _, r in s.iterrows():
        lines.append(f"| {LABEL.get(r.scenario, r.scenario)} | {r.tool} | {r.roas_error:.0%} | {r.coverage:.0%} | "
                     f"±{r.interval_width / 2:.0%} | {int(r.flagged_runs)} of {r.runs} |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(RES / "results.csv")
    # keep the longest-chain run for each tool / scenario / seed
    df = (df.assign(_n=df.draws * df.chains).sort_values("_n", ascending=False)
            .drop_duplicates(["tool", "scenario", "seed", "channel"]))
    s = summarise(df)
    chart(s, RES / "recovery.png")
    md = table(s)
    (RES / "summary.md").write_text(md + "\n")
    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        txt = re.sub(r"<!-- results:start -->.*<!-- results:end -->",
                     f"<!-- results:start -->\n{md}\n<!-- results:end -->", txt, flags=re.S)
        readme.write_text(txt)
    print(md)


if __name__ == "__main__":
    main()

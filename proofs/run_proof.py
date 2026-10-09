"""Run proof notebooks: turn proofs/src/<id>.py into a notebook, execute it, save the outputs.

    python proofs/run_proof.py cc01_collinearity --seeds 200
    python proofs/run_proof.py all --seeds 5          # quick smoke test of every notebook
    python proofs/run_proof.py --index                # rebuild proofs/index.md from saved results
    python proofs/run_proof.py --summary              # markdown with each notebook's link and verdict

Each run writes proofs/<id>/<id>.ipynb (executed, with outputs), results.json and chart.png.
A notebook is a jupytext "percent" script, so diffs stay readable and the script runs on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "proofs" / "src"
ID_RE = re.compile(r"^[a-z0-9_]+$")
REPO = os.environ.get("GITHUB_REPOSITORY", "vivemeasurement/mmm-recovery-bench")


def ids() -> list[str]:
    return sorted(p.stem for p in SRC.glob("*.py"))


def run(pid: str, seeds: int) -> None:
    import jupytext
    import nbformat
    from nbclient import NotebookClient

    if not ID_RE.match(pid) or not (SRC / f"{pid}.py").exists():
        sys.exit(f"Unknown proof id {pid!r}. Available: {', '.join(ids())}")
    out = ROOT / "proofs" / pid
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PROOF_OUT"] = str(out)
    os.environ["PROOF_SEEDS"] = str(seeds)

    nb = jupytext.read(SRC / f"{pid}.py")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.validator.normalize(nb)
    NotebookClient(nb, timeout=1500, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, out / f"{pid}.ipynb")

    res_path = out / "results.json"
    if not res_path.exists():
        sys.exit(f"{pid}: the notebook ran but did not write results.json")
    res = json.loads(res_path.read_text())
    res["run_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    res_path.write_text(json.dumps(res, indent=2))
    print(f"{pid}: {res['verdict']} ({res['detail']})")


def results() -> list[dict]:
    out = []
    for p in sorted((ROOT / "proofs").glob("*/results.json")):
        out.append(json.loads(p.read_text()))
    return out


def link(pid: str) -> str:
    return f"https://github.com/{REPO}/blob/main/proofs/{pid}/{pid}.ipynb"


def rebuild_index() -> None:
    lines = [
        "# Proof notebooks",
        "",
        "Every Vive Measurement post that makes a claim links to a notebook here. Each one runs on simulated data",
        "with a known truth, and each states its limits. Rerun any of them with",
        "`python proofs/run_proof.py <id> --seeds 200`.",
        "",
        "| Notebook | Kind | Verdict | Seeds | Last run (UTC) |",
        "|---|---|---|---:|---|",
    ]
    for r in results():
        lines.append(f"| [{r['title']}](./{r['id']}/{r['id']}.ipynb) | {r['kind']} | {r['verdict']} | {r['seeds']} | {r.get('run_at_utc', '')} |")
    (ROOT / "proofs" / "index.md").write_text("\n".join(lines) + "\n")


def summary() -> str:
    lines = ["## Proof notebooks", "", "Paste the link into the Source link column (column G) of the Stream tab.", ""]
    for r in results():
        lines += [f"**{r['title']}**", f"- Verdict: {r['verdict']}. {r['detail']}.", f"- Link once pushed to main: {link(r['id'])}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("proof", nargs="?", help="proof id, or 'all'")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--index", action="store_true")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()

    if a.summary:
        print(summary())
        return
    if not a.index:
        if not a.proof:
            p.error("give a proof id, 'all', --index or --summary")
        for pid in (ids() if a.proof == "all" else [a.proof]):
            run(pid, a.seeds)
    rebuild_index()


if __name__ == "__main__":
    main()

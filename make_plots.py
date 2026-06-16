"""
make_plots.py

Generate before/after comparison figures for the SayItSofter upgrade
(context / strict_prompt / self_refine).

Reads:
  - the eval row_level CSV  (LLM-judge metrics + failure_tag_counts), and
  - the run CSV             (objective nli_min per variant).

Produces PNGs under <outdir>:
  1. metrics_3way.png        grouped bars: 6 metrics + overall, per variant
  2. added_information.png    added_information failure-tag count per variant
  3. nli_min.png              mean objective NLI_min per variant

If paths are omitted, the newest files under results/claude_refine3/ are used.

Usage:
  python make_plots.py
  python make_plots.py --eval-csv <row_level_eval.csv> --run-csv <run.csv> --outdir figures
"""

import argparse
import csv
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

VARIANTS = ["context", "strict_prompt", "self_refine"]
VARIANT_LABELS = {"context": "baseline", "strict_prompt": "strict prompt", "self_refine": "self-refine"}
COLORS = {"context": "#9aa0a6", "strict_prompt": "#1a73e8", "self_refine": "#34a853"}
METRICS = ["meaning_preservation", "tone_match", "contextual_appropriateness",
           "naturalness", "conciseness", "formality_fit", "overall"]
METRIC_SHORT = ["meaning", "tone", "context", "natural", "concise", "formality", "overall"]


def newest(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def parse_args():
    p = argparse.ArgumentParser(description="Plot 3-way before/after comparison.")
    p.add_argument("--eval-csv", default=None, help="row_level_eval.csv")
    p.add_argument("--run-csv", default=None, help="run CSV with nli_min")
    p.add_argument("--outdir", default=None, help="output dir for PNGs")
    return p.parse_args()


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_metric(rows, metric):
    xs = [float(r["mean_" + metric]) for r in rows
          if r.get("mean_" + metric) not in (None, "", "None")]
    return st.mean(xs) if xs else 0.0


def plot_metrics(eval_rows, outpath):
    groups = {v: [r for r in eval_rows if r["prompt_variant"] == v] for v in VARIANTS}
    x = range(len(METRICS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, v in enumerate(VARIANTS):
        vals = [mean_metric(groups[v], m) for m in METRICS]
        offs = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offs, vals, width, label=VARIANT_LABELS[v], color=COLORS[v])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(METRIC_SHORT)
    ax.set_ylim(3.0, 5.2)
    ax.set_ylabel("Mean judge score (1-5)")
    ax.set_title("LLM-judge metrics by variant (gpt + gemini)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_added_info(eval_rows, outpath):
    counts = {}
    for v in VARIANTS:
        c = Counter()
        for r in eval_rows:
            if r["prompt_variant"] == v:
                for t, n in json.loads(r["failure_tag_counts"]).items():
                    c[t] += n
        counts[v] = c.get("added_information", 0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([VARIANT_LABELS[v] for v in VARIANTS],
                  [counts[v] for v in VARIANTS],
                  color=[COLORS[v] for v in VARIANTS])
    for b, v in zip(bars, VARIANTS):
        ax.text(b.get_x() + b.get_width() / 2, counts[v] + 0.1, str(counts[v]),
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("added_information failure tags")
    ax.set_title("Added-information failures by variant (lower = better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_nli(run_rows, outpath):
    means = {}
    for v in VARIANTS:
        xs = [float(r["nli_min"]) for r in run_rows
              if r["prompt_variant"] == v and r.get("nli_min") not in (None, "", "None")]
        means[v] = st.mean(xs) if xs else 0.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([VARIANT_LABELS[v] for v in VARIANTS],
                  [means[v] for v in VARIANTS],
                  color=[COLORS[v] for v in VARIANTS])
    for b, v in zip(bars, VARIANTS):
        ax.text(b.get_x() + b.get_width() / 2, means[v] + 0.005, f"{means[v]:.3f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Mean objective NLI_min (higher = more faithful)")
    ax.set_title("Objective meaning preservation (independent NLI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    eval_csv = args.eval_csv or newest("results/claude_refine3/evals/*/row_level_eval.csv")
    run_csv = args.run_csv or newest("results/claude_refine3/runs/*.csv")
    if not eval_csv or not run_csv:
        raise SystemExit("Could not locate eval/run CSVs. Pass --eval-csv and --run-csv.")
    outdir = Path(args.outdir) if args.outdir else Path(eval_csv).parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    eval_rows = load(eval_csv)
    run_rows = load(run_csv)

    plot_metrics(eval_rows, outdir / "metrics_3way.png")
    plot_added_info(eval_rows, outdir / "added_information.png")
    plot_nli(run_rows, outdir / "nli_min.png")

    print(f"eval CSV : {eval_csv}")
    print(f"run  CSV : {run_csv}")
    print("Saved figures:")
    for p in ["metrics_3way.png", "added_information.png", "nli_min.png"]:
        print(f"  {outdir / p}")


if __name__ == "__main__":
    main()

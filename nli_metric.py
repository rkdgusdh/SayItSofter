"""
nli_metric.py

Add an OBJECTIVE meaning-preservation metric to any SayItSofter run CSV.

The first report relied entirely on LLM-as-a-judge scores and itself noted this
as a limitation. This script complements those scores with a model-free (well,
encoder-based) NLI entailment metric between the original message and each
rewrite, then reports the mean by prompt_variant / situation / model.

It works on ANY run CSV that has 'raw_message' (or 'original_message'/'message')
and 'output_text' columns -- including the combined baseline/self_refine CSV
produced by run_refine.py, giving you a clean before/after on an objective metric.

API-only, no GPU. Set HF_TOKEN (HF NLI backend) and/or ANTHROPIC_API_KEY (LLM
fallback). See nli_utils.py.

Usage:
  python nli_metric.py --input-csv results/claude_refine/runs/<run>.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from nli_utils import nli_scores, DEFAULT_HF_NLI_MODEL

RAW_ALIASES = ["raw_message", "original_message", "message"]
OUT_ALIASES = ["output_text", "output", "response"]


def pick(row, aliases, default=""):
    for a in aliases:
        if row.get(a):
            return row[a]
    return default


def parse_args():
    p = argparse.ArgumentParser(description="Compute NLI meaning-preservation metric.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", default=None,
                   help="Where to write the per-row CSV with NLI columns. "
                        "Default: <input>_nli.csv")
    p.add_argument("--nli-model", default=DEFAULT_HF_NLI_MODEL,
                   help="HF NLI model (only used when --nli-backend=hf/auto).")
    p.add_argument("--nli-backend",
                   choices=["gemini", "openai", "anthropic", "hf", "auto"],
                   default="gemini",
                   help="Who scores faithfulness. Default 'gemini' = independent "
                        "of the Claude generator (recommended).")
    p.add_argument("--llm-model", default=None,
                   help="Override the NLI judge model.")
    p.add_argument("--group-by", nargs="+",
                   default=["prompt_variant", "situation"],
                   help="Columns to report mean nli_min by.")
    return p.parse_args()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def main():
    args = parse_args()
    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv) if args.output_csv else in_path.with_name(in_path.stem + "_nli.csv")

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {in_path}")

    enriched = []
    for i, row in enumerate(rows):
        original = pick(row, RAW_ALIASES)
        rewrite = pick(row, OUT_ALIASES)
        scores = nli_scores(original, rewrite, hf_model=args.nli_model,
                            backend=args.nli_backend, llm_model=args.llm_model)
        out = dict(row)
        out.update({
            "nli_entail_o2r": scores["entail_o2r"],
            "nli_entail_r2o": scores["entail_r2o"],
            "nli_min": scores["nli_min"],
            "nli_backend": scores["nli_backend"],
        })
        enriched.append(out)
        print(f"[{i+1}/{len(rows)}] nli_min={scores['nli_min']} "
              f"({scores['nli_backend']}) variant={row.get('prompt_variant','')}")

    fieldnames = list(enriched[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(enriched)

    # Reports
    print("\n=== Overall ===")
    print("  mean nli_min     :", mean([float(r["nli_min"]) for r in enriched if r["nli_min"] not in (None, "")]))
    print("  mean entail_r2o  :", mean([float(r["nli_entail_r2o"]) for r in enriched if r["nli_entail_r2o"] not in (None, "")]),
          "(low => added_information)")
    print("  mean entail_o2r  :", mean([float(r["nli_entail_o2r"]) for r in enriched if r["nli_entail_o2r"] not in (None, "")]),
          "(low => dropped/changed meaning)")

    for key in args.group_by:
        if key not in enriched[0]:
            continue
        groups = defaultdict(list)
        for r in enriched:
            if r["nli_min"] not in (None, ""):
                groups[r.get(key, "")].append(float(r["nli_min"]))
        print(f"\n=== mean nli_min by {key} ===")
        for g, vals in sorted(groups.items()):
            print(f"  {g:<16} n={len(vals):<3} mean_nli_min={mean(vals)}")

    print(f"\nSaved per-row NLI CSV to {out_path}")


if __name__ == "__main__":
    main()

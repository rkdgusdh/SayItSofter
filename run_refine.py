"""
run_refine.py

Faithfulness-aware self-refine rewriting for SayItSofter (Claude, API-only).

Pipeline (per data row):
  1. GENERATE  : produce a first rewrite using the same prompt as run_claude.py.
  2. CRITIQUE  : an LLM critic compares original vs rewrite and flags
                 added_information / meaning_drift; an NLI score (nli_utils)
                 gives an objective bidirectional entailment measure.
  3. REFINE    : if the critic flags an issue OR nli_min < threshold, regenerate
                 the rewrite with the feedback. Repeat up to --max-rounds.

This directly targets the main finding of the first report: meaning preservation
is the weakest metric, and the dominant failures are added_information and
meaning_drift.

For every row we emit TWO rows in the output CSV so the eval can do before/after:
  - prompt_variant="context"      (base_mode=True)  -> the 1-pass baseline
  - prompt_variant="self_refine"  (base_mode=False) -> the refined output

The output CSV INCLUDES raw_message (unlike run_claude.py), so eval.py reads the
original message correctly when judging meaning preservation.

Related work this implements:
  - Self-Refine (Madaan et al., NeurIPS 2023): generate -> self-feedback -> refine
  - Reflexion (Shinn et al., NeurIPS 2023): verbal feedback drives the retry
  - NLI faithfulness checking, in the spirit of SummaC (Laban et al., TACL 2022)

Usage:
  export ANTHROPIC_API_KEY=...
  export HF_TOKEN=...            # optional; enables the HF NLI backend
  python run_refine.py --data data.csv --outdir results/claude_refine
"""

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from nli_utils import nli_scores, DEFAULT_HF_NLI_MODEL

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
# Note: claude-sonnet-4-6 rejects temperature AND top_p together, so we default
# to temperature only (the report's best temperature was 1.0).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = None

GENERATE_PROMPT = (
    "Rewrite the following message to fit the given social context.\n\n"
    "Relation: {relation}\n"
    "Situation: {situation}\n"
    "Target tone: {target_tone}\n"
    "Original message: \"{raw_message}\"\n\n"
    "Keep the original intent and core meaning.\n"
    "Make it sound natural and appropriate for this relation and situation.\n"
    "Prioritize matching the target tone.\n"
    "Do not make it more formal than necessary.\n"
    "Return only the rewritten message."
)

# Single-pass variant with an explicit no-addition instruction. Used as the third
# arm of the 3-way comparison (baseline / strict-prompt / self-refine) to test
# whether the refine LOOP earns its extra cost over just a stronger prompt.
STRICT_PROMPT = (
    "Rewrite the following message to fit the given social context.\n\n"
    "Relation: {relation}\n"
    "Situation: {situation}\n"
    "Target tone: {target_tone}\n"
    "Original message: \"{raw_message}\"\n\n"
    "Hard constraints:\n"
    "- Do NOT add any facts, reasons, excuses, times, or commitments that are not "
    "in the original message.\n"
    "- Do NOT drop or change the original core intent or any important detail.\n"
    "- You MAY adjust greeting, politeness, warmth, and tone freely.\n"
    "Make it natural and matched to the target tone, not more formal than necessary.\n"
    "Return only the rewritten message."
)

CRITIC_PROMPT = (
    "You are a strict faithfulness reviewer for message rewriting.\n\n"
    "Original message: {raw_message}\n"
    "Relation: {relation}\n"
    "Situation: {situation}\n"
    "Target tone: {target_tone}\n"
    "Rewritten message: {rewrite}\n\n"
    "Compare the rewrite ONLY against the original message at the level of factual "
    "and intent content.\n"
    "IMPORTANT: greetings, politeness, apologies-as-tone, warmth, and softening "
    "(e.g. 'Hey', 'I'm sorry', 'thanks', 'if that's okay', 'I hope that's fine') "
    "are EXPECTED in a rewrite. Do NOT flag them as added_information.\n\n"
    "Flag a problem ONLY when:\n"
    "- added_information: the rewrite introduces NEW facts, reasons, excuses, "
    "times, or commitments that are not in the original; OR\n"
    "- meaning_drift: the rewrite changes, weakens, or drops the original core "
    "intent or an important detail.\n\n"
    "Return ONLY valid JSON with this schema:\n"
    "{{\n"
    '  "added_information": boolean,\n'
    '  "meaning_drift": boolean,\n'
    '  "issues": [string],            // short phrases naming each problem span\n'
    '  "fix_instruction": string      // one sentence telling how to fix it\n'
    "}}\n"
    "If the rewrite is faithful, set both booleans to false, issues to [], and "
    "fix_instruction to an empty string."
)

REFINE_PROMPT = (
    "You previously rewrote a message, but a reviewer found faithfulness problems.\n\n"
    "Relation: {relation}\n"
    "Situation: {situation}\n"
    "Target tone: {target_tone}\n"
    "Original message: \"{raw_message}\"\n"
    "Current rewrite: \"{rewrite}\"\n\n"
    "Reviewer feedback:\n"
    "- Problems found: {issues}\n"
    "- Fix instruction: {fix_instruction}\n\n"
    "Produce a corrected rewrite that:\n"
    "- removes ONLY the unsupported factual content the reviewer flagged,\n"
    "- restores any original meaning that was changed or dropped,\n"
    "- KEEPS the warmth, greeting, politeness, and tone of the current rewrite "
    "(do not make it blunt or colder than necessary),\n"
    "- still matches the target tone for this relation and situation,\n"
    "- is not more formal than necessary.\n"
    "Return only the corrected rewritten message."
)


def parse_args():
    p = argparse.ArgumentParser(description="Self-refine rewriting for SayItSofter (Claude).")
    p.add_argument("--data", default="data.csv")
    p.add_argument("--outdir", default="results/claude_refine")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-rounds", type=int, default=2, help="Max refine iterations.")
    p.add_argument("--nli-threshold", type=float, default=0.5,
                   help="Trigger a refine if nli_min < threshold.")
    p.add_argument("--nli-model", default=DEFAULT_HF_NLI_MODEL,
                   help="HF NLI model (only used when --nli-backend=hf/auto).")
    p.add_argument("--nli-backend",
                   choices=["gemini", "openai", "anthropic", "hf", "auto", "off"],
                   default="gemini",
                   help="Who scores faithfulness. Default 'gemini' = independent "
                        "of the Claude generator (recommended). 'off' disables NLI.")
    p.add_argument("--nli-llm-model", default=None,
                   help="Override the NLI judge model (e.g. a specific gemini/gpt model).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    return p.parse_args()


def load_rows(csv_path: Path, limit=None):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    required = {"id", "relation", "situation", "raw_message", "target_tone"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return rows


def stable_hash(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def extract_text(message) -> str:
    return "\n".join(
        getattr(b, "text", "") for b in getattr(message, "content", [])
        if getattr(b, "type", None) == "text"
    ).strip()


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def call_claude(client, model, prompt, temperature, top_p, max_tokens):
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    # This model rejects temperature and top_p together; prefer temperature.
    if temperature is not None:
        body["temperature"] = temperature
    elif top_p is not None:
        body["top_p"] = top_p
    return client.messages.create(**body)


def critique(client, model, row, rewrite):
    prompt = CRITIC_PROMPT.format(
        raw_message=json.dumps(row["raw_message"], ensure_ascii=False),
        relation=row["relation"], situation=row["situation"],
        target_tone=row["target_tone"],
        rewrite=json.dumps(rewrite, ensure_ascii=False),
    )
    # Critic uses low temperature for stable judgments.
    msg = call_claude(client, model, prompt, temperature=0.0, top_p=None, max_tokens=300)
    try:
        obj = parse_json_object(extract_text(msg))
        return {
            "added_information": bool(obj.get("added_information", False)),
            "meaning_drift": bool(obj.get("meaning_drift", False)),
            "issues": [str(x) for x in obj.get("issues", []) if x],
            "fix_instruction": str(obj.get("fix_instruction", "")).strip(),
        }
    except Exception:
        # On parse failure, do not block the pipeline.
        return {"added_information": False, "meaning_drift": False,
                "issues": [], "fix_instruction": ""}


def main():
    args = parse_args()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    (outdir / "jsonl").mkdir(parents=True, exist_ok=True)
    (outdir / "runs").mkdir(parents=True, exist_ok=True)
    rows = load_rows(data_path, args.limit)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"claude_refine_{timestamp}"
    jsonl_path = outdir / "jsonl" / f"{run_id}.jsonl"
    csv_path = outdir / "runs" / f"{run_id}.csv"

    backend = args.nli_backend
    use_nli = backend != "off"
    summary_rows = []

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            gen_prompt = GENERATE_PROMPT.format(**row)

            # --- Step 1: baseline generation ---
            base_msg = call_claude(client, args.model, gen_prompt,
                                   args.temperature, args.top_p, args.max_tokens)
            baseline_text = extract_text(base_msg)

            # --- Variant 2: single-pass "strict" prompt (no-addition instruction) ---
            strict_msg = call_claude(client, args.model, STRICT_PROMPT.format(**row),
                                     args.temperature, args.top_p, args.max_tokens)
            strict_text = extract_text(strict_msg)

            # --- Steps 2-3: critique + refine loop ---
            current = baseline_text
            history = []
            rounds_used = 0
            for _ in range(args.max_rounds):
                crit = critique(client, args.model, row, current)
                nli = (nli_scores(row["raw_message"], current,
                                  hf_model=args.nli_model, backend=backend,
                                  llm_model=args.nli_llm_model)
                       if use_nli else
                       {"entail_o2r": None, "entail_r2o": None, "nli_min": None, "nli_backend": None})

                nli_fail = (use_nli and nli["nli_min"] is not None
                            and nli["nli_min"] < args.nli_threshold)
                needs_fix = crit["added_information"] or crit["meaning_drift"] or nli_fail
                history.append({"round": rounds_used, "rewrite": current,
                                "critique": crit, "nli": nli, "needs_fix": needs_fix})
                if not needs_fix:
                    break

                refine_prompt = REFINE_PROMPT.format(
                    relation=row["relation"], situation=row["situation"],
                    target_tone=row["target_tone"], raw_message=row["raw_message"],
                    rewrite=current,
                    issues="; ".join(crit["issues"]) or "(unspecified)",
                    fix_instruction=crit["fix_instruction"] or "Remove unsupported content and restore original meaning.",
                )
                ref_msg = call_claude(client, args.model, refine_prompt,
                                      args.temperature, args.top_p, args.max_tokens)
                current = extract_text(ref_msg)
                rounds_used += 1

            final_nli = (nli_scores(row["raw_message"], current,
                                    hf_model=args.nli_model, backend=backend,
                                    llm_model=args.nli_llm_model)
                         if use_nli else
                         {"entail_o2r": None, "entail_r2o": None, "nli_min": None, "nli_backend": None})
            base_nli = (nli_scores(row["raw_message"], baseline_text,
                                   hf_model=args.nli_model, backend=backend,
                                   llm_model=args.nli_llm_model)
                        if use_nli else
                        {"entail_o2r": None, "entail_r2o": None, "nli_min": None, "nli_backend": None})
            strict_nli = (nli_scores(row["raw_message"], strict_text,
                                     hf_model=args.nli_model, backend=backend,
                                     llm_model=args.nli_llm_model)
                          if use_nli else
                          {"entail_o2r": None, "entail_r2o": None, "nli_min": None, "nli_backend": None})

            jf.write(json.dumps({
                "run_id": run_id, "row": row,
                "baseline_text": baseline_text, "baseline_nli": base_nli,
                "strict_text": strict_text, "strict_nli": strict_nli,
                "final_text": current, "final_nli": final_nli,
                "rounds_used": rounds_used, "history": history,
                "model": args.model,
                "hyperparameters": {"temperature": args.temperature,
                                    "top_p": args.top_p, "max_tokens": args.max_tokens},
            }, ensure_ascii=False) + "\n")

            # Three output rows for the 3-way comparison (eval breaks down by prompt_variant).
            for variant, text, nli, base_mode in [
                ("context", baseline_text, base_nli, True),
                ("strict_prompt", strict_text, strict_nli, False),
                ("self_refine", current, final_nli, False),
            ]:
                exp_id = stable_hash({"row_id": row["id"], "prompt_variant": variant,
                                      "model": args.model, "run_id": run_id})
                summary_rows.append({
                    "experiment_id": exp_id,
                    "provider": "anthropic",
                    "model": args.model,
                    "row_id": row["id"],
                    "relation": row["relation"],
                    "situation": row["situation"],
                    "raw_message": row["raw_message"],     # <-- included so eval reads the original
                    "target_tone": row["target_tone"],
                    "prompt_variant": variant,
                    "base_mode": base_mode,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "output_text": text,
                    "refine_rounds": rounds_used if variant == "self_refine" else 0,
                    "nli_entail_o2r": nli["entail_o2r"],
                    "nli_entail_r2o": nli["entail_r2o"],
                    "nli_min": nli["nli_min"],
                    "nli_backend": nli["nli_backend"],
                })

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {len(summary_rows)} rows ({len(rows)} items x 3 variants) to {csv_path}")
    print(f"Per-item detail (critique + refine history) in {jsonl_path}")
    print("Next: evaluate before/after with eval.py, e.g.")
    print(f"  python eval.py --input-csv {csv_path} --generator-model claude")


if __name__ == "__main__":
    main()

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TEMPERATURES = [0.2, 0.7]
DEFAULT_TOP_PS = [0.8, 1.0]
DEFAULT_MAX_OUTPUT_TOKENS = [80, 160]

PROMPT_TEMPLATES = {
    "context": (
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
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenAI experiments for SayItSofter.")
    parser.add_argument("--data", default="data.csv", help="Path to input CSV.")
    parser.add_argument("--outdir", default="results/gpt", help="Directory for outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--top-ps", nargs="+", type=float, default=DEFAULT_TOP_PS)
    parser.add_argument("--max-output-tokens", nargs="+", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--prompt-variants", nargs="+", choices=list(PROMPT_TEMPLATES.keys()), default=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--base", action="store_true", help="Use provider/model default hyperparameters by not sending sampling params.")
    return parser.parse_args()


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


def ensure_dirs(outdir: Path):
    (outdir / "jsonl").mkdir(parents=True, exist_ok=True)
    (outdir / "runs").mkdir(parents=True, exist_ok=True)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)


def format_prompt(row, variant):
    return PROMPT_TEMPLATES[variant].format(**row)


def stable_hash(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def try_extract_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def iter_hyperparameter_settings(args):
    if args.base:
        yield None, None, None
        return
    for temperature, top_p, max_output_tokens in iter_hyperparameter_settings(args):
                yield temperature, top_p, max_output_tokens


def main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    rows = load_rows(data_path, args.limit)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"gpt_{timestamp}"
    jsonl_path = outdir / "jsonl" / f"{run_id}.jsonl"
    csv_summary_path = outdir / "runs" / f"{run_id}.csv"
    manifest_path = outdir / "artifacts" / f"{run_id}_manifest.json"

    run_manifest = {
        "provider": "openai",
        "model": args.model,
        "temperatures": [] if args.base else args.temperatures,
        "top_ps": [] if args.base else args.top_ps,
        "max_output_tokens": [] if args.base else args.max_output_tokens,
        "prompt_variants": args.prompt_variants,
        "data_path": str(data_path),
        "row_count": len(rows),
        "started_at_utc": timestamp,
        "base_mode": args.base,
    }

    summary_rows = []

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            for prompt_variant in args.prompt_variants:
                prompt = format_prompt(row, prompt_variant)
                for temperature, top_p, max_output_tokens in iter_hyperparameter_settings(args):
                            request_body = {
                                    "model": args.model,
                                    "input": prompt,
                            }
                            if not args.base:
                                request_body["temperature"] = temperature
                                request_body["top_p"] = top_p
                                request_body["max_output_tokens"] = max_output_tokens
                            item_id = stable_hash({
                                "row_id": row["id"],
                                "prompt_variant": prompt_variant,
                                "base_mode": args.base,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_output_tokens": max_output_tokens,
                                "model": args.model,
                                "base_mode": args.base,
                            })
                            started = time.time()
                            response = client.responses.create(**request_body)
                            latency = round(time.time() - started, 3)
                            output_text = response.output_text.strip()
                            usage = try_extract_usage(response)

                            record = {
                                "experiment_id": item_id,
                                "provider": "openai",
                                "model": args.model,
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "row": row,
                                "prompt_variant": prompt_variant,
                                "prompt_text": prompt,
                                "base_mode": args.base,
                                "hyperparameters": {} if args.base else {
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "max_output_tokens": max_output_tokens,
                                },
                                "output_text": output_text,
                                "usage": usage,
                                "latency_seconds": latency,
                                "raw_response": response.model_dump(),
                            }
                            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                            summary_rows.append({
                                "experiment_id": item_id,
                                "provider": "openai",
                                "model": args.model,
                                "row_id": row["id"],
                                "relation": row["relation"],
                                "situation": row["situation"],
                                "target_tone": row["target_tone"],
                                "prompt_variant": prompt_variant,
                                "base_mode": args.base,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_output_tokens": max_output_tokens,
                                "output_text": output_text,
                                "latency_seconds": latency,
                                "input_tokens": usage.get("input_tokens"),
                                "output_tokens": usage.get("output_tokens"),
                                "total_tokens": usage.get("total_tokens"),
                            })
                            if args.sleep_seconds > 0:
                                time.sleep(args.sleep_seconds)

    with csv_summary_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved OpenAI results to {jsonl_path} and {csv_summary_path}")


if __name__ == "__main__":
    main()

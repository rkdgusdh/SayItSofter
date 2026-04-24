
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

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
    parser = argparse.ArgumentParser(description="Run Claude experiments for SayItSofter.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--outdir", default="results/claude")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="If omitted, do not send temperature and let the provider use its default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="If omitted, do not send top_p and let the provider use its default.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Max output tokens. If omitted, uses script default {DEFAULT_MAX_TOKENS} because Claude requires max_tokens.",
    )
    parser.add_argument("--prompt-variant", choices=list(PROMPT_TEMPLATES.keys()), default="context")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
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


def try_extract_usage(message):
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    out = {}
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def extract_text(message):
    texts = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "text":
            texts.append(getattr(block, "text", ""))
    return "\n".join(texts).strip()


def get_effective_hparams(args):
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS,
        "provider_default_used": {
            "temperature": args.temperature is None,
            "top_p": args.top_p is None,
            "max_tokens": args.max_tokens is None,
        },
    }


def main():
    args = parse_args()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    ensure_dirs(outdir)
    rows = load_rows(data_path, args.limit)

    hparams = get_effective_hparams(args)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"claude_{timestamp}"
    jsonl_path = outdir / "jsonl" / f"{run_id}.jsonl"
    csv_summary_path = outdir / "runs" / f"{run_id}.csv"
    manifest_path = outdir / "artifacts" / f"{run_id}_manifest.json"

    summary_rows = []
    run_manifest = {
        "provider": "anthropic",
        "model": args.model,
        "prompt_variant": args.prompt_variant,
        "data_path": str(data_path),
        "row_count": len(rows),
        "started_at_utc": timestamp,
        "hyperparameters": {
            "temperature": hparams["temperature"],
            "top_p": hparams["top_p"],
            "max_tokens": hparams["max_tokens"],
        },
        "provider_default_used": hparams["provider_default_used"],
        "note": "temperature/top_p are omitted from the request when not provided. max_tokens is always sent because Anthropic Messages API requires it.",
    }

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            prompt = format_prompt(row, args.prompt_variant)

            request_body = {
                "model": args.model,
                "max_tokens": hparams["max_tokens"],
                "messages": [{"role": "user", "content": prompt}],
            }
            if hparams["temperature"] is not None:
                request_body["temperature"] = hparams["temperature"]
            if hparams["top_p"] is not None:
                request_body["top_p"] = hparams["top_p"]

            item_id = stable_hash({
                "row_id": row["id"],
                "prompt_variant": args.prompt_variant,
                "temperature": hparams["temperature"],
                "top_p": hparams["top_p"],
                "max_tokens": hparams["max_tokens"],
                "model": args.model,
                "provider_default_used": hparams["provider_default_used"],
            })

            started = time.time()
            message = client.messages.create(**request_body)
            latency = round(time.time() - started, 3)
            output_text = extract_text(message)
            usage = try_extract_usage(message)

            record = {
                "experiment_id": item_id,
                "provider": "anthropic",
                "model": args.model,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "row": row,
                "prompt_variant": args.prompt_variant,
                "prompt_text": prompt,
                "hyperparameters": {
                    "temperature": hparams["temperature"],
                    "top_p": hparams["top_p"],
                    "max_tokens": hparams["max_tokens"],
                },
                "provider_default_used": hparams["provider_default_used"],
                "output_text": output_text,
                "usage": usage,
                "latency_seconds": latency,
                "raw_response": message.model_dump(),
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary_rows.append({
                "experiment_id": item_id,
                "provider": "anthropic",
                "model": args.model,
                "row_id": row["id"],
                "relation": row["relation"],
                "situation": row["situation"],
                "target_tone": row["target_tone"],
                "prompt_variant": args.prompt_variant,
                "temperature": hparams["temperature"],
                "top_p": hparams["top_p"],
                "max_tokens": hparams["max_tokens"],
                "provider_default_temperature": hparams["provider_default_used"]["temperature"],
                "provider_default_top_p": hparams["provider_default_used"]["top_p"],
                "provider_default_max_tokens": hparams["provider_default_used"]["max_tokens"],
                "output_text": output_text,
                "latency_seconds": latency,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            })

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    with csv_summary_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Claude results to {jsonl_path} and {csv_summary_path}")


if __name__ == "__main__":
    main()

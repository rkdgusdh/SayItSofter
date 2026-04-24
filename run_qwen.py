import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_TEMPERATURES = [0.2, 0.7]
DEFAULT_TOP_PS = [0.8, 1.0]
DEFAULT_MAX_NEW_TOKENS = [80, 160]

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
    parser = argparse.ArgumentParser(description="Run local Qwen experiments for SayItSofter.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--outdir", default="results/qwen")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--top-ps", nargs="+", type=float, default=DEFAULT_TOP_PS)
    parser.add_argument("--max-new-tokens", nargs="+", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt-variants", nargs="+", choices=list(PROMPT_TEMPLATES.keys()), default=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--base", action="store_true", help="Use model defaults by not sending generation hyperparameters.")
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


def resolve_dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_chat_messages(prompt: str):
    return [
        {"role": "system", "content": "You are a careful rewriting assistant."},
        {"role": "user", "content": prompt},
    ]


def load_model_and_tokenizer(model_name: str, device_map: str, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return tokenizer, model


def render_prompt_for_model(tokenizer, prompt: str):
    messages = build_chat_messages(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def iter_hyperparameter_settings(args):
    if args.base:
        yield None, None, None
        return
    for temperature, top_p, max_new_tokens in iter_hyperparameter_settings(args):
                yield temperature, top_p, max_new_tokens


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    ensure_dirs(outdir)
    rows = load_rows(data_path, args.limit)

    torch_dtype = resolve_dtype(args.torch_dtype)
    tokenizer, model = load_model_and_tokenizer(args.model, args.device_map, torch_dtype)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen_{timestamp}"
    jsonl_path = outdir / "jsonl" / f"{run_id}.jsonl"
    csv_summary_path = outdir / "runs" / f"{run_id}.csv"
    manifest_path = outdir / "artifacts" / f"{run_id}_manifest.json"

    summary_rows = []
    run_manifest = {
        "provider": "huggingface_local",
        "model": args.model,
        "temperatures": [] if args.base else args.temperatures,
        "top_ps": [] if args.base else args.top_ps,
        "max_new_tokens": [] if args.base else args.max_new_tokens,
        "prompt_variants": args.prompt_variants,
        "data_path": str(data_path),
        "row_count": len(rows),
        "started_at_utc": timestamp,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "base_mode": args.base,
    }

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            for prompt_variant in args.prompt_variants:
                prompt = format_prompt(row, prompt_variant)
                rendered_prompt = render_prompt_for_model(tokenizer, prompt)
                model_inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
                prompt_token_count = int(model_inputs["input_ids"].shape[1])

                for temperature, top_p, max_new_tokens in iter_hyperparameter_settings(args):
                            item_id = stable_hash({
                                "row_id": row["id"],
                                "prompt_variant": prompt_variant,
                                "base_mode": args.base,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_new_tokens": max_new_tokens,
                                "model": args.model,
                                "base_mode": args.base,
                            })
                            started = time.time()
                            generate_kwargs = {
                                **model_inputs,
                                "pad_token_id": tokenizer.eos_token_id,
                            }
                            if not args.base:
                                generate_kwargs.update({
                                    "max_new_tokens": max_new_tokens,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "do_sample": True,
                                })
                            outputs = model.generate(**generate_kwargs)
                            latency = round(time.time() - started, 3)
                            generated_ids = outputs[0][prompt_token_count:]
                            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                            output_token_count = int(generated_ids.shape[0])

                            record = {
                                "experiment_id": item_id,
                                "provider": "huggingface_local",
                                "model": args.model,
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "row": row,
                                "prompt_variant": prompt_variant,
                                "prompt_text": prompt,
                                "rendered_prompt": rendered_prompt,
                                "base_mode": args.base,
                                "hyperparameters": {} if args.base else {
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "max_new_tokens": max_new_tokens,
                                    "device_map": args.device_map,
                                    "torch_dtype": args.torch_dtype,
                                },
                                "output_text": output_text,
                                "usage": {
                                    "prompt_token_count": prompt_token_count,
                                    "output_token_count": output_token_count,
                                },
                                "latency_seconds": latency,
                                "raw_response": {
                                    "generated_token_ids_preview": generated_ids[:50].tolist(),
                                },
                            }
                            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                            summary_rows.append({
                                "experiment_id": item_id,
                                "provider": "huggingface_local",
                                "model": args.model,
                                "row_id": row["id"],
                                "relation": row["relation"],
                                "situation": row["situation"],
                                "target_tone": row["target_tone"],
                                "prompt_variant": prompt_variant,
                                "base_mode": args.base,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_new_tokens": max_new_tokens,
                                "output_text": output_text,
                                "latency_seconds": latency,
                                "prompt_token_count": prompt_token_count,
                                "output_token_count": output_token_count,
                            })

    with csv_summary_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Qwen results to {jsonl_path} and {csv_summary_path}")


if __name__ == "__main__":
    main()

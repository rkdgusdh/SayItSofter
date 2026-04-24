import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Optional imports are loaded lazily in client builders.

JUDGE_PROVIDER_ORDER = ["gpt", "gemini", "claude", "qwen"]
DEFAULT_MODEL_NAMES = {
    "gpt": "gpt-5.4",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-6",
    "qwen": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

ALLOWED_FAILURE_TAGS = [
    "too_formal",
    "too_casual",
    "meaning_drift",
    "added_information",
    "unnatural",
    "too_verbose",
    "tone_mismatch",
    "context_mismatch",
]

METRICS = [
    "meaning_preservation",
    "tone_match",
    "contextual_appropriateness",
    "naturalness",
    "conciseness",
    "formality_fit",
]

METRIC_WEIGHTS = {
    "meaning_preservation": 0.25,
    "tone_match": 0.20,
    "contextual_appropriateness": 0.20,
    "naturalness": 0.20,
    "conciseness": 0.10,
    "formality_fit": 0.05,
}

CSV_ALIASES = {
    "provider": ["provider"],
    "model": ["model"],
    "row_id": ["row_id", "id"],
    "relation": ["relation"],
    "situation": ["situation"],
    "raw_message": ["raw_message", "message", "original_message"],
    "target_tone": ["target_tone", "tone"],
    "prompt_variant": ["prompt_variant"],
    "output_text": ["output_text", "output", "response"],
    "temperature": ["temperature"],
    "top_p": ["top_p"],
    "max_output_tokens": ["max_output_tokens"],
    "max_new_tokens": ["max_new_tokens"],
    "max_tokens": ["max_tokens"],
    "base_mode": ["base_mode"],
    "experiment_id": ["experiment_id"],
}

JUDGE_PROMPT_TEMPLATE = """You are evaluating a rewritten message for social appropriateness.

Original message: {raw_message}
Relation: {relation}
Situation: {situation}
Target tone: {target_tone}
Rewritten message: {output_text}

Evaluate the rewritten message on a 1-5 integer scale for:
1. meaning_preservation
2. tone_match
3. contextual_appropriateness
4. naturalness
5. conciseness
6. formality_fit

Definitions:
- meaning_preservation: Does it preserve the original intent and core meaning?
- tone_match: Does it match the requested target tone?
- contextual_appropriateness: Is it socially appropriate for the given relation and situation?
- naturalness: Does it sound like something a real person would naturally say?
- conciseness: Is it clear and not unnecessarily wordy?
- formality_fit: Is the level of formality appropriate for this context?

Also assign zero or more failure_tags from this exact list:
{allowed_failure_tags}

Rules:
- Use only integers 1 through 5 for the six scores.
- Do not add tags outside the allowed list.
- If the rewrite is good, failure_tags may be an empty list.
- Keep brief_rationale to one sentence.

Return valid JSON only with this schema:
{{
  "meaning_preservation": int,
  "tone_match": int,
  "contextual_appropriateness": int,
  "naturalness": int,
  "conciseness": int,
  "formality_fit": int,
  "failure_tags": [string],
  "brief_rationale": string
}}
"""


@dataclass
class EvalExample:
    experiment_id: str
    provider: str
    model: str
    row_id: str
    relation: str
    situation: str
    raw_message: str
    target_tone: str
    prompt_variant: str
    output_text: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_generation_tokens: Optional[int]
    base_mode: Optional[bool]
    original_row: Dict[str, Any]


class JudgeClient:
    provider: str
    model_name: str

    def judge(self, prompt: str, max_output_tokens: int) -> Tuple[str, Dict[str, Any], float]:
        raise NotImplementedError


class OpenAIJudgeClient(JudgeClient):
    def __init__(self, model_name: str):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self.provider = "gpt"
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def judge(self, prompt: str, max_output_tokens: int) -> Tuple[str, Dict[str, Any], float]:
        started = time.time()
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        latency = round(time.time() - started, 3)
        raw = response.model_dump() if hasattr(response, "model_dump") else {"response": str(response)}
        text = getattr(response, "output_text", "") or ""
        usage = {}
        if getattr(response, "usage", None):
            usage_obj = response.usage
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
        return text.strip(), {"usage": usage, "raw_response": raw}, latency


class GeminiJudgeClient(JudgeClient):
    def __init__(self, model_name: str):
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self.provider = "gemini"
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def judge(self, prompt: str, max_output_tokens: int) -> Tuple[str, Dict[str, Any], float]:
        from google.genai import types

        started = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency = round(time.time() - started, 3)
        raw = response.model_dump(exclude_none=False) if hasattr(response, "model_dump") else {"response": str(response)}
        usage = {}
        usage_obj = getattr(response, "usage_metadata", None)
        if usage_obj is not None:
            usage = {
                "prompt_token_count": getattr(usage_obj, "prompt_token_count", None),
                "candidates_token_count": getattr(usage_obj, "candidates_token_count", None),
                "total_token_count": getattr(usage_obj, "total_token_count", None),
            }
        text = (getattr(response, "text", "") or "").strip()
        return text, {"usage": usage, "raw_response": raw}, latency


class ClaudeJudgeClient(JudgeClient):
    def __init__(self, model_name: str):
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self.provider = "claude"
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def judge(self, prompt: str, max_output_tokens: int) -> Tuple[str, Dict[str, Any], float]:
        started = time.time()
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = round(time.time() - started, 3)
        raw = message.model_dump() if hasattr(message, "model_dump") else {"response": str(message)}
        texts = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", ""))
        usage = {}
        usage_obj = getattr(message, "usage", None)
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
            }
        return "\n".join(texts).strip(), {"usage": usage, "raw_response": raw}, latency


class QwenJudgeClient(JudgeClient):
    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.provider = "qwen"
        self.model_name = model_name
        self._torch = torch
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map[torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def judge(self, prompt: str, max_output_tokens: int) -> Tuple[str, Dict[str, Any], float]:
        messages = [
            {"role": "system", "content": "You are a careful evaluator that returns only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt

        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.model.device)
        prompt_token_count = int(model_inputs["input_ids"].shape[1])
        started = time.time()
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_output_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        latency = round(time.time() - started, 3)
        generated_ids = outputs[0][prompt_token_count:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        usage = {
            "prompt_token_count": prompt_token_count,
            "output_token_count": int(generated_ids.shape[0]),
        }
        return text, {"usage": usage, "raw_response": {"generated_token_ids_preview": generated_ids[:50].tolist()}}, latency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rewritten outputs with three judge models.")
    parser.add_argument("--input-csv", required=True, help="Path to run CSV file to evaluate.")
    parser.add_argument("--generator-model", required=True, choices=JUDGE_PROVIDER_ORDER, help="Provider that generated the outputs in the CSV.")
    parser.add_argument("--temperature", type=float, default=None, help="Filter to a specific temperature. Omit to evaluate all.")
    parser.add_argument("--top-p", type=float, default=None, help="Filter to a specific top_p. Omit to evaluate all.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Filter to a specific max generation tokens value. Omit to evaluate all.")
    parser.add_argument("--prompt-variant", default=None, help="Optional filter for prompt_variant.")
    parser.add_argument("--relation", default=None, help="Optional filter for relation.")
    parser.add_argument("--situation", default=None, help="Optional filter for situation.")
    parser.add_argument("--target-tone", default=None, help="Optional filter for target_tone.")
    parser.add_argument("--base-mode", choices=["any", "true", "false"], default="any", help="Filter base_mode if present.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of examples after filtering.")
    parser.add_argument("--judge-models", nargs="+", choices=JUDGE_PROVIDER_ORDER, default=None, help="Override judge providers. Default: all except generator.")
    parser.add_argument("--gpt-model-name", default=DEFAULT_MODEL_NAMES["gpt"])
    parser.add_argument("--gemini-model-name", default=DEFAULT_MODEL_NAMES["gemini"])
    parser.add_argument("--claude-model-name", default=DEFAULT_MODEL_NAMES["claude"])
    parser.add_argument("--qwen-model-name", default=DEFAULT_MODEL_NAMES["qwen"])
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--qwen-torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--judge-max-output-tokens", type=int, default=300, help="Max tokens for judge outputs.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per judge call on parse failure or transient errors.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional sleep between judge calls.")
    parser.add_argument("--outdir", default=None, help="Output directory. Default: sibling evals/<run_name>")
    parser.add_argument("--resume", action="store_true", help="Resume by skipping already judged (experiment_id, judge_provider) pairs from raw_judgments.jsonl.")
    return parser.parse_args()


def get_field(row: Dict[str, Any], canonical_name: str, default: Any = None) -> Any:
    for candidate in CSV_ALIASES.get(canonical_name, [canonical_name]):
        if candidate in row and row[candidate] not in (None, ""):
            return row[candidate]
    return default



def normalize_provider(provider_value: Optional[str], model_value: Optional[str]) -> str:
    provider_value = (provider_value or "").lower()
    model_value = (model_value or "").lower()
    if provider_value in {"openai", "gpt"} or model_value.startswith("gpt"):
        return "gpt"
    if provider_value in {"google", "gemini"} or model_value.startswith("gemini"):
        return "gemini"
    if provider_value in {"anthropic", "claude"} or model_value.startswith("claude"):
        return "claude"
    if provider_value in {"huggingface_local", "qwen"} or "qwen" in model_value:
        return "qwen"
    return provider_value or "unknown"



def parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def parse_optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None



def parse_optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None



def rows_from_csv(csv_path: Path) -> List[EvalExample]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    examples: List[EvalExample] = []
    for i, row in enumerate(rows):
        provider = normalize_provider(get_field(row, "provider"), get_field(row, "model"))
        model = str(get_field(row, "model", ""))
        experiment_id = str(get_field(row, "experiment_id", f"row_{i}"))
        raw_message = str(get_field(row, "raw_message", ""))
        output_text = str(get_field(row, "output_text", ""))
        max_gen = (
            parse_optional_int(get_field(row, "max_new_tokens"))
            or parse_optional_int(get_field(row, "max_output_tokens"))
            or parse_optional_int(get_field(row, "max_tokens"))
        )

        examples.append(
            EvalExample(
                experiment_id=experiment_id,
                provider=provider,
                model=model,
                row_id=str(get_field(row, "row_id", f"{i}")),
                relation=str(get_field(row, "relation", "")),
                situation=str(get_field(row, "situation", "")),
                raw_message=raw_message,
                target_tone=str(get_field(row, "target_tone", "")),
                prompt_variant=str(get_field(row, "prompt_variant", "")),
                output_text=output_text,
                temperature=parse_optional_float(get_field(row, "temperature")),
                top_p=parse_optional_float(get_field(row, "top_p")),
                max_generation_tokens=max_gen,
                base_mode=parse_optional_bool(get_field(row, "base_mode")),
                original_row=row,
            )
        )
    return examples



def filter_examples(examples: List[EvalExample], args: argparse.Namespace) -> List[EvalExample]:
    filtered: List[EvalExample] = []
    for ex in examples:
        if ex.provider != args.generator_model:
            continue
        if args.temperature is not None and ex.temperature != args.temperature:
            continue
        if args.top_p is not None and ex.top_p != args.top_p:
            continue
        if args.max_new_tokens is not None and ex.max_generation_tokens != args.max_new_tokens:
            continue
        if args.prompt_variant is not None and ex.prompt_variant != args.prompt_variant:
            continue
        if args.relation is not None and ex.relation != args.relation:
            continue
        if args.situation is not None and ex.situation != args.situation:
            continue
        if args.target_tone is not None and ex.target_tone != args.target_tone:
            continue
        if args.base_mode == "true" and ex.base_mode is not True:
            continue
        if args.base_mode == "false" and ex.base_mode is not False:
            continue
        filtered.append(ex)

    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered



def choose_judges(generator_model: str, judge_models: Optional[List[str]]) -> List[str]:
    if judge_models:
        unique = []
        for provider in judge_models:
            if provider == generator_model:
                continue
            if provider not in unique:
                unique.append(provider)
        if not unique:
            raise ValueError("After excluding the generator model, no judge models remain.")
        return unique
    return [p for p in JUDGE_PROVIDER_ORDER if p != generator_model]



def make_judge_client(provider: str, args: argparse.Namespace) -> JudgeClient:
    if provider == "gpt":
        return OpenAIJudgeClient(args.gpt_model_name)
    if provider == "gemini":
        return GeminiJudgeClient(args.gemini_model_name)
    if provider == "claude":
        return ClaudeJudgeClient(args.claude_model_name)
    if provider == "qwen":
        return QwenJudgeClient(args.qwen_model_name, device_map=args.qwen_device_map, torch_dtype=args.qwen_torch_dtype)
    raise ValueError(f"Unsupported judge provider: {provider}")



def build_judge_prompt(ex: EvalExample) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        raw_message=json.dumps(ex.raw_message, ensure_ascii=False),
        relation=ex.relation,
        situation=ex.situation,
        target_tone=ex.target_tone,
        output_text=json.dumps(ex.output_text, ensure_ascii=False),
        allowed_failure_tags=json.dumps(ALLOWED_FAILURE_TAGS, ensure_ascii=False),
    )



def extract_json_object(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty judge output")
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in judge output")
    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate



def clamp_metric(value: Any) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Metric value is not an int: {value!r}")
    if value < 1 or value > 5:
        raise ValueError(f"Metric value out of range 1-5: {value}")
    return value



def normalize_failure_tags(tags: Any) -> List[str]:
    if tags in (None, ""):
        return []
    if not isinstance(tags, list):
        raise ValueError("failure_tags must be a list")
    normalized = []
    for tag in tags:
        tag = str(tag)
        if tag in ALLOWED_FAILURE_TAGS and tag not in normalized:
            normalized.append(tag)
    return normalized



def parse_judge_json(text: str) -> Dict[str, Any]:
    obj = json.loads(extract_json_object(text))
    parsed = {}
    for metric in METRICS:
        parsed[metric] = clamp_metric(obj.get(metric))
    parsed["failure_tags"] = normalize_failure_tags(obj.get("failure_tags", []))
    parsed["brief_rationale"] = str(obj.get("brief_rationale", "")).strip()
    parsed["overall"] = round(sum(parsed[m] * METRIC_WEIGHTS[m] for m in METRICS), 4)
    return parsed



def safe_mean(values: List[float]) -> Optional[float]:
    return round(float(statistics.mean(values)), 4) if values else None



def safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    return round(float(statistics.pstdev(values)), 4)



def load_completed_pairs(raw_jsonl_path: Path) -> set:
    completed = set()
    if not raw_jsonl_path.exists():
        return completed
    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            exp_id = record.get("experiment_id")
            judge_provider = record.get("judge_provider")
            parse_error = record.get("parse_error", False)
            if exp_id and judge_provider and not parse_error:
                completed.add((exp_id, judge_provider))
    return completed



def aggregate_row_level(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r["experiment_id"]].append(r)

    rows = []
    for exp_id, group in grouped.items():
        first = group[0]
        row = {
            "experiment_id": exp_id,
            "generator_provider": first["generator_provider"],
            "generator_model": first["generator_model"],
            "row_id": first["row_id"],
            "relation": first["relation"],
            "situation": first["situation"],
            "target_tone": first["target_tone"],
            "prompt_variant": first["prompt_variant"],
            "temperature": first["temperature"],
            "top_p": first["top_p"],
            "max_generation_tokens": first["max_generation_tokens"],
            "base_mode": first["base_mode"],
            "raw_message": first["raw_message"],
            "output_text": first["output_text"],
            "judge_count": len(group),
        }

        for metric in METRICS + ["overall"]:
            vals = [float(r[metric]) for r in group if r.get(metric) is not None]
            row[f"mean_{metric}"] = safe_mean(vals)
            row[f"std_{metric}"] = safe_std(vals)

        tags = Counter()
        rationales = []
        for r in group:
            for tag in r.get("failure_tags", []):
                tags[tag] += 1
            if r.get("brief_rationale"):
                rationales.append(f"[{r['judge_provider']}] {r['brief_rationale']}")
            for metric in METRICS + ["overall"]:
                row[f"{r['judge_provider']}_{metric}"] = r.get(metric)
        row["failure_tag_counts"] = json.dumps(tags, ensure_ascii=False, sort_keys=True)
        row["judge_rationales"] = " | ".join(rationales)
        rows.append(row)
    return rows



def summarize_records(records: List[Dict[str, Any]], judge_providers: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_judgments": len(records),
        "judge_providers": judge_providers,
    }

    overall_metrics = {}
    for metric in METRICS + ["overall"]:
        vals = [float(r[metric]) for r in records if r.get(metric) is not None]
        overall_metrics[metric] = {"mean": safe_mean(vals), "std": safe_std(vals)}
    summary["overall_metrics"] = overall_metrics

    by_judge = {}
    for judge in judge_providers:
        subset = [r for r in records if r["judge_provider"] == judge]
        judge_stats = {}
        for metric in METRICS + ["overall"]:
            vals = [float(r[metric]) for r in subset if r.get(metric) is not None]
            judge_stats[metric] = {"mean": safe_mean(vals), "std": safe_std(vals)}
        by_judge[judge] = judge_stats
    summary["by_judge"] = by_judge

    failure_counts = Counter()
    for r in records:
        for tag in r.get("failure_tags", []):
            failure_counts[tag] += 1
    summary["failure_tag_counts"] = dict(failure_counts)

    judge_agreement = {}
    row_level = aggregate_row_level(records)
    for metric in METRICS + ["overall"]:
        vals = [row[f"std_{metric}"] for row in row_level if row.get(f"std_{metric}") is not None]
        judge_agreement[metric] = {"mean_std_across_judges": safe_mean(vals), "max_std": max(vals) if vals else None}
    summary["judge_agreement"] = judge_agreement

    return summary



def summarize_breakdown(row_level_rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in row_level_rows:
        groups[str(row.get(key, ""))].append(row)

    output = {}
    for group_name, rows in groups.items():
        stats = {"n": len(rows)}
        for metric in METRICS + ["overall"]:
            vals = [float(r[f"mean_{metric}"]) for r in rows if r.get(f"mean_{metric}") is not None]
            stats[metric] = {"mean": safe_mean(vals), "std": safe_std(vals)}
        output[group_name] = stats
    return output



def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def render_report(args: argparse.Namespace, examples: List[EvalExample], judge_providers: List[str], summary: Dict[str, Any], row_level_rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(f"- Input CSV: `{args.input_csv}`")
    lines.append(f"- Generator provider: `{args.generator_model}`")
    lines.append(f"- Judge providers: `{', '.join(judge_providers)}`")
    lines.append(f"- Evaluated examples: `{len(examples)}`")
    lines.append("")
    lines.append("## Filters")
    lines.append("")
    lines.append(f"- temperature: `{args.temperature}`")
    lines.append(f"- top_p: `{args.top_p}`")
    lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    lines.append(f"- prompt_variant: `{args.prompt_variant}`")
    lines.append(f"- relation: `{args.relation}`")
    lines.append(f"- situation: `{args.situation}`")
    lines.append(f"- target_tone: `{args.target_tone}`")
    lines.append(f"- base_mode: `{args.base_mode}`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---:|---:|")
    for metric in METRICS + ["overall"]:
        stat = summary["overall_metrics"][metric]
        lines.append(f"| {metric} | {stat['mean']} | {stat['std']} |")
    lines.append("")
    lines.append("## By Judge")
    lines.append("")
    for judge in judge_providers:
        lines.append(f"### {judge}")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|---|---:|---:|")
        for metric in METRICS + ["overall"]:
            stat = summary["by_judge"][judge][metric]
            lines.append(f"| {metric} | {stat['mean']} | {stat['std']} |")
        lines.append("")
    lines.append("## Failure Tags")
    lines.append("")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for tag, count in sorted(summary["failure_tag_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {tag} | {count} |")
    lines.append("")
    lines.append("## Judge Agreement")
    lines.append("")
    lines.append("| Metric | Mean row std | Max row std |")
    lines.append("|---|---:|---:|")
    for metric, stat in summary["judge_agreement"].items():
        lines.append(f"| {metric} | {stat['mean_std_across_judges']} | {stat['max_std']} |")
    lines.append("")
    lines.append("## Breakdown")
    lines.append("")
    for key in ["relation", "situation", "target_tone", "prompt_variant", "base_mode"]:
        lines.append(f"### by {key}")
        lines.append("")
        lines.append("| Group | N | Overall mean | Overall std |")
        lines.append("|---|---:|---:|---:|")
        breakdown = summarize_breakdown(row_level_rows, key)
        for group_name, stat in sorted(breakdown.items()):
            lines.append(f"| {group_name} | {stat['n']} | {stat['overall']['mean']} | {stat['overall']['std']} |")
        lines.append("")
    lines.append("## Lowest-scoring Examples")
    lines.append("")
    worst_rows = sorted(row_level_rows, key=lambda r: (r.get("mean_overall") is None, r.get("mean_overall", 999)))[:10]
    for row in worst_rows:
        lines.append(f"- `{row['experiment_id']}` overall={row.get('mean_overall')} relation={row.get('relation')} situation={row.get('situation')} tone={row.get('target_tone')}")
        lines.append(f"  - original: {row.get('raw_message')}")
        lines.append(f"  - output: {row.get('output_text')}")
        lines.append(f"  - notes: {row.get('judge_rationales')}")
    lines.append("")
    lines.append("## Highest disagreement Examples")
    lines.append("")
    disagreement_rows = sorted(row_level_rows, key=lambda r: (r.get("std_overall") is None, -(r.get("std_overall") or -1)))[:10]
    for row in disagreement_rows:
        lines.append(f"- `{row['experiment_id']}` std_overall={row.get('std_overall')} mean_overall={row.get('mean_overall')}")
        lines.append(f"  - original: {row.get('raw_message')}")
        lines.append(f"  - output: {row.get('output_text')}")
        lines.append(f"  - notes: {row.get('judge_rationales')}")
    lines.append("")
    return "\n".join(lines)



def ensure_output_dir(args: argparse.Namespace) -> Path:
    input_csv = Path(args.input_csv)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        run_name = input_csv.stem
        outdir = input_csv.parent.parent / "evals" / f"{run_name}_eval"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir



def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    input_csv = Path(args.input_csv)
    examples = rows_from_csv(input_csv)
    filtered = filter_examples(examples, args)
    if not filtered:
        raise ValueError("No examples matched the requested filters.")

    judge_providers = choose_judges(args.generator_model, args.judge_models)
    outdir = ensure_output_dir(args)
    raw_jsonl_path = outdir / "raw_judgments.jsonl"
    row_level_csv_path = outdir / "row_level_eval.csv"
    judgment_level_csv_path = outdir / "judgment_level_eval.csv"
    summary_json_path = outdir / "summary.json"
    report_md_path = outdir / "report.md"
    manifest_path = outdir / "manifest.json"

    completed_pairs = load_completed_pairs(raw_jsonl_path) if args.resume else set()

    clients: Dict[str, JudgeClient] = {}
    judgment_records: List[Dict[str, Any]] = []
    if args.resume and raw_jsonl_path.exists():
        with raw_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("parse_error", False):
                    judgment_records.append(rec)

    raw_file_mode = "a" if args.resume and raw_jsonl_path.exists() else "w"
    with raw_jsonl_path.open(raw_file_mode, encoding="utf-8") as raw_f:
        for ex in filtered:
            prompt = build_judge_prompt(ex)
            for judge_provider in judge_providers:
                if (ex.experiment_id, judge_provider) in completed_pairs:
                    continue
                if judge_provider not in clients:
                    clients[judge_provider] = make_judge_client(judge_provider, args)
                client = clients[judge_provider]

                last_error = None
                for attempt in range(args.retries + 1):
                    parse_error = False
                    parsed = None
                    judge_text = ""
                    meta = {}
                    latency = None
                    try:
                        judge_text, meta, latency = client.judge(prompt, args.judge_max_output_tokens)
                        parsed = parse_judge_json(judge_text)
                    except Exception as exc:
                        last_error = repr(exc)
                        parse_error = True

                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "experiment_id": ex.experiment_id,
                        "generator_provider": ex.provider,
                        "generator_model": ex.model,
                        "row_id": ex.row_id,
                        "relation": ex.relation,
                        "situation": ex.situation,
                        "raw_message": ex.raw_message,
                        "target_tone": ex.target_tone,
                        "prompt_variant": ex.prompt_variant,
                        "output_text": ex.output_text,
                        "temperature": ex.temperature,
                        "top_p": ex.top_p,
                        "max_generation_tokens": ex.max_generation_tokens,
                        "base_mode": ex.base_mode,
                        "judge_provider": judge_provider,
                        "judge_model": client.model_name,
                        "judge_latency_seconds": latency,
                        "judge_prompt": prompt,
                        "judge_raw_text": judge_text,
                        "judge_usage": meta.get("usage", {}),
                        "judge_raw_response": meta.get("raw_response", {}),
                        "parse_error": parse_error,
                        "error": last_error,
                    }
                    if parsed is not None:
                        record.update(parsed)
                        raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        raw_f.flush()
                        judgment_records.append(record)
                        completed_pairs.add((ex.experiment_id, judge_provider))
                        break
                    if attempt == args.retries:
                        raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        raw_f.flush()
                    elif args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    successful_judgments = [r for r in judgment_records if not r.get("parse_error", False)]
    if not successful_judgments:
        raise RuntimeError("No successful judgments were collected.")

    row_level_rows = aggregate_row_level(successful_judgments)
    summary = summarize_records(successful_judgments, judge_providers)
    summary["breakdown"] = {
        "relation": summarize_breakdown(row_level_rows, "relation"),
        "situation": summarize_breakdown(row_level_rows, "situation"),
        "target_tone": summarize_breakdown(row_level_rows, "target_tone"),
        "prompt_variant": summarize_breakdown(row_level_rows, "prompt_variant"),
        "base_mode": summarize_breakdown(row_level_rows, "base_mode"),
    }

    write_csv(judgment_level_csv_path, successful_judgments)
    write_csv(row_level_csv_path, row_level_rows)
    report_md = render_report(args, filtered, judge_providers, summary, row_level_rows)
    report_md_path.write_text(report_md, encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "generator_model": args.generator_model,
        "judge_providers": judge_providers,
        "filters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "prompt_variant": args.prompt_variant,
            "relation": args.relation,
            "situation": args.situation,
            "target_tone": args.target_tone,
            "base_mode": args.base_mode,
            "limit": args.limit,
        },
        "n_filtered_examples": len(filtered),
        "n_successful_judgments": len(successful_judgments),
        "output_files": {
            "raw_judgments_jsonl": str(raw_jsonl_path),
            "judgment_level_csv": str(judgment_level_csv_path),
            "row_level_csv": str(row_level_csv_path),
            "summary_json": str(summary_json_path),
            "report_md": str(report_md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved raw judgments to {raw_jsonl_path}")
    print(f"Saved judgment-level CSV to {judgment_level_csv_path}")
    print(f"Saved row-level CSV to {row_level_csv_path}")
    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved report to {report_md_path}")


if __name__ == "__main__":
    main()

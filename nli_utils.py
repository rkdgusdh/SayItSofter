"""
nli_utils.py

API-only (no GPU) faithfulness scoring for SayItSofter.

We measure meaning preservation between the original message and a rewrite as a
bidirectional Natural Language Inference (NLI) entailment score:

    entail_o2r = P(entailment | premise=original,  hypothesis=rewrite)
    entail_r2o = P(entailment | premise=rewrite,   hypothesis=original)
    nli_min    = min(entail_o2r, entail_r2o)

A high nli_min means the two messages entail each other (same information, no
unsupported additions and no dropped content). A low entail_r2o specifically
flags "added_information" (the rewrite says things the original does not), while
a low entail_o2r flags dropped / drifted meaning.

Backends (chosen automatically, in priority order):
  1. "hf"  : HuggingFace Serverless Inference API (cross-encoder NLI model).
             Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN). No GPU needed.
  2. "llm" : Fallback. Ask an Anthropic model to act as an NLI classifier and
             return entailment probabilities. Needs ANTHROPIC_API_KEY.

This module is imported by both run_refine.py and nli_metric.py.
"""

import json
import os
import time
from typing import Dict, Optional

import requests

DEFAULT_HF_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"


def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def _label_to_entailment(scores) -> Optional[float]:
    """Pull the 'entailment' probability out of an HF text-classification result."""
    # HF returns either a list of {label, score} or a nested list.
    flat = scores
    if isinstance(flat, list) and flat and isinstance(flat[0], list):
        flat = flat[0]
    if not isinstance(flat, list):
        return None
    for item in flat:
        if isinstance(item, dict) and str(item.get("label", "")).lower().startswith("entail"):
            return float(item["score"])
    return None


def _hf_entailment(premise: str, hypothesis: str, model: str, retries: int = 2,
                   timeout: float = 30.0) -> Optional[float]:
    """Single-direction entailment probability via HF Inference API, or None on failure."""
    token = _hf_token()
    if not token:
        return None
    url = HF_API_URL.format(model=model)
    headers = {"Authorization": f"Bearer {token}"}
    # Cross-encoder NLI models accept a sentence pair. Try the two payload shapes
    # the serverless API is known to accept across model versions.
    payloads = [
        {"inputs": {"text": premise, "text_pair": hypothesis}},
        {"inputs": [[premise, hypothesis]]},
    ]
    last_err = None
    for attempt in range(retries + 1):
        for payload in payloads:
            try:
                payload = dict(payload)
                payload["options"] = {"wait_for_model": True}
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 503:
                    # model loading; wait and retry
                    last_err = "503 model loading"
                    time.sleep(2.0)
                    continue
                resp.raise_for_status()
                ent = _label_to_entailment(resp.json())
                if ent is not None:
                    return ent
            except Exception as exc:  # noqa: BLE001
                last_err = repr(exc)
        time.sleep(1.0)
    return None


# ---- LLM fallback (Anthropic) -------------------------------------------------

_LLM_NLI_PROMPT = """You are a strict faithfulness checker for message rewriting.

PREMISE (treat as the only ground truth):
{premise}

HYPOTHESIS:
{hypothesis}

Decide whether the HYPOTHESIS is fully supported by the PREMISE at the level of
factual and intent content.
- IGNORE differences in greetings, politeness, tone, formality, and emotional
  softening (e.g. "Hey", "I'm sorry", "thanks", "if that's okay"). These are
  expected in a rewrite and must NOT lower the score.
- ONLY penalize when the HYPOTHESIS adds NEW factual content, reasons, excuses,
  times, or commitments that are not in the PREMISE, or when it changes, weakens,
  or drops the PREMISE's core intent or an important detail.

Return ONLY valid JSON: {{"entailment": <float between 0 and 1>}}
1.0 = fully supported (no new factual content, nothing important changed or dropped)
0.0 = contradicts the premise or adds significant unsupported content."""


# Default judge model per provider (cheap, and different from the Claude generator).
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-5.4",
    "anthropic": "claude-sonnet-4-6",
}

# Lightweight client cache so we don't rebuild SDK clients per call.
_LLM_CLIENTS: Dict[str, object] = {}


def _parse_entail(text: str) -> Optional[float]:
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start:end + 1])
    return max(0.0, min(1.0, float(obj["entailment"])))


def _one_llm_call(prompt: str, provider: str, model: str, client) -> str:
    """Return the raw text from one provider call (may raise)."""
    if provider == "anthropic":
        msg = client.messages.create(model=model, max_tokens=64,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    if provider == "openai":
        resp = client.responses.create(model=model, input=prompt, max_output_tokens=64)
        return (getattr(resp, "output_text", "") or "").strip()
    if provider == "gemini":
        from google.genai import types
        # gemini-2.5-pro cannot disable thinking (budget=0 is invalid) and spends
        # output tokens on thinking, so give pro a large budget; flash turns it off.
        is_pro = "pro" in (model or "").lower()
        cfg = types.GenerateContentConfig(
            max_output_tokens=4096 if is_pro else 64,
            thinking_config=None if is_pro else types.ThinkingConfig(thinking_budget=0),
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return (getattr(resp, "text", "") or "").strip()
    return ""


def _get_llm_client(provider: str, client):
    if client is not None:
        return client
    cached = _LLM_CLIENTS.get(provider)
    if cached is not None:
        return cached
    if provider == "anthropic":
        import anthropic
        cached = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "openai":
        from openai import OpenAI
        cached = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "gemini":
        from google import genai
        cached = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
    _LLM_CLIENTS[provider] = cached
    return cached


def _llm_entailment(premise: str, hypothesis: str, provider: str, model: Optional[str],
                    client=None, retries: int = 2) -> Optional[float]:
    """
    Single-direction entailment via an LLM acting as an NLI classifier.
    provider: 'gemini' | 'openai' | 'anthropic'. Returns None on failure.

    Using a provider OTHER than the generator (Claude) avoids self-evaluation
    bias when scoring faithfulness. Retries on empty/unparseable output (pro
    models can spend the whole budget on thinking and return no text).
    """
    model = model or DEFAULT_LLM_MODELS.get(provider)
    prompt = _LLM_NLI_PROMPT.format(premise=premise, hypothesis=hypothesis)
    try:
        client = _get_llm_client(provider, client)
    except Exception:  # noqa: BLE001
        return None
    for attempt in range(retries + 1):
        try:
            text = _one_llm_call(prompt, provider, model, client)
            if text:
                return _parse_entail(text)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return None


# ---- Public API ---------------------------------------------------------------

def nli_scores(original: str, rewrite: str,
               hf_model: str = DEFAULT_HF_NLI_MODEL,
               backend: str = "gemini",
               llm_model: Optional[str] = None,
               llm_client=None) -> Dict[str, Optional[float]]:
    """
    Bidirectional entailment between original and rewrite.

    backend:
      'gemini' | 'openai' | 'anthropic' : use that LLM as an NLI classifier
      'hf'   : HuggingFace cross-encoder NLI (needs HF_TOKEN)
      'auto' : try hf first, then gemini
      'off'  : skip (returns all None)  [handled by callers]

    Recommended: 'gemini' or 'openai' (a provider different from the Claude
    generator) so faithfulness is scored by an independent model.

    Returns dict with: entail_o2r, entail_r2o, nli_min, nli_backend.
    """
    used = None
    o2r = r2o = None

    if backend in ("auto", "hf"):
        o2r = _hf_entailment(original, rewrite, hf_model)
        r2o = _hf_entailment(rewrite, original, hf_model)
        if o2r is not None and r2o is not None:
            used = "hf"

    if used is None:
        provider = "gemini" if backend in ("auto", "hf") else backend
        if provider in ("gemini", "openai", "anthropic"):
            o2r = _llm_entailment(original, rewrite, provider, llm_model, client=llm_client)
            r2o = _llm_entailment(rewrite, original, provider, llm_model, client=llm_client)
            if o2r is not None and r2o is not None:
                used = provider

    nli_min = min(o2r, r2o) if (o2r is not None and r2o is not None) else None
    return {
        "entail_o2r": round(o2r, 4) if o2r is not None else None,
        "entail_r2o": round(r2o, 4) if r2o is not None else None,
        "nli_min": round(nli_min, 4) if nli_min is not None else None,
        "nli_backend": used,
    }

# SayItSofter

**SayItSofter** is a small research project on **socially contextualized message rewriting** with large language models (LLMs).

The goal is simple: given a short message, a relationship, a situation, and a target tone, the model rewrites the message so that it sounds **more natural, socially appropriate, and polished**, while still preserving the original intent.

---

## Project Goal

This project studies whether LLMs can help users rewrite **sensitive everyday messages** in a socially appropriate way.

We focus on common interpersonal situations such as:

- apology
- refusal
- request
- reschedule

and on different recipient relationships such as:

- friend
- teammate
- professor

The main research question is whether current LLMs can balance:

- **meaning preservation**
- **tone control**
- **contextual appropriateness**
- **naturalness**
- **conciseness**
- **formality fit**

---

## Repository Structure

```text
SayItSofter/
├── data.csv
├── environment.yml
├── run_gpt.py
├── run_claude.py
├── run_gemini.py
├── run_qwen.py
├── eval.py
├── run_refine.py     # Term Project #2: faithfulness-aware rewriting (baseline / strict / self-refine)
├── nli_utils.py      # objective bidirectional NLI faithfulness scoring (API-only)
├── nli_metric.py     # append the NLI meaning-preservation metric to any run CSV
└── make_plots.py     # before/after comparison figures
```

### File descriptions

- **`data.csv`**  
  Input dataset used for generation. Each row contains the social context and the original message to rewrite.

- **`environment.yml`**  
  Conda environment file for reproducing the project setup. It specifies the Python environment and required packages.

- **`run_gpt.py`**  
  Runs generation experiments with the GPT model.

- **`run_claude.py`**  
  Runs generation experiments with the Claude model.

- **`run_gemini.py`**  
  Runs generation experiments with the Gemini model.

- **`run_qwen.py`**  
  Runs generation experiments with the Qwen model.

- **`eval.py`**  
  Evaluates generated outputs using a multi-judge setup and aggregates the results.

---

## Data Format

The input file `data.csv` is expected to include the following columns:

- `id`
- `relation`
- `situation`
- `raw_message`
- `target_tone`

Each example defines:

- who the message is for
- in what situation it is used
- what tone it should have
- what the original user-written message is

---

## Environment Setup

We recommend using **Conda**.

### 1. Create the environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate sayitsofter
```

If your environment name is different, use the name defined inside `environment.yml`.

---

## API Keys

Some scripts require API access.

Set the required API keys as environment variables before running the scripts.

Examples:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GEMINI_API_KEY="your_gemini_key"
```

Do **not** hardcode API keys in the source code.

---

## Running Generation

Example usage:

```bash
python run_gpt.py
python run_claude.py
python run_gemini.py
python run_qwen.py
```

Depending on the script version, some hyperparameters may be configurable through command-line arguments.

---

## Running Evaluation

After generation, run:

```bash
python eval.py
```

The evaluation script is used to score generated outputs and summarize results for analysis.

---

## Evaluation Dimensions

The project evaluates outputs using the following criteria:

- Meaning Preservation
- Tone Match
- Contextual Appropriateness
- Naturalness
- Conciseness
- Formality Fit
- Overall Score

In addition to score-based evaluation, the project also analyzes failure types such as:

- added information
- meaning drift
- context mismatch
- tone mismatch
- unnaturalness

---

## Term Project #2: Faithfulness Upgrade

The first study found that **meaning preservation** was the weakest dimension and
that **added information** / **meaning drift** were the dominant failures: models
improved politeness by inserting content the user never wrote. This upgrade targets
that bottleneck directly, using the strongest base model (Claude) in an API-only
setting (no GPU; Qwen is therefore omitted).

We compare three rewriting strategies and add an objective faithfulness metric:

- **`run_refine.py`** — generates each message with three strategies and writes one
  combined CSV (so `eval.py` can break down before/after by `prompt_variant`):
  - `context` — the original baseline prompt;
  - `strict_prompt` — single-pass prompt that forbids adding facts while allowing free tone change;
  - `self_refine` — a generate → critique → revise loop with a calibrated critic,
    gated by the objective NLI score.
- **`nli_utils.py` / `nli_metric.py`** — a generator-independent meaning-preservation
  metric based on **bidirectional NLI entailment** (scored by Gemini, an independent
  model), complementing the LLM-as-a-judge scores.
- **`make_plots.py`** — produces the before/after comparison figures.

Example:

```bash
export ANTHROPIC_API_KEY=...   # generation (Claude)
export GEMINI_API_KEY=...      # NLI metric (Gemini)
export OPENAI_API_KEY=...      # judge

python run_refine.py --data data.csv --outdir results/claude_refine3
python eval.py --input-csv results/claude_refine3/runs/<run>.csv \
               --generator-model claude --judge-models gpt gemini
python nli_metric.py --input-csv results/claude_refine3/runs/<run>.csv
python make_plots.py
```

**Key finding — prevention beats cure.** A single-pass *strict* prompt reduced
added-information failures the most (9 → 2) and raised objective faithfulness
(NLI 0.806 → 0.950) with no tone loss, outperforming the more expensive
self-refine loop (9 → 4). A calibrated refine loop is best used as a targeted
safety net rather than the primary mechanism.

---

## Reproducibility

This repository is intended to make the generation and evaluation pipeline reproducible.

To reproduce the project:

1. Create the environment from `environment.yml`
2. Prepare API keys
3. Run one or more `run_*.py` scripts
4. Run `eval.py`

---

## Notes

- The dataset in this repository is a project dataset for socially contextualized rewriting experiments.
- Model outputs may vary depending on model version, API behavior, and generation settings.

---

## License

This repository is provided for academic and research use.

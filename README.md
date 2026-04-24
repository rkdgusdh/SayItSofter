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

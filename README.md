# BeyondQuestions (BeQu)

**An open knowledge evaluation benchmark for Large Language Models.**

BeQu systematically measures how well LLMs can elicit factual knowledge as RDF triples, evaluating both **precision** (are the elicited triples correct?) and **recall** (do they cover ground truth knowledge?) across 20+ models, 10,000+ entities, and 7 experimental scenarios.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Experimental Settings](#experimental-settings)
- [Tested Models](#tested-models)
- [Setup](#setup)
- [Usage](#usage)
- [Data Formats](#data-formats)
- [Results](#results)
- [License](#license)

---

## Overview

Traditional NLP benchmarks rely on fixed question-answering pairs. BeQu goes further: instead of asking *specific* questions, it prompts LLMs to freely elicit structured knowledge about entities, then measures coverage and correctness against a reference corpus built from Wikipedia and the web.

**Key capabilities:**
- Prompt LLMs to extract RDF triples (subject, predicate, object) for any set of Wikipedia entities
- Build ground truth corpora from Wikipedia articles and web search results
- Evaluate elicited triples via RAG-based LLM verification (precision) and coverage checking (recall)
- Compare results across models, domains, popularity tiers, and prompt strategies

---

## Architecture

```
BeyondQuestions/
├── BeQu.py                          # Main CLI entry point
├── experiment_tracker.py            # Experiment deduplication
├── combine_results.py               # Result aggregation
├── index.html                       # GitHub Pages results dashboard
│
├── ELICITATION/                     # Knowledge elicitation pipeline
│   ├── main.py                      # OpenAI Batch API orchestration
│   ├── local.py                     # OpenRouter / SCADS API wrapper
│   ├── gpt_kbc.py                   # GPT KBC runner (workspace isolation)
│   ├── template_utils.py            # Jinja2 prompt template loading
│   └── templates/prompts/           # Prompt templates
│       ├── GPTKB.jinja              # Direct triple elicitation
│       ├── GPTKB_2x_repetition.jinja
│       ├── GPTKB_3x_repetition.jinja
│       ├── wikidata_schema.jinja
│       ├── schemaorg_schema.jinja
│       └── LMCRAWL/                 # Two-step elicitation
│           ├── predicates.jinja     # Step 1: elicit predicates
│           └── objects.jinja        # Step 2: elicit objects
│
├── EVALUATION/                      # Triple evaluation pipeline
│   ├── main.py                      # Evaluation coordinator
│   ├── request.py                   # LLM verification (JSON schema output)
│   ├── process_request.py           # Batch evaluation
│   ├── wikipedia_triple_extractor.py # Ground truth construction
│   ├── rag_retriever.py             # Dense embedding retrieval
│   ├── retrieve_passages.py         # Passage utilities
│   ├── wikipedia_utils.py           # Wikipedia API wrapper
│   ├── wikidata_utils.py            # Wikidata API wrapper
│   └── network_utils.py             # Retry decorator
│
├── ENTITY LISTS/                    # Test entity sets
│   ├── RANDOM/                      # 10K random Wikipedia entities
│   ├── DOMAINS/                     # Entities grouped by Wikipedia domain
│   └── POPULARITY/                  # Entities grouped by page-view popularity
│
├── REFERENCE CORPUS/                # Ground truth (Git LFS)
│   ├── random/GT.json.gz
│   ├── domains/GT.json.gz
│   └── popularity/GT.json.gz
│
├── ELICITED TRIPLES/                # Model outputs (20+ models)
└── RESULTS/                         # Evaluation results (~887 MB)
    └── combined_results_manual.csv
```

### Elicitation Pipeline

LLMs are prompted to extract RDF triples for each entity using configurable Jinja2 templates. Supported APIs:

| API | Description |
|-----|-------------|
| `openai_batched` | OpenAI Batch API (50% cost reduction, asynchronous) |
| `openrouter` | Any OpenRouter-served model |
| `scads` | SCADS AI local LLM serving |

Two elicitation strategies are supported:
- **Single-step** (`GPTKB.jinja`): Prompt the model to elicit all triples in one shot
- **Two-step** (`LMCRAWL/`): First elicit predicates, then fill in objects per predicate

### Ground Truth Construction

For each entity, the pipeline:
1. Fetches the full Wikipedia article text
2. Retrieves the top-20 web search results via Brave Search API
3. Uses an LLM to extract RDF triples from all sources
4. Caches everything (text, URLs, triples) in a compressed `GT.json.gz`

### Evaluation Pipeline

**Precision (RAG-based):**
1. For each elicited triple, concatenate subject + predicate + object as a query
2. Retrieve the top-10 most similar passages from the ground truth corpus using dense embeddings (`text-embedding-3-small`)
3. A judge LLM classifies: **entailment** / **contradiction** / **neutral**
4. Precision = fraction of triples classified as entailment

**Recall (Coverage Check):**
1. For each ground truth triple, check if any elicited triple covers it
2. Verify with the judge LLM using all ground truth sources
3. Recall = fraction of ground truth triples covered

Result categories per triple:

| Category | Meaning |
|----------|---------|
| Entailment | Verified as factually correct |
| Contradiction | Contradicts ground truth |
| Neutral | Truth cannot be determined |
| Error | Malformed triple or extraction failure |

---

## Experimental Settings

BeQu supports 7+ configurations to test different aspects of LLM knowledge:

| Setting | Description |
|---------|-------------|
| `random` | 10,000 random Wikipedia entities (baseline) |
| `domains` | 1,000 entities across Wikipedia topic domains |
| `popularity` | Entities bucketed by Wikipedia page-view frequency |
| `ranges` | Vary the number of expected triples per entity |
| `non_existing` | Made-up entities to test hallucination robustness |
| `multiple_prompts` | Run multiple elicitation templates per model |
| `reasoning_effort` | Vary LLM reasoning effort (low / medium / high) |

---

## Tested Models

BeQu has been evaluated on 20+ models:

**Commercial:**
| Provider | Model |
|----------|-------|
| OpenAI | GPT-5, GPT-oss-120b |
| Anthropic | Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5 |
| Google | Gemini 3.1 Pro, Gemini 3 Flash, Gemma 3 (4B/12B/27B) |
| xAI | Grok 4.1 Fast |

**Open / API-Served:**
| Provider | Model |
|----------|-------|
| Meta | Llama-4-Scout-17B-16E-Instruct |
| Mistral AI | mistral-large-2512 |
| DeepSeek | DeepSeek-V3.2 |
| Qwen | Qwen 3.5-27B |
| Moonshot | Kimi-K2.5 |
| MiniMax | MiniMax-M2.5 |

---

## Setup

### Prerequisites

- Python 3.8+
- Git LFS (for reference corpus and large result files)

### Installation

```bash
git clone https://github.com/Knowledge-aware-AI/BeyondQuestions.git
cd BeyondQuestions

# Pull large files (ground truth corpus, detailed results)
git lfs pull

pip install openai anthropic requests pandas loguru tqdm \
            sentence-transformers torch jinja2 fire
```

### API Keys

Set the relevant environment variables for the APIs you intend to use:

```bash
export OPENAI_API_KEY="..."         # OpenAI Batch API
export OPENROUTER_API_KEY="..."     # OpenRouter
export BRAVE_SEARCH_API_KEY="..."   # Web search for ground truth construction
export SCADS_API_KEY="..."          # SCADS AI (optional)
```

---

## Usage

All workflows are driven through the `BeQu.py` CLI (powered by [Fire](https://github.com/google/python-fire)).

### 1 — Build Ground Truth Only

Construct the reference corpus from Wikipedia + web search without running any model:

```bash
python BeQu.py \
  --entities_file_path "ENTITY LISTS/RANDOM/10000_random_entities_wikipedia.json" \
  --ground_truth_dir_path "REFERENCE CORPUS/random/200" \
  --build_ground_truth_only
```

### 2 — Elicit and Evaluate a Single Model

```bash
python BeQu.py \
  --entities_file_path "ENTITY LISTS/RANDOM/10000_random_entities_wikipedia.json" \
  --api openai_batched \
  --model_elicitation gpt-4o \
  --prompt_template_dir_elicitation ELICITATION/templates/prompts/GPTKB.jinja \
  --reasoning_effort_elicitation medium \
  --elicited_triples_dir "ELICITED TRIPLES" \
  --ground_truth_dir_path "REFERENCE CORPUS/random/200" \
  --results_dir_path RESULTS \
  --llm_judge "meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  --sample_size 500 \
  --seed 42
```

### 3 — Skip Elicitation (Evaluate Pre-existing Triples)

```bash
python BeQu.py \
  --skip_elicitation \
  --elicited_triples_dir "ELICITED TRIPLES" \
  --ground_truth_dir_path "REFERENCE CORPUS/random/200" \
  --results_dir_path RESULTS
```

### 4 — Batch Evaluate All Models

```bash
python BeQu.py \
  --evaluate_all_models \
  --elicited_triples_dir "ELICITED TRIPLES" \
  --results_dir_path RESULTS
```

### 5 — Combine Results into a Single CSV

```bash
python combine_results.py --results_dir RESULTS
```

### Key CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--api` | API backend: `openai_batched`, `scads`, `openrouter` | required |
| `--model_elicitation` | Model identifier (e.g. `gpt-4o`) | required |
| `--sample_size` | Entities to sample per experiment | `500` |
| `--seed` | Random seed for reproducibility | `42` |
| `--llm_judge` | Model used as evaluator | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| `--evaluate_by_category` | Report metrics per Wikipedia domain | `False` |
| `--evaluate_by_popularity` | Report metrics per popularity bucket | `False` |
| `--non_existing_entities` | Test robustness with fake entities | `False` |
| `--use_all_prompts` | Run all available prompt templates | `False` |
| `--recover_errors` | Retry failed triple evaluations | `False` |
| `--build_ground_truth_only` | Skip elicitation and evaluation | `False` |
| `--skip_elicitation` | Use existing triples, evaluate only | `False` |
| `--skip_evaluation` | Elicit triples only, skip evaluation | `False` |

---

## Data Formats

### Entity List (input JSON)

```json
[
  {"title": "Albert Einstein", "length": 5974},
  {"title": "Marie Curie", "length": 4801}
]
```

### Elicited Triples (output CSV)

```
subject,predicate,object
Albert_Einstein,instanceOf,physicist
Albert_Einstein,birthPlace,Ulm
Albert_Einstein,fieldOfWork,theoretical physics
```

### Evaluation Results (per-model CSV)

```
Model,Metric,Setting,Total #Triples,Entailment,Contradiction,Neutral
claude-sonnet-4-6,Precision (RAG),random,500,320,15,165
```

### Combined Results (`combined_results_manual.csv`)

```
Model,Setting,Domain,Entailment_F1,Entailment_Precision,Entailment_Recall
claude-sonnet-4-6,random,,0.85,0.87,0.83
```

---

## Results

Aggregated results are available in `RESULTS/combined_results_manual.csv` and visualized on the [GitHub Pages dashboard](https://knowledge-aware-ai.github.io/BeyondQuestions/).

Individual per-model evaluation outputs are stored under `RESULTS/{model}/{setting}/` and include:

| File | Contents |
|------|----------|
| `results.csv` | Aggregate precision / recall metrics |
| `results_by_category.csv` | Metrics broken down by Wikipedia domain |
| `results_by_popularity.csv` | Metrics broken down by popularity bucket |
| `results_detailed.csv` | Per-triple entailment labels (Git LFS) |

---

## Design Notes

- **Experiment deduplication**: Configuration hashes (SHA-256) are stored in `.experiment_tracking.json` so runs are never repeated accidentally.
- **Memory efficiency**: Evaluation results are streamed to CSV incrementally rather than held in memory.
- **Cost efficiency**: OpenAI Batch API cuts inference costs by 50%; dense RAG retrieval minimizes the number of LLM judge calls.
- **Parallelism**: Up to 5 models can be evaluated concurrently; elicitation threads are configurable.
- **Reproducibility**: Fix `--seed` and `--sample_size` to reproduce any published result exactly.

---

## License

This project is released under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE) license. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

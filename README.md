# BeyondQuestions (BeQu)

**An open knowledge evaluation benchmark for Large Language Models.**

BeQu evaluates what LLMs *actually know* by prompting them to freely surface structured knowledge about entities, then verifying every generated statement against a reference corpus built from Wikipedia and the web. Unlike fixed Q&A benchmarks, BeQu measures both **precision** (are the elicited triples correct?) and **recall** (how much of the reference knowledge does the model cover?).

> Paper under peer review at ARR May 2026. All data, code, and elicited triples are available in this repository.

---

## Table of Contents

- [Overview](#overview)
- [Website](#website)
- [Architecture](#architecture)
- [Experimental Settings](#experimental-settings)
- [Datasets](#datasets)
- [Tested Models](#tested-models)
- [Key Findings](#key-findings)
- [Setup](#setup)
- [Usage](#usage)
- [Data Formats](#data-formats)
- [Results](#results)
- [License](#license)

---

## Overview

Traditional NLP benchmarks rely on fixed question-answering pairs, which introduces *availability bias*: facts not explicitly queried are invisible to evaluation, even when a model possesses them. BeQu goes further by prompting LLMs to freely emit (subject, predicate, object) triples about Wikipedia entities, then measuring coverage and correctness against a per-entity reference corpus.

**Key capabilities:**
- Prompt LLMs to generate knowledge triples for any set of Wikipedia entities
- Build reference corpora from Wikipedia articles and web search results
- Evaluate elicited triples via RAG-based LLM verification (precision) and coverage checking (recall)
- Compare results across 20 models, multiple prompt formats, reasoning effort levels, entity popularity, and knowledge domains

---

## Website

The repository includes a self-contained interactive website (`index.html`) with six sections:

### Leaderboard
The primary view. Shows all 20 models ranked by Entailment F1, with sortable columns for F1, Precision, Recall, and Contradiction rate. Results can be filtered by commercial vs. open-weight models and searched by name. The view includes:
- A **Precision–Recall scatter plot** for all 20 models (no model achieves both high precision and high recall — a clean frontier emerges)
- A **Key Findings** panel summarising the four main experimental takeaways
- A **F1 by knowledge domain** bar chart for three models (GPT-5.4, DeepSeek V3.2, Llama 4 Scout 17B) across 10 domains
- **Six experiment tabs**: Overall ranking · By reasoning effort · By domain · By prompt format · By triple range · By entity popularity

### Methodology
Explains the four-step pipeline: entity selection → knowledge elicitation → reference corpus construction → two-way verification. Covers evaluation scale details: 500 sampled triples per model per direction, top-10 RAG retrieval, `text-embedding-3-small` embeddings, Llama 4 Scout as the NLI judge (90% agreement with human labels).

### Datasets
Describes the four entity lists:

| # | Name | Size | Purpose |
|---|------|------|---------|
| 01 | Random entities | 10,000 | Primary benchmark; hard-filtered + LLM-judged |
| 02 | Domain-balanced | 10 × 100 | Per-domain F1 analysis across 10 Wikipedia categories |
| 03 | Non-existent entities | 10 | Hallucination probe (hand-crafted fictional entities) |
| App. | Popularity tiers | 3 × 200 | Effect of entity popularity on model performance |

Also shows a hallucination bar chart for the non-existent entity experiment: GPT-5.4 fully abstains (0 triples); Llama 4 Scout generates 32 triples over 7 entities; DeepSeek V3.2 generates 131 triples over 4 entities.

### About
Background on the benchmark motivation and design philosophy.

### Entity Lists
An interactive browser for the four Wikipedia entity sets used across BeQu experiments. Clicking a dataset card fetches the corresponding JSON from GitHub and renders a searchable, paginated table (50 rows per page). Four datasets are available:

| Card | Dataset | Columns |
|------|---------|---------|
| Dataset 01 | Random entities (10,000) | Entity |
| Dataset 02 | By domain (10 × 100) | Entity · Wikidata ID |
| Appendix E | By popularity (3 × ~200) | Entity · Popularity · Wikidata statements · Wikidata ID |
| Dataset 03 | Non-existent entities (10) | Entity |

The Domains dataset exposes subset chips for each of the 10 domains (Person, Organization, Location, …). The Popularity dataset exposes chips for each tier (High 66–100%, Mid 33–66%, Low 0–33%). Entity titles link to Wikipedia; Wikidata IDs link to Wikidata.

### Elicited Triples
An interactive data browser that loads the raw `elicited_triples.csv` files directly from the repository via the GitHub raw content API. Select any model from the sidebar, then choose an experiment or subset to browse the full triple table (subject · predicate · object) with live search and pagination (50 rows per page). All 20 models and their respective experiment subsets are available.

To use the website, open `index.html` in any modern browser. The Entity Lists and Elicited Triples sections fetch data from GitHub and therefore require an internet connection.

---

## Architecture

```
BeyondQuestions/
├── BeQu.py                          # Main CLI entry point
├── experiment_tracker.py            # Experiment deduplication
├── combine_results.py               # Result aggregation
│
├── ELICITATION/                     # Knowledge elicitation pipeline
│   ├── main.py                      # OpenAI Batch API orchestration
│   ├── local.py                     # OpenRouter / local API wrapper
│   ├── gpt_kbc.py                   # GPT KBC runner (workspace isolation)
│   ├── template_utils.py            # Jinja2 prompt template loading
│   └── templates/prompts/           # Prompt templates
│       ├── GPTKB.jinja              # Direct triple elicitation (default)
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
│   ├── wikipedia_triple_extractor.py # Reference construction
│   ├── rag_retriever.py             # Dense embedding retrieval
│   ├── retrieve_passages.py         # Passage utilities
│   ├── wikipedia_utils.py           # Wikipedia API wrapper
│   ├── wikidata_utils.py            # Wikidata API wrapper
│   └── network_utils.py             # Retry decorator
│
├── ENTITY LISTS/                    # Test entity sets
│   ├── RANDOM/                      # 10K random Wikipedia entities
│   ├── DOMAINS/                     # Entities grouped by domain
│   └── POPULARITY/                  # Entities grouped by page-view popularity
│
├── REFERENCE CORPUS/
│   ├── random/GT.json.gz
│   ├── domains/GT.json.gz
│   └── popularity/GT.json.gz
│
├── ELICITED TRIPLES/                # Model outputs (20 models × experiments)
├── RESULTS/                         # Evaluation results
│   └── combined_results_manual.csv
└── index.html                       # Interactive website (single file, no build step)
```

### Elicitation Pipeline

LLMs are prompted to generate (subject, predicate, object) triples for each entity using configurable Jinja2 templates. The default prompt is the GPTKB style; seven additional templates cover schema-constrained, repeated, and two-step elicitation variants.

### Reference Corpus Construction

For each entity, the pipeline:
1. Fetches the full Wikipedia article text
2. Retrieves the top-20 web search results via Brave Search API
3. Uses an LLM to extract triples from all sources
4. Caches everything (text, URLs, triples) in a JSON file

### Evaluation Pipeline

**Precision (RAG-based):**
1. For each elicited triple, concatenate subject + predicate + object as a query
2. Retrieve the top-10 most similar passages from the reference corpus (`text-embedding-3-small`)
3. A judge LLM (Llama 4 Scout) classifies: **entailment** / **contradiction** / **neutral**
4. Precision = fraction of triples classified as entailment

**Recall (Coverage Check):**
1. For each ground truth triple, check if any elicited triple covers it
2. Verify with the judge LLM using all ground truth sources
3. Recall = fraction of ground truth triples covered

NLI label meanings:

| Label | Meaning |
|-------|---------|
| Entailment | Verified as factually correct |
| Contradiction | Contradicts reference corpus |
| Neutral | Truth cannot be determined from reference |

---

## Experimental Settings

BeQu covers five experimental configurations:

| # | Setting | Description |
|---|---------|-------------|
| 01 | `random` | 200-entity sample from the 10K random list — the primary BeQu ranking |
| 02 | `reasoning_effort` | Low / medium / high reasoning effort on Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro |
| 03 | `domains` | 100 entities per domain; includes a non-existent entity hallucination probe |
| 04 | `prompts` | Eight prompt templates (GPTKB, LMCRAWL, Schema.org, Wikidata schema, and repetition variants) run on Kimi K2.5 |
| 05 | `ranges` | Vary the number of expected triples per entity (six range buckets) |
| App. | `popularity` | Three popularity tiers (0–33%, 33–66%, 66–100% by Wikidata statement count) |

---

## Datasets

### Random entities (10,000)
Randomly sampled Wikipedia article titles passing hard filters: minimum article length (≥ 2,000 chars), minimum Wikidata statement count (≥ 10), non-disambiguation page. Survivors are then judged by an LLM for informativeness, ambiguity, and suitability (Likert ≥ 3 on all axes). This is the primary benchmark dataset.

### Domain-balanced (10 × 100)
100 entities per domain across: *person, organisation, location, event, work of art, artifact, scientific concept, cultural concept, animal, plant*. Single-domain entities only, enabling per-domain F1 analysis.

### Non-existent entities (10)
Hand-crafted plausible and absurd fictional entities — e.g. *Valdora Strait, U-Bahn Dresden, iPhone 19 Pro, Helios Prize for Digital Arts, Gulf of Varennes*. No reference corpus; the test is whether models abstain from generating triples at all.

### Popularity tiers (3 × 200)
The random subset partitioned into low / mid / high popularity buckets by Wikidata statement count. Used in a supplementary experiment.

---

## Tested Models

BeQu has been evaluated on 20 models:

**Commercial:**
| Provider | Model | ID |
|----------|-------|----|
| OpenAI | GPT-5.4 | `gpt-5.4-2026-03-05` |
| OpenAI | GPT-5 Mini | `gpt-5-mini-2025-08-07` |
| OpenAI | GPT-5 Nano | `gpt-5-nano-2025-08-07` |
| OpenAI | GPT-OSS-120B | `openai_gpt-oss-120b` |
| Anthropic | Claude Opus 4.6 | `anthropic_claude-opus-4.6` |
| Anthropic | Claude Sonnet 4.6 | `anthropic_claude-sonnet-4.6` |
| Anthropic | Claude Haiku 4.5 | `anthropic_claude-haiku-4.5` |
| Google DeepMind | Gemini 3.1 Pro | `google_gemini-3.1-pro-preview` |
| Google DeepMind | Gemini 3 Flash | `google_gemini-3-flash-preview` |
| Google DeepMind | Gemini 3.1 Flash Lite | `google_gemini-3.1-flash-lite-preview` |
| xAI | Grok 4.1 Fast | `x-ai_grok-4.1-fast` |
| MiniMax | MiniMax M2.5 | `MiniMaxAI_MiniMax-M2.5` |

**Open / API-Served:**
| Provider | Model | ID |
|----------|-------|----|
| Google | Gemma 3 4B | `google_gemma-3-4b-it` |
| Google | Gemma 3 12B | `google_gemma-3-12b-it` |
| Google | Gemma 3 27B | `google_gemma-3-27b-it` |
| Meta AI | Llama 4 Scout 17B | `meta-llama_Llama-4-Scout-17B-16E-Instruct` |
| Mistral AI | Mistral Large | `mistralai_mistral-large-2512` |
| DeepSeek | DeepSeek V3.2 | `deepseek-ai_DeepSeek-V3.2` |
| Alibaba | Qwen3.5 27B | `qwen_qwen3.5-27b` |
| Moonshot AI | Kimi K2.5 | `moonshotai_kimi-k2.5` |

---

## Key Findings

**Finding 01 — Benchmark durability.** All models are far from saturating open-ended knowledge generation. Entailment F1 spans 0.171–0.473 across the 20 models. Open-source models are competitive: Kimi K2.5 ranks 2nd overall.

**Finding 02 — Reasoning has negligible effect.** Unlike most NLP tasks, reasoning effort makes almost no difference for open-ended knowledge expression. All nine model–effort combinations cluster within F1 0.400–0.484.

**Finding 03 — Schemas vs. creativity.** Schema enforcement boosts precision (Schema.org reaches 77.2%) but significantly lowers recall to only 14.0%. Open-ended prompts win on F1.

**Finding 04 — Hard-wired operating points.** Explicit precision–recall steering is limited. Prompt repetition (3×) gives a modest recall gain (27.2 → 31.6%), but further repetitions degrade performance sharply.

---

## Setup

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone [ANONYMIZED]
cd BeyondQuestions

# Pull large files (reference corpus, detailed results)
git lfs pull

pip install openai anthropic requests pandas loguru tqdm \
            sentence-transformers torch jinja2 fire
```

---

## Usage

All workflows are driven through the `BeQu.py` CLI.

### 1 — Build Reference Corpus Only

```bash
python BeQu.py \
  --entities_file_path "path/to/json/file" \
  --ground_truth_dir_path "path/to/output" \
  --build_ground_truth_only
```

### 2 — Elicit and Evaluate a Single Model

```bash
python BeQu.py \
  --entities_file_path "path/to/json/file" \
  --api [YOUR_API] \
  --model_elicitation [MODEL] \
  --prompt_template_dir_elicitation ELICITATION/templates/prompts/ \
  --reasoning_effort_elicitation [EFFORT] \
  --elicited_triples_dir "ELICITED TRIPLES" \
  --ground_truth_dir_path "path/to/reference" \
  --results_dir_path RESULTS \
  --llm_judge [JUDGE] \
  --sample_size 500 \
  --seed 42
```

### 3 — Skip Elicitation (Evaluate Pre-existing Triples)

```bash
python BeQu.py \
  --skip_elicitation \
  --elicited_triples_dir "ELICITED TRIPLES" \
  --ground_truth_dir_path "path/to/reference" \
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
subject,predicate,object,subject_name
Albert_Einstein,instanceOf,physicist,Albert Einstein
Albert_Einstein,birthPlace,Ulm,Albert Einstein
Albert_Einstein,fieldOfWork,theoretical physics,Albert Einstein
```

### Evaluation Results (per-model CSV)

```
Model,Metric,Setting,Total #Triples,Entailment,Contradiction,Neutral
claude-sonnet-4-6,Precision (RAG),random,500,320,15,165
```

### Combined Results (`combined_results_manual.csv`)

```
Model,Setting,Domain,Entailment_F1,Entailment_Precision,Entailment_Recall
claude-sonnet-4-6,random,,0.4316,0.646,0.324
```

---

## Results

Aggregated results are in `RESULTS/combined_results_manual.csv`. Individual per-model outputs are under `RESULTS/{model}/{setting}/`:

| File | Contents |
|------|----------|
| `results.csv` | Aggregate precision / recall metrics |
| `results_by_category.csv` | Metrics broken down by Wikipedia domain |
| `results_by_popularity.csv` | Metrics broken down by popularity bucket |
| `results_detailed.csv` | Per-triple entailment labels (Git LFS) |

**Overall leaderboard (Entailment F1, random setting, 200 entities):**

| Rank | Model | F1 | Precision | Recall |
|------|-------|----|-----------|--------|
| 1 | Claude Opus 4.6 | 0.473 | 59.6% | 39.2% |
| 2 | Kimi K2.5 | 0.450 | 54.4% | 38.4% |
| 3 | Gemini 3 Flash | 0.444 | 47.8% | 41.4% |
| 4 | Claude Sonnet 4.6 | 0.432 | 64.6% | 32.4% |
| 5 | GPT-5.4 | 0.421 | 76.6% | 29.0% |
| 6 | Gemini 3.1 Pro | 0.401 | 79.2% | 26.8% |
| 7 | Mistral Large | 0.396 | 53.4% | 31.4% |
| 8 | Gemini 3.1 Flash Lite | 0.383 | 64.6% | 27.2% |
| 9 | DeepSeek V3.2 | 0.372 | 46.0% | 31.2% |
| 10 | GPT-5 Mini | 0.353 | 62.6% | 24.6% |
| 11 | Grok 4.1 Fast | 0.338 | 53.2% | 24.8% |
| 12 | MiniMax M2.5 | 0.323 | 60.8% | 22.0% |
| 13 | GPT-OSS-120B | 0.318 | 41.8% | 25.6% |
| 14 | Claude Haiku 4.5 | 0.299 | 61.4% | 19.8% |
| 15 | Qwen3.5 27B | 0.296 | 49.0% | 21.2% |
| 16 | Gemma 3 27B | 0.282 | 33.4% | 24.4% |
| 17 | GPT-5 Nano | 0.271 | 88.0% | 16.0% |
| 18 | Gemma 3 12B | 0.254 | 38.4% | 19.0% |
| 19 | Llama 4 Scout 17B | 0.253 | 63.2% | 15.8% |
| 20 | Gemma 3 4B | 0.171 | 36.4% | 11.2% |

---

## Design Notes

- **Experiment deduplication**: Configuration hashes (SHA-256) are stored in `.experiment_tracking.json` so runs are never repeated accidentally.
- **Memory efficiency**: Evaluation results are streamed to CSV incrementally rather than held in memory.
- **Reproducibility**: Fix `--seed` and `--sample_size` to reproduce any published result exactly.

---

## License

This project is released under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE) license. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

Please cite our work: [TBD upon publication]

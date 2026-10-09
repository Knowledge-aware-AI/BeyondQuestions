# BeyondQuestions (BeQu)

**An open knowledge evaluation benchmark for Large Language Models.**

BeQu evaluates what LLMs *actually know* by prompting them to freely surface structured knowledge about entities, then verifying every generated statement against a reference corpus built from Wikipedia and the web. Unlike fixed Q&A benchmarks, BeQu measures both **precision** (are the elicited triples correct?) and **recall** (how much of the reference knowledge does the model cover?).

> Paper under peer review. All data, code, and elicited triples are available in this repository.
> **v2.0 (October 2026):** all experiments were rerun on 500 random entities (Exp1: 5 independent runs), every setting is evaluated on 1,000 sampled triples per direction, and the judge is now Gemma 4 26B-A4B-it. The results of the earlier version (200 entities, Llama 4 Scout judge) are in the git history.

---

## Table of Contents

- [Website](#website)
- [Repository layout](#repository-layout)
- [Pipeline](#pipeline)
- [Experiments](#experiments)
- [Datasets](#datasets)
- [Tested models](#tested-models)
- [Key findings](#key-findings)
- [Results](#results)
- [Setup and usage](#setup-and-usage)
- [Data formats](#data-formats)
- [License](#license)

---

## Website

`index.html` is a self-contained interactive website (also served via GitHub Pages):

- **Leaderboard**: all 20 models ranked by entailment F1 (mean over the 5 Exp1 runs), sortable by precision, recall and contradiction rate, with a precision–recall scatter, the key findings, F1 by knowledge domain, and one tab per experiment.
- **Methodology** and **Datasets**: the pipeline, the evaluation scale, the four entity lists, and the hallucination probe on non-existent entities.
- **Entity Lists**, **Reference Corpora**, **Elicited Triples**, **Detailed Results**: data browsers that load the files of this repository from GitHub (internet connection required). *Detailed Results* shows, for any setting and up to 3 models, every sampled triple of an entity coloured by the judge's label.

---

## Repository layout

```
BeyondQuestions/
├── BeQu.py                      # CLI entry point: elicitation, reference corpus, evaluation
├── experiment_tracker.py        # experiment deduplication
├── combine_results.py           # aggregation of results.csv files
├── combine_results_popularity.py
├── compute_stats.py             # triple statistics
├── BeQu_ALL_RESULTS.csv / .md   # ALL experiments, settings and models in one table (181 rows)
├── elicitation/                 # elicitation pipeline + prompt templates
│   └── templates/
│       ├── prompts/             # GPTKB (default), LMCRAWL (two-step), Wikidata / Schema.org schema (± predicate list)
│       └── extra_prompt_formats/ # additional formats not used in the paper's experiments
├── eval/                        # evaluation pipeline (reference corpus, RAG retrieval, NLI judge)
├── ENTITY_LISTS/                # RANDOM (10K pool + the 500 used), DOMAINS (10 × 100, 100 non-existent), POPULARITY
├── REFERENCE_CORPUS/            # per-entity reference corpora (GT.json.gz, Git LFS) + index.json for the website
│   ├── random/500/              # the 500 entities of Exp1, 2, 4, 5
│   ├── random/10k/
│   ├── domains/
│   └── popularity/
├── elicited_triples/
│   ├── exp1/<run 1-5>/<model>/random/medium/
│   ├── exp2/<model>/random/<low|medium|high>/
│   ├── exp3/<model>/<domains|non-existent>/
│   ├── exp4/<model>/prompts/<prompt>/
│   ├── exp5/<model>/ranges/<range>/
│   ├── additional_exp2_popularity/<model>/popularity/<low|mid|high>/
│   └── gpt_evolution_experiment/  # Additional Experiment 1 (GPT family evolution)
├── RESULTS/                     # same layout; one <judge>_rag_eval/ folder per setting, plus expN/expN_summary.csv/.md
└── index.html                   # website (single file, no build step)
```

Each elicited-triples folder holds `elicited_triples.csv`, plus `empty_entity_results.jsonl` (entities for which the model returned no triples) and `entity_extraction_errors.jsonl` where applicable. Each results folder holds `results.csv` (precision / recall counts), `results_by_category.csv`, `results_detailed.csv` (one row per judged triple with the judge's label and explanation) and `results_index.json` (compact version used by the website).

---

## Pipeline

1. **Elicitation.** Each model is prompted (via OpenRouter) to emit (subject, predicate, object) triples about an entity. The default prompt is GPTKB; the templates are Jinja2 files in `elicitation/templates/prompts/`.
2. **Reference corpus.** For each entity: the full Wikipedia article plus up to 20 web documents (Brave Search API). An LLM extracts reference triples from all sources; the full texts are split into passages and serve as the retrieval corpus.
3. **Precision.** For each sampled elicited triple, the top-10 passages are retrieved (`text-embedding-3-small`), and the judge labels the triple **entailed** / **contradicted** / **neutral**.
4. **Recall.** For each sampled reference triple, the judge checks whether the model's full set of elicited triples entails it.

Sampling: 1,000 triples per direction per setting (entity first, then triple; seed 42). Judge: **Gemma 4 26B-A4B-it**. F1 is the harmonic mean of entailment precision and entailment recall.

---

## Experiments

| # | Question | Models | Entities | Setting |
|---|---|---|---|---|
| Exp1 | Overall ranking | all 20 | 500 random | GPTKB prompt, medium reasoning, **5 independent runs** |
| Exp2 | Reasoning effort | Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro | 500 random | low / medium / high (medium = Exp1 run 1) |
| Exp3 | Knowledge domains, hallucination | GPT-5.4, DeepSeek V3.2, Llama 4 Scout | 10 domains × 100, + 100 non-existent | medium reasoning |
| Exp4 | Prompt format | GPT-5.4, DeepSeek V3.2, Llama 4 Scout | 500 random | 6 prompts: GPTKB, LMCRAWL, Wikidata / Schema.org schema with and without predicate list |
| Exp5 | Requested number of triples | GPT-5.4, DeepSeek V3.2, Llama 4 Scout | 500 random | 7 ranges (e.g. 50–100 triples for popular entities, 5–10 for less popular) |
| Add. 1 | Evolution across the GPT family | 9 GPT models, GPT-3.5 Turbo … GPT-5.5 | 200 random | GPTKB prompt; judged by Llama 4 Scout, 500 triples per direction (run in June 2026, before the main rerun) |
| Add. 2 | Entity popularity | GPT-5.4, DeepSeek V3.2, Llama 4 Scout | 3 buckets × 200, by Wikidata statement count | judged by Llama 4 Scout, 500 triples per direction and bucket (June 2026) |

**Reasoning setting of Exp4 and Exp5.** These two experiments were run without a reasoning parameter, i.e. at the API default, which is *no reasoning* for GPT-5.4 and DeepSeek V3.2 (Llama 4 Scout does not reason). Comparisons inside Exp4 or Exp5 are fair; their absolute scores are not directly comparable with Exp1 (GPT-5.4 with the same GPTKB prompt: F1 0.340 in Exp4 vs 0.378 in Exp1).

---

## Datasets

| Name | Size | Purpose |
|---|---|---|
| Random entities | 10,000 pool; 500 used | Wikipedia titles passing hard filters (≥ 2,000 characters, ≥ 10 Wikidata statements, no disambiguation page) and an LLM judge for informativeness, ambiguity and suitability (Likert ≥ 3). Primary benchmark. |
| Domain-balanced | 10 × 100 | person, organisation, location, event, work of art, artifact, scientific concept, cultural concept, animal, plant |
| Non-existent entities | 100 | Plausible and absurd fictional entities (e.g. *Valdora Strait, U-Bahn Dresden, iPhone 19 Pro*); no reference corpus — tests whether models abstain |
| Popularity tiers | 3 × 200 | Low / mid / high popularity by Wikidata statement count |

---

## Tested models

All models were queried through OpenRouter.

| Provider | Model | OpenRouter ID | Type |
|---|---|---|---|
| Alibaba | Qwen3.5 27B | `qwen/qwen3.5-27b` | open weights |
| Anthropic | Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | commercial |
| Anthropic | Claude Opus 4.6 | `anthropic/claude-opus-4.6` | commercial |
| Anthropic | Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | commercial |
| DeepSeek | DeepSeek V3.2 | `deepseek/deepseek-v3.2` | open weights |
| Google | Gemma 3 12B | `google/gemma-3-12b-it` | open weights |
| Google | Gemma 3 27B | `google/gemma-3-27b-it` | open weights |
| Google | Gemma 3 4B | `google/gemma-3-4b-it` | open weights |
| Google DeepMind | Gemini 3 Flash | `google/gemini-3-flash-preview` | commercial |
| Google DeepMind | Gemini 3.1 Flash Lite | `google/gemini-3.1-flash-lite` | commercial |
| Google DeepMind | Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | commercial |
| Meta AI | Llama 4 Scout 17B | `meta-llama/llama-4-scout` | open weights |
| MiniMax | MiniMax M2.5 | `minimax/minimax-m2.5` | open weights |
| Mistral AI | Mistral Large 3 | `mistralai/mistral-large-2512` | open weights |
| Moonshot AI | Kimi K2.5 | `moonshotai/kimi-k2.5` | open weights |
| OpenAI | GPT-5 Mini | `openai/gpt-5-mini` | commercial |
| OpenAI | GPT-5 Nano | `openai/gpt-5-nano` | commercial |
| OpenAI | GPT-5.4 | `openai/gpt-5.4` | commercial |
| OpenAI | GPT-OSS-120B | `openai/gpt-oss-120b` | open weights |
| xAI | Grok 4.3 | `x-ai/grok-4.3` | commercial |

---

## Key findings

1. **Benchmark durability.** No model saturates open-ended knowledge generation: mean F1 spans 0.086–0.378. Open-weight models are competitive (Kimi K2.5 ranks 4th).
2. **Reasoning effort barely matters.** GPT-5.4 and Gemini 3.1 Pro change by less than 0.03 F1 between low and high effort. Claude Opus 4.6 is the exception (0.277 → 0.391), mainly because it answers more entities at higher effort.
3. **Schemas vs. creativity.** Predicate lists raise precision (Schema.org: 77.3% for GPT-5.4) but cut recall (11.7%); open-ended prompts give the best F1 for every model.
4. **Hard-wired operating points.** Requesting 20× more triples changes GPT-5.4's output only 2.3× (11 → 26 triples per entity) and its F1 by at most 0.05.
5. **No model consistently abstains on non-existent entities.** GPT-5.4 answers 22 of 100 (213 triples), DeepSeek V3.2 43 (1,944 triples), Llama 4 Scout 91 (617 triples).

---

## Results

`BeQu_ALL_RESULTS.csv` contains every experiment, setting and model (Exp1 per run; Exp3 per domain and overall). `RESULTS/expN/expN_summary.csv/.md` give the per-experiment summary tables.

**Exp1 leaderboard** (500 random entities, mean ± SD over 5 runs; contradiction = share of sampled elicited triples contradicted by the reference):

| Rank | Model | F1 | Precision | Recall | Contradiction |
|---|---|---|---|---|---|
| 1 | GPT-5.4 | 0.378 ± 0.010 | 67.1% | 26.4% | 4.7% |
| 2 | Gemini 3.1 Pro | 0.370 ± 0.017 | 70.9% | 25.0% | 2.8% |
| 3 | Gemini 3 Flash | 0.370 ± 0.008 | 39.3% | 34.9% | 4.1% |
| 4 | Kimi K2.5 | 0.356 ± 0.009 | 52.8% | 26.9% | 5.9% |
| 5 | Claude Opus 4.6 | 0.345 ± 0.017 | 58.2% | 24.5% | 5.8% |
| 6 | Claude Sonnet 4.6 | 0.328 ± 0.018 | 60.2% | 22.5% | 7.0% |
| 7 | Gemini 3.1 Flash Lite | 0.323 ± 0.016 | 59.6% | 22.2% | 5.9% |
| 8 | Mistral Large 3 | 0.307 ± 0.006 | 46.9% | 22.9% | 7.6% |
| 9 | DeepSeek V3.2 | 0.296 ± 0.012 | 37.5% | 24.6% | 7.4% |
| 10 | GPT-5 Mini | 0.292 ± 0.006 | 61.2% | 19.2% | 2.0% |
| 11 | Grok 4.3 | 0.234 ± 0.013 | 73.2% | 14.0% | 4.2% |
| 12 | MiniMax M2.5 | 0.232 ± 0.013 | 52.3% | 14.9% | 8.5% |
| 13 | GPT-OSS-120B | 0.222 ± 0.007 | 21.7% | 22.8% | 8.0% |
| 14 | Qwen3.5 27B | 0.220 ± 0.012 | 33.1% | 16.6% | 10.5% |
| 15 | Claude Haiku 4.5 | 0.207 ± 0.010 | 64.8% | 12.4% | 7.2% |
| 16 | Gemma 3 27B | 0.192 ± 0.009 | 22.3% | 16.9% | 10.2% |
| 17 | Gemma 3 12B | 0.162 ± 0.009 | 16.4% | 16.1% | 7.4% |
| 18 | GPT-5 Nano | 0.160 ± 0.017 | 82.7% | 8.9% | 2.5% |
| 19 | Llama 4 Scout 17B | 0.117 ± 0.010 | 44.5% | 6.7% | 18.2% |
| 20 | Gemma 3 4B | 0.086 ± 0.004 | 18.7% | 5.6% | 10.8% |

### Notes on the data
- `results_detailed.csv` omits the retrieved passages (to keep the repository small); passages can be regenerated from the reference corpus with `--top_k 10`.
- Reasoning traces are not included.
- Six settings have 1,999 instead of 2,000 judged rows: one triple containing an invalid Unicode character could not be written. Their `results.csv` metrics are computed on all 2,000.
- `REFERENCE_CORPUS/*/GT.json.gz` are stored with Git LFS (`git lfs pull`).

---

## Setup and usage

```bash
git clone [ANONYMIZED]
cd BeyondQuestions
git lfs pull
pip install openai requests pandas loguru tqdm sentence-transformers torch jinja2 fire
```

All workflows go through `BeQu.py` (Python Fire CLI; `python BeQu.py --help` lists every option).

```bash
# build the reference corpus only
python BeQu.py --entities_file_path ENTITY_LISTS/RANDOM/EXPERIMENTS_500_random_entities_wikipedia.json \
  --ground_truth_dir_path REFERENCE_CORPUS/random/500 --build_ground_truth_only

# elicit and evaluate one model (Exp1 setting)
python BeQu.py --entities_file_path ENTITY_LISTS/RANDOM/EXPERIMENTS_500_random_entities_wikipedia.json \
  --api openrouter --model_elicitation openai/gpt-5.4 --reasoning_effort_elicitation medium \
  --prompt_template_dir_elicitation elicitation/templates/prompts/ \
  --elicited_triples_dir elicited_triples --ground_truth_dir_path REFERENCE_CORPUS/random/500 \
  --results_dir_path RESULTS --llm_judge google/gemma-4-26B-A4B-it --sample_size 1000 --seed 42

# evaluate existing triples only
python BeQu.py --skip_elicitation --elicited_triples_dir elicited_triples \
  --ground_truth_dir_path REFERENCE_CORPUS/random/500 --results_dir_path RESULTS
```

API keys are read from the environment: `OPENROUTER_API_KEY` (elicitation and embeddings), the judge endpoint's key and base URL (default: a university-hosted inference service; or `--llm_judge_api openrouter`), `OPENAI_API_KEY` for the OpenAI Batch API, and `BRAVE_API_KEY` only for building new reference corpora.

---

## Data formats

**Entity list** (`ENTITY_LISTS/...json`): `[{"title": "Albert Einstein", "length": 5974}, ...]`

**Elicited triples** (`elicited_triples.csv`): `subject,predicate,object,subject_name`

**Judged triples** (`results_detailed.csv`): one row per sampled triple with the metric (precision / recall), the judge's label (a = entailment, b = contradiction, c = neutral) and its explanation.

---

## License

This project is released under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE) license. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

Please cite our work: [TBD upon publication]

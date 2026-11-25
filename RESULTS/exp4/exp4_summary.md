# exp4: prompt templates

_Generated 2026-11-25 19:42 by ANALYSIS/make_all_tables.py. Precision / Recall = entailment ratio of the 1000-triple precision / recall sample (judge: gemma-4-26B-A4B-it, RAG); F1 = harmonic mean._

18 of 18 rows evaluated


## F1

| Model | GPTKB | LMCRAWL | schemaorg_schema | schemaorg_schema_no_constraints | wikidata_schema | wikidata_schema_no_constraints |
|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.361 | 0.305 | 0.216 | 0.327 | 0.189 | 0.244 |
| llama-4-scout | 0.114 | 0.198 | 0.124 | 0.164 | 0.139 | 0.147 |
| gpt-5.4 | 0.366 | 0.296 | 0.205 | 0.338 | 0.220 | 0.310 |

## Precision

| Model | GPTKB | LMCRAWL | schemaorg_schema | schemaorg_schema_no_constraints | wikidata_schema | wikidata_schema_no_constraints |
|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.567 | 0.392 | 0.750 | 0.578 | 0.715 | 0.495 |
| llama-4-scout | 0.517 | 0.439 | 0.635 | 0.557 | 0.586 | 0.482 |
| gpt-5.4 | 0.714 | 0.517 | 0.839 | 0.709 | 0.825 | 0.738 |

## Recall

| Model | GPTKB | LMCRAWL | schemaorg_schema | schemaorg_schema_no_constraints | wikidata_schema | wikidata_schema_no_constraints |
|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.265 | 0.250 | 0.126 | 0.228 | 0.109 | 0.162 |
| llama-4-scout | 0.064 | 0.128 | 0.069 | 0.096 | 0.079 | 0.087 |
| gpt-5.4 | 0.246 | 0.207 | 0.117 | 0.222 | 0.127 | 0.196 |

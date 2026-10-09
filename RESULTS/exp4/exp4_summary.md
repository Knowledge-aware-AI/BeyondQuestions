# exp4: prompt templates

_Generated 2026-10-06 13:24 by ANALYSIS/make_all_tables.py. Precision / Recall = entailment ratio of the 1000-triple precision / recall sample (judge: gemma-4-26B-A4B-it, RAG); F1 = harmonic mean._

18 of 18 rows evaluated


## F1

| Model | GPTKB | LMCRAWL | schemaorg_schema | schemaorg_schema_no_constraints | wikidata_schema | wikidata_schema_no_constraints |
|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.300 | 0.255 | 0.212 | 0.297 | 0.185 | 0.224 |
| llama-4-scout | 0.112 | 0.191 | 0.123 | 0.159 | 0.137 | 0.141 |
| gpt-5.4 | 0.340 | 0.270 | 0.203 | 0.319 | 0.217 | 0.295 |

## Precision

| Model | GPTKB | LMCRAWL | schemaorg_schema | schemaorg_schema_no_constraints | wikidata_schema | wikidata_schema_no_constraints |
|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.345 | 0.260 | 0.676 | 0.428 | 0.605 | 0.361 |
| llama-4-scout | 0.431 | 0.372 | 0.579 | 0.456 | 0.526 | 0.379 |
| gpt-5.4 | 0.549 | 0.386 | 0.773 | 0.567 | 0.742 | 0.597 |

## Recall

| Model | GPTKB | LMCRAWL | schemaorg_schema | schemaorg_schema_no_constraints | wikidata_schema | wikidata_schema_no_constraints |
|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.265 | 0.250 | 0.126 | 0.228 | 0.109 | 0.162 |
| llama-4-scout | 0.064 | 0.128 | 0.069 | 0.096 | 0.079 | 0.087 |
| gpt-5.4 | 0.246 | 0.207 | 0.117 | 0.222 | 0.127 | 0.196 |

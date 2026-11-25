# exp3: 10 domains x 100 entities

_Generated 2026-11-25 19:42 by ANALYSIS/make_all_tables.py. Precision / Recall = entailment ratio of the 1000-triple precision / recall sample (judge: gemma-4-26B-A4B-it, RAG); F1 = harmonic mean._

33 of 33 rows evaluated

"ALL" pools the 10 domains (sum of entailed / sum of judged triples).

## F1

| Model | animal | artifact | cultural_concept | event | location | organization | person | plant | scientific_concept | work_of_art | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.406 | 0.322 | 0.317 | 0.476 | 0.278 | 0.308 | 0.186 | 0.332 | 0.447 | 0.296 | 0.339 |
| llama-4-scout | 0.065 | 0.081 | 0.053 | 0.172 | 0.085 | 0.088 | 0.042 | 0.063 | 0.109 | 0.095 | 0.086 |
| gpt-5.4 | 0.354 | 0.323 | 0.383 | 0.542 | 0.330 | 0.414 | 0.232 | 0.284 | 0.433 | 0.423 | 0.377 |

## Precision

| Model | animal | artifact | cultural_concept | event | location | organization | person | plant | scientific_concept | work_of_art | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.611 | 0.461 | 0.564 | 0.664 | 0.505 | 0.551 | 0.353 | 0.620 | 0.598 | 0.484 | 0.541 |
| llama-4-scout | 0.710 | 0.381 | 0.469 | 0.582 | 0.449 | 0.530 | 0.429 | 0.705 | 0.527 | 0.398 | 0.505 |
| gpt-5.4 | 0.874 | 0.839 | 0.840 | 0.857 | 0.816 | 0.876 | 0.816 | 0.921 | 0.843 | 0.893 | 0.858 |

## Recall

| Model | animal | artifact | cultural_concept | event | location | organization | person | plant | scientific_concept | work_of_art | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.304 | 0.247 | 0.220 | 0.371 | 0.192 | 0.214 | 0.126 | 0.227 | 0.357 | 0.213 | 0.247 |
| llama-4-scout | 0.034 | 0.045 | 0.028 | 0.101 | 0.047 | 0.048 | 0.022 | 0.033 | 0.061 | 0.054 | 0.047 |
| gpt-5.4 | 0.222 | 0.200 | 0.248 | 0.396 | 0.207 | 0.271 | 0.135 | 0.168 | 0.291 | 0.277 | 0.241 |

# exp3: 10 domains x 100 entities

_Generated 2026-10-06 13:24 by ANALYSIS/make_all_tables.py. Precision / Recall = entailment ratio of the 1000-triple precision / recall sample (judge: gemma-4-26B-A4B-it, RAG); F1 = harmonic mean._

33 of 33 rows evaluated

"ALL" pools the 10 domains (sum of entailed / sum of judged triples).

## F1

| Model | animal | artifact | cultural_concept | event | location | organization | person | plant | scientific_concept | work_of_art | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.367 | 0.279 | 0.275 | 0.380 | 0.256 | 0.271 | 0.175 | 0.301 | 0.377 | 0.255 | 0.298 |
| llama-4-scout | 0.065 | 0.079 | 0.052 | 0.165 | 0.085 | 0.087 | 0.042 | 0.062 | 0.107 | 0.093 | 0.085 |
| gpt-5.4 | 0.340 | 0.310 | 0.353 | 0.469 | 0.318 | 0.390 | 0.226 | 0.273 | 0.403 | 0.386 | 0.355 |

## Precision

| Model | animal | artifact | cultural_concept | event | location | organization | person | plant | scientific_concept | work_of_art | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.462 | 0.320 | 0.368 | 0.390 | 0.386 | 0.368 | 0.287 | 0.448 | 0.399 | 0.317 | 0.374 |
| llama-4-scout | 0.634 | 0.327 | 0.378 | 0.447 | 0.437 | 0.452 | 0.405 | 0.567 | 0.452 | 0.333 | 0.429 |
| gpt-5.4 | 0.721 | 0.688 | 0.612 | 0.576 | 0.684 | 0.694 | 0.691 | 0.721 | 0.656 | 0.637 | 0.668 |

## Recall

| Model | animal | artifact | cultural_concept | event | location | organization | person | plant | scientific_concept | work_of_art | ALL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.304 | 0.247 | 0.220 | 0.371 | 0.192 | 0.214 | 0.126 | 0.227 | 0.357 | 0.213 | 0.247 |
| llama-4-scout | 0.034 | 0.045 | 0.028 | 0.101 | 0.047 | 0.048 | 0.022 | 0.033 | 0.061 | 0.054 | 0.047 |
| gpt-5.4 | 0.222 | 0.200 | 0.248 | 0.396 | 0.207 | 0.271 | 0.135 | 0.168 | 0.291 | 0.277 | 0.241 |

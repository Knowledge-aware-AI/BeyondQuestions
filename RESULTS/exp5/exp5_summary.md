# exp5: triple-count ranges in the prompt (popular / long-tail)

_Generated 2026-11-25 19:42 by ANALYSIS/make_all_tables.py. Precision / Recall = entailment ratio of the 1000-triple precision / recall sample (judge: gemma-4-26B-A4B-it, RAG); F1 = harmonic mean._

21 of 21 rows evaluated

Ranges ordered from the smallest to the largest requested number of facts.

## F1

| Model | pop_5_10_lt_1 | pop_10_20_lt_1_2 | pop_25_50_lt_3_5 | pop_33_66_lt_3_7 | pop_50_100_lt_5_10 | pop_75_150_lt_8_15 | pop_100_200_lt_10_20 |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.305 | 0.315 | 0.349 | 0.371 | 0.358 | 0.348 | 0.351 |
| llama-4-scout | 0.114 | 0.098 | 0.100 | 0.127 | 0.130 | 0.121 | 0.103 |
| gpt-5.4 | 0.327 | 0.327 | 0.367 | 0.402 | 0.380 | 0.387 | 0.396 |

## Precision

| Model | pop_5_10_lt_1 | pop_10_20_lt_1_2 | pop_25_50_lt_3_5 | pop_33_66_lt_3_7 | pop_50_100_lt_5_10 | pop_75_150_lt_8_15 | pop_100_200_lt_10_20 |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.686 | 0.638 | 0.546 | 0.550 | 0.544 | 0.503 | 0.494 |
| llama-4-scout | 0.601 | 0.555 | 0.531 | 0.543 | 0.525 | 0.484 | 0.522 |
| gpt-5.4 | 0.777 | 0.801 | 0.734 | 0.720 | 0.710 | 0.715 | 0.715 |

## Recall

| Model | pop_5_10_lt_1 | pop_10_20_lt_1_2 | pop_25_50_lt_3_5 | pop_33_66_lt_3_7 | pop_50_100_lt_5_10 | pop_75_150_lt_8_15 | pop_100_200_lt_10_20 |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.196 | 0.209 | 0.257 | 0.280 | 0.267 | 0.266 | 0.272 |
| llama-4-scout | 0.063 | 0.054 | 0.055 | 0.072 | 0.074 | 0.069 | 0.057 |
| gpt-5.4 | 0.207 | 0.205 | 0.245 | 0.279 | 0.259 | 0.265 | 0.274 |

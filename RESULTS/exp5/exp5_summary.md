# exp5: triple-count ranges in the prompt (popular / long-tail)

_Generated 2026-10-06 13:24 by ANALYSIS/make_all_tables.py. Precision / Recall = entailment ratio of the 1000-triple precision / recall sample (judge: gemma-4-26B-A4B-it, RAG); F1 = harmonic mean._

21 of 21 rows evaluated

Ranges ordered from the smallest to the largest requested number of facts.

## F1

| Model | pop_5_10_lt_1 | pop_10_20_lt_1_2 | pop_25_50_lt_3_5 | pop_33_66_lt_3_7 | pop_50_100_lt_5_10 | pop_75_150_lt_8_15 | pop_100_200_lt_10_20 |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.290 | 0.293 | 0.300 | 0.311 | 0.299 | 0.285 | 0.294 |
| llama-4-scout | 0.113 | 0.097 | 0.098 | 0.125 | 0.127 | 0.118 | 0.101 |
| gpt-5.4 | 0.315 | 0.312 | 0.346 | 0.367 | 0.352 | 0.351 | 0.366 |

## Precision

| Model | pop_5_10_lt_1 | pop_10_20_lt_1_2 | pop_25_50_lt_3_5 | pop_33_66_lt_3_7 | pop_50_100_lt_5_10 | pop_75_150_lt_8_15 | pop_100_200_lt_10_20 |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.556 | 0.490 | 0.359 | 0.349 | 0.341 | 0.307 | 0.321 |
| llama-4-scout | 0.531 | 0.487 | 0.447 | 0.471 | 0.439 | 0.400 | 0.442 |
| gpt-5.4 | 0.659 | 0.655 | 0.588 | 0.537 | 0.549 | 0.522 | 0.551 |

## Recall

| Model | pop_5_10_lt_1 | pop_10_20_lt_1_2 | pop_25_50_lt_3_5 | pop_33_66_lt_3_7 | pop_50_100_lt_5_10 | pop_75_150_lt_8_15 | pop_100_200_lt_10_20 |
|---|---|---|---|---|---|---|---|
| deepseek-v3.2 | 0.196 | 0.209 | 0.257 | 0.280 | 0.267 | 0.266 | 0.272 |
| llama-4-scout | 0.063 | 0.054 | 0.055 | 0.072 | 0.074 | 0.069 | 0.057 |
| gpt-5.4 | 0.207 | 0.205 | 0.245 | 0.279 | 0.259 | 0.265 | 0.274 |

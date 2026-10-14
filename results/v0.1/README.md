# LabelBench by Bayanat Labs: results

Scores are computed on fixed, checksummed manifests. SAC95 is the fraction that can be accepted while the 95% Wilson lower confidence bound remains at or above 95% accuracy. For tied-confidence structured outputs, zero SAC is a routing diagnostic rather than proof that selective automation is impossible.

| Task | Model | n | Primary score (95% bootstrap CI) | Macro-F1 | SAC95 |
|---|---|---:|---:|---:|---:|
| Binary toxicity | `char_ngram_nb` | 100 | 97.0% (93.0–100.0) | 95.0% | 89.0% |
| Binary toxicity | `ollama_hf.co_yasserrmd_Fanar-1-9B-Instruct-gguf_Q4_K_M_zero_shot` | 100 | 88.0% (82.0–94.0) | 83.1% | 0.0% |
| Binary toxicity | `ollama_qwen2.5_3b_zero_shot` | 100 | 78.0% (70.0–86.0) | 69.8% | 0.0% |
| Dialect region identification | `char_ngram_nb` | 500 | 69.8% (65.8–74.0) | 68.8% | 0.0% |
| Dialect region identification | `ollama_hf.co_yasserrmd_Fanar-1-9B-Instruct-gguf_Q4_K_M_dialect_region_zero_shot` | 500 | 47.8% (43.4–52.4) | 47.3% | 0.0% |
| Dialect region identification | `ollama_llama3_latest_dialect_region_zero_shot` | 500 | 24.6% (21.0–28.6) | 18.7% | 0.0% |
| Dialect region identification | `ollama_qwen2_latest_dialect_region_zero_shot` | 500 | 22.6% (19.0–26.2) | 16.2% | 0.0% |
| Dialect region identification | `ollama_qwen2.5_3b_dialect_region_zero_shot` | 500 | 22.4% (18.8–26.6) | 14.9% | 0.0% |
| Intent classification | `char_ngram_nb` | 2974 | 74.2% (72.7–75.8) | 65.1% | 39.5% |
| Multi-label toxicity | `char_ngram_nb_ovr_tuned` | 300 | 49.0% (43.7–54.3) | 73.2% | 0.0% |
| Multi-label toxicity | `ollama_hf.co_yasserrmd_Fanar-1-9B-Instruct-gguf_Q4_K_M_multilabel_zero_shot` | 300 | 31.3% (26.3–36.3) | 50.5% | 0.0% |
| Multi-label toxicity | `ollama_qwen2.5_3b_multilabel_zero_shot` | 300 | 25.3% (20.3–29.7) | 39.3% | 0.0% |

> Local-model results are a reproducible launch baseline, not a claim about untested frontier systems. Evaluation-only source texts are never redistributed.

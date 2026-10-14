# LabelBench v0.1 verified results

Run date: 2026-07-06

## Headline

Qwen 2.5 3B reaches 78.0% accuracy on a seeded 100-item binary AraTox
sample, then falls to 25.3% exact-set accuracy on a seeded 300-item sample from
the original seven-label taxonomy. On balanced five-region native DART text it
reaches 22.4%, only 2.4 points above chance. Arabic-specialized Fanar 1 9B
improves all three paired tasks, but still reaches 0% SAC95 under the
v0.1 tied-confidence JSON protocol.

Small supervised character diagnostics score 97.0%, 49.0%, and 69.8% on the
same three held-out manifests. The benchmark signal is learnable; zero-shot
annotation without calibrated routing and measured human review is the failure.

## Matched held-out comparisons

| Task | Model | n | Exact / accuracy | Macro-F1 | SAC95 |
|---|---|---:|---:|---:|---:|
| Binary toxicity | Qwen 2.5 3B zero-shot | 100 | 78.0% | 69.8% | 0% |
| Binary toxicity | Fanar 1 9B zero-shot | 100 | 88.0% | 83.1% | 0% |
| Binary toxicity | Character NB | 100 | 97.0% | 95.0% | 89.0% |
| Multi-label toxicity | Qwen 2.5 3B zero-shot | 300 | 25.3% | 39.3% | 0% |
| Multi-label toxicity | Fanar 1 9B zero-shot | 300 | 31.3% | 50.5% | 0% |
| Multi-label toxicity | Tuned character OVR | 300 | 49.0% | 73.2% | 0% |
| Dialect region | Qwen 2.5 3B zero-shot | 500 | 22.4% | 14.9% | 0% |
| Dialect region | Qwen 2 7B zero-shot | 500 | 22.6% | 16.2% | 0% |
| Dialect region | Llama 3 8B zero-shot | 500 | 24.6% | 18.7% | 0% |
| Dialect region | Fanar 1 9B zero-shot | 500 | 47.8% | 47.3% | 0% |
| Dialect region | Character NB | 500 | 69.8% | 68.8% | 0% |

## Dialect collapse

Qwen 2.5 3B recall by region:

| Region | Recall |
|---|---:|
| Egyptian | 85% |
| Gulf | 4% |
| Iraqi | 6% |
| Levantine | 13% |
| Maghrebi | 4% |

It predicts Egyptian for 70% to 80% of every non-Egyptian region. Qwen 2 7B
shifts the dominant class toward Levantine but reaches nearly identical
accuracy. The failure is systematic class collapse, not uniform five-way
uncertainty.

Fanar 1 9B reaches 47.8% accuracy: a 23.2-point gain over Llama 3 and a
25.4-point gain over Qwen 2.5. Its regional recall is 60% Egyptian, 75% Gulf,
13% Iraqi, 50% Levantine, and 41% Maghrebi. Arabic specialization materially
helps. The remaining 22-point gap to the supervised diagnostic and
tied-confidence 0% SAC95 leave substantial room for task-specific data and
human routing.

## Paired significance

Every comparison below uses identical item IDs and gold labels. The final
column applies Holm correction across the eight paired McNemar tests.

| A minus B | Accuracy difference | Paired bootstrap 95% CI | Raw McNemar p | Holm p |
|---|---:|---:|---:|---:|
| Binary character NB minus Qwen | +19.0 points | +11.0 to +28.0 | 2.10e-5 | 8.39e-5 |
| Fanar minus Qwen on binary | +10.0 points | +2.0 to +18.0 | 0.031 | 0.090 |
| Binary character NB minus Fanar | +9.0 points | +2.0 to +17.0 | 0.035 | 0.090 |
| Multi-label character OVR minus Qwen | +23.7 points | +17.7 to +29.7 | 1.43e-13 | 1.00e-12 |
| Fanar minus Qwen on multi-label | +6.0 points | +1.0 to +11.3 | 0.030 | 0.090 |
| Multi-label character OVR minus Fanar | +17.7 points | +11.7 to +23.7 | 8.53e-8 | 4.27e-7 |
| Fanar minus Qwen on dialect | +25.4 points | +20.2 to +30.6 | 1.91e-19 | 1.53e-18 |
| Character NB minus Fanar on dialect | +22.0 points | +16.2 to +27.6 | 2.95e-13 | 1.77e-12 |

The Fanar-over-Qwen binary and multi-label gains are promising but should be
treated as exploratory after family-wise correction. The larger diagnostic
gaps and Fanar's dialect improvement remain statistically strong.

## Taxonomy collapse

On the seven-label AraTox sample, Qwen predicts approximately the correct label
cardinality (1.40 predicted versus 1.38 gold) but selects the wrong categories.
Per-label F1 ranges from 10.3% for violence to 68.1% for `not`. This rules out
simple under-labeling as the explanation for low exact match.

Fanar improves exact match to 31.3%, micro-F1 to 55.0%, and macro-F1 to 50.5%.
Its strongest labels are sexual (71.8%) and `not` (69.9%); appearance remains
weak at 20.0%. Specialization helps without making the workflow safely
automatic.

## Controls and caveats

- MASSIVE ar-SA character NB: 74.24% accuracy, 65.15% macro-F1, 39.51% SAC95.
  Removing 8.41% normalized train/test overlap lowers accuracy to 73.27%.
- Full AraTox binary character NB: 96.27% accuracy, 95.47% macro-F1, 100%
  SAC95. Normalized overlap is 0.49% with no conflicting binary labels.
- Full AraTox multi-label tuned character OVR: 53.73% exact set, 74.70%
  micro-F1, 73.35% macro-F1, and 0% SAC95.
- Structured-output confidence is categorical, not calibrated. Zero SAC is a
  diagnostic of the current prompt/harness, not proof that generative models
  cannot support selective automation. The code now supports verbalized
  confidence and calibration-split SAC for the next experiment matrix.
- v0.1 contains no frontier API results and no human adjudication results.
  The API adapter and review bundles are ready for the next release.

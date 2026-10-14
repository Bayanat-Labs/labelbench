# LabelBench specification

Status: **v0.1 implemented release specification**
Public title: **Arabic Annotation Is Not Binary**
Paper title: **Arabic Annotation Is Not Binary: LabelBench, a Reproducible
Audit of Automated Data Labeling**

## 1. Research question

For a specified Arabic labeling task, what fraction of examples can an AI
system label automatically while satisfying a predeclared quality threshold,
and how does that fraction change by task, variety, domain, and prompting
protocol?

## 2. Primary estimand

For model or workflow \(m\), task slice \(s\), and required quality \(q\), the
primary estimand is:

```text
safe_coverage(m, s, q) =
  max fraction of examples accepted by confidence
  such that the 95% Wilson lower bound on accepted-label accuracy is >= q
```

The planned quality targets are 0.95 and 0.98. Empirical selective accuracy and
coverage are reported alongside the conservative Wilson-bound result.

This metric is called **Safe Automation Coverage (SAC)** in the research code.
SAC is an operational Wilson-bound diagnostic, not a replacement for selective
prediction, the Alternative Annotator Test, or conformal risk-control methods.
The preferred protocol is to choose the confidence threshold on a calibration
partition, then report realized coverage and accuracy on a disjoint evaluation
partition. If the threshold is selected on the same evaluation items, the result
must be labeled exploratory and optimistic.

## 3. Secondary metrics

- Accuracy and macro-F1 for single-label classification
- Entity/span F1 for extraction tasks
- Exact-set and label-wise F1 for multi-label tasks
- Expected calibration error and Brier score where probabilities are available
- Worst-variety performance and max/min variety gap
- Cost per 10,000 attempted and correctly accepted labels
- Median and tail latency
- Abstention rate

All central estimates will include uncertainty intervals. Pairwise model
comparisons will use paired tests and multiplicity correction.
Selective-automation thresholds will be selected on held-out calibration splits
where sample size permits. Top-tier annotator-replacement claims require an
alt-test or conformal risk-control analysis in addition to aggregate model
scores.

## 4. Task families

Implemented in v0.1:

1. Intent classification
2. Binary toxicity
3. Multi-label toxicity
4. Broad-region dialect identification

Planned after v0.1:

5. Sentiment and sarcasm
6. Named entities and PII
7. Code-switching
8. Stance and culturally dependent interpretation

Tasks enter the benchmark only after a provenance and license review. Source
text without a clear redistribution license is never redistributed. For those
sources, public releases report only aggregate scores and SHA-256 hashes (see
DATA.md).

## 5. Arabic coverage

The target matrix includes MSA plus Saudi, Gulf, Egyptian, Levantine, Iraqi,
Moroccan, Algerian, Tunisian, Yemeni, and Sudanese varieties. Dataset-level
labels are not assumed to prove city-level or sub-dialect coverage.

Translated Arabic, human-localized Arabic, native-authored Arabic, synthetic
Arabic, and naturally occurring Arabic must be analyzed as separate provenance
groups. They must never be silently pooled into a single "Arabic" result.

## 6. Evaluation protocols

- `zero_shot`: task rubric and closed label set
- `few_shot`: fixed development examples, never test examples
- `self_consistency`: repeated inference with a declared aggregation rule
- `ensemble_route`: accept only when independent systems agree
- `human_route`: AI pre-labeling with uncertainty-based escalation

Prompt text, model version, decoding parameters, date, and raw response hashes
must be preserved for every run.

Generative models must expose a routing signal before SAC is interpreted:
token log probabilities where available, verbalized confidence, calibrated
self-consistency agreement, or a held-out risk-control score. A fixed parser
confidence of 1.0 records JSON validity only and cannot justify selective
automation.

## 7. Dataset quality policy

Every dataset registry entry records:

- canonical source and immutable revision where possible;
- license and redistribution status;
- collection method;
- annotation method and annotator population;
- Arabic provenance (native, natural, translated, post-edited, synthetic);
- dialect/variety claim and the evidence supporting it;
- known limitations;
- inclusion status and intended role.

Public benchmark claims will be based on gold sets with documented human
annotation. Model/dataset disagreements may be independently adjudicated by
qualified native raters. Existing labels are not automatically presumed
error-free.

## 8. Leakage and reproducibility

- Public training and test examples remain separated.
- Few-shot examples come only from a development or training split.
- Dataset fingerprints and normalized-file hashes are written to each run.
- Item-level predictions are cached and never overwritten silently.
- Closed-model results identify the exact API model string and run date.
- Random operations use recorded seeds.

## 9. Release scope

The v0.1 release evaluates MASSIVE `ar-SA`, AraTox binary and multi-label
toxicity, and a common-source balanced five-region DART slice from IADD.
MASSIVE remains an engineering control because it is localized from English.
AraTox and DART provide naturally occurring, human-annotated Arabic.

The release includes majority and character n-gram diagnostics, a tuned
one-vs-rest multi-label diagnostic, local structured-output language models,
and a credential-ready dated OpenAI Responses API adapter.

## 10. Release artifacts

- Technical report: LaTeX source in `paper/`, PDF on the project page
- Evaluation harness: `labelbench/`
- Aggregate results and machine-readable tables: `results/v0.1/`
- Offline blinded review app: `labelbench/templates/review_template.html`

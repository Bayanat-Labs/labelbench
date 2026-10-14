# LabelBench by Bayanat Labs

**Arabic annotation is not binary.**

LabelBench is a reproducible audit of automated Arabic data labeling from [Bayanat Labs](https://www.bayanatlabs.com). It measures how much of a labeling queue can be accepted at a declared quality floor, then breaks the answer down by taxonomy, task and Arabic variety.

[Project page](https://www.bayanatlabs.com/research/labelbench) · [Technical report (PDF)](https://www.bayanatlabs.com/research/labelbench-paper-v0.1.pdf) · Code Apache-2.0 · Results CC BY 4.0 · Version 0.1.0

## The v0.1 finding

A local language model (Qwen 2.5 3B) scored 78% on binary Arabic toxicity. On the real seven-label taxonomy, exact-match accuracy fell to 25.3% on the fixed held-out manifest. Across five dialect regions it reached 22.4%, where chance is 20%. We call this the annotation complexity cliff.

| Held-out task | Qwen 2.5 3B zero-shot | Fanar 1 9B zero-shot | Small supervised diagnostic |
|---|---:|---:|---:|
| Binary toxicity | 78.0% | 88.0% | 97.0% |
| Seven-label toxicity (exact set) | 25.3% | 31.3% | 49.0% |
| Five-region dialect ID | 22.4% | 47.8% | 69.8% |

The toxicity rows use seeded AraTox samples of 100 and 300 items. The dialect row uses a balanced 500-item manifest of DART text from IADD, with 100 items for each region.

The supervised systems are intentionally small character n-gram models. Their purpose is to establish that the held-out signal is learnable, not to claim state of the art. Arabic specialization helps, but the v0.1 generative runs still have 0% Safe Automation Coverage at a 95% quality target under their tied-confidence JSON protocol. That is a routing-diagnostic limitation, not proof that generative models cannot support selective automation when confidence is properly elicited and calibrated.

Every model and baseline, the paired significance tests and per-region recall are in [RESULTS.md](RESULTS.md).

## What is included

- Checksummed ingestion for MASSIVE 1.1, AraTox, and IADD/DART
- Leakage and conflicting-label audits
- Binary, multi-label, intent, and dialect-region tasks
- Character n-gram and one-vs-rest diagnostics
- Local structured-output inference through Ollama
- A dated OpenAI Responses API adapter with strict JSON Schema and `store=false`
- Accuracy, exact-set, F1, calibration, bootstrap intervals, confusion matrices
- Conservative Safe Automation Coverage using a Wilson lower bound
- Resumable item-level logs with raw responses, model digests, usage, and hashes
- A blinded offline RTL review app and three-rater aggregation
- Paper-ready JSON, CSV, Markdown, and SVG release generation
- The LaTeX source of the technical report

```text
BENCHMARK.md              evaluation specification
RESULTS.md                verified v0.1 findings
DATA.md                   third-party sources, pinned revisions and licenses
labelbench/               dependency-free benchmark package and CLI
labelbench/templates/     blinded review template
registry/                 data and model provenance gates
adjudication/             offline human-review protocol
paper/                    LaTeX source and figure script for the technical report
tests/                    deterministic unit tests
results/                  aggregate-only public releases
```

## Install

LabelBench needs Python 3.10 or newer and has no third-party runtime dependencies.

```bash
pip install bayanat-labelbench
labelbench --help
```

Or install from source:

```bash
git clone https://github.com/Bayanat-Labs/labelbench.git
cd labelbench
pip install -e .
labelbench --help
```

From a checkout you can also run `python3 -m labelbench` without installing. Tested on Python 3.10 to 3.13.

## Reproduce the core results

Each command downloads its source from the original location at a pinned revision and checks its SHA-256 before use (see [DATA.md](DATA.md)). Downloads and normalized manifests go to `$LABELBENCH_CACHE_DIR` when it is set, and otherwise to `labelbench/` in your user cache directory (`~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows). Every data command also accepts `--cache-dir`.

The v0.1 runs kept downloads inside the checkout. To do the same, run these commands from the repository root:

```bash
export LABELBENCH_CACHE_DIR=data

python3 -m unittest discover -s tests -v

labelbench pilot-aratox \
  --test-limit 100 --random-test \
  --output runs/aratox-binary-sample100

labelbench pilot-aratox-multilabel \
  --test-limit 300 --random-test --tune-thresholds \
  --output runs/aratox-multilabel-sample300

labelbench pilot-ollama-aratox-multilabel \
  --model qwen2.5:3b --test-limit 300 \
  --output runs/ollama-aratox-multilabel-pilot-300

labelbench pilot-ollama-aratox \
  --model hf.co/yasserrmd/Fanar-1-9B-Instruct-gguf:Q4_K_M \
  --test-limit 100 \
  --output runs/ollama-aratox-random-pilot-100

labelbench pilot-ollama-aratox \
  --model qwen2.5:3b --test-limit 100 \
  --output runs/ollama-aratox-random-pilot-100

labelbench pilot-ollama-aratox-multilabel \
  --model hf.co/yasserrmd/Fanar-1-9B-Instruct-gguf:Q4_K_M --test-limit 300 \
  --output runs/ollama-aratox-multilabel-pilot-300

labelbench pilot-ollama-iadd \
  --model hf.co/yasserrmd/Fanar-1-9B-Instruct-gguf:Q4_K_M --per-region 100 \
  --output runs/ollama-iadd-region-pilot-500

labelbench pilot-iadd-supervised \
  --train-per-region 100 --test-per-region 100 \
  --output runs/iadd-supervised-diagnostic

labelbench pilot-ollama-iadd \
  --model qwen2.5:3b --per-region 100 \
  --output runs/ollama-iadd-region-pilot-500
```

`data/` and `runs/` are created locally by these commands. Both are listed in `.gitignore` and are not part of this repository.

The Ollama commands need a local [Ollama](https://ollama.com) server with the model already pulled, for example `ollama pull qwen2.5:3b`.

All inference commands resume safely. A cached prediction is accepted only when its stable ID and gold label match the current manifest.

The exact v0.1 generative results used categorical JSON confidence. For the next confidence-enabled experiment matrix, add `--confidence-mode verbalized` to the Ollama commands. Those runs are intentionally new model IDs rather than exact reproductions of the v0.1 reports.

### Build a public release

Pass the exact reports intended for publication:

```bash
labelbench build-release \
  --release-id labelbench-v0.1 \
  --output runs/release-v0.1 \
  --reports \
    runs/aratox-binary-sample100/char_ngram_nb/report.json \
    runs/ollama-aratox-random-pilot-100/ollama_qwen2.5_3b_zero_shot/report.json \
    runs/ollama-aratox-random-pilot-100/ollama_hf.co_yasserrmd_Fanar-1-9B-Instruct-gguf_Q4_K_M_zero_shot/report.json \
    runs/aratox-multilabel-sample300/char_ngram_nb_ovr_tuned/report.json \
    runs/ollama-aratox-multilabel-pilot-300/ollama_qwen2.5_3b_multilabel_zero_shot/report.json \
    runs/ollama-aratox-multilabel-pilot-300/ollama_hf.co_yasserrmd_Fanar-1-9B-Instruct-gguf_Q4_K_M_multilabel_zero_shot/report.json \
    runs/iadd-supervised-diagnostic/char_ngram_nb/report.json \
    runs/ollama-iadd-region-pilot-500/ollama_qwen2.5_3b_dialect_region_zero_shot/report.json \
    runs/ollama-iadd-region-pilot-500/ollama_hf.co_yasserrmd_Fanar-1-9B-Instruct-gguf_Q4_K_M_dialect_region_zero_shot/report.json
```

The published `results/v0.1` also includes the Qwen 2 7B, Llama 3 8B and MASSIVE intent reports, plus eight `compare-runs` outputs passed with `--comparisons`. Write test builds outside `results/` so the published files stay unchanged.

The output includes `summary.json`, `leaderboard.csv`, a human-readable table, a task chart, and a dialect confusion heatmap. No source text is copied into the release.

### Human adjudication

Create a blinded review bundle (the examples path assumes `LABELBENCH_CACHE_DIR=data`, as above):

```bash
labelbench make-review-bundle \
  --examples data/normalized/aratox_multilabel_test_sample300_seed42.jsonl \
  --manifest adjudication/bundles/aratox-review-manifest.json \
  --bundle adjudication/bundles/aratox-review.html
```

The review template ships with the package. Pass `--template` to use your own. Reviewers work offline. Model predictions are never shown. The application enforces the exclusive `not` label, stores progress locally, and exports a pseudonymous JSON submission. See [adjudication/README.md](adjudication/README.md).

### Frontier API adapter

The OpenAI adapter follows the current Responses API structured-output shape:

- <https://developers.openai.com/api/docs/guides/structured-outputs>
- <https://developers.openai.com/api/reference/resources/responses/methods/create>

Copy `.env.example`, export `OPENAI_API_KEY`, and use a dated model snapshot:

```bash
labelbench pilot-openai-iadd \
  --model gpt-5.4-mini-2026-03-17 \
  --per-region 100 \
  --confidence-mode verbalized \
  --output runs/openai-iadd-region-pilot-500
```

Credentials are never read from files or written to run artifacts.

Do not paste API keys into chats, issues, or notebooks. Export them in the shell environment for the single run and rotate any key that was ever shared in plain text.

## Data sources and licenses

- Code: Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
- Results, registry, specification and paper source written by Bayanat Labs: CC BY 4.0. See [LICENSE-data](LICENSE-data).
- Third-party data is not stored in this repository. The commands download each source from its original location and check its SHA-256. Each source keeps its own license. See [DATA.md](DATA.md).
- The Bayanat Labs name and logo are not licensed.

MASSIVE and AraTox are pinned CC BY 4.0 sources. IADD/DART text is treated as evaluation-only because the repository does not include an explicit data license, even though the accompanying Data in Brief article is CC BY. Public releases contain aggregate results and cryptographic identifiers only.

Raw data, normalized text, review submissions, and model outputs are ignored by Git. Do not publish those directories without a separate source and privacy review.

## Results and paper

- [results/v0.1/](results/v0.1/) holds `summary.json`, `leaderboard.csv`, eight paired comparison files and two charts. `summary.json` and `leaderboard.csv` match the downloads on the project page byte for byte.
- [RESULTS.md](RESULTS.md) has the verified v0.1 findings, and [BENCHMARK.md](BENCHMARK.md) the evaluation specification.
- The technical report is on the [project page](https://www.bayanatlabs.com/research/labelbench) as a [PDF](https://www.bayanatlabs.com/research/labelbench-paper-v0.1.pdf). Its LaTeX source is in [paper/](paper/). `paper/make_figures.py` needs `reportlab`, which LabelBench itself does not use.

## Limitations

- **Small local models.** The v0.1 generative results come from four-bit quantized local models of 3B to 9B parameters, run through Ollama. They do not describe frontier or API models. v0.1 contains no frontier API results and no human adjudication results. The API adapter and review bundles are ready for the next release.
- **Tied-confidence protocol.** The v0.1 generative runs emit categorical JSON confidence, not calibrated confidence. Their 0% Safe Automation Coverage is a limitation of this harness setup, not evidence that selective automation is impossible.
- **v0.1 scope.** v0.1 covers four task families on fixed samples of 100 to 2,974 items. The dialect task uses five coarse regions from one source (DART) and cannot establish country-level performance. MASSIVE is localized from English and serves only as an engineering control. AraTox rows carry no per-item dialect labels.

## Cite

If you use LabelBench, please cite the technical report. GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff). A Zenodo DOI will be added with the first GitHub release.

```bibtex
@techreport{bayanat2026labelbench,
  title       = {Arabic Annotation Is Not Binary: LabelBench, a Reproducible Audit of Automated Data Labeling},
  author      = {{Bayanat Labs Research}},
  institution = {Bayanat Labs},
  year        = {2026},
  month       = jul,
  number      = {v0.1},
  url         = {https://www.bayanatlabs.com/research/labelbench}
}
```

## Contributing

Issues and pull requests are welcome. Please read the [contributing guide](https://github.com/Bayanat-Labs/.github/blob/main/CONTRIBUTING.md) and open an issue before larger changes. For questions about the data, email [hello@bayanatlabs.com](mailto:hello@bayanatlabs.com).

---

### About Bayanat Labs

[Bayanat Labs](https://www.bayanatlabs.com) is a Riyadh-based Arabic data company for AI. We work only on data, from sourcing and annotation to alignment and evaluation, across 25+ Arabic varieties, with vetted native speakers, linguists and licensed domain professionals.

[Run a pilot](https://www.bayanatlabs.com/pilot) · [Datasets](https://www.bayanatlabs.com/products) · [LinkedIn](https://www.linkedin.com/company/bayanat-labs/) · [hello@bayanatlabs.com](mailto:hello@bayanatlabs.com)

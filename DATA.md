# Data sources

LabelBench does not store any third-party dataset in this repository or in its release files. When you run a command, the code downloads each source from its original location at a pinned revision, checks its SHA-256, and writes normalized copies to your local cache directory (see the README). Each source keeps its own license, and you are responsible for following it.

Public LabelBench releases (`results/`) contain aggregate scores, SHA-256 hashes of the normalized manifests and prediction files, and model metadata. They contain no source text.

| Source | Version or revision | License | URL | What we fetch | Redistributed here |
| --- | --- | --- | --- | --- | --- |
| MASSIVE 1.1, `ar-SA` locale (Amazon) | Release archive 1.1, SHA-256 `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` | CC BY 4.0 | https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz | The archive; only `1.1/data/ar-SA.jsonl` is extracted | No |
| AraTox | Hugging Face `Alshargi/Aratox`, revision `438f12260480318b976b15b23212ed3a661c088f` | CC BY 4.0 (dataset card) | https://huggingface.co/datasets/Alshargi/Aratox | `aratox_train.csv` (SHA-256 `0a10c60fa15b15dd0a73855d8a46d0beed4db3a3fe1daebf5594c052f35d4e3e`) and `aratox_test.csv` (SHA-256 `27dc4b80ffe79f46a4d2abe3aad362c3273827c68dfcc581ab4341ecd7636a7c`) | No |
| IADD, DART subset | GitHub `JihadZa/IADD`, commit `340b5a6135f70878b95481edcaae4e3afb2067ac` | None stated in the repository. The accompanying Data in Brief article is CC BY 4.0. Evaluation-only | https://github.com/JihadZa/IADD | `IADD.json` (SHA-256 `a28c106d1825debabb6b16979c6c6ca66abdfd0d7802fd899b5ec4d7aee0f734`); only rows with `DataSource` DART are used | No |

Licenses were checked at each source on 2026-10-13:

- MASSIVE: the Hugging Face card for `AmazonScience/massive` lists `cc-by-4.0`, and `NOTICE.md` in https://github.com/alexa/massive states that the dataset is licensed under CC BY 4.0. The pinned record in `registry/datasets.json` was verified against the `LICENSE` file at Hugging Face revision `e6c0d86b14b23cb7843fa5b6d8be335b1b4506ab`.
- AraTox: the dataset card lists `cc-by-4.0` at the pinned revision `438f122` and at the current revision.
- IADD: the GitHub repository has no license file at the pinned commit `340b5a6` (its latest commit, from 2021).

## MASSIVE 1.1

Amazon's multilingual voice-assistant dataset, localized from the English SLURP dataset. LabelBench uses the Saudi Arabia (`ar-SA`) locale for 60-way intent classification as an engineering control. Because it is localized from English, it cannot support a claim about native Saudi Arabic.

Paper: FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages", https://arxiv.org/abs/2204.08582

## AraTox

Arabic social-media text with human multi-label toxicity annotation (six toxic categories plus `not`). LabelBench uses the binary view and the full seven-label view.

Paper: Alshargi et al., "Aratox: a multi-dialect, multi-label Arabic dataset and model benchmark for toxicity detection", Language Resources and Evaluation, 2026, https://doi.org/10.1007/s10579-026-09917-9

## IADD (DART subset): evaluation-only

IADD integrates five Arabic dialect corpora. LabelBench uses only the DART rows, so all five region labels (Gulf, Egyptian, Levantine, Maghrebi and Iraqi) come from one source.

The IADD repository does not state a license for the data. LabelBench therefore treats it as evaluation-only:

- The code downloads `IADD.json` from the original repository at the pinned commit when you run an IADD command (`pilot-ollama-iadd`, `pilot-iadd-supervised`, `pilot-openai-iadd`).
- No IADD rows are stored in this repository or in any release file.
- Releases report aggregate scores and SHA-256 hashes of normalized text only.
- Do not publish the normalized IADD manifests that the code writes to your cache directory.

Paper: Zahir, J., "IADD: An integrated Arabic dialect identification dataset", Data in Brief 40, 2022, https://doi.org/10.1016/j.dib.2021.107777

DART: Alsarsour, I., Mohamed, E., Suwaileh, R. and Elsayed, T., "DART: A Large Dataset of Dialectal Arabic Tweets", LREC 2018.

If you are an author or rights holder of one of these sources and want it handled differently, email [hello@bayanatlabs.com](mailto:hello@bayanatlabs.com).

## Not downloaded

`registry/datasets.json` also lists ArBanking77 with the status `blocked_license_review`. No LabelBench code downloads it.

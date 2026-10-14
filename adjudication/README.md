# Blinded human adjudication

The review bundle is a self-contained HTML file. It makes no network requests,
stores progress in the browser's local storage, and exports a JSON submission.
Gold labels and model predictions are deliberately excluded from the bundle.

The template is `labelbench/templates/review_template.html` and ships with the
package. `labelbench make-review-bundle` uses it unless you pass `--template`.

## Workflow

1. Generate a task manifest from the frozen evaluation sample.
2. Build one offline HTML bundle from that manifest.
3. Assign opaque reviewer IDs; never put reviewer names in benchmark files.
4. Each reviewer opens the HTML file, completes labels and confidence, and
   exports JSON.
5. Validate and aggregate submissions with the LabelBench CLI.
6. Route items without exact-set majority consensus to a senior adjudicator.

Three independent reviewers are the minimum for majority consensus. Reviewers
must be allowed to flag ambiguity, distressing content, possible bad gold
labels, and PII. They may stop at any time.

Bundles and submissions are ignored by Git because they can contain source
texts and internal reviewer metadata.

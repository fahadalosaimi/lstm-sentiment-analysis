# Final trained-model inference gate — CPU only

**Current status: PASSED on the owner's Colab CPU run.** The returned evidence
records 8/8 tests, zero failures/errors/skips, inference_executed=true and
retraining=false. All four reload deltas are 0.0. The ZIP/member, checker, source,
weight and metadata hashes were verified locally without executing the model.
See the [report](results/improved/run_20261105T101047_013570Z/final_inference/final_inference_report.json)
and [log](results/improved/run_20261105T101047_013570Z/final_inference/final_inference.log).
The earlier local not_run record remains historical, because local PyTorch is absent.

## Run the independent Colab check

1. Open `notebooks/final_inference_cpu.ipynb` in Colab and select a standard **CPU**
   runtime. Do not use the GPU training notebook for this gate.
2. Run all cells. At its one upload prompt, select both local files:
   - `Upload/lstm_sentiment_colab.zip` (or its identical preserved dist/ copy) — the source used by the run.
   - `Upload/run_20261105T101047_013570Z_results.zip` — the actual returned results.
3. The notebook checks the exact archive hashes and every manifest member before
   extracting only four inference source files and the final model/metadata pair.
   No CSV, training entry point or partial checkpoint is extracted.
4. It runs eight explicit CPU inference tests and downloads
   `final_inference_cpu_evidence.zip`, including failures/not-run evidence if available.
5. Preserve the resulting evidence. The existing completed gate is already audited;
   a future repeat must be assessed on its own outputs and must not change test scores.

The notebook embeds an exact copy of `tests/final_artifact_cpu_check.py`, so it
does not need a repository, GitHub connection, script download, Drive access,
GPU, raw dataset or package installation. Standard Colab's preinstalled PyTorch
and NumPy are used. If PyTorch is absent, it stops with not_run; choose a standard
CPU runtime with compatible dependencies. No authentication is automated.

## Checks and required outcome

- Real final weights and metadata SHA-256; original inference source fingerprints.
- Strict checkpoint loading against the saved architecture; all seven tensor
  shapes, 2,117,889 parameters, finite float32 CPU values and zero PAD row.
- The 20,000-entry vocabulary, PAD/UNK, saved config, preprocessing, labels and threshold.
- Two newly authored positive and two negative English examples, with actual
  probabilities and expected/observed labels recorded. A wrong label fails the
  diagnostic check; do not change thresholds/examples or retrain to hide it.
- Blank, whitespace, markup-only, Arabic, Chinese and non-string input rejection.
- Unknown English words, padding invariance and a single-item batch.
- Repeated predictions and a freshly reloaded model, tolerance 1e-7 on CPU.
- Unchanged model weights, input artifact files and absent gradients after inference.

Only `status=passed`, `tests_run=8`, zero errors/failures/skips,
`inference_executed=true` and `retraining=false` closes this gate. Hash the returned
report/logs against EVIDENCE_MANIFEST.json and review the recorded examples and
environment. A self-produced manifest is integrity evidence, not a trusted signature.
These four examples are diagnostics, **not** another accuracy/Macro-F1 estimate.

If the semantic examples fail but loading/reload checks pass, report the observed
model weakness separately; do not claim the full gate passed. The audited
86.09% / 0.8607 experiment metrics remain unchanged.

## Inputs are pinned, not bundled for publication

    source ZIP: e49f872b0118bbd481c9838e17d60a9d3a894e62d39b043bbbf56e474071996d
    results ZIP: bc16b5a4c13e21495b50d7433f906dc7554bcad149037c62ca08e4bfdd2e435e
    model: 740572862066b6e39dcefd2ddafa4c77f2fbd8560590cd24a559851d65953718
    metadata: f774d20762c5d1b8c9e18f2ceeedd7139a43a5de932984d3dd3c4487f0e27f7d

A regenerated source ZIP will intentionally fail the archive pin, even if its
runtime files appear similar. Use the preserved local archive. Input ZIPs and
weights are not added to PUBLICATION_FILES.txt. A public clone will not contain
these private inference assets until redistribution rights are resolved.

For an already provisioned local CPU environment, the same explicit check is:

    python -B tests/final_artifact_cpu_check.py --artifact artifacts/improved/run_20261105T101047_013570Z --report outputs/final_inference/report.json

This command is documented, not claimed successful on this laptop. It never
imports a training entry point, calls fit/backward/optimizer, or reads a dataset.

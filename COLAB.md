# Complete Google Colab GPU workflow

The owner has now executed this workflow in Colab and returned a complete results
archive. The audited improved run achieved **86.09% accuracy / 0.8607 Macro-F1**;
see the [verified report](results/improved/run_20261105T101047_013570Z/reports/results.md).
The local audit did not access Google or repeat training. Original **86.29%** is
a separately documented historical result under a different protocol.

For **final trained-model inference only**, use
[the separate CPU notebook](notebooks/final_inference_cpu.ipynb) and
[FINAL_INFERENCE.md](FINAL_INFERENCE.md). Do not rerun this full-training workflow
to check saved weights. The CPU notebook has no training step and needs no dataset.

That final CPU gate is now completed in the owner's returned evidence: 8/8 tests
passed, zero failures/errors/skips, with no retraining. See
[the evidence guide](results/README.md). Repeating this GPU workflow would create
a new experiment; it is not necessary to inspect the verified results.

## Start with minimal manual work

1. From a source checkout, run `python -B -m tools.package_colab` to create
   dist/lstm_sentiment_source_current.zip. Open notebooks/colab_gpu.ipynb separately.
2. Open the notebook via **File → Upload notebook** in Google Colab.
3. Select a fresh **free GPU** runtime via **Runtime → Change runtime type**.
   Do not purchase credits or storage. If no free GPU is available, retry later.
4. Choose **Run all**. Upload lstm_sentiment_source_current.zip at the first prompt and
   IMDB Dataset.csv at the later prompt. Normally these are the only file selections.
   Keep the session connected and allow the final browser download.

**Changed default:** RUN_FULL_TRAINING=True. Running all cells in Colab now requests
full improved-model training **after all gates succeed**. Set False in Settings
for verification only. Preparing the notebook locally has not executed it.

## Stage order and gates

1. **Project upload:** extract into a new /content/lstm_colab_.../lstm_sentiment_analysis
   directory. Reject unsafe paths, symlinks, duplicate entries and manifest
   mismatches. The manifest verifies integrity, not publisher authenticity.
2. **Environment:** install requirements-colab.txt remotely only. Keep Colab's
   existing CUDA PyTorch build; do not install a CPU wheel. Require Python >=3.11,
   torch >=2.6,<3, NumPy >=1.26,<3, matplotlib >=3.9,<4 and CUDA. These are target
   ranges, not a validated lockfile. Start fresh; restart first if packages were
   imported before installation.
3. **All unit tests:** run the entire suite in a fresh subprocess. Require **zero
   failures, errors and skips**, plus all six original TorchTests. Tiny synthetic
   CPU optimization/checkpoint tests use no real dataset. Ordinary unittest treats
   skips as success; the strict Colab runner rejects them.
4. **GPU smoke:** one synthetic optimizer step through production fit, checking
   forward shape, padding invariance, finite values, an actual update, fixed PAD
   embedding and GPU/CPU save/load. This is not IMDb evaluation.
5. **Dataset:** upload a separate CSV copy into Data, or set EXISTING_DATA_PATH.
   Existing files are read, never overwritten. Enforce review/sentiment columns,
   supported text, no missing values, valid labels, 50,000 records and the inspected
   SHA-256. Review text is never printed.
6. **Training:** recheck source/input hashes, validate the saved config, train on
   training only. Select the best checkpoint and early-stop using **validation
   loss only**. Defaults: seed 42, batch 64, maximum 10 epochs, patience 3.
   Vocabulary is training-only; normalized deduplication precedes the split.
7. **Final evaluation:** restore validation-selected weights, then evaluate test
   once with fixed probability threshold 0.5. Test never determines vocabulary,
   selection, early stopping or threshold. Do not tune future runs from test scores.
8. **Reporting/export:** save portable model/vocabulary/config, metrics/history,
   figures/reports, build and download a verified output ZIP. A caught failure or
   interrupt exports explicitly failed/interrupted diagnostics when possible.

Gates are held in memory for the current run; old JSON cannot bypass verification.
A settings change is a new experiment. This is procedural guarding, not a security
boundary against someone deliberately modifying the program.

## Clear output paths

Default: /content/lstm_runs/run_<UTC timestamp>/

| Path inside run | Purpose |
|---|---|
| status.json | Workflow stage/status and gate evidence |
| config.json, source_manifest.json | Exact settings and source hashes |
| environment.json | Python/library/CUDA/GPU details |
| tests.json, tests.log | Actual counts and full unit output |
| smoke.json, smoke.log | GPU synthetic checks |
| dataset.json | Input schema/counts/fingerprint, no raw reviews |
| setup.log, training.log, workflow_error.log | Setup/training/error diagnostics |
| model/model.pt | Final CPU-portable selected weights |
| model/metadata.json | Vocabulary, preprocessing, config, labels, weight checksum |
| model/metrics.json, history.json, status.json | Measured evaluation, history, model status |
| model/partial/ | Best completed-epoch diagnostic weights and history |
| figures/loss.png, accuracy.png, confusion_matrix.png | New-run plots |
| reports/results.md, metrics.csv, provenance.json | New-run report and machine-readable results |

Results ZIP: /content/lstm_runs/run_<UTC timestamp>_results.zip.
It contains known output files plus RESULTS_MANIFEST.json with hashes/status.
The results ZIP intentionally includes model weights; it excludes the CSV,
credentials and unrelated files. The project/source ZIP contains neither data nor
weights and ships **one notebook**.

Workflow status **complete** plus model status **trained** denotes completed
evaluation. running, failed, interrupted and verified_not_trained do not.
If reporting fails after training, recover measured model outputs and repair
reporting; do not invent graphs or tune the model from its test results.

## Interruptions, GPU failure and persistence

- Progress, complete-epoch history and best diagnostic weights are saved after
  every epoch. A mid-epoch stop preserves only earlier completed-epoch checkpoints.
- Exact optimizer/shuffle/RNG resume is **not implemented**. Restart a new seeded
  experiment after a full reset. Partial weights are not final artifact bundles.
- A browser disconnect may leave remote training running. Reconnect and inspect
  status/logs before launching another run. Do not start duplicate jobs.
- A killed/deleted runtime cannot execute cleanup or initiate downloads. Temporary
  /content files may disappear. Optionally set PERSIST_TO_DRIVE=True before
  starting. User-interactive Drive mounting preserves already-written files at
  /content/drive/MyDrive/LSTM_Sentiment_Analysis/runs. No paid plan is purchased.
  Persistence does not guarantee survival of every storage/cloud failure.
- If automatic download fails while files remain, use the last recovery cell or
  Colab Files panel. Recovery only rebuilds/downloads outputs; it never resumes
  training or reruns test evaluation.
- CUDA/determinism errors stop the workflow. Do not silently disable checks.
  For GPU OOM, keep diagnostics and start a new experiment through all gates with
  a smaller batch size. There is no automatic CPU fallback or paid GPU request.

## Reproducibility and interpretation

Python, NumPy, PyTorch and loader seeds are fixed. Deterministic algorithms and
CUBLAS workspace are configured; cross-device/library bit identity is not promised.
Every run stores config, environment, source/data/split fingerprints and selection
criterion. Dataset provenance still needs confirmation per Data/README.md; a hash
identifies bytes, not redistribution rights.

Reports are populated only from completed improved-run metrics. Original 86.29%
is never copied into the improved score. Fixture losses are technical checks,
not model-quality evidence. Limitations: English regex tokens, first-200-token
truncation, one seed, custom split, no movie-group/near-duplicate control,
calibration or out-of-domain validation.

References: [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html),
[Colab FAQ](https://research.google.com/colaboratory/faq.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

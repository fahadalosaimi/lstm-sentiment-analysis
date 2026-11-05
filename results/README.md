# Selected portfolio evidence

The public candidate is a subset of the complete local experiment archive.
No raw data, learned vocabulary, model weights or ZIPs are included. Unselected
local evidence and preparation reports are retained on disk, not deleted.

## Measured experiment

- [Exact metrics](improved/run_20261105T101047_013570Z/model/metrics.json):
  Accuracy 86.09%, Macro-F1 0.8607, test size 7,436, best epoch 5.
- [History](improved/run_20261105T101047_013570Z/model/history.json) and
  [report](improved/run_20261105T101047_013570Z/reports/results.md).
- [Training environment](improved/run_20261105T101047_013570Z/environment.json)
  and [source snapshot](improved/run_20261105T101047_013570Z/source_manifest.json).
- [Original Colab test summary](improved/run_20261105T101047_013570Z/tests.json)
  and [GPU smoke summary](improved/run_20261105T101047_013570Z/smoke.json).
- Three figures are retained under the run's figures/ directory.

Final test scores were checked against the saved matrix, JSON and Markdown.
The full original result ZIP has SHA-256
bc16b5a4c13e21495b50d7433f906dc7554bcad149037c62ca08e4bfdd2e435e.
Its full manifest is not the manifest of this shortened public subset.

source_manifest.json is the frozen training-source snapshot, not a manifest of
the current portfolio tree. Documentation and source-packaging configuration
were subsequently refined; model, preprocessing and inference implementation
remain unchanged. Later edits do not rewrite the original experiment evidence.

## Final trained-model CPU gate

[Report](improved/run_20261105T101047_013570Z/final_inference/final_inference_report.json),
[log](improved/run_20261105T101047_013570Z/final_inference/final_inference.log),
[preparation record](improved/run_20261105T101047_013570Z/final_inference/preparation.json)
and [manifest](improved/run_20261105T101047_013570Z/final_inference/EVIDENCE_MANIFEST.json)
are copied byte-for-byte from the returned owner-provided evidence ZIP.

The four-file ZIP passed CRC and exact membership checks. All three manifest-listed
file sizes and SHA-256 hashes match. The checker digest, four inference source
digests and actual final model/metadata digests match local files. The report and
log agree: 8 passed, no failures/errors/skips, inference executed, no retraining.
All four expected labels match; all seven invalid cases were rejected. Each
repeated/reloaded prediction matches exactly on the four recorded CPU examples.
Environment: Python 3.13.18, torch 2.11.0+cpu, NumPy 2.4.6, one thread.

The earlier preparation record correctly says prepared_not_executed; the later
report establishes actual inference. These are different stages, not contradictory
claims. Self-produced hashes verify integrity and consistency, not trusted signatures.
No model was rerun during the final local evidence review.

Evidence ZIP SHA-256:
63e31125a18731f83958312f8e248e69bd4697148ed674b898ae97eaee43f5f4.

PORTFOLIO_MANIFEST.json covers only the published candidate result files and this
guide. The private full archives and earlier full curated manifest remain separate.
These diagnostics do not change the measured test accuracy or establish universal
reload equality across environments. The original 86.29% experiment is separately
summarized in ../reports/original_results.md under a different protocol.

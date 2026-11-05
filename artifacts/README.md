# Local inference artifact — excluded from the proposed public files

The selected trained model is retained locally at:

    artifacts/improved/run_20261105T101047_013570Z/model.pt
    artifacts/improved/run_20261105T101047_013570Z/metadata.json

Only these two files are required for Predictor.load. Metadata contains the exact
20,000-entry vocabulary, PAD/UNK IDs, architecture/configuration, preprocessing,
labels, threshold, provenance and expected weight SHA-256:

    740572862066b6e39dcefd2ddafa4c77f2fbd8560590cd24a559851d65953718

The complete original results ZIP is retained locally, unchanged. To reconstruct
the inference directory on another machine, copy its selected model/model.pt and
model/metadata.json into the same directory. Do not substitute the diagnostic
partial checkpoint for the final artifact bundle.

With compatible dependencies already available, from the project root:

    python -m sentiment.predict --artifact artifacts/improved/run_20261105T101047_013570Z --text "The acting was excellent."

This command is an instruction, not a prediction executed during the local audit.
The actual trained checkpoint has now passed an independent Colab CPU load/predict/
reload check: 8 tests, zero failures/errors/skips, unchanged weights, no retraining.
See [the recorded evidence](../results/improved/run_20261105T101047_013570Z/final_inference/final_inference_report.json).
Local PyTorch remains unavailable, and no model was executed during this review.

Load only trusted artifacts. The loader uses weights_only=True, CPU map_location,
schema validation and a weight checksum; a checksum is not a publisher signature.
No dataset is required for inference. Blank/unsupported inputs raise an error;
positive_probability is not calibrated confidence. To release weights later,
complete the rights/license review first. All artifact subdirectories remain ignored.

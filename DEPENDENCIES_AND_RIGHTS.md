# Dependency and content rights review — 2026-11-05

This documents checked sources and unresolved facts. It is **not a license grant
or legal clearance** for this project. No open-source license was added for the
owner's academic code, per their instruction.

## Direct dependencies

| Dependency | Upstream license reviewed | Scope |
|---|---|---|
| PyTorch | BSD 3-clause-style, [v2.11.0 LICENSE](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/LICENSE) | Runtime/inference/training; recorded Colab version 2.11.0+cu130 |
| NumPy | BSD 3-Clause, [upstream LICENSE](https://raw.githubusercontent.com/numpy/numpy/main/LICENSE.txt) | Runtime seed/numerical dependency; recorded version 2.4.6 |
| Matplotlib | Matplotlib Development Team license agreement, [v3.10.0 LICENSE](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/LICENSE/LICENSE) | Reporting/plots; recorded version 3.10.0 |
| JupyterLab | BSD 3-Clause, [upstream LICENSE](https://raw.githubusercontent.com/jupyterlab/jupyterlab/main/LICENSE) | Optional local notebook environment |
| ipykernel | BSD 3-Clause, [upstream LICENSE](https://raw.githubusercontent.com/ipython/ipykernel/main/LICENSE) | Optional local notebook kernel |

The reviewed BSD-style texts require preserving notices/conditions when their
code or binaries are redistributed and prohibit using names for endorsement
without permission. Matplotlib has its own agreement and conditions, including
notices and a change summary for distributed modifications; it is not simply
labeled BSD here. Read the linked texts for the actual conditions.

The candidate repository contains imports, dependency requirements and original
project code, not vendored dependency sources, wheels or runtime environments.
The owner must recheck applicable version-specific and bundled third-party terms
if later distributing a container, executable, CUDA libraries or dependency copies.
This was not a full transitive dependency/SBOM or NVIDIA redistribution audit.
NumPy/Jupyter links describe upstream current text; no exact installed wheel
license inventory was obtainable from the saved run. Python standard-library
modules are used, but the Python interpreter is not redistributed here.

## Code and academic contributions

The owner describes this as their academic project, with final rights review
pending. Source headers/references were inspected; no explicit external copied
code license notice was found in the project modules. **Absence of a notice does
not prove sole ownership, originality, or institutional permission.** Course
templates, supervisor/coauthor contributions, assignment policies and any copied
snippets need owner confirmation. Do not remove legitimate notices or add an
open-source license on another contributor's behalf.

The new English inference examples are authored diagnostic sentences, not copied
review excerpts. Reports/PNG charts are generated experiment outputs; the prior
audit verified their consistency. This does not establish every downstream right
associated with the underlying review corpus.

## Dataset, weights and existing content

The owner confirmed the [Kaggle source](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
Its publisher page still lists **Other (specified in description)** and refers
to Stanford. That designation is not an independently verified redistribution
grant. A research citation does not substitute for licensing terms. No CC0/MIT
license is asserted for the CSV or learned model/vocabulary.

Raw CSV, the original notebook with review excerpts, all weight/vocabulary bundles,
input/results ZIPs, caches, backups and temporary outputs remain excluded.
The candidate public results contain aggregate metrics, plots and reviewed logs,
not raw reviews. Paper links are references only; no paper PDF/web artwork is copied.
Dataset/derived-asset redistribution remains unresolved and requires review before
a model release or raw-content publication. See LICENSING.md.

## Remaining decisions

- Confirm the portions the owner can publish, any institutional/coauthor consent
  and required attribution, then select a code license at final review.
- Confirm applicable dataset/derived-model rights before distributing omitted assets.
- Approve the exact final file allowlist. No repository/public upload is authorized
  by this document, and no external publication has occurred.

# Package validation — 2026-11-08

**Software validation only. These are not full-budget benchmark results or evidence of improved generalization.**

- 13 automated tests passed: exact public parameter counts, both architectures and both profiles fitting/reloading, new feature IDs, padding invariance, unordered patch invariance, missing structures, training-only normalization, positive-class/cutoff semantics, five distinct holdouts, sequence alignment and family-leakage rejection.
- Four public XLSX files match every JSON record (2,320 total). All six delivered workbooks contain no formula/error cells; previews inspected. The independent reader is openpyxl/pandas; release authoring uses artifact-tool. Portable runtime exports use XlsxWriter.
- All ten frozen PSI split specifications (five random, five grouped) pass membership/class/OOF checks; grouped family overlap is zero. PSI has zero retained sequence/family overlap with Garcia, NESG and eSOL.
- All 202 descriptors were regenerated and compared with the original feature matrices. The release archive has SHA-256 checks for itself and each member; ESM encoder/revision, token precision and pooling policy are recorded.
- The quickstart notebook executed all four code cells in a real Jupyter kernel with downloads/full training disabled; the complete source notebook is validated against the Python files.
- Both public architectures ran two-epoch capped nested fitting on the same deliberately balanced 128-row PSI software fixture (64 insoluble/64 soluble, 128 distinct families). Five distinct random holdouts and five distinct grouped holdouts per model, three OOF fits plus one outer fit per seed: 80 fitting jobs total, each with separate inner selection and fresh refit. All selected epochs were at most2; all inner family overlaps were zero.
- Each outer model also inferred Garcia284, NESG152, and eSOL198; binary and continuous metrics are retained. External labels never supplied fitting/stopping/threshold inputs. These capped results must not be used to select an architecture or claim scientific performance.
- The `fit-final` / `predict` path was exercised with a two-epoch MLP on the smoke fixture and284 Garcia candidates. Checkpoint reload worked and every reject flag matched `score >= stored_threshold`.

## Smoke metrics (mean ± sample SD over five seeds)

The fixture is deliberately balanced and contains one sequence per selected family, so its performance is not representative of full PSI or an internal assay. Nontrivial multi-member family isolation is separately checked on the complete frozen PSI split manifest.

|Architecture|Mode|Partition|Insoluble AP|AUROC|
|---|---|---|---:|---:|
|residual_mlp|grouped|holdout|0.710 ± 0.086|0.726 ± 0.081|
|residual_mlp|grouped|oof|0.629 ± 0.073|0.625 ± 0.065|
|residual_mlp|random|holdout|0.738 ± 0.115|0.714 ± 0.097|
|residual_mlp|random|oof|0.655 ± 0.054|0.669 ± 0.021|
|bilstm|grouped|holdout|0.712 ± 0.084|0.745 ± 0.062|
|bilstm|grouped|oof|0.603 ± 0.035|0.615 ± 0.056|
|bilstm|random|holdout|0.751 ± 0.125|0.747 ± 0.095|
|bilstm|random|oof|0.608 ± 0.055|0.638 ± 0.050|

Per-seed OOF, holdout and external metrics, predictions, fit histories and memberships are in `smoke/`; XLSX summaries are beside this report. PNGs explicitly identify the results as smoke checks and display positive prevalence. Prior full-budget results were not rerun or relabeled as results of this implementation. Compact structure profiles use synthetic test inputs only; no internally measured structures or labels were available. Parameter initialization ordering differs from earlier scripts, so bitwise equality to historical trained checkpoints is not claimed.

CI runs the unit tests, XLSX/JSON/frozen-split validation and notebook source checks. Full ESM extraction is not part of CI; the pinned feature cache supplies the tested public inputs.

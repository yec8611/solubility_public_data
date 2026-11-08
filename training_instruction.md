# Training instructions

Package creation date: **2026-11-08**. Target: **insoluble=1**, soluble=0. Choose one architecture before examining held-out outcomes. Start with the residual MLP for approximately 2,000 internally labeled proteins; the BiLSTM adds sequence-order capacity that should justify itself on family-disjoint validation.

## 1. Install and verify

Use Python 3.11 or later. CPU training is supported; CUDA can accelerate it. Frozen ESM extraction needs substantially more memory than training either small head. The release cache avoids ESM extraction for the 2,320 public proteins. Apple MPS recurrent/deterministic operators are not guaranteed; CPU is the supported default.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,esm]'
python scripts/download_features.py --output features/public
python scripts/validate_data.py
pytest -q
```

The downloader verifies SHA-256 for the archive and every member. It refuses a nonempty destination and unsafe archive members. No pretrained solubility classifier is bundled; the release contains frozen input embeddings. Installed dependencies are specified in `pyproject.toml`; the tested local versions are recorded in `validation/environment_20261108.json`.

## 2. Data and leakage boundaries

The public primary pool has 1,686 proteins (368 insoluble, 1,318 soluble) in 737 sequence-family groups. It is not the user's approximately 2,000 internal proteins. External panels are Garcia 284, NESG 152, and eSOL 198 (151 binary extremes, 47 continuous-only). The training command rejects `role=external` rows in the fitting dataset and rejects exact sequence/group overlaps with external panels.

Use `data/public/psi_records_20261108.xlsx`, sheet `Records`. Input columns needed for training are `record_id`, `sequence`, `insoluble`, `group_id`, and `role=train`. `dataset` is an output label. IDs and canonical sequences must match the feature cache exactly; sequences must be 20–100 aa, uppercase canonical residues. No truncation, silent label inversion, source IDs, assay results, or solubility readouts enter X. Assay conditions in the source files are incompletely recorded; keep future internal assays and endpoints consistent.

For your internal XLSX, use the same `Records` schema. Group related sequences by parent/scaffold and sequence homology before splitting, including aliases/tag-stripped forms. The existing public families used connected components at 40% identity with 80% bidirectional coverage plus alias relationships. The package consumes group assignments; it does **not** pretend that arbitrary unique row IDs are valid homology groups. Experimental duplicates/variants must be resolved without separating the same family across folds. This is essential for the intended design-to-synthesis setting.

## 3. Features

Encoder: `facebook/esm2_t33_650M_UR50D`, pinned revision `08e4846e537177426273712802403f7ba8261b6c`, final hidden layer. It remains in evaluation mode with no gradient updates. Exclude BOS, EOS, and padding. Compute coordinate-wise mean and maximum over real residue tokens in float32 and concatenate to 2,560 values. Residue caches store float16 and are converted to float32 in each batch. The 202 sequence-only descriptors are listed in `data/manifests/descriptor_names_20261108.json` and implemented in `descriptors.py`.

To create features for internal or new prediction sequences:

```bash
python scripts/extract_features.py --records private_data/internal_20261108.xlsx --output features/internal --device cpu --batch-size 8
```

Feature cache filenames have the schema version date `20261108`; run outputs use their creation date. For prospective scores, reuse the same pinned encoder and feature definitions. Structure profiles need an additional NPZ sidecar described in [architecture_and_schema_20261108.md](docs/architecture_and_schema_20261108.md). No experimental structure values are inferred or invented by this package.

## 4. Exact public architectures

Every projection block is Linear → LayerNorm → GELU → Dropout(0.2), except the ESM residue projection/dropout, which uses 0.1.

**Residual fusion MLP:** descriptors 202→64; concatenated ESM mean/max 2560→64. Concatenate the two representations to 128; gate MLP 128→32→2 with GELU then softmax. Multiply each 64-D representation by its gate and concatenate (do not sum). Apply two blocks `LayerNorm(128)→Linear(128,96)→GELU→Dropout(.2)→Linear(96,128)`. Each block updates `x ← x + sigmoid(r) * F(x)`, where the trainable vector r starts at zero (scale starts at 0.5, not zero). Final projection 128→96 then Linear96→1. Total: **244,387** trainable parameters.

**BiLSTM fusion:** the same descriptor and global ESM projections, plus per-residue ESM1280→64→one bidirectional LSTM with hidden size32 per direction. Pack sequences before recurrence so neither direction reads padding. Apply output dropout0.1, masked mean+max over the 64-D outputs, then128→64. The three 64-D channels give192 inputs to gate192→32→3. Use two residual192→96→192 blocks, projection192→96, and Linear96→1. Total: **393,284** trainable parameters.

During public-profile fitting, independently drop each view with probability0.15. If all are dropped, retain the pooled ESM view. Softmax excludes dropped views. Gate weights are input-dependent; residual scales are learned static vectors. Neither is a causal feature attribution.

## 5. Loss and optimizer

- AdamW, learning rate0.001, weight decay0.01 applied to all trainable parameters; no scheduler or warmup.
- Positive class weight is `sqrt(n_soluble/n_insoluble)`, recomputed from the rows actually fitted. For 1,600/400 this is2.0, not4.0.
- Uniform shuffled protein minibatches, no class oversampling. Public MLP batch128; BiLSTM and structure profiles batch64.
- Two dropout passes: `loss = 0.5*(weighted_BCE(z1,y)+weighted_BCE(z2,y)) + 0.5*Dsym`, where `Dsym=0.5*(KL(Bern(sigmoid(z1)) || Bern(sigmoid(z2))) + reverse KL)` averaged over proteins.
- Clip gradient norm at5.0. Frozen ESM is excluded from optimization.
- Public profiles independently resample view masks across the two passes to retain the historical protocol. Compact structure profiles share a structure-drop mask across passes and independently sample ordinary unit dropout.
- Maximum60 epochs, patience8, improvement tolerance1e-5, selection metric **inner-validation insoluble average precision**. Seeds11,29,47,71,101. CPU uses two threads and deterministic PyTorch operations. CUDA users should set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python if deterministic matrix operations require it.

## 6. Five-seed OOF and holdout protocol

For each seed, produce a different approximately80/20 outer train/holdout partition. Random mode is class-stratified; grouped mode takes the first fold of shuffled five-fold StratifiedGroupKFold. Group sizes make the grouped test fraction vary. The frozen PSI random splits contain1,348 training and338 holdout rows for each seed, with294/74 insoluble cases. Grouped train/holdout sizes are1261/425,1370/316,1423/263,1382/304,1399/287.

Within each outer training partition, create three OOF folds (stratified in random mode; stratified and family-grouped in grouped mode). Each OOF model receives only the other two folds. Inside **every fit**, including each OOF fit:

1. Generate100 group-disjoint candidates with validation fraction0.2 and RNG `fit_seed+101`. Require both classes in fitting and validation. Choose the candidate minimizing class-prevalence deviation plus size deviation; no model performance is used to choose the partition.
2. Fit pooled/descriptor standardization using only that inner fitting subset. For structure data, compute mean imputation and scaling only from available valid items belonging to inner fitting proteins. ESM residue states are not dataset-standardized; they use learned projection and LayerNorm.
3. Train with early stopping on inner validation AP and remember the best epoch count.
4. Reset model weights, RNG, and optimizer from the same fit seed. Recompute preprocessing on **all rows supplied to this fit** and train from scratch for the selected count. No warm starts or inner-validation checkpoint continuation.
5. Predict only the reserved OOF fold. After all OOF folds, independently repeat this fit procedure on the entire outer training subset, then predict the untouched outer holdout and external panels.

Fit seeds are `outer_seed + fold*10000`; folds0–2 are OOF, fold3 is the outer training refit. Both label classes must occur in each fitting/evaluation split. Five holdout memberships must be distinct. Random results permit family overlap and are a sensitivity analysis; grouped results are the principal novelty test. The same external panel is inferred by five models trained on different outer training sets; the external panel itself is not split or used for selection.

```bash
solubility evaluate --records data/public/psi_records_20261108.xlsx --features features/public --config configs/residual_mlp_public.json --splits data/splits/psi_five_seed_splits_20261108.json --external data/public/garcia_records_20261108.xlsx data/public/nesg_records_20261108.xlsx data/public/esol_records_20261108.xlsx --output runs/residual_mlp_20261108
solubility evaluate --records data/public/psi_records_20261108.xlsx --features features/public --config configs/bilstm_public.json --splits data/splits/psi_five_seed_splits_20261108.json --external data/public/garcia_records_20261108.xlsx data/public/nesg_records_20261108.xlsx data/public/esol_records_20261108.xlsx --output runs/bilstm_20261108
```

For internal data omit `--splits` so the five sets are generated from internal labels/groups. Pass `--structure private_data/structure_20261108.npz` with `configs/residual_mlp_structure.json` or `configs/bilstm_structure.json`. Outputs contain individual OOF/holdout/external metrics, mean/SD/count across seeds, per-protein predictions, split membership, inner histories, full-fit metadata, five checkpoints per split mode, XLSX, PNG, and Markdown. Undefined precision or accepted-set contamination is missing, not zero. `--max-epochs 1` is only a smoke check and marks its report accordingly.

## 7. Rejection thresholds and one deployment model

Thresholds are chosen solely from outer-training OOF predictions: select the **largest** cutoff achieving at least the requested insoluble recall (default0.90). Reject when `score >= cutoff`. Report both default0.5 and the OOF-selected cutoff. Metrics include AP, AUROC, MCC, precision, recall, specificity, soluble retention, rejected fraction, and insolubility among accepted sequences, plus Brier/log loss for the uncalibrated scores. eSOL also reports Spearman correlation with negative measured soluble fraction. OOF metrics at a threshold selected on those same OOF labels are apparent; holdout metrics are the assessment.

This rule does not guarantee90% recall or a low accepted-set failure rate in a new assay. For synthesis decisions, prespecify the acceptable tradeoff, assess it on assay-matched family-disjoint holdouts, and verify it prospectively. Sigmoid outputs are **not calibrated probabilities**; calibration is intentionally not fitted on the external panels. If a probability target is required, add a separate training-only calibration design before assessing an untouched confirmation set.

Once an architecture and operating rule are fixed, train **one** deployment model on all permitted internal rows:

```bash
solubility fit-final --records private_data/internal_20261108.xlsx --features features/internal --structure private_data/structure_20261108.npz --config configs/residual_mlp_structure.json --output runs/deployment_20261108 --seed 11 --recall 0.90
solubility predict --records private_data/candidates_20261108.xlsx --features features/candidates --structure private_data/candidate_structure_20261108.npz --checkpoint runs/deployment_20261108/deployment_20261108.pt --output runs/candidate_scores_20261108
```

The output checkpoint uses the actual run date, so substitute that date when predicting. `fit-final` obtains a cutoff from grouped three-fold OOF over the entire training dataset, then performs the inner-selection/fresh-refit procedure on all rows. It creates no new independent holdout; use `evaluate` first. Prediction records can omit labels and group IDs, but require IDs and sequences. A checkpoint can score new feature-cache IDs; structure feature names and order must match training. `private_data/`, `features/`, `runs/`, and checkpoint files are ignored by Git.

## 8. Optional weak-label primary training

```bash
python scripts/import_weak.py --input /path/to/weak_20261106.npz
python scripts/extract_features.py --records private_data/weak_records_20261108.json --output features/weak
```

This imports exactly17,638 independent weak rows (6,042 positive). It does not republish them or silently mix them into PSI. To reproduce the weak-primary budget, copy the chosen public config and set maximum40 epochs, patience6, and batch256 for the MLP (batch128 for BiLSTM). Generate five-seed splits from the weak group IDs and audit overlap with every external panel before evaluating. Upstream terms vary; see `DATA_LICENSES.md`. Joint multi-head weak-auxiliary and ensemble models are outside this two-model package.

## 9. Reproducibility and scope

Run `python scripts/build_notebooks.py` after changing Python source. The complete source notebook contains every Python module/script/test with annotations; the quickstart is executable with expensive training/download disabled initially. Run `python scripts/build_notebooks.py --check` in CI.

The public head topologies and training protocol are ported, but parameter ordering and framework differences mean this package is **not asserted to reproduce prior checkpoint scores bit-for-bit**. The release validation checks code/data integrity and runs capped fitting; it is not a fresh full-budget model comparison. Compact structure profiles have only software/synthetic validation. No ensemble, external-label tuning, full ESM finetuning, or new scientific superiority claim is introduced.

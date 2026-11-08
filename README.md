# solubility_public_data

Two reproducible neural architectures for filtering **20–100 aa mini-proteins** before synthesis. **Insolubility is the positive class (1)**. The package contains public sequence/label data, frozen five-seed splits, feature extraction/download scripts, training and prediction code, optional structure-feature profiles, notebooks, tests, and a complete [training guide](training_instruction.md).

| Dataset | Role | Proteins | Insoluble | Soluble | Continuous-only |
|---|---|---:|---:|---:|---:|
| PSI | Training | 1,686 | 368 | 1,318 | 0 |
| Garcia | External retrospective evaluation | 284 | 44 | 240 | 0 |
| NESG | External retrospective evaluation | 152 | 32 | 120 | 0 |
| eSOL | External retrospective evaluation | 198 | 24 | 127 | 47 |

Data are available as **XLSX**, with identical JSON records for programmatic loading. eSOL's binary subset uses strict `<0.30` insoluble and `>0.70` soluble cutoffs; all 198 measured fractions remain available. These panels have been analyzed previously and are not blinded confirmation datasets. Mini-protein length refers to the released canonical/core sequence, which can differ from the assayed tagged construct. Antibody screening is documented but cannot rule out every short immunoglobulin fragment.

| Architecture | Public inputs | Trainable parameters |
|---|---|---:|
| Residual fusion MLP | 202 sequence descriptors + ESM650 mean/max | 244,387 |
| BiLSTM fusion | Same inputs + ESM650 residue states through a packed BiLSTM | 393,284 |

ESM650 stays frozen. The compact `structure` profiles replace the descriptor channel with global/residue/patch structural features; they are implemented and tested on synthetic fixtures, **not validated on experimental structure features**. No internal experimental data are distributed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,esm]'
python scripts/download_features.py
python scripts/validate_data.py
pytest -q
solubility evaluate \
  --records data/public/psi_records_20261108.xlsx \
  --features features/public \
  --config configs/residual_mlp_public.json \
  --splits data/splits/psi_five_seed_splits_20261108.json \
  --external data/public/garcia_records_20261108.xlsx data/public/nesg_records_20261108.xlsx data/public/esol_records_20261108.xlsx \
  --output runs/residual_mlp_20261108
```

Use `configs/bilstm_public.json` and a different output directory for the second model. Both random and grouped modes run by default: **five distinct holdouts, three-fold training OOF, inner group-disjoint stopping, fresh refit**, and thresholds selected from training OOF scores. The external panels never select models, epochs, features, or cutoffs. This is a release/validation package, not a claim of a new performance improvement; see [validation](validation/package_validation_20261108.md).

The existing 17,638 mixed weak-label records are supported by a **local importer**; the combined dataset is not republished because upstream redistribution terms differ. See [data rights and attribution](DATA_LICENSES.md). The code license does not relicense third-party data.

- [Full training and deployment instructions](training_instruction.md)
- [Architecture and structure schema](docs/architecture_and_schema_20261108.md)
- [Quickstart notebook](notebooks/quickstart_20261108.ipynb)
- [All Python source merged into one annotated notebook](notebooks/complete_source_20261108.ipynb)
- [Dataset provenance and caveats](docs/data_card_20261108.md)

Runtime tables use XLSX; plots use PNG; generated artifacts include their creation date. Model scores are uncalibrated and require prospective, assay-matched validation before setting a synthesis rejection policy.

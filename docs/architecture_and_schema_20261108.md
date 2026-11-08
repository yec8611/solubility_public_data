# Architecture and structure schema — 2026-11-08

The two public networks use the exact widths documented in `training_instruction.md`. Their trainable parameter counts are checked in automated tests. The encoder is frozen ESM650, not ESM150. Mean/max means concatenation of two1280-D vectors before the projection; it does not introduce two separate ESM channels.

## Compact structure profiles

These are optional software implementations of the internal-data proposal. They have not been validated on experimental structure features.

| Channel | Processing | Output |
|---|---|---:|
| Global ESM | Mean+max2560 → Linear64 → LN → GELU → Dropout.2 |64|
| Global structure | Dg → Linear32 → LN → GELU → Dropout.2 |32|
| Residue structure | Shared Dr→32 block, masked mean+max,64→16 block |16|
| Patch structure | Shared Dp→32 block, masked mean+max,64→16 block |16|
| Combined structure | Concatenate32+16+16; LayerNorm64 |64|
| Optional ESM BiLSTM |1280→64 projection; packed BiLSTM32 each direction; masked mean+max;128→64 |64|

The residual MLP gates structure and pooled ESM: `(128+6)→16→2`; BiLSTM adds the sequence branch: `(192+6)→16→3`. Six extra gate inputs are three availability flags and three quality scores. Weighted concatenation is followed by **one** residual block of width128 or192 with hidden64, then a32-D projection and one logit. The residual scale vector starts at0.5. Structure-only branch dropout is0.15, with the same mask in both R-Drop passes. ESM remains available. Exact parameter counts depend on Dg,Dr,Dp and are written into each fit report.

Global summaries, sequence-position features and spatial patches have different coordinate systems. The separate encoders prevent arbitrary patch order from becoming a sequence signal. Residue structural summaries in this compact profile are permutation-invariant; sequence order is already represented by ESM and, in the BiLSTM architecture, a dedicated ESM sequence branch. Spatial patch membership/order is never treated as peptide order.

## Structure NPZ sidecar

Use `numpy.savez_compressed`; no object arrays or pickle. ID order may differ from the feature cache; alignment is by ID and SHA-256 of the exact sequence. Store one structural feature definition/order consistently across training and inference.

| Key | Shape / meaning |
|---|---|
| `ids` |Unicode[N], unique record IDs|
| `sequence_sha256` |Unicode[N], SHA-256 of each uppercase canonical sequence|
| `global_values` |float[N,Dg]|
| `global_available` |bool[N], structural calculation succeeded|
| `global_feature_names` |Unicode[Dg], unique names in feature order|
| `residue_values` |float[sum(L),Dr], one row per residue when available|
| `residue_offsets` |int[N+1], first0, last total items, monotonically nondecreasing|
| `residue_mask` |bool[sum(L)], valid observed/predicted residues|
| `residue_available` |bool[N]|
| `residue_feature_names` |Unicode[Dr]|
| `patch_values` |float[sum(P),Dp], variable unordered patches|
| `patch_offsets` |int[N+1], first0, last total patches|
| `patch_mask` |bool[sum(P)], valid patches|
| `patch_available` |bool[N]|
| `patch_feature_names` |Unicode[Dp]|
| `quality` |float[N,3] in[0,1], predeclared confidence/coverage scores for global/residue/patch calculations|

Dg,Dr,Dp must be positive. Unavailable residue/patch collections may have zero items; a successful patch calculation finding no patches is `patch_available=1` with zero patches, whereas a failed calculation is `patch_available=0`. This distinction is preserved in gate inputs. Empty set encodings become zero. Padded items are never pooled. Feature-wise missing numerical values use fitting-only mean imputation and standardization; item validity and modality availability are separate. Do not put insolubility, yield, expression success, assay readouts, or post-hoc outcome-informed quality into these fields.

Normalization is refitted independently inside every OOF/holdout fitting job and again for its final refit. Structural normalization uses valid items from fitting proteins only. This implementation weights each valid residue/patch equally when estimating its feature statistics; long proteins or proteins with many patches contribute more items. Global summary statistics weight proteins equally. Checkpoints persist normalization and ordered feature names. Scoring a candidate with a changed feature definition requires retraining.

## Equations

For representations h_j and a presence/drop mask m_j:

`g = softmax(gate(concat(m_j*h_j), availability, quality), unavailable=-infinity)`

`x = concat(g_j * m_j * h_j)`

`x_next = x + sigmoid(r) * residual_MLP(x)`

`score = sigmoid(linear(output_projection(x_next)))`

The public profile omits the availability/quality inputs. All available views receive normalized weights summing to1. Weights measure how the fitted network combines its channels; they are not causal importance estimates or calibrated reliability probabilities.

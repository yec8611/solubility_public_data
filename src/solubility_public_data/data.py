"""ID/sequence-checked data interfaces. Source metadata never enters features."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch

AA = set('ACDEFGHIKLMNPQRSTVWY')
SEEDS = [11, 29, 47, 71, 101]


def read_records(path, training=False):
    path = Path(path)
    if path.suffix == '.xlsx':
        records = pd.read_excel(path, sheet_name='Records', keep_default_na=False).to_dict('records')
    else:
        records = json.loads(path.read_text())
    if not records:
        raise ValueError('Empty records')
    ids = [str(r['record_id']) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate record IDs')
    for r in records:
        seq = r['sequence']
        if not isinstance(seq, str) or not 20 <= len(seq) <= 100 or set(seq) - AA:
            raise ValueError('Require canonical 20-100 aa sequences, without silent truncation')
        label = r.get('insoluble')
        if label is None or label == '' or pd.isna(label):
            r['insoluble'] = None
        elif label not in (0, 1):
            raise ValueError('insoluble must be 1, 0, or missing')
        if 'solubility_fraction' in r and (r['solubility_fraction'] == '' or r['solubility_fraction'] is None or pd.isna(r['solubility_fraction'])):
            r['solubility_fraction'] = None
        if training:
            if r.get('role') != 'train' or r['insoluble'] is None:
                raise ValueError('Training requires role=train and binary labels; external records prohibited')
            if not r.get('group_id') or pd.isna(r['group_id']):
                raise ValueError('Training requires sequence/scaffold family group_id')
    if training and (len({r['sequence'] for r in records}) != len(records) or {r['insoluble'] for r in records} != {0, 1}):
        raise ValueError('Training requires unique sequences and both classes')
    return records


def subset(data, indices):
    return {k: v[indices] for k, v in data.items()}


class FeatureStore:
    """Portable pooled NPZ and optional memory-mapped residue array."""
    def __init__(self, directory, structure=None):
        self.directory = Path(directory)
        with np.load(self.directory / 'pooled_20261108.npz', allow_pickle=False) as a:
            self.pooled = {k: a[k] for k in a.files}
        ids = self.pooled['ids'].astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate feature IDs')
        self.lookup = {s: i for i, s in enumerate(ids)}
        n = len(ids)
        for key, width in [('desc', 202), ('esm650', 1280), ('esm650_max', 1280)]:
            if self.pooled[key].shape != (n, width) or not np.isfinite(self.pooled[key]).all():
                raise ValueError(f'Invalid {key} feature matrix')
        self.tokens = None
        if (self.directory / 'residues_20261108.npy').exists():
            self.tokens = np.load(self.directory / 'residues_20261108.npy', mmap_mode='r', allow_pickle=False)
            off = self.pooled['offsets']
            if off.shape != (n+1,) or off[0] != 0 or off[-1] != len(self.tokens) or not np.array_equal(np.diff(off), [len(s) for s in self.pooled['sequences']]):
                raise ValueError('Residue offsets/sequence lengths mismatch')
            if self.tokens.shape[1:] != (1280,):
                raise ValueError('ESM650 token width must be 1280')
        self.structure = None
        if structure:
            with np.load(structure, allow_pickle=False) as s:
                self.structure = {k: s[k] for k in s.files}
            self._validate_structure()

    def _validate_structure(self):
        s = self.structure
        ids = s['ids'].astype(str); n = len(ids)
        if len(set(ids)) != n:
            raise ValueError('Duplicate structure IDs')
        self.structure_lookup = {v: i for i, v in enumerate(ids)}
        if s['global_values'].ndim != 2 or s['global_values'].shape[0] != n or s['quality'].shape != (n, 3):
            raise ValueError('Bad structure global/quality shape')
        self.structure_dims = {'global': s['global_values'].shape[1]}
        for key in ['global', 'residue', 'patch']:
            a = s[key+'_available']
            if a.shape != (n,) or not np.isin(a, [0, 1]).all():
                raise ValueError('Availability flags must be binary per protein')
            if key == 'global':
                continue
            v, o, mask = s[key+'_values'], s[key+'_offsets'], s[key+'_mask']
            if v.ndim != 2 or o.shape != (n+1,) or o[0] != 0 or o[-1] != len(v) or np.any(np.diff(o) < 0) or mask.shape != (len(v),) or not np.isin(mask, [0, 1]).all():
                raise ValueError('Invalid ragged structure schema')
            self.structure_dims[key] = v.shape[1]
        if not np.isfinite(s['quality']).all() or np.any((s['quality'] < 0) | (s['quality'] > 1)):
            raise ValueError('Declare quality values in [0,1], never use assay outcomes')
        for key in self.structure_dims:
            names = s[key+'_feature_names']
            if len(names) != self.structure_dims[key] or len(set(names)) != len(names):
                raise ValueError('Feature names must be unique and match dimensions')
        self.structure_schema = {k: s[k+'_feature_names'].tolist() for k in self.structure_dims}

    def align(self, records, need_tokens=False, need_structure=False):
        ix = np.asarray([self.lookup[str(r['record_id'])] for r in records])
        sequences = np.asarray([r['sequence'] for r in records])
        if not np.array_equal(sequences, self.pooled['sequences'][ix]):
            raise ValueError('Sequence mismatch for feature ID')
        if need_tokens and self.tokens is None:
            raise ValueError('BiLSTM requires residue cache; download or extract features first')
        data = {k: self.pooled[k][ix] for k in ['desc', 'esm650', 'esm650_max']}
        data['rows'] = ix
        if need_structure:
            if self.structure is None:
                raise ValueError('Structure profile requires a structure NPZ sidecar')
            si = np.asarray([self.structure_lookup[str(r['record_id'])] for r in records])
            sha = [hashlib.sha256(s.encode()).hexdigest() for s in sequences]
            if not np.array_equal(self.structure['sequence_sha256'][si], sha):
                raise ValueError('Structure sequence hash mismatch')
            offsets = self.structure['residue_offsets']
            # Available residue structures must align one item to every sequence residue.
            for i, seq in zip(si, sequences):
                if self.structure['residue_available'][i] and offsets[i+1]-offsets[i] != len(seq):
                    raise ValueError('Structure residue coordinates must align to sequence')
            data['structure_rows'] = si
        return data

    def scalers(self, data, ix):
        result = {}
        for key in ['desc', 'esm650', 'esm650_max']:
            x = data[key][ix].astype(np.float64)
            result[key] = self._stats(x)
        if 'structure_rows' in data:
            s, rows = self.structure, data['structure_rows'][ix]
            for key in ['global', 'residue', 'patch']:
                pieces = []
                for row in rows:
                    if not s[key+'_available'][row]:
                        continue
                    if key == 'global':
                        pieces.append(s['global_values'][row:row+1])
                    else:
                        lo, hi = s[key+'_offsets'][row:row+2]
                        pieces.append(s[key+'_values'][lo:hi][s[key+'_mask'][lo:hi].astype(bool)])
                width = self.structure_dims[key]
                x = np.concatenate(pieces) if pieces else np.empty((0, width))
                result['structure_'+key] = self._stats(x)
        return result

    @staticmethod
    def _stats(x):
        # Missing individual structure values: fit-only mean imputation, then z-score.
        valid = np.isfinite(x)
        count = valid.sum(0)
        mean = np.where(valid, x, 0.).sum(0) / np.maximum(count, 1)
        var = np.where(valid, x-mean, 0.)**2
        scale = np.sqrt(var.sum(0) / np.maximum(count, 1))
        scale[scale < 1e-12] = 1.
        return {'mean': mean, 'scale': scale}

    def batch(self, data, ix, scalers, device='cpu', need_tokens=False):
        def tensor(a, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        def norm(a, key):
            stat = scalers[key]
            return (np.where(np.isfinite(a), a, stat['mean'])-stat['mean'])/stat['scale']
        b = {k: tensor(norm(data[k][ix], k)) for k in ['desc', 'esm650', 'esm650_max']}
        if need_tokens:
            rows = data['rows'][ix]; offsets = self.pooled['offsets']
            lengths = offsets[rows+1]-offsets[rows]
            values = np.zeros((len(ix), int(lengths.max()), 1280), np.float32)
            for j, row in enumerate(rows):
                raw = self.tokens[offsets[row]:offsets[row+1]]
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite valid residue states')
                values[j, :lengths[j]] = raw
            b.update(residue=tensor(values), mask=tensor(np.arange(values.shape[1])[None, :] < lengths[:, None], torch.bool), lengths=torch.as_tensor(lengths, dtype=torch.long))
        if 'structure_rows' in data:
            s, rows = self.structure, data['structure_rows'][ix]
            available = np.column_stack([s[k+'_available'][rows] for k in ['global', 'residue', 'patch']])
            b['structure_available'] = tensor(available)
            b['structure_quality'] = tensor(s['quality'][rows] * available)
            b['structure_global'] = tensor(norm(s['global_values'][rows], 'structure_global') * available[:, :1])
            for key in ['residue', 'patch']:
                offsets = s[key+'_offsets']; lengths = offsets[rows+1]-offsets[rows]
                width = max(1, int(lengths.max()))
                values = np.zeros((len(rows), width, self.structure_dims[key]), np.float32)
                mask = np.zeros((len(rows), width), bool)
                for j, row in enumerate(rows):
                    lo, hi = offsets[row:row+2]
                    values[j, :hi-lo] = norm(s[key+'_values'][lo:hi], 'structure_'+key)
                    mask[j, :hi-lo] = s[key+'_mask'][lo:hi].astype(bool) & bool(s[key+'_available'][row])
                b['structure_'+key] = tensor(values)
                b['structure_'+key+'_mask'] = tensor(mask, torch.bool)
        return b

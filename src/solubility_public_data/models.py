"""Two gated residual architectures; public and optional structure profiles.

The public profiles reproduce the previously studied head topology. Compact
structure profiles are proposals, not experimentally validated improvements.
"""
import torch
from torch import nn


def block(n, m, p=.2):
    return nn.Sequential(nn.Linear(n, m), nn.LayerNorm(m), nn.GELU(), nn.Dropout(p))


def masked_pool(x, mask):
    """Mean/max over valid positions; empty sets map to exact zeros."""
    mask = mask.bool().unsqueeze(-1)
    clean = x.masked_fill(~mask, 0.)
    mean = clean.sum(1) / mask.sum(1).clamp_min(1)
    maximum = x.masked_fill(~mask, -torch.inf).max(1).values
    maximum = torch.where(mask.any(1), maximum, torch.zeros_like(maximum))
    return torch.cat([mean, maximum], -1)


class SetEncoder(nn.Module):
    """Shared item MLP, permutation-invariant masked pooling, 16-D output."""
    def __init__(self, width):
        super().__init__()
        self.items = block(width, 32)
        self.output = block(64, 16)

    def forward(self, x, mask):
        h = self.items(x.masked_fill(~mask.unsqueeze(-1), 0.))
        result = self.output(masked_pool(h, mask))
        return result * mask.any(1, keepdim=True)


class StructureEncoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.global_branch = block(dims['global'], 32)
        self.residue_branch = SetEncoder(dims['residue'])
        self.patch_branch = SetEncoder(dims['patch'])
        self.norm = nn.LayerNorm(64)

    def forward(self, b):
        global_h = self.global_branch(b['structure_global']) * b['structure_available'][:, :1]
        residue_h = self.residue_branch(b['structure_residue'], b['structure_residue_mask'])
        patch_h = self.patch_branch(b['structure_patch'], b['structure_patch_mask'])
        return self.norm(torch.cat([global_h, residue_h, patch_h], 1))


class ResidueBiLSTM(nn.Module):
    """One bidirectional layer, 32 units/direction, no padding in recurrence."""
    def __init__(self):
        super().__init__()
        self.projection = block(1280, 64, .1)
        self.encoder = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(.1)
        self.output = block(128, 64)

    def forward(self, b):
        mask = b['mask']
        h = self.projection(b['residue'].masked_fill(~mask.unsqueeze(-1), 0.))
        packed = nn.utils.rnn.pack_padded_sequence(h, b['lengths'].cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=h.shape[1])
        return self.output(masked_pool(self.dropout(h), mask))


class FusionModel(nn.Module):
    def __init__(self, architecture='residual_mlp', profile='public', structure_dims=None):
        super().__init__()
        if architecture not in {'residual_mlp', 'bilstm'} or profile not in {'public', 'structure'}:
            raise ValueError('Unknown architecture/profile')
        self.architecture, self.profile = architecture, profile
        self.structure_dims = structure_dims
        self.esm = block(2560, 64)
        if profile == 'public':
            self.other = block(202, 64)
        else:
            if not structure_dims or any(structure_dims[k] < 1 for k in ['global', 'residue', 'patch']):
                raise ValueError('Specify positive global/residue/patch dimensions')
            self.other = StructureEncoder(structure_dims)
        self.sequence = ResidueBiLSTM() if architecture == 'bilstm' else None
        views = 3 if self.sequence else 2
        width = 64 * views
        compact = profile == 'structure'
        quality_dim = 6 if compact else 0  # availability 3 + declared quality 3
        self.gate = nn.Sequential(nn.Linear(width + quality_dim, 16 if compact else 32), nn.GELU(), nn.Linear(16 if compact else 32, views))
        hidden = 64 if compact else 96
        count = 1 if compact else 2
        self.residuals = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden, width)) for _ in range(count)])
        self.residual_gates = nn.Parameter(torch.zeros(count, width))
        self.output = block(width, 32 if compact else 96)
        self.head = nn.Linear(32 if compact else 96, 1)

    def sample_keep(self, b):
        n = len(b['esm650']); v = 3 if self.sequence else 2
        keep = torch.ones(n, v, dtype=torch.bool, device=b['esm650'].device)
        if self.profile == 'structure':
            keep[:, 0] = b['structure_available'].bool().any(1)
            if self.training:
                keep[:, 0] &= torch.rand(n, device=keep.device) >= .15
        elif self.training:
            keep = torch.rand(n, v, device=keep.device) >= .15
            keep[:, 1] |= ~keep.any(1)
        return keep

    def forward(self, b, keep=None):
        # Ordering is descriptor/structure, pooled ESM, optional ESM BiLSTM.
        h = self.other(b['desc']) if self.profile == 'public' else self.other(b)
        views = [h, self.esm(torch.cat([b['esm650'], b['esm650_max']], 1))]
        if self.sequence:
            views.append(self.sequence(b))
        views = torch.stack(views, 1)
        keep = self.sample_keep(b) if keep is None else keep
        visible = views * keep.unsqueeze(-1)
        gate_inputs = visible.flatten(1)
        if self.profile == 'structure':
            gate_inputs = torch.cat([gate_inputs, b['structure_available'], b['structure_quality']], 1)
        gates = torch.softmax(self.gate(gate_inputs).masked_fill(~keep, -torch.inf), 1)
        x = (visible * gates.unsqueeze(-1)).flatten(1)
        for i, layer in enumerate(self.residuals):
            x = x + torch.sigmoid(self.residual_gates[i]) * layer(x)
        return {'logit': self.head(self.output(x)).squeeze(-1), 'gates': gates}


def symmetric_kl(a, b):
    la = torch.stack([nn.functional.logsigmoid(-a), nn.functional.logsigmoid(a)], -1)
    lb = torch.stack([nn.functional.logsigmoid(-b), nn.functional.logsigmoid(b)], -1)
    return .5 * ((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1)).mean()

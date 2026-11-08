"""Sequence-only solubility descriptors and train-only neural classifiers.

Created 2026-11-06. All features use the input sequence only. Experimental solubility, expression
yield, assay readouts, and source identities must never be passed as X.
The classifier chooses its epoch count on an internal training-only validation
split, then refits from scratch on all rows supplied to fit(). Outer validation
and holdout rows must be supplied only to predict_proba().
"""

from __future__ import annotations

import copy
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils import ProtParamData

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA)}
HYDROPHOBIC = set("AILMFWVY")
AA_GROUPS = {
    "hydrophobic": HYDROPHOBIC,
    "aromatic": set("FWY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "small": set("AGST"),
    "polar_uncharged": set("NQST"),
    "pro_gly": set("PG"),
}


def _longest_run(sequence, residue_set):
    best = current = 0
    for residue in sequence:
        current = current + 1 if residue in residue_set else 0
        best = max(best, current)
    return best


def sequence_features(sequences, include_dipeptides=False):
    """Return interpretable descriptors in fixed, deterministic column order.

    Janin and Kyte-Doolittle scales are the BioPython ProtParamData tables.
    The pH 7 charge is an explicit approximate Henderson-Hasselbalch descriptor,
    not an observed charge or a pI estimate. Terminal windows characterize
    local sequence composition without using assay or source metadata.
    """
    rows = []
    for sequence in sequences:
        sequence = str(sequence).strip().upper()
        if not sequence or set(sequence).difference(AA):
            raise ValueError(f"Expected nonempty canonical sequence, got {sequence!r}")
        n = len(sequence)
        counts = Counter(sequence)
        kd = np.fromiter((ProtParamData.kd[a] for a in sequence), float)
        janin = np.fromiter((ProtParamData.ja[a] for a in sequence), float)
        fracs = np.array([counts[a] / n for a in AA])
        record = {
            "length": n,
            "janin_mean": float(janin.mean()),
            "kd_mean": float(kd.mean()),
            "kd_std": float(kd.std()),
            "composition_entropy": float(-np.sum(fracs[fracs > 0] * np.log(fracs[fracs > 0]))),
            "unique_residues": len(counts),
            "approx_charge_pH7": (
                1 / (1 + 10 ** (7 - 9.69)) - 1 / (1 + 10 ** (2.34 - 7))
                + counts["K"] / (1 + 10 ** (7 - 10.5))
                + counts["R"] / (1 + 10 ** (7 - 12.4))
                + counts["H"] / (1 + 10 ** (7 - 6.0))
                - counts["D"] / (1 + 10 ** (3.86 - 7))
                - counts["E"] / (1 + 10 ** (4.25 - 7))
                - counts["C"] / (1 + 10 ** (8.33 - 7))
                - counts["Y"] / (1 + 10 ** (10.07 - 7))
            ),
        }
        for aa in AA:
            record[f"count_{aa}"] = counts[aa]
        for name, subsequence in [
            ("global", sequence), ("n5", sequence[:5]),
            ("c5", sequence[-5:]), ("c10", sequence[-10:]),
            ("c20", sequence[-20:]),
        ]:
            subcount = Counter(subsequence)
            for aa in AA:
                record[f"{name}_frac_{aa}"] = subcount[aa] / len(subsequence)
            record[f"{name}_kd_mean"] = np.mean([ProtParamData.kd[a] for a in subsequence])
        for aa in AA:
            record[f"nterm_{aa}"] = int(sequence[0] == aa)
            record[f"cterm_{aa}"] = int(sequence[-1] == aa)
        for name, group in AA_GROUPS.items():
            record[f"{name}_fraction"] = sum(counts[a] for a in group) / n
            record[f"{name}_longest_run"] = _longest_run(sequence, group)
        for width in [3, 5, 7, 10]:
            size = min(width, n)
            rolling = np.convolve(kd, np.ones(size) / size, mode="valid")
            record[f"kd_window{width}_max"] = rolling.max()
            record[f"kd_window{width}_min"] = rolling.min()
        for motif in ["DP", "DG", "DS", "DT", "NG", "NS", "PP", "CC"]:
            record[f"motif_{motif}_fraction"] = sum(
                sequence[i:i + 2] == motif for i in range(n - 1)
            ) / max(n - 1, 1)
        if include_dipeptides:
            pairs = Counter(sequence[i:i + 2] for i in range(n - 1))
            for a in AA:
                for b in AA:
                    record[f"dipeptide_{a}{b}"] = pairs[a + b] / max(n - 1, 1)
        rows.append(record)
    return pd.DataFrame.from_records(rows).astype(np.float32)


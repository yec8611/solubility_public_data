"""Five distinct holdouts and training-only OOF threshold selection."""
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                            matthews_corrcoef, brier_score_loss, log_loss)
from .data import SEEDS


def make_splits(records, mode):
    y = np.array([r['insoluble'] for r in records]); groups = np.array([r['group_id'] for r in records])
    result = {}
    for seed in SEEDS:
        if mode == 'random':
            tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
            cv = StratifiedKFold(3, shuffle=True, random_state=seed+1000)
            folds_iter = cv.split(tr, y[tr])
        elif mode == 'grouped':
            tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups))
            folds_iter = StratifiedGroupKFold(3, shuffle=True, random_state=seed+1000).split(tr, y[tr], groups[tr])
        else:
            raise ValueError(mode)
        folds = np.full(len(y), -1, int)
        for k, (_, valid) in enumerate(folds_iter):
            folds[tr[valid]] = k
        result[str(seed)] = {'train': tr.tolist(), 'test': te.tolist(), 'folds': folds.tolist()}
    validate_splits(records, result, mode)
    return result


def validate_splits(records, splits, mode):
    n = len(records); y = np.array([r['insoluble'] for r in records]); g = np.array([r['group_id'] for r in records]); hashes = []
    if set(splits) != set(map(str, SEEDS)):
        raise ValueError('Exactly five prescribed seeds required')
    for s in splits.values():
        tr, te, f = np.array(s['train']), np.array(s['test']), np.array(s['folds'])
        if f.shape != (n,) or len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) & set(te) or set(tr) | set(te) != set(range(n)):
            raise ValueError('Invalid split partition')
        if set(f[tr]) != {0, 1, 2} or not np.all(f[te] == -1):
            raise ValueError('Invalid OOF assignments')
        for a, b in [(tr, te)]+[(tr[f[tr] != k], tr[f[tr] == k]) for k in range(3)]:
            if len(np.unique(y[a])) != 2 or len(np.unique(y[b])) != 2:
                raise ValueError('Every fitting/evaluation partition needs both classes')
            if mode == 'grouped' and set(g[a]) & set(g[b]):
                raise ValueError('Family leakage')
        hashes.append(hashlib.sha256(np.sort(te).astype('<i8').tobytes()).hexdigest())
    if len(set(hashes)) != 5:
        raise ValueError('Holdout memberships must differ across all five seeds')


def choose_threshold(y, p, recall=.9):
    """Largest score cutoff attaining requested insoluble recall on training OOF."""
    positive = np.sort(np.asarray(p)[np.asarray(y) == 1])
    if not len(positive) or not 0 < recall <= 1:
        raise ValueError('Positive labels and valid recall required')
    return float(positive[max(0, len(positive)-int(np.ceil(recall*len(positive))))])


def metrics(y, p, threshold=.5):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if not len(y) or not np.isfinite(p).all():
        raise ValueError('Empty/nonfinite predictions')
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    divide = lambda a, b: float(a/b) if b else None
    both = len(np.unique(y)) == 2
    return dict(n=len(y), n_insoluble=int(y.sum()), prevalence=float(y.mean()),
        auroc=float(roc_auc_score(y, p)) if both else None,
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
        threshold=float(threshold), precision=divide(tp, tp+fp), recall=divide(tp, tp+fn),
        specificity=divide(tn, tn+fp), mcc=float(matthews_corrcoef(y, pred)),
        true_positive=int(tp), false_positive=int(fp), true_negative=int(tn), false_negative=int(fn),
        rejection_fraction=float(pred.mean()), soluble_retention=divide(tn, tn+fp),
        accepted_insoluble_fraction=divide(fn, tn+fn), accepted_n=int(tn+fn))

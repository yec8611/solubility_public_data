"""Group-disjoint inner stopping followed by a freshly initialized full refit."""
import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from .models import FusionModel, symmetric_kl

DEFAULTS = dict(max_epochs=60, patience=8, batch_size=128, lr=.001,
                weight_decay=.01, gradient_clip=5., rdrop_weight=.5,
                min_delta=1e-5, validation_candidates=100, num_threads=2)


def seed_all(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def inner_split(y, groups, seed, candidates=100):
    choices = []
    for tr, va in GroupShuffleSplit(n_splits=candidates, test_size=.2, random_state=seed+101).split(y, y, groups):
        if len(np.unique(y[tr])) == len(np.unique(y[va])) == 2:
            distance = abs(y[va].mean()-y.mean()) + abs(len(va)/len(y)-.2)
            choices.append((distance, tr, va))
    if not choices:
        raise ValueError('Too few class-supported groups for inner validation')
    _, tr, va = min(choices, key=lambda x: x[0])
    return tr, va


class Estimator:
    def __init__(self, config, store, seed=11, device='cpu'):
        self.config = {**DEFAULTS, **config}
        self.store, self.seed, self.device = store, int(seed), device
        self.need_tokens = self.config['architecture'] == 'bilstm'
        if self.config['profile'] == 'structure':
            self.config['structure_dims'] = store.structure_dims

    def network(self):
        return FusionModel(self.config['architecture'], self.config['profile'], self.config.get('structure_dims')).to(self.device)

    def _predict(self, model, data, ix, scalers):
        model.eval(); chunks = []
        with torch.inference_mode():
            for start in range(0, len(ix), 64):
                batch = self.store.batch(data, ix[start:start+64], scalers, self.device, self.need_tokens)
                chunks.append(torch.sigmoid(model(batch)['logit']).cpu().numpy())
        return np.concatenate(chunks).astype(float) if chunks else np.array([])

    def _run(self, data, y, ix, scalers, epochs, validation=None):
        c = self.config; seed_all(self.seed, c['num_threads'])
        model = self.network()
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        w = float(np.sqrt((len(ix)-y[ix].sum())/y[ix].sum()))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=self.device))
        best, best_epoch, stale, history = -1., 1, 0, []
        for epoch in range(1, int(epochs)+1):
            model.train(); order = ix[torch.randperm(len(ix)).numpy()]; total = 0.
            for start in range(0, len(order), c['batch_size']):
                rows = order[start:start+c['batch_size']]
                batch = self.store.batch(data, rows, scalers, self.device, self.need_tokens)
                labels = torch.as_tensor(y[rows], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                # Compact structure profile shares its modality mask across R-Drop passes.
                # Public profiles preserve independently redrawn historical view masks.
                keep = model.sample_keep(batch) if c['profile'] == 'structure' else None
                a = model(batch, keep)['logit']; loss = loss_fn(a, labels)
                if c['rdrop_weight']:
                    b = model(batch, keep)['logit']
                    loss = .5*(loss+loss_fn(b, labels)) + c['rdrop_weight']*symmetric_kl(a, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip']); optimizer.step()
                total += float(loss.detach().cpu())*len(rows)
            row = {'epoch': epoch, 'training_loss': total/len(ix)}
            if validation is not None:
                ap = float(average_precision_score(y[validation], self._predict(model, data, validation, scalers)))
                row['inner_insoluble_ap'] = ap
                if ap > best+c['min_delta']:
                    best, best_epoch, stale = ap, epoch, 0
                else:
                    stale += 1
            history.append(row)
            if validation is not None and stale >= c['patience']:
                break
        return model, best_epoch if validation is not None else int(epochs), history

    def fit(self, data, y, groups):
        y = np.asarray(y, np.float32); groups = np.asarray(groups).astype(str)
        if y.shape != groups.shape or set(np.unique(y)) != {0., 1.}:
            raise ValueError('Aligned groups and both classes required')
        tr, va = inner_split(y, groups, self.seed, self.config['validation_candidates'])
        inner_scalers = self.store.scalers(data, tr)
        _, epochs, history = self._run(data, y, tr, inner_scalers, self.config['max_epochs'], va)
        self.scalers = self.store.scalers(data, np.arange(len(y)))
        self.model, _, refit = self._run(data, y, np.arange(len(y)), self.scalers, epochs)
        self.model.eval()
        self.metadata = dict(n_training=len(y), n_insoluble=int(y.sum()), n_soluble=int((1-y).sum()),
            n_groups=len(set(groups)), seed=self.seed, selected_epochs=epochs,
            positive_weight=float(np.sqrt((len(y)-y.sum())/y.sum())),
            trainable_parameters=sum(p.numel() for p in self.model.parameters()),
            inner_train=tr.tolist(), inner_validation=va.tolist(),
            inner_group_overlap=len(set(groups[tr]) & set(groups[va])),
            inner_history=history, refit_history=refit, reinitialized_before_refit=True,
            positive_class='insoluble', scores_calibrated=False)
        return self

    def predict(self, data):
        return self._predict(self.model, data, np.arange(len(data['rows'])), self.scalers)

    def save(self, path, threshold=.5):
        torch.save(dict(format='mini_solubility_v1', config=self.config, seed=self.seed,
            state={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            scalers={k: {a: torch.as_tensor(v) for a, v in s.items()} for k, s in self.scalers.items()},
            metadata=self.metadata, threshold=float(threshold),
            structure_schema=getattr(self.store, 'structure_schema', None)), path)

    @classmethod
    def load(cls, path, store, device='cpu'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['format'] != 'mini_solubility_v1':
            raise ValueError('Unknown checkpoint format')
        if saved['structure_schema'] != getattr(store, 'structure_schema', None):
            raise ValueError('Structure feature names/order mismatch')
        obj = cls(saved['config'], store, saved['seed'], device)
        obj.model = obj.network(); obj.model.load_state_dict(saved['state']); obj.model.eval()
        obj.scalers = {k: {a: v.numpy() for a, v in s.items()} for k, s in saved['scalers'].items()}
        obj.metadata, obj.threshold = saved['metadata'], saved['threshold']
        return obj

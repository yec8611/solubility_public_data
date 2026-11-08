"""Train/evaluate/deploy without selecting settings on external labels."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .data import FeatureStore, read_records, subset, SEEDS
from .training import Estimator
from .evaluation import make_splits, validate_splits, choose_threshold, metrics


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def setup(args, training=True):
    records = read_records(args.records, training=training)
    store = FeatureStore(args.features, getattr(args, 'structure', None))
    config = json.loads(Path(args.config).read_text()) if getattr(args, 'config', None) else None
    return records, store, config


def train(args):
    records, store, config = setup(args)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory for each run')
    out.mkdir(parents=True, exist_ok=True)
    date = args.date
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    data = store.align(records, config['architecture'] == 'bilstm', config['profile'] == 'structure')
    y = np.array([r['insoluble'] for r in records], np.float32)
    groups = np.array([r['group_id'] for r in records])
    external = []
    for path in args.external:
        rr = read_records(path)
        if any(r.get('role') != 'external' for r in rr):
            raise ValueError('External panels require role=external')
        if {r['sequence'] for r in rr} & {r['sequence'] for r in records} or {r['group_id'] for r in rr} & set(groups):
            raise ValueError('External exact-sequence/family overlap with training')
        external.append((rr, store.align(rr, config['architecture'] == 'bilstm', config['profile'] == 'structure')))
    frozen = dict(created=date, config=config, seeds=SEEDS, n=len(y), n_insoluble=int(y.sum()), n_groups=len(set(groups)),
        positive_class='insoluble', epochs_capped_for_smoke=args.max_epochs is not None,
        records_sha256=hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
        threshold_policy='training OOF insoluble recall >= '+str(args.recall),
        external_used_for_selection=False)
    write_json(out/f'protocol_{date}.json', frozen)
    rows, predictions, fits, manifest = [], [], [], {}
    for mode in args.modes:
        if args.splits:
            s = json.loads(Path(args.splits).read_text())
            if s['record_ids'] != [r['record_id'] for r in records]:
                raise ValueError('Split record ID order mismatch')
            splits = s[mode]
        else:
            splits = make_splits(records, mode)
        validate_splits(records, splits, mode); manifest[mode] = splits
        for seed in SEEDS:
            spec = splits[str(seed)]; tr = np.array(spec['train']); te = np.array(spec['test']); folds = np.array(spec['folds'])
            p = np.full(len(y), np.nan)
            for fold in range(4):
                fitting = tr[folds[tr] != fold] if fold < 3 else tr
                testing = tr[folds[tr] == fold] if fold < 3 else te
                estimator = Estimator(config, store, seed+fold*10000, args.device).fit(subset(data, fitting), y[fitting], groups[fitting])
                p[testing] = estimator.predict(subset(data, testing))
                fits.append(dict(mode=mode, outer_seed=seed, fold=fold, fitting_ids=[records[i]['record_id'] for i in fitting], **estimator.metadata))
            threshold = choose_threshold(y[tr], p[tr], args.recall)
            estimator.save(out/f'{mode}_seed{seed}_{date}.pt', threshold)
            for split, ix in [('oof', tr), ('holdout', te)]:
                for op, cutoff in [('default', .5), ('oof_recall', threshold)]:
                    rows.append(dict(mode=mode, seed=seed, dataset=records[0].get('dataset', 'internal'), split=split, operating_point=op, **metrics(y[ix], p[ix], cutoff)))
                predictions.extend(dict(mode=mode, seed=seed, split=split, record_id=records[i]['record_id'], insoluble=int(y[i]), score=float(p[i])) for i in ix)
            for rr, xx in external:
                pp = estimator.predict(xx)
                valid = np.array([r['insoluble'] is not None for r in rr]); yy = np.array([r['insoluble'] or 0 for r in rr])
                for op, cutoff in [('default', .5), ('oof_recall', threshold)]:
                    rows.append(dict(mode=mode, seed=seed, dataset=rr[0]['dataset'], split='external', operating_point=op, **metrics(yy[valid], pp[valid], cutoff)))
                quantitative = [i for i,r in enumerate(rr) if r.get('solubility_fraction') is not None]
                if len(quantitative) > 2:
                    rho = float(spearmanr([-rr[i]['solubility_fraction'] for i in quantitative], pp[quantitative]).statistic)
                    rows.append(dict(mode=mode, seed=seed, dataset=rr[0]['dataset'], split='external_continuous', operating_point='ranking', n=len(quantitative), spearman=rho if np.isfinite(rho) else None))
                predictions.extend(dict(mode=mode, seed=seed, split='external', record_id=r['record_id'], insoluble=r['insoluble'], score=float(v)) for r,v in zip(rr,pp))
            print(f'{mode} seed={seed}: OOF AP={metrics(y[tr],p[tr])["average_precision"]:.3f}; holdout AP={metrics(y[te],p[te])["average_precision"]:.3f}', flush=True)
            write_json(out/f'metrics_{date}.json', rows)
            write_json(out/f'predictions_{date}.json', predictions)
            write_json(out/f'fits_{date}.json', fits)
    write_json(out/f'splits_{date}.json', {'record_ids':[r['record_id'] for r in records], **manifest})
    from .reporting import report
    report(out, date, export_xlsx=not args.json_only)


def fit_final(args):
    """One deployment model; threshold from full-training grouped OOF predictions."""
    records, store, config = setup(args)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    data = store.align(records, config['architecture']=='bilstm', config['profile']=='structure')
    y = np.array([r['insoluble'] for r in records], np.float32); groups = np.array([r['group_id'] for r in records])
    from sklearn.model_selection import StratifiedGroupKFold
    p = np.full(len(y), np.nan); fits = []
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=args.seed+1000).split(y,y,groups)):
        est = Estimator(config, store, args.seed+fold*10000, args.device).fit(subset(data,tr), y[tr], groups[tr])
        p[va] = est.predict(subset(data,va)); fits.append(est.metadata)
    threshold = choose_threshold(y,p,args.recall)
    est = Estimator(config, store, args.seed, args.device).fit(data,y,groups)
    est.save(out/f'deployment_{args.date}.pt', threshold)
    write_json(out/f'deployment_fit_{args.date}.json', dict(oof_metrics=metrics(y,p,threshold), threshold_selection='full training grouped OOF, apparent OOF threshold metrics', fits=fits, final_fit=est.metadata, independent_holdout_metrics='Run evaluate separately; no holdout created by fit-final'))
    prediction_rows=[dict(record_id=r['record_id'],insoluble=int(label),oof_score=float(score)) for r,label,score in zip(records,y,p)]
    write_json(out/f'deployment_oof_{args.date}.json',prediction_rows)
    if not args.json_only:
        from .reporting import export_tables
        export_tables(out/f'deployment_oof_{args.date}.xlsx', {'OOF_predictions':prediction_rows,'OOF_metrics':[metrics(y,p,threshold)]})


def predict(args):
    records = read_records(args.records)
    store = FeatureStore(args.features, args.structure)
    estimator = Estimator.load(args.checkpoint, store, args.device)
    x = store.align(records, estimator.need_tokens, estimator.config['profile']=='structure')
    p = estimator.predict(x)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = [dict(record_id=r['record_id'], sequence=r['sequence'], insolubility_score=float(v), threshold=estimator.threshold, reject_before_synthesis=bool(v>=estimator.threshold)) for r,v in zip(records,p)]
    write_json(out/f'predictions_{args.date}.json',rows)
    if not args.json_only:
        from .reporting import export_tables
        export_tables(out/f'predictions_{args.date}.xlsx', {'Predictions':rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    for name in ['evaluate','fit-final','predict']:
        p = sub.add_parser(name); p.add_argument('--records',required=True);p.add_argument('--features',required=True)
        p.add_argument('--structure'); p.add_argument('--output',required=True);p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
        p.add_argument('--date',default=datetime.now().strftime('%Y%m%d'));p.add_argument('--json-only',action='store_true')
        if name != 'predict':
            p.add_argument('--config',required=True);p.add_argument('--recall',type=float,default=.9)
        else:
            p.add_argument('--checkpoint',required=True)
    p = sub.choices['evaluate'];p.add_argument('--splits');p.add_argument('--modes',nargs='+',choices=['random','grouped'],default=['random','grouped']);p.add_argument('--external',nargs='*',default=[]);p.add_argument('--max-epochs',type=int)
    sub.choices['fit-final'].add_argument('--seed',type=int,default=11)
    args=parser.parse_args()
    {'evaluate':train,'fit-final':fit_final,'predict':predict}[args.command](args)


if __name__ == '__main__':
    main()

"""Validate release records, exact frozen splits, class polarity and XLSX parity."""
import hashlib,json
from pathlib import Path
from solubility_public_data.data import read_records
from solubility_public_data.evaluation import validate_splits


def main():
    root=Path(__file__).resolve().parents[1];allrows={};summary=[]
    for name,n,positive in [('psi',1686,368),('garcia',284,44),('nesg',152,32),('esol',198,24)]:
        path=root/f'data/public/{name}_records_20261108.json';rows=read_records(path,name=='psi')
        assert len(rows)==n and sum(r['insoluble']==1 for r in rows)==positive
        xlsx=read_records(path.with_suffix('.xlsx'),name=='psi')
        # Pandas can represent empty numeric fields as NaN; normalize first.
        for r in xlsx:
            if r.get('solubility_fraction')!=r.get('solubility_fraction'):r['solubility_fraction']=None
        assert rows==xlsx, f'XLSX/JSON mismatch: {name}'
        allrows[name]=rows;summary.append(dict(dataset=name,n=n,insoluble=positive,json_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    psi=allrows['psi'];s=json.loads((root/'data/splits/psi_five_seed_splits_20261108.json').read_text())
    assert s['record_ids']==[r['record_id'] for r in psi]
    for mode in ['random','grouped']:validate_splits(psi,s[mode],mode)
    for name in ['garcia','nesg','esol']:
        assert not {r['sequence'] for r in psi}&{r['sequence'] for r in allrows[name]}
        assert not {r['group_id'] for r in psi}&{r['group_id'] for r in allrows[name]}
    print(json.dumps(dict(status='passed',rows=summary,distinct_seeds=5,oof_folds=3,external_training_family_overlap=0),indent=2))


if __name__=='__main__':main()

"""Local-only importer for the previously curated 17,638 independent weak rows.

Input is the old weak_20261106.npz. Nothing is downloaded, published or pushed.
Review the upstream terms before sharing inputs or derivative artifacts.
"""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='private_data/weak_records_20261108.json');a=p.parse_args()
    z=np.load(a.input,allow_pickle=False)
    if len(z['ids'])!=17638 or int(z['y'].sum())!=6042:raise ValueError('Unexpected curated weak pool')
    records=[dict(record_id=str(i),sequence=str(s),insoluble=int(y),group_id=str(g),dataset='independent_weak',role='train',label_quality='weak mixed endpoint; review terms before redistribution') for i,s,y,g in zip(z['ids'],z['sequences'],z['y'],z['groups'])]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(records,indent=2)+'\n')
    from solubility_public_data.reporting import export_tables
    export_tables(out.with_suffix('.xlsx'),{'Records':records})
    print(f'Imported {len(records)} rows locally. Use extract_features.py; keep private_data ignored.')


if __name__=='__main__':main()

"""Maintainer export from the earlier research directory; no new model fitting.

Usage: python scripts/export_research_archive.py --archive PATH --feature-output PATH
The archive is not distributed. Public data, checksums and split membership are.
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--feature-output',type=Path,required=True);args=p.parse_args()
    src=args.archive;root=Path(__file__).resolve().parents[1];dest=args.feature_output;dest.mkdir(parents=True,exist_ok=True)
    master=json.loads((src/'data/processed/universal_benchmark_export_20261106.json').read_text())['records']
    lookup={r['benchmark_record_id']:r for r in master}
    token_index=np.load(src/'sequence_architectures_20261107/data/residue_index_20261107.npz',allow_pickle=False)
    token_values=np.load(src/'sequence_architectures_20261107/data/esm650_residues_20261107.npy',mmap_mode='r',allow_pickle=False)
    token_lookup={str(s):i for i,s in enumerate(token_index['ids'])}
    all_records=[];blocks=[];chunks=[];summary=[];provenance=[]
    for name,oldname in [('psi','psi'),('garcia','external_garcia'),('nesg','external_nesg'),('esol',None)]:
        if name=='esol':
            rows=json.loads((src/'additional_evaluation_20261107/data/esol_evaluation_records_20261107.json').read_text())
            features=np.load(src/'additional_evaluation_20261107/data/esol_features_20261107.npz',allow_pickle=False)
            index=np.load(src/'additional_evaluation_20261107/data/esol_residue_index_20261107.npz',allow_pickle=False)
            values=np.load(src/'additional_evaluation_20261107/data/esol_esm650_residues_20261107.npy',mmap_mode='r',allow_pickle=False)
            ilook={str(s):i for i,s in enumerate(index['ids'])}
        else:
            features=np.load(src/f'esm650_pooling_20261106/data/{oldname}_20261106.npz',allow_pickle=False)
            rows=[lookup[str(i)] for i in features['ids']];index=token_index;values=token_values;ilook=token_lookup
        records=[]
        for i,(raw,identifier,seq) in enumerate(zip(rows,features['ids'],features['sequences'])):
            identifier,seq=str(identifier),str(seq)
            label=raw['binary_insoluble'] if name=='esol' else int(features['y'][i])
            record=dict(record_id=identifier,sequence=seq,length_aa=len(seq),insoluble=label,
                group_id=raw['family40_group'] if name=='esol' else str(features['groups'][i]),
                dataset=name,role='train' if name=='psi' else 'external',
                solubility_fraction=raw.get('solubility_fraction'),source_record_id=raw['source_record_id'],
                source_publication=raw['source_publication'],source_url=raw['source_urls'],
                endpoint=raw['source_label_endpoint'],antibody_screen=raw['antibody_screen'],
                assay_construct_caveat=raw.get('assay_construct_caveat','Canonical/core sequence; assay context is incomplete'),
                prior_exposure='Previously analyzed; retrospective panel, not a new blinded test',creation_date='2026-11-08')
            assert 20<=len(seq)<=100 and (name=='esol' or label==1-int(raw['binary_soluble']))
            records.append(record)
            ri=ilook[identifier];lo,hi=index['offsets'][ri:ri+2]
            assert np.array_equal(index['sequences'][ri],seq) and hi-lo==len(seq)
            chunks.append((values,int(lo),int(hi)))
        file=root/f'data/public/{name}_records_20261108.json';file.write_text(json.dumps(records,indent=2)+'\n')
        all_records.extend(records);blocks.append({k:features[k] for k in ['ids','sequences','desc','esm650','esm650_max']})
        summary.append(dict(dataset=name,n=len(records),insoluble=sum(r['insoluble']==1 for r in records),soluble=sum(r['insoluble']==0 for r in records),unclassified=sum(r['insoluble'] is None for r in records),groups=len({r['group_id'] for r in records}),role=records[0]['role']))
        provenance.append(dict(dataset=name,json_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),source_url=records[0]['source_url']))
    pooled={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    pooled['offsets']=np.concatenate([[0],np.cumsum([hi-lo for _,lo,hi in chunks])]).astype(np.int64)
    np.savez_compressed(dest/'pooled_20261108.npz',**pooled)
    values=np.lib.format.open_memmap(dest/'residues_20261108.npy',mode='w+',dtype=np.float16,shape=(int(pooled['offsets'][-1]),1280))
    for i,(a,lo,hi) in enumerate(chunks):values[pooled['offsets'][i]:pooled['offsets'][i+1]]=a[lo:hi]
    values.flush();del values
    old=json.loads((src/'esm650_pooling_20261106/data/splits_20261106.json').read_text())
    split={'record_ids':blocks[0]['ids'].tolist(),'creation_date':'2026-11-08','positive_class':'insoluble',**old['psi']}
    (root/'data/splits/psi_five_seed_splits_20261108.json').write_text(json.dumps(split,indent=2)+'\n')
    (root/'data/manifests/dataset_summary_20261108.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'data/manifests/provenance_20261108.json').write_text(json.dumps(provenance,indent=2)+'\n')
    from solubility_public_data.descriptors import sequence_features
    desc=sequence_features(pooled['sequences'])
    if not np.allclose(desc.to_numpy(),pooled['desc'],atol=1e-5,rtol=1e-5):raise ValueError('Descriptor definition mismatch')
    (root/'data/manifests/descriptor_names_20261108.json').write_text(json.dumps(desc.columns.tolist(),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

"""Frozen pinned ESM2-650M embeddings; canonical residues only, no labels used."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel
from solubility_public_data.data import read_records
from solubility_public_data.descriptors import sequence_features

MODEL='facebook/esm2_t33_650M_UR50D'
REVISION='08e4846e537177426273712802403f7ba8261b6c'


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
    records=[r for path in a.records for r in read_records(path)]
    if len({r['record_id'] for r in records})!=len(records):raise ValueError('Duplicate IDs across inputs')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use an empty feature directory')
    seq=[r['sequence'] for r in records];offsets=np.r_[0,np.cumsum(list(map(len,seq)))].astype(np.int64)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=EsmModel.from_pretrained(MODEL,revision=REVISION).eval().to(a.device)
    tokens=np.lib.format.open_memmap(out/'residues_20261108.npy',mode='w+',dtype=np.float16,shape=(offsets[-1],1280))
    means=[];maxima=[]
    with torch.inference_mode():
        for start in range(0,len(seq),a.batch_size):
            ss=seq[start:start+a.batch_size]
            batch=tokenizer(ss,padding=True,return_tensors='pt',return_special_tokens_mask=True)
            special=batch.pop('special_tokens_mask').bool();valid=batch['attention_mask'].bool() & ~special
            hidden=model(**{k:v.to(a.device) for k,v in batch.items()}).last_hidden_state.cpu()
            for j,s in enumerate(ss):
                v=hidden[j][valid[j]].float().numpy();i=start+j
                if v.shape!=(len(s),1280):raise ValueError('Tokenizer residue mismatch')
                tokens[offsets[i]:offsets[i+1]]=v.astype(np.float16);means.append(v.mean(0));maxima.append(v.max(0))
            print(f'{min(start+a.batch_size,len(seq))}/{len(seq)}',flush=True)
    tokens.flush()
    np.savez_compressed(out/'pooled_20261108.npz',ids=np.array([r['record_id'] for r in records]),sequences=np.array(seq),desc=sequence_features(seq).to_numpy(),esm650=np.array(means),esm650_max=np.array(maxima),offsets=offsets)
    (out/'feature_metadata_20261108.json').write_text(json.dumps(dict(model=MODEL,revision=REVISION,labels_used=False,pooling='float32 mean/max of valid residues; token cache float16',n=len(seq)),indent=2)+'\n')


if __name__=='__main__':main()

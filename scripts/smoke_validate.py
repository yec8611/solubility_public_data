"""Two-epoch software validation on 128 real PSI proteins; not a model study."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from solubility_public_data.data import read_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',type=Path,default=Path('runs/smoke_20261108'));a=p.parse_args()
    root=Path(__file__).resolve().parents[1];records=read_records(root/'data/public/psi_records_20261108.json',True)
    rng=np.random.default_rng(90210);used=set();picked=[]
    for label in [1,0]:
        candidates=[r for r in records if r['insoluble']==label];count=0
        for i in rng.permutation(len(candidates)):
            r=candidates[i]
            if r['group_id'] in used:continue
            picked.append(r);used.add(r['group_id']);count+=1
            if count==64:break
        if count!=64:raise ValueError('Insufficient unique groups for smoke fixture')
    a.output.mkdir(parents=True,exist_ok=True);input_path=a.output/'smoke_records_20261108.json';input_path.write_text(json.dumps(picked,indent=2)+'\n')
    for architecture in ['residual_mlp','bilstm']:
        subprocess.run([sys.executable,'-m','solubility_public_data.cli','evaluate','--records',str(input_path),'--features',a.features,'--config',str(root/f'configs/{architecture}_public.json'),'--max-epochs','2','--json-only','--date','20261108','--external',*[str(root/f'data/public/{name}_records_20261108.json') for name in ['garcia','nesg','esol']],'--output',str(a.output/architecture)],check=True)


if __name__=='__main__':main()

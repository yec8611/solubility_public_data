"""Generate a runnable quickstart and an annotated, complete source notebook."""
import argparse,hashlib,json
from pathlib import Path
import nbformat as nbf


def build(root):
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    quick=nbf.v4.new_notebook(cells=[
        md('# Mini-protein insolubility — quickstart\n\nCreated 2026-11-08. Insoluble = 1. Run from the repository or notebooks folder. Install `pip install -e ".[dev,esm]"` first. Full fitting is opt-in; the default cells only inspect public data and model topology.'),
        code("from pathlib import Path\nimport sys, json\nroot = Path.cwd() if (Path.cwd() / 'pyproject.toml').exists() else Path.cwd().parent\nsys.path.insert(0, str(root / 'src'))\nfrom solubility_public_data.data import read_records\nfrom solubility_public_data.models import FusionModel\nrecords = read_records(root / 'data/public/psi_records_20261108.xlsx', training=True)\nprint({'n': len(records), 'insoluble': sum(r['insoluble'] for r in records), 'groups': len({r['group_id'] for r in records})})"),
        md('## Two public heads\n\nFrozen ESM650 mean/max inputs; BiLSTM also reads frozen residue states. The source notebook documents every module. Structure profiles require measured or predicted structure features supplied separately.'),
        code("for architecture in ['residual_mlp', 'bilstm']:\n    model = FusionModel(architecture)\n    print(architecture, sum(p.numel() for p in model.parameters()))\n    print(model)"),
        md('## Download the feature cache\n\nThis verifies archive and member hashes. It downloads a large public asset only when enabled.'),
        code("import subprocess\nDOWNLOAD_FEATURES = False\nif DOWNLOAD_FEATURES:\n    subprocess.run([sys.executable, 'scripts/download_features.py'], cwd=root, check=True)"),
        md('## Prespecified five-seed evaluation\n\nBoth random and family-grouped splits; three-fold OOF within each training partition. Full fitting may take substantial CPU time. External labels are used only for scoring.'),
        code("RUN_TRAINING = False\nif RUN_TRAINING:\n    for architecture in ['residual_mlp', 'bilstm']:\n        subprocess.run([sys.executable, '-m', 'solubility_public_data.cli', 'evaluate', '--records', 'data/public/psi_records_20261108.xlsx', '--features', 'features/public', '--config', f'configs/{architecture}_public.json', '--splits', 'data/splits/psi_five_seed_splits_20261108.json', '--external', 'data/public/garcia_records_20261108.xlsx', 'data/public/nesg_records_20261108.xlsx', 'data/public/esol_records_20261108.xlsx', '--output', f'runs/{architecture}_20261108'], cwd=root, check=True)"),
        md('## Deployment\n\nUse `solubility fit-final` after evaluating the chosen architecture. This fits one model and sets its cutoff from full-training grouped OOF scores. Use `solubility predict` for new sequences. See `training_instruction.md` for internal structure schema and threshold interpretation.')
    ])
    quick.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    files=sorted((root/'src').rglob('*.py'))+sorted((root/'scripts').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    cells=[md('# Complete source and training protocol — 2026-11-08\n\nAll Python modules, scripts and tests are merged below in readable functional sections. These are source listings, not execution cells: relative imports and command-line entry points belong to their files. Use the companion quickstart for execution. Rebuild with `python scripts/build_notebooks.py` after changing source files.')]
    hashes={}
    for path in files:
        relative=path.relative_to(root).as_posix();source=path.read_text();hashes[relative]=hashlib.sha256(source.encode()).hexdigest()
        cells.append(md(f'## {relative}\n\n```python\n{source.rstrip()}\n```'))
    source=nbf.v4.new_notebook(cells=cells);source.metadata={'source_sha256':hashes}
    return {'quickstart_20261108.ipynb':quick,'complete_source_20261108.ipynb':source}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]
    for name,book in build(root).items():
        nbf.validate(book);path=root/'notebooks'/name
        if a.check:
            old=nbf.read(path,as_version=4);nbf.validate(old)
            assert [c.source for c in old.cells]==[c.source for c in book.cells], f'Stale notebook: {name}'
        else:nbf.write(book,path)
    print('Notebook sources validated' if a.check else 'Notebooks generated')


if __name__=='__main__':main()

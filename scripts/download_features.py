"""Download the public release cache with SHA-256 and safe member validation."""
import argparse, hashlib, json, shutil, tarfile, tempfile, urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('features/public'));a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data/manifests/feature_archive_20261108.json').read_text())
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose an empty feature directory')
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'features.tar.gz';digest=hashlib.sha256()
        with urllib.request.urlopen(manifest['url']) as source,archive.open('wb') as target:
            while chunk:=source.read(1024*1024):target.write(chunk);digest.update(chunk)
        if digest.hexdigest()!=manifest['sha256']:raise ValueError('Archive checksum mismatch')
        a.output.mkdir(parents=True,exist_ok=True)
        expected=manifest['files']
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            if {m.name for m in members}!=set(expected) or len(members)!=len(expected):raise ValueError('Unexpected archive members')
            for member in members:
                if not member.isfile() or Path(member.name).name!=member.name:raise ValueError('Unsafe archive member')
                target=a.output/member.name
                with tar.extractfile(member) as inp,target.open('wb') as dest:shutil.copyfileobj(inp,dest)
                h=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
                if h!=expected[member.name]['sha256']:raise ValueError('Member checksum mismatch')
    print(f'Verified {manifest["n"]} proteins in {a.output}')


if __name__=='__main__':main()

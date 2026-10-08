#!/usr/bin/env python3
"""Create the offline handoff, excluding Git, environments and run outputs."""
import argparse,hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SKIP={'.git','.venv','__pycache__','runtime','runs'}
def included(path):
    relative=path.relative_to(ROOT)
    # Resources are the curated dependency package. Its payload includes named
    # runs that are actual input resources; do not exclude those recursively.
    if relative.parts[0]=='resources':
        return not any(p in {'.git','.venv','__pycache__'} for p in relative.parts)
    return not any(p in SKIP for p in relative.parts)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT==out.parent or ROOT in out.parents:raise ValueError('Archive must be new and outside repository')
    assert (ROOT/'resources/manifest.json').is_file(),'Missing resource manifest'
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and included(p)]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as raw:
        import gzip
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as archive:
                for path in files:archive.add(path,arcname=str(Path(ROOT.name)/path.relative_to(ROOT)),recursive=False)
    h=hashlib.sha256()
    with out.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    out.with_suffix(out.suffix+'.sha256').write_text(h.hexdigest()+'  '+out.name+'\n')
    print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':h.hexdigest(),'files':len(files)},indent=2))
if __name__=='__main__':main()

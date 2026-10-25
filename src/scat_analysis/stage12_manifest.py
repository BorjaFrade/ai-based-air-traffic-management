"""SHA-256 inventory of Stage 12 code and generated outputs."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
from stage12_audit_inputs import OUT

ROOT=Path(__file__).resolve().parents[2]
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def main():
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.csv']
    files+=sorted((ROOT/'src/scat_analysis').glob('stage12_*.py'))
    files+=sorted((ROOT/'src/scat_analysis/tests').glob('test_stage12_*.py'))
    rows=[]
    for p in sorted(set(files)):
        if p.is_relative_to(OUT):name=str(p.relative_to(OUT));kind='artifact';section=p.relative_to(OUT).parts[0]
        else:name=str(p.relative_to(ROOT));kind='source_code';section='src'
        rows.append(dict(path=name,kind=kind,section=section,bytes=p.stat().st_size,sha256=sha256(p)))
    dest=OUT/'manifest.csv';pd.DataFrame(rows).to_csv(dest,index=False)
    print(dest,'files',len(rows),'total_bytes',sum(x['bytes'] for x in rows))

if __name__=='__main__':main()

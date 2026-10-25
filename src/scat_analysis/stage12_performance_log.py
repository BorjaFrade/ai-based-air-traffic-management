"""Summarize recorded wall time, peak RSS and output volume for pair discovery."""
from __future__ import annotations
import json
import pandas as pd
from stage12_audit_inputs import OUT,WEEKS


def main():
    rows=[]
    for w in WEEKS:
        a=json.loads((OUT/'diagnostics'/f'{w}_pair_build_audit.json').read_text())
        p=OUT/'tables'/f'{w}_candidate_pairs_150nm.parquet'
        rows.append(dict(week=w,stage='native_snapshot_and_pair_candidate_build',
            elapsed_seconds=a['elapsed_seconds'],peak_rss_mb=a.get('peak_rss_mb'),
            native_rows=a['native_rows'],candidate_rows=a['candidate_rows_within_nm']['150'],
            output_bytes=p.stat().st_size))
    dest=OUT/'diagnostics'/'performance_log.csv';pd.DataFrame(rows).to_csv(dest,index=False)
    print(dest)

if __name__=='__main__':main()

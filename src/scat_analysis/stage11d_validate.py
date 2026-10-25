"""Scientific invariants, artifact links, and manifest for Stage 11D."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stage11d_core import OUT, PILOT, REPLICATION, CRITERIA, PARAMS


def validate():
    checks=[]
    audits={w:json.loads((OUT/"diagnostics"/f"{w}_coverage.json").read_text()) for w in (PILOT,REPLICATION)}
    assert audits[PILOT]["parameters"]==audits[REPLICATION]["parameters"]==PARAMS
    checks.append("parameter parity")
    for week in (PILOT,REPLICATION):
        p=OUT/"tables"/f"{week}_criterion_snapshots.parquet"
        f=pd.read_parquet(p,columns=["tick","flight_key","criterion","neighbour_key",
            "horizontal_nm","focal_source_epoch","neighbour_source_epoch","event_count","exposure_hours"])
        assert len(f)==5*audits[week]["risk_ticks"]
        assert not f.duplicated(["tick","flight_key","criterion"]).any()
        assert set(f.criterion)==set(CRITERIA)
        assert (f.focal_source_epoch.dropna()<=f.loc[f.focal_source_epoch.notna(),"tick"]+1e-6).all()
        assert (f.neighbour_source_epoch.dropna()<=f.loc[f.neighbour_source_epoch.notna(),"tick"]+1e-6).all()
        assert f.loc[f.neighbour_key.notna(),"horizontal_nm"].le(100+1e-5).all()
        assert not f.loc[f.neighbour_key.notna(),"flight_key"].eq(f.loc[f.neighbour_key.notna(),"neighbour_key"]).any()
        assert f.groupby("criterion").event_count.sum().eq(audits[week]["eligible_events"]).all()
        assert np.allclose(f.groupby("criterion").exposure_hours.sum(),audits[week]["focal_hours"])
        reference=f.loc[f.criterion.eq(CRITERIA[0]),["tick","flight_key"]].sort_values(["tick","flight_key"]).reset_index(drop=True)
        for criterion in CRITERIA[1:]:
            other=f.loc[f.criterion.eq(criterion),["tick","flight_key"]].sort_values(["tick","flight_key"]).reset_index(drop=True)
            assert reference.equals(other)
        checks.append(f"{week}: focal parity, pair identity, pre-event times, units, counts")
    report=OUT/"reports/STAGE11D_VALIDATION_REPORT.md"
    content=report.read_text(encoding="utf-8")
    for link in re.findall(r"!\[[^]]*\]\(([^)]+)\)",content):
        assert (report.parent/link).resolve().exists(),link
    pngs=sorted((OUT/"figures").rglob("*.png"))
    assert len(pngs)==22,len(pngs)
    assert len(re.findall(r"!\[Figura \d+",content))==21
    assert (OUT/"figures/stage11d_five_panel_dashboard.svg").exists()
    checks.append("21 detailed figures, dashboard PNG/SVG and report links")
    pdf=OUT/"reports/STAGE11D_VALIDATION_REPORT.pdf"
    if pdf.exists():
        raw=pdf.read_bytes()
        pages=len(re.findall(rb"/Type\s*/Page\b",raw))
        assert raw.startswith(b"%PDF-") and b"%%EOF" in raw[-1024:] and pages>=10
        checks.append(f"PDF structure: {pages} pages (rendered separately for visual QA)")
    pd.DataFrame({"check":checks,"result":"PASS"}).to_csv(OUT/"diagnostics/validation_checks.csv",index=False)
    scripts=(sorted(Path(__file__).parent.glob("stage11d_*.py"))+
             [Path(__file__).parent/"stage11_core.py",Path(__file__).parent/"stage11_episodes.py",
              Path(__file__).parent/"tests/test_stage11d_validation.py"])
    pd.DataFrame([dict(path=str(p.relative_to(Path(__file__).resolve().parents[2])),
          sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in scripts]).to_csv(
          OUT/"diagnostics/source_code_sha256.csv",index=False)
    rows=[]
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name=="manifest.csv":continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=digest))
    pd.DataFrame(rows).to_csv(OUT/"manifest.csv",index=False)
    print(json.dumps({"result":"PASS","checks":checks,"files":len(rows)},ensure_ascii=False,indent=2))


if __name__=="__main__":validate()

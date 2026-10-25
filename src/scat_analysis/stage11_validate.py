"""End-to-end Stage 11 artifact and numerical consistency gate."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from stage11_core import OUT


def main():
    a=pd.read_parquet(OUT/"tables/11a_anchor_pairs.parquet")
    trace=pd.read_parquet(OUT/"tables/11a_fixed_pair_timeline.parquet")
    risk=pd.read_parquet(OUT/"tables/11b_population_risk_set.parquet")
    event=pd.read_parquet(OUT/"tables/11b_eligible_events.parquet")
    rate=pd.read_csv(OUT/"tables/11b_rate_map_h2_v250.csv")
    assert a.window_id.is_unique and a.ATC_INTERVENTION.value_counts().to_dict()=={0:4098,1:4098}
    assert a.anchor_source.eq("STAGE10B_BACKWARD_ASOF_NATIVE").all()
    assert (a.time_epoch<=a.event_epoch+1e-6).all()
    assert (a.other_source_epoch.dropna()<=a.loc[a.other_source_epoch.notna(),"event_epoch"]+1e-6).all()
    assert trace.merge(a[["window_id","other_flight_key"]],on=["window_id","other_flight_key"],validate="many_to_one").shape[0]==len(trace)
    assert not trace.duplicated(["window_id","relative_bin_s"]).any()
    zero=trace.loc[trace.relative_bin_s.eq(0)].merge(a[["window_id","horizontal_separation_nm","vertical_separation_ft"]],
                                                       on="window_id",validate="one_to_one",suffixes=("_trace","_anchor"))
    assert zero.window_id.nunique()==a.other_flight_key.notna().sum()
    assert np.allclose(zero.horizontal_separation_nm_trace,zero.horizontal_separation_nm_anchor)
    assert np.allclose(zero.vertical_separation_ft_trace,zero.vertical_separation_ft_anchor)
    assert not risk.duplicated(["tick","flight_key"]).any()
    assert np.allclose(risk.exposure_hours,30/3600)
    assert (risk.focal_source_epoch<=risk.tick+1e-6).all()
    assert (risk.neighbour_source_epoch.dropna()<=risk.loc[risk.neighbour_source_epoch.notna(),"tick"]+1e-6).all()
    assert int(risk.event_count.sum())==len(event)
    assert ((event.clearance_epoch>event.tick)&(event.clearance_epoch<=event.tick+30)).all()
    mask=risk.horizontal_nm.ge(0)&risk.horizontal_nm.lt(50)&risk.vertical_ft.ge(0)&risk.vertical_ft.lt(6000)
    assert int(rate.events.sum())==int(risk.loc[mask,"event_count"].sum())
    assert np.isclose(rate.exposure_hours.sum(),risk.loc[mask,"exposure_hours"].sum())
    assert rate.supported.eq((rate.exposure_hours>=10)&(rate.events>=3)&(rate.flights>=20)).all()
    assert rate.loc[rate.supported,"ci_low_per_1000h"].le(rate.loc[rate.supported,"rate_per_1000h"]).all()
    assert rate.loc[rate.supported,"ci_high_per_1000h"].ge(rate.loc[rate.supported,"rate_per_1000h"]).all()
    for p in (OUT/"figures").rglob("*.png"):
        assert p.with_suffix(".svg").exists(),p
        assert p.with_suffix(".json").exists(),p
        if "case_studies" not in str(p):
            assert p.with_suffix(".csv").exists(),p
        with Image.open(p) as im:
            assert im.width>=1000 and im.height>=600,p
            dpi=im.info.get("dpi")
            assert dpi and min(dpi)>=290,p
    report=OUT/"reports/STAGE11_INTERVENTION_GEOMETRY_REPORT.md"
    markdown=report.read_text(encoding="utf-8")
    paths=re.findall(r"!\[[^]]*\]\(([^)]+)\)",markdown)
    assert paths and all((report.parent/p).exists() for p in paths)
    assert len(re.findall(r"^## \d+\.",markdown,re.M))==16
    pdf=report.with_suffix(".pdf")
    assert pdf.exists() and pdf.stat().st_size>100_000
    result={"matched_windows":len(a),"matched_pairs_with_neighbour":int(a.other_flight_key.notna().sum()),
            "population_snapshots":len(risk),"population_hours":float(risk.exposure_hours.sum()),
            "eligible_events":len(event),"map_events":int(rate.events.sum()),"map_hours":float(rate.exposure_hours.sum()),
            "supported_map_cells":int(rate.supported.sum()),"figures_png":len(list((OUT/"figures").rglob("*.png"))),
            "report_images":len(paths),"result":"PASS"}
    path=OUT/"diagnostics/11_final_validation.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="manifest.csv")
    result["manifest_items"]=len(files)
    path.write_text(json.dumps(result,indent=2)+"\n")
    manifest=pd.DataFrame([{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"kind":p.suffix.lstrip(".")}
                           for p in files])
    manifest.to_csv(OUT/"manifest.csv",index=False)
    assert set(manifest.path)=={str(p.relative_to(OUT)) for p in files}
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

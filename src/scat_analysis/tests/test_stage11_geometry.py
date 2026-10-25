"""Stage 11 denominator, timing, pair and artifact invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import stage11_core as c
import stage11_visualize as v
from stage11_episodes import observed_segments


def test_pre_event_interval_and_asof_edges():
    event=np.array([0,0.001,29.999,30,30.001,60])
    assert c.event_tick(event,30).tolist()==[-30,0,0,0,30,30]
    native=np.array([0,14.999,15,15.001,30])
    ticks=c.snapshot_tick(native,30)
    assert ticks.tolist()==[0,30,30,30,30]
    assert ((ticks-native)<=15).tolist()==[True,False,True,True,True]
    assert np.all(c.event_tick(event,30)<event)
    assert np.all(event<=c.event_tick(event,30)+30)


def test_nearest_pair_units_and_same_pair_prediction():
    f=pd.DataFrame({"tick":[30,30,30],"flight_key":["A","B","C"],
        "time_epoch":[20.,21.,22.],"latitude":[50.,50.,50.],"longitude":[0.,1/60,5/60],
        "altitude_ft":[30000.,31000.,35000.],"ground_speed_kt":[0.,0.,0.],
        "track_deg":[0.,0.,0.],"vertical_rate_fpm":[0.,0.,0.],
        "source_archive":["x"]*3,"plot_index":[1,2,3]})
    g=c._snapshot_geometry(f)
    a=g.set_index("flight_key").loc["A"]
    assert a.neighbour_key=="B"
    assert abs(a.horizontal_nm-np.cos(np.deg2rad(50)))<.002
    assert a.vertical_ft==1000
    assert np.isnan(a.tcpa_min)  # zero relative speed; no invented CPA
    assert a.focal_source_epoch<=a.tick and a.neighbour_source_epoch<=a.tick


def test_binned_denominator_rate_and_zero_event_interval():
    f=pd.DataFrame({"flight_key":["A","B","C","D"],"horizontal_nm":[.5,.6,1.5,8.],
                    "vertical_ft":[100,150,100,100],"exposure_hours":[.5,.5,1.,1.],"event_count":[1,0,0,2]})
    d=c.binned_rate(f,"horizontal_nm",np.array([0,1,2]),"vertical_ft",np.array([0,200]))
    assert d.events.sum()==1  # out-of-range event retained in source, never silently added
    assert d.exposure_hours.sum()==2
    near=d.loc[d.x_low.eq(0)].iloc[0]
    far=d.loc[d.x_low.eq(1)].iloc[0]
    assert near.rate_per_1000h==1000
    assert far.rate_per_1000h==0
    assert far.ci_low_per_1000h==0 and far.ci_high_per_1000h>0


def test_rate_is_deterministic_and_no_missing_denominator_is_zero():
    f=pd.DataFrame({"flight_key":["A"],"horizontal_nm":[.2],"exposure_hours":[.5],"event_count":[1]})
    a=c.binned_rate(f,"horizontal_nm",np.array([0,1,2]))
    b=c.binned_rate(f,"horizontal_nm",np.array([0,1,2]))
    pdt.assert_frame_equal(a,b)
    assert np.isnan(a.loc[a.x_low.eq(1),"rate_per_1000h"].iloc[0])


def test_figure_exports_png_svg_csv_and_metadata(tmp_path,monkeypatch):
    monkeypatch.setattr(v,"FIG",tmp_path)
    fig,ax=plt.subplots();ax.plot([0,1],[1,2])
    relative=v.finish(fig,"check","unit_export","A real plotted series",pd.DataFrame({"x":[0,1],"y":[1,2]}),
                      {"population":"test","unit":"NM"})
    folder=tmp_path/"check"
    assert relative.endswith("unit_export.png")
    assert all((folder/f"unit_export.{extension}").stat().st_size>0 for extension in ("png","svg","csv","json"))


def test_output_pair_and_population_invariants_if_built():
    a_path=c.OUT/"tables/11a_anchor_pairs.parquet"
    t_path=c.OUT/"tables/11a_fixed_pair_timeline.parquet"
    p_path=c.OUT/"tables/11b_population_risk_set.parquet"
    if not all(p.exists() for p in (a_path,t_path,p_path)):
        return
    a=pd.read_parquet(a_path,columns=["window_id","other_flight_key","time_epoch","other_source_epoch","event_epoch","anchor_source"])
    t=pd.read_parquet(t_path,columns=["window_id","other_flight_key","relative_bin_s"])
    p=pd.read_parquet(p_path,columns=["tick","flight_key","neighbour_key","focal_source_epoch","neighbour_source_epoch","event_count","exposure_hours"])
    assert a.window_id.is_unique and len(a)==8196
    assert a.anchor_source.eq("STAGE10B_BACKWARD_ASOF_NATIVE").all()
    assert (a.time_epoch<=a.event_epoch+1e-6).all()
    assert (a.other_source_epoch.dropna()<=a.loc[a.other_source_epoch.notna(),"event_epoch"]+1e-6).all()
    assert t.merge(a[["window_id","other_flight_key"]],on=["window_id","other_flight_key"],validate="many_to_one").shape[0]==len(t)
    assert not t.duplicated(["window_id","relative_bin_s"]).any()
    assert not p.duplicated(["tick","flight_key"]).any()
    assert (p.focal_source_epoch<=p.tick+1e-6).all()
    assert (p.neighbour_source_epoch.dropna()<=p.loc[p.neighbour_source_epoch.notna(),"tick"]+1e-6).all()
    assert np.allclose(p.exposure_hours,30/3600)
    assert (p.event_count>=0).all()


def test_observation_segments_are_pair_specific_and_gap_sensitive():
    f=pd.DataFrame({"window_id":["I:a"]*3+["I:b"],"other_flight_key":["N"]*3+["M"],
        "focal_flight_key":["A"]*3+["B"],"time_epoch":[100.,130.,201.,130.],
        "relative_seconds":[-30.,0.,71.,0.],"relative_bin_s":[-30,0,60,0],
        "ATC_INTERVENTION":[1]*4,"clearance_type":["LEVEL"]*4,
        "horizontal_separation_nm":[5.,4.,6.,9.],"vertical_separation_ft":[1000.]*4})
    a=observed_segments(f,60)
    b=observed_segments(f,90)
    assert len(a)==3 and len(b)==2
    assert a.loc[a.window_id.eq("I:a"),"n_pair_bins"].tolist()==[2,1]
    assert a.loc[a.window_id.eq("I:a"),"contains_anchor_bin"].tolist()==[True,False]

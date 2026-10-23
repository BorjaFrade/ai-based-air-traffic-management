"""Adapter equivalence and production detector boundary checks for Stage 10B."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "10b_exact_dynamic_rule_cruise.py"
SPEC = importlib.util.spec_from_file_location("stage10b", SCRIPT)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)
prod = stage.production
shared = sys.modules["opensky_safety"]


def scat_pair(*, distance_nm=6.0, track_a=90.0, track_b=270.0,
              altitude_b=31000.0, rate_a=0.0, rate_b=0.0) -> pd.DataFrame:
    # Equatorial longitude separation is close to distance_nm/60 degrees.
    return pd.DataFrame([
        {"flight_key":"a","callsign":"A","latitude":0.0,"longitude":0.0,
         "altitude_ft":30000.0,"ground_speed_ms":230.0,"track_deg":track_a,
         "heading_deg":123.0,"vertical_rate_fpm":rate_a},
        {"flight_key":"b","callsign":"B","latitude":0.0,"longitude":distance_nm/60.1,
         "altitude_ft":altitude_b,"ground_speed_ms":230.0,"track_deg":track_b,
         "heading_deg":246.0,"vertical_rate_fpm":rate_b},
    ])


def test_adapter_exact_production_input_and_output():
    native=scat_pair()
    adapted=stage.detector_frame(native)
    canonical=pd.DataFrame({"icao24":["a","b"],"callsign":["A","B"],
       "latitude":[0.0,0.0],"longitude":[0.0,6.0/60.1],
       "baro_altitude_ft":[30000.0,31000.0],"velocity":[230.0,230.0],
       "true_track":[90.0,270.0],"vertical_rate_fpm":[0.0,0.0],
       "on_ground":[False,False]})
    pd.testing.assert_frame_equal(adapted,canonical)
    assert prod.analyse_snapshot_fast(adapted)==prod.analyse_snapshot_fast(canonical)


def test_units_and_heading_track_choice():
    native=scat_pair()
    adapted=stage.detector_frame(native)
    assert adapted.true_track.tolist()==[90.0,270.0]
    assert adapted.velocity.tolist()==[230.0,230.0]
    assert adapted.baro_altitude_ft.tolist()==[30000.0,31000.0]
    assert adapted.vertical_rate_fpm.tolist()==[0.0,0.0]
    h,v=prod.pair_separation_at_time(adapted.iloc[0],adapted.iloc[1],0.0)
    assert abs(h-6.0)<.1 and v==1000.0


def test_heading_wrap_and_true_track_projection():
    native=scat_pair(track_a=359.0,track_b=1.0)
    adapted=stage.detector_frame(native)
    native.heading_deg=[180.0,180.0]
    pd.testing.assert_frame_equal(adapted,stage.detector_frame(native))
    assert adapted.true_track.tolist()==[359.0,1.0]
    lon,lat,alt=shared.project_aircraft_state(adapted.iloc[0],60.0)
    assert lat>0 and abs(lon)<.01 and alt==30000.0


def test_parallel_converging_diverging_trajectories():
    parallel=stage.detector_frame(scat_pair(track_a=90,track_b=90))
    closing=stage.detector_frame(scat_pair(track_a=90,track_b=270))
    diverging=stage.detector_frame(scat_pair(track_a=270,track_b=90))
    hp0,_=prod.pair_separation_at_time(parallel.iloc[0],parallel.iloc[1],0)
    hp10,_=prod.pair_separation_at_time(parallel.iloc[0],parallel.iloc[1],10)
    hc10,_=prod.pair_separation_at_time(closing.iloc[0],closing.iloc[1],10)
    hd10,_=prod.pair_separation_at_time(diverging.iloc[0],diverging.iloc[1],10)
    assert abs(hp10-hp0)<.02
    assert hc10<hp0<hd10
    # Uses full unchanged production pair construction/forecast path.
    assert prod.analyse_snapshot_fast(closing)[1]["pairs_total"]==1


def test_vertical_separation_deadband_and_missing_field():
    native=scat_pair(altitude_b=30950,rate_a=50,rate_b=0)
    adapted=stage.detector_frame(native)
    _,v0=prod.pair_separation_at_time(adapted.iloc[0],adapted.iloc[1],0,
                    vertical_rate_deadband_fpm=prod.VERTICAL_RATE_DEADBAND_FPM)
    _,v60=prod.pair_separation_at_time(adapted.iloc[0],adapted.iloc[1],60,
                    vertical_rate_deadband_fpm=prod.VERTICAL_RATE_DEADBAND_FPM)
    assert v0==v60==950.0
    adapted.loc[1,"velocity"]=np.nan
    assert pd.isna(adapted.loc[1,"velocity"])
    # Production accepts a missing velocity and keeps that state stationary.
    assert prod.analyse_snapshot_fast(adapted)[1]["pairs_total"]==1


def test_stale_state_and_no_future_leakage():
    native=scat_pair()
    native["archive_week"]=stage.WEEK
    native["time_epoch"]=[100.0,84.0]
    native["on_ground"]=[None,None]
    windows=pd.DataFrame([{"window_id":"w","matched_pair_id":0,"ATC_INTERVENTION":1,
         "split":"test","anchor_epoch":100.0,"focal_flight_key":"a"}])
    states,cov=stage.reconstruct(windows,native.sort_values("time_epoch").reset_index(drop=True),15)
    assert not cov.evaluable.iloc[0] and cov.unevaluable_reason.iloc[0]=="NO_CONTEMPORANEOUS_NEIGHBOUR"
    native.loc[1,"time_epoch"]=101.0
    states,cov=stage.reconstruct(windows,native.sort_values("time_epoch").reset_index(drop=True),15)
    assert not cov.evaluable.iloc[0]
    assert (states.source_timestamp<=states.anchor_timestamp).all()


def test_multiple_aircraft_and_severity_aggregation():
    native=scat_pair(distance_nm=3.0,altitude_b=30500.0)
    third=native.iloc[[1]].copy()
    third.loc[:,'flight_key']="c"
    third.loc[:,'callsign']="C"
    third.loc[:,'longitude']=.5
    native=pd.concat([native,third],ignore_index=True)
    frame=stage.detector_frame(native)
    events,stats=prod.analyse_snapshot_fast(frame)
    assert stats["pairs_total"]==3
    focal=[event for event in events if 0 in (event["index_a"],event["index_b"])]
    assert any(event["event_state"]==prod.STATE_CONFLICT for event in focal)
    assert max(stage.SEVERITY[stage.COLOURS[x["event_state"]]] for x in focal)==stage.SEVERITY["RED"]
    native["window_id"]="w"
    windows=pd.DataFrame([{"window_id":"w","matched_pair_id":0,"ATC_INTERVENTION":1,
         "split":"test","anchor_epoch":100.0,"focal_flight_key":"a"}])
    coverage=pd.DataFrame([{"window_id":"w","matched_pair_id":0,"ATC_INTERVENTION":1,
         "split":"test","anchor_timestamp":100.0,"focal_flight_key":"a",
         "n_aircraft":3,"n_neighbours":2,"evaluable":True,
         "unevaluable_reason":"EVALUABLE","focal_age_s":0.0}])
    result=stage.execute(windows,native,coverage,save_raw=False,label="test")
    assert result.focal_dynamic_state.iloc[0]=="RED"
    assert result.number_red_pairs.iloc[0]>=1
    assert result.number_green_pairs.iloc[0]+result.number_yellow_pairs.iloc[0]+result.number_orange_pairs.iloc[0]+result.number_red_pairs.iloc[0]==2

"""Synthetic calculation tests; no synthetic results enter scientific outputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from stage11_core import event_tick
from stage11d_core import PARAMS, pair_group, select_neighbors, vertical_status
from stage11d_tables import rates


def test_vertical_state_and_pair_classification():
    s=vertical_status(np.array([np.nan,-201,-200,0,200,201]),200)
    assert s.tolist()==["UNKNOWN","DESCENDING","LEVEL_FLIGHT","LEVEL_FLIGHT","LEVEL_FLIGHT","CLIMBING"]
    g=pair_group(np.array(["LEVEL_FLIGHT","LEVEL_FLIGHT","CLIMBING","DESCENDING","UNKNOWN"]),
                 np.array(["LEVEL_FLIGHT","CLIMBING","DESCENDING","DESCENDING","LEVEL_FLIGHT"]))
    assert g.tolist()==["BOTH_LEVEL","LEVEL_AND_CLIMB","OPPOSITE_MOTION","BOTH_DESCEND","UNKNOWN"]


def test_neighbor_criteria_choose_different_valid_aircraft():
    t=1000.0
    f=pd.DataFrame(dict(flight_key=["FOCAL","NEAR","FAR"],latitude=[0,0,0],longitude=[0,.1,.2],
         altitude_ft=[35000,37000,36000],track_deg=[90,90,270],ground_speed_kt=[300,300,300],
         vertical_rate_fpm=[0,0,0],time_epoch=[t-5]*3))
    z=select_neighbors(f,np.array(["FOCAL"]))
    assert z["NEAREST_HORIZONTAL"]["neighbour_key"][0]=="NEAR"
    assert z["MINIMUM_3D_DISTANCE"]["neighbour_key"][0]=="NEAR"
    assert z["MINIMUM_FUTURE_CPA"]["neighbour_key"][0]=="FAR"
    assert z["MAXIMUM_CLOSING"]["neighbour_key"][0]=="FAR"
    assert z["NEAREST_VERTICAL"]["neighbour_key"][0]=="FAR"
    assert z["NEAREST_HORIZONTAL"]["horizontal_nm"][0]<z["MINIMUM_FUTURE_CPA"]["horizontal_nm"][0]
    assert z["MINIMUM_FUTURE_CPA"]["cpa_nm"][0]<.01
    assert z["MINIMUM_FUTURE_CPA"]["neighbour_source_epoch"][0]<=t


def test_neighbor_radius_and_unknown():
    f=pd.DataFrame(dict(flight_key=["A","B"],latitude=[0,0],longitude=[0,3],altitude_ft=[35000,35000],
        track_deg=[90,270],ground_speed_kt=[300,300],vertical_rate_fpm=[0,0],time_epoch=[1000,1000]))
    z=select_neighbors(f,np.array(["A"]))
    assert all(v["neighbour_key"][0] is None for v in z.values())
    assert PARAMS["candidate_radius_nm"]==100


def test_event_tick_is_strictly_pre_event():
    epochs=np.array([120.,120.01,149.99,150.])
    ticks=event_tick(epochs,30)
    assert ticks.tolist()==[90,120,120,120]
    assert np.all(epochs>ticks)
    assert np.all(epochs<=ticks+30)


def test_denominator_and_uncertainty_from_synthetic_counts():
    f=pd.DataFrame(dict(bin=[0,0,0,2,2],flight_key=["A","B","C","A","B"],
        exposure_hours=[.5,.5,.5,1,1],event_count=[1,0,0,2,0]))
    r=rates(f,["bin"]).set_index("bin")
    assert r.loc[0,"events"]==1 and r.loc[0,"exposure_hours"]==1.5
    assert np.isclose(r.loc[0,"rate_per_1000h"],1000/1.5)
    assert r.loc[0,"ci_low_per_1000h"]<=r.loc[0,"rate_per_1000h"]<=r.loc[0,"ci_high_per_1000h"]
    assert r.loc[2,"ci_high_per_1000h"]>r.loc[2,"rate_per_1000h"]

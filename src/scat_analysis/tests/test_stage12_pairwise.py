"""Synthetic invariants for Stage 12 encounter reconstruction."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage12_build_encounters import canonical_pair,pair_geometry
from stage12_compute_features import segment_flags,_episode_table
from stage12_validate_predictions import forecast
from stage12_boundary_audit import haversine_nm,classify_boundary


def state(key,lon=0,speed=60,track=90,alt=10000):
    return dict(entity_key=key,flight_key=key,source_archive='archive',source_member=key,
        time_epoch=0,plot_index=0,latitude=0,longitude=lon,altitude_ft=alt,
        ground_speed_kt=speed,track_deg=track,vertical_rate_fpm=0,flight_phase='CRUISE',active_centre_id='C')

def test_canonical_and_identity_reuse():
    assert canonical_pair('w','A','B')==canonical_pair('w','B','A')
    assert canonical_pair('w','archive:1::member1','X')[2]!=canonical_pair('w','archive:1::member2','X')[2]
    with pytest.raises(ValueError):canonical_pair('w','A','A')

def test_geographic_radius_and_no_order_duplicates():
    g=pd.DataFrame([state('A'),state('B',lon=1),state('C',lon=3)])
    p=pair_geometry(g,0,'w',100)
    assert len(p)==1 and p.pair_id.is_unique
    assert p.iloc[0].a_key=='A' and p.iloc[0].b_key=='B'
    assert abs(p.iloc[0].horizontal_nm-60.04)<.1
    assert abs(haversine_nm(0,0,0,1)-60.04)<.1

def test_simultaneous_encounters_are_all_retained():
    g=pd.DataFrame([state('A',lon=0),state('B',lon=.4),state('C',lon=.8)])
    p=pair_geometry(g,0,'w',100)
    assert len(p)==3
    assert p.pair_id.nunique()==3
    assert int(p.loc[p.a_key.eq('A'),'concurrent_neighbors_a'].iloc[0])==2

def test_relative_motion_vertical_units_and_simultaneity():
    g=pd.DataFrame([state('A',speed=60,track=90,alt=10000),
                    state('B',lon=1,speed=60,track=270,alt=11000)])
    p=pair_geometry(g,0,'w',100).iloc[0]
    assert p.vertical_ft==1000 and p.signed_vertical_ft==1000
    assert abs(p.relative_vx_nm_min+2)<1e-6
    assert p.closing_nm_min>0 and abs(p.tcpa_raw_min-30)<.1

def test_gap_and_same_pair_continuity():
    d=pd.DataFrame({'pair_id':['AB']*4+['AC'],'tick':[0,30,90,210,210],
                    'horizontal_nm':[10,9,8,7,5]})
    assert segment_flags(d,45).tolist()==[True,False,True,True,True]
    assert segment_flags(d,60).tolist()==[True,False,False,True,True]
    assert segment_flags(d,90).tolist()==[True,False,False,True,True]

def test_boundary_censoring_vs_radius_crossing():
    c=classify_boundary(np.array([True,False,True]),np.array([105,np.nan,90]),
                        np.array([False,True,False]),'start')
    assert c.tolist()==['OBSERVED_RADIUS_CROSSING','DATA_EDGE_CENSORED','PHYSICAL_OR_GRID_BREAK']

def test_future_only_cpa_and_no_future_input():
    valid,t,cpa,v=forecast(10,0,1,0,500,0,-10,5)
    assert not bool(valid) and np.isnan(cpa)
    valid,t,cpa,v=forecast(10,0,-1,0,500,-100,10,5)
    assert bool(valid) and t==5 and cpa==5 and v==0
    assert forecast(10,0,-1,0,500,-100,10,2)[2]==8

def test_simultaneous_minimum_not_independent_minimum():
    # The vertical minimum occurs at a different tick from the horizontal one.
    base={'episode_id':['E']*3,'pair_id':['AB']*3,'a_key':['A']*3,'b_key':['B']*3,
          'a_flight_key':['A']*3,'b_flight_key':['B']*3,'a_phase':['C']*3,'b_phase':['C']*3,
          'a_sector':['S']*3,'b_sector':['S']*3,'tick':[0,30,60],
          'horizontal_nm':[10,2,8],'vertical_ft':[0,2000,0],'signed_vertical_ft':[0,2000,0],
          'a_lat':[0]*3,'a_lon':[0]*3,'b_lat':[0]*3,'b_lon':[.1]*3,
          'a_alt_ft':[10000]*3,'b_alt_ft':[10000,12000,10000],
          'a_speed_kt':[200]*3,'b_speed_kt':[210]*3,'a_track_deg':[0]*3,'b_track_deg':[90]*3,
          'a_vrate_fpm':[0]*3,'b_vrate_fpm':[0]*3,'relative_east_nm':[10,2,8],
          'relative_north_nm':[0]*3,'relative_vx_nm_min':[-1]*3,'relative_vy_nm_min':[0]*3,
          'tcpa_raw_min':[10,2,8],'cpa_unbounded_nm':[0]*3,'closing_nm_min':[1,0,-1],
          'crossing_angle_deg':[90]*3,'a_age_s':[0]*3,'b_age_s':[0]*3,
          'concurrent_neighbors_a':[1]*3,'concurrent_neighbors_b':[1]*3}
    row=_episode_table(pd.DataFrame(base)).iloc[0]
    assert row.min_horizontal_nm==2 and row.vertical_at_min_horizontal_ft==2000
    assert row.min_vertical_ft==0 and row.horizontal_at_min_vertical_nm==10
    assert row.observed_duration_s==60

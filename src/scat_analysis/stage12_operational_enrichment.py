"""Attach operational context only after the physical structure is frozen."""
from __future__ import annotations
import json
import pandas as pd
import numpy as np
from stage12_audit_inputs import OUT,WEEKS,BASE


def main():
    freeze=OUT/'diagnostics'/'structure_discovery.json'
    if not freeze.exists():raise RuntimeError('Physical structure must be frozen first')
    clear=pd.read_parquet(BASE/'03_clearances/scat_clearances.parquet',columns=[
        'clearance_id','flight_key','archive_week','clearance_epoch','clearance_type','ATC_INTERVENTION'])
    rows=[]
    for week in WEEKS:
        ep=pd.read_parquet(OUT/'episodes'/f'{week}_episode_summary.parquet')
        ev=clear.loc[clear.archive_week.eq(week)&clear.ATC_INTERVENTION.eq(1)].copy()
        ev=ev.loc[np.isfinite(ev.clearance_epoch)].sort_values('clearance_epoch')
        lookup={key:g for key,g in ev.groupby('flight_key',sort=False)}
        n=len(ep);counts=np.zeros(n,dtype=np.int32);closest=np.full(n,np.nan)
        level=np.zeros(n,dtype=np.int32);heading=level.copy();speed=level.copy()
        nearest_ambiguity=np.full(n,np.nan)
        per_flight={}
        for i,side in enumerate(('a','b')):
            for key,indices in ep.groupby(f'{side}_flight_key',sort=False).indices.items():
                if key not in lookup:continue
                g=lookup[key];ts=g.clearance_epoch.to_numpy(float);typ=g.clearance_type.astype(str).to_numpy()
                ix=np.asarray(indices);starts=ep.start_tick.to_numpy()[ix]-300;ends=ep.end_tick.to_numpy()[ix]+300
                lo=np.searchsorted(ts,starts,'left');hi=np.searchsorted(ts,ends,'right')
                for j,a,b in zip(ix,lo,hi):
                    if a==b:continue
                    times=ts[a:b];types=typ[a:b]
                    counts[j]+=b-a;level[j]+=np.char.find(types.astype(str),'LEVEL').__ge__(0).sum()
                    heading[j]+=np.char.find(types.astype(str),'HEADING').__ge__(0).sum()
                    speed[j]+=np.char.find(types.astype(str),'SPEED').__ge__(0).sum()
                    midpoint=(ep.start_tick.iat[j]+ep.end_tick.iat[j])/2
                    delta=times-midpoint;candidate=delta[np.argmin(abs(delta))]
                    if not np.isfinite(closest[j]) or abs(candidate)<abs(closest[j]):closest[j]=candidate
        # Ambiguity is number of candidate episodes for the intervened flight at each clearance.
        for key,g in ev.groupby('flight_key',sort=False):
            parts=[]
            for side in ('a','b'):
                parts.append(ep.loc[ep[f'{side}_flight_key'].eq(key),['start_tick','end_tick']])
            e=pd.concat(parts,ignore_index=True)
            if e.empty:continue
            starts=np.sort(e.start_tick.to_numpy());ends=np.sort(e.end_tick.to_numpy())
            ts=g.clearance_epoch.to_numpy(float)
            ambiguity=np.searchsorted(starts,ts+300,side='right')-np.searchsorted(ends,ts-300,side='left')
            per_flight[key]=dict(zip(g.clearance_id.astype(str),ambiguity.astype(int)))
        # For each episode count whether a temporally associated clearance is unique to that pair.
        unique=np.zeros(n,dtype=np.int32);ambiguous=np.zeros(n,dtype=np.int32)
        for side in ('a','b'):
            for key,indices in ep.groupby(f'{side}_flight_key',sort=False).indices.items():
                if key not in lookup:continue
                g=lookup[key];ts=g.clearance_epoch.to_numpy(float);ids=g.clearance_id.astype(str).to_numpy()
                ix=np.asarray(indices);lo=np.searchsorted(ts,ep.start_tick.to_numpy()[ix]-300)
                hi=np.searchsorted(ts,ep.end_tick.to_numpy()[ix]+300,'right')
                mapping=per_flight.get(key,{})
                for j,a,b in zip(ix,lo,hi):
                    if a==b:continue
                    v=np.asarray([mapping.get(x,0) for x in ids[a:b]])
                    unique[j]+=(v==1).sum();ambiguous[j]+=(v>1).sum()
        result=ep[['episode_id','pair_id','a_flight_key','b_flight_key','start_tick','end_tick',
                   'min_horizontal_nm','vertical_at_min_horizontal_ft','start_closing_nm_min']].copy()
        result['temporally_associated_clearances']=counts
        result['nearest_clearance_minus_episode_midpoint_s']=closest
        result['level_count']=level;result['heading_count']=heading;result['speed_count']=speed
        result['unique_pair_context_count']=unique;result['ambiguous_pair_context_count']=ambiguous
        result['atc_context']=counts>0
        result.to_parquet(OUT/'tables'/f'{week}_operational_enrichment.parquet',index=False)
        rows.append(dict(week=week,episodes=n,episodes_with_atc_context=int((counts>0).sum()),
                         linked_clearance_contexts=int(counts.sum()),unique_pair_contexts=int(unique.sum()),
                         ambiguous_pair_contexts=int(ambiguous.sum()),
                         caveat='Events are linked by participant and +/-300 s around the episode, not attributed causally to a pair.'))
        print(rows[-1],flush=True)
    # Stage 10B colours are sampled focal-window context only for the pilot week.
    d=pd.read_parquet(BASE/'10_dynamic_rule_cruise/focal_dynamic_states.parquet',columns=[
        'window_id','focal_flight_key','anchor_epoch','focal_dynamic_state','ATC_INTERVENTION'])
    ep=pd.read_parquet(OUT/'episodes'/f'{WEEKS[0]}_episode_summary.parquet',columns=[
        'episode_id','a_flight_key','b_flight_key','start_tick','end_tick'])
    by_flight={}
    for side in ('a','b'):
        for key,g in ep.groupby(f'{side}_flight_key',sort=False):
            by_flight.setdefault(key,[]).append(g[['episode_id','start_tick','end_tick']])
    by_flight={k:pd.concat(v,ignore_index=True) for k,v in by_flight.items()}
    contexts=[]
    for row in d.itertuples(index=False):
        e=by_flight.get(row.focal_flight_key)
        if e is None:continue
        candidates=e.loc[e.start_tick.le(row.anchor_epoch)&e.end_tick.ge(row.anchor_epoch)]
        for match in candidates.itertuples(index=False):
            contexts.append(dict(episode_id=match.episode_id,window_id=row.window_id,
                focal_flight_key=row.focal_flight_key,anchor_epoch=row.anchor_epoch,
                focal_dynamic_state=row.focal_dynamic_state,ATC_INTERVENTION=row.ATC_INTERVENTION,
                concurrent_pair_contexts=len(candidates)))
    context=pd.DataFrame(contexts);context.to_parquet(OUT/'tables'/'pilot_detector_context.parquet',index=False)
    rows.append(dict(pilot_detector_windows=len(d),pilot_window_episode_links=len(context),
        linked_unique_windows=context.window_id.nunique() if len(context) else 0,
        caveat='Focal colours from Stage 10B matched windows; no per-pair detector score and no replication colours.'))
    (OUT/'diagnostics'/'operational_enrichment.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(rows[-1],flush=True)

if __name__=='__main__':main()

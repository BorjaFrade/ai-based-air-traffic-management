#!/usr/bin/env python3
"""Decompose pair-instant lift by position within the frozen 360 s test episodes."""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import units_ladder as u
import units_multiplicity as mult

OUT = u.ROOT / "results/benchmarks_units/mechanism"
WEEKS = ("20170107_20170113", "20170401_20170407",
         "20170624_20170630", "20170916_20170922")
KBINS = ("1", "2", "3", "4+")
TBINS = ("0 s", "(0,300] s", "(300,600] s", ">600 s")
LBINS = ("1", "2", "3", "4-5", "6-10", ">10")
DBINS = ("0 s", "(0,300] s", "(300,600] s", "(600,1200] s", "(1200,3600] s", ">3600 s")
PBINS = ("0", "1", "2", "3", "4+")


def positions(indices: np.ndarray, ep_id: np.ndarray, t: np.ndarray) -> np.ndarray:
    """One-based order within each episode, keeping Parquet order for ties."""
    out = np.zeros(len(t), np.int32)
    order = indices[np.lexsort((indices, t[indices], ep_id[indices]))]
    if len(order):
        cut = np.r_[True, ep_id[order[1:]] != ep_id[order[:-1]]]
        starts = np.flatnonzero(cut)
        out[order] = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)])) + 1
    return out


def first_events(t: np.ndarray, a: np.ndarray, b: np.ndarray,
                 index: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = np.full(len(t), np.inf)
    for keys in (a, b):
        for key, rows in pd.Series(keys).groupby(keys, sort=False).indices.items():
            entry = index.get(key)
            if entry is None:
                continue
            epochs = entry[0]
            ix = np.searchsorted(epochs, t[rows], side="right")
            valid = ix < len(epochs)
            candidate = np.full(len(rows), np.inf)
            candidate[valid] = epochs[ix[valid]]
            out[rows] = np.minimum(out[rows], candidate)
    return out


def add_tensor(rec: dict, day: np.ndarray, stratum: np.ndarray, y: np.ndarray,
               ep: np.ndarray, n_days: int) -> None:
    flat = day * 4 + stratum
    rec["den"] += np.bincount(flat, minlength=n_days*4).reshape(n_days, 4)
    rec["num"] += np.bincount(flat, weights=y, minlength=n_days*4).reshape(n_days, 4)
    rec["episodes"] += len(np.unique(ep))


def rate_lift(rec: dict, base: dict, weights: np.ndarray) -> tuple[float, float, float]:
    n, z = rec["den"], rec["num"]
    nb, zb = base["den"], base["num"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.divide(zb.sum(0), nb.sum(0), out=np.zeros(4), where=nb.sum(0)>0)
        point = z.sum() / (n.sum(0)*rb).sum()
        ns, zs = weights@n, weights@z
        nbs, zbs = weights@nb, weights@zb
        rbs = np.divide(zbs, nbs, out=np.zeros_like(zbs), where=nbs>0)
        boot = zs.sum(1)/(ns*rbs).sum(1)
    u.check(np.isfinite(boot).all(), "Non-finite mechanism bootstrap lift")
    return float(point), float(np.quantile(boot,.025)), float(np.quantile(boot,.975))


def fmt(row: pd.Series, col: str) -> str:
    return f"{row[col]:.2f} ({row[col+'_lo']:.2f}–{row[col+'_hi']:.2f})"


def main() -> None:
    start = time.perf_counter()
    u.check(not OUT.exists(), f"Output already exists: {OUT}")
    audit = json.loads((u.ROOT/"data/processed/scat/19_calib_selection/tables/audit.json").read_text())
    u.check(tuple(audit["test"]) == WEEKS, "Frozen test weeks changed")
    days, day_lookup = u.test_days(list(WEEKS))
    nd = len(days)
    weights = np.random.default_rng(u.SEED).multinomial(nd, np.ones(nd)/nd, size=u.B)
    def empty():
        return {"den": np.zeros((nd,4)), "num": np.zeros((nd,4)), "episodes": 0}
    refs = {k: empty() for k in (("four_class","obs"),("four_class","ep"),
                                 ("binary_1","obs"),("binary_1","ep"))}
    acc = defaultdict(empty)
    lengths, clearances = [], []
    diagnostics = {"seed":u.SEED,"replicates":u.B,"weeks":list(WEEKS),"weekly":{},"assertions":{}}
    attr, event_indexes = mult.sources()
    pf = pq.ParquetFile(u.ROOT/"data/processed/scat/02_traffic/scat_states.parquet")
    for week in WEEKS:
        f = pd.read_parquet(u.ROOT/"results/stage20/pairs"/f"pairs_{week}.parquet",
                            columns=["T","flight_key_a","flight_key_b","source_member_a",
                                     "source_member_b","m_min","color","stage16_label_300"])
        t = f["T"].to_numpy(float)
        m = f.m_min.to_numpy(float)
        y = f.stage16_label_300.to_numpy(bool)
        rank = f.color.map({c:i for i,c in enumerate(u.COLORS)}).to_numpy(np.int8)
        a = f.flight_key_a.astype(str).str.cat(f.source_member_a.astype(str),sep="\x1f")
        b = f.flight_key_b.astype(str).str.cat(f.source_member_b.astype(str),sep="\x1f")
        keys = pd.Index(pd.concat([a,b],ignore_index=True).unique())
        ac, bc = keys.get_indexer(a), keys.get_indexer(b)
        pair = np.minimum(ac,bc).astype(np.int64)*len(keys)+np.maximum(ac,bc)
        unique_t, tc = np.unique(t,return_inverse=True)
        inst = u.snapshot_strata(week, unique_t, pf)[tc]
        day = np.array([day_lookup[int(v)] for v in (t//86400).astype(int)], dtype=np.int8)
        ep = u.make_episodes(t,pair,rank,y,m)
        eid = ep["id"]
        first = ep["first"]
        last = np.full(ep["count"],-np.inf)
        np.maximum.at(last,eid,t)
        duration = last-t[first]
        positive_per_ep = np.bincount(eid,weights=y,minlength=ep["count"]).astype(int)
        diagnostics["weekly"][week] = {"observations":len(t),"episodes":ep["count"],
                                        "non_green_episodes":int((ep["max_rank"]>0).sum()),
                                        "green_episodes":int((ep["max_rank"]==0).sum())}
        for scheme, obs_mask, ep_mask in (("four_class",rank==0,ep["max_rank"]==0),
                                          ("binary_1",m>=1,ep["min_m"]>=1)):
            for typ, ix in (("obs",np.flatnonzero(obs_mask)),("ep",first[ep_mask])):
                add_tensor(refs[(scheme,typ)],day[ix],inst[ix],y[ix],eid[ix],nd)
        kep = positions(np.arange(len(t)),eid,t)
        elapsed = t-t[first][eid]
        time_bin = np.select([elapsed==0,elapsed<=300,elapsed<=600],[0,1,2],default=3)
        length_bin = np.select([ep["size"]==1,ep["size"]==2,ep["size"]==3,
                                ep["size"]<=5,ep["size"]<=10],[0,1,2,3,4],default=5)
        duration_bin = np.select([duration==0,duration<=300,duration<=600,
                                  duration<=1200,duration<=3600],[0,1,2,3,4],default=5)
        for scheme,classes in (("four_class",[(c,rank==i) for i,c in enumerate(u.COLORS[1:],1)]),
                               ("binary_1",[("positive",m<1)])):
            for cls,mask in classes:
                kcls = positions(np.flatnonzero(mask),eid,t)
                bins = (("k_ep",np.minimum(kep-1,3),KBINS),
                        ("k_cls",np.minimum(kcls-1,3),KBINS),
                        ("time_from_T0",time_bin,TBINS))
                for variant,values,labels in bins:
                    for bi,label in enumerate(labels):
                        ix=np.flatnonzero(mask & (values==bi))
                        add_tensor(acc[(scheme,cls,variant,label)],day[ix],inst[ix],y[ix],eid[ix],nd)
                ix=np.flatnonzero(mask)
                add_tensor(acc[(scheme,cls,"all","all")],day[ix],inst[ix],y[ix],eid[ix],nd)
        for cls, epmask in [(c,ep["max_rank"]==i) for i,c in enumerate(u.COLORS)]+[("binary_1",ep["min_m"]<1)]:
            for bi,label in enumerate(LBINS):
                selected=epmask & (length_bin==bi)
                if selected.any():
                    lengths.append(dict(week=week,section="length", **{"class":cls},bin=label,
                                        n_episodes=int(selected.sum()),observations=int(ep["size"][selected].sum()),
                                        duration_mean_s=float(duration[selected].mean()),
                                        duration_median_s=float(np.median(duration[selected])),
                                        duration_sum_s=float(duration[selected].sum()),
                                        start_positives=int(y[first[selected]].sum()),
                                        start_label_rate=float(y[first[selected]].mean())))
            for bi,label in enumerate(DBINS):
                selected=epmask & (duration_bin==bi)
                lengths.append(dict(week=week,section="duration", **{"class":cls},bin=label,
                                    n_episodes=int(selected.sum()),observations=int(ep["size"][selected].sum()),
                                    duration_mean_s=float(duration[selected].mean()) if selected.any() else np.nan,
                                    duration_median_s=float(np.median(duration[selected])) if selected.any() else np.nan,
                                    duration_sum_s=float(duration[selected].sum()),
                                    start_positives=int(y[first[selected]].sum()),
                                    start_label_rate=float(y[first[selected]].mean()) if selected.any() else np.nan))
        for cls,epmask in [(c,ep["max_rank"]==i) for i,c in enumerate(u.COLORS[1:],1)]+[("binary_1",ep["min_m"]<1)]:
            pos_bin=np.minimum(positive_per_ep,4)
            for bi,label in enumerate(PBINS):
                selected=epmask & (pos_bin==bi)
                lengths.append(dict(week=week,section="positive_count",**{"class":cls},bin=label,
                                    n_episodes=int(selected.sum())))
        # The Stage 20 audit has zero label disagreements with these eligible events.
        iy=np.flatnonzero(y)
        event=first_events(t[iy],a.iloc[iy].to_numpy(str),b.iloc[iy].to_numpy(str),event_indexes[week])
        delta=event-t[iy]
        u.check(np.isfinite(delta).all() and ((delta>0)&(delta<=300)).all(),
                f"Eligible clearance times do not reproduce positive labels in {week}")
        dt_bin=np.minimum(np.ceil(delta/60).astype(int)-1,4)
        for cls,mask in [(c,rank[iy]==i) for i,c in enumerate(u.COLORS[1:],1)]+[("binary_1",m[iy]<1)]:
            for bi in range(5):
                clearances.append(dict(week=week,section="observation_delay",**{"class":cls},
                                       bin=f"({bi*60},{(bi+1)*60}] s",n=int((mask&(dt_bin==bi)).sum())))
        first_clear=np.full(ep["count"],np.inf)
        np.minimum.at(first_clear,eid[iy],event)
        before=np.bincount(eid,weights=(t<first_clear[eid]).astype(int),minlength=ep["count"]).astype(int)
        for cls,epmask in [(c,ep["max_rank"]==i) for i,c in enumerate(u.COLORS[1:],1)]+[("binary_1",ep["min_m"]<1)]:
            selected=epmask & np.isfinite(first_clear)
            for k in np.unique(before[selected]):
                ix=selected & (before==k)
                clearances.append(dict(week=week,section="episode_clearance_position",**{"class":cls},
                                       bin=str(k),n=int(ix.sum()),
                                       episode_length_sum=int(ep["size"][ix].sum()),
                                       relative_position_mean=float((before[ix]/ep["size"][ix]).mean())))
        print(f"Mechanism: {week}: {len(t):,} observations",flush=True)
    rows=[]
    for (scheme,cls,variant,bin_),rec in acc.items():
        n=int(rec["den"].sum()); pos=int(rec["num"].sum())
        allrec=acc[(scheme,cls,"all","all")]
        lift0,lo0,hi0=rate_lift(rec,refs[(scheme,"obs")],weights)
        lifte,loe,hie=rate_lift(rec,refs[(scheme,"ep")],weights)
        rate_boot=(weights@rec["num"]).sum(1)/(weights@rec["den"]).sum(1)
        u.check(np.isfinite(rate_boot).all(),"Non-finite mechanism bootstrap rate")
        rows.append(dict(scheme=scheme,**{"class":cls},variant=variant,bin=bin_,n_obs=n,
                         n_episodes=rec["episodes"],positives=pos,rate=pos/n,
                         rate_lo=np.quantile(rate_boot,.025),rate_hi=np.quantile(rate_boot,.975),
                         share_observations=n/allrec["den"].sum(),
                         share_positives=pos/allrec["num"].sum(),
                         lift_ref0=lift0,lift_ref0_lo=lo0,lift_ref0_hi=hi0,
                         lift_refEp=lifte,lift_refEp_lo=loe,lift_refEp_hi=hie))
    positions_df=pd.DataFrame(rows)
    original=pd.read_csv(u.ROOT/"results/benchmarks_units/ladder/ladder_rates_lifts.csv")
    for scheme,cls,variant,bin_,column,step,name in (
        ("four_class","rojo","all","all","lift_ref0","0","i_rojo"),
        ("binary_1","positive","all","all","lift_ref0","0","i_binary_1"),
        ("four_class","rojo","k_cls","1","lift_refEp","3","ii_rojo"),
        ("binary_1","positive","k_cls","1","lift_refEp","3","ii_binary_1")):
        got=positions_df.loc[(positions_df.scheme==scheme)&(positions_df["class"]==cls)&
                             (positions_df.variant==variant)&(positions_df.bin==bin_),column].iloc[0]
        expected=original.loc[(original.scheme==scheme)&(original["class"]==cls)&
                              (original.stratifier=="instant")&(original.step==step),"lift"].iloc[0]
        difference=abs(got-expected)
        diagnostics["assertions"][name]={"actual":float(got),"expected":float(expected),
                                         "absolute_difference":float(difference),"passed":bool(difference<1e-12)}
        u.check(difference<1e-12,f"A1 assertion {name} failed: {got} vs {expected}")
    refrows=[]
    for scheme in ("four_class","binary_1"):
        row={"scheme":scheme}
        for typ,col in (("obs","ref_rate_obs"),("ep","ref_rate_ep")):
            r=refs[(scheme,typ)]
            n,z=r["den"].sum(),r["num"].sum()
            nb,zb=(weights@r["den"]).sum(1),(weights@r["num"]).sum(1)
            bs=zb/nb
            row.update({col:z/n,col+"_lo":np.quantile(bs,.025),col+"_hi":np.quantile(bs,.975),
                        col+"_n":int(n),col+"_positives":int(z)})
        refrows.append(row)
    diagnostics["elapsed_seconds"]=round(time.perf_counter()-start,1)
    OUT.mkdir(parents=True)
    positions_df.to_csv(OUT/"position_rates_lifts.csv",index=False)
    pd.DataFrame(refrows).to_csv(OUT/"reference_rates.csv",index=False)
    length_df=pd.DataFrame(lengths)
    length_df["share_episodes"] = length_df.n_episodes/length_df.groupby(["week","section","class"]).n_episodes.transform("sum")
    length_df.to_csv(OUT/"episode_length.csv",index=False)
    clearance_df=pd.DataFrame(clearances)
    clearance_df["share"] = clearance_df.n/clearance_df.groupby(["week","section","class"]).n.transform("sum")
    clearance_df.to_csv(OUT/"time_to_clearance.csv",index=False)
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    lines=["# Mechanism of the lift by position", "",
           "Command: `python3 paper/figure_scripts/units_mechanism.py`.",
           f"Measured time: {diagnostics['elapsed_seconds']} s. Seed {u.SEED}; {u.B} whole-day bootstrap replicates (28 days).",
           "", "Episodes: green and non-green runs kept separate, gap >360 s. `k_ep` and `k_cls` start at 1; ties follow the order of the Parquet file. `T0` is the first instant of the episode.",
           "The rate is positives/n_obs and has its own CI. `lift_ref0` uses the reference of Step 0 by instantaneous stratum; `lift_refEp` uses the reference of Step 3, one row per episode at its anchor. The CIs are the 2.5th and 97.5th percentiles of the day resampling. The row proportions are computed within each class.",
           "`episode_length.csv` includes distributions by length and duration (last minus first instant, s), as well as the number of positive observations per non-green episode; `share_episodes` uses the total of each class, week and section. The start label uses (T0,T0+300 s]. The time to clearance uses the first eligible clearance in (t,t+300 s]. For episodes, the position indicates how many observations precede the first later clearance; the quotient divides by the length.",
           "", "| Class | Variant | Bin | lift_ref0 (CI) | lift_refEp (CI) |", "|---|---|---|---|---|"]
    for scheme,cls in (("four_class","rojo"),("binary_1","positive")):
        for variant in ("k_ep","k_cls"):
            for bin_ in KBINS:
                row=positions_df.loc[(positions_df.scheme==scheme)&(positions_df["class"]==cls)&
                                     (positions_df.variant==variant)&(positions_df.bin==bin_)].iloc[0]
                lines.append(f"| {'red' if cls=='rojo' else cls} | {variant} | {bin_} | {fmt(row,'lift_ref0')} | {fmt(row,'lift_refEp')} |")
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    print(f"Mechanism complete in {diagnostics['elapsed_seconds']} s",flush=True)


if __name__ == "__main__":
    main()

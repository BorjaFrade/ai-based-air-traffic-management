#!/usr/bin/env python3
"""Distance from the last observed position of tracks that end before the
20-min endpoint (in-band) to the edge of the recorded SCAT footprint.

Footprint = occupancy grid (about 11 km cells) of all surveillance positions
of the aircraft involved in the 54,441 non-green episodes. Reference: same
distance for positions of evaluable episodes at the anchor time.
Outputs results/benchmarks/track_end_footprint.csv and a printed summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scat_analysis"))
import stage21_maneuver_screening as s21  # noqa: E402

OUT = ROOT / "results/benchmarks"
TRACKS = ROOT / "data/processed/scat/02_traffic/scat_states.parquet"
LAT0, LAT1, LON0, LON1 = 50.0, 72.0, 0.0, 34.0
DLAT, DLON = 0.1, 0.2          # 11.1 km x 11.1 km at 60N
KM_PER_CELL = 11.1
NM_PER_KM = 1 / 1.852


def cell(lat, lon):
    return (np.clip(((np.asarray(lat) - LAT0) / DLAT).astype(int), 0, int((LAT1 - LAT0) / DLAT) - 1),
            np.clip(((np.asarray(lon) - LON0) / DLON).astype(int), 0, int((LON1 - LON0) / DLON) - 1))


def edge_distance_cells(occ, ii, jj):
    """Distance (in cells) from cells (ii, jj) to the nearest empty cell; 0 if the cell is empty."""
    ei, ej = np.nonzero(~occ)
    empties = np.column_stack([ei, ej]).astype(np.float32)
    key = ii.astype(np.int64) * occ.shape[1] + jj
    uniq, inv = np.unique(key, return_inverse=True)
    ui, uj = (uniq // occ.shape[1]).astype(np.float32), (uniq % occ.shape[1]).astype(np.float32)
    out = np.empty(len(uniq), np.float32)
    for a in range(0, len(uniq), 400):
        di = ui[a:a + 400, None] - empties[None, :, 0]
        dj = uj[a:a + 400, None] - empties[None, :, 1]
        out[a:a + 400] = np.sqrt(di * di + dj * dj).min(1)
    return out[inv]


def main():
    ep = pd.read_csv(ROOT / "results/stage20/stage20_episodes.csv",
                     usecols=["episode_id", "week", "entity1", "entity2"])
    v = pd.read_csv(ROOT / "results/stage21/stage21_encounters.csv",
                    usecols=["episode_id", "week", "color", "anchor_time_utc",
                             "observed_evaluable", "start_observed"])
    v = v.merge(ep, on=["episode_id", "week"], validate="one_to_one")
    assert len(v) == 54441
    pf = pq.ParquetFile(TRACKS)
    shape = (int((LAT1 - LAT0) / DLAT), int((LON1 - LON0) / DLON))
    counts = np.zeros(shape, np.int64)
    rows = []
    for week, g in v.groupby("week", sort=False):
        keys = set(g.entity1) | set(g.entity2)
        tracks, audit = s21.load_tracks_week(pf, week, keys, "ft")
        print(week, audit, flush=True)
        archive_end = pd.Timestamp(week.split("_")[1], tz="UTC").timestamp() + 86400
        for tr in tracks.values():
            i, j = cell(tr[:, 1], tr[:, 2])
            np.add.at(counts, (i, j), 1)
        for e in g.itertuples(index=False):
            anchor = pd.Timestamp(e.anchor_time_utc).timestamp()
            end = anchor + 1200
            if e.observed_evaluable:
                for k, ent in enumerate((e.entity1, e.entity2)):
                    p = s21.interpolate(tracks.get(ent), np.array([anchor]), 30)[0]
                    if np.isfinite(p).all():
                        rows.append(dict(episode_id=e.episode_id, color=e.color, kind="evaluable_anchor",
                                         lat=p[0], lon=p[1], remaining_s=np.nan))
                continue
            if end >= archive_end or not e.start_observed:
                continue
            best = None
            for ent in (e.entity1, e.entity2):
                tr = tracks.get(ent)
                if tr is None or len(tr) == 0 or np.isfinite(s21.interpolate(tr, np.array([end]), 30)).all():
                    continue
                t = tr[:, 0]
                prior = np.searchsorted(t, end, side="right") - 1
                later = np.searchsorted(t, end, side="left")
                if prior < 0 or later < len(t):
                    continue  # bracketed by a gap or no prior state: not an in-band track end
                if not (29000 <= tr[prior, 3] <= 41000):
                    continue
                cand = (t[prior], tr[prior, 1], tr[prior, 2])
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is not None:
                rows.append(dict(episode_id=e.episode_id, color=e.color, kind="track_end_in_band",
                                 lat=best[1], lon=best[2], remaining_s=end - best[0]))
        del tracks
    d = pd.DataFrame(rows)
    res = {}
    for thr in (20, 100, 500):
        occ = counts >= thr
        i, j = cell(d.lat.values, d.lon.values)
        d[f"edge_nm_{thr}"] = edge_distance_cells(occ, i, j) * KM_PER_CELL * NM_PER_KM
    d.to_csv(OUT / "track_end_footprint.csv", index=False)
    for thr in (20, 100, 500):
        c = f"edge_nm_{thr}"
        print("\n=== threshold", thr, "rows/cell ===")
        for (kind, color), s in d.groupby(["kind", "color"]):
            x = s[c]
            print(f"{kind:18s} {color:9s} n={len(x):6d} median={x.median():6.1f} NM "
                  f"p25={x.quantile(.25):6.1f} p75={x.quantile(.75):6.1f} "
                  f"<=25NM={100*(x<=25).mean():5.1f}% <=50NM={100*(x<=50).mean():5.1f}% <=100NM={100*(x<=100).mean():5.1f}%")
    print("\nremaining seconds before endpoint (track ends):")
    print(d[d.kind == "track_end_in_band"].groupby("color").remaining_s.describe().to_string())


if __name__ == "__main__":
    main()

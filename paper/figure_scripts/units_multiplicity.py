#!/usr/bin/env python3
"""Clearance multiplicity by colour threshold and initial horizontal radius."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scat_analysis"))
import stage15_predicted_separation as s15  # noqa: E402
import stage16_pair_hazard as s16  # noqa: E402

OUT = ROOT / "results/benchmarks_units"
RADII = (350, 100, 50)
SCENARIOS = ("all", "non_green", "m_lt_1")
NS = 1_000_000_000


def sources() -> tuple[pd.DataFrame, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    attr = pd.read_csv(ROOT / "results/stage20/stage20_clearance_attribution.csv.gz")
    assert len(attr) == 51950 and attr.clearance_id.is_unique
    stage03 = pd.read_parquet(ROOT / "data/processed/scat/03_clearances/scat_clearances.parquet",
                              columns=["clearance_id", "clearance_epoch"])
    epoch = attr.clearance_id.map(stage03.set_index("clearance_id").clearance_epoch)
    assert epoch.notna().all()
    attr["epoch"] = epoch.to_numpy(float)
    indexes = {}
    for week, group in attr.groupby("week", sort=False):
        indexes[week] = {}
        keys = group.flight.astype(str).str.cat(group.source_member.astype(str), sep="\x1f")
        frame = pd.DataFrame({"key": keys, "epoch": group.epoch.to_numpy(float),
                              "row": group.index.to_numpy(np.int32)})
        for key, g in frame.groupby("key", sort=False):
            g = g.sort_values("epoch")
            indexes[week][key] = (g.epoch.to_numpy(float), g.row.to_numpy(np.int32))
    return attr, indexes


def pair_geometry(week: str, states_pf: pq.ParquetFile) -> pd.DataFrame:
    path = ROOT / "results/stage20/pairs" / f"pairs_{week}.parquet"
    columns = ["T", "flight_key_a", "flight_key_b", "source_member_a",
               "source_member_b", "m_min", "color"]
    pairs = pd.read_parquet(path, columns=columns)
    assert np.all(np.diff(pairs["T"].to_numpy(float)) >= 0)
    a = pairs.flight_key_a.astype(str).str.cat(pairs.source_member_a.astype(str), sep="\x1f")
    b = pairs.flight_key_b.astype(str).str.cat(pairs.source_member_b.astype(str), sep="\x1f")
    keys = pd.Index(pd.concat([a, b], ignore_index=True).unique())
    pairs["a"] = keys.get_indexer(a).astype(np.int32)
    pairs["b"] = keys.get_indexer(b).astype(np.int32)
    # Stage 16 selected the latest valid aircraft state at each snapshot.
    states, _ = s15.load_week_states_lean(states_pf, week)
    states["key"] = states.flight_key.astype(str).str.cat(states.source_member.astype(str), sep="\x1f")
    available = states[["time_epoch", "key", "latitude", "longitude"]].copy()
    del states
    left = pd.concat([pd.DataFrame({"T": pairs["T"], "key": a}),
                      pd.DataFrame({"T": pairs["T"], "key": b})], ignore_index=True).drop_duplicates()
    left.sort_values("T", kind="mergesort", inplace=True)
    available.sort_values("time_epoch", kind="mergesort", inplace=True)
    selected = pd.merge_asof(left, available, left_on="T", right_on="time_epoch", by="key",
                             direction="backward", tolerance=s15.MAX_AGE_S)
    assert selected.latitude.notna().all(), f"Missing selected state in {week}"
    px, py = s16.to_plane(selected.latitude.to_numpy(float), selected.longitude.to_numpy(float))
    index = pd.MultiIndex.from_frame(selected[["T", "key"]])
    t = pairs["T"].to_numpy(float)
    ia = index.get_indexer(pd.MultiIndex.from_arrays([t, a]))
    ib = index.get_indexer(pd.MultiIndex.from_arrays([t, b]))
    assert (ia >= 0).all() and (ib >= 0).all()
    radius = np.hypot(px[ia] - px[ib], py[ia] - py[ib])
    assert radius.max() <= 350 + 1e-7, (week, radius.max())
    pairs["radius_nm"] = radius
    pairs["key_a"] = a.to_numpy(str)
    pairs["key_b"] = b.to_numpy(str)
    return pairs[["T", "a", "b", "key_a", "key_b", "m_min", "color", "radius_nm"]]


def count_pair_links(pairs: pd.DataFrame, group: pd.DataFrame,
                     counts: dict, distinct: dict) -> None:
    t = pairs["T"].to_numpy(float)
    aa, bb = pairs.a.to_numpy(np.int32), pairs.b.to_numpy(np.int32)
    colors = pairs.color.to_numpy(str)
    m = pairs.m_min.to_numpy(float)
    radius = pairs.radius_nm.to_numpy(float)
    lookup = {k: code for code, k in enumerate(pd.Index(pd.concat([
        pairs.key_a, pairs.key_b], ignore_index=True).unique()))}
    # Pair codes above were assigned from concatenated endpoint keys in this order.
    for row in group.itertuples():
        key = f"{row.flight}\x1f{row.source_member}"
        code = lookup.get(key, -1)
        if code < 0:
            continue
        lo, hi = np.searchsorted(t, [row.epoch - 300, row.epoch], side="left")
        if lo == hi:
            continue
        source = (aa[lo:hi] == code) | (bb[lo:hi] == code)
        if not source.any():
            continue
        partner = np.where(aa[lo:hi] == code, bb[lo:hi], aa[lo:hi])
        for rad in RADII:
            spatial = radius[lo:hi] <= rad
            for scenario in SCENARIOS:
                mask = source & spatial
                if scenario == "non_green":
                    mask &= colors[lo:hi] != "verde"
                elif scenario == "m_lt_1":
                    mask &= m[lo:hi] < 1
                if mask.any():
                    counts[(scenario, rad)][row.Index] = mask.sum()
                    distinct[(scenario, rad)][row.Index] = len(np.unique(partner[mask]))


def episode_links(pairs: pd.DataFrame, week: str, rad: int,
                  event_index: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Filter observations by radius, then regroup non-green runs at 360 seconds."""
    obs = pairs.loc[pairs.color.ne("verde") & pairs.radius_nm.le(rad),
                    ["T", "key_a", "key_b", "color"]].copy()
    swap = obs.key_a.to_numpy(str) > obs.key_b.to_numpy(str)
    a, b = obs.key_a.to_numpy(str), obs.key_b.to_numpy(str)
    obs["entity1"], obs["entity2"] = np.where(swap, b, a), np.where(swap, a, b)
    obs["time_ns"] = np.rint(obs["T"].to_numpy(float) * NS).astype(np.int64)
    obs.sort_values(["entity1", "entity2", "time_ns"], kind="mergesort", inplace=True)
    cut = (obs.entity1.ne(obs.entity1.shift()) | obs.entity2.ne(obs.entity2.shift()) |
           obs.time_ns.diff().gt(360 * NS))
    obs["run"] = cut.cumsum()
    severity = {"amarillo": 1, "naranja": 2, "rojo": 3}
    obs["rank"] = obs.color.map(severity)
    out = []
    for _, g in obs.groupby("run", sort=False):
        tt = g["T"].to_numpy(float)
        start, end = tt[0], tt[-1]
        color = max(severity, key=lambda c: severity[c] if c in set(g.color) else -1)
        episode = int(g.run.iloc[0])
        linked = set()
        for key in (g.entity1.iloc[0], g.entity2.iloc[0]):
            entry = event_index.get(key)
            if entry is None:
                continue
            times, ids = entry
            lo = np.searchsorted(times, start - 600, side="left")
            hi = np.searchsorted(times, end + 300, side="right")
            for event_t, order in zip(times[lo:hi], ids[lo:hi]):
                prior = 0 < start - event_t <= 600
                j = np.searchsorted(tt, event_t, side="left") - 1
                after = event_t > start and j >= 0 and 0 < event_t - tt[j] <= 300
                if prior or after:
                    linked.add(int(order))
        for order in linked:
            out.append((order, f"{week}:{episode}", color))
    return pd.DataFrame(out, columns=["order_row", "episode", "color"])


def summarize(counts: dict, distinct: dict, n: int) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for rad in RADII:
            c, p = counts[(scenario, rad)], distinct[(scenario, rad)]
            active = c > 0
            rows.append(dict(scenario=scenario, radius_nm=rad, eligible_clearances=n,
                             linked_clearances=int(active.sum()), zero_links=int((~active).sum()),
                             pair_instant_links=int(c.sum()),
                             pair_instants_mean=float(c[active].mean()) if active.any() else np.nan,
                             pair_instants_median=float(np.median(c[active])) if active.any() else np.nan,
                             pair_instants_p90=float(np.quantile(c[active], .9)) if active.any() else np.nan,
                             distinct_pairs_mean=float(p[active].mean()) if active.any() else np.nan,
                             distinct_pairs_median=float(np.median(p[active])) if active.any() else np.nan,
                             distinct_pairs_p90=float(np.quantile(p[active], .9)) if active.any() else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    attr, indexes = sources()
    counts = {(s, r): np.zeros(len(attr), dtype=np.int32) for s in SCENARIOS for r in RADII}
    distinct = {(s, r): np.zeros(len(attr), dtype=np.int32) for s in SCENARIOS for r in RADII}
    links = {r: [] for r in RADII}
    states_pf = pq.ParquetFile(ROOT / "data/processed/scat/02_traffic/scat_states.parquet")
    for week, group in attr.groupby("week", sort=False):
        pairs = pair_geometry(week, states_pf)
        count_pair_links(pairs, group, counts, distinct)
        for rad in RADII:
            links[rad].append(episode_links(pairs, week, rad, indexes[week]))
        print(f"Multiplicity: {week}", flush=True)
    pair = summarize(counts, distinct, len(attr))
    assert np.array_equal(counts[("all", 350)], attr.matched_pair_instants.to_numpy(np.int32))
    assert np.array_equal(counts[("non_green", 350)],
                          attr[["matched_amarillo", "matched_naranja", "matched_rojo"]].sum(axis=1).to_numpy(np.int32))
    assert np.array_equal(counts[("m_lt_1", 350)],
                          attr[["matched_naranja", "matched_rojo"]].sum(axis=1).to_numpy(np.int32))
    base = pair.loc[pair.scenario.eq("all") & pair.radius_nm.eq(350)].iloc[0]
    assert (base.linked_clearances, base.zero_links, base.pair_instant_links,
            base.pair_instants_median, base.pair_instants_p90) == (47339, 4611, 3176638, 65, 108)
    episode_rows = []
    detail = attr[["clearance_id"]].copy()
    for scenario in SCENARIOS:
        for rad in RADII:
            detail[f"{scenario}_{rad}_pair_instants"] = counts[(scenario, rad)]
            detail[f"{scenario}_{rad}_distinct_pairs"] = distinct[(scenario, rad)]
    for rad in RADII:
        x = pd.concat(links[rad], ignore_index=True).drop_duplicates(["order_row", "episode"])
        if rad == 350:
            published = pd.read_csv(ROOT / "results/stage21/stage21_clearance_maneuvers.csv",
                                    usecols=["clearance_id", "episode_id"])
            expected = published.groupby("clearance_id").episode_id.nunique()
            actual = x.groupby("order_row").episode.nunique()
            actual.index = attr.clearance_id.iloc[actual.index].to_numpy()
            assert actual.sort_index().equals(expected.sort_index())
        by_order = x.groupby("order_row").agg(episodes=("episode", "nunique"),
                                               colors=("color", "nunique"))
        detail[f"non_green_{rad}_episodes"] = np.bincount(x.order_row.to_numpy(np.int32),
                                                              minlength=len(attr))
        color_count = np.zeros(len(attr), dtype=np.int8)
        color_count[by_order.index.to_numpy(np.int32)] = by_order.colors.to_numpy(np.int8)
        detail[f"non_green_{rad}_colors"] = color_count
        episode_rows.append(dict(radius_nm=rad, eligible_clearances=len(attr),
                                 linked_clearances=len(by_order), zero_links=len(attr)-len(by_order),
                                 episode_clearance_links=len(x),
                                 clearances_multiple_episodes=int(by_order.episodes.gt(1).sum()),
                                 share_multiple_episodes=float(by_order.episodes.gt(1).mean()),
                                 clearances_multiple_colors=int(by_order.colors.gt(1).sum())))
    episode = pd.DataFrame(episode_rows)
    row = episode.loc[episode.radius_nm.eq(350)].iloc[0]
    assert (row.linked_clearances, row.episode_clearance_links,
            row.clearances_multiple_episodes, row.clearances_multiple_colors) == (28324, 45463, 11474, 6383)
    OUT.mkdir(parents=True, exist_ok=True)
    pair.to_csv(OUT / "multiplicity_pairs.csv", index=False)
    episode.to_csv(OUT / "multiplicity_episodes.csv", index=False)
    detail.to_csv(OUT / "multiplicity_by_clearance.csv.gz", index=False, compression="gzip")
    print(pair.to_string(index=False))
    print(episode.to_string(index=False))
    print(f"Runtime: {time.perf_counter()-start:.1f} s", flush=True)


if __name__ == "__main__":
    main()

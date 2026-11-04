#!/usr/bin/env python3
"""Audit Stage 16 pair snapshots against the same eligible ATC orders.

The source of events is Stage 03, restricted by the frozen Stage 13/14 eligibility
used in Stage 16. Association is descriptive; an order is not a conflict label.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stage15_predicted_separation as s15
from scat_stage_io import (describe, flight_norm, header, input_files, load_mapping,
                           mapping_template, read_batches, resolve, save_json, to_ns)

COLORS = ("verde", "amarillo", "naranja", "rojo")
NULL = np.iinfo(np.int64).min


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--clearances", required=True)
    p.add_argument("--out", default="results/stage20")
    p.add_argument("--mapping")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--horizon-sec", type=int, default=300)
    p.add_argument("--lookback-sec", type=int, default=600)
    p.add_argument("--encounter-gap-sec", type=int, default=360)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if min(a.batch_size, a.horizon_sec, a.lookback_sec, a.encounter_gap_sec) <= 0:
        p.error("batch size and time windows must be positive")
    return a


def event_key(week, flight, member):
    return np.char.add(np.char.add(np.char.add(week.astype(str), "\x1f"),
                                   flight.astype(str)),
                       np.char.add("\x1f", member.astype(str)))


def load_events(files, mapping, batch_size):
    selected, selection_audit = s15.load_events()
    wanted = set(selected.clearance_id.astype(str))
    pieces = []
    raw_count = 0
    heading_level_count = 0
    for raw in read_batches(files, batch_size,
                            [mapping[x] for x in ("eventid", "time", "flight", "kind")]
                            + ["archive_week", "source_member"]):
        raw_count += len(raw)
        heading_level_count += int(raw[mapping["kind"]].astype(str).str.contains("HEADING|LEVEL").sum())
        r = raw.loc[raw[mapping["eventid"]].astype(str).isin(wanted)].copy()
        if r.empty:
            continue
        pieces.append(pd.DataFrame({
            "clearance_id": r[mapping["eventid"]].astype(str),
            "week": r.archive_week.astype(str),
            "flight": flight_norm(r[mapping["flight"]]),
            "source_member": r.source_member.astype(str),
            "time_ns": to_ns(r[mapping["time"]], "clearances.time"),
            "kind": r[mapping["kind"]].astype(str).str.upper(),
        }))
    if not pieces:
        raise ValueError("No Stage 16 eligible events found in Stage 03 clearances")
    events = pd.concat(pieces, ignore_index=True)
    if events.clearance_id.duplicated().any() or set(events.clearance_id) != wanted:
        raise ValueError("Stage 03 clearance IDs do not match the Stage 16 eligible cohort")
    expected = selected.set_index(selected.clearance_id.astype(str))
    aligned = expected.loc[events.clearance_id]
    if (events.week.to_numpy() != aligned.archive_week.astype(str).to_numpy()).any() or \
       (events.flight.to_numpy() != aligned.flight_key.astype(str).to_numpy()).any() or \
       (events.source_member.to_numpy() != aligned.source_member.astype(str).to_numpy()).any() or \
       (np.abs(events.time_ns.to_numpy() / 1e9 - aligned.clearance_epoch.to_numpy(float)) > 0.001).any():
        raise ValueError("Stage 03 event identity, flight, source member, week or time differs from Stage 16")
    if not events.kind.str.contains("HEADING|LEVEL").all():
        raise ValueError("Eligible cohort unexpectedly includes a speed-only order")
    events.sort_values(["week", "flight", "source_member", "time_ns", "clearance_id"],
                       kind="mergesort", inplace=True)
    events.reset_index(drop=True, inplace=True)
    events["event_num"] = np.arange(len(events), dtype=np.int32)
    events["key"] = event_key(events.week.to_numpy(), events.flight.to_numpy(),
                              events.source_member.to_numpy())
    return events, {"stage03_rows": raw_count,
                    "stage03_heading_or_level_unique_events": heading_level_count,
                    "eligible_unique_events": len(events),
                    "eligibility": "Stage 13/14 selection used by Stage 16",
                    "selection_audit": selection_audit, "deduplication": "unique clearance_id; no time-based collapse"}


def event_index(events):
    return {k: (g.time_ns.to_numpy(np.int64), g.event_num.to_numpy(np.int32))
            for k, g in events.groupby("key", sort=False)}


def nearest(keys, ns, index, seconds, future):
    ids = np.full(len(keys), -1, dtype=np.int32)
    lag = np.full(len(keys), np.nan, dtype=np.float32)
    for key, rows in pd.Series(keys).groupby(keys, sort=False).indices.items():
        if key not in index:
            continue
        t, e = index[key]
        q = ns[rows]
        pos = np.searchsorted(t, q, side="right" if future else "left")
        pos = pos if future else pos - 1
        ok = (pos >= 0) & (pos < len(t))
        rr = np.asarray(rows)[ok]
        pp = pos[ok]
        delta = (t[pp] - ns[rr] if future else ns[rr] - t[pp]) / 1e9
        good = (delta > 0) & (delta <= seconds)
        ids[rr[good]] = e[pp[good]]
        lag[rr[good]] = delta[good]
    return ids, lag


def count_all_events(keys, ns, colors, index, seconds, counts):
    """Count every eligible order in (T,T+W], even when several occur per flight."""
    for key, rows in pd.Series(keys).groupby(keys, sort=False).indices.items():
        if key not in index:
            continue
        t, ids = index[key]
        rr = np.asarray(rows)
        starts = np.searchsorted(t, ns[rr], side="right")
        ends = np.searchsorted(t, ns[rr] + seconds * 1_000_000_000, side="right")
        for ci, color in enumerate(COLORS):
            take = colors[rr] == color
            if not take.any():
                continue
            diff = np.bincount(starts[take], minlength=len(t) + 1) - \
                   np.bincount(ends[take], minlength=len(t) + 1)
            counts[ci, ids] += np.cumsum(diff)[:-1]


def color_for(raw):
    m = pd.to_numeric(raw.m_min, errors="coerce").to_numpy(float)
    tau = pd.to_numeric(raw.tau_sec, errors="coerce").to_numpy(float)
    if not np.isfinite(m).all() or not np.isfinite(tau).all():
        raise ValueError("Non-finite m_min/tau_sec in canonical pair export")
    color = np.full(len(raw), "verde", dtype="<U9")
    color[(m >= 1.2) & (m < 2)] = "amarillo"
    color[(m < 1) & (tau >= 300)] = "naranja"
    color[(m < 1) & (tau < 300)] = "rojo"
    if (raw.color.astype(str).to_numpy() != color).any():
        raise ValueError("Exported colors differ from frozen Stage 19 classification")
    return color


def build_episodes(non_green_path, out, gap, index, lookback):
    obs = pd.read_csv(non_green_path, compression="gzip", low_memory=False)
    if obs.empty:
        raise ValueError("No non-green observations; cannot construct episodes")
    obs.sort_values(["week", "entity1", "entity2", "time_ns"],
                    kind="mergesort", inplace=True)
    obs.reset_index(drop=True, inplace=True)
    cut = obs.week.ne(obs.week.shift()) | obs.entity1.ne(obs.entity1.shift()) | \
          obs.entity2.ne(obs.entity2.shift()) | (obs.time_ns.diff() > gap * 1e9)
    obs["episode_id"] = cut.cumsum().astype(np.int64)
    obs.to_csv(non_green_path, index=False, compression="gzip")
    g = obs.groupby("episode_id", sort=False)
    ep = g.agg(week=("week", "first"), split=("split", "first"),
               flight1=("flight1", "first"), flight2=("flight2", "first"),
               entity1=("entity1", "first"), entity2=("entity2", "first"),
               start_ns=("time_ns", "min"), end_ns=("time_ns", "max"),
               observations=("color", "size"),
               red_n=("color", lambda x: int((x == "rojo").sum())),
               orange_n=("color", lambda x: int((x == "naranja").sum())),
               yellow_n=("color", lambda x: int((x == "amarillo").sum())),
               with_next=("with_next", "sum"), with_previous=("with_previous", "sum")).reset_index()
    ep["highest_color"] = np.select([ep.red_n > 0, ep.orange_n > 0, ep.yellow_n > 0],
                                     ["rojo", "naranja", "amarillo"], default="verde")
    start = ep.start_ns.to_numpy(np.int64)
    p1, _ = nearest(ep.entity1.to_numpy(str), start, index, lookback, False)
    p2, _ = nearest(ep.entity2.to_numpy(str), start, index, lookback, False)
    ep["prior_to_episode_start"] = (p1 >= 0) | (p2 >= 0)
    ep["prior_event_a"], ep["prior_event_b"] = p1, p2
    ep["any_next_clearance"] = ep.with_next > 0
    ep["no_order_in_windows"] = ~ep.prior_to_episode_start & ~ep.any_next_clearance
    ep["start_utc"] = pd.to_datetime(ep.start_ns, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ep["end_utc"] = pd.to_datetime(ep.end_ns, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ep["duration_min"] = (ep.end_ns - ep.start_ns) / 60e9
    ep.drop(columns=["start_ns", "end_ns"]).to_csv(out / "stage20_episodes.csv", index=False)
    result = {}
    for split in ("calibration", "test", "all"):
        sub = ep if split == "all" else ep.loc[ep.split.eq(split)]
        result[split] = {c: {"encounters": int((sub.highest_color == c).sum()),
                             "with_prior_to_start": int((sub.highest_color.eq(c) & sub.prior_to_episode_start).sum()),
                             "with_any_next": int((sub.highest_color.eq(c) & sub.any_next_clearance).sum()),
                             "without_order_in_windows": int((sub.highest_color.eq(c) & sub.no_order_in_windows).sum())}
                         for c in COLORS[1:]}
    return result


def main():
    a = cli()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = load_mapping(a.mapping)
    pfiles, cfiles = input_files(a.pairs), input_files(a.clearances)
    if a.inspect:
        describe(a.pairs, "pairs", overrides)
        describe(a.clearances, "clearances", overrides)
        mapping_template(out / "mapping_example.json")
        return
    if (out / "stage20_summary.json").exists() and not a.force:
        raise FileExistsError("Stage 20 output already exists; use a new --out or explicit --force")
    pc = resolve("pairs", header(pfiles), overrides,
                 ["time", "flight1", "flight2", "mmin", "tau", "color", "existing_label"])
    cc = resolve("clearances", header(cfiles), overrides,
                 ["time", "flight", "kind", "eventid"])
    expected = {"time": "T", "flight1": "flight_key_a", "flight2": "flight_key_b",
                "mmin": "m_min", "tau": "tau_sec", "color": "color",
                "existing_label": "stage16_label_300"}
    if pc != expected or not {"week", "split", "source_member_a", "source_member_b"} <= set(header(pfiles)):
        raise ValueError("Stage 20 requires the canonical full Stage 16 pair export")
    if cc["time"] != "clearance_epoch" or cc["eventid"] != "clearance_id" or \
       not {"archive_week", "source_member"} <= set(header(cfiles)):
        raise ValueError("Stage 20 requires the original Stage 03 clearance table")
    audit = json.loads((s15.BASE / "19_calib_selection/tables/audit.json").read_text())
    test_weeks = set(audit["test"])
    events, event_audit = load_events(cfiles, cc, a.batch_size)
    index = event_index(events)
    by_event = np.zeros((len(COLORS), len(events)), dtype=np.int64)
    split_counts = {s: np.zeros((len(COLORS), 4), dtype=np.int64)
                    for s in ("calibration", "test")}
    unique_pairs = {s: {c: set() for c in COLORS} for s in ("calibration", "test")}
    non_green = out / "stage20_non_green_observations.csv.gz"
    near_cols = ["week", "split", "time_ns", "time_utc", "flight1", "flight2",
                 "entity1", "entity2", "color", "mmin", "tau_min", "next_event_a",
                 "next_event_b", "prev_event_a", "prev_event_b", "next_lag_s_a",
                 "next_lag_s_b", "with_next", "with_previous"]
    total = 0
    label_disagreements = 0
    weeks_seen = set()
    first_write = True
    with gzip.open(non_green, "wt", encoding="utf-8", newline="") as f:
        for raw in read_batches(pfiles, a.batch_size, header(pfiles)):
            if a.max_pairs and total >= a.max_pairs:
                break
            if a.max_pairs:
                raw = raw.iloc[:a.max_pairs - total].copy()
            total += len(raw)
            if raw.empty:
                continue
            weeks = raw.week.astype(str).to_numpy()
            if not set(weeks) <= set(audit["weeks"]):
                raise ValueError("Pair rows contain weeks absent from Stage 19")
            splits = np.where(np.isin(weeks, list(test_weeks)), "test", "calibration")
            if (raw.split.astype(str).to_numpy() != splits).any():
                raise ValueError("Pair split differs from Stage 19 frozen split")
            weeks_seen.update(weeks)
            ns = to_ns(raw["T"], "pairs.T")
            if (ns == NULL).any():
                raise ValueError("Invalid UTC timestamp in pair export")
            color = color_for(raw)
            f1, f2 = flight_norm(raw.flight_key_a).to_numpy(str), flight_norm(raw.flight_key_b).to_numpy(str)
            m1, m2 = raw.source_member_a.astype(str).to_numpy(), raw.source_member_b.astype(str).to_numpy()
            if (f1 == "").any() or (f2 == "").any():
                raise ValueError("Missing flight key in pair export")
            k1, k2 = event_key(weeks, f1, m1), event_key(weeks, f2, m2)
            if (k1 == k2).any():
                raise ValueError("Self-pair in pair export")
            na, da = nearest(k1, ns, index, a.horizon_sec, True)
            nb, db = nearest(k2, ns, index, a.horizon_sec, True)
            pa, _ = nearest(k1, ns, index, a.lookback_sec, False)
            pb, _ = nearest(k2, ns, index, a.lookback_sec, False)
            yes, previous = (na >= 0) | (nb >= 0), (pa >= 0) | (pb >= 0)
            label_disagreements += int((yes != raw.stage16_label_300.to_numpy(bool)).sum())
            count_all_events(k1, ns, color, index, a.horizon_sec, by_event)
            count_all_events(k2, ns, color, index, a.horizon_sec, by_event)
            swap = k1 > k2
            left, right = np.where(swap, k2, k1), np.where(swap, k1, k2)
            uniq = pd.DataFrame({"split": splits, "color": color, "left": left, "right": right}).drop_duplicates()
            for s, c, l, r in uniq.itertuples(index=False, name=None):
                unique_pairs[s][c].add((l, r))
            for s in ("calibration", "test"):
                sm = splits == s
                for ci, c in enumerate(COLORS):
                    mask = sm & (color == c)
                    split_counts[s][ci] += (int(mask.sum()), int((mask & yes).sum()),
                                            int((mask & previous).sum()), int((mask & yes & previous).sum()))
            ix = color != "verde"
            if ix.any():
                near = pd.DataFrame({"week": weeks[ix], "split": splits[ix],
                    "time_ns": ns[ix],
                    "time_utc": pd.to_datetime(ns[ix], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "flight1": np.where(swap, f2, f1)[ix],
                    "flight2": np.where(swap, f1, f2)[ix],
                    "entity1": left[ix], "entity2": right[ix],
                    "color": color[ix], "mmin": raw.m_min.to_numpy()[ix],
                    "tau_min": raw.tau_sec.to_numpy()[ix] / 60,
                    "next_event_a": np.where(swap, nb, na)[ix],
                    "next_event_b": np.where(swap, na, nb)[ix],
                    "prev_event_a": np.where(swap, pb, pa)[ix],
                    "prev_event_b": np.where(swap, pa, pb)[ix],
                    "next_lag_s_a": np.where(swap, db, da)[ix],
                    "next_lag_s_b": np.where(swap, da, db)[ix],
                    "with_next": yes[ix].astype(int), "with_previous": previous[ix].astype(int)})[near_cols]
                near.to_csv(f, index=False, header=first_write)
                first_write = False
            if total % 1_000_000 < a.batch_size:
                print(f"Processed {total:,} pair snapshots", flush=True)
    if first_write:
        raise ValueError("No non-green pairs found")
    ep_summary = build_episodes(non_green, out, a.encounter_gap_sec, index, a.lookback_sec)
    rows = []
    for split in ("calibration", "test", "all"):
        for ci, c in enumerate(COLORS):
            values = (split_counts["calibration"][ci] + split_counts["test"][ci]
                      if split == "all" else split_counts[split][ci])
            pairs = (unique_pairs["calibration"][c] | unique_pairs["test"][c]
                     if split == "all" else unique_pairs[split][c])
            mask = np.ones(len(events), dtype=bool) if split == "all" else \
                   events.week.isin(test_weeks).to_numpy() == (split == "test")
            rows.append({"split": split, "color": c,
                         "pair_instants": int(values[0]), "unique_flight_pairs": len(pairs),
                         "with_next_clearance": int(values[1]),
                         "next_clearance_rate": float(values[1] / values[0]) if values[0] else None,
                         "with_prior_clearance": int(values[2]),
                         "with_both_prior_and_next": int(values[3]),
                         "distinct_next_clearances_attributed": int(np.count_nonzero(by_event[ci, mask]))})
    pd.DataFrame(rows).to_csv(out / "stage20_by_color.csv", index=False)
    attribution = events[["event_num", "clearance_id", "week", "flight", "source_member",
                          "time_ns", "kind"]].copy()
    attribution["split"] = np.where(attribution.week.isin(test_weeks), "test", "calibration")
    attribution["time_utc"] = pd.to_datetime(attribution.time_ns, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for ci, c in enumerate(COLORS):
        attribution["matched_" + c] = by_event[ci]
    attribution["matched_pair_instants"] = by_event.sum(axis=0)
    attribution.drop(columns="time_ns").to_csv(
        out / "stage20_clearance_attribution.csv.gz", index=False, compression="gzip")
    summary = {"stage": 20, "pairs_input": str(a.pairs), "clearances_input": str(a.clearances),
               "pair_mapping": pc, "clearance_mapping": cc, "pairs_read": total,
               "weeks_seen": sorted(weeks_seen), "nominal_days_expected": 91,
               "calibration_weeks": audit["calib"], "test_weeks": audit["test"],
               "event_cohort": event_audit, "by_color": rows, "encounters": ep_summary,
               "stage16_label_300_disagreements": label_disagreements,
               "next_window_seconds": a.horizon_sec, "previous_window_seconds": a.lookback_sec,
               "encounter_gap_sec": a.encounter_gap_sec, "partial_run": bool(a.max_pairs),
               "interpretation": {"observed": "clearance records and surveillance states",
                                  "predicted": "m_min, tau_sec and frozen Stage 19 color",
                                  "heuristic": "temporal clearance attribution and contiguous-episode grouping"},
               "limitations": ["An order may concern any of several pairs or a non-conflict task.",
                               "An associated order is not evidence of conflict resolution.",
                               "Non-green episodes depend on the 360-second gap rule; green episodes are not constructed."]}
    save_json(out / "stage20_summary.json", summary)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Stage 16 label disagreements:", label_disagreements)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR STAGE 20: {exc}", file=sys.stderr)
        raise

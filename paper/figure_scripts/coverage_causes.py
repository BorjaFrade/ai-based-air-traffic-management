#!/usr/bin/env python3
"""Classify observed evidence for unevaluable Stage 21 follow-up.

This does not infer an airspace exit without a supplied geographic boundary.
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


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def missing_end_reason(track, end):
    if track is None or len(track) == 0:
        return "no_usable_individual_track"
    t = track[:, 0]
    at_end = s21.interpolate(track, np.array([end], float), 30)
    if np.isfinite(at_end).all():
        return None
    prior = np.searchsorted(t, end, side="right") - 1
    later = np.searchsorted(t, end, side="left")
    if prior >= 0 and later < len(t) and t[later] > end:
        return "surveillance_gap_bracketing_endpoint"
    if prior >= 0:
        last_alt = track[prior, 3]
        if last_alt < 29000 or last_alt > 41000:
            return "last_observed_outside_FL290_410"
        return "track_ends_before_endpoint_in_band"
    return "no_state_before_endpoint"


def main():
    ep = pd.read_csv(ROOT / "results/stage20/stage20_episodes.csv",
                     usecols=["episode_id", "week", "entity1", "entity2"])
    v = pd.read_csv(ROOT / "results/stage21/stage21_encounters.csv",
                    usecols=["episode_id", "week", "color", "anchor_time_utc",
                             "observed_evaluable", "start_observed", "end_observed"])
    check(len(v) == 54441 and v.episode_id.is_unique, "Stage 21 episode keys changed")
    v = v.loc[~v.observed_evaluable].merge(ep, on=["episode_id", "week"], validate="one_to_one")
    check(len(v) == 26990, "Expected 26,990 unevaluable non-green episodes")
    pf = pq.ParquetFile(TRACKS)
    rows = []
    for week, g in v.groupby("week", sort=False):
        keys = set(g.entity1) | set(g.entity2)
        tracks, audit = s21.load_tracks_week(pf, week, keys, "ft")
        print(week, audit, flush=True)
        archive_end = pd.Timestamp(week.split("_")[1], tz="UTC").timestamp() + 86400
        for e in g.itertuples(index=False):
            end = pd.Timestamp(e.anchor_time_utc).timestamp() + 1200
            if end >= archive_end:
                reason = "nominal_archive_boundary"
            elif not e.start_observed:
                reason = "missing_start_state"
            else:
                a = missing_end_reason(tracks.get(e.entity1), end)
                b = missing_end_reason(tracks.get(e.entity2), end)
                kinds = {a, b} - {None}
                if not kinds:
                    reason = "interior_grid_gap_or_other"
                elif "surveillance_gap_bracketing_endpoint" in kinds:
                    reason = "surveillance_gap_bracketing_endpoint"
                elif "last_observed_outside_FL290_410" in kinds:
                    reason = "last_observed_outside_FL290_410"
                elif "no_usable_individual_track" in kinds:
                    reason = "no_usable_individual_track"
                elif "track_ends_before_endpoint_in_band" in kinds:
                    reason = "track_ends_before_endpoint_in_band"
                else:
                    reason = "no_state_before_endpoint"
            rows.append({"episode_id": e.episode_id, "week": week,
                         "color": e.color, "evidence_category": reason})
        del tracks
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    check(len(detail) == 26990 and detail.episode_id.is_unique,
          "Coverage evidence does not partition unevaluable episodes")
    detail.to_csv(OUT / "coverage_evidence_detail.csv", index=False)
    summary = detail.groupby(["color", "evidence_category"]).size().rename("episodes").reset_index()
    summary.to_csv(OUT / "coverage_evidence_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

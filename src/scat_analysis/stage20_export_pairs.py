#!/usr/bin/env python3
"""Replay Stage 16 on processed SCAT inputs and export every R pair snapshot.

Outputs are separate from Stages 1–19. The exported population is restricted to
the 91 nominal days used by Stage 19; Stage 16's border-day inputs are retained
for prediction continuity but are not counted as study observations.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import stage16_pair_hazard as s16

TEST_WEEKS = {
    "20170107_20170113", "20170401_20170407",
    "20170624_20170630", "20170916_20170922",
}
COLUMNS = ["week", "split", "T", "flight_key_a", "flight_key_b",
           "source_member_a", "source_member_b", "m_min", "tau_sec",
           "color", "stage16_label_300"]


class PairWriter:
    def __init__(self, path: Path, week: str, batch_rows: int):
        self.path, self.week, self.batch_rows = path, week, batch_rows
        self.start = pd.Timestamp(week[:8], tz="UTC").timestamp()
        self.end = pd.Timestamp(week[-8:], tz="UTC").timestamp() + 86400
        self.parts: list[pd.DataFrame] = []
        self.buffered = 0
        self.count = 0
        self.colors = {c: 0 for c in ("verde", "amarillo", "naranja", "rojo")}
        self.labeled = {c: 0 for c in self.colors}
        self.writer: pq.ParquetWriter | None = None

    def __call__(self, week, T, states, a, b, features, labels):
        if not (self.start <= T < self.end) or not len(a):
            return
        m = np.asarray(features["m_min"], dtype="float64")
        tau = np.asarray(features["tau"], dtype="float64")
        if not np.isfinite(m).all() or not np.isfinite(tau).all():
            raise ValueError(f"Non-finite pair geometry in {week} at {T}")
        color = np.full(len(a), "verde", dtype="<U9")
        color[(m >= 1.2) & (m < 2)] = "amarillo"
        color[(m < 1) & (tau >= 300)] = "naranja"
        color[(m < 1) & (tau < 300)] = "rojo"
        f = pd.DataFrame({
            "week": week,
            "split": "test" if week in TEST_WEEKS else "calibration",
            "T": float(T),
            "flight_key_a": states.flight_key.iloc[a].astype(str).to_numpy(),
            "flight_key_b": states.flight_key.iloc[b].astype(str).to_numpy(),
            "source_member_a": states.source_member.iloc[a].astype(str).to_numpy(),
            "source_member_b": states.source_member.iloc[b].astype(str).to_numpy(),
            "m_min": m, "tau_sec": tau, "color": color,
            "stage16_label_300": np.asarray(labels, dtype=bool),
        })[COLUMNS]
        self.parts.append(f)
        self.buffered += len(f)
        self.count += len(f)
        for c in self.colors:
            ix = color == c
            self.colors[c] += int(ix.sum())
            self.labeled[c] += int(np.asarray(labels, dtype=bool)[ix].sum())
        if self.buffered >= self.batch_rows:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        table = pa.Table.from_pandas(pd.concat(self.parts, ignore_index=True), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.parts.clear()
        self.buffered = 0

    def close(self):
        self.flush()
        if self.writer is None:
            raise ValueError(f"No nominal-day pairs found for {self.week}")
        self.writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage20/pairs"))
    ap.add_argument("--weeks", default="all", help="Comma-separated weeks or all")
    ap.add_argument("--batch-rows", type=int, default=100000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.batch_rows <= 0:
        ap.error("--batch-rows must be positive")
    audit = json.loads((s16.BASE / "19_calib_selection/tables/audit.json").read_text())
    all_weeks = audit["weeks"]
    if set(audit["test"]) != TEST_WEEKS:
        raise ValueError("Stage 19 test-week definition differs from the frozen split")
    weeks = all_weeks if args.weeks == "all" else args.weeks.split(",")
    if not set(weeks) <= set(all_weeks):
        ap.error("--weeks includes a week absent from Stage 19")
    args.out.mkdir(parents=True, exist_ok=True)
    rules = s16.s15.load_rules()
    states_pf = pq.ParquetFile(s16.STATES)
    pred_pf = pq.ParquetFile(s16.PREDICTIONS)
    events, _ = s16.s15.load_events()
    replay = argparse.Namespace(snapshot_step_s=120.0, horizon_s=1200.0, grid_s=10.0,
                                pair_radius_nm=350.0, keep_below=12.0,
                                max_pred_age_s=900.0, sample_below=-1.0,
                                sample_rate=0.0, val_per_snapshot=0,
                                seed=16, max_snapshots=0)
    for week in weeks:
        path = args.out / f"pairs_{week}.parquet"
        manifest = args.out / f"pairs_{week}.json"
        if path.exists() or manifest.exists():
            if path.exists() and manifest.exists() and not args.force:
                print("CACHED", week, flush=True)
                continue
            if not args.force:
                raise FileExistsError(f"Incomplete or existing export for {week}; inspect before --force")
        t0 = time.monotonic()
        states, source_audit = s16.s15.load_week_states_lean(states_pf, week)
        writer = PairWriter(path, week, args.batch_rows)
        s16.analyse_week(week, states, pred_pf, rules, replay, events, pair_sink=writer)
        writer.close()
        meta = {"week": week, "split": "test" if week in TEST_WEEKS else "calibration",
                "pairs": writer.count, "by_color": writer.colors,
                "stage16_labeled_300_by_color": writer.labeled,
                "source": {"states": str(s16.STATES), "predictions": str(s16.PREDICTIONS),
                           "stage16_script": "stage16_pair_hazard.py",
                           "stage19_audit": str(s16.BASE / "19_calib_selection/tables/audit.json")},
                "source_audit": source_audit,
                "parameters": vars(replay), "created_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - t0, 1)}
        manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(week, json.dumps({"pairs": writer.count, "colors": writer.colors,
                                "elapsed_seconds": meta["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

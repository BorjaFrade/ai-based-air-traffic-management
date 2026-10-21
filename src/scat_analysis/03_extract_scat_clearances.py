#!/usr/bin/env python3
"""Extract time-varying SCAT controller clearances as ATC interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    action_fields,
    archive_paths,
    attach_context,
    build_context,
    clearance_type,
    compact_json,
    extract_traffic_rows,
    finite_number,
    flight_key,
    iter_flights,
    numeric_delta,
    normalise_for_compare,
    parse_timestamp,
    update_read_me_first,
)


STATE_MATCH_COLUMNS = [
    "flight_key",
    "timestamp",
    "time_epoch",
    "callsign",
    "latitude",
    "longitude",
    "altitude_ft",
    "flight_level",
    "ground_speed_kt",
    "track_deg",
    "heading_deg",
    "vertical_rate_fpm",
    "flight_phase",
    "active_centre_id",
]
MATCH_NUMERIC_COLUMNS = {
    "time_epoch", "latitude", "longitude", "altitude_ft", "flight_level",
    "ground_speed_kt", "track_deg", "heading_deg", "vertical_rate_fpm",
}
CLEARANCE_NUMERIC_COLUMNS = MATCH_NUMERIC_COLUMNS | {
    "clearance_record_index", "clearance_epoch", "clearance_cfl", "clearance_heading_deg",
    "clearance_speed", "matched_state_delta_s", "surveillance_match_age_s",
    "time_to_previous_clearance_s", "time_to_next_clearance_s", "ATC_INTERVENTION",
}


def normalise_clearance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make mixed JSON scalar types deterministic before Parquet writing."""

    frame = frame.copy()
    for column in frame.columns:
        if column in CLEARANCE_NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif column == "surveillance_match_valid":
            frame[column] = frame[column].fillna(False).astype(bool)
        else:
            frame[column] = frame[column].map(
                lambda value: None if pd.isna(value) else str(value)
            )
    return frame


def actual_changed_fields(previous: dict, current: dict) -> list[str]:
    keys = set(previous) | set(current)
    changed = []
    for key in sorted(keys):
        if key == "time_stamp":
            continue
        old = previous.get(key)
        new = current.get(key)
        if normalise_for_compare(old) != normalise_for_compare(new):
            changed.append(key)
    return changed


def event_rows_for_flight(archive: Path, member: str, flight: dict) -> list[dict]:
    key = flight_key(archive, flight, member)
    context = build_context(flight)
    records = sorted(
        [item for item in (flight.get("fpl", {}).get("fpl_clearance") or []) if isinstance(item, dict)],
        key=lambda item: parse_timestamp(item.get("time_stamp")) or float("inf"),
    )
    events = []
    previous = {}
    for record_index, record in enumerate(records):
        event_epoch = parse_timestamp(record.get("time_stamp"))
        if event_epoch is None:
            previous = record
            continue
        present = action_fields(record)
        changed = actual_changed_fields(previous, record)
        # TopSky stores empty initial snapshots in fpl_clearance.  They are
        # state records, not controller interventions.  Repeated unchanged
        # snapshots are also not new interventions.
        if not present or not changed:
            previous = record
            continue
        previous_values = {field: previous.get(field) for field in changed}
        new_values = {field: record.get(field) for field in changed}
        delta_values = {
            field: numeric_delta(previous.get(field), record.get(field))
            for field in changed
        }
        context_row = attach_context(context, event_epoch)
        event_id = f"{key}:{record_index}:{int(round(event_epoch))}"
        row = {
            "clearance_id": event_id,
            "flight_key": key,
            "flight_id": str(flight.get("id", Path(member).stem)),
            "archive_week": archive.stem.removeprefix("scat"),
            "source_archive": archive.name,
            "source_member": member,
            "clearance_record_index": record_index,
            "clearance_timestamp": record.get("time_stamp", ""),
            "clearance_epoch": event_epoch,
            "callsign": context_row.get("callsign", ""),
            "clearance_type": clearance_type(changed),
            "changed_fields": compact_json(changed),
            "previous_value": compact_json(previous_values),
            "new_value": compact_json(new_values),
            "delta_value": compact_json(delta_values),
            "clearance_record_json": compact_json(record),
            "previous_record_json": compact_json(previous),
            "clearance_cfl": finite_number(record.get("cfl")),
            "clearance_cfl_unit": record.get("cfl_unit", ""),
            "clearance_heading_deg": finite_number(record.get("assigned_heading_val")),
            "clearance_heading_beacon": record.get("assign_heading_beacon", ""),
            "clearance_speed": finite_number(record.get("assigned_speed_val")),
            "clearance_speed_unit": record.get("assigned_speed_unit", ""),
            "active_centre_id": context_row.get("active_centre_id", ""),
            "ATC_INTERVENTION": 1,
            "target_name": "ATC_INTERVENTION",
        }
        # Keep the flight-plan context that was active when the clearance was
        # recorded, but do not mistake it for a conflict label.
        for field in (
            "fpl_adep", "fpl_ades", "fpl_aircraft_type", "fpl_wtc",
            "fpl_flight_rules", "fpl_route", "fpl_rfl", "fpl_tas",
            "fpl_copn", "fpl_copn_pel", "fpl_copn_pel_unit", "fpl_copx",
            "fpl_copx_pel", "fpl_copx_pel_unit", "predicted_next_fix",
            "predicted_next_fix_eta",
        ):
            row[field] = context_row.get(field, "")
        events.append(row)
        previous = record

    events.sort(key=lambda item: item["clearance_epoch"])
    for index, row in enumerate(events):
        previous_epoch = events[index - 1]["clearance_epoch"] if index else np.nan
        next_epoch = events[index + 1]["clearance_epoch"] if index + 1 < len(events) else np.nan
        row["time_to_previous_clearance_s"] = (
            row["clearance_epoch"] - previous_epoch if np.isfinite(previous_epoch) else np.nan
        )
        row["time_to_next_clearance_s"] = (
            next_epoch - row["clearance_epoch"] if np.isfinite(next_epoch) else np.nan
        )
    return events


def match_surveillance(
    clearances: pd.DataFrame,
    states_path: Path,
    max_surveillance_age_s: float = 10.0,
) -> pd.DataFrame:
    if not states_path.exists():
        raise FileNotFoundError(
            f"Missing native-resolution states file: {states_path}. "
            "Run 02_extract_scat_traffic.py first."
        )
    available = pd.read_parquet(states_path, columns=STATE_MATCH_COLUMNS)
    available["time_epoch"] = pd.to_numeric(available["time_epoch"], errors="coerce")
    available = available.dropna(subset=["time_epoch", "flight_key"]).copy()
    available.sort_values(["flight_key", "time_epoch"], inplace=True, kind="mergesort")
    grouped = {key: group.reset_index(drop=True) for key, group in available.groupby("flight_key", sort=False)}
    matched_rows = []
    for row in clearances.itertuples(index=False):
        group = grouped.get(row.flight_key)
        match = None
        if group is not None and not group.empty:
            times = group["time_epoch"].to_numpy(dtype=float)
            index = int(np.searchsorted(times, row.clearance_epoch, side="right")) - 1
            if index >= 0:
                match = group.iloc[index]
        matched = {
            "matched_state_timestamp": match.get("timestamp", "") if match is not None else "",
            "matched_state_epoch": match.get("time_epoch", np.nan) if match is not None else np.nan,
            "matched_state_delta_s": (
                match.get("time_epoch", np.nan) - row.clearance_epoch
                if match is not None else np.nan
            ),
            "surveillance_match_age_s": (
                row.clearance_epoch - match.get("time_epoch", np.nan)
                if match is not None else np.nan
            ),
            "surveillance_match_valid": bool(
                match is not None
                and row.clearance_epoch - match.get("time_epoch", np.nan) <= max_surveillance_age_s
            ),
            "surveillance_match_method": "same_flight_previous_plot",
        }
        if match is not None:
            assert matched["surveillance_match_age_s"] >= 0
        for field in STATE_MATCH_COLUMNS:
            if field in {"flight_key", "timestamp", "time_epoch"}:
                continue
            default = np.nan if field in MATCH_NUMERIC_COLUMNS else ""
            matched[f"matched_{field}"] = match.get(field, default) if match is not None else default
        matched_rows.append(matched)
    return pd.concat([clearances.reset_index(drop=True), pd.DataFrame(matched_rows)], axis=1)


def match_events_to_raw_flight(
    events: list[dict],
    archive: Path,
    member: str,
    flight: dict,
    max_surveillance_age_s: float,
) -> list[dict]:
    """Attach only the previous native surveillance state from the same JSON.

    Reading the same flight JSON here avoids loading the complete 70M-row
    traffic Parquet into memory merely to perform an as-of join.
    """

    if not events:
        return events
    states, _ = extract_traffic_rows(archive, member, flight)
    states.sort(key=lambda item: float(item["time_epoch"]))
    times = np.asarray([float(item["time_epoch"]) for item in states], dtype=float)
    for event in events:
        index = int(np.searchsorted(times, event["clearance_epoch"], side="right")) - 1
        match = states[index] if index >= 0 else None
        age = (
            float(event["clearance_epoch"] - match["time_epoch"])
            if match is not None else np.nan
        )
        if match is not None:
            assert match["time_epoch"] <= event["clearance_epoch"]
        event["matched_state_timestamp"] = match.get("timestamp", "") if match else ""
        event["matched_state_epoch"] = match.get("time_epoch", np.nan) if match else np.nan
        event["matched_state_delta_s"] = -age if match is not None else np.nan
        event["surveillance_match_age_s"] = age
        event["surveillance_match_valid"] = bool(match is not None and age <= max_surveillance_age_s)
        event["surveillance_match_method"] = "same_flight_previous_plot"
        for field in STATE_MATCH_COLUMNS:
            if field in {"flight_key", "timestamp", "time_epoch"}:
                continue
            default = np.nan if field in MATCH_NUMERIC_COLUMNS else ""
            event[f"matched_{field}"] = match.get(field, default) if match else default
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--states-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "02_traffic" / "scat_states.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "03_clearances")
    parser.add_argument("--weeks", nargs="*", help="Optional ZIP names or week tokens")
    parser.add_argument("--max-flights-per-archive", type=int, default=0)
    parser.add_argument(
        "--max-surveillance-age-s",
        type=float,
        default=10.0,
        help="Maximum age of the previous same-flight surveillance state (seconds).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-archive clearance checkpoints under output-root/checkpoints.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "scat_clearances.parquet"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}; use --overwrite")
    if output_path.exists():
        output_path.unlink()
    if args.max_surveillance_age_s < 0:
        raise SystemExit("--max-surveillance-age-s must be non-negative")

    archives = archive_paths(args.input_root, args.weeks)
    rows = []
    archive_stats = []
    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for archive_index, archive in enumerate(archives, start=1):
        checkpoint_path = checkpoint_dir / f"{archive.stem}_clearances.parquet"
        if args.resume and checkpoint_path.exists():
            checkpoint = pd.read_parquet(checkpoint_path)
            rows.extend(checkpoint.to_dict(orient="records"))
            archive_stats.append({
                "archive": archive.name,
                "flights_scanned": np.nan,
                "interventions": len(checkpoint),
                "status": "resumed_checkpoint",
            })
            print(f"[{archive_index}/{len(archives)}] Reusing {checkpoint_path.name}", flush=True)
            continue
        print(f"[{archive_index}/{len(archives)}] Extracting clearances from {archive.name}", flush=True)
        before = len(rows)
        flights = 0
        for current_archive, member, flight in iter_flights([archive], args.max_flights_per_archive):
            flights += 1
            events = event_rows_for_flight(current_archive, member, flight)
            rows.extend(
                match_events_to_raw_flight(
                    events,
                    current_archive,
                    member,
                    flight,
                    args.max_surveillance_age_s,
                )
            )
        archive_stats.append(
            {"archive": archive.name, "flights_scanned": flights, "interventions": len(rows) - before, "status": "extracted"}
        )
        archive_frame = normalise_clearance_frame(pd.DataFrame(rows[before:]))
        archive_frame.to_parquet(checkpoint_path, index=False, compression="zstd")
    if not rows:
        raise RuntimeError("No actionable fpl_clearance records were found in the selected input")
    clearances = pd.DataFrame(rows)
    clearances.sort_values(["clearance_epoch", "flight_key"], inplace=True, kind="mergesort")
    clearances = normalise_clearance_frame(clearances)
    clearances.to_parquet(output_path, index=False, compression="zstd")
    pd.DataFrame(archive_stats).to_csv(args.output_root / "archive_clearance_stats.csv", index=False)
    config = {
        "input_root": str(args.input_root),
        "states_path": str(args.states_path),
        "archives": [path.name for path in archives],
        "max_flights_per_archive": args.max_flights_per_archive,
        "max_surveillance_age_s": args.max_surveillance_age_s,
        "resume": args.resume,
        "checkpoint_dir": str(checkpoint_dir),
        "matching_source": "same-flight raw JSON surveillance plots; predecessor-only as-of join",
        "output": str(output_path),
        "rows": len(clearances),
        "target": "ATC_INTERVENTION",
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    type_counts = clearances["clearance_type"].value_counts().rename_axis("clearance_type").reset_index(name="count")
    type_counts.to_csv(args.output_root / "clearance_type_counts.csv", index=False)
    valid = clearances["surveillance_match_valid"].astype(bool)
    valid_type_counts = (
        clearances.loc[valid, "clearance_type"]
        .value_counts()
        .rename_axis("clearance_type")
        .reset_index(name="valid_match_count")
    )
    type_counts = type_counts.merge(valid_type_counts, on="clearance_type", how="left").fillna({"valid_match_count": 0})
    type_counts.to_csv(args.output_root / "clearance_type_counts.csv", index=False)
    pd.DataFrame(
        [{
            "actionable_clearance_events": len(clearances),
            "flights_with_clearances": int(clearances["flight_key"].nunique()),
            "valid_surveillance_matches": int(valid.sum()),
            "rejected_surveillance_matches": int((~valid).sum()),
            "max_surveillance_age_s": args.max_surveillance_age_s,
        }]
    ).to_csv(args.output_root / "surveillance_match_quality.csv", index=False)
    report = [
        "# SCAT clearance extraction",
        "",
        f"- Actionable clearance/intervention rows: **{len(clearances)}**",
        f"- Flights with actionable clearances: **{clearances['flight_key'].nunique()}**",
        f"- Valid previous-surveillance matches: **{int(valid.sum())}**",
        f"- Rejected by match-age rule: **{int((~valid).sum())}**",
        f"- Maximum surveillance age: **{args.max_surveillance_age_s:g} s**",
        f"- Source: `{output_path}`",
        "- `ATC_INTERVENTION` is set to 1 for these rows.",
        "- Empty initial clearance snapshots and unchanged repeated snapshots are not counted as new interventions.",
        "- Each row is matched only to the previous native surveillance state of the same flight; future surveillance is never used.",
        "- Rows failing the maximum-age rule are retained for audit but must not be used to construct precise traffic-state windows.",
        "",
        "This stage does not infer conflict, interaction, safety or severity. A clearance is an operational intervention and may have many non-conflict causes.",
        "",
        "## Clearance types",
        "",
    ]
    report.extend(f"- `{row.clearance_type}`: {row.count}" for row in type_counts.itertuples(index=False))
    (args.output_root / "clearance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Clearance extraction complete: {args.output_root}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Audit the real SCAT JSON structure before analytical assumptions are made."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import zipfile
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    SPECIAL_MEMBERS,
    archive_paths,
    flight_members,
    parse_timestamp,
    update_read_me_first,
)


def type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def shorten(value, limit: int = 160):
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def walk_schema(value, path: str, registry: dict[str, dict]) -> None:
    entry = registry.setdefault(
        path,
        {"objects_seen": 0, "non_null": 0, "types": set(), "examples": []},
    )
    entry["objects_seen"] += 1
    entry["types"].add(type_name(value))
    if value is not None:
        entry["non_null"] += 1
    if len(entry["examples"]) < 3 and value is not None:
        example = shorten(value)
        if example not in entry["examples"]:
            entry["examples"].append(example)
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            walk_schema(child, child_path, registry)
    elif isinstance(value, list):
        list_path = f"{path}[*]"
        if not value:
            walk_schema(None, list_path, registry)
        else:
            for child in value[:3]:
                walk_schema(child, list_path, registry)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p05", "median", "p95", "max")}
    values = sorted(values)
    def percentile(frac: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * frac
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {
        "min": values[0],
        "p05": percentile(0.05),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": values[-1],
    }


def audit_archive(
    archive: Path,
    schema_registry: dict[str, dict],
    clearance_fields: dict[str, dict],
    max_flights: int = 0,
) -> tuple[dict, dict, int]:
    inventory = {
        "archive": archive.name,
        "archive_start": "",
        "archive_end": "",
        "zip_members": 0,
        "flight_members": 0,
        "flights_scanned": 0,
        "invalid_json": 0,
        "flights_with_plots": 0,
        "plots": 0,
        "fpl_base_records": 0,
        "fpl_plan_update_records": 0,
        "clearance_records": 0,
        "nonempty_clearance_records": 0,
        "predicted_trajectory_updates": 0,
        "centre_control_records": 0,
        "plot_timestamp_min": "",
        "plot_timestamp_max": "",
        "airspace_bytes": 0,
        "weather_bytes": 0,
        "unique_callsigns": 0,
        "unique_flight_ids": 0,
    }
    match = archive.name.lower().removeprefix("scat").removesuffix(".zip")
    if "_" in match:
        inventory["archive_start"], inventory["archive_end"] = match.split("_", 1)

    interval_samples: list[float] = []
    callsigns = set()
    flight_ids = set()
    with zipfile.ZipFile(archive) as zipped:
        infos = {Path(info.filename).name.lower(): info for info in zipped.infolist()}
        inventory["zip_members"] = len(zipped.namelist())
        inventory["flight_members"] = len(flight_members(zipped))
        for special in SPECIAL_MEMBERS:
            inventory[f"{special.removesuffix('.json')}_bytes"] = infos.get(special).file_size if special in infos else 0
        members = flight_members(zipped)
        if max_flights:
            members = members[:max_flights]
        for member in members:
            inventory["flights_scanned"] += 1
            try:
                with zipped.open(member) as stream:
                    flight = json.load(stream)
            except (OSError, ValueError, json.JSONDecodeError):
                inventory["invalid_json"] += 1
                continue
            if not isinstance(flight, dict):
                inventory["invalid_json"] += 1
                continue
            walk_schema(flight, "", schema_registry)
            flight_id = str(flight.get("id", Path(member).stem))
            flight_ids.add(flight_id)
            fpl = flight.get("fpl") or {}
            base_records = fpl.get("fpl_base") or []
            plan_records = fpl.get("fpl_plan_update") or []
            clearance_records = fpl.get("fpl_clearance") or []
            inventory["fpl_base_records"] += len(base_records)
            inventory["fpl_plan_update_records"] += len(plan_records)
            inventory["clearance_records"] += len(clearance_records)
            inventory["predicted_trajectory_updates"] += len(flight.get("predicted_trajectory") or [])
            inventory["centre_control_records"] += len(flight.get("centre_ctrl") or [])
            for base in base_records:
                if isinstance(base, dict) and base.get("callsign"):
                    callsigns.add(str(base["callsign"]))
            plot_times = []
            for plot in flight.get("plots") or []:
                if not isinstance(plot, dict):
                    continue
                inventory["plots"] += 1
                epoch = parse_timestamp(plot.get("time_of_track"))
                if epoch is not None:
                    plot_times.append(epoch)
                walk_schema(plot, "plots[*]", schema_registry)
            if plot_times:
                inventory["flights_with_plots"] += 1
                plot_times.sort()
                if not inventory["plot_timestamp_min"] or plot_times[0] < parse_timestamp(inventory["plot_timestamp_min"]):
                    inventory["plot_timestamp_min"] = pd.Timestamp(plot_times[0], unit="s", tz="UTC").isoformat()
                if not inventory["plot_timestamp_max"] or plot_times[-1] > parse_timestamp(inventory["plot_timestamp_max"]):
                    inventory["plot_timestamp_max"] = pd.Timestamp(plot_times[-1], unit="s", tz="UTC").isoformat()
                for first, second in zip(plot_times, plot_times[1:]):
                    diff = second - first
                    if diff >= 0 and len(interval_samples) < 200_000:
                        interval_samples.append(diff)
            for record in clearance_records:
                if not isinstance(record, dict):
                    continue
                present = [key for key, value in record.items() if key != "time_stamp" and value not in (None, "")]
                if present:
                    inventory["nonempty_clearance_records"] += 1
                for key, value in record.items():
                    entry = clearance_fields.setdefault(
                        key,
                        {"records": 0, "non_null": 0, "examples": set()},
                    )
                    entry["records"] += 1
                    if value not in (None, ""):
                        entry["non_null"] += 1
                        if len(entry["examples"]) < 5:
                            entry["examples"].add(shorten(value))
        inventory["unique_callsigns"] = len(callsigns)
        inventory["unique_flight_ids"] = len(flight_ids)
    sampling = {
        "archive": archive.name,
        "flights_scanned": inventory["flights_scanned"],
        "plots": inventory["plots"],
        "interval_count": len(interval_samples),
        **{f"interval_{key}_s": value for key, value in quantiles(interval_samples).items()},
    }
    return inventory, sampling, len(interval_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "01_audit")
    parser.add_argument("--weeks", nargs="*", help="Optional ZIP names or week tokens")
    parser.add_argument(
        "--max-flights-per-archive",
        type=int,
        default=0,
        help="0 scans every flight; use a positive value for a quick audit smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    archives = archive_paths(args.input_root, args.weeks)
    schema_registry: dict[str, dict] = {}
    clearance_fields: dict[str, dict] = {}
    inventory_rows = []
    sampling_rows = []
    for index, archive in enumerate(archives, start=1):
        print(f"[{index}/{len(archives)}] Auditing {archive.name}", flush=True)
        inventory, sampling, _ = audit_archive(
            archive,
            schema_registry,
            clearance_fields,
            max_flights=args.max_flights_per_archive,
        )
        inventory_rows.append(inventory)
        sampling_rows.append(sampling)

    schema_json = {
        "dataset": "SCAT",
        "scan": {
            "input_root": str(args.input_root),
            "archives": [path.name for path in archives],
            "max_flights_per_archive": args.max_flights_per_archive,
            "note": "Schema paths are observed from the real JSON; arrays use [*].",
        },
        "paths": {
            path: {
                **entry,
                "types": sorted(entry["types"]),
            }
            for path, entry in sorted(schema_registry.items())
        },
    }
    (args.output_root / "scat_schema.json").write_text(
        json.dumps(schema_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(inventory_rows).to_csv(args.output_root / "scat_inventory.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(args.output_root / "sampling_inventory.csv", index=False)
    clearance_rows = []
    for field, entry in sorted(clearance_fields.items()):
        clearance_rows.append(
            {
                "field": field,
                "records": entry["records"],
                "non_null": entry["non_null"],
                "missing_rate": 1 - (entry["non_null"] / entry["records"] if entry["records"] else 0),
                "examples": " | ".join(sorted(entry["examples"])),
            }
        )
    pd.DataFrame(clearance_rows).to_csv(args.output_root / "clearance_field_inventory.csv", index=False)

    total = pd.DataFrame(inventory_rows).sum(numeric_only=True)
    report = [
        "# SCAT audit report",
        "",
        "This report is structural only. It does not define a conflict label and does not apply the OpenSky safety states.",
        "",
        f"- Archives selected: **{len(archives)}**",
        f"- Flights scanned: **{int(total.get('flights_scanned', 0))}**",
        f"- Surveillance plots: **{int(total.get('plots', 0))}**",
        f"- Flight-plan update records: **{int(total.get('fpl_plan_update_records', 0))}**",
        f"- Clearance records: **{int(total.get('clearance_records', 0))}**",
        f"- Non-empty clearance records: **{int(total.get('nonempty_clearance_records', 0))}**",
        f"- Predicted-trajectory updates: **{int(total.get('predicted_trajectory_updates', 0))}**",
        f"- Invalid JSON files: **{int(total.get('invalid_json', 0))}**",
        "",
        "## Observed clearance fields",
        "",
    ]
    if clearance_rows:
        report.extend(f"- `{row['field']}`: {row['non_null']} non-null values" for row in clearance_rows)
    else:
        report.append("No clearance fields were observed in the selected scan.")
    report.extend(
        [
            "",
            "## Outputs",
            "",
            "- `scat_schema.json`: observed nested JSON paths and examples.",
            "- `scat_inventory.csv`: one row per weekly archive.",
            "- `clearance_field_inventory.csv`: field availability in `fpl_clearance`.",
            "- `sampling_inventory.csv`: surveillance timestamp interval statistics.",
            "",
            "The extraction stages must use these observed fields and preserve source archive/member/plot indices for traceability.",
        ]
    )
    (args.output_root / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Audit complete: {args.output_root}")


if __name__ == "__main__":
    main()


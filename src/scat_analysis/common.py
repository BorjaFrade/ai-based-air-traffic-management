#!/usr/bin/env python3
"""Shared SCAT readers and normalisation helpers.

The analysis pipeline deliberately reads the original JSON files directly.
The existing 10-second adapter is useful for the rule-based detector, but the
exploratory SCAT analysis must preserve the native surveillance resolution and
the time-varying ATC context.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "scat" / "weekly_archives"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "scat"

ARCHIVE_RE = re.compile(
    r"^scat(?P<start>\d{8})_(?P<end>\d{8})\.zip$", re.IGNORECASE
)
SPECIAL_MEMBERS = {"airspace.json", "grib_meteo.json"}

MPS_TO_KT = 1.9438444924406
MPS_TO_FPM = 196.8503937007874
FT_TO_M = 0.3048


def parse_timestamp(value) -> float | None:
    """Return an epoch in UTC for SCAT's naive ISO-8601 timestamps."""

    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).timestamp()


def iso_utc(epoch: float | None) -> str:
    if epoch is None:
        return ""
    return (
        dt.datetime.fromtimestamp(float(epoch), tz=dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compact_json(value) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_value(value):
    """Convert values into stable JSON-comparable primitives."""

    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def normalise_for_compare(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return json.dumps(json_value(value), sort_keys=True, ensure_ascii=False, default=str)


def archive_paths(input_root: Path = DEFAULT_RAW_ROOT, weeks: Iterable[str] | None = None) -> list[Path]:
    paths = sorted(input_root.glob("scat*.zip"))
    if not paths:
        raise FileNotFoundError(f"No SCAT ZIP archives found under {input_root}")
    if not weeks:
        return paths
    wanted = set()
    for week in weeks:
        token = Path(str(week)).name.lower()
        if not token.endswith(".zip"):
            token = f"scat{token}.zip" if not token.startswith("scat") else f"{token}.zip"
        wanted.add(token)
    selected = [path for path in paths if path.name.lower() in wanted]
    if not selected:
        raise FileNotFoundError(
            f"None of the requested SCAT weeks were found. Available: "
            f"{[path.name for path in paths]}"
        )
    return selected


def flight_members(zipped: zipfile.ZipFile) -> list[str]:
    members = []
    for name in zipped.namelist():
        basename = Path(name).name.lower()
        if not basename.endswith(".json") or basename in SPECIAL_MEMBERS:
            continue
        members.append(name)
    return sorted(members)


def iter_flights(
    archives: Iterable[Path],
    max_flights_per_archive: int = 0,
) -> Iterator[tuple[Path, str, dict]]:
    """Yield ``(archive, member, flight_json)`` without extracting ZIPs."""

    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            members = flight_members(zipped)
            if max_flights_per_archive:
                members = members[:max_flights_per_archive]
            for member in members:
                try:
                    with zipped.open(member) as stream:
                        flight = json.load(stream)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(flight, dict):
                    yield archive, member, flight


def records_with_time(items: Iterable[dict] | None) -> list[tuple[float, dict]]:
    records = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        epoch = parse_timestamp(item.get("time_stamp", item.get("start_time")))
        if epoch is not None:
            records.append((epoch, item))
    records.sort(key=lambda pair: pair[0])
    return records


def latest_record_with_age(
    records: list[tuple[float, dict]], epoch: float
) -> tuple[dict, float | None]:
    """Return the latest record available *as of* ``epoch`` and its age.

    A record published after ``epoch`` is never a valid predictor for that
    timestamp.  The old implementation fell back to the first record when no
    prior record existed, which could silently introduce future context.
    """

    if not records:
        return {}, None
    times = [item[0] for item in records]
    index = bisect.bisect_right(times, epoch) - 1
    if index < 0:
        return {}, None
    source_epoch, record = records[index]
    assert source_epoch <= epoch, "as-of context must not come from the future"
    return record, float(epoch - source_epoch)


def latest_record(records: list[tuple[float, dict]], epoch: float) -> dict:
    """Return the latest record at or before ``epoch``; otherwise ``{}``."""

    record, _ = latest_record_with_age(records, epoch)
    return record


def route_summary(route: list[dict] | None) -> tuple[str, str, str]:
    if not isinstance(route, list):
        return "", "", ""
    valid = [item for item in route if isinstance(item, dict)]
    names = [str(item.get("fix_name", "") or "").strip() for item in valid]
    names = [name for name in names if name]
    return (
        ">".join(names),
        names[0] if names else "",
        str(valid[0].get("eto", "") or "") if valid else "",
    )


def flight_key(archive: Path, flight: dict, member: str) -> str:
    identifier = str(flight.get("id", Path(member).stem)).strip()
    return f"{archive.stem}:{identifier}"


def build_context(flight: dict) -> dict:
    fpl = flight.get("fpl") or {}
    return {
        "flight_key": "",
        "base_records": records_with_time(fpl.get("fpl_base")),
        "plan_records": records_with_time(fpl.get("fpl_plan_update")),
        "clearance_records": records_with_time(fpl.get("fpl_clearance")),
        "arrival_records": records_with_time(fpl.get("fpl_arr")),
        "departure_records": records_with_time(fpl.get("fpl_dep")),
        "holding_records": records_with_time(fpl.get("fpl_holding")),
        "centre_records": records_with_time(flight.get("centre_ctrl")),
        "prediction_records": records_with_time(flight.get("predicted_trajectory")),
    }


def _subitem_number(item: dict, key: str = "altitude") -> float | None:
    if not isinstance(item, dict):
        return None
    return finite_number(item.get(key))


def extract_plot_kinematics(plot: dict) -> dict | None:
    position = plot.get("I062/105") or {}
    lat = finite_number(position.get("lat"))
    lon = finite_number(position.get("lon"))
    epoch = parse_timestamp(plot.get("time_of_track"))
    if None in (lat, lon, epoch):
        return None

    measured_fl = finite_number((plot.get("I062/136") or {}).get("measured_flight_level"))
    item_380 = plot.get("I062/380") or {}
    selected_altitude_ft = _subitem_number(item_380.get("subitem6") or {})
    final_selected_altitude_ft = _subitem_number(item_380.get("subitem7") or {})
    altitude_ft = measured_fl * 100.0 if measured_fl is not None else selected_altitude_ft
    if altitude_ft is None:
        altitude_ft = final_selected_altitude_ft

    velocity = plot.get("I062/185") or {}
    vx = finite_number(velocity.get("vx"))
    vy = finite_number(velocity.get("vy"))
    speed_ms = math.hypot(vx, vy) if vx is not None and vy is not None else None
    track_deg = (
        math.degrees(math.atan2(vx, vy)) % 360.0
        if vx is not None and vy is not None and speed_ms
        else None
    )
    magnetic_heading = finite_number((item_380.get("subitem3") or {}).get("mag_hdg"))
    vertical_rate_fpm = finite_number((plot.get("I062/220") or {}).get("rocd"))
    if vertical_rate_fpm is None:
        vertical_rate_fpm = finite_number(
            (item_380.get("subitem13") or {}).get("baro_vert_rate")
        )

    if altitude_ft is None or not math.isfinite(altitude_ft):
        return None
    return {
        "epoch": epoch,
        "lat": lat,
        "lon": lon,
        "flight_level": measured_fl,
        "altitude_ft": altitude_ft,
        "selected_altitude_ft": selected_altitude_ft,
        "final_selected_altitude_ft": final_selected_altitude_ft,
        "speed_ms": speed_ms,
        "ground_speed_kt": speed_ms * MPS_TO_KT if speed_ms is not None else None,
        "track_deg": track_deg,
        "heading_deg": magnetic_heading,
        "vertical_rate_fpm": vertical_rate_fpm,
        "vertical_rate_ms": (
            vertical_rate_fpm / MPS_TO_FPM if vertical_rate_fpm is not None else None
        ),
    }


def phase_label(altitude_ft, vertical_rate_fpm, on_ground=None) -> str:
    altitude = finite_number(altitude_ft)
    rate = finite_number(vertical_rate_fpm)
    if on_ground is True:
        return "GROUND"
    if altitude is None:
        return "UNKNOWN"
    if altitude < 10_000:
        return "TERMINAL"
    if rate is not None and rate > 300:
        return "CLIMB"
    if rate is not None and rate < -300:
        return "DESCENT"
    return "CRUISE_OR_LEVEL"


def attach_context(context: dict, epoch: float) -> dict:
    base, base_age = latest_record_with_age(context["base_records"], epoch)
    plan, plan_age = latest_record_with_age(context["plan_records"], epoch)
    clearance, clearance_age = latest_record_with_age(context["clearance_records"], epoch)
    arrival, arrival_age = latest_record_with_age(context["arrival_records"], epoch)
    departure, departure_age = latest_record_with_age(context["departure_records"], epoch)
    holding, holding_age = latest_record_with_age(context["holding_records"], epoch)
    centre, centre_age = latest_record_with_age(context["centre_records"], epoch)
    prediction, prediction_age = latest_record_with_age(context["prediction_records"], epoch)
    route, next_fix, next_eta = route_summary(prediction.get("route"))
    ages = [
        age for age in (
            base_age, plan_age, clearance_age, arrival_age, departure_age,
            holding_age, centre_age, prediction_age,
        ) if age is not None
    ]
    return {
        "callsign": str(base.get("callsign", "") or "").strip(),
        "fpl_adep": base.get("adep", ""),
        "fpl_ades": base.get("ades", ""),
        "fpl_adar": base.get("adar", ""),
        "fpl_aircraft_type": base.get("aircraft_type", ""),
        "fpl_wtc": base.get("wtc", ""),
        "fpl_flight_rules": base.get("flight_rules", ""),
        "fpl_equip_status_rvsm": base.get("equip_status_rvsm", ""),
        "fpl_route": plan.get("icao_route", ""),
        "fpl_rfl": plan.get("rfl_string", ""),
        "fpl_tas": plan.get("tas_string", ""),
        "fpl_copn": plan.get("copn", ""),
        "fpl_copn_pel": plan.get("copn_pel", ""),
        "fpl_copn_pel_unit": plan.get("copn_pel_unit", ""),
        "fpl_copx": plan.get("copx", ""),
        "fpl_copx_pel": plan.get("copx_pel", ""),
        "fpl_copx_pel_unit": plan.get("copx_pel_unit", ""),
        "fpl_update_timestamp": plan.get("time_stamp", ""),
        "clearance_cfl": clearance.get("cfl", ""),
        "clearance_cfl_unit": clearance.get("cfl_unit", ""),
        "clearance_heading_deg": clearance.get("assigned_heading_val", ""),
        "clearance_heading_beacon": clearance.get("assign_heading_beacon", ""),
        "clearance_speed": clearance.get("assigned_speed_val", ""),
        "clearance_speed_unit": clearance.get("assigned_speed_unit", ""),
        "clearance_timestamp": clearance.get("time_stamp", ""),
        "active_centre_id": centre.get("centre_id", ""),
        "centre_since": centre.get("start_time", ""),
        "arrival_approach_clearance": arrival.get("approach_clearance", ""),
        "arrival_runway": arrival.get("arrival_runway", ""),
        "arrival_star": arrival.get("star", ""),
        "departure_runway": departure.get("departure_runway", ""),
        "departure_sid": departure.get("sid", ""),
        "holding_status": holding.get("holding_status", ""),
        "predicted_route_timestamp": prediction.get("time_stamp", ""),
        "predicted_next_fix": next_fix,
        "predicted_next_fix_eta": next_eta,
        "predicted_route": route,
        "predicted_route_json": compact_json(prediction.get("route", [])),
        "context_age_seconds": max(ages) if ages else None,
        "base_context_age_seconds": base_age,
        "plan_context_age_seconds": plan_age,
        "clearance_context_age_seconds": clearance_age,
        "arrival_context_age_seconds": arrival_age,
        "departure_context_age_seconds": departure_age,
        "holding_context_age_seconds": holding_age,
        "centre_context_age_seconds": centre_age,
        "prediction_context_age_seconds": prediction_age,
    }


def extract_traffic_rows(
    archive: Path,
    member: str,
    flight: dict,
    include_raw_plot_json: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return native-resolution surveillance rows and prediction sidecar rows."""

    key = flight_key(archive, flight, member)
    context = build_context(flight)
    context["flight_key"] = key
    rows = []
    for plot_index, plot in enumerate(flight.get("plots") or []):
        if not isinstance(plot, dict):
            continue
        kinematics = extract_plot_kinematics(plot)
        if kinematics is None:
            continue
        context_row = attach_context(context, kinematics["epoch"])
        on_ground = None
        row = {
            "flight_key": key,
            "flight_id": str(flight.get("id", Path(member).stem)),
            "archive_week": archive.stem.removeprefix("scat"),
            "source_archive": archive.name,
            "source_member": member,
            "plot_index": plot_index,
            "timestamp": iso_utc(kinematics["epoch"]),
            "time_epoch": kinematics["epoch"],
            "callsign": context_row.pop("callsign", ""),
            "latitude": kinematics["lat"],
            "longitude": kinematics["lon"],
            "altitude_ft": kinematics["altitude_ft"],
            "flight_level": kinematics["flight_level"],
            "selected_altitude_ft": kinematics["selected_altitude_ft"],
            "final_selected_altitude_ft": kinematics["final_selected_altitude_ft"],
            "ground_speed_ms": kinematics["speed_ms"],
            "ground_speed_kt": kinematics["ground_speed_kt"],
            "track_deg": kinematics["track_deg"],
            "heading_deg": kinematics["heading_deg"],
            "vertical_rate_fpm": kinematics["vertical_rate_fpm"],
            "vertical_rate_ms": kinematics["vertical_rate_ms"],
            "on_ground": on_ground,
            "flight_phase": phase_label(
                kinematics["altitude_ft"], kinematics["vertical_rate_fpm"], on_ground
            ),
            "source_plot_time": plot.get("time_of_track", ""),
            "clearance_record_json": compact_json(
                latest_record(context["clearance_records"], kinematics["epoch"])
                or ""
            ),
            "plot_source_json": compact_json(plot) if include_raw_plot_json else "",
            **context_row,
        }
        rows.append(row)

    prediction_rows = []
    for prediction_index, prediction in enumerate(flight.get("predicted_trajectory") or []):
        if not isinstance(prediction, dict):
            continue
        timestamp = prediction.get("time_stamp", "")
        prediction_rows.append(
            {
                "flight_key": key,
                "flight_id": str(flight.get("id", Path(member).stem)),
                "archive_week": archive.stem.removeprefix("scat"),
                "source_archive": archive.name,
                "source_member": member,
                "prediction_index": prediction_index,
                "prediction_timestamp": timestamp,
                "prediction_epoch": parse_timestamp(timestamp),
                "route_json": compact_json(prediction.get("route", [])),
            }
        )
    return rows, prediction_rows


def action_fields(record: dict) -> list[str]:
    return [
        key for key, value in record.items()
        if key != "time_stamp" and is_present(value)
    ]


def clearance_type(changed_fields: Iterable[str]) -> str:
    changed = set(changed_fields)
    types = []
    if changed & {"assigned_heading_val", "assign_heading_beacon"}:
        types.append("HEADING")
    if "cfl" in changed or "cfl_unit" in changed:
        types.append("LEVEL")
    if changed & {"assigned_speed_val", "assigned_speed_unit"}:
        types.append("SPEED")
    other = sorted(
        field for field in changed
        if field not in {
            "assigned_heading_val", "assign_heading_beacon",
            "cfl", "cfl_unit", "assigned_speed_val", "assigned_speed_unit",
        }
    )
    types.extend(f"OTHER:{field}" for field in other)
    return "+".join(types) if types else "OTHER"


def numeric_delta(previous, current):
    old = finite_number(previous)
    new = finite_number(current)
    if old is None or new is None:
        return None
    return new - old


def update_read_me_first(base_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Create the compact investigator landing page for completed stages."""

    base_root.mkdir(parents=True, exist_ok=True)
    stage_names = [
        "01_audit", "02_traffic", "03_clearances", "04_windows", "05_features",
        "06_exploration", "07_explainability", "08_bayesian", "09_counterfactual",
        "10_opensky_comparison",
    ]
    lines = [
        "# SCAT exploratory analysis — read first",
        "",
        "This pipeline studies `ATC_INTERVENTION`, not conflict. A recorded clearance is not treated as a conflict label. The OpenSky GREEN/YELLOW/ORANGE/RED rules are deliberately deferred to stage 10.",
        "",
        "## Stage status",
        "",
    ]
    for stage in stage_names:
        directory = base_root / stage
        if not directory.exists():
            lines.append(f"- `{stage}`: not run")
            continue
        reports = sorted(directory.glob("*_report.md")) + sorted(directory.glob("*_summary.md"))
        artifacts = sorted(directory.glob("*.parquet")) + sorted(directory.glob("*.csv"))
        lines.append(
            f"- `{stage}`: available — "
            + ", ".join(f"[{path.name}]({path.relative_to(base_root)})" for path in reports[:3])
            + (f"; {len(artifacts)} data artifacts" if artifacts else "")
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `ATC_INTERVENTION = 1` means that an actionable clearance/intervention record exists at the event time.",
            "- `LIKELY_INTERACTION_RELATED` is a later provisional label, not a confirmed conflict.",
            "- Separation, CPA, TCPA and closing rates remain continuous until the final comparison stage.",
            "- Any suspiciously strong model result must be checked for temporal leakage and same-event duplication.",
            "",
            "Run the stages from `src/scat_analysis/README.md`. Each stage records its configuration and input/output paths.",
        ]
    )
    target = base_root / "00_READ_ME_FIRST.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


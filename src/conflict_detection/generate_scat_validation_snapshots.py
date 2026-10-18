#!/usr/bin/env python3
"""
Render random SCAT validation snapshots from a detector run.

The selection logic mirrors the OpenSky validation workflow: up to N unique
timestamps are sampled independently for current YELLOW, ORANGE and RED
states.  The full SCAT traffic snapshot is then reloaded, the pair is
recomputed with the shared safety model and a map is rendered with the
flight-plan and controller-clearance context in a validation footer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
ACQUISITION_DIR = SRC_DIR / "acquisition"

if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import detect_opensky_conflicts as opensky_detector  # noqa: E402
from area_presets import (  # noqa: E402
    load_area_presets,
    load_corridors_for_area,
    load_waypoints_for_area,
    resolve_area_preset,
)
from opensky_plot import (  # noqa: E402
    build_shared_validation_render_config,
    render_map,
)
from opensky_safety import (  # noqa: E402
    analyse_aircraft_snapshot,
    enabled_states,
    relevant_pair_results,
    state_colours,
    state_display_names,
)


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "scat_10s"
SEPARATION_ROOT = PROJECT_ROOT / "data" / "processed" / "scat_separation"
BBOX_PRESETS_FILE = ACQUISITION_DIR / "bbox_presets.json"
WAYPOINTS_FILE = ACQUISITION_DIR / "waypoints.json"
CORRIDORS_FILE = ACQUISITION_DIR / "corridors.json"

EVENT_FILES = (
    "yellow_proximity_events.csv",
    "orange_high_risk_events.csv",
    "red_loss_of_separation_events.csv",
)

STATE_TO_COLOUR = {
    opensky_detector.STATE_PROXIMITY: "YELLOW",
    opensky_detector.STATE_HIGH_RISK: "ORANGE",
    opensky_detector.STATE_CONFLICT: "RED",
}
STATE_ORDER = {
    opensky_detector.STATE_PROXIMITY: 0,
    opensky_detector.STATE_HIGH_RISK: 1,
    opensky_detector.STATE_CONFLICT: 2,
}


def render_config(area: dict, no_basemap: bool) -> tuple[int, int, int, dict]:
    render = area.get("render", {})
    width = int(render.get("width_px", 3840))
    height = int(render.get("height_px", 2160))
    dpi = 160
    if width <= 0 or height <= 0:
        raise ValueError("Area render dimensions must be positive")
    safety_colours = state_colours(opensky_detector.SAFETY_STATE_RULES)
    safety_names = state_display_names(opensky_detector.SAFETY_STATE_RULES)
    config = build_shared_validation_render_config(
        area,
        width_px=width,
        height_px=height,
        dpi=dpi,
        title_prefix="SCAT Traffic Snapshot",
        show_basemap=not no_basemap,
        overrides={
            "SEPARATION_PROFILE": opensky_detector.SEPARATION_PROFILE,
            "STATE_NORMAL": opensky_detector.STATE_NORMAL,
            "STATE_PROXIMITY": opensky_detector.STATE_PROXIMITY,
            "STATE_HIGH_RISK": opensky_detector.STATE_HIGH_RISK,
            "STATE_CONFLICT": opensky_detector.STATE_CONFLICT,
            "ENABLED_SAFETY_STATES": enabled_states(opensky_detector.SAFETY_STATE_RULES),
            "SAFETY_STATE_COLORS": safety_colours,
            "SAFETY_STATE_DISPLAY_NAMES": safety_names,
            "NORMAL_AIRCRAFT_COLOR": safety_colours[opensky_detector.STATE_NORMAL],
            "PROXIMITY_AIRCRAFT_COLOR": safety_colours[opensky_detector.STATE_PROXIMITY],
            "HIGH_RISK_AIRCRAFT_COLOR": safety_colours[opensky_detector.STATE_HIGH_RISK],
            "CONFLICT_AIRCRAFT_COLOR": safety_colours[opensky_detector.STATE_CONFLICT],
            "WARNING_AIRCRAFT_COLOR": safety_colours[opensky_detector.STATE_PROXIMITY],
            "MIN_HORIZONTAL_SEPARATION_NM": opensky_detector.MIN_HORIZONTAL_SEPARATION_NM,
            "MIN_VERTICAL_SEPARATION_FT": opensky_detector.MIN_VERTICAL_SEPARATION_FT,
            "LOOKAHEAD_MINUTES": [1, 2, 5],
            "LOOKAHEAD_INTERVAL_STATES": {},
        },
    )
    return width, height, dpi, config


def run_directory(area: str, run_name: str) -> Path:
    path = SEPARATION_ROOT / area / "runs" / run_name
    if not path.exists():
        raise FileNotFoundError(f"SCAT detector run not found: {path}")
    return path


def load_events(run_dir: Path) -> pd.DataFrame:
    frames = []
    required = {
        "timestamp",
        "state_current",
        "event_state",
        "icao24_a",
        "icao24_b",
        "horizontal_sep_nm",
        "vertical_sep_ft",
    }
    for filename in EVENT_FILES:
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing detector event file: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.loc[frame["timestamp"].notna()].copy()
        frame["timestamp"] = frame["timestamp"].round().astype("int64")
        for column in ("state_current", "event_state"):
            frame[column] = frame[column].astype(str).str.strip().str.upper()
        frame["source_event_file"] = filename
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=sorted(required))
    return pd.concat(frames, ignore_index=True)


def select_samples(events: pd.DataFrame, n: int, seed: int | None) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    selected = []
    for state in STATE_ORDER:
        candidates = events.loc[events["state_current"] == state].copy()
        if candidates.empty:
            continue
        order = rng.permutation(len(candidates))
        candidates = candidates.iloc[order].drop_duplicates(
            subset=["timestamp"], keep="first"
        )
        count = min(n, len(candidates))
        if count:
            sample = candidates.iloc[
                rng.choice(len(candidates), size=count, replace=False)
            ].copy()
            sample["validation_state"] = state
            sample["validation_colour"] = STATE_TO_COLOUR[state]
            selected.append(sample)
    if not selected:
        return pd.DataFrame()
    result = pd.concat(selected, ignore_index=True)
    result["_state_order"] = result["validation_state"].map(STATE_ORDER)
    result.sort_values(["_state_order", "timestamp"], inplace=True)
    result.drop(columns=["_state_order"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def daily_path(area: str, timestamp: int) -> Path:
    day = pd.to_datetime(timestamp, unit="s", utc=True).strftime("%Y-%m-%d")
    return (
        PROCESSED_ROOT
        / area
        / f"date={day}"
        / f"scat_{day}_10s.csv"
    )


def load_selected_snapshots(
    area: str,
    timestamps: list[int],
    chunksize: int,
) -> dict[int, pd.DataFrame]:
    by_file: dict[Path, set[int]] = {}
    for timestamp in timestamps:
        by_file.setdefault(daily_path(area, timestamp), set()).add(timestamp)

    pieces: dict[int, list[pd.DataFrame]] = {timestamp: [] for timestamp in timestamps}
    for path, wanted in by_file.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing processed SCAT daily file: {path}")
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
            chunk = chunk.loc[chunk["time"].notna()].copy()
            chunk["_selected_timestamp"] = chunk["time"].round().astype("int64")
            chunk = chunk.loc[chunk["_selected_timestamp"].isin(wanted)]
            if chunk.empty:
                continue
            for timestamp, group in chunk.groupby("_selected_timestamp"):
                pieces[int(timestamp)].append(
                    group.drop(columns=["_selected_timestamp"])
                )

    snapshots = {}
    for timestamp, groups in pieces.items():
        if not groups:
            continue
        snapshot = pd.concat(groups, ignore_index=True)
        snapshot.sort_values(["icao24", "lastcontact"], inplace=True, kind="mergesort")
        snapshot.drop_duplicates("icao24", keep="last", inplace=True)
        snapshots[timestamp] = snapshot.reset_index(drop=True)
    return snapshots


def prepare_for_plot(snapshot: pd.DataFrame) -> pd.DataFrame:
    result = snapshot.copy()
    result["latitude"] = pd.to_numeric(result["lat"], errors="coerce")
    result["longitude"] = pd.to_numeric(result["lon"], errors="coerce")
    result["baro_altitude_ft"] = pd.to_numeric(
        result["baro_altitude_ft"], errors="coerce"
    )
    result["velocity_kts"] = pd.to_numeric(result["ground_speed_kt"], errors="coerce")
    result["vertical_rate_fpm"] = pd.to_numeric(
        result["vertical_rate_fpm"], errors="coerce"
    )
    result["on_ground"] = result["on_ground"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    return result


def find_pair_result(pair_results: list[dict], icao_a: str, icao_b: str) -> dict | None:
    wanted = {str(icao_a), str(icao_b)}
    for result in pair_results:
        if {str(result.get("icao24_a", "")), str(result.get("icao24_b", ""))} == wanted:
            return result
    return None


def display(value, limit: int = 140) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def add_validation_footer(
    fig,
    selected_row: pd.Series,
    pair_result: dict | None,
    validation_colour: str,
) -> None:
    lines = [
        f"VALIDATION {validation_colour} | current={display(selected_row.get('state_current'))} | "
        f"worst_5min={display(selected_row.get('event_state'))}",
        f"A {display(selected_row.get('callsign_a'))} ({display(selected_row.get('icao24_a'))}) "
        f"{display(selected_row.get('fpl_adep_a'))}->{display(selected_row.get('fpl_ades_a'))} "
        f"clearance CFL {display(selected_row.get('clearance_cfl_a'))}{display(selected_row.get('clearance_cfl_unit_a'))}",
        f"B {display(selected_row.get('callsign_b'))} ({display(selected_row.get('icao24_b'))}) "
        f"{display(selected_row.get('fpl_adep_b'))}->{display(selected_row.get('fpl_ades_b'))} "
        f"clearance CFL {display(selected_row.get('clearance_cfl_b'))}{display(selected_row.get('clearance_cfl_unit_b'))}",
        f"detector H={display(selected_row.get('horizontal_sep_nm'))} NM | "
        f"V={display(selected_row.get('vertical_sep_ft'))} ft | "
        f"route A: {display(selected_row.get('fpl_route_a'), 110)}",
        f"route B: {display(selected_row.get('fpl_route_b'), 110)}",
    ]
    if pair_result is not None:
        horizontal = pair_result.get("horizontal_sep_nm")
        try:
            horizontal_text = f"{float(horizontal):.3f}"
        except (TypeError, ValueError):
            horizontal_text = display(horizontal)
        lines.append(
            f"recomputed H={horizontal_text} NM | "
            f"V={display(pair_result.get('vertical_sep_ft'))} ft | "
            f"state={display(pair_result.get('state_current'))}"
        )
    fig.text(
        0.045,
        0.018,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.94, "pad": 5},
    )


def highlight_target_pair(ax, snapshot: pd.DataFrame, selected_row: pd.Series) -> None:
    colour = {
        "YELLOW": "#FFD400",
        "ORANGE": "#FF6F00",
        "RED": "#D7191C",
    }.get(str(selected_row.get("validation_colour", "")), "#FFFFFF")
    points = []
    for suffix in ("a", "b"):
        wanted = str(selected_row.get(f"icao24_{suffix}", "")).strip()
        matches = snapshot.loc[snapshot["icao24"].astype(str).str.strip() == wanted]
        if matches.empty:
            continue
        aircraft = matches.iloc[0]
        points.append((float(aircraft["longitude"]), float(aircraft["latitude"])))
    if len(points) == 2:
        ax.plot(
            [points[0][0], points[1][0]],
            [points[0][1], points[1][1]],
            color=colour,
            linewidth=2.0,
            linestyle="--",
            alpha=0.75,
            zorder=13.5,
            solid_capstyle="round",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random SCAT validation snapshots from a detector run."
    )
    parser.add_argument("--area", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n", type=int, default=10, help="Maximum images per colour")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--no-basemap", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_directory(args.area, args.run_name)
    events = load_events(run_dir)
    selected = select_samples(events, args.n, args.seed) if not events.empty else pd.DataFrame()
    seed_label = "random" if args.seed is None else f"random_seed_{args.seed}"
    output_dir = run_dir / "validation_screenshots" / seed_label
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selected_random_moments.csv", index=False)

    if selected.empty:
        report = pd.DataFrame(
            [{"generated": 0, "reason": "No non-normal current-state events"}]
        )
        report.to_csv(output_dir / "validation_report.csv", index=False)
        if not args.allow_empty:
            raise RuntimeError(
                "The detector run has no PROXIMITY, HIGH_RISK or CONFLICT current-state events."
            )
        print(f"No validation events. Wrote {output_dir}")
        return

    snapshots = load_selected_snapshots(
        args.area,
        selected["timestamp"].astype(int).tolist(),
        args.chunksize,
    )
    presets = load_area_presets(BBOX_PRESETS_FILE)
    area = resolve_area_preset(args.area, presets)
    bbox = area["bbox"]
    width, height, dpi, config = render_config(area, args.no_basemap)
    waypoints = load_waypoints_for_area(
        WAYPOINTS_FILE,
        area_key=args.area,
        area=area,
    )
    corridors = load_corridors_for_area(
        CORRIDORS_FILE,
        area_key=args.area,
        area=area,
    )
    report_rows = []

    for number, selected_row in selected.iterrows():
        timestamp = int(selected_row["timestamp"])
        snapshot = snapshots.get(timestamp)
        if snapshot is None:
            raise RuntimeError(f"No traffic snapshot found for timestamp {timestamp}")
        snapshot = prepare_for_plot(snapshot)
        snapshot.reset_index(drop=True, inplace=True)

        pair_results, aircraft_states, interval_states, skipped_pairs = analyse_aircraft_snapshot(
            snapshot,
            min_horizontal_nm=opensky_detector.MIN_HORIZONTAL_SEPARATION_NM,
            min_vertical_ft=opensky_detector.MIN_VERTICAL_SEPARATION_FT,
            state_rules=opensky_detector.SAFETY_STATE_RULES,
            lookahead_minutes=(1, 2, 5),
            prediction_step_seconds=5,
            check_ground_aircraft=False,
        )
        relevant = relevant_pair_results(pair_results)
        warnings = [
            item for item in relevant
            if item.get("state_current") in {
                opensky_detector.STATE_PROXIMITY,
                opensky_detector.STATE_HIGH_RISK,
            }
        ]
        conflicts = [
            item for item in relevant
            if item.get("state_current") == opensky_detector.STATE_CONFLICT
        ]
        pair_result = find_pair_result(
            pair_results,
            selected_row["icao24_a"],
            selected_row["icao24_b"],
        )
        config["LOOKAHEAD_INTERVAL_STATES"] = interval_states
        dt_value = pd.to_datetime(timestamp, unit="s", utc=True).to_pydatetime()
        colour = str(selected_row["validation_colour"])
        filename = (
            f"{colour.lower()}_{number + 1:02d}_"
            f"{dt_value.strftime('%Y%m%dT%H%M%SZ')}.png"
        )
        image_path = output_dir / filename
        fig, ax = render_map(
            aircraft=snapshot,
            area=area,
            bbox=bbox,
            aircraft_states=aircraft_states,
            warnings=warnings,
            conflicts=conflicts,
            snapshot_time=dt_value,
            map_file=image_path,
            config=config,
            waypoints=waypoints,
            corridors=corridors,
            show=False,
            save=False,
        )
        highlight_target_pair(ax, snapshot, selected_row)
        add_validation_footer(fig, selected_row, pair_result, colour)
        fig.set_size_inches(width / dpi, height / dpi, forward=True)
        fig.savefig(image_path, dpi=dpi, facecolor="white")
        plt.close(fig)

        recomputed_state = pair_result.get("state_current") if pair_result else ""
        report_rows.append(
            {
                "validation_colour": colour,
                "timestamp": timestamp,
                "timestamp_utc": dt_value.isoformat(),
                "icao24_a": selected_row["icao24_a"],
                "callsign_a": selected_row.get("callsign_a", ""),
                "icao24_b": selected_row["icao24_b"],
                "callsign_b": selected_row.get("callsign_b", ""),
                "event_state_worst_5min": selected_row.get("event_state", ""),
                "detector_state_current": selected_row.get("state_current", ""),
                "recomputed_state_current": recomputed_state,
                "state_matches": recomputed_state == selected_row.get("state_current"),
                "detector_horizontal_nm": selected_row.get("horizontal_sep_nm", ""),
                "recomputed_horizontal_nm": pair_result.get("horizontal_sep_nm", "") if pair_result else "",
                "detector_vertical_ft": selected_row.get("vertical_sep_ft", ""),
                "recomputed_vertical_ft": pair_result.get("vertical_sep_ft", "") if pair_result else "",
                "skipped_pairs_missing_vertical": skipped_pairs,
                "pair_found_on_map": pair_result is not None,
                "image_width_px": width,
                "image_height_px": height,
                "screenshot": filename,
            }
        )
        print(f"Saved {image_path}")

    pd.DataFrame(report_rows).to_csv(output_dir / "validation_report.csv", index=False)
    print(f"Generated {len(report_rows)} screenshots in {output_dir}")


if __name__ == "__main__":
    main()

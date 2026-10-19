#!/usr/bin/env python3
"""Validate SCAT unit conversions in processed 10-second CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
ACQUISITION_DIR = SRC_DIR / "acquisition"

import sys

if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))

from scat_units import (  # noqa: E402
    FT_TO_M,
    MPS_TO_FPM,
    MPS_TO_KT,
    conversion_error_summary,
)


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "scat_10s"
REQUIRED_COLUMNS = {
    "baroaltitude",
    "baro_altitude_ft",
    "measured_flight_level",
    "velocity",
    "ground_speed_kt",
    "vertical_rate",
    "vertical_rate_fpm",
    "true_track",
}


def discover_files(area: str | None, inputs: list[Path]) -> list[Path]:
    if inputs:
        return sorted(path for path in inputs if path.is_file())
    root = PROCESSED_ROOT / area if area else PROCESSED_ROOT
    return sorted(root.rglob("scat_*_10s.csv"))


def validate_file(path: Path) -> dict:
    dataframe = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    errors = conversion_error_summary(dataframe)
    altitude = pd.to_numeric(dataframe["baro_altitude_ft"], errors="coerce")
    measured_fl = pd.to_numeric(
        dataframe["measured_flight_level"], errors="coerce"
    )
    valid_fl = altitude.notna() & measured_fl.notna()
    if valid_fl.any():
        errors["flight_level_max_error_ft"] = float(
            np.max(np.abs(altitude.loc[valid_fl] - measured_fl.loc[valid_fl] * 100.0))
        )
    else:
        errors["flight_level_max_error_ft"] = 0.0

    speed = pd.to_numeric(dataframe["ground_speed_kt"], errors="coerce")
    track = pd.to_numeric(dataframe["true_track"], errors="coerce")
    errors["negative_speed_rows"] = int((speed.dropna() < 0).sum())
    errors["invalid_track_rows"] = int(
        ((track.dropna() < 0) | (track.dropna() > 360)).sum()
    )
    errors["altitude_unit_check"] = (
        errors["altitude_max_error_ft"] <= 1e-6
        and errors["flight_level_max_error_ft"] <= 1e-6
    )
    errors["horizontal_speed_unit_check"] = errors["speed_max_error_kt"] <= 1e-6
    errors["vertical_speed_unit_check"] = (
        errors["vertical_rate_max_error_fpm"] <= 1e-6
    )
    errors["physical_range_check"] = (
        errors["negative_speed_rows"] == 0
        and errors["invalid_track_rows"] == 0
    )
    errors["path"] = str(path)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check SCAT feet/metres, m/s/kt and vertical-rate conversions."
    )
    parser.add_argument("--area", help="Processed SCAT area key")
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_files(args.area, args.input or [])
    if not paths:
        raise SystemExit("No processed SCAT CSV files found")

    rows = [validate_file(path) for path in paths]
    failed = [row for row in rows if not all(
        row[key]
        for key in (
            "altitude_unit_check",
            "horizontal_speed_unit_check",
            "vertical_speed_unit_check",
            "physical_range_check",
        )
    )]
    payload = {
        "dataset": "SCAT",
        "files": rows,
        "passed": not failed,
        "constants": {
            "ft_to_m": FT_TO_M,
            "mps_to_kt": MPS_TO_KT,
            "mps_to_fpm": MPS_TO_FPM,
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"{'PASS' if row['altitude_unit_check'] and row['horizontal_speed_unit_check'] and row['vertical_speed_unit_check'] else 'FAIL'} "
            f"{row['path']} | rows={row['rows']:,} | "
            f"alt_ft={row['altitude_max_error_ft']:.3g} | "
            f"speed_kt={row['speed_max_error_kt']:.3g} | "
            f"vr_fpm={row['vertical_rate_max_error_fpm']:.3g}"
        )
    if failed:
        raise SystemExit(f"{len(failed)} file(s) failed SCAT unit validation")


if __name__ == "__main__":
    main()

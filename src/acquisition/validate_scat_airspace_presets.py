#!/usr/bin/env python3
"""Validate SCAT airspace-backed bbox presets against a downloaded ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BBOX_PRESETS_FILE = SCRIPT_DIR / "bbox_presets.json"
DEFAULT_ARCHIVE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scat"
    / "weekly_archives"
    / "scat20161015_20161021.zip"
)


def read_airspace(archive: Path) -> list[dict]:
    with zipfile.ZipFile(archive) as zipped:
        return json.loads(zipped.read("airspace.json"))


def sector_bbox(airspace: list[dict], centre: str, sector: str) -> dict:
    centre_entry = next(item for item in airspace if item.get("name") == centre)
    sector_entry = next(
        item for item in centre_entry.get("sectors", [])
        if str(item.get("name")) == str(sector)
    )
    points = [
        point
        for volume in sector_entry.get("volumes", [])
        for point in volume.get("coordinates", [])
    ]
    if not points:
        raise ValueError(f"No coordinates for {centre}/{sector}")
    return {
        "lamin": min(float(point["lat"]) for point in points),
        "lamax": max(float(point["lat"]) for point in points),
        "lomin": min(float(point["lon"]) for point in points),
        "lomax": max(float(point["lon"]) for point in points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SCAT airspace sector envelopes with bbox_presets.json"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--area", action="append", help="Preset key; repeatable")
    args = parser.parse_args()

    presets = json.loads(BBOX_PRESETS_FILE.read_text(encoding="utf-8"))
    targets = args.area or [
        key for key, value in presets.items()
        if isinstance(value, dict) and value.get("airspace", {}).get("dataset") == "SCAT"
    ]
    airspace = read_airspace(args.archive)
    failures = 0
    for key in targets:
        preset = presets[key]
        metadata = preset["airspace"]
        expected = preset["vertices"]
        actual = sector_bbox(
            airspace,
            metadata["centre"],
            metadata["sector"],
        )
        differences = {
            name: abs(float(expected[name]) - float(actual[name]))
            for name in ("lamin", "lamax", "lomin", "lomax")
        }
        passed = max(differences.values()) <= 1e-6
        print(
            f"{'PASS' if passed else 'FAIL'} {key}: "
            f"{metadata['centre']}/{metadata['sector']} "
            f"bbox={actual}"
        )
        if not passed:
            failures += 1
            print(f"  differences={differences}")
    if failures:
        raise SystemExit(f"{failures} preset(s) do not match airspace.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the existing OpenSky rule states only after SCAT exploration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT, update_read_me_first

ACQUISITION_DIR = PROJECT_ROOT / "src" / "acquisition"
if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))
import opensky_safety  # noqa: E402


def event_rows(features: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, group in features.groupby("window_id", sort=False):
        eligible = group.loc[group["predictor_eligible"].astype(bool) & (pd.to_numeric(group["relative_seconds"], errors="coerce") <= 0)]
        if eligible.empty:
            continue
        selected.append(eligible.iloc[pd.to_numeric(eligible["relative_seconds"], errors="coerce").abs().argmin()])
    return pd.DataFrame(selected).reset_index(drop=True)


def classify(horizontal, vertical, rules):
    if pd.isna(horizontal):
        return "UNKNOWN"
    return opensky_safety.classify_pair(float(horizontal), None if pd.isna(vertical) else float(vertical), state_rules=rules)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / "interaction_features.parquet")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "09_counterfactual" / "provisional_interaction_labels.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "10_opensky_comparison")
    parser.add_argument("--profile", default="ENROUTE")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "opensky_rule_comparison.csv"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}; use --overwrite")
    if not args.features_path.exists() or not args.labels_path.exists():
        raise FileNotFoundError("Run stages 05 and 09 before stage 10")
    features = event_rows(pd.read_parquet(args.features_path))
    labels = pd.read_parquet(args.labels_path)
    labels = labels[["clearance_id", "provisional_label", "reason_score", "confidence_score"]].drop_duplicates("clearance_id")
    events = features.merge(labels, on="clearance_id", how="left")
    profile_path = ACQUISITION_DIR / "separation_profiles.json"
    profile = opensky_safety.load_safety_profile(profile_path, args.profile)
    rules = profile["states"]
    events["opensky_current_state"] = [classify(h, v, rules) for h, v in zip(events["horizontal_separation_nm"], events["vertical_separation_ft"])]
    events["opensky_predicted_state"] = [classify(h, v, rules) for h, v in zip(events["predicted_min_horizontal_sep_nm"], events["predicted_min_vertical_sep_ft"])]
    columns = [
        "window_id", "clearance_id", "ATC_INTERVENTION", "provisional_label", "reason_score", "confidence_score",
        "horizontal_separation_nm", "vertical_separation_ft", "cpa_nm", "tcpa_min",
        "predicted_min_horizontal_sep_nm", "predicted_min_vertical_sep_ft",
        "opensky_current_state", "opensky_predicted_state",
    ]
    events[[column for column in columns if column in events]].to_csv(output_path, index=False)
    distribution = events.groupby(["provisional_label", "opensky_current_state"], dropna=False).size().reset_index(name="count")
    distribution.to_csv(args.output_root / "state_distribution.csv", index=False)
    mismatch = events.loc[
        events["provisional_label"].isin(["LIKELY_INTERACTION_RELATED", "LIKELY_NON_INTERACTION_OPERATIONAL"])
        & (events["opensky_current_state"] == "NORMAL")
    ]
    mismatch[[column for column in columns if column in mismatch]].to_csv(args.output_root / "mismatch_cases.csv", index=False)
    config = {
        "features_path": str(args.features_path),
        "labels_path": str(args.labels_path),
        "profile": args.profile,
        "rules_source": str(profile_path),
        "stage": "final comparison only",
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = [
        "# SCAT versus OpenSky rule comparison",
        "",
        f"- Event rows compared: **{len(events)}**",
        f"- OpenSky profile: **{args.profile}**",
        "",
        "This is the first stage that applies the existing GREEN/NORMAL, YELLOW/PROXIMITY, ORANGE/HIGH_RISK and RED/CONFLICT rule states. It is a comparison, not the target definition used in stages 01–09.",
        "Review `mismatch_cases.csv` for provisional interaction-related interventions that the existing rule profile classifies as NORMAL.",
        "The comparison cannot establish a confirmed conflict; it measures agreement or mismatch between two analytical views.",
    ]
    (args.output_root / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"OpenSky rule comparison complete: {args.output_root}")


if __name__ == "__main__":
    main()


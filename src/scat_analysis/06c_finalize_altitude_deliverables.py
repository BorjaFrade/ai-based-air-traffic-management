#!/usr/bin/env python3
"""Complete requested Stage 06 tables from existing first-week pilot outputs only.

This is a lightweight reporting pass. It never recomputes Stages 01–05, clustering,
or another SCAT week, and it does not execute Stage 07.
"""

from __future__ import annotations

import pandas as pd

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK, assert_pilot_frame


BANDS = ("VERY_LOW", "LOW", "MID", "CRUISE_FL290_410")
OUTPUT = DEFAULT_OUTPUT_ROOT / "06_exploration"
ROBUST_FEATURES = {
    "horizontal_sep_at_event", "min_horizontal_sep_within_30nm",
    "vertical_sep_at_event", "min_cpa_within_30nm",
    "min_tcpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
    "max_vertical_closing_within_30nm", "median_relative_heading",
    "median_crossing_angle", "median_relative_bearing", "median_traffic_density",
    "n_unique_neighbours_within_30nm", "median_nearby_aircraft_30nm",
    "median_relative_velocity_kt", "ground_speed_at_anchor_kt",
    "vertical_rate_at_anchor_fpm", "analysis_altitude_ft",
    "min_topsky_predicted_sep_nm_proxy",
    *(f"{axis}_sep_trend_{seconds}s" for axis in ("horizontal", "vertical") for seconds in (60, 120, 300, 600)),
}


def complete_matching(inventory: pd.DataFrame, balance: pd.DataFrame) -> None:
    cohort = pd.read_parquet(DEFAULT_OUTPUT_ROOT / "04_windows" / f"primary_matching_cohort_{PILOT_WEEK}.parquet")
    windows = pd.read_parquet(
        DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_v2_{PILOT_WEEK}.parquet",
        columns=["window_id", "archive_week", "altitude_band"],
    )
    assert_pilot_frame(cohort)
    assert_pilot_frame(windows)
    if len(cohort) != 10097 or len(windows) != 20194 or windows.window_id.duplicated().any():
        raise SystemExit("First-week source count or window uniqueness changed")
    wide = cohort.merge(windows[["window_id", "altitude_band"]], left_on="intervention_window_id", right_on="window_id", validate="one_to_one")
    wide = wide.merge(windows[["window_id", "altitude_band"]], left_on="control_window_id", right_on="window_id", suffixes=("_I", "_C"), validate="one_to_one")
    if len(wide) != len(cohort):
        raise SystemExit("A matched pair lost a V2 window")
    same = wide.altitude_band_I.eq(wide.altitude_band_C)
    if int(same.sum()) != 10086 or int((~same).sum()) != 11:
        raise SystemExit("The pilot same-band/cross-band split changed")
    inventory = inventory.set_index("band")
    rows = []
    for band in (*BANDS, "ABOVE_FL410"):
        item = inventory.loc[band]
        part = wide.loc[same & wide.altitude_band_I.eq(band)]
        if len(part) != int(item.valid_matched_pairs):
            raise SystemExit(f"Inventory and matched cohort disagree for {band}")
        if len(part) == 0:
            rows.append({"band": band, "intervention_windows": int(item.intervention_windows),
                         "control_windows": int(item.control_windows), "valid_matched_pairs": 0,
                         "cross_band_pairs_excluded": int(item.cross_band_excluded),
                         "unique_focal_flights_in_pairs": 0, "matching_status": "UNDETERMINED",
                         "largest_failed_covariate": "", "largest_failed_SMD": float("nan"),
                         "local_traffic_density_SMD": float("nan"),
                         "analysis_altitude_ft_SMD": float("nan"),
                         "ground_speed_at_anchor_kt_SMD": float("nan"),
                         "control_time_distance_median_s": float("nan"),
                         "control_time_distance_p95_s": float("nan"),
                         "control_time_distance_max_s": float("nan")})
            continue
        b = balance.loc[balance.band.eq(band) & ~balance.status.eq("DIAGNOSTIC")].copy()
        failing = b.loc[b.status.eq("FAIL")].sort_values("SMD", key=lambda x: x.abs(), ascending=False)
        status = "FAIL" if len(failing) else "WARNING" if b.status.eq("WARNING").any() else "PASS"
        time = pd.to_numeric(part.control_time_difference_s, errors="coerce")
        rows.append({"band": band, "intervention_windows": int(item.intervention_windows),
                     "control_windows": int(item.control_windows), "valid_matched_pairs": len(part),
                     "cross_band_pairs_excluded": int(item.cross_band_excluded),
                     "unique_focal_flights_in_pairs": int(item.unique_focal_flights),
                     "matching_status": status,
                     "largest_failed_covariate": failing.iloc[0].feature if len(failing) else "",
                     "largest_failed_SMD": failing.iloc[0].SMD if len(failing) else float("nan"),
                     "local_traffic_density_SMD": b.set_index("feature").loc["local_traffic_density", "SMD"],
                     "analysis_altitude_ft_SMD": b.set_index("feature").loc["analysis_altitude_ft", "SMD"],
                     "ground_speed_at_anchor_kt_SMD": b.set_index("feature").loc["ground_speed_at_anchor_kt", "SMD"],
                     "control_time_distance_median_s": time.median(),
                     "control_time_distance_p95_s": time.quantile(.95),
                     "control_time_distance_max_s": time.max()})
    summary = pd.DataFrame(rows)
    lines = ["# First-week Stage 06 — altitude-band matching", "",
             f"Scope: `FIRST_WEEK_PILOT`, archive week `{PILOT_WEEK}`. Only original primary intervention/control pairs are used; cross-band pairs are excluded, not re-matched. This report does not establish conflict or causal effects.", "",
             "| Band | I windows | C windows | Valid pairs | Cross-band excluded | Unique focal flights | Matching | Largest failed covariate (SMD) | Control time distance median / p95 |", "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for row in summary.itertuples(index=False):
        failure = f"{row.largest_failed_covariate} ({row.largest_failed_SMD:+.3f})" if row.largest_failed_covariate else "—"
        distance = f"{row.control_time_distance_median_s:.0f} / {row.control_time_distance_p95_s:.0f} s" if row.valid_matched_pairs else "—"
        lines.append(f"| {row.band} | {row.intervention_windows:,} | {row.control_windows:,} | {row.valid_matched_pairs:,} | {row.cross_band_pairs_excluded} | {row.unique_focal_flights_in_pairs:,} | {row.matching_status} | {failure} | {distance} |")
    lines.extend(["", "The time distance is `control_time_difference_s` from Stage 04; the primary cohort has no spatial matching-distance field. UTC hour, phase and centre are evaluated as category-specific standardized mean differences (SMDs), not only as means. The full [balance table](altitude_band_matching_balance.csv) includes their category shares and SMDs, plus local traffic density, anchor altitude/FL and ground speed. |SMD| < 0.10 = PASS; 0.10–0.20 = WARNING; > 0.20 = FAIL. The pre-event median speed is diagnostic only.", "",
                  "**Interpretation gate:** all four analysed bands have at least one FAIL. ABOVE_FL410 has zero same-band pairs and is UNDETERMINED. Intervention/control differences and cluster patterns are descriptive only; no band passes the matching screen for scientific effect interpretation.", "",
                  "[Machine-readable summary](altitude_band_matching_summary.csv) · [Cross-band pairs](cross_band_pairs.csv)", ""])
    summary.to_csv(OUTPUT / "altitude_band_matching_summary.csv", index=False)
    (OUTPUT / "altitude_band_matching_report.md").write_text("\n".join(lines), encoding="utf-8")


def complete_feature_tables() -> None:
    cruise_desc = pd.read_csv(OUTPUT / "cruise_FL290_410_descriptive_statistics.csv")
    cruise_comp = pd.read_csv(OUTPUT / "cruise_FL290_410_intervention_vs_control.csv")
    smd = cruise_comp.set_index("feature").smd
    cruise_desc["SMD"] = cruise_desc.feature.map(smd)
    cruise_desc.to_csv(OUTPUT / "descriptive_CRUISE_FL290_410.csv", index=False)
    for label, prefix in ((1, "intervention"), (0, "control")):
        grouped = cruise_desc.loc[cruise_desc.ATC_INTERVENTION.eq(label)].set_index("feature")
        cruise_comp[f"{prefix}_mean"] = cruise_comp.feature.map(grouped["mean"])
        cruise_comp[f"{prefix}_std"] = cruise_comp.feature.map(grouped["std"])
    cruise_comp.to_csv(OUTPUT / "cruise_FL290_410_intervention_vs_control.csv", index=False)
    all_features = pd.read_csv(OUTPUT / "altitude_band_feature_comparison.csv")
    all_features = all_features.rename(columns={c: c.replace("_effect", "_SMD") for c in all_features if c.endswith("_effect")})
    all_features.to_csv(OUTPUT / "altitude_band_feature_comparison.csv", index=False)
    top = []
    inventory = pd.read_csv(OUTPUT / "altitude_band_inventory.csv").set_index("band")
    for band in BANDS:
        path = OUTPUT / ("cruise_FL290_410_intervention_vs_control.csv" if band == "CRUISE_FL290_410" else f"intervention_vs_control_{band}.csv")
        comp = pd.read_csv(path)
        n = int(inventory.loc[band, "valid_matched_pairs"])
        allowed = comp.loc[comp.feature.isin(ROBUST_FEATURES) & comp.intervention_count.ge(30)
                           & comp.control_count.ge(30) & comp.smd.notna()
                           & comp.intervention_missing.le(.2 * n) & comp.control_missing.le(.2 * n)
                           & (comp.intervention_missing - comp.control_missing).abs().le(.1 * n)].copy()
        allowed = allowed.sort_values("smd", key=lambda x: x.abs(), ascending=False).head(10)
        for rank, row in enumerate(allowed.itertuples(index=False), 1):
            top.append({"band": band, "rank": rank, "feature": row.feature, "SMD": row.smd,
                        "intervention_median": row.intervention_median, "control_median": row.control_median})
    pd.DataFrame(top).to_csv(OUTPUT / "altitude_band_top_features.csv", index=False)
    composition = pd.read_csv(OUTPUT / "altitude_band_clearance_composition.csv")
    types = ("LEVEL", "HEADING", "SPEED", "OTHER")
    rows = []
    for band in BANDS:
        part = composition.loc[composition.band.eq(band)].set_index("clearance_type")
        count = int(part["count"].sum())
        for typ in types:
            n = int(part.loc[typ, "count"]) if typ in part.index else 0
            rows.append({"band": band, "clearance_type": typ, "count": n, "percentage": n / count * 100})
    pd.DataFrame(rows).to_csv(OUTPUT / "clearance_types_by_altitude_band.csv", index=False)


def complete_mapping() -> None:
    path = OUTPUT / "scat_to_opensky_feature_mapping.csv"
    mapping = pd.read_csv(path).set_index("SCAT_feature", drop=False)
    extra = {
        "far30_pair_fraction": ("time,icao24,lat,lon", "Fraction of aligned observed pairs beyond 30 NM", "NO", "SCAT fraction depends on its pair-selection procedure"),
        "horizontal_closing_rate_at_min_cpa": ("time,icao24,lat,lon,velocity,heading", "Line-of-sight closing at the selected kinematic CPA pair", "TEST", "Pair selection and time alignment must be checked"),
        "median_nearby_aircraft_30nm": ("time,icao24,lat,lon", "Median concurrent neighbour count within 30 NM", "TEST", "Requires full contemporaneous state sets"),
        "median_relative_velocity_kt": ("time,icao24,velocity,heading", "Median relative horizontal speed over selected pairs", "TEST", "Ground-track conversion and alignment required"),
        "min_horizontal_separation": ("time,icao24,lat,lon", "Minimum pairwise distance over a window", "NO", "Unbounded SCAT extreme; prefer within-30-NM variant"),
        "p05_horizontal_separation": ("time,icao24,lat,lon", "Fifth percentile pairwise distance over a window", "NO", "Original SCAT pair population included distant aircraft"),
        "p05_vertical_separation": ("time,icao24,baroaltitude", "Fifth percentile pairwise altitude difference", "TEST", "Pair-selection population must be defined"),
        "vertical_sep_at_min_cpa": ("time,icao24,baroaltitude,velocity,heading,lat,lon", "Vertical separation for the selected kinematic CPA pair", "TEST", "CPA is a constant-velocity proxy"),
        **{f"vertical_sep_trend_{s}s": ("time,icao24,baroaltitude", f"Current pair vertical separation minus that {s} seconds earlier", "TEST", "Requires same-pair historical coverage") for s in (60, 120, 300)},
    }
    for feature, (source, derivation, priority, limitation) in extra.items():
        if feature not in mapping.index:
            mapping.loc[feature] = {"SCAT_feature": feature, "importance_by_band": "",
                "available_in_opensky": "PARTIAL", "opensky_source_column": source,
                "derivable": "YES", "derivation": derivation,
                "recommended_for_opensky": priority, "limitations": limitation}
    measured = set(pd.read_csv(OUTPUT / "altitude_band_feature_comparison.csv").feature)
    if measured - set(mapping.index):
        raise SystemExit(f"OpenSky mapping is incomplete: {sorted(measured - set(mapping.index))}")
    proxy = {"flight_phase", "centre"}
    scat_only = {"min_topsky_predicted_sep_nm_proxy", "clearance_type", "ATC_INTERVENTION"}
    mapping["availability_class"] = [
        "PROXY_REQUIRED" if feature in proxy else
        "SCAT_ONLY" if feature in scat_only else
        "AVAILABLE_IN_OPENSKY" if available == "YES" else
        "DERIVABLE_FROM_OPENSKY" if derivable == "YES" else
        "SCAT_ONLY"
        for feature, available, derivable in zip(mapping.SCAT_feature, mapping.available_in_opensky, mapping.derivable)
    ]
    mapping.to_csv(path, index=False)


def main() -> None:
    if not OUTPUT.exists():
        raise SystemExit("Existing Stage 06 outputs are required")
    inventory = pd.read_csv(OUTPUT / "altitude_band_inventory.csv")
    balance = pd.read_csv(OUTPUT / "altitude_band_matching_balance.csv")
    complete_matching(inventory, balance)
    complete_feature_tables()
    complete_mapping()
    print("Completed Stage 06 deliverable schemas from existing FIRST_WEEK_PILOT outputs only; Stage 07 not run.", flush=True)


if __name__ == "__main__":
    main()

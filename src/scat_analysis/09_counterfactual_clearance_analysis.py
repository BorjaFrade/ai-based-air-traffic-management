#!/usr/bin/env python3
"""Estimate whether an intervention changed an interaction-related trajectory.

This is a transparent screening analysis, not a causal claim. It compares the
last pre-clearance kinematic prediction with the observed post-clearance pair
trajectory and keeps the resulting labels provisional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_ROOT, update_read_me_first


def mad_scale(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 1.0
    median = values.median()
    mad = (values - median).abs().median()
    return float(mad * 1.4826 if mad > 1e-9 else max(values.std(), 1.0))


def build_pair_counterfactuals(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    interventions = features.loc[features["ATC_INTERVENTION"] == 1].copy()
    for (clearance_id, other_key), group in interventions.groupby(["clearance_id", "other_flight_key"], sort=False):
        group = group.sort_values("relative_seconds", kind="mergesort")
        pre = group.loc[group["relative_seconds"] <= 0]
        post = group.loc[group["relative_seconds"] >= 0]
        if pre.empty or post.empty:
            continue
        anchor = pre.iloc[(pre["relative_seconds"].abs()).argmin()]
        post_min_h = pd.to_numeric(post["horizontal_separation_nm"], errors="coerce").min()
        post_min_v = pd.to_numeric(post["vertical_separation_ft"], errors="coerce").min()
        post_min_cpa = pd.to_numeric(post["cpa_nm"], errors="coerce").min()
        baseline_h = pd.to_numeric(pd.Series([anchor.get("predicted_min_horizontal_sep_nm")]), errors="coerce").iloc[0]
        baseline_v = pd.to_numeric(pd.Series([anchor.get("predicted_min_vertical_sep_ft")]), errors="coerce").iloc[0]
        baseline_cpa = pd.to_numeric(pd.Series([anchor.get("cpa_nm")]), errors="coerce").iloc[0]
        baseline_tcpa = pd.to_numeric(pd.Series([anchor.get("tcpa_min")]), errors="coerce").iloc[0]
        topsky = pd.to_numeric(pd.Series([anchor.get("topsky_predicted_min_sep_nm_proxy")]), errors="coerce").iloc[0]
        post_tcpa = pd.to_numeric(post["tcpa_min"], errors="coerce").median()
        post_closing = pd.to_numeric(post["horizontal_closing_rate_nm_min"], errors="coerce").median()
        anchor_closing = pd.to_numeric(pd.Series([anchor.get("horizontal_closing_rate_nm_min")]), errors="coerce").iloc[0]
        rows.append(
            {
                "clearance_id": clearance_id,
                "window_id": anchor["window_id"],
                "focal_flight_key": anchor["focal_flight_key"],
                "other_flight_key": other_key,
                "focal_callsign": anchor.get("focal_callsign", ""),
                "clearance_epoch": anchor.get("event_epoch", np.nan),
                "anchor_relative_seconds": anchor["relative_seconds"],
                "pre_rows": len(pre),
                "post_rows": len(post),
                "counterfactual_min_horizontal_sep_nm": baseline_h,
                "observed_post_min_horizontal_sep_nm": post_min_h,
                "delta_min_horizontal_sep_nm": post_min_h - baseline_h if pd.notna(post_min_h) and pd.notna(baseline_h) else np.nan,
                "counterfactual_min_vertical_sep_ft": baseline_v,
                "observed_post_min_vertical_sep_ft": post_min_v,
                "delta_min_vertical_sep_ft": post_min_v - baseline_v if pd.notna(post_min_v) and pd.notna(baseline_v) else np.nan,
                "counterfactual_cpa_nm": baseline_cpa,
                "observed_post_min_cpa_nm": post_min_cpa,
                "delta_CPA_nm": post_min_cpa - baseline_cpa if pd.notna(post_min_cpa) and pd.notna(baseline_cpa) else np.nan,
                "counterfactual_TCPA_min": baseline_tcpa,
                "observed_post_median_TCPA_min": post_tcpa,
                "delta_TCPA_min": post_tcpa - baseline_tcpa if pd.notna(post_tcpa) and pd.notna(baseline_tcpa) else np.nan,
                "delta_closing_rate_nm_min": anchor_closing - post_closing if pd.notna(anchor_closing) and pd.notna(post_closing) else np.nan,
                "counterfactual_topsky_min_sep_nm": topsky,
                "delta_predicted_interaction_nm": post_min_h - topsky if pd.notna(post_min_h) and pd.notna(topsky) else np.nan,
                "event_horizontal_separation_nm": anchor.get("horizontal_separation_nm", np.nan),
                "event_vertical_separation_ft": anchor.get("vertical_separation_ft", np.nan),
            }
        )
    return pd.DataFrame(rows)


def add_provisional_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    scales = {
        "delta_min_horizontal_sep_nm": mad_scale(frame["delta_min_horizontal_sep_nm"]),
        "delta_min_vertical_sep_ft": mad_scale(frame["delta_min_vertical_sep_ft"]),
        "delta_closing_rate_nm_min": mad_scale(frame["delta_closing_rate_nm_min"]),
    }
    components = []
    for column, scale in scales.items():
        median = pd.to_numeric(frame[column], errors="coerce").median()
        components.append((pd.to_numeric(frame[column], errors="coerce") - median) / scale)
    matrix = pd.concat(components, axis=1)
    frame["reason_score"] = matrix.mean(axis=1, skipna=True)
    frame["evidence_count"] = matrix.notna().sum(axis=1)
    valid = frame.loc[frame["evidence_count"] >= 2, "reason_score"].dropna()
    q25 = valid.quantile(0.25) if len(valid) else np.nan
    q75 = valid.quantile(0.75) if len(valid) else np.nan
    frame["provisional_label"] = "UNKNOWN"
    if pd.notna(q25) and pd.notna(q75) and q75 > q25:
        frame.loc[(frame["evidence_count"] >= 2) & (frame["reason_score"] >= q75), "provisional_label"] = "LIKELY_INTERACTION_RELATED"
        frame.loc[(frame["evidence_count"] >= 2) & (frame["reason_score"] <= q25), "provisional_label"] = "LIKELY_NON_INTERACTION_OPERATIONAL"
    frame["confidence_score"] = np.minimum(
        1.0,
        frame["reason_score"].abs() / max(abs(q75 - q25), 1.0) * frame["evidence_count"].clip(upper=3) / 3.0,
    )
    frame.attrs["q25"] = q25
    frame.attrs["q75"] = q75
    frame.attrs["scales"] = scales
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / "interaction_features.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "09_counterfactual")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "clearance_counterfactuals.parquet"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}; use --overwrite")
    if not args.features_path.exists():
        raise FileNotFoundError("Run 05_build_interaction_features.py first")
    features = pd.read_parquet(args.features_path)
    features["ATC_INTERVENTION"] = pd.to_numeric(features["ATC_INTERVENTION"], errors="coerce").fillna(0).astype(int)
    features["relative_seconds"] = pd.to_numeric(features["relative_seconds"], errors="coerce")
    counterfactuals = build_pair_counterfactuals(features)
    if counterfactuals.empty:
        raise RuntimeError("No intervention pairs contain both pre- and post-clearance observations")
    counterfactuals = add_provisional_labels(counterfactuals)
    counterfactuals.to_parquet(output_path, index=False, compression="zstd")
    labels = counterfactuals.sort_values(["clearance_id", "event_horizontal_separation_nm"], kind="mergesort").drop_duplicates("clearance_id")
    labels.to_parquet(args.output_root / "provisional_interaction_labels.parquet", index=False, compression="zstd")
    labels[["clearance_id", "focal_callsign", "other_flight_key", "provisional_label", "reason_score", "confidence_score"]].to_csv(args.output_root / "interaction_examples.csv", index=False)
    if len(counterfactuals):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            directory = args.output_root / "figures" / "before_after"
            directory.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            for label, group in labels.groupby("provisional_label"):
                ax.scatter(group["counterfactual_min_horizontal_sep_nm"], group["observed_post_min_horizontal_sep_nm"], s=18, alpha=0.7, label=label)
            limits = ax.get_xlim()
            ax.plot(limits, limits, "k--", alpha=0.5)
            ax.set_xlabel("Counterfactual predicted minimum H (NM)")
            ax.set_ylabel("Observed post-clearance minimum H (NM)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(directory / "counterfactual_vs_observed_horizontal.png", dpi=160)
            plt.close(fig)
        except ImportError:
            pass
    config = {
        "features_path": str(args.features_path),
        "counterfactual": "constant pre-clearance kinematic prediction compared with observed post-clearance pair trajectory",
        "labels": ["UNKNOWN", "LIKELY_INTERACTION_RELATED", "LIKELY_NON_INTERACTION_OPERATIONAL"],
        "quantile_thresholds": {"q25": counterfactuals.attrs.get("q25"), "q75": counterfactuals.attrs.get("q75")},
        "scales": counterfactuals.attrs.get("scales"),
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    counts = labels["provisional_label"].value_counts()
    report = [
        "# SCAT provisional counterfactual clearance analysis",
        "",
        f"- Clearance/nearby-aircraft pair cases: **{len(counterfactuals)}**",
        f"- Clearance-level representative labels: **{len(labels)}**",
        *[f"- `{label}`: **{int(counts.get(label, 0))}**" for label in ["UNKNOWN", "LIKELY_INTERACTION_RELATED", "LIKELY_NON_INTERACTION_OPERATIONAL"]],
        "",
        "The counterfactual is a transparent constant-kinematics screen, not a causal model. It asks whether the observed post-clearance trajectory improved relative to the last pre-clearance kinematic prediction.",
        "The labels are provisional and deliberately avoid `CONFLICT`/`NO_CONFLICT`. Their data-driven quantile thresholds and evidence counts are stored in `run_config.json`.",
    ]
    (args.output_root / "counterfactual_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Counterfactual clearance analysis complete: {args.output_root}")


if __name__ == "__main__":
    main()


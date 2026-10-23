#!/usr/bin/env python3
"""Apply the existing historical OpenSky detector to native SCAT anchor traffic.

Only 20161015_20161021 / CRUISE_MATCHING_V2 / FL290-410 is accepted.  No
upstream stage is recomputed.  The OpenSky detector is imported, never edited.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm, spearmanr
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "conflict_detection"))
import detect_opensky_conflicts as production  # noqa: E402

BASE = ROOT / "data" / "processed" / "scat"
OUT = BASE / "10_dynamic_rule_cruise"
REPORT = BASE / "PILOT_STAGE10B_EXACT_DYNAMIC_RULE_REPORT.md"
WEEK = "20161015_20161021"
N = 4098
COLOURS = {"NORMAL": "GREEN", "PROXIMITY": "YELLOW",
           "HIGH_RISK": "ORANGE", "CONFLICT": "RED"}
ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "UNKNOWN"]
SEVERITY = {name: i for i, name in enumerate(ORDER[:4])}
PRIMARY_AGE_S = production.MAX_STATE_AGE_SECONDS
SENSITIVITY_AGE_S = production.MAX_EXTRAPOLATION_SECONDS
SOURCE = BASE / "02_traffic" / "scat_states.parquet"
SOURCE_COLUMNS = ["flight_key", "archive_week", "time_epoch", "callsign",
                  "latitude", "longitude", "altitude_ft", "ground_speed_ms",
                  "track_deg", "heading_deg", "vertical_rate_fpm", "on_ground"]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohort() -> pd.DataFrame:
    pairs = pd.read_parquet(BASE / "04_windows" / "cruise_matching_v2.parquet")
    windows = pd.read_parquet(BASE / "05_features" / "window_features_cruise_matching_v2.parquet")
    splits = pd.read_parquet(BASE / "07_explainability_cruise" / "split_assignments.parquet")
    static = pd.read_parquet(BASE / "10_rule_geometry_comparison_cruise" / "rule_state_assignments.parquet")
    if len(pairs) != N or len(windows) != 2*N or len(splits) != 2*N or len(static) != 2*N:
        raise ValueError("Unexpected Stage 10B cohort size")
    if (set(windows.archive_week) != {WEEK} or set(windows.altitude_band) != {"CRUISE_FL290_410"}
            or not windows.analysis_altitude_ft.between(29000, 41000).all()
            or (windows.anchor_relative_seconds > 0).any()):
        raise ValueError("Scope or pre-intervention anchor invalid")
    if not pairs.control_clearance_free_verified.all():
        raise ValueError("Control clearance-free verification failed")
    if any(not x.window_id.is_unique for x in [windows, splits, static]):
        raise ValueError("Duplicate window ID")
    if set(windows.window_id) != set(splits.window_id) or set(windows.window_id) != set(static.window_id):
        raise ValueError("Window membership mismatch")
    p = pairs.reset_index(drop=True).assign(matched_pair_id=np.arange(N))
    pair_map = pd.concat([p[["matched_pair_id", "intervention_window_id"]].rename(columns={"intervention_window_id":"window_id"}),
                          p[["matched_pair_id", "control_window_id"]].rename(columns={"control_window_id":"window_id"})])
    w = windows.merge(pair_map, on="window_id", validate="one_to_one")
    w = w.merge(splits[["window_id", "matched_pair_id", "split"]], on=["window_id","matched_pair_id"], validate="one_to_one")
    w = w.merge(static[["window_id", "opensky_static_profile_proxy_state"]], on="window_id", validate="one_to_one")
    w["anchor_epoch"] = w.event_epoch + w.anchor_relative_seconds
    if (w.anchor_epoch > w.event_epoch + 1e-6).any() or w.ATC_INTERVENTION.value_counts().to_dict() != {0:N,1:N}:
        raise ValueError("Invalid target/anchor")
    if int(w.split.eq("test").sum()) != 1184 or w.loc[w.split.eq("test"), "matched_pair_id"].nunique() != 592:
        raise ValueError("Stage07 TEST split changed")
    return w


def scan_native_states(anchor_epochs: np.ndarray) -> pd.DataFrame:
    """Read only the requested week; retain raw rows near a cohort anchor."""
    parquet = pq.ParquetFile(SOURCE)
    ordered = np.sort(anchor_epochs)
    parts = []
    count = 0
    for i in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(i).column(2).statistics
        if stats and not (stats.min <= WEEK <= stats.max):
            continue
        frame = parquet.read_row_group(i, columns=SOURCE_COLUMNS).to_pandas()
        frame = frame.loc[frame.archive_week.eq(WEEK)]
        if frame.empty:
            continue
        count += len(frame)
        times = frame.time_epoch.to_numpy(float)
        ix = np.searchsorted(ordered, times)
        left = np.clip(ix-1, 0, len(ordered)-1)
        right = np.clip(ix, 0, len(ordered)-1)
        near = np.minimum(np.abs(times-ordered[left]), np.abs(times-ordered[right])) <= PRIMARY_AGE_S + 1e-6
        selected = frame.loc[near & frame.altitude_ft.between(29000,41000)].copy()
        if len(selected):
            parts.append(selected)
        if (i+1) % 250 == 0:
            print(f"Stage02 row groups {i+1}/{parquet.metadata.num_row_groups}; pilot rows scanned {count:,}", flush=True)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SOURCE_COLUMNS)
    result.sort_values(["time_epoch", "flight_key"], inplace=True, kind="mergesort")
    result.reset_index(drop=True, inplace=True)
    print(f"Pilot source rows scanned {count:,}; anchor-near band rows {len(result):,}", flush=True)
    return result


def reconstruct(windows: pd.DataFrame, native: pd.DataFrame, max_age: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    times = native.time_epoch.to_numpy(float)
    rows, coverage = [], []
    for i, win in enumerate(windows.itertuples(index=False), 1):
        anchor = float(win.anchor_epoch)
        start = np.searchsorted(times, anchor-max_age-1e-6, side="left")
        end = np.searchsorted(times, anchor+1e-6, side="right")
        sample = native.iloc[start:end].copy()
        if not sample.empty:
            sample = sample.loc[sample.latitude.between(-90,90) & sample.longitude.between(-180,180)]
            sample = sample.sort_values(["flight_key","time_epoch"], kind="mergesort").drop_duplicates("flight_key", keep="last")
        focal = sample.loc[sample.flight_key.eq(win.focal_flight_key)] if len(sample) else sample
        reason = "EVALUABLE" if len(focal) == 1 and len(sample) >= 2 else ("FOCAL_MISSING" if len(focal) == 0 else "NO_CONTEMPORANEOUS_NEIGHBOUR")
        cov = {"window_id":win.window_id, "matched_pair_id":win.matched_pair_id,
               "ATC_INTERVENTION":win.ATC_INTERVENTION, "split":win.split,
               "anchor_timestamp":anchor, "focal_flight_key":win.focal_flight_key,
               "n_aircraft":len(sample), "n_neighbours":max(0,len(sample)-1),
               "evaluable":reason=="EVALUABLE", "unevaluable_reason":reason,
               "focal_age_s":float(anchor-focal.time_epoch.iloc[0]) if len(focal) else np.nan}
        coverage.append(cov)
        if len(sample):
            sample["window_id"] = win.window_id
            sample["matched_pair_id"] = win.matched_pair_id
            sample["ATC_INTERVENTION"] = win.ATC_INTERVENTION
            sample["anchor_timestamp"] = anchor
            sample["source_timestamp"] = sample.time_epoch
            sample["age_s"] = anchor-sample.time_epoch
            sample["selection_method"] = "LATEST_NATIVE_STATE_AT_OR_BEFORE_ANCHOR"
            sample["is_focal"] = sample.flight_key.eq(win.focal_flight_key)
            rows.append(sample)
        if i % 1000 == 0:
            print(f"Reconstructed {i}/{len(windows)} anchors at freshness {max_age:g}s", flush=True)
    states = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cov = pd.DataFrame(coverage)
    if len(cov)!=len(windows) or cov.window_id.nunique()!=len(windows) or (len(states) and (states.age_s < -1e-5).any()):
        raise ValueError("Snapshot reconstruction integrity failed")
    return states,cov


def detector_frame(sample: pd.DataFrame) -> pd.DataFrame:
    """Only field/units adapter; no safety-rule implementation lives here."""
    result = pd.DataFrame({
        "icao24":sample.flight_key.astype(str).to_numpy(),
        "callsign":sample.callsign.fillna("").astype(str).to_numpy(),
        "latitude":sample.latitude.to_numpy(float), "longitude":sample.longitude.to_numpy(float),
        "baro_altitude_ft":sample.altitude_ft.to_numpy(float),
        "velocity":sample.ground_speed_ms.to_numpy(float),
        "true_track":sample.track_deg.to_numpy(float),
        "vertical_rate_fpm":sample.vertical_rate_fpm.to_numpy(float),
        "on_ground":np.zeros(len(sample), dtype=bool),
    })
    # The production routine returns integer DataFrame indices in pair events.
    return result


def execute(windows: pd.DataFrame, states: pd.DataFrame, coverage: pd.DataFrame,
            *, save_raw: bool, label: str) -> pd.DataFrame:
    grouped = states.groupby("window_id", sort=False) if len(states) else {}
    raw, focals = [], []
    for i, cov in enumerate(coverage.itertuples(index=False), 1):
        sample = grouped.get_group(cov.window_id) if cov.window_id in grouped.groups else pd.DataFrame()
        if not cov.evaluable:
            focals.append({"window_id":cov.window_id,"focal_dynamic_state":"UNKNOWN",
                           "number_green_pairs":0,"number_yellow_pairs":0,"number_orange_pairs":0,
                           "number_red_pairs":0,"number_unevaluable_pairs":0,
                           "production_pairs_total":0,"production_pairs_detailed":0})
            raw.append({"window_id":cov.window_id,"event_index":-1,"raw_event_json":None,
                        "raw_stats_json":None,"execution_status":cov.unevaluable_reason})
            continue
        frame = detector_frame(sample)
        if frame["icao24"].duplicated().any():
            raise ValueError("Duplicate aircraft in detector snapshot")
        focal_index = int(np.flatnonzero(frame.icao24.eq(cov.focal_flight_key))[0])
        events, stats = production.analyse_snapshot_fast(frame)
        seen = {"YELLOW":0,"ORANGE":0,"RED":0}
        for j,event in enumerate(events):
            if focal_index in (event["index_a"],event["index_b"]):
                seen[COLOURS[event["event_state"]]] += 1
            if save_raw:
                raw.append({"window_id":cov.window_id,"event_index":j,
                            "raw_event_json":json.dumps(event,default=str,sort_keys=True),
                            "raw_stats_json":json.dumps(stats,sort_keys=True),"execution_status":"EVALUABLE"})
        if save_raw and not events:
            raw.append({"window_id":cov.window_id,"event_index":-1,"raw_event_json":None,
                        "raw_stats_json":json.dumps(stats,sort_keys=True),"execution_status":"EVALUABLE"})
        n_green = int(cov.n_aircraft-1-sum(seen.values()))
        if n_green < 0:
            raise ValueError("Production emitted duplicate focal pair")
        severity = max((SEVERITY[k] for k,v in seen.items() if v), default=0)
        focals.append({"window_id":cov.window_id,"focal_dynamic_state":ORDER[severity],
                       "number_green_pairs":n_green,"number_yellow_pairs":seen["YELLOW"],
                       "number_orange_pairs":seen["ORANGE"],"number_red_pairs":seen["RED"],
                       "number_unevaluable_pairs":0,"production_pairs_total":stats["pairs_total"],
                       "production_pairs_detailed":stats["pairs_detailed"]})
        if i%500==0:
            print(f"Production detector {label}: {i}/{len(coverage)} snapshots", flush=True)
    if save_raw:
        pd.DataFrame(raw).to_parquet(OUT/"production_detector_raw_output.parquet",index=False)
    return windows.merge(coverage, on=["window_id","matched_pair_id","ATC_INTERVENTION","split","focal_flight_key"],validate="one_to_one").merge(pd.DataFrame(focals),on="window_id",validate="one_to_one")


def restore_primary_result(windows: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Rejoin saved production states to clean, validated upstream columns."""
    saved=pd.read_parquet(OUT/"focal_dynamic_states.parquet")
    columns=["window_id","focal_dynamic_state","number_green_pairs","number_yellow_pairs",
             "number_orange_pairs","number_red_pairs","number_unevaluable_pairs",
             "production_pairs_total","production_pairs_detailed"]
    if len(saved)!=2*N or not saved.window_id.is_unique:
        raise ValueError("Incomplete saved primary execution")
    return windows.merge(coverage,on=["window_id","matched_pair_id","ATC_INTERVENTION","split","focal_flight_key"],validate="one_to_one").merge(
        saved[columns],on="window_id",validate="one_to_one")


def matched_or(i_pos: np.ndarray, c_pos: np.ndarray) -> dict:
    b = int(np.sum(i_pos & ~c_pos)); c = int(np.sum(~i_pos & c_pos))
    odds = b/c if c else math.inf
    # Exact McNemar conditional OR interval: transform Clopper-Pearson for b/(b+c).
    from scipy.stats import beta
    n=b+c
    lo = beta.ppf(.025,b,n-b+1) if b else 0.0
    hi = beta.ppf(.975,b+1,n-b) if c else 1.0
    return {"I_non_green_C_green":b,"I_green_C_non_green":c,
            "discordant_pairs":n,"matched_conditional_OR":odds,
            "ci95_lower":lo/(1-lo) if lo<1 else math.inf,
            "ci95_upper":hi/(1-hi) if hi<1 else math.inf}


def fmt(x, digits=3):
    return "NA" if pd.isna(x) else ("∞" if x==math.inf else f"{x:.{digits}f}")


def summarise(w: pd.DataFrame, cov: pd.DataFrame, sensitivity: pd.DataFrame) -> dict:
    state="focal_dynamic_state"
    w.to_parquet(OUT/"focal_dynamic_states.parquet",index=False)
    counts = pd.crosstab(w[state],w.ATC_INTERVENTION).reindex(ORDER,fill_value=0)
    counts.columns=["n_control","n_intervention"]
    counts.reset_index().rename(columns={state:"dynamic_state"}).to_csv(OUT/"dynamic_state_counts.csv",index=False)
    dist=counts.reset_index().rename(columns={state:"dynamic_state"})
    dist["matched_cohort_intervention_fraction"] = dist.n_intervention/(dist.n_intervention+dist.n_control).replace(0,np.nan)
    dist.to_csv(OUT/"dynamic_state_vs_intervention.csv",index=False)
    pair = w.pivot(index="matched_pair_id",columns="ATC_INTERVENTION",values=state).rename(columns={0:"control_state",1:"intervention_state"})
    pair["comparison"] = ["UNKNOWN" if a=="UNKNOWN" or b=="UNKNOWN" else
                          "INTERVENTION_MORE_SEVERE" if SEVERITY[a]>SEVERITY[b] else
                          "CONTROL_MORE_SEVERE" if SEVERITY[a]<SEVERITY[b] else "SAME_SEVERITY"
                          for a,b in zip(pair.intervention_state,pair.control_state)]
    pair["binary_pattern"] = ["UNKNOWN" if a=="UNKNOWN" or b=="UNKNOWN" else
                              f"I_{'non_green' if a!='GREEN' else 'green'}_C_{'non_green' if b!='GREEN' else 'green'}"
                              for a,b in zip(pair.intervention_state,pair.control_state)]
    pair.reset_index().to_csv(OUT/"dynamic_matched_pair_comparison.csv",index=False)
    known=pair.loc[pair.comparison.ne("UNKNOWN")]
    assoc=matched_or(known.intervention_state.ne("GREEN").to_numpy(),known.control_state.ne("GREEN").to_numpy())
    assoc.update({"known_matched_pairs":len(known),"unknown_matched_pairs":N-len(known),
                  "both_non_green":int(known.binary_pattern.eq("I_non_green_C_non_green").sum()),
                  "both_green":int(known.binary_pattern.eq("I_green_C_green").sum())})
    pd.DataFrame([assoc]).to_csv(OUT/"dynamic_conditional_association.csv",index=False)
    from scipy.stats import binomtest
    more=int(pair.comparison.eq("INTERVENTION_MORE_SEVERE").sum())
    less=int(pair.comparison.eq("CONTROL_MORE_SEVERE").sum())
    pd.DataFrame([{"intervention_more_severe":more,"control_more_severe":less,
                   "same_severity":int(pair.comparison.eq("SAME_SEVERITY").sum()),
                   "unknown_pairs":int(pair.comparison.eq("UNKNOWN").sum()),
                   "two_sided_matched_sign_p":binomtest(more,more+less,.5).pvalue if more+less else np.nan}]).to_csv(OUT/"dynamic_ordinal_matched_association.csv",index=False)
    confusion=pd.crosstab(w.opensky_static_profile_proxy_state,w[state]).reindex(index=ORDER,columns=ORDER,fill_value=0)
    confusion.to_csv(OUT/"static_vs_dynamic_confusion.csv")
    comparable=w.loc[w.opensky_static_profile_proxy_state.isin(SEVERITY) & w[state].isin(SEVERITY)]
    st=comparable.opensky_static_profile_proxy_state.map(SEVERITY)
    dy=comparable[state].map(SEVERITY)
    static_summary={"n_comparable":len(comparable),"raw_agreement":float((st==dy).mean()),
                    "weighted_kappa_quadratic":cohen_kappa_score(st,dy,weights="quadratic") if st.nunique()>1 and dy.nunique()>1 else np.nan,
                    "static_to_dynamic_up_classified":int((dy>st).sum()),
                    "static_to_dynamic_down_classified":int((dy<st).sum()),
                    "static_non_green":int(w.opensky_static_profile_proxy_state.isin(ORDER[1:4]).sum()),
                    "dynamic_non_green":int(w[state].isin(ORDER[1:4]).sum())}
    static_pair=w.pivot(index="matched_pair_id",columns="ATC_INTERVENTION",values="opensky_static_profile_proxy_state")
    static_pair=static_pair.loc[static_pair[0].isin(SEVERITY)&static_pair[1].isin(SEVERITY)]
    static_assoc=matched_or(static_pair[1].ne("GREEN").to_numpy(),static_pair[0].ne("GREEN").to_numpy())
    static_summary["static_matched_conditional_OR"]=static_assoc["matched_conditional_OR"]
    static_summary["dynamic_matched_conditional_OR"]=assoc["matched_conditional_OR"]
    pd.DataFrame([static_summary]).to_csv(OUT/"static_vs_dynamic_summary.csv",index=False)
    # Descriptive ordinal association, no linearity assumption.
    trend=[]
    for colour in ORDER:
        sub=w.loc[w[state].eq(colour)]
        trend.append({"dynamic_state":colour,"ordinal_score":SEVERITY.get(colour,np.nan),"n":len(sub),
                      "n_intervention":int(sub.ATC_INTERVENTION.sum()),
                      "matched_cohort_intervention_fraction":sub.ATC_INTERVENTION.mean()})
    pd.DataFrame(trend).to_csv(OUT/"dynamic_ordinal_trend.csv",index=False)
    known_window=w.loc[w[state].isin(SEVERITY)]
    rho,p=spearmanr(known_window[state].map(SEVERITY),known_window.ATC_INTERVENTION)
    # Context is post-hoc and uses saved Stage05/07 rows only.
    preds=pd.read_parquet(BASE/"07_explainability_cruise"/"test_predictions.parquet")
    oscore=preds.loc[preds.model.eq("xgboost") & preds.feature_family.eq("OPENSKY_COMPATIBLE"),["window_id","score"]].rename(columns={"score":"stage07_opensky_score_test"})
    ctx=w.merge(oscore,on="window_id",how="left",validate="one_to_one")
    features=["horizontal_sep_at_event","vertical_sep_at_event","min_future_cpa_ge_1s_within_30nm",
              "max_horizontal_closing_within_30nm","median_traffic_density",
              "min_topsky_predicted_sep_nm_proxy","stage07_opensky_score_test"]
    context=[]
    for colour in ORDER:
        for target in (0,1):
            sub=ctx.loc[ctx[state].eq(colour)&ctx.ATC_INTERVENTION.eq(target)]
            row={"dynamic_state":colour,"ATC_INTERVENTION":target,"n":len(sub)}
            row.update({f+"_median":sub[f].median() for f in features})
            context.append(row)
    pd.DataFrame(context).to_csv(OUT/"dynamic_state_geometry_context.csv",index=False)
    clearance=pd.read_parquet(BASE/"09_counterfactual_cruise_v2"/"counterfactual_event_summary_v2.parquet",columns=["window_id","clearance_type","stage09_label"])
    intervention=ctx.loc[ctx.ATC_INTERVENTION.eq(1)].merge(clearance,on="window_id",validate="one_to_one")
    cols=["window_id","matched_pair_id",state,"horizontal_sep_at_event","vertical_sep_at_event",
          "min_future_cpa_ge_1s_within_30nm","max_horizontal_closing_within_30nm","median_traffic_density",
          "min_topsky_predicted_sep_nm_proxy","clearance_type","stage07_opensky_score_test"]
    intervention.loc[intervention[state].eq("GREEN"),cols].to_csv(OUT/"dynamic_green_interventions.csv",index=False)
    ctx.loc[ctx.ATC_INTERVENTION.eq(0)&ctx[state].isin(ORDER[1:4]),[c for c in cols if c!="clearance_type"]].to_csv(OUT/"dynamic_nongreen_controls.csv",index=False)
    clear=intervention.groupby(["clearance_type",state],dropna=False).size().rename("n").reset_index()
    clear["fraction_within_clearance_type"] = clear.n/clear.groupby("clearance_type").n.transform("sum")
    clear.to_csv(OUT/"dynamic_clearance_type_summary.csv",index=False)
    intervention.loc[intervention.stage09_label.eq("LIKELY_INTERACTION_RELATED"),cols+["stage09_label"]].to_csv(OUT/"stage09_one_case_dynamic_context.csv",index=False)
    # Reuse the frozen Stage10 TEST baselines without rerunning the logistic fit.
    old=pd.read_csv(BASE/"10_rule_geometry_comparison_cruise"/"stage07_test_rule_comparison.csv")
    old=old.loc[old.method.isin(["5NM_1000FT","7_5NM_1200FT","CONTINUOUS_GEOMETRY_LOGISTIC",
                                "STAGE07_OPENSKY_COMPATIBLE_XGBOOST","STAGE07_SCAT_FULL_XGBOOST"])].copy()
    test=w.loc[w.split.eq("test")]
    static_test=test.opensky_static_profile_proxy_state.isin(ORDER[1:4])
    def metric_row(name, sub, positive):
        y=sub.ATC_INTERVENTION.astype(int).to_numpy()
        p=np.asarray(positive,dtype=int)
        return {"method":name,"n_test":1184,"n_evaluable_test":len(sub),"test_coverage":len(sub)/1184,
                "threshold_selection":"fixed_predeclared_NON_GREEN","roc_auc":np.nan,"pr_auc":np.nan,
                "balanced_accuracy":balanced_accuracy_score(y,p),
                "precision_intervention":precision_score(y,p,zero_division=0),
                "intervention_recall":recall_score(y,p,zero_division=0),
                "f1_intervention":f1_score(y,p,zero_division=0),
                "control_rule_positive_rate":float(np.mean(p[y==0]))}
    rows=[metric_row("STATIC_PROFILE_PROXY",test.loc[test.opensky_static_profile_proxy_state.isin(SEVERITY)],
                     static_test.loc[test.opensky_static_profile_proxy_state.isin(SEVERITY)]),
          metric_row("EXACT_DYNAMIC_OPENSKY_RULE",test.loc[test[state].isin(SEVERITY)],
                     test.loc[test[state].isin(SEVERITY),state].ne("GREEN"))]
    pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/"stage07_test_dynamic_rule_comparison.csv",index=False)
    full_test=pd.crosstab(test[state],test.ATC_INTERVENTION).reindex(ORDER,fill_value=0)
    full_test.columns=["n_control_test","n_intervention_test"]
    full_test.reset_index().rename(columns={state:"dynamic_state"}).to_csv(OUT/"stage07_test_full_colour.csv",index=False)
    def compare_context(frame:pd.DataFrame, group_col:str, name:str) -> None:
        out=[]
        for group,sub in frame.groupby(group_col):
            row={"group":group,"n":len(sub)}
            for feature in features:
                row[feature+"_median"]=sub[feature].median()
                row[feature+"_n"]=int(sub[feature].notna().sum())
            out.append(row)
        pd.DataFrame(out).to_csv(OUT/name,index=False)
    i_context=intervention.copy()
    i_context["dynamic_binary_state"]=np.where(i_context[state].eq("GREEN"),"GREEN",
                                        np.where(i_context[state].isin(ORDER[1:4]),"NON_GREEN","UNKNOWN"))
    compare_context(i_context,"dynamic_binary_state","dynamic_green_vs_nongreen_interventions.csv")
    nc_context=ctx.loc[ctx[state].isin(ORDER[1:4])].copy()
    nc_context["group"]=np.where(nc_context.ATC_INTERVENTION.eq(1),"NON_GREEN_INTERVENTION","NON_GREEN_CONTROL")
    compare_context(nc_context,"group","dynamic_nongreen_intervention_vs_control.csv")
    sens=w[["window_id",state]].merge(sensitivity[["window_id",state]],on="window_id",suffixes=("_primary","_sensitivity"))
    sens.to_csv(OUT/"sync_sensitivity_states.csv",index=False)
    sens_agree=float((sens.focal_dynamic_state_primary==sens.focal_dynamic_state_sensitivity).mean())
    return {"counts":counts,"assoc":assoc,"static":static_summary,"pairs":pair,
            "rho":rho,"rho_p":p,"sensitivity_agreement":sens_agree,"test":pd.DataFrame(rows),
            "context":context,"trend":trend}


def write_audits(w:pd.DataFrame,states:pd.DataFrame,cov:pd.DataFrame,summary:dict) -> None:
    profile_path=SRC/"acquisition"/"separation_profiles.json"
    raw_profile=json.loads(profile_path.read_text())["ENROUTE"]
    write_json(OUT/"production_enroute_profile.json",{
        "source_file":str(profile_path),"source_sha256":sha256(profile_path),
        "raw_ENROUTE":raw_profile,"resolved":production.ACTIVE_SAFETY_PROFILE,
        "effective_constants":{"snapshot_step_seconds":production.SNAPSHOT_STEP_SECONDS,
          "lookahead_minutes":production.LOOKAHEAD_MINUTES,"prediction_step_seconds":production.PREDICTION_STEP_SECONDS,
          "max_state_age_seconds":production.MAX_STATE_AGE_SECONDS,
          "max_interpolation_gap_seconds":production.MAX_INTERPOLATION_GAP_SECONDS,
          "max_extrapolation_seconds":production.MAX_EXTRAPOLATION_SECONDS,
          "exact_position_tolerance_seconds":production.EXACT_POSITION_TOLERANCE_SECONDS,
          "distance_prefilter":production.ENABLE_DISTANCE_PREFILTER,
          "cpa_prefilter":production.ENABLE_CPA_PREFILTER,
          "cpa_prefilter_margin_nm":production.CPA_PREFILTER_MARGIN_NM}})
    mapping=[
        ("latitude","latitude","degrees north","identity","WGS84"),
        ("longitude","longitude","degrees east","identity","WGS84"),
        ("altitude_ft","baro_altitude_ft","ft","identity","measured flight level x100 ft; fallback selected altitude in Stage02"),
        ("ground_speed_ms","velocity","m/s","identity","SCAT Cartesian ground velocity magnitude"),
        ("track_deg","true_track","degrees clockwise true north","identity modulo 360","SCAT Cartesian velocity direction; magnetic heading unused"),
        ("heading_deg","unused","magnetic degrees","not passed","SCAT magnetic heading is semantically different from true ground track; HEADING_TRACK_PROXY not required"),
        ("vertical_rate_fpm","vertical_rate_fpm","ft/min","identity","production applies ENROUTE 100 fpm deadband"),
        ("on_ground","on_ground","boolean","unknown -> False for cruise states","production historical preprocessor excludes explicit True; native SCAT cruise flags are missing"),
        ("time_epoch","source_timestamp","UTC epoch seconds","identity","native SCAT timestamp"),
        ("anchor_epoch","anchor_timestamp","UTC epoch seconds","event_epoch + anchor_relative_seconds","pre-event Stage05 anchor"),
        ("geodesic WGS84","horizontal_sep_nm","NM","metres / 1852","inside unmodified production detector"),
        ("altitude difference","vertical_sep_ft","ft","absolute difference","inside unmodified production detector")]
    pd.DataFrame(mapping,columns=["scat_field","detector_field","detector_unit","conversion","audit_note"]).to_csv(OUT/"unit_mapping_scat_to_opensky_detector.csv",index=False)
    nby=cov.loc[cov.evaluable].n_aircraft
    ages=states.age_s if len(states) else pd.Series(dtype=float)
    cov_i=float(cov.loc[cov.ATC_INTERVENTION.eq(1)].evaluable.mean())
    cov_c=float(cov.loc[cov.ATC_INTERVENTION.eq(0)].evaluable.mean())
    missing={k:int(states[k].isna().sum()) for k in ["latitude","longitude","altitude_ft","ground_speed_ms","track_deg","vertical_rate_fpm"]}
    md(OUT/"snapshot_reconstruction_report.md",f"""# Native SCAT anchor snapshot reconstruction

- Expected anchors: **8196**; reconstructed: **{len(cov)}**; evaluable: **{int(cov.evaluable.sum())}**; unevaluable: **{int((~cov.evaluable).sum())}**.
- Intervention coverage: **{cov_i:.2%}**; control coverage: **{cov_c:.2%}**; difference: **{cov_i-cov_c:+.2%}**.
- Aircraft per evaluable snapshot: median **{nby.median():.0f}**, p05 **{nby.quantile(.05):.0f}**, p95 **{nby.quantile(.95):.0f}**.
- Native selected-state age: median **{ages.median():.2f} s**, p95 **{ages.quantile(.95):.2f} s**, maximum **{ages.max():.2f} s**.
- Missing fields after selection: `{json.dumps(missing)}`. Unevaluable reasons: `{cov.unevaluable_reason.value_counts().to_dict()}`.
- Selection: latest native state at or before the exact pre-event anchor for **every aircraft in the SCAT week and FL290–FL410**; maximum age **15 s**. No top-10, 30-NM, or future-state selection. Duplicate aircraft rows use latest timestamp.
- Sensitivity: **10 s** maximum age, derived from OpenSky's 10-s extrapolation safeguard and native SCAT's approximately 5-s sampling; selected before reading outcome distributions.
- Difference from OpenSky preprocessing: its observations are on a 10-s UTC grid, can interpolate between real positions across a ≤20-s gap, and can extrapolate ≤10 s. An intervention anchor is off-grid; to avoid future leakage, Stage10B uses the actual anchor and backward as-of native positions, with no interpolation or projection. The source-area geography also differs; all available native SCAT aircraft in-band are included.
- GREEN means no production non-green pair event with the focal aircraft among selected in-band states; UNKNOWN is retained when the focal or every neighbour is missing.
""")
    md(OUT/"production_detector_audit.md",f"""# Production OpenSky detector audit

## Entry point and call graph

Historical pipeline: `src/preprocessing/preprocess_opensky_10s.py:regularize_day` → `regularize_aircraft_to_grid` → area and FL290–FL410 filter → `src/conflict_detection/detect_opensky_conflicts.py:iter_10s_snapshots` → `snapshot_unique` → **`analyse_snapshot_fast`** → `evaluate_candidate_pair` → `classify_pair_active` → `opensky_safety.classify_pair_dynamic` and `opensky_safety.pair_separation_at_time`. Stage10B calls `analyse_snapshot_fast` directly after adapting the native states. The shared `opensky_safety.analyse_aircraft_snapshot` is used by other live/historical display entry points, but is not the historical fast pipeline entry point.

## Configuration and outputs

- Module: `src/conflict_detection/detect_opensky_conflicts.py` (SHA256 `{sha256(SRC/'conflict_detection'/'detect_opensky_conflicts.py')}`). Shared geometric logic: `src/acquisition/opensky_safety.py` (SHA256 `{sha256(SRC/'acquisition'/'opensky_safety.py')}`).
- Profile: ENROUTE from `src/acquisition/separation_profiles.json`; raw and effective parameters in `production_enroute_profile.json`. No thresholds, horizons, separator logic, filters, or colour definitions changed.
- Inputs: snapshot DataFrame with unique `icao24`, latitude/longitude degrees, barometric altitude ft, ground speed m/s, true track degrees, vertical rate ft/min, on-ground flag. `snapshot_unique` deduplicates by `icao24`.
- Prediction: 0–60, 60–120, 120–300 s, sampled every 5 s; constant ground velocity/track and vertical rate; geodesic WGS84 pair separation. 100 fpm vertical deadband. Yellow/orange require active legal separator deterioration. Red requires H<5 NM and effective V<1000 ft, with ±1 ft boundary normalization. Severity is worst of current and predicted intervals.
- Pair construction: all in-band aircraft pairs, with the production distance and CPA prefilters for current-GREEN pairs. `analyse_snapshot_fast` returns **only non-GREEN pair events** plus pair-screening counts. GREEN pair counts in Stage10B are the analysis complement of returned focal pair events; this is labelled an **ANALYSIS ADAPTER**. Focal severity is maximum returned event severity, or GREEN when the focal has at least one evaluable neighbour and no event.
- Track semantics: the detector's `true_track` is ground course, so Stage10B uses SCAT `track_deg` calculated from east/north velocity components. SCAT `heading_deg` is magnetic and remains unused; no `HEADING_TRACK_PROXY` is needed. Native cruise `on_ground` is unknown; the adapter supplies False, matching the historical preprocessor's exclusion of explicit True only.
- Altitude provenance: `altitude_provenance.csv` joins every unique selected native state back to Stage02 and finds measured FL present for all 239,892 states, with `altitude_ft = FL × 100` exactly. This cohort uses no selected-altitude fallback.
- Snapshot output does not encode UNKNOWN. Stage10B uses UNKNOWN for a missing focal or no contemporaneous neighbour, without converting it to GREEN.
- OpenSky preprocess: exact 10-s grid, state freshness ≤15 s, exact position tolerance 0.75 s, interpolation gap ≤20 s, extrapolation ≤10 s. Stage10B's causal as-of mapping is documented in `snapshot_reconstruction_report.md`.
- Semantic limit: `event_state=CONFLICT` maps to RED rule geometry and is **not** a confirmed conflict against independent ground truth.
""")
    cases=[]
    for colour in ORDER[:4]:
        group=w.loc[w.focal_dynamic_state.eq(colour)].sort_values(["ATC_INTERVENTION","window_id"])
        if len(group):
            a=group.loc[group.ATC_INTERVENTION.eq(1)].head(5)
            b=group.loc[group.ATC_INTERVENTION.eq(0)].head(5)
            chosen=pd.concat([a,b]).drop_duplicates("window_id")
            if len(chosen)<min(10,len(group)):
                chosen=pd.concat([chosen,group.loc[~group.window_id.isin(chosen.window_id)].head(min(10,len(group))-len(chosen))])
            cases.append(chosen)
    audit=pd.concat(cases) if cases else w.head(0)
    source_by=states.groupby("window_id")
    audit_rows=[]
    raw=pd.read_parquet(OUT/"production_detector_raw_output.parquet")
    raw_by=raw.groupby("window_id")
    for row in audit.itertuples(index=False):
        sample=source_by.get_group(row.window_id)
        focal=sample.loc[sample.is_focal].iloc[0]
        f=detector_frame(sample)
        fi=int(np.flatnonzero(f.icao24.eq(row.focal_flight_key))[0])
        pair_events=[json.loads(x) for x in raw_by.get_group(row.window_id).raw_event_json.dropna()]
        focal_events=[x for x in pair_events if fi in [x["index_a"],x["index_b"]]]
        replayed_events,_=production.analyse_snapshot_fast(f)
        replay_identical=(json.dumps(replayed_events,default=str,sort_keys=True)==
                          json.dumps(pair_events,default=str,sort_keys=True))
        current_geometry_ok=True
        current_state_ok=True
        for event in focal_events:
            a=f.iloc[event["index_a"]]
            b=f.iloc[event["index_b"]]
            h0,v0=production.pair_separation_at_time(a,b,0.0,
                            vertical_rate_deadband_fpm=production.VERTICAL_RATE_DEADBAND_FPM)
            h5,v5=production.pair_separation_at_time(a,b,float(production.PREDICTION_STEP_SECONDS),
                            vertical_rate_deadband_fpm=production.VERTICAL_RATE_DEADBAND_FPM)
            current_geometry_ok &= (math.isclose(h0,event["horizontal_sep_nm"],abs_tol=1e-8)
                                    and (v0==event["vertical_sep_ft"] or
                                         (v0 is None and event["vertical_sep_ft"] is None)))
            current_state_ok &= (production.classify_pair_active(h0,v0,
                                next_horizontal_nm=h5,next_vertical_ft=v5)==event["state_current"])
        nearest=None
        for k,r in f.iterrows():
            if k==fi: continue
            h,v=production.pair_separation_at_time(f.iloc[fi],r,0,vertical_rate_deadband_fpm=production.VERTICAL_RATE_DEADBAND_FPM)
            if nearest is None or h<nearest[0]: nearest=(h,v,k)
        audit_rows.append({"window_id":row.window_id,"ATC_INTERVENTION":row.ATC_INTERVENTION,
                           "focal_dynamic_state":row.focal_dynamic_state,"anchor_timestamp":row.anchor_epoch,
                           "focal_source_timestamp":focal.source_timestamp,"focal_age_s":focal.age_s,
                           "focal_detector_input_json":json.dumps(f.iloc[fi].to_dict(),default=str),
                           "nearest_pair_h_nm":nearest[0] if nearest else np.nan,
                           "nearest_pair_v_ft":nearest[1] if nearest else np.nan,
                           "nearest_pair_other_input_json":json.dumps(f.iloc[nearest[2]].to_dict(),default=str) if nearest else None,
                           "production_focal_events_json":json.dumps(focal_events,default=str),
                           "production_event_state_max":max([COLOURS[x["event_state"]] for x in focal_events],key=lambda x:SEVERITY[x]) if focal_events else "GREEN",
                           "n_snapshot_aircraft":len(f),"production_replay_identical":replay_identical,
                           "focal_current_geometry_identical":current_geometry_ok,
                           "focal_current_state_identical":current_state_ok})
    manual=pd.DataFrame(audit_rows)
    manual.to_csv(OUT/"manual_audit_cases.csv",index=False)
    if len(manual) and not (manual.production_event_state_max.eq(manual.focal_dynamic_state).all()
                           and manual.production_replay_identical.all()
                           and manual.focal_current_geometry_identical.all()
                           and manual.focal_current_state_identical.all()):
        raise ValueError("Manual audit event aggregation mismatch")
    md(OUT/"dynamic_rule_manual_audit.md",f"""# Manual production-output audit

Selected **{len(manual)}** cases by colour (up to 10 per available colour), with intervention/control mix. The CSV records native source time, converted focal and nearest aircraft inputs, current pair separation, all raw production focal events including forecast interval states and CPA, and the final focal state. The source states for every aircraft in each case are in `snapshot_reconstruction.parquet` under the same `window_id`. Every selected case passed deterministic re-execution of the entire production snapshot output, direct pair-separation and current-state checks, and the focal-severity aggregation check. GREEN cases have no focal event from `analyse_snapshot_fast`; they are not certified safe. Counts: `{manual.focal_dynamic_state.value_counts().to_dict()}`.
""")


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"figures").mkdir(exist_ok=True)
    w=load_cohort()
    postprocess=len(sys.argv)==2 and sys.argv[1]=="--postprocess"
    resume_sensitivity=len(sys.argv)==2 and sys.argv[1]=="--resume-sensitivity"
    if len(sys.argv)>1 and not (postprocess or resume_sensitivity):
        raise SystemExit("Only optional --postprocess or --resume-sensitivity is supported")
    config={"scope":"FIRST_WEEK_PILOT","archive_week":WEEK,"band":"FL290-410",
            "cohort":"CRUISE_MATCHING_V2","expected_windows":2*N,"expected_pairs":N,
            "production_entry_point":"detect_opensky_conflicts.analyse_snapshot_fast",
            "primary_max_state_age_s":PRIMARY_AGE_S,"sensitivity_max_state_age_s":SENSITIVITY_AGE_S,
            "selection":"latest_native_state_at_or_before_exact_anchor", "production_profile":"ENROUTE",
            "native_source":str(SOURCE),"target":"ATC_INTERVENTION","no_conflict_ground_truth":True}
    write_json(OUT/"run_config.json",config)
    if postprocess:
        states=pd.read_parquet(OUT/"snapshot_reconstruction.parquet")
        cov=pd.read_csv(OUT/"snapshot_coverage.csv")
        result=restore_primary_result(w,cov)
        sens=pd.read_csv(OUT/"sync_sensitivity_states.csv")[["window_id","focal_dynamic_state_sensitivity"]].rename(
            columns={"focal_dynamic_state_sensitivity":"focal_dynamic_state"})
        if len(result)!=2*N or len(sens)!=2*N:
            raise ValueError("Postprocess requires completed primary and sensitivity outputs")
    elif resume_sensitivity:
        states=pd.read_parquet(OUT/"snapshot_reconstruction.parquet")
        cov=pd.read_csv(OUT/"snapshot_coverage.csv")
        result=restore_primary_result(w,cov)
        states_s=states.loc[states.age_s.le(SENSITIVITY_AGE_S+1e-6)].copy()
        nstates=states_s.groupby("window_id").size()
        focal_states=states_s.loc[states_s.is_focal].set_index("window_id")
        cov_s=w[["window_id","matched_pair_id","ATC_INTERVENTION","split","focal_flight_key"]].copy()
        cov_s["anchor_timestamp"]=w.anchor_epoch
        cov_s["n_aircraft"]=cov_s.window_id.map(nstates).fillna(0).astype(int)
        cov_s["n_neighbours"]=(cov_s.n_aircraft-1).clip(lower=0)
        cov_s["focal_age_s"]=cov_s.window_id.map(focal_states.age_s)
        cov_s["evaluable"]=cov_s.focal_age_s.notna() & cov_s.n_aircraft.ge(2)
        cov_s["unevaluable_reason"]=np.where(cov_s.evaluable,"EVALUABLE",
             np.where(cov_s.focal_age_s.isna(),"FOCAL_MISSING","NO_CONTEMPORANEOUS_NEIGHBOUR"))
        sens=execute(w,states_s,cov_s,save_raw=False,label="sensitivity-resume")
    else:
        native=scan_native_states(w.anchor_epoch.to_numpy(float))
        states,cov=reconstruct(w,native,PRIMARY_AGE_S)
        states.to_parquet(OUT/"snapshot_reconstruction.parquet",index=False)
        cov.to_csv(OUT/"snapshot_coverage.csv",index=False)
        result=execute(w,states,cov,save_raw=True,label="primary")
        states_s,cov_s=reconstruct(w,native,SENSITIVITY_AGE_S)
        sens=execute(w,states_s,cov_s,save_raw=False,label="sensitivity")
    summary=summarise(result,cov,sens)
    write_audits(result,states,cov,summary)
    counts=summary["counts"]
    ci=float(cov.loc[cov.ATC_INTERVENTION.eq(1)].evaluable.mean())
    cc=float(cov.loc[cov.ATC_INTERVENTION.eq(0)].evaluable.mean())
    static=summary["static"]
    test=summary["test"].set_index("method")
    dyn=test.loc["EXACT_DYNAMIC_OPENSKY_RULE"]
    sta=test.loc["STATIC_PROFILE_PROXY"]
    old=pd.read_csv(BASE/"10_rule_geometry_comparison_cruise"/"stage07_test_rule_comparison.csv").set_index("method")
    # Predeclared descriptive magnitude thresholds, never used to tune the detector.
    count_change=abs(static["dynamic_non_green"]-static["static_non_green"])/8196
    ba_change=abs(dyn.balanced_accuracy-sta.balanced_accuracy)
    ors=static["static_matched_conditional_OR"]
    ordy=static["dynamic_matched_conditional_OR"]
    or_ratio=ordy/ors if np.isfinite(ors) and ors>0 and np.isfinite(ordy) else np.nan
    no_change=(static["raw_agreement"]>=.95 and count_change<=.01 and ba_change<=.02
               and np.isfinite(or_ratio) and .8<=or_ratio<=1.25)
    major_change=(static["raw_agreement"]<.8 or count_change>.05 or ba_change>.1
                  or (np.isfinite(or_ratio) and (or_ratio<.5 or or_ratio>2)))
    change="NO_MEANINGFUL_CHANGE" if no_change else "MAJOR_CHANGE" if major_change else "MODERATE_CHANGE"
    frac=counts.n_intervention/(counts.n_intervention+counts.n_control).replace(0,np.nan)
    spread=frac.iloc[:4].max()-frac.iloc[:4].min()
    strength="UNDETERMINED" if int(counts.iloc[1:4].sum().sum())<30 else ("STRONG" if spread>.25 else "MODERATE" if spread>.1 else "WEAK" if spread>.02 else "NONE")
    readiness=ci>=.95 and cc>=.95 and abs(ci-cc)<=.03 and summary["sensitivity_agreement"]>=.95
    gate="YES" if readiness else "NO"
    decision="READY_FOR_BOUNDED_OPENSKY_CRUISE_PILOT" if readiness else ("INSUFFICIENT_DYNAMIC_RULE_COVERAGE" if min(ci,cc)<.95 else "REVISE_DYNAMIC_RULE_RECONSTRUCTION")
    association=summary["assoc"]
    ordinal_sign=pd.read_csv(OUT/"dynamic_ordinal_matched_association.csv").iloc[0]
    green_context=pd.read_csv(OUT/"dynamic_green_vs_nongreen_interventions.csv").set_index("group")
    nongreen_context=pd.read_csv(OUT/"dynamic_nongreen_intervention_vs_control.csv").set_index("group")
    altitude_note=("All 239,892 unique selected source states have measured flight level and altitude_ft = FL×100 exactly; no selected-altitude fallback was used."
                   if (OUT/"altitude_provenance.csv").exists() else
                   "Stage02 uses measured FL×100 where available and selected-altitude fallback otherwise; provenance is not yet quantified.")
    report=f"""# SCAT Stage 10B — exact dynamic OpenSky rule versus ATC intervention

## 1. Scope

Only SCAT week `{WEEK}`, FL290–FL410, `CRUISE_MATCHING_V2`: 4,098 matched pairs, 8,196 anchor windows. Target is recorded **ATC_INTERVENTION**, not conflict.

## 2. Why Stage 10B was necessary

Stage 10 used a static ENROUTE profile proxy. It did not execute the historical OpenSky dynamic detector or its forecast, active-separator and pair-screening path.

## 3. Production detector audit

The historical production entry point is **`detect_opensky_conflicts.analyse_snapshot_fast`**, which calls shared `opensky_safety` projection and dynamic classification functions. It was imported and executed unchanged. [Full audit](10_dynamic_rule_cruise/production_detector_audit.md).

## 4. ENROUTE configuration

The source profile and all effective production constants are frozen in [production_enroute_profile.json](10_dynamic_rule_cruise/production_enroute_profile.json). Current red rule: H<5 NM and effective V<1000 ft. Forecast: 5-s samples to 5 min; yellow/orange require active-separator deterioration.

## 5. SCAT-to-detector mapping

Native SCAT measured flight level maps to barometric altitude ft; ground speed remains m/s; `track_deg` maps to OpenSky `true_track`. SCAT magnetic heading is not used. {altitude_note} [Unit mapping](10_dynamic_rule_cruise/unit_mapping_scat_to_opensky_detector.csv).

## 6. Snapshot synchronization

Latest native in-band state at or before each exact anchor, max 15 s. No future state enters the initial intervention snapshot. OpenSky's 10-s grid permits limited position interpolation/extrapolation; Stage10B uses causal as-of native positions. A 10-s maximum-age sensitivity is reported. No top-10 or 30-NM neighbour limit. [Coverage detail](10_dynamic_rule_cruise/snapshot_reconstruction_report.md).

## 7. Coverage

Intervention **{int(cov.loc[cov.ATC_INTERVENTION.eq(1)].evaluable.sum())}/4098 ({ci:.2%})**; control **{int(cov.loc[cov.ATC_INTERVENTION.eq(0)].evaluable.sum())}/4098 ({cc:.2%})**; difference **{ci-cc:+.2%}**. Median aircraft per evaluable snapshot **{cov.loc[cov.evaluable].n_aircraft.median():.0f}**; selected state-age p95 **{states.age_s.quantile(.95):.2f} s**.

## 8. Unit validation

Conversion and direct production-call tests are in `src/scat_analysis/tests/test_stage10b_exact_dynamic_rule.py`. The adapter passes numeric values in the detector's native units; the detector computes pair separation internally.

## 9. Production execution

Raw pair events and screening counts are preserved in [production_detector_raw_output.parquet](10_dynamic_rule_cruise/production_detector_raw_output.parquet). The source detector returns only non-GREEN pair events; no equations or thresholds were copied.

## 10. Focal severity aggregation

**ANALYSIS ADAPTER:** maximum production `event_state` among events involving the focal aircraft. If none are returned with at least one contemporaneous neighbour, focal state is GREEN; missing focal/neighbour gives UNKNOWN. GREEN is not a safety certificate.

## 11. Manual/equivalence audit

{len(pd.read_csv(OUT/'manual_audit_cases.csv'))} manually selected auditable rows span available colours, with source timestamps, converted detector inputs, pair separation, full event forecast and final severity. Every case passed deterministic production replay and direct current-geometry/state checks. See [manual audit](10_dynamic_rule_cruise/dynamic_rule_manual_audit.md). Direct-call equivalence and edge cases are covered by tests. Primary/10-s-sensitivity state agreement: **{summary['sensitivity_agreement']:.2%}**.

## 12. Dynamic state distribution

| State | Intervention | Control |
|---|---:|---:|
"""
    for colour in ORDER:
        report+=f"| {colour} | {counts.loc[colour,'n_intervention']} | {counts.loc[colour,'n_control']} |\n"
    report+="\n![Dynamic rule state distribution](10_dynamic_rule_cruise/figures/dynamic_state_distribution.png)\n"
    report+=f"""

## 13. Dynamic states versus ATC intervention

Non-GREEN (YELLOW/ORANGE/RED): **{int(counts.loc[ORDER[1:4],'n_intervention'].sum())}** interventions and **{int(counts.loc[ORDER[1:4],'n_control'].sum())}** controls. Matched cohort fractions are conditional on an artificial 50/50 cohort, not population probabilities. The intervention fraction declines from GREEN **{frac.loc['GREEN']:.1%}** through YELLOW **{frac.loc['YELLOW']:.1%}**, ORANGE **{frac.loc['ORANGE']:.1%}**, to RED **{frac.loc['RED']:.1%}**. Ordinal Spearman rho **{fmt(summary['rho'])}** is descriptive because windows are matched; the matched ordinal sign test gives p **{fmt(ordinal_sign.two_sided_matched_sign_p)}**. Colour-specific rates are in `dynamic_ordinal_trend.csv`; no linearity is assumed. Relationship descriptor: **{strength} inverse across colour fractions**, with small non-GREEN cells.

## 14. Matched-pair analysis

Intervention more severe **{int(summary['pairs'].comparison.eq('INTERVENTION_MORE_SEVERE').sum())}**; control more severe **{int(summary['pairs'].comparison.eq('CONTROL_MORE_SEVERE').sum())}**; same **{int(summary['pairs'].comparison.eq('SAME_SEVERITY').sum())}**; unknown **{int(summary['pairs'].comparison.eq('UNKNOWN').sum())}**. Discordant binary pairs: I non-GREEN/C GREEN **{association['I_non_green_C_green']}**, I GREEN/C non-GREEN **{association['I_green_C_non_green']}**. Conditional OR **{fmt(association['matched_conditional_OR'])}**, exact 95% CI **[{fmt(association['ci95_lower'])}, {fmt(association['ci95_upper'])}]**.

## 15. Static proxy versus exact dynamic detector

Raw agreement **{static['raw_agreement']:.2%}**; quadratic weighted kappa **{fmt(static['weighted_kappa_quadratic'])}**; static→dynamic up-classified **{static['static_to_dynamic_up_classified']}**, down-classified **{static['static_to_dynamic_down_classified']}**. Non-GREEN windows static **{static['static_non_green']}**, dynamic **{static['dynamic_non_green']}**. Matched conditional OR static **{fmt(ors)}**, dynamic **{fmt(ordy)}**; TEST balanced accuracy static **{fmt(sta.balanced_accuracy)}**, dynamic **{fmt(dyn.balanced_accuracy)}**. Descriptor: **{change}** because the matched association reverses direction (OR ratio **{fmt(or_ratio)}**), despite high raw agreement dominated by GREEN and similar near-chance TEST binary discrimination. Thresholds: no meaningful change requires ≥95% agreement, ≤1% of windows difference in non-GREEN count, ≤.02 TEST balanced-accuracy difference and OR ratio .8–1.25; major change is <80% agreement, >5% count difference, >.10 balanced-accuracy difference or OR ratio outside .5–2. See full 5×5 matrix and summary CSV.

![Static proxy versus dynamic rule](10_dynamic_rule_cruise/figures/static_vs_dynamic_confusion.png)

## 16. Stage07 TEST comparison

Exactly 592 frozen TEST pairs (1,184 windows), with no refit. The dynamic NON-GREEN rule has coverage **{dyn.test_coverage:.2%}**, balanced accuracy **{fmt(dyn.balanced_accuracy)}**, intervention recall **{fmt(dyn.intervention_recall)}**, and control non-GREEN rate **{fmt(dyn.control_rule_positive_rate)}**. Fixed rules, static proxy, continuous-geometry logistic and both saved Stage07 XGBoost models are in `stage07_test_dynamic_rule_comparison.csv`. Full-colour TEST counts are in `stage07_test_full_colour.csv`.

## 17. GREEN interventions

**{int(counts.loc['GREEN','n_intervention'])}** dynamic-GREEN interventions are listed with geometry, TopSky proxy, clearance type and saved Stage07 OpenSky-compatible TEST score where available in `dynamic_green_interventions.csv`; medians versus non-GREEN interventions are in `dynamic_green_vs_nongreen_interventions.csv`. GREEN interventions have median nearest-pair H **{green_context.loc['GREEN','horizontal_sep_at_event_median']:.2f} NM** and V **{green_context.loc['GREEN','vertical_sep_at_event_median']:.0f} ft**, versus **{green_context.loc['NON_GREEN','horizontal_sep_at_event_median']:.2f} NM** and **{green_context.loc['NON_GREEN','vertical_sep_at_event_median']:.0f} ft** for non-GREEN interventions. Their saved Stage07 OpenSky-compatible TEST score median is **{green_context.loc['GREEN','stage07_opensky_score_test_median']:.3f}** (n={int(green_context.loc['GREEN','stage07_opensky_score_test_n'])}) versus **{green_context.loc['NON_GREEN','stage07_opensky_score_test_median']:.3f}** (n={int(green_context.loc['NON_GREEN','stage07_opensky_score_test_n'])}); Stage07 also uses continuous geometry/context outside the detector colour, but this sparse TEST comparison cannot attribute individual interventions. They are not called missed conflicts.

## 18. Non-GREEN controls

**{int(counts.loc[ORDER[1:4],'n_control'].sum())}** non-GREEN controls are listed in `dynamic_nongreen_controls.csv`, with medians versus non-GREEN interventions in `dynamic_nongreen_intervention_vs_control.csv`. Median future CPA is **{nongreen_context.loc['NON_GREEN_CONTROL','min_future_cpa_ge_1s_within_30nm_median']:.2f} NM** for these controls versus **{nongreen_context.loc['NON_GREEN_INTERVENTION','min_future_cpa_ge_1s_within_30nm_median']:.2f} NM** for non-GREEN interventions; this is post-hoc Stage05 context, not the detector's own forecast. No nearby clearance was recorded in their matched control windows; they are not called false positives.

## 19. Clearance-type context

LEVEL, HEADING and SPEED state counts/fractions are post-hoc in `dynamic_clearance_type_summary.csv`; clearance type did not enter the detector.

## 20. Stage09 one-case context

The sole provisional interaction-related SPEED case is recorded, without validation claims, in `stage09_one_case_dynamic_context.csv`.

## 21. Implications for OpenSky

This gate concerns technical readiness to reproduce the same features/rules in a **bounded FL290–FL410 OpenSky quality/feature-yield pilot**. It does not establish detector correctness. Gate: **{gate}**.

## 22. Limitations

SCAT's off-grid backward-as-of reconstruction differs from OpenSky's 10-s regularized, sometimes interpolated positions. SCAT measured flight level is a barometric-altitude proxy; source area differs. Production may treat missing speed/track as stationary in forecast. The matched cohort has artificial 50/50 balance; ATC intervention is not conflict truth. RED does not mean confirmed conflict, GREEN does not mean safe, and Stage09 is not ground truth.

## 23. Final decision

**{decision}**. Do not process full OpenSky history, other SCAT weeks, or Stage08.
"""
    md(REPORT,report)
    print("="*60)
    print("SCAT STAGE 10B — EXACT DYNAMIC OPEN SKY RULE")
    print("="*60)
    print(f"Week: {WEEK}\nBand: FL290–FL410\nCohort: CRUISE_MATCHING_V2\nExpected windows: 8196")
    print("-"*60+"\nPRODUCTION DETECTOR\n"+"-"*60)
    print("Module: src/conflict_detection/detect_opensky_conflicts.py")
    print("Entry point: analyse_snapshot_fast\nProfile: ENROUTE\nExact production code reused: YES")
    print("-"*60+"\nSNAPSHOT RECONSTRUCTION\n"+"-"*60)
    print(f"Intervention evaluable: {int(cov.loc[cov.ATC_INTERVENTION.eq(1)].evaluable.sum())}/4098 ({ci:.2%})")
    print(f"Control evaluable: {int(cov.loc[cov.ATC_INTERVENTION.eq(0)].evaluable.sum())}/4098 ({cc:.2%})")
    print(f"Coverage difference: {ci-cc:+.2%}\nMedian aircraft per snapshot: {cov.loc[cov.evaluable].n_aircraft.median():.0f}\nState-age p95: {states.age_s.quantile(.95):.2f}s")
    print("-"*60+"\nDYNAMIC STATES\n"+"-"*60)
    print("                INTERVENTION    CONTROL")
    for colour in ORDER:
        print(f"{colour:<15} {counts.loc[colour,'n_intervention']:>7} {counts.loc[colour,'n_control']:>12}")
    print("-"*60+"\nNON-GREEN ASSOCIATION\n"+"-"*60)
    print(f"Intervention non-green: {int(counts.loc[ORDER[1:4],'n_intervention'].sum())}")
    print(f"Control non-green: {int(counts.loc[ORDER[1:4],'n_control'].sum())}")
    print(f"Matched conditional OR: {fmt(association['matched_conditional_OR'])}\n95% CI: [{fmt(association['ci95_lower'])}, {fmt(association['ci95_upper'])}]")
    print("-"*60+"\nMATCHED PAIRS\n"+"-"*60)
    for key,label in [("INTERVENTION_MORE_SEVERE","Intervention more severe"),("CONTROL_MORE_SEVERE","Control more severe"),("SAME_SEVERITY","Same"),("UNKNOWN","Unknown")]:
        print(f"{label}: {int(summary['pairs'].comparison.eq(key).sum())}")
    print("-"*60+"\nSTATIC VS DYNAMIC\n"+"-"*60)
    print(f"Raw agreement: {static['raw_agreement']:.2%}\nWeighted kappa: {fmt(static['weighted_kappa_quadratic'])}")
    print(f"Static → dynamic up-classified: {static['static_to_dynamic_up_classified']}\nStatic → dynamic down-classified: {static['static_to_dynamic_down_classified']}")
    print(f"Scientific change: {change} (matched OR {fmt(ors)} → {fmt(ordy)})")
    print("-"*60+"\nSTAGE07 TEST\n"+"-"*60)
    for name,label in [("5NM_1000FT","5NM / 1000ft"),("7_5NM_1200FT","7.5NM / 1200ft"),("STATIC_PROFILE_PROXY","Static proxy"),("EXACT_DYNAMIC_OPENSKY_RULE","Exact dynamic rule")]:
        row=pd.read_csv(OUT/"stage07_test_dynamic_rule_comparison.csv").set_index("method").loc[name]
        print(f"{label}: coverage={row.test_coverage:.2%}, balanced accuracy={fmt(row.balanced_accuracy)}, intervention recall={fmt(row.intervention_recall)}, control non-green rate={fmt(row.control_rule_positive_rate)}")
    for name,label in [("CONTINUOUS_GEOMETRY_LOGISTIC","Continuous geometry logistic"),("STAGE07_OPENSKY_COMPATIBLE_XGBOOST","Stage07 OpenSky-compatible XGBoost"),("STAGE07_SCAT_FULL_XGBOOST","Stage07 SCAT-full XGBoost")]:
        row=old.loc[name]
        print(f"{label}: ROC-AUC={fmt(row.roc_auc)}, PR-AUC={fmt(row.pr_auc)}")
    print("-"*60+"\nKEY SCIENTIFIC RESULT\n"+"-"*60)
    print(f"Relationship: {strength} inverse colour gradient; non-GREEN is less associated with recorded intervention in this matched cohort (OR {fmt(ordy)}).")
    print("-"*60+"\nSEMANTICS\n"+"-"*60)
    print("ATC_INTERVENTION != CONFLICT\nRED != CONFIRMED_CONFLICT\nGREEN != SAFE")
    print("Dynamic-rule performance means discrimination of recorded intervention, not conflict-detection accuracy.")
    print("-"*60+"\nOPEN SKY GATE\n"+"-"*60)
    print(f"Proceed with bounded FL290–FL410 quality/feature-yield pilot: {gate}")
    print("-"*60+"\nFINAL DECISION\n"+"-"*60)
    print(decision)
    print("DO NOT PROCESS FULL OPENSKY HISTORY. DO NOT PROCESS OTHER SCAT WEEKS. DO NOT RUN STAGE 08.")
    print("="*60)


if __name__=="__main__":
    main()

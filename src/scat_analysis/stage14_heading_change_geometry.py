#!/usr/bin/env python3
"""Stage 14: traffic geometry at SCAT assigned-heading change clearances.

The unit is one controller clearance, with the closest distinct aircraft at
the clearance timestamp. Both aircraft must have native surveillance plots no
more than 10 seconds old. No future state, detector colour, or conflict label
enters the calculation. Inputs from earlier stages are read only.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm


BASE = Path(__file__).resolve().parents[2] / "data/processed/scat"
OUT = BASE / "14_heading_change_geometry"
STATES = BASE / "02_traffic/scat_states.parquet"
CLEARANCES = BASE / "03_clearances/scat_clearances.parquet"
EARTH_RADIUS_NM = 3440.065
MAX_AGE_S = 10.0
FL_LOW, FL_HIGH = 290.0, 410.0
RADII_NM = (50, 100)
STATE_COLS = [
    "archive_week", "flight_key", "source_member", "plot_index", "time_epoch",
    "latitude", "longitude", "altitude_ft", "flight_level",
    "ground_speed_kt", "track_deg", "vertical_rate_fpm", "on_ground",
]
CLEARANCE_COLS = [
    "clearance_id", "archive_week", "flight_key", "source_member",
    "clearance_epoch", "clearance_timestamp", "clearance_type",
    "previous_value", "new_value", "previous_record_json", "clearance_cfl_unit",
    "surveillance_match_valid", "surveillance_match_age_s", "matched_state_epoch",
    "matched_latitude", "matched_longitude", "matched_altitude_ft",
    "matched_flight_level", "matched_ground_speed_kt", "matched_track_deg",
    "matched_heading_deg", "matched_vertical_rate_fpm",
]


def json_field(value: object, key: str) -> object:
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return None
    return data.get(key) if isinstance(data, dict) else None


def shortest_signed_heading_change(previous: pd.Series, new: pd.Series) -> pd.Series:
    """Return the shortest clockwise-positive signed angle in [-180, 180)."""
    return (new - previous + 180) % 360 - 180


def prepare_events(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    source = source.copy()
    audit = {"all_clearance_rows": len(source)}
    heading = source.loc[source.clearance_type.fillna("").str.contains("HEADING")].copy()
    audit["heading_records"] = len(heading)
    audit["heading_only_records"] = int(heading.clearance_type.eq("HEADING").sum())
    audit["combined_heading_records"] = int(heading.clearance_type.ne("HEADING").sum())
    heading["previous_heading_deg"] = pd.to_numeric(
        heading.previous_value.map(lambda x: json_field(x, "assigned_heading_val")),
        errors="coerce")
    heading["new_heading_deg"] = pd.to_numeric(
        heading.new_value.map(lambda x: json_field(x, "assigned_heading_val")),
        errors="coerce")
    numeric = heading.previous_heading_deg.notna() & heading.new_heading_deg.notna()
    audit["both_assigned_headings_numeric"] = int(numeric.sum())
    valid_new_heading = heading.new_heading_deg.between(0, 360)
    valid_angles = numeric & heading.previous_heading_deg.between(0, 360) & valid_new_heading
    audit["both_headings_valid_degrees"] = int(valid_angles.sum())
    audit["new_assigned_heading_valid_degrees"] = int(valid_new_heading.sum())
    heading["signed_heading_change_deg"] = shortest_signed_heading_change(
        heading.previous_heading_deg, heading.new_heading_deg)
    heading["absolute_heading_change_deg"] = heading.signed_heading_change_deg.abs()
    actual_change = valid_angles & heading.absolute_heading_change_deg.gt(1e-9)
    audit["actual_numeric_heading_changes"] = int(actual_change.sum())
    heading["matched_heading_deg"] = pd.to_numeric(heading.matched_heading_deg, errors="coerce")
    heading["signed_heading_instruction_from_observed_deg"] = shortest_signed_heading_change(
        heading.matched_heading_deg, heading.new_heading_deg)
    heading["absolute_heading_instruction_from_observed_deg"] = \
        heading.signed_heading_instruction_from_observed_deg.abs()
    heading["heading_event_kind"] = np.where(
        valid_angles & heading.absolute_heading_change_deg.gt(1e-9),
        "REASSIGNMENT", "INITIAL_ASSIGNMENT")
    heading_instruction = valid_new_heading & (
        heading.previous_heading_deg.isna() | actual_change)
    audit["new_numeric_heading_instructions"] = int(heading_instruction.sum())
    heading["matched_flight_level"] = pd.to_numeric(
        heading.matched_flight_level, errors="coerce")
    for field in ["clearance_epoch", "matched_state_epoch", "matched_latitude",
                  "matched_longitude", "matched_altitude_ft", "matched_ground_speed_kt",
                  "matched_track_deg", "matched_vertical_rate_fpm",
                  "surveillance_match_age_s"]:
        heading[field] = pd.to_numeric(heading[field], errors="coerce")

    valid_focal = heading_instruction & heading.surveillance_match_valid.fillna(False).astype(bool) & \
        heading.surveillance_match_age_s.between(0, MAX_AGE_S) & \
        heading.matched_latitude.between(-90, 90) & \
        heading.matched_longitude.between(-180, 180) & \
        heading.matched_altitude_ft.between(0, 60000)
    audit["valid_previous_focal_state"] = int(valid_focal.sum())
    observed_band = valid_focal & heading.matched_flight_level.between(FL_LOW, FL_HIGH)
    audit["focal_observed_in_fl_band"] = int(observed_band.sum())
    audit["eligible_clearance_events"] = int(observed_band.sum())
    events = heading.loc[observed_band, [
        "clearance_id", "archive_week", "flight_key", "source_member",
        "clearance_epoch", "clearance_timestamp", "matched_state_epoch",
        "surveillance_match_age_s", "matched_latitude", "matched_longitude",
        "matched_altitude_ft", "matched_flight_level", "matched_ground_speed_kt",
        "matched_track_deg", "matched_heading_deg", "matched_vertical_rate_fpm",
        "previous_heading_deg", "new_heading_deg", "signed_heading_change_deg",
        "absolute_heading_change_deg", "signed_heading_instruction_from_observed_deg",
        "absolute_heading_instruction_from_observed_deg", "heading_event_kind",
    ]].copy()
    events["turn_direction"] = np.select(
        [events.signed_heading_instruction_from_observed_deg.abs().lt(180) &
         events.signed_heading_instruction_from_observed_deg.gt(0),
         events.signed_heading_instruction_from_observed_deg.abs().lt(180) &
         events.signed_heading_instruction_from_observed_deg.lt(0)],
        ["right", "left"], default="180_deg_ambiguous_or_unavailable")
    if events.clearance_id.isna().any() or events.clearance_id.duplicated().any():
        raise ValueError("Clearance IDs are missing or duplicated")
    if (events.matched_state_epoch > events.clearance_epoch + 1e-6).any():
        raise ValueError("A focal surveillance state is after its clearance")
    return events.sort_values(["archive_week", "clearance_epoch", "clearance_id"],
                              kind="mergesort").reset_index(drop=True), audit


def load_events() -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_parquet(CLEARANCES, columns=CLEARANCE_COLS)
    return prepare_events(source)


def week_row_groups(parquet: pq.ParquetFile, week: str) -> list[int]:
    column = parquet.schema.names.index("archive_week")
    indices = []
    for i in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(i).column(column).statistics
        if stats is None or stats.min <= week <= stats.max:
            indices.append(i)
    return indices


def load_week_states(parquet: pq.ParquetFile, week: str) -> tuple[pd.DataFrame, dict]:
    groups = week_row_groups(parquet, week)
    if not groups:
        raise ValueError(f"No surveillance row groups for {week}")
    states = parquet.read_row_groups(groups, columns=STATE_COLS, use_threads=True).to_pandas()
    states = states.loc[states.archive_week.eq(week)].copy()
    native_rows = len(states)
    for field in ["time_epoch", "latitude", "longitude", "altitude_ft",
                  "flight_level", "ground_speed_kt", "track_deg", "vertical_rate_fpm"]:
        states[field] = pd.to_numeric(states[field], errors="coerce")
    eligible = states.flight_key.notna() & states.source_member.notna() & \
        states.time_epoch.notna() & states.latitude.between(-90, 90) & \
        states.longitude.between(-180, 180) & states.altitude_ft.between(0, 60000) & \
        states.flight_level.between(FL_LOW, FL_HIGH) & \
        ~states.on_ground.fillna(False).astype(bool)
    states = states.loc[eligible].copy()
    collisions = states.groupby("flight_key", sort=False).source_member.nunique()
    collision_keys = set(collisions.index[collisions.gt(1)].astype(str))
    if collision_keys:
        states["entity_key"] = np.where(
            states.flight_key.isin(collision_keys),
            states.flight_key.astype(str) + "::" + states.source_member.astype(str),
            states.flight_key.astype(str))
    else:
        states["entity_key"] = states.flight_key.astype(str)
    states.sort_values(["time_epoch", "plot_index"], kind="mergesort", inplace=True)
    states.reset_index(drop=True, inplace=True)
    audit = {"week": week, "native_rows": native_rows, "in_band_valid_airborne_rows": len(states),
             "colliding_flight_keys": len(collision_keys), "row_groups": len(groups)}
    return states, audit


def nearest_geometry(event, positions: np.ndarray, arrays: dict[str, np.ndarray],
                     focal_entity: str) -> dict:
    result = {"n_neighbours_50nm": 0, "n_neighbours_100nm": 0,
              "nearest_flight_key": None, "nearest_source_member": None,
              "nearest_horizontal_nm": np.nan, "nearest_vertical_ft": np.nan,
              "nearest_observed_fl": np.nan, "nearest_state_age_s": np.nan,
              "nearest_relative_speed_kt": np.nan, "nearest_tcpa_signed_min": np.nan,
              "nearest_cpa_horizontal_nm": np.nan, "nearest_vertical_at_cpa_ft": np.nan}
    if not len(positions):
        return result
    # Native rows are time sorted. The first row after reversing is the latest
    # observation of each entity in the 10 s lookback interval.
    _, first_reversed = np.unique(arrays["entity_code"][positions][::-1], return_index=True)
    p = positions[-1 - first_reversed]
    p = p[arrays["entity_key"][p] != focal_entity]
    if not len(p):
        return result
    lat0 = math.radians(float(event.matched_latitude))
    lon0 = math.radians(float(event.matched_longitude))
    dlat = arrays["lat_rad"][p] - lat0
    dlon = (arrays["lon_rad"][p] - lon0 + np.pi) % (2 * np.pi) - np.pi
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(arrays["lat_rad"][p]) * \
        np.sin(dlon / 2) ** 2
    distance = 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    result["n_neighbours_50nm"] = int(np.count_nonzero(distance <= 50))
    result["n_neighbours_100nm"] = int(np.count_nonzero(distance <= 100))
    nearest = int(np.argmin(distance))
    if distance[nearest] > 100:
        return result
    j = int(p[nearest])
    h = float(distance[nearest])
    result.update(nearest_flight_key=str(arrays["flight_key"][j]),
                  nearest_source_member=str(arrays["source_member"][j]),
                  nearest_horizontal_nm=h,
                  nearest_vertical_ft=abs(float(arrays["altitude_ft"][j]) -
                                          float(event.matched_altitude_ft)),
                  nearest_observed_fl=float(arrays["flight_level"][j]),
                  nearest_state_age_s=float(event.clearance_epoch - arrays["time_epoch"][j]))

    focal_speed = float(event.matched_ground_speed_kt)
    focal_track = float(event.matched_track_deg)
    other_speed = float(arrays["ground_speed_kt"][j])
    other_track = float(arrays["track_deg"][j])
    if not all(map(math.isfinite, [focal_speed, focal_track, other_speed, other_track])):
        return result
    # Tangent-plane position supports the constant-velocity CPA calculation;
    # the displayed current separation above remains an exact great circle.
    dx = EARTH_RADIUS_NM * float(dlon[nearest]) * \
        math.cos((lat0 + float(arrays["lat_rad"][j])) / 2)
    dy = EARTH_RADIUS_NM * float(dlat[nearest])
    rvx = (other_speed * math.sin(math.radians(other_track)) -
           focal_speed * math.sin(math.radians(focal_track))) / 60
    rvy = (other_speed * math.cos(math.radians(other_track)) -
           focal_speed * math.cos(math.radians(focal_track))) / 60
    rv2 = rvx * rvx + rvy * rvy
    result["nearest_relative_speed_kt"] = 60 * math.sqrt(rv2)
    if rv2 <= 1e-12:
        return result
    tcpa = -(dx * rvx + dy * rvy) / rv2
    result["nearest_tcpa_signed_min"] = tcpa
    if tcpa <= 0:
        return result
    result["nearest_cpa_horizontal_nm"] = math.hypot(dx + rvx * tcpa, dy + rvy * tcpa)
    focal_vr = float(event.matched_vertical_rate_fpm)
    other_vr = float(arrays["vertical_rate_fpm"][j])
    if math.isfinite(focal_vr) and math.isfinite(other_vr):
        result["nearest_vertical_at_cpa_ft"] = abs(
            float(arrays["altitude_ft"][j]) - float(event.matched_altitude_ft) +
            (other_vr - focal_vr) * tcpa)
    return result


def analyze_week(events: pd.DataFrame, states: pd.DataFrame, collision_keys: set[str] | None = None) -> pd.DataFrame:
    collision_keys = collision_keys or set()
    arrays = {name: states[name].to_numpy() for name in [
        "flight_key", "source_member", "entity_key", "time_epoch", "altitude_ft",
        "flight_level", "ground_speed_kt", "track_deg", "vertical_rate_fpm"]}
    arrays["lat_rad"] = np.deg2rad(states.latitude.to_numpy(dtype=float))
    arrays["lon_rad"] = np.deg2rad(states.longitude.to_numpy(dtype=float))
    arrays["entity_code"] = pd.factorize(states.entity_key, sort=False)[0]
    times = arrays["time_epoch"].astype(float)
    if np.any(np.diff(times) < 0):
        raise ValueError("Surveillance states are not sorted by time")
    event_times = events.clearance_epoch.to_numpy(dtype=float)
    starts = np.searchsorted(times, event_times - MAX_AGE_S, side="left")
    ends = np.searchsorted(times, event_times, side="right")
    rows = []
    for event, start, end in zip(events.itertuples(index=False), starts, ends):
        focal_entity = str(event.flight_key)
        if focal_entity in collision_keys:
            focal_entity += "::" + str(event.source_member)
        geometry = nearest_geometry(event, np.arange(start, end), arrays, focal_entity)
        rows.append(geometry)
    result = pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if result.nearest_horizontal_nm.notna().any():
        observed = result.loc[result.nearest_horizontal_nm.notna()]
        if observed.nearest_state_age_s.lt(-1e-6).any() or \
                observed.nearest_state_age_s.gt(MAX_AGE_S + 1e-6).any():
            raise ValueError("A neighbour state violates the 10 s lookback")
    if (result.n_neighbours_50nm > result.n_neighbours_100nm).any():
        raise ValueError("The 50 NM neighbours are not nested inside 100 NM")
    return result


def histogram(values: pd.Series, edges: np.ndarray, radius: int, metric: str,
              denominator: int) -> pd.DataFrame:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    counts, _ = np.histogram(clean, bins=edges)
    return pd.DataFrame({"radius_nm": radius, "metric": metric,
                         "left": edges[:-1], "right": edges[1:], "count": counts,
                         "share_all_eligible_events": counts / denominator,
                         "share_metric_observations": counts / len(clean) if len(clean) else 0.0})


def render_radius(all_events: pd.DataFrame, radius: int) -> dict:
    eligible = len(all_events)
    selected = all_events.loc[all_events.nearest_horizontal_nm.le(radius)].copy()
    h_edges = np.arange(0, radius + (2 if radius == 50 else 5), 2 if radius == 50 else 5)
    v_edges = np.arange(0, 12500, 500)
    t_edges = np.arange(0, 31, 1)
    turn_edges = np.linspace(0, 180, 37)
    future = selected.loc[selected.nearest_tcpa_signed_min.gt(0), "nearest_tcpa_signed_min"]
    hist = pd.concat([
        histogram(selected.absolute_heading_change_deg, turn_edges, radius,
                  "absolute_change_between_assigned_headings_deg", eligible),
        histogram(selected.absolute_heading_instruction_from_observed_deg, turn_edges, radius,
                  "absolute_new_heading_vs_observed_magnetic_heading_deg", eligible),
        histogram(selected.nearest_horizontal_nm, h_edges, radius, "nearest_horizontal_nm", eligible),
        histogram(selected.nearest_vertical_ft, v_edges, radius, "vertical_at_nearest_ft", eligible),
        histogram(future.loc[future.le(30)], t_edges, radius, "positive_tcpa_min_0_30", eligible),
    ], ignore_index=True)
    expected_metric_counts = {
        "absolute_change_between_assigned_headings_deg": int(
            selected.absolute_heading_change_deg.notna().sum()),
        "absolute_new_heading_vs_observed_magnetic_heading_deg": int(
            selected.absolute_heading_instruction_from_observed_deg.notna().sum()),
        "nearest_horizontal_nm": len(selected),
        "vertical_at_nearest_ft": len(selected),
    }
    for metric, expected in expected_metric_counts.items():
        if int(hist.loc[hist.metric.eq(metric), "count"].sum()) != expected:
            raise ValueError(f"Histogram {metric} omits events inside {radius} NM")
    hist.to_csv(OUT / "tables" / f"histograms_{radius}nm.csv", index=False)
    joint, _, _ = np.histogram2d(selected.nearest_horizontal_nm,
                                 selected.nearest_vertical_ft, bins=[h_edges, v_edges])
    joint_rows = [dict(radius_nm=radius, horizontal_left_nm=h_edges[i],
                       horizontal_right_nm=h_edges[i + 1], vertical_left_ft=v_edges[j],
                       vertical_right_ft=v_edges[j + 1], count=int(joint[i, j]))
                  for i in range(len(h_edges) - 1) for j in range(len(v_edges) - 1)]
    if int(joint.sum()) != len(selected):
        raise ValueError(f"Joint histogram omits events inside {radius} NM")
    pd.DataFrame(joint_rows).to_csv(OUT / "tables" / f"joint_horizontal_vertical_{radius}nm.csv", index=False)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#f5f8fb")
    fig.subplots_adjust(left=.06, right=.94, top=.90, bottom=.20, hspace=.38, wspace=.30)
    for item in ax.flat:
        item.set_facecolor("white")
    fig.suptitle(f"SCAT Stage 14 · órdenes de cambio de rumbo · radio {radius} NM",
                 fontsize=16, weight="bold", color="#153d59")
    strict_changes = selected.absolute_heading_change_deg.dropna()
    observed_heading_offsets = selected.absolute_heading_instruction_from_observed_deg.dropna()
    ax[0, 0].hist(strict_changes, bins=turn_edges,
                  color="#0f766e", edgecolor="white")
    ax[0, 0].set(xlabel="Magnitud del cambio entre rumbo anterior y nuevo (°)", ylabel="Órdenes",
                 title=f"Reasignaciones · N={len(strict_changes):,}")
    ax[0, 1].hist(observed_heading_offsets, bins=turn_edges,
                  color="#0891b2", edgecolor="white")
    ax[0, 1].set(xlabel="Magnitud angular: rumbo nuevo vs heading observado (°)", ylabel="Órdenes",
                 title=f"Desviación de la orden nueva · N={len(observed_heading_offsets):,}")
    ax[0, 2].hist(selected.nearest_horizontal_nm, bins=h_edges,
                  color="#155e75", edgecolor="white")
    ax[0, 2].set(xlabel="Distancia horizontal al vecino más próximo (NM)", ylabel="Órdenes",
                 title="Separación horizontal observada")
    ax[1, 0].hist(selected.nearest_vertical_ft, bins=v_edges,
                  color="#a21caf", edgecolor="white")
    ax[1, 0].set(xlabel="Separación vertical con ese vecino (ft)", ylabel="Órdenes",
                 title="Separación vertical observada")
    ax[1, 1].hist(future.loc[future.le(30)], bins=t_edges,
                  color="#d97706", edgecolor="white")
    ax[1, 1].set(xlabel="Tiempo hasta CPA horizontal (min)", ylabel="Órdenes",
                 title="TCPA positivo del mismo vecino · 0–30 min")
    mesh = ax[1, 2].pcolormesh(h_edges, v_edges, joint.T, cmap="YlGnBu",
                               norm=LogNorm(vmin=1, vmax=max(2, int(joint.max()))), shading="auto")
    fig.colorbar(mesh, ax=ax[1, 2], label="Órdenes por celda")
    ax[1, 2].set(xlabel="Distancia horizontal (NM)", ylabel="Separación vertical (ft)",
                 title="Distribución conjunta")
    n_future = int(selected.nearest_tcpa_signed_min.gt(0).sum())
    n_past = int(selected.nearest_tcpa_signed_min.le(0).sum())
    n_missing = int(selected.nearest_tcpa_signed_min.isna().sum())
    notes = [
        f"N={eligible:,} cambios de rumbo elegibles; {len(selected):,} con vecino ≤{radius} NM.",
        f"TCPA futuro: {n_future:,}; pasado/divergente: {n_past:,}; sin cálculo: {n_missing:,}; "
        f"futuro >30 min: {int(future.gt(30).sum()):,}.",
        "Los dos histogramas angulares usan referencias distintas (orden previa; heading magnético observado).",
        "Estados radar separados en el tiempo hasta 10 s; TCPA supone velocidad constante y no confirma colisión.",
    ]
    for y, note in zip([.145, .11, .075, .04], notes):
        fig.text(.07, y, note, ha="left", va="bottom", fontsize=9, color="#334155")
    dest = OUT / "figures" / f"stage14_heading_change_{radius}nm.png"
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"radius_nm": radius, "eligible_events": eligible, "with_neighbour": len(selected),
            "n_numeric_heading_reassignments": int(selected.absolute_heading_change_deg.notna().sum()),
            "median_absolute_heading_change_deg": float(selected.absolute_heading_change_deg.median()),
            "p90_absolute_heading_change_deg": float(selected.absolute_heading_change_deg.quantile(.9)),
            "n_observed_heading_offsets": int(observed_heading_offsets.size),
            "median_absolute_heading_offset_from_observed_deg": float(observed_heading_offsets.abs().median()),
            "right_turns": int(selected.turn_direction.eq("right").sum()),
            "left_turns": int(selected.turn_direction.eq("left").sum()),
            "without_neighbour": eligible - len(selected), "future_tcpa": n_future,
            "past_or_diverging_tcpa": n_past, "missing_tcpa": n_missing,
            "future_tcpa_over_30min": int(future.gt(30).sum()),
            "median_nearest_horizontal_nm": float(selected.nearest_horizontal_nm.median()),
            "median_vertical_at_nearest_ft": float(selected.nearest_vertical_ft.median()),
            "median_positive_tcpa_min": float(future.median()),
            "median_horizontal_cpa_for_future_pairs_nm": float(
                selected.loc[selected.nearest_tcpa_signed_min.gt(0),
                             "nearest_cpa_horizontal_nm"].median()),
            "figure": str(dest.relative_to(OUT))}


def write_report(events: pd.DataFrame, clearance_audit: dict, weeks: list[dict],
                 radii: list[dict], invocation_elapsed: float, full_build: dict) -> None:
    focal_speed = pd.to_numeric(events.matched_ground_speed_kt, errors="coerce").median()
    relative_speed = pd.to_numeric(
        events.loc[events.nearest_horizontal_nm.le(100), "nearest_relative_speed_kt"],
        errors="coerce").median()
    build_time = (f"{full_build['elapsed_seconds']:.1f} s"
                  if full_build.get("elapsed_seconds") is not None else "no registrado")
    build_memory = (f"{full_build['peak_rss_mib']:.0f} MiB"
                    if full_build.get("peak_rss_mib") is not None else "no registrado")
    lines = [
        "# Stage 14 — Geometría en órdenes de cambio de rumbo",
        "",
        "## Pregunta y alcance",
        "",
        "Distribución de los cambios de rumbo asignado y de la separación horizontal/vertical "
        "con la aeronave más próxima cuando se registra una modificación de rumbo, para radios "
        "de 50 y 100 NM.",
        "Se muestran dos magnitudes angulares: la diferencia mínima entre dos rumbos asignados "
        "consecutivos (solo reasignaciones) y la diferencia entre el rumbo nuevo y el heading "
        "magnético medido justo antes de la orden (también disponible para asignaciones iniciales). "
        "Ambas usan el intervalo 0–180°. Se excluyen cancelaciones, actualizaciones solo de "
        "beacon y equivalencias como 0°↔360°.",
        "El TCPA mostrado corresponde al mismo vecino más próximo y es el tiempo hasta su CPA "
        "horizontal bajo velocidad constante. No es tiempo de colisión confirmado ni mide la "
        "respuesta posterior del avión a la instrucción.",
        "",
        "Se incluyen registros cuyo tipo contiene HEADING, también si en el mismo registro "
        "cambia otro campo. La cohorte geométrica exige un nuevo rumbo asignado numérico, "
        "destinatario observado en FL290–410 y vecino observado también en FL290–410. Cuando "
        "hay rumbo asignado anterior se conserva la reasignación como medida estricta; cuando "
        "no lo hay, se compara la orden nueva con el heading magnético observado. Se usa el "
        "último estado radar de cada aeronave anterior a la orden y de antigüedad máxima 10 s; "
        "no se interpolan trayectorias y los dos estados no tienen por qué ser simultáneos. "
        "Cada orden aporta como máximo una observación por radio. "
        "50 NM es subconjunto de 100 NM.",
        "",
        "## Cobertura de órdenes",
        "",
        "| Filtro | Registros |", "|---|---:|",
    ]
    filter_names = {
        "all_clearance_rows": "Todas las autorizaciones SCAT",
        "heading_records": "Registros cuyo tipo incluye HEADING",
        "heading_only_records": "Registros solo HEADING",
        "combined_heading_records": "Registros HEADING combinados con otros cambios",
        "both_assigned_headings_numeric": "Rumbo asignado anterior y nuevo numéricos",
        "both_headings_valid_degrees": "Ambos rumbos entre 0° y 360°",
        "actual_numeric_heading_changes": "Reasignaciones previas con diferencia angular no nula",
        "new_assigned_heading_valid_degrees": "Nuevo rumbo numérico entre 0° y 360°",
        "new_numeric_heading_instructions": "Nuevas asignaciones (iniciales o reasignaciones)",
        "valid_previous_focal_state": "Estado radar previo válido del destinatario",
        "focal_observed_in_fl_band": "Destinatario observado en FL290–410",
        "eligible_clearance_events": "Órdenes elegibles finales",
    }
    for key, value in clearance_audit.items():
        lines.append(f"| {filter_names.get(key, key)} | {value:,} |")
    lines += ["", "## Cobertura por semana", "",
              "| Semana | Órdenes elegibles | Vecino ≤50 NM | Vecino ≤100 NM | Estados radar en banda |",
              "|---|---:|---:|---:|---:|"]
    for w in weeks:
        subset = events.loc[events.archive_week.eq(w["week"])]
        lines.append(f"| {w['week']} | {len(subset):,} | "
                     f"{int(subset.nearest_horizontal_nm.le(50).sum()):,} | "
                     f"{int(subset.nearest_horizontal_nm.le(100).sum()):,} | "
                     f"{w['in_band_valid_airborne_rows']:,} |")
    lines += ["", "## Histogramas", "",
              "Las figuras muestran órdenes con vecino observado dentro del radio. Las tablas "
              "CSV conservan conteos y proporción respecto de todas las órdenes elegibles. "
              "Para TCPA, se grafica solo el TCPA positivo de hasta 30 min; los casos futuros "
              "más largos, pasados/divergentes y no calculables se contabilizan aparte.", "",
              "El primer histograma angular usa solo reasignaciones con rumbo anterior numérico; "
              "el segundo compara toda orden con el heading magnético observado y, por ello, "
              "incluye asignaciones iniciales. No son la misma población ni la misma referencia.", "",
              "| Sentido del nuevo rumbo respecto al heading magnético observado | Órdenes elegibles |",
              "|---|---:|",
              f"| Giro a la derecha | {int(events.turn_direction.eq('right').sum()):,} |",
              f"| Giro a la izquierda | {int(events.turn_direction.eq('left').sum()):,} |",
              f"| 180° o heading observado ausente | {int(events.turn_direction.eq('180_deg_ambiguous_or_unavailable').sum()):,} |", ""]
    for r in radii:
        n = r["radius_nm"]
        lines += [f"### Radio {n} NM", "",
                  f"![Histogramas para {n} NM](figures/stage14_heading_change_{n}nm.png)", "",
                  f"- Con vecino dentro del radio: {r['with_neighbour']:,}/{r['eligible_events']:,} "
                  f"({100*r['with_neighbour']/r['eligible_events']:.1f} %).",
                  f"- Reasignaciones con orden previa: {r['n_numeric_heading_reassignments']:,}; "
                  f"cambio mediano anterior→nuevo: {r['median_absolute_heading_change_deg']:.1f}° "
                  f"(P90 {r['p90_absolute_heading_change_deg']:.1f}°).",
                  f"- Desviación mediana del nuevo rumbo respecto al heading magnético observado: "
                  f"{r['median_absolute_heading_offset_from_observed_deg']:.1f}° "
                  f"(N={r['n_observed_heading_offsets']:,}).",
                  f"- Mediana horizontal del vecino: {r['median_nearest_horizontal_nm']:.1f} NM; "
                  f"vertical: {r['median_vertical_at_nearest_ft']:.0f} ft.",
                  f"- TCPA futuro: {r['future_tcpa']:,}; pasado/divergente: "
                  f"{r['past_or_diverging_tcpa']:,}; sin cálculo: {r['missing_tcpa']:,}; "
                  f"futuro >30 min: {r['future_tcpa_over_30min']:,}. "
                  f"Mediana del TCPA futuro: {r['median_positive_tcpa_min']:.1f} min; "
                  f"distancia horizontal mediana prevista en ese CPA: "
                  f"{r['median_horizontal_cpa_for_future_pairs_nm']:.1f} NM.", ""]
    lines += [
        "## Interpretación y límites", "",
        "La separación horizontal es la distancia actual a la aeronave más próxima dentro del "
        "radio; la separación vertical y el TCPA pertenecen a ese mismo avión. La orden se "
        "analiza con estados nativos anteriores de hasta 10 s de antigüedad, que no tienen por "
        "qué ser simultáneos entre aeronaves. Por ello, estos resultados "
        "describen el contexto de tráfico al emitir la orden y no prueban que el cambio se "
        "ejecutara ni que estuviera motivado por un conflicto.",
        "El TCPA usa velocidad sobre el suelo y track observado, supone movimiento rectilíneo "
        "uniforme y no modela la maniobra solicitada. Puede ser negativo si la pareja ya pasó "
        "su CPA; un TCPA positivo tampoco significa que vaya a producirse una pérdida de "
        "separación. 50 y 100 NM son radios descriptivos, no horizontes temporales fijos.",
        "La atribución del vecino a una orden individual es ambigua: una instrucción puede "
        "responder a varios tráficos, restricciones de ruta, coordinación u otros motivos. "
        "Los cambios de rumbo asignado tampoco equivalen a cambios observados de track.",
        f"La velocidad observada mediana del destinatario es {focal_speed:.1f} kt; la velocidad "
        f"relativa mediana respecto al vecino más próximo dentro de 100 NM es {relative_speed:.1f} kt.",
        "",
        "## Reproducibilidad", "",
        f"- Fuente de autorizaciones: `{CLEARANCES.relative_to(BASE)}`.",
        f"- Fuente de estados nativos: `{STATES.relative_to(BASE)}`.",
        "- Script: `src/scat_analysis/stage14_heading_change_geometry.py`.",
        "- Eventos semanales y tabla consolidada: `tables/heading_events_*.parquet` y "
        "`tables/heading_change_event_geometry.parquet`.",
        "- Histogramas y distribuciones conjuntas: `tables/histograms_*.csv` y "
        "`tables/joint_horizontal_vertical_*.csv`.",
        f"- Construcción completa de Stage 14: {build_time}; pico de memoria de proceso: {build_memory}.",
        f"- Última ejecución del script: {invocation_elapsed:.1f} s.",
        "- No se reejecutaron ni modificaron fases anteriores.", "",
    ]
    (OUT / "STAGE14_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute Stage 14 week shards")
    args = parser.parse_args()
    start = time.monotonic()
    for subdir in ["tables", "figures", "diagnostics"]:
        (OUT / subdir).mkdir(parents=True, exist_ok=True)
    events, clearance_audit = load_events()
    weeks = sorted(events.archive_week.unique())
    source = pq.ParquetFile(STATES)
    weekly_audit = []
    shards = []
    all_weeks_recomputed = True
    for week in weeks:
        path = OUT / "tables" / f"heading_events_{week}.parquet"
        audit_path = OUT / "diagnostics" / f"week_{week}.json"
        if path.exists() and audit_path.exists() and not args.force:
            print("CACHED", week, flush=True)
            all_weeks_recomputed = False
            weekly_audit.append(json.loads(audit_path.read_text()))
            shards.append(path)
            continue
        t0 = time.monotonic()
        states, audit = load_week_states(source, week)
        collisions = states.groupby("flight_key", sort=False).source_member.nunique()
        collision_keys = set(collisions.index[collisions.gt(1)].astype(str))
        week_events = events.loc[events.archive_week.eq(week)].copy()
        result = analyze_week(week_events, states, collision_keys)
        audit.update(eligible_clearances=len(result), with_neighbour_50nm=int(result.nearest_horizontal_nm.le(50).sum()),
                     with_neighbour_100nm=int(result.nearest_horizontal_nm.le(100).sum()),
                     elapsed_seconds=round(time.monotonic() - t0, 2))
        result.to_parquet(path, index=False, compression="zstd")
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        weekly_audit.append(audit)
        shards.append(path)
        print(week, json.dumps(audit), flush=True)
        del states, result
    all_events = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    if len(all_events) != len(events) or all_events.clearance_id.duplicated().any():
        raise ValueError("Stage 14 event coverage is incomplete or duplicated")
    if all_events.nearest_horizontal_nm.notna().any() and \
            all_events.nearest_horizontal_nm.max() > 100 + 1e-6:
        raise ValueError("An exported nearest neighbour exceeds 100 NM")
    all_events.to_parquet(OUT / "tables/heading_change_event_geometry.parquet",
                          index=False, compression="zstd")
    radius_audit = [render_radius(all_events, radius) for radius in RADII_NM]
    elapsed = time.monotonic() - start
    peak_rss_mib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                         (1024**2 if sys.platform == "darwin" else 1024), 1)
    full_build_path = OUT / "diagnostics/full_build.json"
    if all_weeks_recomputed:
        full_build = {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": round(elapsed, 2), "peak_rss_mib": peak_rss_mib,
                      "weeks_recomputed": len(weeks)}
        full_build_path.write_text(json.dumps(full_build, indent=2) + "\n", encoding="utf-8")
    elif full_build_path.exists():
        full_build = json.loads(full_build_path.read_text(encoding="utf-8"))
    else:
        full_build = {"elapsed_seconds": None, "peak_rss_mib": None,
                      "weeks_recomputed": 0}
    diagnostics = {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_files": {"clearances": str(CLEARANCES), "states": str(STATES)},
                   "parameters": {"radii_nm": RADII_NM, "observed_fl_band": [FL_LOW, FL_HIGH],
                                  "strict_reassignment_delta": "shortest signed difference between consecutive numeric assigned headings; exact-equivalent angle excluded",
                                  "all_instruction_delta": "shortest signed difference from latest observed magnetic heading to new assigned heading",
                                  "heading_only_beacon_updates_included": False,
                                  "initial_numeric_heading_assignments_included": True,
                                  "heading_cancellations_included": False,
                                  "max_state_age_s": MAX_AGE_S, "interpolation": False,
                                  "time_metric": "signed horizontal TCPA, nearest current neighbour, constant velocity"},
                   "clearance_filter": clearance_audit, "weeks": weekly_audit,
                   "radii": radius_audit, "invocation_elapsed_seconds": round(elapsed, 2),
                   "invocation_peak_rss_mib": peak_rss_mib, "full_build": full_build}
    (OUT / "diagnostics/run.json").write_text(json.dumps(diagnostics, indent=2) + "\n",
                                                 encoding="utf-8")
    write_report(all_events, clearance_audit, weekly_audit, radius_audit, elapsed, full_build)
    print(json.dumps({"events": len(all_events), "radii": radius_audit,
                      "invocation_elapsed_seconds": round(elapsed, 2),
                      "invocation_peak_rss_mib": peak_rss_mib,
                      "full_build": full_build}, indent=2), flush=True)


if __name__ == "__main__":
    main()

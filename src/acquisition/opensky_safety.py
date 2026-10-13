from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Geod


GEOD = Geod(ellps="WGS84")

STATE_NORMAL = "NORMAL"
STATE_PROXIMITY = "PROXIMITY"
STATE_HIGH_RISK = "HIGH_RISK"
STATE_CONFLICT = "CONFLICT"

STATE_PRIORITY = {
    STATE_NORMAL: 0,
    STATE_PROXIMITY: 1,
    STATE_HIGH_RISK: 2,
    STATE_CONFLICT: 3,
}


def combine_states(current_state: str, new_state: str) -> str:
    return (
        new_state
        if STATE_PRIORITY[new_state] > STATE_PRIORITY[current_state]
        else current_state
    )


def classify_pair(
    horizontal_nm: float,
    vertical_ft: float | None,
    *,
    min_horizontal_nm: float,
    min_vertical_ft: float,
    warning_horizontal_nm: float,
    warning_vertical_ft: float,
) -> str:
    """
    Four-state classification used by the research visualisation.

    RED / CONFLICT:
        H < minimum horizontal AND V < minimum vertical.

    ORANGE / HIGH_RISK:
        Inside the warning envelope AND exactly one minimum has already
        been infringed.

    YELLOW / PROXIMITY:
        Inside the warning envelope, but neither minimum is infringed.

    GREEN / NORMAL:
        Everything else.

    Missing vertical information can never be classified as a confirmed
    loss of separation. A close horizontal pair is conservatively marked as
    PROXIMITY only when it is inside the horizontal warning threshold.
    """
    if vertical_ft is None or not np.isfinite(vertical_ft):
        return (
            STATE_PROXIMITY
            if horizontal_nm < warning_horizontal_nm
            else STATE_NORMAL
        )

    h_min = horizontal_nm < min_horizontal_nm
    v_min = vertical_ft < min_vertical_ft

    if h_min and v_min:
        return STATE_CONFLICT

    inside_warning = (
        horizontal_nm < warning_horizontal_nm
        and vertical_ft < warning_vertical_ft
    )

    if inside_warning and (h_min ^ v_min):
        return STATE_HIGH_RISK

    if inside_warning:
        return STATE_PROXIMITY

    return STATE_NORMAL


def _numeric(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = row.get(name, np.nan)
        try:
            if pd.notna(value) and np.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def _speed_ms(row: pd.Series) -> float | None:
    speed = _numeric(row, "velocity")
    if speed is not None:
        return speed

    speed_kts = _numeric(row, "velocity_kts")
    if speed_kts is not None:
        return speed_kts / 1.9438444924406

    return None


def _track_deg(row: pd.Series) -> float | None:
    return _numeric(row, "true_track", "heading")


def _vertical_rate_fpm(row: pd.Series) -> float:
    value = _numeric(row, "vertical_rate_fpm")
    if value is not None:
        return value

    value_ms = _numeric(row, "vertical_rate", "vertrate")
    if value_ms is not None:
        return value_ms * 196.8503937008

    return 0.0


def _baro_altitude_ft(row: pd.Series) -> float | None:
    value = _numeric(row, "baro_altitude_ft")
    if value is not None:
        return value

    value_m = _numeric(row, "baro_altitude", "baroaltitude")
    if value_m is not None:
        return value_m * 3.2808398950131

    return None


def _on_ground(row: pd.Series) -> bool:
    value = row.get("on_ground", row.get("onground", False))
    if pd.isna(value):
        return False
    return bool(value)


def _callsign(row: pd.Series) -> str:
    value = str(row.get("callsign", "") or "").strip()
    if value:
        return value
    return str(row.get("icao24", "") or "").upper()


def project_aircraft_state(
    row: pd.Series,
    time_seconds: float,
) -> tuple[float, float, float | None]:
    """
    Constant-velocity/track/vertical-rate kinematic projection.
    """
    lon = _numeric(row, "longitude", "lon")
    lat = _numeric(row, "latitude", "lat")
    if lon is None or lat is None:
        raise ValueError("Aircraft position is missing")

    speed_ms = _speed_ms(row)
    track_deg = _track_deg(row)

    if speed_ms is not None and track_deg is not None:
        projected_lon, projected_lat, _ = GEOD.fwd(
            lon,
            lat,
            track_deg,
            speed_ms * float(time_seconds),
        )
    else:
        projected_lon, projected_lat = lon, lat

    altitude_ft = _baro_altitude_ft(row)
    if altitude_ft is not None:
        altitude_ft += (
            _vertical_rate_fpm(row)
            * float(time_seconds)
            / 60.0
        )

    return float(projected_lon), float(projected_lat), altitude_ft


def pair_separation_at_time(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
    time_seconds: float,
) -> tuple[float, float | None]:
    lon_a, lat_a, alt_a = project_aircraft_state(aircraft_a, time_seconds)
    lon_b, lat_b, alt_b = project_aircraft_state(aircraft_b, time_seconds)

    _, _, distance_m = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    horizontal_nm = float(distance_m) / 1852.0

    vertical_ft = (
        None
        if alt_a is None or alt_b is None
        else abs(float(alt_a) - float(alt_b))
    )

    return horizontal_nm, vertical_ft


def _continuous_tcpa_seconds(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
    max_seconds: float,
) -> float | None:
    """
    Approximate 2-D TCPA using a local tangent-plane relative-motion model.

    The final CPA distance is recomputed geodesically at the resulting time.
    """
    lon_a = _numeric(aircraft_a, "longitude", "lon")
    lat_a = _numeric(aircraft_a, "latitude", "lat")
    lon_b = _numeric(aircraft_b, "longitude", "lon")
    lat_b = _numeric(aircraft_b, "latitude", "lat")

    speed_a = _speed_ms(aircraft_a)
    speed_b = _speed_ms(aircraft_b)
    track_a = _track_deg(aircraft_a)
    track_b = _track_deg(aircraft_b)

    if None in (lon_a, lat_a, lon_b, lat_b, speed_a, speed_b, track_a, track_b):
        return None

    azimuth_ab, _, distance_m = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    az = math.radians(azimuth_ab)
    # Local EN coordinates of B relative to A.
    rx = distance_m * math.sin(az)
    ry = distance_m * math.cos(az)

    ta = math.radians(track_a)
    tb = math.radians(track_b)

    vax = speed_a * math.sin(ta)
    vay = speed_a * math.cos(ta)
    vbx = speed_b * math.sin(tb)
    vby = speed_b * math.cos(tb)

    rvx = vbx - vax
    rvy = vby - vay

    denom = rvx * rvx + rvy * rvy
    if denom <= 1e-9:
        return 0.0

    tcpa = -((rx * rvx) + (ry * rvy)) / denom
    return float(min(max(tcpa, 0.0), float(max_seconds)))


def _interval_boundaries_seconds(
    lookahead_minutes: Iterable[float],
) -> list[tuple[float, float]]:
    horizons = sorted(float(v) for v in lookahead_minutes)
    if not horizons or horizons[0] <= 0:
        raise ValueError("LOOKAHEAD_MINUTES must contain positive values")

    intervals: list[tuple[float, float]] = []
    start = 0.0
    for minutes in horizons:
        end = minutes * 60.0
        intervals.append((start, end))
        start = end
    return intervals


def _interval_column_name(start_s: float, end_s: float) -> str:
    start_min = int(round(start_s / 60.0))
    end_min = int(round(end_s / 60.0))
    return f"worst_status_{start_min}_{end_min}min"


def analyse_aircraft_snapshot(
    aircraft: pd.DataFrame,
    *,
    min_horizontal_nm: float,
    min_vertical_ft: float,
    warning_factor: float,
    lookahead_minutes: Iterable[float] = (1, 2, 5),
    prediction_step_seconds: int = 5,
    check_ground_aircraft: bool = False,
) -> tuple[list[dict], dict, dict, int]:
    """
    Analyse current and predicted pairwise separation.

    Returns:
      pair_results
      aircraft_current_states
      aircraft_interval_states
      skipped_pairs
    """
    if prediction_step_seconds <= 0:
        raise ValueError("prediction_step_seconds must be positive")

    warning_horizontal_nm = min_horizontal_nm * warning_factor
    warning_vertical_ft = min_vertical_ft * warning_factor

    intervals = _interval_boundaries_seconds(lookahead_minutes)
    max_seconds = intervals[-1][1]

    rows = list(aircraft.iterrows())

    aircraft_states = {
        index: STATE_NORMAL
        for index in aircraft.index
    }

    interval_states = {
        index: [STATE_NORMAL for _ in intervals]
        for index in aircraft.index
    }

    pair_results: list[dict] = []
    skipped_pairs = 0

    for i in range(len(rows)):
        index_a, aircraft_a = rows[i]

        if _on_ground(aircraft_a) and not check_ground_aircraft:
            continue

        for j in range(i + 1, len(rows)):
            index_b, aircraft_b = rows[j]

            if _on_ground(aircraft_b) and not check_ground_aircraft:
                continue

            horizontal_nm, vertical_ft = pair_separation_at_time(
                aircraft_a,
                aircraft_b,
                0.0,
            )

            if vertical_ft is None:
                skipped_pairs += 1

            current_state = classify_pair(
                horizontal_nm,
                vertical_ft,
                min_horizontal_nm=min_horizontal_nm,
                min_vertical_ft=min_vertical_ft,
                warning_horizontal_nm=warning_horizontal_nm,
                warning_vertical_ft=warning_vertical_ft,
            )

            aircraft_states[index_a] = combine_states(
                aircraft_states[index_a],
                current_state,
            )
            aircraft_states[index_b] = combine_states(
                aircraft_states[index_b],
                current_state,
            )

            pair_interval_states: list[str] = []
            pair_min_horizontal_nm = horizontal_nm
            pair_min_vertical_at_cpa_ft = vertical_ft
            pair_tcpa_s = 0.0

            # Continuous horizontal TCPA/CPA estimate.
            continuous_tcpa = _continuous_tcpa_seconds(
                aircraft_a,
                aircraft_b,
                max_seconds,
            )
            if continuous_tcpa is not None:
                cpa_h, cpa_v = pair_separation_at_time(
                    aircraft_a,
                    aircraft_b,
                    continuous_tcpa,
                )
                pair_min_horizontal_nm = cpa_h
                pair_min_vertical_at_cpa_ft = cpa_v
                pair_tcpa_s = continuous_tcpa

            for interval_index, (start_s, end_s) in enumerate(intervals):
                worst = STATE_NORMAL

                # Evaluate the full interval. The end point is included so that
                # a conflict exactly at 1/2/5 min is not missed.
                sample_times = np.arange(
                    start_s,
                    end_s + prediction_step_seconds * 0.5,
                    prediction_step_seconds,
                    dtype=float,
                )

                if sample_times.size == 0 or sample_times[-1] < end_s:
                    sample_times = np.append(sample_times, end_s)

                for time_s in sample_times:
                    h_nm, v_ft = pair_separation_at_time(
                        aircraft_a,
                        aircraft_b,
                        float(time_s),
                    )

                    state = classify_pair(
                        h_nm,
                        v_ft,
                        min_horizontal_nm=min_horizontal_nm,
                        min_vertical_ft=min_vertical_ft,
                        warning_horizontal_nm=warning_horizontal_nm,
                        warning_vertical_ft=warning_vertical_ft,
                    )
                    worst = combine_states(worst, state)

                pair_interval_states.append(worst)

                interval_states[index_a][interval_index] = combine_states(
                    interval_states[index_a][interval_index],
                    worst,
                )
                interval_states[index_b][interval_index] = combine_states(
                    interval_states[index_b][interval_index],
                    worst,
                )

            result = {
                "index_a": index_a,
                "index_b": index_b,
                "icao24_a": str(aircraft_a.get("icao24", "") or ""),
                "icao24_b": str(aircraft_b.get("icao24", "") or ""),
                "callsign_a": _callsign(aircraft_a),
                "callsign_b": _callsign(aircraft_b),
                "horizontal_sep_nm": horizontal_nm,
                "vertical_sep_ft": vertical_ft,
                "state_current": current_state,
                "status_current": current_state,
                "cpa_nm": pair_min_horizontal_nm,
                "tcpa_s": pair_tcpa_s,
                "vertical_sep_at_cpa_ft": pair_min_vertical_at_cpa_ft,
                "predicted_conflict": any(
                    state == STATE_CONFLICT
                    for state in pair_interval_states
                ),
            }

            for (start_s, end_s), state in zip(
                intervals,
                pair_interval_states,
            ):
                result[_interval_column_name(start_s, end_s)] = state

            pair_results.append(result)

    return (
        pair_results,
        aircraft_states,
        interval_states,
        skipped_pairs,
    )


def relevant_pair_results(pair_results: Iterable[dict]) -> list[dict]:
    """Keep pairs that are non-normal now or in any look-ahead interval."""
    output = []
    for pair in pair_results:
        current = pair.get("state_current", STATE_NORMAL)
        future_states = [
            value
            for key, value in pair.items()
            if key.startswith("worst_status_")
        ]
        if (
            current != STATE_NORMAL
            or any(state != STATE_NORMAL for state in future_states)
        ):
            output.append(dict(pair))
    return output

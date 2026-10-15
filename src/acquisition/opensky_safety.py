from __future__ import annotations

import json
import math
from pathlib import Path
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



DEFAULT_STATE_COLOURS = {
    STATE_NORMAL: "#00A651",
    STATE_PROXIMITY: "#FFD400",
    STATE_HIGH_RISK: "#FF6F00",
    STATE_CONFLICT: "#D7191C",
}

DEFAULT_STATE_DISPLAY_NAMES = {
    STATE_NORMAL: "Normal separation",
    STATE_PROXIMITY: "Proximity warning",
    STATE_HIGH_RISK: "High-risk separation",
    STATE_CONFLICT: "Loss of separation",
}


def normalize_state_rules(
    states_config: dict | None,
) -> dict[str, dict]:
    """
    Convert descriptive JSON keys such as ORANGE_HIGH_RISK into a mapping keyed
    by the canonical internal state names.

    A state is active when:
      - its block exists, and
      - enabled is omitted or True.

    Therefore a state can be disabled either by deleting its JSON block or by
    setting "enabled": false.
    """
    output: dict[str, dict] = {}

    if not isinstance(states_config, dict):
        return output

    for config_key, raw_rule in states_config.items():
        if not isinstance(raw_rule, dict):
            continue

        if raw_rule.get("enabled", True) is False:
            continue

        state = str(
            raw_rule.get("state", "")
        ).strip().upper()

        if state not in STATE_PRIORITY:
            continue

        rule = dict(raw_rule)
        rule["config_key"] = str(config_key)
        output[state] = rule

    return output


def resolve_safety_profile(
    profile: dict,
) -> dict:
    """
    Resolve a profile from separation_profiles.json.

    New format:
        profile["states"]

    Legacy format:
        horizontal_nm
        vertical_ft

    Legacy profiles are automatically expanded to the four-state model using
    warning_factor=1.5 unless another value is present.
    """
    if not isinstance(profile, dict):
        raise ValueError("Separation profile must be a JSON object")

    state_rules = normalize_state_rules(
        profile.get("states")
    )

    if not state_rules:
        min_horizontal = float(
            profile["horizontal_nm"]
        )
        min_vertical = float(
            profile["vertical_ft"]
        )
        warning_factor = float(
            profile.get(
                "warning_factor",
                1.5,
            )
        )

        state_rules = {
            STATE_NORMAL: {
                "state": STATE_NORMAL,
                "display_name": DEFAULT_STATE_DISPLAY_NAMES[STATE_NORMAL],
                "colour": DEFAULT_STATE_COLOURS[STATE_NORMAL],
                "rule": "otherwise",
            },
            STATE_PROXIMITY: {
                "state": STATE_PROXIMITY,
                "display_name": DEFAULT_STATE_DISPLAY_NAMES[STATE_PROXIMITY],
                "colour": DEFAULT_STATE_COLOURS[STATE_PROXIMITY],
                "horizontal_nm": min_horizontal * warning_factor,
                "vertical_ft": min_vertical * warning_factor,
                "rule": "inside_envelope_if_no_more_severe_state_matches",
            },
            STATE_HIGH_RISK: {
                "state": STATE_HIGH_RISK,
                "display_name": DEFAULT_STATE_DISPLAY_NAMES[STATE_HIGH_RISK],
                "colour": DEFAULT_STATE_COLOURS[STATE_HIGH_RISK],
                "horizontal_nm": min_horizontal * warning_factor,
                "vertical_ft": min_vertical * warning_factor,
                "rule": "inside_envelope_and_exactly_one_red_minimum_is_breached",
            },
            STATE_CONFLICT: {
                "state": STATE_CONFLICT,
                "display_name": DEFAULT_STATE_DISPLAY_NAMES[STATE_CONFLICT],
                "colour": DEFAULT_STATE_COLOURS[STATE_CONFLICT],
                "horizontal_nm": min_horizontal,
                "vertical_ft": min_vertical,
                "rule": "both_below",
            },
        }

    if STATE_CONFLICT not in state_rules:
        raise ValueError(
            "The separation profile must contain an enabled CONFLICT / "
            "RED_LOSS_OF_SEPARATION state because it defines the regulatory "
            "minimum separation thresholds."
        )

    red_rule = state_rules[STATE_CONFLICT]

    for field in ("horizontal_nm", "vertical_ft"):
        if field not in red_rule:
            raise ValueError(
                f"CONFLICT state is missing required field: {field}"
            )

    return {
        "profile_name": str(
            profile.get(
                "profile_name",
                "Unnamed separation profile",
            )
        ),
        "states": state_rules,
        "legacy_horizontal_nm": float(
            profile.get(
                "horizontal_nm",
                red_rule["horizontal_nm"],
            )
        ),
        "legacy_vertical_ft": float(
            profile.get(
                "vertical_ft",
                red_rule["vertical_ft"],
            )
        ),
    }


def load_safety_profile(
    path: str | Path,
    profile_name: str,
) -> dict:
    path = Path(path)

    with path.open(
        encoding="utf-8"
    ) as profile_file:
        payload = json.load(
            profile_file
        )

    if profile_name not in payload:
        raise ValueError(
            f"Unknown separation profile: {profile_name}"
        )

    resolved = resolve_safety_profile(
        payload[profile_name]
    )

    resolved["profile_key"] = profile_name

    return resolved


def state_rule(
    state_rules: dict[str, dict] | None,
    state: str,
) -> dict | None:
    if not state_rules:
        return None

    # Accept both already-normalised rules and raw JSON states.
    if state in state_rules:
        rule = state_rules[state]
        if (
            isinstance(rule, dict)
            and rule.get("enabled", True) is not False
        ):
            return rule

    return normalize_state_rules(
        state_rules
    ).get(
        state
    )


def enabled_states(
    state_rules: dict[str, dict] | None,
) -> set[str]:
    if not state_rules:
        return set(
            STATE_PRIORITY
        )

    if all(
        key in STATE_PRIORITY
        for key in state_rules
    ):
        return {
            state
            for state, rule in state_rules.items()
            if (
                isinstance(rule, dict)
                and rule.get("enabled", True) is not False
            )
        }

    return set(
        normalize_state_rules(
            state_rules
        )
    )


def state_colours(
    state_rules: dict[str, dict] | None,
) -> dict[str, str]:
    colours = dict(
        DEFAULT_STATE_COLOURS
    )

    if not state_rules:
        return colours

    rules = (
        state_rules
        if all(
            key in STATE_PRIORITY
            for key in state_rules
        )
        else normalize_state_rules(
            state_rules
        )
    )

    for state, rule in rules.items():
        colour = rule.get(
            "colour",
            rule.get(
                "color"
            ),
        )

        if colour:
            colours[state] = str(
                colour
            )

    return colours


def state_display_names(
    state_rules: dict[str, dict] | None,
) -> dict[str, str]:
    names = dict(
        DEFAULT_STATE_DISPLAY_NAMES
    )

    if not state_rules:
        return names

    rules = (
        state_rules
        if all(
            key in STATE_PRIORITY
            for key in state_rules
        )
        else normalize_state_rules(
            state_rules
        )
    )

    for state, rule in rules.items():
        display_name = rule.get(
            "display_name"
        )

        if display_name:
            names[state] = str(
                display_name
            )

    return names


def _inside_rule_envelope(
    horizontal_nm: float,
    vertical_ft: float | None,
    rule: dict | None,
) -> bool:
    if rule is None:
        return False

    horizontal_limit = rule.get(
        "horizontal_nm"
    )
    vertical_limit = rule.get(
        "vertical_ft"
    )

    if horizontal_limit is None:
        return False

    if horizontal_nm >= float(
        horizontal_limit
    ):
        return False

    if (
        vertical_ft is None
        or not np.isfinite(
            vertical_ft
        )
    ):
        # With no vertical information, only proximity can be asserted
        # conservatively. Callers decide whether that state is enabled.
        return True

    if vertical_limit is None:
        return True

    return (
        vertical_ft
        < float(
            vertical_limit
        )
    )

def combine_states(current_state: str, new_state: str) -> str:
    return (
        new_state
        if STATE_PRIORITY[new_state] > STATE_PRIORITY[current_state]
        else current_state
    )



def _normalised_state_rules(
    state_rules: dict[str, dict] | None,
) -> dict[str, dict]:
    if not state_rules:
        return {}

    if all(
        key in STATE_PRIORITY
        for key in state_rules
    ):
        return state_rules

    return normalize_state_rules(
        state_rules
    )


def vertical_separation_for_rule(
    vertical_ft: float | None,
    *,
    red_vertical_ft: float,
    tolerance_ft: float = 0.0,
) -> float | None:
    """
    Apply the numerical tolerance used by the separation rule.

    Example for ENROUTE:
        red_vertical_ft = 1000
        tolerance_ft = 1

    999.0 .. 1001.0 ft is treated as exactly 1000 ft. This prevents a
    metres->feet conversion / CSV quantisation artefact such as 999.9998 ft
    from being labelled as a loss of separation.
    """
    if (
        vertical_ft is None
        or not np.isfinite(
            vertical_ft
        )
    ):
        return None

    value = float(
        vertical_ft
    )

    tolerance_ft = max(
        0.0,
        float(
            tolerance_ft
        ),
    )

    if (
        abs(
            value
            - float(
                red_vertical_ft
            )
        )
        <= tolerance_ft
    ):
        return float(
            red_vertical_ft
        )

    return value


def profile_vertical_rate_deadband_fpm(
    state_rules: dict[str, dict] | None,
) -> float:
    """
    Return the vertical-rate deadband configured in the CONFLICT rule.
    """
    rules = _normalised_state_rules(
        state_rules
    )

    red_rule = rules.get(
        STATE_CONFLICT,
        {},
    )

    try:
        return max(
            0.0,
            float(
                red_rule.get(
                    "vertical_rate_deadband_fpm",
                    0.0,
                )
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def classify_pair_dynamic(
    horizontal_nm: float,
    vertical_ft: float | None,
    *,
    next_horizontal_nm: float | None,
    next_vertical_ft: float | None,
    state_rules: dict[str, dict],
) -> str:
    """
    Dynamic ENROUTE-style classification.

    RED
        H < red horizontal minimum AND effective V < red vertical minimum.

    ORANGE / YELLOW
        No current RED.
        Pair lies inside the state's 2-D envelope.
        The separator that is currently keeping the pair legal is actually
        deteriorating between the current and next prediction samples.

    Active separator logic
    ----------------------
    H < red H, V >= red V:
        vertical separation is the active separator. It must be decreasing.

    H >= red H, V < red V:
        horizontal separation is the active separator. It must be decreasing.

    H >= red H, V >= red V:
        both dimensions independently satisfy the minima. Both must be
        decreasing before a yellow/orange dynamic warning is asserted.

    This deliberately leaves a stable RVSM crossing such as:
        H = 0.3 NM, V = 1000 ft, relative vertical trend = 0
    as NORMAL rather than HIGH_RISK/CONFLICT.
    """
    rules = _normalised_state_rules(
        state_rules
    )

    red_rule = rules.get(
        STATE_CONFLICT
    )

    if red_rule is None:
        raise ValueError(
            "state_rules must contain an enabled CONFLICT state"
        )

    red_horizontal = float(
        red_rule[
            "horizontal_nm"
        ]
    )

    red_vertical = float(
        red_rule[
            "vertical_ft"
        ]
    )

    vertical_tolerance = float(
        red_rule.get(
            "vertical_tolerance_ft",
            0.0,
        )
    )

    vertical_rule = (
        vertical_separation_for_rule(
            vertical_ft,
            red_vertical_ft=red_vertical,
            tolerance_ft=vertical_tolerance,
        )
    )

    next_vertical_rule = (
        vertical_separation_for_rule(
            next_vertical_ft,
            red_vertical_ft=red_vertical,
            tolerance_ft=vertical_tolerance,
        )
    )

    vertical_valid = (
        vertical_rule is not None
        and np.isfinite(
            vertical_rule
        )
    )

    h_red = (
        float(
            horizontal_nm
        )
        < red_horizontal
    )

    v_red = (
        vertical_valid
        and float(
            vertical_rule
        )
        < red_vertical
    )

    if (
        vertical_valid
        and h_red
        and v_red
    ):
        return STATE_CONFLICT

    yellow_rule = rules.get(
        STATE_PROXIMITY
    )

    orange_rule = rules.get(
        STATE_HIGH_RISK
    )

    # Missing vertical information cannot justify orange/red. Preserve the
    # conservative legacy behaviour by allowing YELLOW on horizontal proximity.
    if not vertical_valid:
        if (
            yellow_rule is not None
            and _inside_rule_envelope(
                horizontal_nm,
                vertical_ft,
                yellow_rule,
            )
        ):
            return STATE_PROXIMITY

        return STATE_NORMAL

    if (
        next_horizontal_nm is None
        or not np.isfinite(
            next_horizontal_nm
        )
    ):
        return STATE_NORMAL

    if (
        next_vertical_rule is None
        or not np.isfinite(
            next_vertical_rule
        )
    ):
        return STATE_NORMAL

    # Use the looser yellow rule for trend epsilons when available, otherwise
    # fall back to small numerical epsilons.
    trend_rule = (
        yellow_rule
        or orange_rule
        or {}
    )

    horizontal_epsilon_nm = float(
        trend_rule.get(
            "horizontal_deterioration_epsilon_nm",
            0.01,
        )
    )

    vertical_epsilon_ft = float(
        trend_rule.get(
            "vertical_deterioration_epsilon_ft",
            1.0,
        )
    )

    horizontal_deteriorating = (
        float(
            next_horizontal_nm
        )
        < float(
            horizontal_nm
        )
        - horizontal_epsilon_nm
    )

    vertical_deteriorating = (
        float(
            next_vertical_rule
        )
        < float(
            vertical_rule
        )
        - vertical_epsilon_ft
    )

    # Determine which dimension is currently preventing a true loss of
    # separation and require that separator to be deteriorating.
    if (
        h_red
        and not v_red
    ):
        active_separator_deteriorating = (
            vertical_deteriorating
        )

    elif (
        not h_red
        and v_red
    ):
        active_separator_deteriorating = (
            horizontal_deteriorating
        )

    else:
        # Both dimensions currently satisfy their regulatory minima.
        # Do not issue a dynamic alert unless both are moving toward the
        # protected volume.
        active_separator_deteriorating = (
            horizontal_deteriorating
            and vertical_deteriorating
        )

    if not active_separator_deteriorating:
        return STATE_NORMAL

    # Tight envelope gets priority over the broader proximity envelope.
    if (
        orange_rule is not None
        and _inside_rule_envelope(
            horizontal_nm,
            vertical_rule,
            orange_rule,
        )
    ):
        return STATE_HIGH_RISK

    if (
        yellow_rule is not None
        and _inside_rule_envelope(
            horizontal_nm,
            vertical_rule,
            yellow_rule,
        )
    ):
        return STATE_PROXIMITY

    return STATE_NORMAL


def classify_pair(
    horizontal_nm: float,
    vertical_ft: float | None,
    *,
    min_horizontal_nm: float | None = None,
    min_vertical_ft: float | None = None,
    warning_horizontal_nm: float | None = None,
    warning_vertical_ft: float | None = None,
    state_rules: dict[str, dict] | None = None,
) -> str:
    """
    Classify one aircraft pair.

    Preferred mode
    --------------
    Pass state_rules loaded from separation_profiles.json.

    In that mode each safety state can be independently present/absent.
    For example, if HIGH_RISK / orange is deleted or disabled, a pair that
    would previously have been orange falls through to PROXIMITY / yellow
    when it is inside the yellow envelope.

    Backward-compatible mode
    ------------------------
    If state_rules is omitted, the original four-state thresholds supplied
    through min_* and warning_* arguments are used.
    """

    if state_rules is not None:
        rules = (
            state_rules
            if all(
                key in STATE_PRIORITY
                for key in state_rules
            )
            else normalize_state_rules(
                state_rules
            )
        )

        red_rule = rules.get(
            STATE_CONFLICT
        )
        orange_rule = rules.get(
            STATE_HIGH_RISK
        )
        yellow_rule = rules.get(
            STATE_PROXIMITY
        )

        if red_rule is None:
            raise ValueError(
                "state_rules must contain an enabled CONFLICT state"
            )

        red_horizontal = float(
            red_rule[
                "horizontal_nm"
            ]
        )
        red_vertical = float(
            red_rule[
                "vertical_ft"
            ]
        )

        vertical_valid = (
            vertical_ft is not None
            and np.isfinite(
                vertical_ft
            )
        )

        h_red = (
            horizontal_nm
            < red_horizontal
        )

        vertical_tolerance = float(
            red_rule.get(
                "vertical_tolerance_ft",
                0.0,
            )
        )

        vertical_rule = (
            vertical_separation_for_rule(
                vertical_ft,
                red_vertical_ft=red_vertical,
                tolerance_ft=vertical_tolerance,
            )
        )

        v_red = (
            vertical_rule is not None
            and np.isfinite(
                vertical_rule
            )
            and vertical_rule
            < red_vertical
        )

        # RED always has first priority.
        if (
            vertical_valid
            and h_red
            and v_red
        ):
            return STATE_CONFLICT

        # Without vertical information we cannot assert orange/red. If yellow
        # exists and the horizontal threshold is breached, mark proximity.
        if not vertical_valid:
            if (
                yellow_rule is not None
                and _inside_rule_envelope(
                    horizontal_nm,
                    vertical_ft,
                    yellow_rule,
                )
            ):
                return STATE_PROXIMITY

            return STATE_NORMAL

        # ORANGE is optional.
        if (
            orange_rule is not None
            and _inside_rule_envelope(
                horizontal_nm,
                vertical_ft,
                orange_rule,
            )
            and (
                h_red
                ^ bool(
                    v_red
                )
            )
        ):
            return STATE_HIGH_RISK

        # YELLOW is optional. If orange is absent, orange-like geometry
        # naturally falls through here and is treated as proximity.
        if (
            yellow_rule is not None
            and _inside_rule_envelope(
                horizontal_nm,
                vertical_ft,
                yellow_rule,
            )
        ):
            return STATE_PROXIMITY

        return STATE_NORMAL

    # -----------------------------------------------------------------
    # Legacy numeric-threshold mode
    # -----------------------------------------------------------------
    if None in (
        min_horizontal_nm,
        min_vertical_ft,
        warning_horizontal_nm,
        warning_vertical_ft,
    ):
        raise ValueError(
            "Either state_rules or all legacy numeric thresholds must be supplied"
        )

    if vertical_ft is None or not np.isfinite(vertical_ft):
        return (
            STATE_PROXIMITY
            if horizontal_nm < float(
                warning_horizontal_nm
            )
            else STATE_NORMAL
        )

    h_min = (
        horizontal_nm
        < float(
            min_horizontal_nm
        )
    )
    v_min = (
        vertical_ft
        < float(
            min_vertical_ft
        )
    )

    if h_min and v_min:
        return STATE_CONFLICT

    inside_warning = (
        horizontal_nm
        < float(
            warning_horizontal_nm
        )
        and vertical_ft
        < float(
            warning_vertical_ft
        )
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
    *,
    vertical_rate_deadband_fpm: float = 0.0,
) -> tuple[float, float, float | None]:
    """
    Constant-velocity/track/vertical-rate kinematic projection.

    Small reported vertical rates can be suppressed with
    vertical_rate_deadband_fpm so nominally level aircraft remain level in
    the prediction model.
    """
    lon = _numeric(row, "longitude", "lon")
    lat = _numeric(row, "latitude", "lat")

    if lon is None or lat is None:
        raise ValueError(
            "Aircraft position is missing"
        )

    speed_ms = _speed_ms(row)
    track_deg = _track_deg(row)

    if (
        speed_ms is not None
        and track_deg is not None
    ):
        projected_lon, projected_lat, _ = GEOD.fwd(
            lon,
            lat,
            track_deg,
            speed_ms
            * float(
                time_seconds
            ),
        )
    else:
        projected_lon, projected_lat = (
            lon,
            lat,
        )

    altitude_ft = _baro_altitude_ft(
        row
    )

    if altitude_ft is not None:
        vertical_rate_fpm = (
            _vertical_rate_fpm(
                row
            )
        )

        if (
            abs(
                vertical_rate_fpm
            )
            <= max(
                0.0,
                float(
                    vertical_rate_deadband_fpm
                ),
            )
        ):
            vertical_rate_fpm = 0.0

        altitude_ft += (
            vertical_rate_fpm
            * float(
                time_seconds
            )
            / 60.0
        )

    return (
        float(
            projected_lon
        ),
        float(
            projected_lat
        ),
        altitude_ft,
    )


def pair_separation_at_time(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
    time_seconds: float,
    *,
    vertical_rate_deadband_fpm: float = 0.0,
) -> tuple[float, float | None]:
    lon_a, lat_a, alt_a = project_aircraft_state(
        aircraft_a,
        time_seconds,
        vertical_rate_deadband_fpm=(
            vertical_rate_deadband_fpm
        ),
    )

    lon_b, lat_b, alt_b = project_aircraft_state(
        aircraft_b,
        time_seconds,
        vertical_rate_deadband_fpm=(
            vertical_rate_deadband_fpm
        ),
    )

    _, _, distance_m = GEOD.inv(
        lon_a,
        lat_a,
        lon_b,
        lat_b,
    )

    horizontal_nm = (
        float(
            distance_m
        )
        / 1852.0
    )

    vertical_ft = (
        None
        if (
            alt_a is None
            or alt_b is None
        )
        else abs(
            float(
                alt_a
            )
            - float(
                alt_b
            )
        )
    )

    return (
        horizontal_nm,
        vertical_ft,
    )


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
    warning_factor: float = 1.5,
    state_rules: dict[str, dict] | None = None,
    lookahead_minutes: Iterable[float] = (1, 2, 5),
    prediction_step_seconds: int = 5,
    check_ground_aircraft: bool = False,
) -> tuple[list[dict], dict, dict, int]:
    """
    Analyse current and predicted pairwise separation.

    With state_rules, the current ENROUTE profile can use dynamic active-
    separator logic for NORMAL / PROXIMITY / HIGH_RISK / CONFLICT.

    Returns:
      pair_results
      aircraft_current_states
      aircraft_interval_states
      skipped_pairs
    """
    if prediction_step_seconds <= 0:
        raise ValueError(
            "prediction_step_seconds must be positive"
        )

    warning_horizontal_nm = (
        min_horizontal_nm
        * warning_factor
    )

    warning_vertical_ft = (
        min_vertical_ft
        * warning_factor
    )

    intervals = _interval_boundaries_seconds(
        lookahead_minutes
    )

    max_seconds = intervals[-1][1]

    rows = list(
        aircraft.iterrows()
    )

    aircraft_states = {
        index: STATE_NORMAL
        for index in aircraft.index
    }

    interval_states = {
        index: [
            STATE_NORMAL
            for _ in intervals
        ]
        for index in aircraft.index
    }

    pair_results: list[dict] = []
    skipped_pairs = 0

    dynamic_rules = (
        _normalised_state_rules(
            state_rules
        )
    )

    use_dynamic = bool(
        dynamic_rules
        and (
            dynamic_rules.get(
                STATE_PROXIMITY,
                {},
            ).get(
                "requires_active_separator_deterioration",
                False,
            )
            or dynamic_rules.get(
                STATE_HIGH_RISK,
                {},
            ).get(
                "requires_active_separator_deterioration",
                False,
            )
        )
    )

    vertical_rate_deadband_fpm = (
        profile_vertical_rate_deadband_fpm(
            state_rules
        )
        if state_rules
        else 0.0
    )

    red_rule = (
        dynamic_rules.get(
            STATE_CONFLICT,
            {},
        )
        if dynamic_rules
        else {}
    )

    red_vertical = float(
        red_rule.get(
            "vertical_ft",
            min_vertical_ft,
        )
    )

    vertical_tolerance = float(
        red_rule.get(
            "vertical_tolerance_ft",
            0.0,
        )
    )

    def separation_at(
        aircraft_a,
        aircraft_b,
        time_s,
    ):
        return pair_separation_at_time(
            aircraft_a,
            aircraft_b,
            float(
                time_s
            ),
            vertical_rate_deadband_fpm=(
                vertical_rate_deadband_fpm
            ),
        )

    def classify_at(
        aircraft_a,
        aircraft_b,
        time_s,
        h_nm,
        v_ft,
    ):
        if (
            use_dynamic
            and state_rules is not None
        ):
            next_time_s = min(
                float(
                    time_s
                )
                + float(
                    prediction_step_seconds
                ),
                float(
                    max_seconds
                ),
            )

            if next_time_s <= float(
                time_s
            ):
                next_h_nm = h_nm
                next_v_ft = v_ft
            else:
                (
                    next_h_nm,
                    next_v_ft,
                ) = separation_at(
                    aircraft_a,
                    aircraft_b,
                    next_time_s,
                )

            return classify_pair_dynamic(
                h_nm,
                v_ft,
                next_horizontal_nm=(
                    next_h_nm
                ),
                next_vertical_ft=(
                    next_v_ft
                ),
                state_rules=(
                    state_rules
                ),
            )

        return classify_pair(
            h_nm,
            v_ft,
            min_horizontal_nm=(
                min_horizontal_nm
            ),
            min_vertical_ft=(
                min_vertical_ft
            ),
            warning_horizontal_nm=(
                warning_horizontal_nm
            ),
            warning_vertical_ft=(
                warning_vertical_ft
            ),
            state_rules=(
                state_rules
            ),
        )

    for i in range(
        len(
            rows
        )
    ):
        index_a, aircraft_a = rows[i]

        if (
            _on_ground(
                aircraft_a
            )
            and not check_ground_aircraft
        ):
            continue

        for j in range(
            i + 1,
            len(
                rows
            ),
        ):
            index_b, aircraft_b = rows[j]

            if (
                _on_ground(
                    aircraft_b
                )
                and not check_ground_aircraft
            ):
                continue

            (
                horizontal_nm,
                vertical_ft,
            ) = separation_at(
                aircraft_a,
                aircraft_b,
                0.0,
            )

            if vertical_ft is None:
                skipped_pairs += 1

            current_state = classify_at(
                aircraft_a,
                aircraft_b,
                0.0,
                horizontal_nm,
                vertical_ft,
            )

            aircraft_states[
                index_a
            ] = combine_states(
                aircraft_states[
                    index_a
                ],
                current_state,
            )

            aircraft_states[
                index_b
            ] = combine_states(
                aircraft_states[
                    index_b
                ],
                current_state,
            )

            pair_interval_states: list[str] = []

            pair_min_horizontal_nm = (
                horizontal_nm
            )

            pair_min_vertical_at_cpa_ft = (
                vertical_ft
            )

            pair_tcpa_s = 0.0

            continuous_tcpa = (
                _continuous_tcpa_seconds(
                    aircraft_a,
                    aircraft_b,
                    max_seconds,
                )
            )

            if continuous_tcpa is not None:
                (
                    cpa_h,
                    cpa_v,
                ) = separation_at(
                    aircraft_a,
                    aircraft_b,
                    continuous_tcpa,
                )

                pair_min_horizontal_nm = (
                    cpa_h
                )

                pair_min_vertical_at_cpa_ft = (
                    cpa_v
                )

                pair_tcpa_s = (
                    continuous_tcpa
                )

            for (
                interval_index,
                (
                    start_s,
                    end_s,
                ),
            ) in enumerate(
                intervals
            ):
                worst = STATE_NORMAL

                sample_times = np.arange(
                    start_s,
                    end_s
                    + prediction_step_seconds
                    * 0.5,
                    prediction_step_seconds,
                    dtype=float,
                )

                if (
                    sample_times.size == 0
                    or sample_times[-1] < end_s
                ):
                    sample_times = np.append(
                        sample_times,
                        end_s,
                    )

                for time_s in sample_times:
                    (
                        h_nm,
                        v_ft,
                    ) = separation_at(
                        aircraft_a,
                        aircraft_b,
                        float(
                            time_s
                        ),
                    )

                    state = classify_at(
                        aircraft_a,
                        aircraft_b,
                        float(
                            time_s
                        ),
                        h_nm,
                        v_ft,
                    )

                    worst = combine_states(
                        worst,
                        state,
                    )

                pair_interval_states.append(
                    worst
                )

                interval_states[
                    index_a
                ][
                    interval_index
                ] = combine_states(
                    interval_states[
                        index_a
                    ][
                        interval_index
                    ],
                    worst,
                )

                interval_states[
                    index_b
                ][
                    interval_index
                ] = combine_states(
                    interval_states[
                        index_b
                    ][
                        interval_index
                    ],
                    worst,
                )

            vertical_rule_ft = (
                vertical_separation_for_rule(
                    vertical_ft,
                    red_vertical_ft=(
                        red_vertical
                    ),
                    tolerance_ft=(
                        vertical_tolerance
                    ),
                )
            )

            result = {
                "index_a": index_a,
                "index_b": index_b,
                "icao24_a": str(
                    aircraft_a.get(
                        "icao24",
                        "",
                    )
                    or ""
                ),
                "icao24_b": str(
                    aircraft_b.get(
                        "icao24",
                        "",
                    )
                    or ""
                ),
                "callsign_a": _callsign(
                    aircraft_a
                ),
                "callsign_b": _callsign(
                    aircraft_b
                ),
                "horizontal_sep_nm": (
                    horizontal_nm
                ),
                "vertical_sep_ft": (
                    vertical_ft
                ),
                "vertical_sep_rule_ft": (
                    vertical_rule_ft
                ),
                "state_current": (
                    current_state
                ),
                "status_current": (
                    current_state
                ),
                "cpa_nm": (
                    pair_min_horizontal_nm
                ),
                "tcpa_s": (
                    pair_tcpa_s
                ),
                "vertical_sep_at_cpa_ft": (
                    pair_min_vertical_at_cpa_ft
                ),
                "predicted_conflict": any(
                    state == STATE_CONFLICT
                    for state
                    in pair_interval_states
                ),
            }

            for (
                (
                    start_s,
                    end_s,
                ),
                state,
            ) in zip(
                intervals,
                pair_interval_states,
            ):
                result[
                    _interval_column_name(
                        start_s,
                        end_s,
                    )
                ] = state

            pair_results.append(
                result
            )

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

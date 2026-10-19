"""Units and common filters for the SCAT ASTERIX-derived fields."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ASTERIX CAT062 decoded values used by the SCAT JSON export.
MPS_TO_KT = 1.9438444924406
MPS_TO_FPM = 196.8503937008
FT_TO_M = 0.3048

# Default first-pass en-route analysis band.  The ingestion layer keeps all
# observed levels; the detector applies this band unless explicitly disabled.
DEFAULT_ANALYSIS_FL_MIN = 290.0
DEFAULT_ANALYSIS_FL_MAX = 410.0


def flight_level_to_ft(flight_level: float) -> float:
    """Convert a flight level (hundreds of feet) to feet."""
    return float(flight_level) * 100.0


def mps_to_kt(value: float) -> float:
    return float(value) * MPS_TO_KT


def mps_to_fpm(value: float) -> float:
    return float(value) * MPS_TO_FPM


def filter_flight_level_band(
    dataframe: pd.DataFrame,
    min_fl: float | None,
    max_fl: float | None,
    *,
    altitude_column: str = "baro_altitude_ft",
) -> pd.DataFrame:
    """Keep rows whose barometric altitude is inside an inclusive FL band."""
    if min_fl is None and max_fl is None:
        return dataframe.copy()
    if min_fl is None or max_fl is None or min_fl > max_fl:
        raise ValueError("Both FL bounds are required and min_fl must be <= max_fl")
    if altitude_column not in dataframe.columns:
        raise ValueError(f"Missing altitude column: {altitude_column}")

    altitude_ft = pd.to_numeric(dataframe[altitude_column], errors="coerce")
    mask = altitude_ft.between(
        flight_level_to_ft(min_fl),
        flight_level_to_ft(max_fl),
        inclusive="both",
    )
    return dataframe.loc[mask].copy()


def conversion_error_summary(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Return maximum internal conversion errors for a processed SCAT table."""
    result: dict[str, float | int] = {"rows": int(len(dataframe))}
    if dataframe.empty:
        result.update(
            {
                "altitude_max_error_ft": 0.0,
                "speed_max_error_kt": 0.0,
                "vertical_rate_max_error_fpm": 0.0,
            }
        )
        return result

    def max_error(left: pd.Series, right: pd.Series) -> float:
        lhs = pd.to_numeric(left, errors="coerce")
        rhs = pd.to_numeric(right, errors="coerce")
        mask = lhs.notna() & rhs.notna()
        if not mask.any():
            return 0.0
        return float(np.max(np.abs(lhs.loc[mask] - rhs.loc[mask])))

    result["altitude_max_error_ft"] = max_error(
        dataframe["baro_altitude_ft"],
        dataframe["baroaltitude"] / FT_TO_M,
    )
    result["speed_max_error_kt"] = max_error(
        dataframe["ground_speed_kt"],
        dataframe["velocity"] * MPS_TO_KT,
    )
    result["vertical_rate_max_error_fpm"] = max_error(
        dataframe["vertical_rate_fpm"],
        dataframe["vertical_rate"] * MPS_TO_FPM,
    )
    return result

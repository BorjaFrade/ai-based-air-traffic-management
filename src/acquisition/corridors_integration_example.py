"""
Minimal integration into the script that already loads WAYPOINTS and calls
opensky_plot.render_map().
"""

from pathlib import Path

from area_presets import load_corridors_for_area


SCRIPT_DIR = Path(__file__).resolve().parent

CORRIDORS_FILE = (
    SCRIPT_DIR
    / "corridors.json"
)

# After AREA has been resolved:
CORRIDORS = load_corridors_for_area(
    CORRIDORS_FILE,
    area_key=SELECTED_REGION,
    area=AREA,
)


# ============================================================
# FLAGS
# ============================================================

# Master flag.
SHOW_CORRIDORS = True

# False -> draw every loaded ATS corridor.
# True  -> show only corridors that have aircraft nearby.
CORRIDORS_ONLY_NEAR_TRAFFIC = True

# Used only when CORRIDORS_ONLY_NEAR_TRAFFIC=True.
CORRIDOR_TRAFFIC_DISTANCE_NM = 20.0


# ============================================================
# RENDER
# ============================================================

fig, ax = render_map(
    aircraft=aircraft,
    area=AREA,
    bbox=BBOX,
    aircraft_states=aircraft_states,
    warnings=warnings,
    conflicts=conflicts,
    snapshot_time=snapshot_time,
    map_file=MAP_FILE,
    config=globals(),
    waypoints=WAYPOINTS,
    corridors=CORRIDORS,
    show=True,
    save=True,
)

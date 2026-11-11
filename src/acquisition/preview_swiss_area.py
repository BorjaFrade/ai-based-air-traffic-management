#!/usr/bin/env python3
"""Render a source-data placement check for the Swiss map configuration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from area_presets import load_area_presets, load_waypoints_for_area, resolve_area_preset


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parents[1] / "results" / "switzerland_area_preview.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    area = resolve_area_preset("SWITZERLAND", load_area_presets(HERE / "bbox_presets.json"))
    waypoints = load_waypoints_for_area(
        HERE / "waypoints.json", area["key"], area)
    bbox = area["bbox"]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_facecolor("#eef3f6")
    ax.set_xlim(bbox["lomin"], bbox["lomax"])
    ax.set_ylim(bbox["lamin"], bbox["lamax"])
    ax.set_aspect(1 / math.cos(math.radians(46.75)))
    ax.grid(alpha=.2)
    for boundary in area["reference_boundaries"]:
        points = boundary["points"]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color="#17507a", linewidth=1.5, linestyle=(0, (5, 3)),
                label="LSAS UIR FL195–FL660 (referencia actual)")
    for city in area["cities"]:
        ax.scatter(city["longitude"], city["latitude"], c="#5d6973", s=28,
                   marker="o", zorder=4)
        ax.annotate(city["name"], (city["longitude"], city["latitude"]),
                    xytext=(4, -10), textcoords="offset points", fontsize=8,
                    color="#4d5963")
    for airport in area["airports"]:
        ax.scatter(airport["lon"], airport["lat"], c="#086b9e", s=65,
                   marker="^", zorder=5)
        ax.annotate(airport["icao"], (airport["lon"], airport["lat"]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8,
                    fontweight="bold", color="#07557e")
    for waypoint in waypoints:
        ax.scatter(waypoint["lon"], waypoint["lat"], c="#b45a19", s=32,
                   marker="s", zorder=5)
        ax.annotate(waypoint["name"], (waypoint["lon"], waypoint["lat"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7,
                    color="#743c16")
    ax.set_title("Suiza: comprobación de capas aeronáuticas y geográficas", loc="left")
    ax.set_xlabel("Longitud (° E, WGS84)")
    ax.set_ylabel("Latitud (° N, WGS84)")
    fig.text(.12, .03,
             "Caja = adquisición OpenSky; línea azul = UIR actual, no límite histórico 2019/2025. "
             "Cuadrados = puntos FRA; triángulos = aeropuertos; círculos = ciudades.",
             fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()

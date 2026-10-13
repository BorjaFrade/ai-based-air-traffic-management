#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the ATC sectorisation published by ENAIRE and generate context maps
for an air traffic management research project.

OFFICIAL ENAIRE SOURCES (ArcGIS REST)
------------------------------------
Sectors:
https://servais.enaire.es/insignia/rest/services/INSIGNIA_SRV/Aero_SRV_VIGOR_Sectores_V1/FeatureServer/2

Airspaces (including FIR/UIR):
https://servais.enaire.es/insignia/rest/services/INSIGNIA_SRV/Aero_SRV_VIGOR_V1/FeatureServer/41

OUTPUTS
-------
data/enaire_airspace/
  enaire_sectors_raw.geojson
  enaire_airspace_raw.geojson
  enaire_sectors.geojson
  enaire_fir_uir.geojson
  enaire_acc_regions.geojson
  enaire_sectors_metadata.csv

results/enaire_airspace/
  spanish_airspace_all.png
  spanish_airspace_peninsula_balearic.png

Map representation:
- Madrid, Barcelona and Canarias FIR/UIR: thick solid line.
- ENAIRE sectors: thin dashed line, with optional sector names/identifiers.
- Areas managed by Sevilla ACC and Palma ACC: thicker dash-dot line.

IMPORTANT
---------
- Sevilla and Palma are NOT treated as independent FIRs. Their outlines are
  built from the sectors that ENAIRE assigns to those dependencies/regions.
- Sectors may overlap laterally because different vertical layers and
  operational configurations may exist. Published lower/upper limits are kept
  in the exported metadata for later analysis.
- ENAIRE GIS layers may contain geometries that GEOS/Shapely considers invalid.
  The script validates and repairs them before union/dissolve operations.
- For strict reproducibility, archive the downloaded GeoJSON files together
  with the AIRAC/date used for the experiment.

Installation (macOS / Linux):
    python3 -m pip install -r requirements_enaire_airspace.txt

Run:
    python3 download_enaire_airspace_map.py
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.lines import Line2D
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

try:
    from shapely import make_valid
except ImportError:  # Shapely < 2.0 fallback
    from shapely.validation import make_valid

try:
    import contextily as ctx
except Exception:
    ctx = None


# =============================================================================
# CONFIGURATION
# =============================================================================

SECTORS_LAYER_URL = (
    "https://servais.enaire.es/insignia/rest/services/"
    "INSIGNIA_SRV/Aero_SRV_VIGOR_Sectores_V1/FeatureServer/2"
)

AIRSPACE_LAYER_URL = (
    "https://servais.enaire.es/insignia/rest/services/"
    "INSIGNIA_SRV/Aero_SRV_VIGOR_V1/FeatureServer/41"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "enaire_airspace"
RESULTS_DIR = PROJECT_ROOT / "results" / "enaire_airspace"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SECTORS_RAW_FILE = DATA_DIR / "enaire_sectors_raw.geojson"
AIRSPACE_RAW_FILE = DATA_DIR / "enaire_airspace_raw.geojson"
SECTORS_FILE = DATA_DIR / "enaire_sectors.geojson"
FIRS_FILE = DATA_DIR / "enaire_fir_uir.geojson"
ACC_FILE = DATA_DIR / "enaire_acc_regions.geojson"
METADATA_FILE = DATA_DIR / "enaire_sectors_metadata.csv"

MAP_ALL_FILE = RESULTS_DIR / "spanish_airspace_all.png"
MAP_PENINSULA_FILE = RESULTS_DIR / "spanish_airspace_peninsula_balearic.png"

# If False, existing raw GeoJSON files are reused. If they do not exist, they
# are downloaded automatically. This is convenient after the first successful
# download and avoids unnecessary requests to ENAIRE.
FORCE_DOWNLOAD = False

# If True, keep only sectors that can be identified as elemental sectors from
# the ENAIRE attribute ELEMENTAL_BOL.
ELEMENTAL_ONLY = False

# Add sector names/identifiers to the map.
# Set to False for a cleaner national overview.
ADD_SECTOR_NAMES = True

# Sector labels use the official ENAIRE identifier only
# (for example LECMSAO instead of "Santiago Oceanic").
SECTOR_LABEL_MODE = "id"

# Draw a small dark box behind sector labels so they remain readable on the
# satellite basemap.
SECTOR_LABEL_BOX = True

# ------------------------------------------------------------------
# Cruise-band sector filtering
# ------------------------------------------------------------------
# Show only ENAIRE sectors whose published vertical limits intersect the
# RVSM study band used in the TFM.
FILTER_BY_FL_RANGE = True
FL_MIN = 290
FL_MAX = 410

# If some records do not expose usable vertical limits, keep them only when
# this flag is True.  The script automatically disables the FL filter if the
# ENAIRE layer contains no usable vertical-limit information at all.
KEEP_SECTORS_WITH_UNKNOWN_VERTICAL_LIMITS = False

# ------------------------------------------------------------------
# Dense-label handling
# ------------------------------------------------------------------
# Labels are normally white.  When several sector labels are crowded in the
# same area, they are highlighted and slightly displaced to remain readable.
NORMAL_LABEL_COLOR = "#FFFFFF"
CROWDED_LABEL_COLOR = "#FFD400"

SECTOR_LABEL_FONTSIZE = 8.0
CROWDED_LABEL_FONTSIZE = 10.5

# Approximate display-space distance used to detect crowded labels.
LABEL_CLUSTER_DISTANCE_PX = 180

# Move labels in crowded clusters around their original sector-centre point.
MOVE_CROWDED_LABELS = True

# Draw a short leader line from a displaced label back to the sector-centre
# point so it remains clear which polygon the label belongs to.
ADD_LABEL_LEADER_LINES = True

# A web basemap is optional. If contextily is not available or tile retrieval
# fails, the aviation geometry is still plotted and exported.
USE_BASEMAP = True

# Request additional Esri World Imagery detail for the 8K export.
# Reduce to 1 or 0 if tile download becomes too heavy.
BASEMAP_ZOOM_ADJUST = 2

REQUEST_TIMEOUT_S = 90
PAGE_SIZE = 5000

OUTPUT_WIDTH_PX = 7680
OUTPUT_HEIGHT_PX = 4320
FIG_DPI = 160
FIGSIZE_8K = (
    OUTPUT_WIDTH_PX / FIG_DPI,
    OUTPUT_HEIGHT_PX / FIG_DPI,
)

# High-contrast overlay styling for satellite imagery. Important boundaries
# are drawn twice: a dark underlay plus a bright foreground stroke.
FIR_LINEWIDTH = 3.0
FIR_UNDERLAY_LINEWIDTH = 5.4
SECTOR_LINEWIDTH = 0.85
SECTOR_UNDERLAY_LINEWIDTH = 1.65
DELEGATED_LINEWIDTH = 2.4
DELEGATED_UNDERLAY_LINEWIDTH = 4.2
SECTOR_LINESTYLE = (0, (4, 3))
DELEGATED_LINESTYLE = "-."

FIR_COLOR = "#FFFFFF"
FIR_UNDERLAY_COLOR = "#101010"
SECTOR_COLOR = "#00E5FF"
SECTOR_UNDERLAY_COLOR = "#101010"
SEVILLA_COLOR = "#FF2EA6"
PALMA_COLOR = "#FFD400"
DELEGATED_UNDERLAY_COLOR = "#101010"


# =============================================================================
# ARCGIS REST DOWNLOAD
# =============================================================================

def query_arcgis_geojson(layer_url: str, where: str = "1=1") -> dict:
    """Download an ArcGIS FeatureServer layer with pagination as WGS84 GeoJSON."""
    query_url = layer_url.rstrip("/") + "/query"
    all_features = []
    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }

        response = requests.get(
            query_url,
            params=params,
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()

        if "error" in payload:
            raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))

        features = payload.get("features", [])
        all_features.extend(features)
        print(f"  downloaded {len(all_features)} features")

        if len(features) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return {
        "type": "FeatureCollection",
        "features": all_features,
    }


def save_geojson(payload: dict, path: Path) -> None:
    """Write a GeoJSON dictionary to disk using UTF-8."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def download_or_load(layer_url: str, path: Path) -> gpd.GeoDataFrame:
    """Download a layer or reuse a previously downloaded local GeoJSON file."""
    if FORCE_DOWNLOAD or not path.exists():
        print(f"Downloading {path.name}...")
        save_geojson(query_arcgis_geojson(layer_url), path)
    else:
        print(f"Reusing {path}")

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


# =============================================================================
# TEXT NORMALISATION AND CLASSIFICATION
# =============================================================================

def norm(value) -> str:
    """Normalise source text for robust comparisons."""
    if value is None:
        return ""

    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass

    text = str(value).upper().strip()
    replacements = str.maketrans("ÁÉÍÓÚÜÑ", "AEIOUUN")
    text = text.translate(replacements)
    return re.sub(r"\s+", " ", text)


def infer_acc(row: pd.Series) -> str:
    """
    Infer the controlling ACC/region from ENAIRE attributes and, as a fallback,
    from the sector identifier prefix.
    """
    ident = norm(row.get("IDENT_TXT", ""))
    dependency = norm(row.get("DEPENDENCIA_CODE", ""))
    region = norm(row.get("REGION_TXT", ""))
    name = norm(row.get("NOMBRE_TXT", ""))
    description = norm(row.get("DESCRIPCION_TXT", ""))
    text = " | ".join([dependency, region, name, description, ident])

    # Check delegated/managed dependencies before the host FIR to avoid
    # misclassifying Sevilla or Palma sectors.
    if "SEVILLA" in text or ident.startswith("LECS"):
        return "Sevilla ACC"
    if "PALMA" in text or ident.startswith("LECP"):
        return "Palma ACC"
    if "CANARIAS" in text or ident.startswith("GCCC"):
        return "Canarias ACC"
    if "BARCELONA" in text or ident.startswith("LECB"):
        return "Barcelona ACC"
    if "MADRID" in text or ident.startswith("LECM"):
        return "Madrid ACC"

    return "Unclassified"


def is_elemental(value) -> bool:
    """Interpret common representations of ENAIRE's elemental-sector flag."""
    return norm(value) in {
        "1", "TRUE", "T", "YES", "Y", "SI", "S", "ELEMENTAL", "E", "X"
    }


def classify_fir(row: pd.Series) -> str | None:
    """Identify the three Spanish FIR/UIR groups from the ENAIRE airspace layer."""
    ident = norm(row.get("IDENT_TXT", ""))
    name = norm(row.get("NAME_TXT", ""))
    type_code = norm(row.get("TYPE_CODE", ""))
    text = f"{ident} | {name} | {type_code}"

    if "FIR" not in text and "UIR" not in text:
        return None

    if "MADRID" in text or ident == "LECM":
        return "Madrid FIR/UIR"
    if "BARCELONA" in text or ident == "LECB":
        return "Barcelona FIR/UIR"
    if "CANARIAS" in text or ident == "GCCC":
        return "Canarias FIR/UIR"

    return None


# =============================================================================
# GEOMETRY VALIDATION AND REPAIR
# =============================================================================

def _polygonal_only(geom):
    """
    Keep only polygonal parts after make_valid().

    make_valid() can return a GeometryCollection when an invalid polygon is
    repaired. For sector outlines we only want Polygon/MultiPolygon content.
    """
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polygonal_parts = [
            part
            for part in geom.geoms
            if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
        ]

        if not polygonal_parts:
            return None

        try:
            return unary_union(polygonal_parts)
        except GEOSException:
            return unary_union([part.buffer(0) for part in polygonal_parts])

    return None


def repair_geometry(geom):
    """
    Repair invalid polygonal geometry from the ENAIRE GIS service.

    The preferred repair is make_valid(). buffer(0) is used only as a fallback.
    Returns None if no usable polygonal geometry can be recovered.
    """
    if geom is None or geom.is_empty:
        return None

    candidate = geom

    try:
        if not candidate.is_valid:
            candidate = make_valid(candidate)
    except Exception:
        try:
            candidate = candidate.buffer(0)
        except Exception:
            return None

    candidate = _polygonal_only(candidate)
    if candidate is None or candidate.is_empty:
        return None

    try:
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
    except Exception:
        return None

    if candidate is None or candidate.is_empty:
        return None

    return candidate


def repair_geodataframe(
    gdf: gpd.GeoDataFrame,
    id_field: str,
    label: str,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Repair invalid geometries and return a diagnostic report.

    The source geometry is replaced only when repair is required. Features that
    cannot be recovered are discarded from geometric operations, but the count
    and identifiers are reported to the console.
    """
    repaired = 0
    dropped = 0
    invalid_ids = []
    dropped_ids = []
    rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        feature_id = str(row.get(id_field, idx))

        if geom is None or geom.is_empty:
            dropped += 1
            dropped_ids.append(feature_id)
            continue

        was_invalid = not geom.is_valid
        fixed = repair_geometry(geom)

        if fixed is None or fixed.is_empty:
            dropped += 1
            dropped_ids.append(feature_id)
            continue

        if was_invalid:
            repaired += 1
            invalid_ids.append(feature_id)

        new_row = row.copy()
        new_row.geometry = fixed
        rows.append(new_row)

    result = gpd.GeoDataFrame(rows, columns=gdf.columns, crs=gdf.crs)

    report = {
        "label": label,
        "input": len(gdf),
        "repaired": repaired,
        "dropped": dropped,
        "invalid_ids": invalid_ids,
        "dropped_ids": dropped_ids,
    }

    print(f"\n=== Geometry validation: {label} ===")
    print(f"  input features : {report['input']}")
    print(f"  repaired       : {report['repaired']}")
    print(f"  dropped        : {report['dropped']}")

    if invalid_ids:
        preview = ", ".join(invalid_ids[:30])
        print(f"  repaired IDs   : {preview}")
        if len(invalid_ids) > 30:
            print(f"                   ... and {len(invalid_ids) - 30} more")

    if dropped_ids:
        preview = ", ".join(dropped_ids[:30])
        print(f"  dropped IDs    : {preview}")
        if len(dropped_ids) > 30:
            print(f"                   ... and {len(dropped_ids) - 30} more")

    return result, report


# =============================================================================
# DATA PREPARATION
# =============================================================================

def _numeric(value):
    """Return a finite float when possible, otherwise None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None

    text = str(value).strip().upper().replace(",", ".")
    if not text:
        return None

    # Accept strings such as "FL345", "345 FL", "29000 FT", etc.
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        number = float(match.group(0))
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def _vertical_limit_to_ft(value, uom=None, code=None, is_upper=False):
    """
    Convert an ENAIRE vertical-limit field to feet.

    The service schema can expose the limit through DISTVERT* fields or through
    simpler LOWER_VAL / UPPER_VAL fields.  This parser is intentionally
    defensive because AIRAC/schema revisions can change formatting.
    """
    text_value = "" if value is None else str(value).strip().upper()
    text_uom = "" if uom is None else str(uom).strip().upper()
    text_code = "" if code is None else str(code).strip().upper()
    combined = f"{text_value} {text_uom} {text_code}"

    if "SFC" in combined or "GND" in combined:
        return 0.0

    if "UNL" in combined or "UNLIMITED" in combined:
        return float("inf")

    number = _numeric(value)
    if number is None:
        # Sometimes the numerical value is embedded in another text field.
        number = _numeric(combined)
    if number is None:
        return None

    # Explicit flight-level notation.
    if "FL" in combined or text_uom in {"FL", "FLEVEL", "FLIGHT LEVEL"}:
        return number * 100.0

    # Metric altitude.
    if text_uom in {"M", "METRE", "METRES", "METER", "METERS"}:
        return number * 3.280839895

    # Feet are the most common representation in the sector layer.
    if text_uom in {"FT", "FEET", "FOOT"}:
        return number

    # Heuristic fallback:
    # values <= 600 are plausible flight levels; larger values are much more
    # likely to already be feet.
    if 0 <= number <= 600:
        return number * 100.0

    return number


def _extract_vertical_bounds_ft(row):
    """Return (lower_ft, upper_ft) from the available ENAIRE sector fields."""
    lower_candidates = [
        (
            row.get("DISTVERTLOWER_VAL"),
            row.get("DISTVERTLOWER_UOM"),
            row.get("DISTVERTLOWER_CODE"),
        ),
        (row.get("LOWER_VAL"), row.get("LOWER_UOM"), row.get("LOWER_CODE")),
    ]
    upper_candidates = [
        (
            row.get("DISTVERTUPPER_VAL"),
            row.get("DISTVERTUPPER_UOM"),
            row.get("DISTVERTUPPER_CODE"),
        ),
        (row.get("UPPER_VAL"), row.get("UPPER_UOM"), row.get("UPPER_CODE")),
    ]

    lower_ft = None
    for value, uom, code in lower_candidates:
        lower_ft = _vertical_limit_to_ft(value, uom, code, is_upper=False)
        if lower_ft is not None:
            break

    upper_ft = None
    for value, uom, code in upper_candidates:
        upper_ft = _vertical_limit_to_ft(value, uom, code, is_upper=True)
        if upper_ft is not None:
            break

    return lower_ft, upper_ft


def filter_sectors_by_fl_range(sectors: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep sectors whose published vertical interval intersects FL_MIN-FL_MAX.

    A sector [lower, upper] is relevant when:
        upper >= FL_MIN * 100
        and
        lower <= FL_MAX * 100
    """
    if not FILTER_BY_FL_RANGE or sectors.empty:
        return sectors.copy()

    target_lower_ft = FL_MIN * 100.0
    target_upper_ft = FL_MAX * 100.0

    work = sectors.copy()

    bounds = work.apply(_extract_vertical_bounds_ft, axis=1)
    work["LOWER_FT_PARSED"] = [b[0] for b in bounds]
    work["UPPER_FT_PARSED"] = [b[1] for b in bounds]

    known_mask = (
        work["LOWER_FT_PARSED"].notna()
        & work["UPPER_FT_PARSED"].notna()
    )

    known_count = int(known_mask.sum())

    if known_count == 0:
        print(
            "WARNING: no usable ENAIRE vertical limits were found in the "
            "sector layer; FL-range filtering has been skipped."
        )
        work["FL_RANGE_MATCH"] = True
        return work

    intersect_mask = known_mask & (
        (work["UPPER_FT_PARSED"] >= target_lower_ft)
        & (work["LOWER_FT_PARSED"] <= target_upper_ft)
    )

    if KEEP_SECTORS_WITH_UNKNOWN_VERTICAL_LIMITS:
        keep_mask = intersect_mask | ~known_mask
    else:
        keep_mask = intersect_mask

    work["FL_RANGE_MATCH"] = keep_mask

    print(
        f"\n=== FL range filter: FL{FL_MIN}-FL{FL_MAX} ===\n"
        f"  sector records before filter : {len(work)}\n"
        f"  records with parsed limits   : {known_count}\n"
        f"  records kept                 : {int(keep_mask.sum())}\n"
        f"  unknown vertical limits      : {int((~known_mask).sum())}"
    )

    return work[keep_mask].copy()


def prepare_sectors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate, classify and optionally filter the ENAIRE sector layer."""
    if "IDENT_TXT" not in gdf.columns:
        raise RuntimeError(
            "ENAIRE sector schema has changed: IDENT_TXT is missing."
        )

    sectors = gdf.copy()
    sectors = sectors[
        sectors.geometry.notna() & ~sectors.geometry.is_empty
    ].copy()

    sectors, _ = repair_geodataframe(
        sectors,
        id_field="IDENT_TXT",
        label="ENAIRE sectors",
    )

    sectors["ACC_GROUP"] = sectors.apply(infer_acc, axis=1)

    if ELEMENTAL_ONLY and "ELEMENTAL_BOL" in sectors.columns:
        mask = sectors["ELEMENTAL_BOL"].apply(is_elemental)

        if mask.any():
            sectors = sectors[mask].copy()
        else:
            print(
                "WARNING: ELEMENTAL_BOL could not be interpreted; "
                "all sectors will be kept."
            )

    sectors = filter_sectors_by_fl_range(sectors)
    return sectors


def prepare_firs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Identify, validate and dissolve Madrid, Barcelona and Canarias FIR/UIR."""
    airspace = gdf.copy()
    airspace["FIR_GROUP"] = airspace.apply(classify_fir, axis=1)
    airspace = airspace[airspace["FIR_GROUP"].notna()].copy()
    airspace = airspace[
        airspace.geometry.notna() & ~airspace.geometry.is_empty
    ].copy()

    if airspace.empty:
        raise RuntimeError(
            "Madrid, Barcelona and Canarias FIR/UIR could not be identified."
        )

    # Use FIR_GROUP as the diagnostic ID because IDENT_TXT may repeat between
    # FIR and UIR records.
    airspace, _ = repair_geodataframe(
        airspace,
        id_field="FIR_GROUP",
        label="Spanish FIR/UIR source geometries",
    )

    # FIR and UIR normally share the same lateral boundary. Dissolving avoids
    # duplicate outlines while preserving a single lateral geometry per group.
    try:
        return airspace.dissolve(by="FIR_GROUP", as_index=False)
    except GEOSException:
        # Extra defensive pass in case a topology issue survives make_valid().
        airspace["geometry"] = airspace.geometry.apply(repair_geometry)
        airspace = airspace[
            airspace.geometry.notna() & ~airspace.geometry.is_empty
        ].copy()
        return airspace.dissolve(by="FIR_GROUP", as_index=False)


def build_acc_regions(sectors: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build an approximate lateral outline for each ACC by unioning its sectors.

    Each geometry is repaired before union. If GEOS still reports a topology
    problem, a buffer(0) fallback is applied before retrying.
    """
    rows = []

    for acc_name, group in sectors.groupby("ACC_GROUP"):
        if acc_name == "Unclassified" or group.empty:
            continue

        valid_geometries = []

        for geom in group.geometry:
            fixed = repair_geometry(geom)
            if fixed is not None and not fixed.is_empty:
                valid_geometries.append(fixed)

        if not valid_geometries:
            print(f"WARNING: no valid geometries are available for {acc_name}")
            continue

        try:
            merged = unary_union(valid_geometries)
        except GEOSException as exc:
            print(
                f"WARNING: topology error while unioning {acc_name}: {exc}\n"
                "         Retrying after buffer(0) repair..."
            )

            repaired_again = []
            for geom in valid_geometries:
                try:
                    fixed = geom.buffer(0)
                    if fixed is not None and not fixed.is_empty:
                        repaired_again.append(fixed)
                except Exception:
                    pass

            if not repaired_again:
                print(f"WARNING: {acc_name} region could not be constructed")
                continue

            merged = unary_union(repaired_again)

        merged = repair_geometry(merged)
        if merged is None or merged.is_empty:
            print(f"WARNING: {acc_name} merged region is empty after repair")
            continue

        rows.append({
            "ACC_GROUP": acc_name,
            "geometry": merged,
        })

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=sectors.crs,
    )


def export_metadata(sectors: gpd.GeoDataFrame) -> None:
    """Export useful non-geometric sector attributes to CSV."""
    wanted = [
        "IDENT_TXT",
        "NOMBRE_TXT",
        "DEPENDENCIA_CODE",
        "REGION_TXT",
        "ELEMENTAL_BOL",
        "DISTVERTLOWER_VAL",
        "DISTVERTLOWER_UOM",
        "DISTVERTLOWER_CODE",
        "DISTVERTUPPER_VAL",
        "DISTVERTUPPER_UOM",
        "DISTVERTUPPER_CODE",
        "LOWER_VAL",
        "UPPER_VAL",
        "LOWER_FT_PARSED",
        "UPPER_FT_PARSED",
        "FL_RANGE_MATCH",
        "CAPACIDAD_ENAIRE",
        "OBSERVACIONES_TXT",
        "DESCRIPCION_TXT",
        "ACC_GROUP",
    ]

    columns = [column for column in wanted if column in sectors.columns]
    sectors.drop(columns="geometry")[columns].to_csv(
        METADATA_FILE,
        index=False,
    )


# =============================================================================
# MAP DRAWING
# =============================================================================

def add_basemap(ax) -> None:
    """
    Add Esri World Imagery, matching the satellite style used by the
    OpenSky visualiser.

    zoom_adjust=1 requests one additional tile zoom level so that the
    7680 x 4320 output remains sharper than the previous national map.
    """
    if not USE_BASEMAP or ctx is None:
        return

    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            crs="EPSG:3857",
            zoom="auto",
            zoom_adjust=BASEMAP_ZOOM_ADJUST,
            attribution_size=5,
            interpolation="bilinear",
            reset_extent=True,
        )
    except Exception as exc:
        print(f"WARNING: Esri World Imagery basemap is not available: {exc}")


def set_bbox(
    ax,
    lonmin: float,
    latmin: float,
    lonmax: float,
    latmax: float,
) -> None:
    """Set a WGS84 bounding box on a Web Mercator plot."""
    bounds = gpd.GeoSeries(
        [box(lonmin, latmin, lonmax, latmax)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    minx, miny, maxx, maxy = bounds.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def set_extent_from_firs(
    ax,
    firs_4326: gpd.GeoDataFrame,
    margin_deg: float = 1.0,
) -> None:
    """Fit the map around all Spanish FIR/UIR geometry with a margin."""
    minx, miny, maxx, maxy = firs_4326.total_bounds
    set_bbox(
        ax,
        minx - margin_deg,
        miny - margin_deg,
        maxx + margin_deg,
        maxy + margin_deg,
    )


def _sector_label_text(row) -> str:
    """
    Return the compact official ENAIRE sector identifier.

    Example:
        LECMSAO

    Long descriptive names are intentionally avoided because they occupy too
    much space on a national-scale map.
    """
    ident = str(row.get("IDENT_TXT", "") or "").strip()
    name = str(row.get("NOMBRE_TXT", "") or "").strip()
    return ident or name


def _sector_label_anchor(geom):
    """
    Return an approximately central label point for a sector polygon.

    For MultiPolygon geometries, use the largest polygon component. Prefer its
    centroid and fall back to representative_point() when the centroid lies
    outside a strongly concave polygon.
    """
    if geom is None or geom.is_empty:
        return None

    target = geom

    if geom.geom_type == "MultiPolygon":
        polygons = [g for g in geom.geoms if g is not None and not g.is_empty]
        if polygons:
            target = max(polygons, key=lambda g: g.area)

    elif geom.geom_type == "GeometryCollection":
        polygons = [
            g for g in geom.geoms
            if g.geom_type in ("Polygon", "MultiPolygon")
            and g is not None
            and not g.is_empty
        ]
        if polygons:
            target = max(polygons, key=lambda g: g.area)

    point = target.centroid

    try:
        if not target.covers(point):
            point = target.representative_point()
    except Exception:
        point = target.representative_point()

    return point


def _cluster_label_indices(pixel_xy, threshold_px):
    """Return connected components of labels closer than threshold_px."""
    n = len(pixel_xy)
    adjacency = [[] for _ in range(n)]

    for i in range(n):
        xi, yi = pixel_xy[i]
        for j in range(i + 1, n):
            xj, yj = pixel_xy[j]
            if math.hypot(xi - xj, yi - yj) < threshold_px:
                adjacency[i].append(j)
                adjacency[j].append(i)

    seen = set()
    clusters = []

    for start in range(n):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)

        clusters.append(component)

    return clusters


def draw_sector_labels(ax, sectors_3857: gpd.GeoDataFrame) -> None:
    """
    Draw compact ENAIRE sector IDs.

    Isolated labels stay at the approximate polygon centre.  Labels that form
    a dense display-space cluster are highlighted in yellow, made slightly
    larger and optionally displaced around their original anchors.
    """
    if not ADD_SECTOR_NAMES or sectors_3857.empty:
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    work = sectors_3857.copy()
    work["_LABEL_TEXT"] = work.apply(_sector_label_text, axis=1)
    work = work[work["_LABEL_TEXT"].astype(str).str.len() > 0].copy()

    if "IDENT_TXT" in work.columns:
        work["_LABEL_KEY"] = work["IDENT_TXT"].fillna("").astype(str).str.strip()
        blank = work["_LABEL_KEY"] == ""
        work.loc[blank, "_LABEL_KEY"] = work.loc[blank, "_LABEL_TEXT"]
    else:
        work["_LABEL_KEY"] = work["_LABEL_TEXT"]

    grouped = work.dissolve(
        by="_LABEL_KEY",
        aggfunc={"_LABEL_TEXT": "first"},
        as_index=False,
    )

    candidates = []

    for _, row in grouped.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        point = _sector_label_anchor(geom)
        if point is None:
            continue

        if not (xmin <= point.x <= xmax and ymin <= point.y <= ymax):
            continue

        candidates.append({
            "text": str(row["_LABEL_TEXT"]),
            "x": point.x,
            "y": point.y,
        })

    if not candidates:
        return

    # Transform sector-centre positions to display pixels.  This makes crowding
    # detection independent of geographic scale.
    pixel_xy = [
        tuple(ax.transData.transform((item["x"], item["y"])))
        for item in candidates
    ]

    clusters = _cluster_label_indices(
        pixel_xy,
        LABEL_CLUSTER_DISTANCE_PX,
    )

    for cluster in clusters:
        crowded = len(cluster) > 1

        if crowded:
            # Larger clusters get a slightly larger displacement radius.
            radius_points = 28.0 + 8.0 * min(len(cluster), 8)
        else:
            radius_points = 0.0

        # Stable ordering gives reproducible placement.
        cluster = sorted(cluster, key=lambda idx: candidates[idx]["text"])

        for position, idx in enumerate(cluster):
            item = candidates[idx]

            if crowded and MOVE_CROWDED_LABELS:
                angle = (
                    2.0 * math.pi * position / len(cluster)
                    - math.pi / 2.0
                )
                dx = radius_points * math.cos(angle)
                dy = radius_points * math.sin(angle)
            else:
                dx = 0.0
                dy = 0.0

            color = (
                CROWDED_LABEL_COLOR if crowded else NORMAL_LABEL_COLOR
            )
            fontsize = (
                CROWDED_LABEL_FONTSIZE
                if crowded
                else SECTOR_LABEL_FONTSIZE
            )

            bbox = None
            if SECTOR_LABEL_BOX:
                bbox = {
                    "facecolor": "black",
                    "alpha": 0.68 if crowded else 0.55,
                    "edgecolor": color if crowded else "none",
                    "linewidth": 0.7 if crowded else 0.0,
                    "pad": 1.5 if crowded else 1.2,
                }

            arrowprops = None
            if (
                crowded
                and MOVE_CROWDED_LABELS
                and ADD_LABEL_LEADER_LINES
                and (dx != 0.0 or dy != 0.0)
            ):
                arrowprops = {
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.7,
                    "alpha": 0.90,
                    "shrinkA": 2,
                    "shrinkB": 2,
                }

            ax.annotate(
                item["text"],
                xy=(item["x"], item["y"]),
                xycoords="data",
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold" if crowded else "semibold",
                color=color,
                ha="center",
                va="center",
                alpha=0.98,
                zorder=22 if not crowded else 24,
                clip_on=True,
                bbox=bbox,
                arrowprops=arrowprops,
            )


def draw_fir_labels(ax, firs_3857: gpd.GeoDataFrame) -> None:
    """Draw FIR/UIR labels only when their label point is inside the viewport."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for _, row in firs_3857.iterrows():
        point = row.geometry.representative_point()
        if not (xmin <= point.x <= xmax and ymin <= point.y <= ymax):
            continue

        ax.text(
            point.x,
            point.y,
            row["FIR_GROUP"],
            fontsize=9,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=25,
            clip_on=True,
            bbox={
                "facecolor": "black",
                "alpha": 0.62,
                "edgecolor": "white",
                "linewidth": 0.5,
                "pad": 2.5,
            },
        )


def draw_map(
    sectors_4326: gpd.GeoDataFrame,
    firs_4326: gpd.GeoDataFrame,
    acc_4326: gpd.GeoDataFrame,
    output: Path,
    peninsula_only: bool,
) -> None:
    """Draw FIR/UIR and ENAIRE sectors over an Esri satellite basemap."""
    sectors = sectors_4326.to_crs("EPSG:3857")
    firs = firs_4326.to_crs("EPSG:3857")
    acc = (
        acc_4326.to_crs("EPSG:3857")
        if not acc_4326.empty
        else acc_4326
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_8K,
        dpi=FIG_DPI,
    )

    # Set the viewport before requesting imagery.
    if peninsula_only:
        set_bbox(ax, -12.5, 34.0, 6.5, 45.5)
        title = (
            f"Spanish ATC sectorisation — Peninsula and Balearic Islands "
            f"(FL{FL_MIN}–FL{FL_MAX})"
            if FILTER_BY_FL_RANGE
            else "Spanish ATC sectorisation — Peninsula and Balearic Islands"
        )
    else:
        set_extent_from_firs(ax, firs_4326, margin_deg=1.0)
        title = (
            f"Spanish airspace ATC sectorisation — FL{FL_MIN}–FL{FL_MAX}"
            if FILTER_BY_FL_RANGE
            else "Spanish airspace ATC sectorisation"
        )

    # Same satellite provider used in opensky_plot.py.
    add_basemap(ax)

    # ENAIRE sectors: thin dashed cyan with dark underlay.
    sectors.boundary.plot(
        ax=ax,
        color=SECTOR_UNDERLAY_COLOR,
        linewidth=SECTOR_UNDERLAY_LINEWIDTH,
        linestyle=SECTOR_LINESTYLE,
        alpha=0.80,
        zorder=5,
    )
    sectors.boundary.plot(
        ax=ax,
        color=SECTOR_COLOR,
        linewidth=SECTOR_LINEWIDTH,
        linestyle=SECTOR_LINESTYLE,
        alpha=0.95,
        zorder=6,
    )

    # FIR/UIR: thick solid white with a black halo.
    firs.boundary.plot(
        ax=ax,
        color=FIR_UNDERLAY_COLOR,
        linewidth=FIR_UNDERLAY_LINEWIDTH,
        linestyle="-",
        alpha=0.95,
        zorder=9,
    )
    firs.boundary.plot(
        ax=ax,
        color=FIR_COLOR,
        linewidth=FIR_LINEWIDTH,
        linestyle="-",
        alpha=1.0,
        zorder=10,
    )

    # Sevilla and Palma managed/delegated regions: dash-dot, distinct from FIR.
    for acc_name, color in [
        ("Sevilla ACC", SEVILLA_COLOR),
        ("Palma ACC", PALMA_COLOR),
    ]:
        if acc_4326.empty:
            continue

        subset = acc[acc["ACC_GROUP"] == acc_name]
        if subset.empty:
            print(
                f"WARNING: {acc_name} could not be identified from "
                "the downloaded sector attributes"
            )
            continue

        subset.boundary.plot(
            ax=ax,
            color=DELEGATED_UNDERLAY_COLOR,
            linewidth=DELEGATED_UNDERLAY_LINEWIDTH,
            linestyle=DELEGATED_LINESTYLE,
            alpha=0.90,
            zorder=12,
        )
        subset.boundary.plot(
            ax=ax,
            color=color,
            linewidth=DELEGATED_LINEWIDTH,
            linestyle=DELEGATED_LINESTYLE,
            alpha=1.0,
            zorder=13,
        )

    draw_fir_labels(ax, firs)
    draw_sector_labels(ax, sectors)

    handles = [
        Line2D([0], [0], color=FIR_COLOR, lw=FIR_LINEWIDTH,
               linestyle="-", label="FIR/UIR"),
        Line2D([0], [0], color=SECTOR_COLOR, lw=1.4,
               linestyle=SECTOR_LINESTYLE,
               label=(
                   f"ENAIRE sectors intersecting FL{FL_MIN}–FL{FL_MAX}"
                   if FILTER_BY_FL_RANGE
                   else "ENAIRE sectors"
               )),
        Line2D([0], [0], color=SEVILLA_COLOR, lw=DELEGATED_LINEWIDTH,
               linestyle=DELEGATED_LINESTYLE,
               label="Sevilla ACC — managed area"),
        Line2D([0], [0], color=PALMA_COLOR, lw=DELEGATED_LINEWIDTH,
               linestyle=DELEGATED_LINESTYLE,
               label="Palma ACC — managed area"),
    ]

    legend = ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.82,
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
    )
    legend.set_zorder(30)


    ax.set_title(
        title + "\nSatellite basemap with official FIR/UIR and ENAIRE sector boundaries",
        fontsize=14,
        pad=12,
    )


    ax.set_axis_off()

    # Keep an exact 7680 x 4320 canvas. Do not use bbox_inches="tight",
    # because it changes the exported pixel dimensions.
    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=0.93,
    )
    fig.set_size_inches(
        OUTPUT_WIDTH_PX / FIG_DPI,
        OUTPUT_HEIGHT_PX / FIG_DPI,
        forward=True,
    )
    fig.savefig(
        output,
        dpi=FIG_DPI,
        facecolor="black",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    print(
        f"Map saved: {output} "
        f"({OUTPUT_WIDTH_PX} x {OUTPUT_HEIGHT_PX} px)"
    )


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    sectors: gpd.GeoDataFrame,
    firs: gpd.GeoDataFrame,
) -> None:
    """Print a short classification summary before map generation."""
    print("\n=== Detected FIR/UIR ===")
    for fir_name in sorted(firs["FIR_GROUP"].unique()):
        print(f"  - {fir_name}")

    print("\n=== Sectors by inferred ACC ===")
    counts = sectors["ACC_GROUP"].value_counts(dropna=False)
    for acc_name, count in counts.items():
        print(f"  {acc_name:18s}: {count}")

    unclassified = sectors[sectors["ACC_GROUP"] == "Unclassified"]
    if not unclassified.empty:
        identifiers = sorted(
            unclassified["IDENT_TXT"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        print("\nWARNING: unclassified sectors (they are still drawn):")
        print("  " + ", ".join(identifiers[:80]))

        if len(identifiers) > 80:
            print(f"  ... and {len(identifiers) - 80} more")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete ENAIRE download, processing and mapping workflow."""
    print("=== ENAIRE · Spanish ATC sectorisation map ===")
    print("Basemap: Esri World Imagery")
    print(
        f"Output resolution: {OUTPUT_WIDTH_PX} x "
        f"{OUTPUT_HEIGHT_PX} px ({FIG_DPI} dpi)"
    )
    print(
        "Sector labels: "
        + ("enabled" if ADD_SECTOR_NAMES else "disabled")
        + " (compact ENAIRE IDs)"
    )
    print(f"Satellite zoom adjustment: +{BASEMAP_ZOOM_ADJUST}")
    if FILTER_BY_FL_RANGE:
        print(f"Sector vertical filter: FL{FL_MIN}-FL{FL_MAX}")
    print(
        "Crowded labels: "
        f"{CROWDED_LABEL_COLOR}, {CROWDED_LABEL_FONTSIZE} pt, "
        f"cluster threshold {LABEL_CLUSTER_DISTANCE_PX} px"
    )
    print()

    sectors_raw = download_or_load(
        SECTORS_LAYER_URL,
        SECTORS_RAW_FILE,
    )
    airspace_raw = download_or_load(
        AIRSPACE_LAYER_URL,
        AIRSPACE_RAW_FILE,
    )

    sectors = prepare_sectors(sectors_raw)
    firs = prepare_firs(airspace_raw)
    acc_regions = build_acc_regions(sectors)

    sectors.to_file(SECTORS_FILE, driver="GeoJSON")
    firs.to_file(FIRS_FILE, driver="GeoJSON")

    if not acc_regions.empty:
        acc_regions.to_file(ACC_FILE, driver="GeoJSON")

    export_metadata(sectors)
    print_summary(sectors, firs)

    draw_map(
        sectors,
        firs,
        acc_regions,
        MAP_ALL_FILE,
        peninsula_only=False,
    )

    draw_map(
        sectors,
        firs,
        acc_regions,
        MAP_PENINSULA_FILE,
        peninsula_only=True,
    )

    print("\nFinished.")
    print(f"Data:    {DATA_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(
        "\nFORCE_DOWNLOAD is currently False, so future runs reuse the raw "
        "GeoJSON files unless they are missing."
    )


if __name__ == "__main__":
    main()

# Switzerland OpenSky study area: audit and provenance

Reviewed 2026-11-07. This is a research map configuration, not an aeronautical
chart for flight operations.

## Existing architecture

| Concern | Existing implementation | Swiss reuse |
| --- | --- | --- |
| Acquisition/analysis bounds | `bbox_presets.json`; `area_presets.load_area_presets` and `resolve_area_preset` | `SWITZERLAND` is a broad WGS84 acquisition box. `GENEVA_ACC` and `ZURICH_ACC` are separate boxes enclosing current upper-sector geometries. |
| Polygon filtering | `area_presets.filter_dataframe_to_area` | Unchanged. The Swiss acquisition box is **not** passed off as a UIR polygon. |
| Waypoints and ATS corridors | `area_presets.load_waypoints_for_area` and `load_corridors_for_area`; `waypoints.json`, `corridors.json` | The Swiss FRA points are stored under the `SWITZERLAND` key in the shared `waypoints.json`. No unverified Swiss route geometry is added. |
| Cities | `opensky_plot.get_relevant_cities` uses `geonamescache` and population/spacing rules | The Swiss preset supplies eight fixed contextual cities; other regions retain the old automatic selection. |
| Aerodromes | `location_presets.json` supplies centres for live acquisition; the old renderer had **no aerodrome layer** | Locations are grouped by study area. Seven current FOCA ARPs under `SWITZERLAND` feed the video through `airport_codes` and are drawn as small triangles with ICAO labels. Other maps retain their previous appearance. |
| Aircraft, safety colours, altitudes, vectors | `opensky_plot.render_map`; `opensky_safety`; `render_opensky_sector_video.py` | Same code, `ENROUTE` profile and FL labels. |
| Video source/gaps | `render_opensky_sector_video.main`, `read_raw_window`, `track_gaps_and_estimates` | `SWITZERLAND` selects raw input by preset. The gap estimator and conflict rules are unchanged. |

## Aeronautical geography and vertical scope

`SWITZERLAND` downloads and plots **45.0–48.5° N, 5.0–11.5° E**. This is a
deliberately broad acquisition domain covering Liechtenstein and neighboring
French, German, Italian and Austrian airspace; it is not a Swiss state border,
an ATS boundary, the LSAS FIR/UIR or the Switzerland UAC. The name is a study
key, not an airspace classification. This box does not cover the southernmost
part of the current Geneva upper-sector geometry (which reaches 44.8° N).
The current reference line displayed on the map is the WGS84 **LSAS
Switzerland UIR** geometry returned by EUROCONTROL Network Manager's EAD-SDO
airspace layer. Its attributes say **FL195–FL660**, so it covers FL290–FL410.
The separate FOCA LSAS FIR dataset has **GND–FL195**; FOCA's CTA Geneva
(`LSAG`) and CTA Zürich (`LSAZ`) have **2000 ft AGL–FL245**. Thus their
published CTA outlines do not describe the operating sector allocation at
FL290–FL410. EUROCONTROL's Switzerland LSSIP labels the Switzerland UAC above
FL245 and the two ACCs below; its 2024 network plan also identifies upper
sectors at both Geneva and Zürich ACCs. Skyguide states that these centres
work delegated neighboring airspace. The current LSAS UIR polygon contains
Vaduz and other sampled Liechtenstein locations; FOCA describes its airspace
geodata extent as Switzerland, Liechtenstein and adjacent regions. The name
"Switzerland" is therefore not a Swiss-national-border mask.

## Geneva and Zürich ACC study boxes

The two additional presets use the **current EUROCONTROL CFMU_SECTOR** layer's
`LSAGUAC` (Geneva) and `LSAZTOTAL` (Zürich) combined upper-sector features.
All published slices from FL245 upward within each feature have identical
horizontal geometry (checked with geometry equality on 2026-11-07). Their
outward-rounded WGS84 bounding boxes are:

| Preset | Latitude (° N) | Longitude (° E) | Current reference sector |
| --- | --- | --- | --- |
| `GENEVA_ACC` | 44.800000–47.243334 | 5.133333–8.225556 | `LSAGUAC` |
| `ZURICH_ACC` | 46.104444–48.015001 | 6.954444–10.488889 | `LSAZTOTAL` |

These are rectangular **download and render domains**; each preset stores its
reference points inline under `reference_boundaries[].polygon.vertices` in
`bbox_presets.json`. The polygons are visual references and do not filter
aircraft. The CFMU sector layer has
both elementary and combined configurations, which can change with AIRAC and
daily operation. The archived geometry is a current reference, not a verified
ACC responsibility map for 2019 or 2025. Do not assign a historical flight to
an ACC from these boxes alone.

Sources:

- [EUROCONTROL Network Manager airspace feature layer](https://ext.eurocontrol.int/arcgis/rest/services/NM_DCT2_1/DATASOURCES/MapServer/8), `IDENT_TXT=LSAS`, `TYPE_CODE=UIR`, `SOURCE_TXT=EAD-SDO`. Retrieved 2026-11-07; feature last modified **2026-10-26**; `VALIDFROM_DATE`/`VALIDTO_DATE` are null. Coordinates are stored in the `SWITZERLAND` preset (WGS84).
- [EUROCONTROL Network Manager CFMU_SECTOR layer](https://ext.eurocontrol.int/arcgis/rest/services/NM_DCT2_1/DATASOURCES/MapServer/7), `AV_AIRSPACE_ID=LSAGUAC` and `LSAZTOTAL`, retrieved 2026-11-07. Representative FL385–999 features are stored in the corresponding ACC presets; source features for all published upper slices were compared before deriving the bboxes.
- [FOCA FIR geodata](https://api3.geo.admin.ch/rest/services/ech/MapServer/ch.bazl.luftraeume-fluginformationsgebiet) and [FOCA CTA geodata](https://api3.geo.admin.ch/rest/services/ech/MapServer/ch.bazl.luftraeume-kontrollbezirke), queried 2026-11-07; layer features created 2026-03-09. These establish lower-airspace vertical limits, not historical upper-sector boundaries.
- [FOCA airspace dataset description](https://www.bazl.admin.ch/de/luftraeume) gives the coverage as Switzerland, Liechtenstein and adjacent regions; [EUROCONTROL LSSIP Switzerland 2022](https://www.eurocontrol.int/sites/default/files/2023-07/eurocontrol-lssip-2022-switzerland.pdf) gives the airspace organisation figure; [EUROCONTROL network plan 2024](https://www.eurocontrol.int/sites/default/files/2024-04/european-nop-rolling-plan-164.pdf) names LSAG and LSAZ UAC sectors above FL245; [Skyguide ACC description](https://skyguide.ch/jobs/air-traffic-controllers) describes the two centres' operational areas.
- [Skyguide AIM](https://skyguide.ch/services/aeronautical-information-management/) identifies the AIP/AIRAC publications and en-route charts as the authoritative operational publications.

The retrieved UIR and points are **current reference data**. Their exact
validity on any date in **2019 or 2025 is not established** by these services.
Do not use the displayed line to classify historical entry/exit or assert
historical ACC responsibility until the corresponding AIRAC eAIP/AIXM release
for each study date is obtained. The raw bounding-box comparison remains
valid without such a classification.

## Points, routes, cities and airports

The `SWITZERLAND` section of `waypoints.json` contains twelve named points from the
[EUROCONTROL FRA_Points layer](https://ext.eurocontrol.int/arcgis/rest/services/NM_DCT2_1/DATASOURCES/MapServer/5), filtered to `AIRSPACE_NAME_TXT=LSASFRA`, retrieved 2026-11-07 (last modified 2026-10-23). The selection spans western, central, northern, eastern and southern connections and avoids a dense point cloud. Its vertical availability fields include FL290–FL410 for the selected points. Actual use frequency by the 2019/2025 flights has not been measured; FRA eligibility is evidence of planning relevance, not an observed traffic count.

EUROCONTROL's [LSAS FRA connections and FRA point resources](https://www.nm.eurocontrol.int/RAD/) and [Swiss en-route chart](https://skybriefing-zrh.skyguide.ch/documents/10156/489218/ENRC_WEF_14MAY2026_layered.pdf/7f983023-3128-24a6-08b2-5a5b231d4671) provide route context. Published connection examples include **Z50/RESIA**, **Z651/CANNE**, **N871/KORED**, **T721/RILAX**, **T723/NATOR**, **T724/EMKIL** and **L856/RAVED** in the [EUROCONTROL RAD](https://www.nm.eurocontrol.int/RAD/index.html?amdt=show). Route geometry and availability by 2019/2025 AIRAC are not yet verified, so Swiss corridors remain hidden and are not fabricated.

The eight city coordinates came from the installed `geonamescache` **3.0.2**
dataset on 2026-11-07 and are only map context. The seven airport reference
points are WGS84 ARPs from the [FOCA aerodromes/heliports layer](https://api3.geo.admin.ch/rest/services/ech/MapServer/ch.bazl.flugplaetze-heliports), queried by ICAO on 2026-11-07. `LFSB` is recorded there with `location=Frankreich`: Basel-Mulhouse lies in France, despite its Swiss service role. Airport coordinates are current ARPs, not a claim of identical historical locations.

## Implementation and use

- `bbox_presets.json`: independent Swiss box and two upper ACC boxes, each with cities, airport codes and inline current-reference polygon points.
- `location_presets.json`: airport locations grouped by study area, with Swiss ARP provenance; `area_presets.load_location_presets` flattens them for existing code-based selection.
- `area_presets.py`: resolves the inline boundary vertices with the same normalisation as `LECMSAN`; there is no separate airspace-boundary file lookup.
- `waypoints.json`: current FRA points classified under `SWITZERLAND`, alongside the other sector entries.
- `opensky_plot.py`: draws reference boundary and airport markers; Swiss city selection is explicit; existing aircraft and safety drawing is unchanged.
- `render_opensky_sector_video.py`: honors the preset's raw source and loads the shared waypoint catalogue.
- `preview_swiss_area.py`: renders a static positioning check to `results/switzerland_area_preview.png`.

Download one bounded study interval with the same FL290–FL410, airborne and
15-second last-contact filters as LECMSAN:

```bash
python3 src/acquisition/download_opensky_sector.py \
  --sector SWITZERLAND --start "2025-01-19 00:00:00" \
  --end "2025-01-20 00:00:00"

python3 src/conflict_detection/render_opensky_sector_video.py \
  --area SWITZERLAND --start "2025-01-19 00:00:00" \
  --end "2025-01-20 00:00:00" --frame-step 10 --fps 10
```

Repeat with a 2019 interval and the same flags. The source uses the same
10-second grid, interpolation and safety classifier. Neither source period is
downloaded as part of this map configuration change.

To study either ACC domain, replace `SWITZERLAND` in both commands with
`GENEVA_ACC` or `ZURICH_ACC`. The separate archives go under the respective
area keys.

`track_gaps.csv` reports `between_observations` and
`after_last_observation`. The latter means **no later observation in the
loaded window**, not proven surveillance loss. With a filtered bounding-box
archive, an aircraft can leave the box, cross FL290/FL410 or exceed the
last-contact age before the next recorded state. The CSV includes last
position and altitude to audit these cases; resolving the cause needs data
outside the box and flight-level band. The UIR reference line is visual only
and does not change this classification.

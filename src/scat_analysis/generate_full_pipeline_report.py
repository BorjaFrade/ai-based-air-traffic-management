#!/usr/bin/env python3
"""Audit existing SCAT outputs and publish equivalent Spanish/English reports.

Run from any working directory with ``python src/scat_analysis/generate_full_pipeline_report.py``.
This program reads reports, small CSV summaries, Parquet metadata and selected PNGs.
It never imports or executes a SCAT analysis stage.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

try:
    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image,
                                   KeepTogether, LongTable, PageBreak,
                                   PageTemplate, Paragraph, Spacer,
                                   TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents
except ImportError as exc:
    raise SystemExit(f"Required local dependency is missing: {exc}. Install pandas, pyarrow, Pillow and reportlab.")


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/scat"
CODE = ROOT / "src/scat_analysis"
OUT = DATA / "full_pipeline_report"
WEEK = "20161015_20161021"
NOT_FOUND = "NOT FOUND"
BLUE = colors.HexColor("#173852")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EAF2F5")
GRAY = colors.HexColor("#53616B")


def stage08(path: Path) -> bool:
    """Exclude the whole branch before opening or parsing any of its files."""
    return any(re.match(r"08[_-]", p, re.I) or
               re.search(r"(?:^|[_-])stage[_ -]?0?8(?:[_-]|\.|$)", p, re.I)
               for p in path.parts)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


@dataclass
class Evidence:
    stage: str
    purpose: str
    patterns: tuple[str, ...]
    path: Path | None = None
    used: bool = False
    detail: str = ""


@dataclass
class Block:
    kind: str
    es: object
    en: object
    source: str = ""


@dataclass
class Section:
    title_es: str
    title_en: str
    blocks: list[Block] = field(default_factory=list)

    def p(self, es: str, en: str, source: str = "") -> None:
        self.blocks.append(Block("p", es, en, source))

    def table(self, es: list[list[str]], en: list[list[str]], source: str = "") -> None:
        self.blocks.append(Block("table", es, en, source))

    def fig(self, path: Path, es: str, en: str) -> None:
        self.blocks.append(Block("figure", (path, es), (path, en), rel(path)))


class Audit:
    def __init__(self) -> None:
        self.discovered: list[Path] = []
        self.ignored08: list[Path] = []
        self.ignored_other: list[Path] = []
        self.evidence: dict[str, Evidence] = {}
        self.warnings: list[str] = []
        self.figures: list[tuple[str, Path, str]] = []
        self.sections: list[Section] = []
        self.headings: dict[str, list[str]] = {}
        self._discover()

    def _discover(self) -> None:
        for base in (DATA, CODE):
            if not base.exists():
                self.warnings.append(f"Search root {rel(base)}: {NOT_FOUND}")
                continue
            for path in base.rglob("*"):
                if not path.is_file() or OUT in path.parents:
                    continue
                if stage08(path.relative_to(base)) or stage08(path):
                    self.ignored08.append(path)
                elif path.suffix.lower() in {".md", ".csv", ".parquet", ".json", ".png", ".py"}:
                    self.discovered.append(path)
                else:
                    self.ignored_other.append(path)

    def add(self, key: str, stage: str, purpose: str, *patterns: str) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        for priority, pat in enumerate(patterns):
            for base in (DATA, CODE):
                for p in base.glob(pat):
                    if p.is_file() and p in self.discovered and not stage08(p):
                        candidates.append((priority, p))
        path = sorted(candidates, key=lambda item: (item[0], -item[1].stat().st_mtime, str(item[1])))[0][1] if candidates else None
        self.evidence[key] = Evidence(stage, purpose, patterns, path)
        if path is None:
            self.warnings.append(f"{key}: {NOT_FOUND} ({', '.join(patterns)})")
        return path

    def use(self, key: str) -> Path | None:
        item = self.evidence[key]
        if item.path:
            item.used = True
        return item.path

    def md(self, key: str) -> str:
        path = self.use(key)
        if path is None:
            return ""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            self.headings[key] = re.findall(r"^#{1,4}\s+(.+)$", content, re.M)
            return content
        except OSError as exc:
            self.warnings.append(f"Cannot read {rel(path)}: {exc}")
            return ""

    def metric(self, key: str, pattern: str, default: str = NOT_FOUND, flags: int = re.I) -> str:
        content = self.md(key)
        match = re.search(pattern, content, flags)
        if not match:
            self.warnings.append(f"Metric {key}/{pattern}: {NOT_FOUND}")
            return default
        return match.group(1).replace(",", "").strip()

    def csv(self, key: str, columns: list[str] | None = None, max_rows: int = 100) -> pd.DataFrame:
        path = self.use(key)
        if path is None:
            return pd.DataFrame()
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                sample = handle.read(8192)
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            return pd.read_csv(path, sep=delimiter, usecols=columns, nrows=max_rows,
                               low_memory=False)
        except Exception as exc:
            self.warnings.append(f"Cannot read CSV {rel(path)}: {exc}")
            return pd.DataFrame()

    def parquet_metadata(self, key: str) -> str:
        path = self.use(key)
        if path is None:
            return NOT_FOUND
        try:
            pf = pq.ParquetFile(path)
            value = f"{pf.metadata.num_rows:,} rows; {len(pf.schema_arrow.names)} columns; {pf.metadata.num_row_groups} row groups"
            self.evidence[key].detail = value
            return value
        except Exception as exc:
            self.warnings.append(f"Cannot read Parquet metadata {rel(path)}: {exc}")
            return NOT_FOUND

    def figure(self, key: str, stage: str, purpose: str, *patterns: str) -> Path | None:
        path = self.add(key, stage, purpose, *patterns)
        if path:
            self.use(key)
            return path
        return None


def clean_number(value: object, places: int = 3) -> str:
    try:
        x = float(value)
        if pd.isna(x):
            return NOT_FOUND
        return f"{x:,.{places}f}" if not x.is_integer() else f"{int(x):,}"
    except (ValueError, TypeError):
        return str(value) if value is not None else NOT_FOUND


def cell(df: pd.DataFrame, match_col: str, match_val: str, result_col: str, places: int = 3) -> str:
    try:
        row = df.loc[df[match_col].astype(str).str.upper() == match_val.upper()].iloc[0]
        return clean_number(row[result_col], places)
    except (KeyError, IndexError):
        return NOT_FOUND


def phase(a: Audit, title_es: str, title_en: str, facets: dict[str, tuple[str, str]], sources: list[str]) -> Section:
    s = Section(title_es, title_en)
    labels = [
        ("objective", "Objetivo", "Objective"),
        ("question", "Pregunta científica", "Scientific question"),
        ("inputs", "Entradas", "Inputs"),
        ("method", "Método", "Method"),
        ("expected", "Hipótesis o resultado esperado", "Expected result or hypothesis"),
        ("actual", "Resultado observado", "Observed result"),
        ("qa", "Validación", "Quality validation"),
        ("interpretation", "Interpretación", "Interpretation"),
        ("limits", "Limitaciones", "Limitations"),
        ("decision", "Decisión", "Decision"),
    ]
    for key, es_label, en_label in labels:
        es, en = facets[key]
        s.p(f"<b>{es_label}.</b> {es}", f"<b>{en_label}.</b> {en}")
    source_paths = [rel(a.evidence[k].path) if a.evidence[k].path else f"{k}: {NOT_FOUND}" for k in sources]
    s.p("<b>Fuentes.</b> " + "; ".join(source_paths),
        "<b>Sources.</b> " + "; ".join(source_paths))
    return s


def make_content(a: Audit) -> str:
    # Priority is declared explicitly: consolidated reports first; CSVs and focused audits
    # supply exact tabular values. Parquet inputs are inspected through metadata only.
    specs = [
        ("audit", "01", "Structural audit", "01_audit/audit_report.md", "**/audit_report.md"),
        ("sampling", "01", "Sampling inventory", "01_audit/sampling_inventory.csv"),
        ("traffic", "02", "Traffic extraction report", "02_traffic/traffic_report.md"),
        ("states", "02", "Native state Parquet metadata", "02_traffic/scat_states.parquet"),
        ("clearance", "03", "Clearance extraction report", "03_clearances/clearance_report.md"),
        ("clearance_types", "03", "Clearance composition", "03_clearances/clearance_type_counts.csv"),
        ("windows", "04", "Initial windows", "04_windows/window_summary.md"),
        ("primary", "04", "Primary pilot matching", "04_windows/primary_matching_report.md"),
        ("cruise", "04", "Completed cruise rematching V2", "04_windows/cruise_matching_v2_report.md", "PILOT_CRUISE_MATCHING_V2_REPORT.md"),
        ("balance", "04", "Cruise balance V2", "04_windows/cruise_matching_v2_balance.csv"),
        ("feature", "05", "Original feature audit", "05_features/window_feature_physical_report.md"),
        ("featurev2", "05", "V2 physical audit", "05_features/window_features_v2_audit_report.md"),
        ("dictionary", "05", "Feature dictionary", "05_features/feature_dictionary.csv"),
        ("explore", "06", "Completed global exploration", "PILOT_STAGE06_REPORT.md", "06_exploration/exploration_report.md"),
        ("strat", "06", "Altitude stratification", "PILOT_STAGE06_ALTITUDE_STRATIFIED_REPORT.md"),
        ("bands", "06", "Band counts", "06_exploration/altitude_band_inventory.csv"),
        ("model", "07", "Completed intervention model", "PILOT_STAGE07_CRUISE_EXPLAINABILITY_REPORT.md"),
        ("models", "07", "Test model metrics", "07_explainability_cruise/model_family_comparison.csv"),
        ("ablation", "07", "TopSky ablation", "07_explainability_cruise/topsky_ablation.csv"),
        ("cf1", "09", "V1 counterfactual report", "PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md"),
        ("labels1", "09", "V1 label counts", "09_counterfactual_cruise/counterfactual_label_counts.csv"),
        ("cf2", "09", "V2 clearance-aware report", "PILOT_STAGE09_V2_CLEARANCE_AWARE_REPORT.md"),
        ("labels2", "09", "V2 label counts", "09_counterfactual_cruise_v2/counterfactual_label_counts_v2.csv"),
        ("level", "09", "LEVEL analysis", "09_counterfactual_cruise_v2/level_counterfactual_report.md"),
        ("rule", "10", "Static rule comparison", "PILOT_STAGE10_RULE_GEOMETRY_VS_ATC_REPORT.md"),
        ("rules", "10", "Rule positive counts", "10_rule_geometry_comparison_cruise/rule_vs_intervention_tables.csv"),
        ("dynamic", "10B", "Exact dynamic detector report", "PILOT_STAGE10B_EXACT_DYNAMIC_RULE_REPORT.md"),
        ("states10", "10B", "Dynamic colour counts", "10_dynamic_rule_cruise/dynamic_state_counts.csv"),
        ("association", "10B", "Matched dynamic association", "10_dynamic_rule_cruise/dynamic_conditional_association.csv"),
        ("staticdynamic", "10B", "Static/dynamic comparison", "10_dynamic_rule_cruise/static_vs_dynamic_summary.csv"),
        ("testdynamic", "10B", "Frozen test comparison", "10_dynamic_rule_cruise/stage07_test_dynamic_rule_comparison.csv"),
        ("plan", "Current", "Processing plan", "OPENSKY_PROCESSING_PLAN_FROM_SCAT.md"),
    ]
    for key, st, purpose, *patterns in specs:
        a.add(key, st, purpose, *patterns)
    # Parse all selected reports for headings and links. This supports source-priority
    # decisions and figure relevance without copying report prose into the output.
    for key in ("audit", "traffic", "clearance", "windows", "primary", "cruise", "feature", "featurev2", "explore", "strat", "model", "cf1", "cf2", "level", "rule", "dynamic", "plan"):
        a.md(key)
    n = {
        "archives": a.metric("audit", r"Archives selected:\s*\*\*(\d+)"),
        "flights": a.metric("audit", r"Flights scanned:\s*\*\*(\d+)"),
        "plots": a.metric("audit", r"Surveillance plots:\s*\*\*(\d+)"),
        "fpl": a.metric("audit", r"Flight-plan update records:\s*\*\*(\d+)"),
        "clearance_all": a.metric("audit", r"Clearance records:\s*\*\*(\d+)"),
        "clearance_nonempty": a.metric("audit", r"Non-empty clearance records:\s*\*\*(\d+)"),
        "trajectory": a.metric("audit", r"Predicted-trajectory updates:\s*\*\*(\d+)"),
        "invalid": a.metric("audit", r"Invalid JSON files:\s*\*\*(\d+)"),
        "native": a.metric("traffic", r"Native-resolution surveillance rows:\s*\*\*(\d+)"),
        "actionable": a.metric("clearance", r"Actionable clearance/intervention rows:\s*\*\*(\d+)"),
        "valid": a.metric("clearance", r"Valid previous-surveillance matches:\s*\*\*(\d+)"),
        "rejected": a.metric("clearance", r"Rejected by match-age rule:\s*\*\*(\d+)"),
        "max_age": a.metric("clearance", r"Maximum surveillance age:\s*\*\*(\d+\s*s)"),
        "iw": a.metric("windows", r"Intervention windows:\s*\*\*(\d+)"),
        "cw": a.metric("windows", r"Control windows:\s*\*\*(\d+)"),
        "pilotpairs": a.metric("explore", r"pilot contains \*\*(\d[\d,]*) pairs"),
        "v1pairs": a.metric("cruise", r"V1 same-band pairs:\s*\*\*([\d,]+)"),
        "v2pairs": a.metric("cruise", r"V2 pairs:\s*\*\*([\d,]+)"),
        "retention": a.metric("cruise", r"retention:\s*\*\*([\d.]+%)"),
        "smdold": a.metric("cruise", r"V1 cruise anchor ground-speed SMD was \*\*([−-]?[\d.]+)"),
        "v2rows": a.metric("featurev2", r"Eligible pre-event pair rows:\s*\*\*([\d,]+)"),
        "far": a.metric("featurev2", r"Pair rows >30 NM:\s*\*\*([\d,]+)"),
        "tcpa": a.metric("featurev2", r"Positive TCPA <1 s rows:\s*\*\*([\d,]+)"),
        "closing": a.metric("featurev2", r"\|vertical closing\| >15,000 ft/min rows:\s*\*\*([\d,]+)"),
        "silhouette": a.metric("explore", r"GMM silhouette is \*\*([\d.]+)"),
        "noise": a.metric("explore", r"HDBSCAN sample assigned \*\*([\d.]+% noise)"),
        "gate": a.metric("dynamic", r"\*\*(READY_FOR_BOUNDED_OPENSKY_CRUISE_PILOT)\*\*"),
    }
    sampling = a.csv("sampling", ["archive", "interval_median_s"], 1)
    n["sampling"] = clean_number(sampling.iloc[0]["interval_median_s"], 1) if not sampling.empty else NOT_FOUND
    n["parquet"] = a.parquet_metadata("states")
    balance = a.csv("balance")
    for key, col in (("speed", "ground_speed_at_anchor_kt"), ("altitude", "analysis_altitude_ft"), ("density", "local_traffic_density")):
        n[key] = cell(balance, "covariate", col, "SMD")
    bands = a.csv("bands", ["band", "valid_matched_pairs"], 12)
    models = a.csv("models", ["model", "feature_family", "roc_auc", "pr_auc", "balanced_accuracy"], 40)
    ablation = a.csv("ablation")
    labels1 = a.csv("labels1", ["stage09_label", "n", "percent"], 10)
    labels2 = a.csv("labels2", ["stage09_label", "n_events", "percent"], 10)
    rules = a.csv("rules")
    states10 = a.csv("states10")
    assoc = a.csv("association", max_rows=1)
    statdyn = a.csv("staticdynamic", max_rows=1)
    testdyn = a.csv("testdynamic")
    n["aucfull"] = cell(models.loc[models["model"].eq("xgboost")] if not models.empty else models, "feature_family", "SCAT_FULL", "roc_auc") if not models.empty else NOT_FOUND
    n["prfull"] = cell(models.loc[models["model"].eq("xgboost")] if not models.empty else models, "feature_family", "SCAT_FULL", "pr_auc") if not models.empty else NOT_FOUND
    n["aucopen"] = cell(models.loc[models["model"].eq("xgboost")] if not models.empty else models, "feature_family", "OPENSKY_COMPATIBLE", "roc_auc") if not models.empty else NOT_FOUND
    n["propen"] = cell(models.loc[models["model"].eq("xgboost")] if not models.empty else models, "feature_family", "OPENSKY_COMPATIBLE", "pr_auc") if not models.empty else NOT_FOUND
    n["topsky_delta"] = cell(ablation, "model", "xgboost", "delta_roc_auc")
    n["or"] = clean_number(assoc.iloc[0]["matched_conditional_OR"]) if not assoc.empty else NOT_FOUND
    n["ci_low"] = clean_number(assoc.iloc[0]["ci95_lower"]) if not assoc.empty else NOT_FOUND
    n["ci_high"] = clean_number(assoc.iloc[0]["ci95_upper"]) if not assoc.empty else NOT_FOUND
    n["agreement"] = clean_number(100 * float(statdyn.iloc[0]["raw_agreement"]), 2) if not statdyn.empty else NOT_FOUND
    n["kappa"] = clean_number(statdyn.iloc[0]["weighted_kappa_quadratic"]) if not statdyn.empty else NOT_FOUND
    n["static_or"] = clean_number(statdyn.iloc[0]["static_matched_conditional_OR"]) if not statdyn.empty else NOT_FOUND
    n["dynamic_bacc"] = cell(testdyn, "method", "EXACT_DYNAMIC_OPENSKY_RULE", "balanced_accuracy")
    if n["dynamic_bacc"] == NOT_FOUND:
        # Current table names the production rule explicitly; find the exact row.
        if not testdyn.empty and "method" in testdyn:
            rows = testdyn[testdyn["method"].astype(str).str.contains("DYNAMIC", case=False, na=False)]
            if not rows.empty:
                n["dynamic_bacc"] = clean_number(rows.iloc[0]["balanced_accuracy"])

    cover = Section("Portada", "Cover")
    cover.p("UNED · TFM · Air Traffic Management Using Artificial Intelligence Models",
            "UNED · TFM · Air Traffic Management Using Artificial Intelligence Models")
    cover.p(f"Ámbito: SCAT; piloto de la primera semana {WEEK}; dominio principal FL290–FL410. Fecha: {dt.date.today().isoformat()}. Proyecto: {ROOT}.",
            f"Scope: SCAT; first-week pilot {WEEK}; primary FL290–FL410 domain. Date: {dt.date.today().isoformat()}. Project: {ROOT}.")
    a.sections.append(cover)

    s = Section("1. Resumen ejecutivo", "1. Executive summary")
    s.p(f"La auditoría de {n['archives']} archivos SCAT dio paso a {n['native']} estados de vigilancia nativos y {n['actionable']} autorizaciones accionables. El análisis piloto se concentró en {WEEK}; tras corregir el emparejamiento de crucero quedaron {n['v2pairs']} pares intervención/control FL290–FL410. Las variables físicas continuas contienen señal moderada sobre la intervención registrada: XGBoost SCAT_FULL alcanzó ROC-AUC {n['aucfull']} y PR-AUC {n['prfull']} en TEST; la versión compatible con OpenSky obtuvo {n['aucopen']} y {n['propen']}.",
        f"The audit of {n['archives']} SCAT archives led to {n['native']} native surveillance states and {n['actionable']} actionable clearances. The pilot focused on {WEEK}; after cruise rematching, {n['v2pairs']} FL290–FL410 intervention/control pairs remained. Continuous physical features carried moderate information about recorded intervention: SCAT_FULL XGBoost reached TEST ROC-AUC {n['aucfull']} and PR-AUC {n['prfull']}; the OpenSky-compatible version reached {n['aucopen']} and {n['propen']}.")
    s.p(f"La agrupación natural fue débil (silueta GMM {n['silhouette']}); el contrafactual V2 dejó {cell(labels2, 'stage09_label', 'UNKNOWN', 'n_events')} de {n['v2pairs']} eventos como UNKNOWN. La regla dinámica exacta de producción dio una OR condicional emparejada de {n['or']} (IC 95% {n['ci_low']}–{n['ci_high']}) para estados no GREEN. El objetivo observable es ATC_INTERVENTION. Ninguno de estos resultados certifica conflictos evitados o seguridad.",
        f"Natural grouping was weak (GMM silhouette {n['silhouette']}); V2 counterfactual analysis left {cell(labels2, 'stage09_label', 'UNKNOWN', 'n_events')} of {n['v2pairs']} events UNKNOWN. The exact production dynamic rule produced a matched conditional OR of {n['or']} (95% CI {n['ci_low']}–{n['ci_high']}) for non-GREEN states. The observable target is ATC_INTERVENTION. None of these results certifies avoided conflicts or safety.")
    s.p(f"La decisión vigente es {n['gate']}: piloto OpenSky acotado de calidad, rendimiento de variables y desplazamiento de dominio, sin inferir prevalencia poblacional ni autorizar procesamiento masivo.",
        f"The current gate is {n['gate']}: a bounded OpenSky pilot for data quality, feature yield and domain shift, without inferring population prevalence or authorizing full-scale processing.")
    a.sections.append(s)

    s = Section("2. Conjunto de datos y alcance", "2. Dataset and scope")
    s.p("Las fases 01–04 inventariaron o prepararon 13 semanas. Las fases 05–10B analizadas aquí son un piloto de una semana; los resultados de crucero usan exclusivamente FL290–FL410. Los controles se eligieron para comparación observacional y no constituyen un muestreo de prevalencia.",
        "Stages 01–04 inventoried or prepared 13 weeks. Stages 05–10B reviewed here form a one-week pilot; cruise findings use FL290–FL410 only. Controls were chosen for an observational comparison and do not sample population prevalence.")
    s.table([["Magnitud", "Valor"], ["Archivos semanales", n["archives"]], ["Vuelos", n["flights"]], ["Plots de vigilancia", n["plots"]], ["Actualizaciones FPL", n["fpl"]], ["Registros de autorización", n["clearance_all"]], ["Autorizaciones no vacías", n["clearance_nonempty"]], ["Actualizaciones de trayectoria", n["trajectory"]], ["JSON inválidos", n["invalid"]], ["Mediana de muestreo, primera semana (s)", n["sampling"]]],
            [["Measure", "Value"], ["Weekly archives", n["archives"]], ["Flights", n["flights"]], ["Surveillance plots", n["plots"]], ["FPL updates", n["fpl"]], ["Clearance records", n["clearance_all"]], ["Nonempty clearances", n["clearance_nonempty"]], ["Trajectory updates", n["trajectory"]], ["Invalid JSON", n["invalid"]], ["First-week median sampling (s)", n["sampling"]]], "audit, sampling")
    a.sections.append(s)

    s = Section("3. Secuencia metodológica", "3. Pipeline overview")
    s.blocks.append(Block("diagram", [], []))
    s.p("La revisión conserva cada revisión cronológicamente. Las correcciones de las fases 04/05 precedieron al modelado; los resultados 09 y 10 motivaron la comparación dinámica exacta de 10B.",
        "The review preserves the chronological revisions. Stage 04/05 corrections preceded modelling; Stage 09 and 10 findings motivated the exact dynamic comparison in 10B.")
    a.sections.append(s)

    s = phase(a, "4. Fase 01 — Auditoría SCAT", "4. Stage 01 — SCAT audit", {
        "objective": ("Determinar qué información operativa existe antes de diseñar etiquetas o variables.", "Establish what operational information exists before designing labels or features."),
        "question": ("¿Son suficientes y trazables los archivos, plots, planes y autorizaciones?", "Are archives, plots, plans and clearances sufficiently available and traceable?"),
        "inputs": ("Archivos semanales SCAT comprimidos, sin alterarlos.", "Compressed weekly SCAT archives, left unchanged."),
        "method": ("Inventario estructural de rutas JSON, conteos por archivo, campos de autorización y intervalos temporales.", "Structural inventory of JSON paths, per-archive counts, clearance fields and sampling intervals."),
        "expected": ("Confirmar una base utilizable y detectar problemas de esquema antes de inferir tráfico.", "Confirm a usable basis and identify schema issues before reconstructing traffic."),
        "actual": (f"Se revisaron {n['archives']} archivos, {n['flights']} vuelos y {n['plots']} plots; se encontraron {n['invalid']} JSON inválidos. El muestreo mediano de la primera semana fue {n['sampling']} s.", f"The audit covered {n['archives']} archives, {n['flights']} flights and {n['plots']} plots; {n['invalid']} invalid JSON files were found. First-week median sampling was {n['sampling']} s."),
        "qa": ("Se conservaron inventarios de esquema, campos y muestreo; las cifras se contrastan con el informe estructural.", "Schema, field and sampling inventories were retained; counts are checked against the structural report."),
        "interpretation": ("Los registros contienen intervención operativa y predicciones TopSky, pero ninguna etiqueta confirmada de conflicto.", "The records contain operational intervention and TopSky predictions, but no confirmed conflict label."),
        "limits": ("La existencia de un campo no garantiza disponibilidad temporal ni semántica uniforme en todos los vuelos.", "A field's presence does not guarantee temporal availability or uniform meaning across flights."),
        "decision": ("Extraer vigilancia nativa conservando procedencia y tiempo de cada registro.", "Extract native surveillance while retaining provenance and each record's timestamp."),
    }, ["audit", "sampling"])
    a.sections.append(s)

    s = phase(a, "5. Fase 02 — Extracción de tráfico", "5. Stage 02 — Traffic extraction", {
        "objective": ("Construir estados de tráfico nativos y trazables.", "Build traceable native traffic states."),
        "question": ("¿Puede representarse la situación antes de cada autorización sin usar información posterior?", "Can the situation before each clearance be represented without later information?"),
        "inputs": ("Plots de vigilancia, planes de vuelo y trayectorias predichas de SCAT.", "SCAT surveillance plots, flight plans and predicted trajectories."),
        "method": ("Una fila por plot válido, con clave de vuelo, archivo, miembro, índice y marca temporal; contexto unido as-of hacia atrás. Se conservan unidades originales y edades del contexto.", "One row per valid plot with flight key, archive, member, index and timestamp; context is backward as-of joined. Original units and context ages are retained."),
        "expected": ("Reconstrucción temporal utilizable de los estados y contexto operacional.", "Usable temporal reconstruction of states and operational context."),
        "actual": (f"{n['native']} estados nativos y {n['trajectory']} actualizaciones de trayectoria predicha en salida lateral; metadatos Parquet: {n['parquet']}.", f"{n['native']} native states and {n['trajectory']} predicted-trajectory updates in a sidecar; Parquet metadata: {n['parquet']}."),
        "qa": ("Procedencia y edad del contexto explícitas; el informe confirma que no se creó una variable de conflicto.", "Provenance and context age are explicit; the report confirms that no conflict variable was created."),
        "interpretation": ("La resolución nativa permite emparejar autorizaciones con estado previo; no implica continuidad perfecta.", "Native resolution supports matching clearances to earlier states; it does not imply perfect continuity."),
        "limits": ("Huecos de vigilancia y disponibilidad as-of pueden reducir cobertura; la altitud exige auditoría de unidades.", "Surveillance gaps and as-of availability can reduce coverage; altitude requires a unit audit."),
        "decision": ("Extraer autorizaciones accionables y exigir antigüedad máxima del estado previo.", "Extract actionable clearances and enforce a maximum age for the preceding state."),
    }, ["traffic", "states"])
    a.sections.append(s)

    s = phase(a, "6. Fase 03 — Autorizaciones", "6. Stage 03 — Clearances", {
        "objective": ("Definir ATC_INTERVENTION a partir de nuevas órdenes del controlador.", "Define ATC_INTERVENTION from new controller instructions."),
        "question": ("¿Qué órdenes son accionables y qué estado de vigilancia las precedía?", "Which instructions are actionable and what surveillance state preceded them?"),
        "inputs": ("Registros de autorización y estados nativos de la fase 02.", "Clearance records and native Stage 02 states."),
        "method": ("Excluir instantáneas iniciales vacías y repeticiones sin cambio; clasificar LEVEL, HEADING, SPEED y combinaciones. Unir sólo el estado anterior del mismo vuelo y aplicar antigüedad máxima.", "Exclude empty initial snapshots and unchanged repeats; classify LEVEL, HEADING, SPEED and combinations. Join only the same flight's preceding state and apply a maximum age."),
        "expected": ("Eventos observables sin fuga temporal, con tipología y cobertura cuantificadas.", "Observable events without temporal leakage, with quantified type and coverage."),
        "actual": (f"{n['actionable']} eventos accionables; {n['valid']} emparejamientos previos válidos y {n['rejected']} rechazados por antigüedad; umbral {n['max_age']}.", f"{n['actionable']} actionable events; {n['valid']} valid previous-state matches and {n['rejected']} age-rule rejections; threshold {n['max_age']}."),
        "qa": ("Las filas rechazadas se conservan para auditoría; ningún estado futuro entra en el emparejamiento.", "Rejected rows remain available for audit; no future state enters the match."),
        "interpretation": ("CLEARANCE != CONFLICT. Una autorización puede responder a muchas causas operativas.", "CLEARANCE != CONFLICT. A clearance may have many operational causes."),
        "limits": ("La autorización no registra por sí sola motivo, riesgo ni respuesta ejecutada.", "The clearance alone does not record motive, risk or completed response."),
        "decision": ("Construir ventanas intervención/control con el estado anterior válido.", "Construct intervention/control windows from valid preceding states."),
    }, ["clearance", "clearance_types"])
    types = a.csv("clearance_types")
    if not types.empty and len(types.columns) >= 2:
        type_rows = [["Tipo", "Eventos"]] + [[str(row.iloc[0]), clean_number(row.iloc[1], 0)] for _, row in types.head(10).iterrows()]
    else:
        type_rows = [["Tipo", "Eventos"]] + [[t, a.metric("clearance", rf"`{re.escape(t)}`:\s*(\d+)")] for t in ("LEVEL", "HEADING", "SPEED", "LEVEL+SPEED", "HEADING+SPEED", "HEADING+LEVEL", "HEADING+LEVEL+SPEED")]
    s.table(type_rows, [["Type", "Events"]] + type_rows[1:], "clearance_types/clearance")
    a.sections.append(s)

    s = phase(a, "7. Fase 04 — Diseño intervención/control", "7. Stage 04 — Intervention/control design", {
        "objective": ("Comparar cada intervención con una situación de tráfico operacionalmente parecida.", "Compare each intervention with an operationally similar traffic situation."),
        "question": ("¿Pueden construirse controles comparables sin emparejar la geometría que se desea estudiar?", "Can comparable controls be built without matching away the geometry under study?"),
        "inputs": ("Autorizaciones con vigilancia previa y estados de tráfico por semana.", "Clearances with preceding surveillance and traffic states by week."),
        "method": ("Ventanas de diez minutos antes/después; sólo el tramo preevento es predictor elegible. Emparejamiento contextual por semana, centro, fase, banda FL, hora UTC, densidad y, cuando procede, tipo de aeronave. Controles excluidos cerca de autorizaciones y uso limitado por par.", "Ten-minute windows before/after; only the pre-event portion is predictor-eligible. Context matching uses week, centre, phase, FL band, UTC hour, density and, when possible, aircraft type. Controls are excluded near clearances and reuse is constrained per pair."),
        "expected": ("Un diseño observacional con balance contextual y procedencia de pares.", "An observational design with context balance and pair provenance."),
        "actual": (f"Construcción inicial: {n['iw']} ventanas de intervención y {n['cw']} controles emparejados. Cohorte primaria de la primera semana: {n['pilotpairs']} pares; el subgrupo crucero V1 necesitó corrección de velocidad.", f"Initial construction: {n['iw']} intervention windows and {n['cw']} matched controls. First-week primary cohort: {n['pilotpairs']} pairs; the V1 cruise subset needed a speed-balance correction."),
        "qa": ("Informes de balance, nivel de relajación y distancia temporal; la selección piloto usó identificadores exactos de ventana.", "Balance, relaxation-level and temporal-distance reports; pilot selection used exact window identifiers."),
        "interpretation": ("El emparejamiento reduce diferencias observadas de contexto, no confusión no medida ni sesgo causal.", "Matching reduces observed context differences, not unmeasured confounding or causal bias."),
        "limits": ("La construcción inicial priorizó cobertura; el crucero V1 conservó desequilibrio de velocidad.", "The initial construction prioritised coverage; V1 cruise retained a speed imbalance."),
        "decision": ("Auditar variables físicas y estratificar por altitud; después rehacer específicamente el emparejamiento de crucero.", "Audit physical features and stratify by altitude; then rematch cruise specifically."),
    }, ["windows", "primary", "cruise"])
    a.sections.append(s)

    s = phase(a, "8. Fase 05 — Variables de interacción física", "8. Stage 05 — Physical interaction features", {
        "objective": ("Resumir la geometría y dinámica entre el vuelo focal y su tráfico vecino antes del evento.", "Summarise geometry and dynamics between the focal aircraft and nearby traffic before the event."),
        "question": ("¿Qué señales físicas continuas distinguen el entorno de una intervención?", "Which continuous physical signals distinguish an intervention environment?"),
        "inputs": ("Ventanas de fase 04, estados de vigilancia y proxy de separación TopSky.", "Stage 04 windows, surveillance states and the TopSky separation proxy."),
        "method": ("Cálculo por pareja/instante de separación horizontal y vertical, velocidades relativas, cierres, rumbo relativo, ángulo de cruce, CPA/TCPA, densidad y vecinos; agregación por ventana sólo con filas preevento elegibles.", "Pair/time calculations of horizontal and vertical separation, relative speed, closing, relative bearing, crossing angle, CPA/TCPA, density and neighbours; window aggregation uses only eligible pre-event rows."),
        "expected": ("Variables físicamente plausibles y temporalmente admisibles para describir y modelar intervención.", "Physically plausible and temporally admissible features for describing and modelling intervention."),
        "actual": (f"V2 conservó {n['v2rows']} filas pareja preevento; {n['far']} estaban a más de 30 NM, {n['tcpa']} tenían TCPA positivo inferior a 1 s y {n['closing']} superaban 15.000 ft/min de cierre vertical absoluto.", f"V2 retained {n['v2rows']} eligible pre-event pair rows; {n['far']} were beyond 30 NM, {n['tcpa']} had positive TCPA below 1 s, and {n['closing']} exceeded 15,000 ft/min absolute vertical closing."),
        "qa": ("V2 corrigió procedencia de altitud y alineación del ancla; cero discrepancias FL×100/altitud. Añadió diagnósticos limitados a 30 NM y preservó valores originales para auditoría.", "V2 corrected altitude provenance and anchor alignment; zero FL×100/altitude mismatches. It added 30-NM-bounded diagnostics and preserved originals for audit."),
        "interpretation": ("Los extremos globales podían representar vecinos lejanos o mínimo estadístico inestable, no necesariamente encuentros relevantes.", "Global extrema could reflect distant neighbours or unstable minima, not necessarily relevant encounters."),
        "limits": ("Persisten discrepancias de alineación de tasa vertical, ausencia de vecinos cercanos y procedencia temporal incompleta de TopSky; no se eliminaron valores automáticamente.", "Vertical-rate alignment differences, missing nearby neighbours and incomplete TopSky publication timing remain; values were not removed automatically."),
        "decision": ("Usar variables acotadas y verificadas en la exploración; evitar TCPA subsegundo como umbral de conflicto.", "Use bounded, audited features in exploration; avoid sub-second TCPA as a conflict threshold."),
    }, ["feature", "featurev2", "dictionary"])
    s.table([["Familia", "Variables / cautela"], ["Separación", "horizontal, vertical, mínimo a 30 NM"], ["Movimiento", "velocidad relativa, cierre horizontal/vertical"], ["Geometría", "rumbo relativo, ángulo de cruce"], ["Predicción", "CPA, TCPA; extremos y horizonte auditados"], ["Contexto", "densidad, vecinos a 30 NM"], ["Plan", "proxy TopSky; publicación no fechada"]],
            [["Family", "Variables / caveat"], ["Separation", "horizontal, vertical, 30-NM minimum"], ["Motion", "relative speed, horizontal/vertical closing"], ["Geometry", "relative bearing, crossing angle"], ["Prediction", "CPA, TCPA; extrema and horizon audited"], ["Context", "density, neighbours within 30 NM"], ["Plan", "TopSky proxy; unstamped publication"]], "featurev2, dictionary")
    a.sections.append(s)

    s = phase(a, "9. Fase 06 — Exploración y estratificación", "9. Stage 06 — Exploration and stratification", {
        "objective": ("Describir diferencias intervención/control y buscar estructura natural sin imponer clases de conflicto.", "Describe intervention/control differences and explore natural structure without imposing conflict classes."),
        "question": ("¿Existen agrupaciones estables, y cambia la comparación con la altitud?", "Are there stable groups, and does the comparison change with altitude?"),
        "inputs": (f"{n['pilotpairs']} pares primarios de la primera semana y sus variables físicas previas.", f"{n['pilotpairs']} first-week primary pairs and their pre-event physical features."),
        "method": ("Estadística descriptiva, estandarización, PCA, GMM y HDBSCAN; después bandas VERY_LOW, LOW, MID y CRUISE_FL290_410. Sólo las intervenciones entraron en el agrupamiento inicial.", "Descriptive statistics, standardisation, PCA, GMM and HDBSCAN; then VERY_LOW, LOW, MID and CRUISE_FL290_410 bands. Only interventions entered the initial clustering."),
        "expected": ("Determinar si la señal es heterogénea y si alguna banda permite una comparación más defendible.", "Determine whether signals are heterogeneous and whether one band supports a more defensible comparison."),
        "actual": (f"Silueta GMM {n['silhouette']}; HDBSCAN asignó {n['noise']}. Las diferencias descriptivas de separación vertical y TopSky fueron visibles, pero las clases naturales fueron débiles. Crucero V1 aportó {cell(bands, 'band', 'CRUISE_FL290_410', 'valid_matched_pairs', 0)} pares válidos de la banda.", f"GMM silhouette {n['silhouette']}; HDBSCAN assigned {n['noise']}. Vertical separation and TopSky showed descriptive differences, but natural classes were weak. Cruise V1 supplied {cell(bands, 'band', 'CRUISE_FL290_410', 'valid_matched_pairs', 0)} valid within-band pairs."),
        "qa": ("Se verificaron cobertura de ventanas, elegibilidad preevento y estabilidad de agrupamiento; el óptimo GMM estaba en el límite del rango probado.", "Window coverage, pre-event eligibility and clustering stability were checked; the GMM optimum lay at the tested range boundary."),
        "interpretation": ("Los componentes son estados de tráfico descriptivos, no mecanismos ni clases confirmadas de conflicto. FL290–FL410 ofrece un dominio operacional más homogéneo.", "Components are descriptive traffic states, not mechanisms or confirmed conflict classes. FL290–FL410 offers a more homogeneous operational domain."),
        "limits": ("Una semana, elección de hiperparámetros y vecindarios; las bandas bajas presentan otras fases y una mezcla distinta de autorizaciones.", "One week, parameter and neighbour choices; lower bands contain other phases and a different clearance mix."),
        "decision": ("Restringir el análisis explicativo principal a crucero y corregir el balance de velocidad antes de fase 07.", "Restrict the main explanatory analysis to cruise and correct speed balance before Stage 07."),
    }, ["explore", "strat", "bands"])
    band_rows = [["Banda", "Pares válidos"]] + [[str(r["band"]), clean_number(r["valid_matched_pairs"], 0)] for _, r in bands.head(10).iterrows()] if not bands.empty else [["Banda", "Pares válidos"], [NOT_FOUND, NOT_FOUND]]
    s.table(band_rows, [["Band", "Valid pairs"]] + band_rows[1:], "bands")
    a.sections.append(s)

    s = phase(a, "10. Refinamiento — CRUISE_MATCHING_V2", "10. Refinement — CRUISE_MATCHING_V2", {
        "objective": ("Corregir el desequilibrio de velocidad manteniendo suficientes pares de crucero.", "Correct speed imbalance while retaining sufficient cruise pairs."),
        "question": ("¿Mejora el balance contextual sin seleccionar por interacción física?", "Can context balance improve without selection on physical interaction?"),
        "inputs": ("Pares V1 de crucero y controles primarios existentes de la primera semana.", "V1 cruise pairs and existing first-week primary controls."),
        "method": ("Asignación bipartita 1:1 de coste mínimo, calipers y exclusión ±600 s alrededor de autorizaciones; estratos exactos fase, centro y hora UTC.", "Minimum-cost 1:1 bipartite assignment, calipers and ±600-s clearance exclusion; exact phase, centre and UTC-hour strata."),
        "expected": ("Reducir |SMD| de velocidad por debajo de 0,10 conservando una muestra útil.", "Reduce speed |SMD| below 0.10 while retaining a useful sample."),
        "actual": (f"De {n['v1pairs']} pares V1 a {n['v2pairs']} V2 ({n['retention']} de retención); SMD velocidad {n['smdold']} → {n['speed']}, altitud {n['altitude']}, densidad {n['density']}.", f"From {n['v1pairs']} V1 pairs to {n['v2pairs']} V2 pairs ({n['retention']} retention); speed SMD {n['smdold']} to {n['speed']}, altitude {n['altitude']}, density {n['density']}."),
        "qa": ("Balance fase/centro/hora PASS; cada control se usa como máximo una vez y nunca pertenece al mismo vuelo que su intervención.", "Phase/centre/hour balance PASS; each control is used at most once and never belongs to its intervention flight."),
        "interpretation": ("Separación, CPA, TCPA, cierre y rumbo quedaron fuera del emparejamiento porque son variables científicas de interés.", "Separation, CPA, TCPA, closing and bearing stayed outside matching because they are scientific variables of interest."),
        "limits": ("La retención reduce tamaño y el ajuste contextual no identifica causalidad.", "Retention reduces sample size and context adjustment does not identify causality."),
        "decision": ("Congelar esta cohorte para fase 07 y fases posteriores.", "Freeze this cohort for Stage 07 and later stages."),
    }, ["cruise", "balance"])
    s.table([["Covariable", "SMD V1", "SMD V2"], ["Velocidad", n["smdold"], n["speed"]], ["Altitud", "-0.044", n["altitude"]], ["Densidad", "+0.024", n["density"]]],
            [["Covariate", "V1 SMD", "V2 SMD"], ["Speed", n["smdold"], n["speed"]], ["Altitude", "-0.044", n["altitude"]], ["Density", "+0.024", n["density"]]], "cruise, balance")
    a.sections.append(s)

    s = phase(a, "11. Fase 07 — Modelado explicable de intervención", "11. Stage 07 — Explainable intervention modelling", {
        "objective": ("Medir y explicar discriminación de ATC_INTERVENTION en el conjunto emparejado.", "Measure and explain ATC_INTERVENTION discrimination in the matched cohort."),
        "question": ("¿Qué información previa predice intervención y cuánto aporta TopSky?", "Which pre-event information predicts intervention, and how much does TopSky add?"),
        "inputs": (f"{n['v2pairs']} pares V2 y variables acotadas previas al ancla.", f"{n['v2pairs']} V2 pairs and bounded pre-anchor features."),
        "method": ("Partición cronológica con vuelos y pares disjuntos, embargo 600 s por lado; logística condicional, regresión logística, árbol y XGBoost. EBM se omitió por dependencia ausente; se usó importancia por permutación, no SHAP.", "Chronological flight- and pair-disjoint split with 600-s embargo on each side; conditional logistic, logistic regression, tree and XGBoost. EBM was skipped because its dependency was absent; permutation importance, not SHAP, was used."),
        "expected": ("Señal moderada de estado físico y contraste SCAT_FULL frente a variables compatibles con OpenSky.", "Moderate physical-state signal and a SCAT_FULL versus OpenSky-compatible contrast."),
        "actual": (f"TRAIN 2.842 pares, validación 580, TEST 592, excluidos 84. XGBoost SCAT_FULL TEST ROC-AUC {n['aucfull']}, PR-AUC {n['prfull']}; compatible OpenSky {n['aucopen']}/{n['propen']}. Ablación TopSky Δ ROC-AUC {n['topsky_delta']}.", f"TRAIN 2,842 pairs, validation 580, TEST 592, excluded 84. SCAT_FULL XGBoost TEST ROC-AUC {n['aucfull']}, PR-AUC {n['prfull']}; OpenSky-compatible {n['aucopen']}/{n['propen']}. TopSky ablation Δ ROC-AUC {n['topsky_delta']}."),
        "qa": ("Lista previa de variables permitidas; separación de pares/vuelos/tiempo; intervalos bootstrap por grupos conectados. Ninguna variable de autorización ni posterior entró como predictor.", "Predeclared feature allowlist; pair/flight/time separation; connected-group bootstrap intervals. No clearance or post-event field entered as predictor."),
        "interpretation": ("Separación TopSky, separación vertical y cierre aportan discriminación. El 50/50 de la cohorte es artificial; los scores no estiman prevalencia poblacional.", "TopSky separation, vertical separation and closing contribute discrimination. The cohort's 50/50 balance is artificial; scores do not estimate population prevalence."),
        "limits": ("Una semana y banda; el instante de publicación de TopSky no está separado; las métricas no validan traslado a OpenSky.", "One week and band; TopSky publication time is not separately stamped; metrics do not validate transfer to OpenSky."),
        "decision": ("Investigar si puede obtenerse evidencia retrospectiva de interacción sin convertir predicción en conflicto confirmado.", "Investigate whether retrospective interaction evidence can be obtained without treating prediction as confirmed conflict."),
    }, ["model", "models", "ablation"])
    model_rows = [["Modelo", "Variables", "ROC-AUC", "PR-AUC"]]
    if not models.empty:
        for model_name, feature_name in (("logistic", "SCAT_FULL"), ("tree", "SCAT_FULL"), ("xgboost", "SCAT_FULL"), ("xgboost", "OPENSKY_COMPATIBLE")):
            rows = models[(models["model"] == model_name) & (models["feature_family"] == feature_name)]
            if not rows.empty:
                r = rows.iloc[0]
                model_rows.append([model_name, feature_name, clean_number(r["roc_auc"]), clean_number(r["pr_auc"])])
    if len(model_rows) == 1:
        model_rows.append([NOT_FOUND] * 4)
    s.table(model_rows, [["Model", "Features", "ROC-AUC", "PR-AUC"]] + model_rows[1:], "models")
    a.sections.append(s)

    for version, title_es, title_en, labeldf, countcol, key in (("V1", "12. Fase 09 V1 — Contrafactual cinemático", "12. Stage 09 V1 — Kinematic counterfactual", labels1, "n", "cf1"), ("V2", "13. Fase 09 V2 — Contrafactual consciente de autorizaciones", "13. Stage 09 V2 — Clearance-aware counterfactual", labels2, "n_events", "cf2")):
        if version == "V1":
            facets = {
                "objective": ("Explorar si la intervención registrada puede asociarse retrospectivamente con interacción de aeronaves.", "Explore whether recorded intervention can retrospectively be linked to aircraft interaction."),
                "question": ("¿Una continuación cinemática sin nueva orden habría cambiado la geometría de un encuentro?", "Would kinematic continuation without a new instruction have changed encounter geometry?"),
                "inputs": ("Cohorte V2, estados nativos preevento, vecinos y controles TRAIN.", "V2 cohort, pre-event native states, neighbours and TRAIN controls."),
                "method": ("Extrapolación cinemática de velocidad/rumbo/tasa vertical, horizontes primarios 60/120/300 s y sensibilidad 600 s, paso 5 s; radio de candidatos 30 NM, sensibilidad 50 NM; calibración del error con controles.", "Kinematic speed/track/vertical-rate extrapolation, primary 60/120/300-s horizons and 600-s sensitivity, 5-s steps; 30-NM candidate radius, 50-NM sensitivity; error calibrated on controls."),
                "expected": ("Etiquetas prudentes sólo cuando cobertura, respuesta y geometría superasen incertidumbre de modelo.", "Conservative labels only where coverage, response and geometry exceeded model uncertainty."),
                "actual": (f"{cell(labeldf, 'stage09_label', 'LIKELY_INTERACTION_RELATED', countcol, 0)} LIKELY_INTERACTION_RELATED, {cell(labeldf, 'stage09_label', 'LIKELY_NON_INTERACTION_OPERATIONAL', countcol, 0)} LIKELY_NON_INTERACTION_OPERATIONAL y {cell(labeldf, 'stage09_label', 'UNKNOWN', countcol, 0)} UNKNOWN ({cell(labeldf, 'stage09_label', 'UNKNOWN', 'percent', 2)}%).", f"{cell(labeldf, 'stage09_label', 'LIKELY_INTERACTION_RELATED', countcol, 0)} LIKELY_INTERACTION_RELATED, {cell(labeldf, 'stage09_label', 'LIKELY_NON_INTERACTION_OPERATIONAL', countcol, 0)} LIKELY_NON_INTERACTION_OPERATIONAL and {cell(labeldf, 'stage09_label', 'UNKNOWN', countcol, 0)} UNKNOWN ({cell(labeldf, 'stage09_label', 'UNKNOWN', 'percent', 2)}%)."),
                "qa": ("Calibración de errores con controles; censura por autorizaciones posteriores, cobertura de muestras y consistencia de horizontes.", "Control-based error calibration; censoring for later clearances, sample coverage and horizon consistency."),
                "interpretation": ("Las etiquetas son cribado provisional; ni el positivo ni el negativo constituyen verdad de conflicto.", "Labels are provisional screening; neither positive nor negative is conflict ground truth."),
                "limits": ("La continuación genérica omite intención de órdenes LEVEL y la incertidumbre crece con el horizonte.", "Generic continuation omits LEVEL-instruction intent and uncertainty grows with horizon."),
                "decision": ("Probar una alternativa que use la orden previa y la nueva.", "Test an alternative using the previous and new instructions."),
            }
        else:
            facets = {
                "objective": ("Poner a prueba una continuación del mandato anterior frente a la respuesta a la nueva autorización.", "Test continuation of the earlier instruction against response to the new clearance."),
                "question": ("¿Aporta intención de CFL/heading/speed evidencia material que V1 no captaba?", "Does CFL/heading/speed intent add material evidence missed by V1?"),
                "inputs": ("Eventos y candidatos V1 congelados, historia de autorizaciones y estados nativos.", "Frozen V1 events and candidates, clearance history and native states."),
                "method": ("CF-A sigue la orden anterior defendible; CF-B conserva cinemática V1. Compara CFL, heading y speed anteriores/nuevos; LEVEL exige mecanismo vertical y materialidad conjunta a 60/120 s.", "CF-A follows a defensible previous instruction; CF-B retains V1 kinematics. Previous/new CFL, heading and speed are compared; LEVEL requires a vertical mechanism and joint materiality at 60/120 s."),
                "expected": ("Comprobar credibilidad del contrafactual, sin perseguir más positivos ni menos UNKNOWN.", "Test counterfactual credibility without seeking more positives or fewer UNKNOWN cases."),
                "actual": (f"{cell(labeldf, 'stage09_label', 'LIKELY_INTERACTION_RELATED', countcol, 0)} positivo provisional, {cell(labeldf, 'stage09_label', 'LIKELY_NON_INTERACTION_OPERATIONAL', countcol, 0)} no interacción probable y {cell(labeldf, 'stage09_label', 'UNKNOWN', countcol, 0)} UNKNOWN ({cell(labeldf, 'stage09_label', 'UNKNOWN', 'percent', 2)}%). Hubo 3.583 LEVEL; 2.141 órdenes previas frescas y 1.806 modelos LEVEL evaluables. CF-A y CF-B no divergieron materialmente en 60/120 s.", f"{cell(labeldf, 'stage09_label', 'LIKELY_INTERACTION_RELATED', countcol, 0)} provisional positive, {cell(labeldf, 'stage09_label', 'LIKELY_NON_INTERACTION_OPERATIONAL', countcol, 0)} likely non-interaction and {cell(labeldf, 'stage09_label', 'UNKNOWN', countcol, 0)} UNKNOWN ({cell(labeldf, 'stage09_label', 'UNKNOWN', 'percent', 2)}%). There were 3,583 LEVEL events; 2,141 fresh previous instructions and 1,806 evaluable LEVEL models. CF-A and CF-B did not materially diverge at 60/120 s."),
                "qa": ("Historial enlazado y versionado; calibración TRAIN, validación de respuesta, misma pareja en ambos horizontes y razones UNKNOWN explícitas.", "Joined and versioned history; TRAIN calibration, response checks, same partner at both horizons and explicit UNKNOWN reasons."),
                "interpretation": ("La nueva intención no produjo una verdad retrospectiva fiable de conflicto; UNKNOWN aumentó de V1 a V2.", "The added intent did not produce reliable retrospective conflict truth; UNKNOWN increased from V1 to V2."),
                "limits": ("Órdenes antiguas o ausentes, respuesta incompleta, censura y modelos de vuelo aproximados.", "Old or missing instructions, incomplete response, censoring and approximate flight models."),
                "decision": ("INSUFFICIENT_COUNTERFACTUAL_EVIDENCE. No usar etiquetas 09 como verdad de entrenamiento o validación de conflicto.", "INSUFFICIENT_COUNTERFACTUAL_EVIDENCE. Do not use Stage 09 labels as conflict training or validation truth."),
            }
        s = phase(a, title_es, title_en, facets, [key, "labels1" if version == "V1" else "labels2"] + (["level"] if version == "V2" else []))
        a.sections.append(s)
    s = Section("13.1 Comparación V1/V2", "13.1 V1/V2 comparison")
    label_rows = [["Etiqueta", "V1", "V2"]]
    for lab in ("LIKELY_INTERACTION_RELATED", "LIKELY_NON_INTERACTION_OPERATIONAL", "UNKNOWN"):
        label_rows.append([lab, cell(labels1, "stage09_label", lab, "n", 0), cell(labels2, "stage09_label", lab, "n_events", 0)])
    s.table(label_rows, label_rows, "labels1, labels2")
    s.p("El cambio de 316 etiquetas y 3.782 persistencias describe una revisión del criterio, no un nuevo resultado causal. El incremento de UNKNOWN refleja mayor rigor de identificación.",
        "The 316 changed labels and 3,782 unchanged labels describe a revision of criteria, not a new causal result. The increase in UNKNOWN reflects stricter identification.")
    a.sections.append(s)

    s = phase(a, "14. Fase 10 — Geometría de reglas frente a intervención", "14. Stage 10 — Rule geometry versus intervention", {
        "objective": ("Comparar regiones geométricas clásicas con ATC_INTERVENTION, sin tratar 09 como verdad.", "Compare classical geometric regions with ATC_INTERVENTION without treating Stage 09 as truth."),
        "question": ("¿Cuántas intervenciones caen dentro de umbrales fijos y qué información añade la geometría continua?", "How many interventions fall within fixed thresholds, and what does continuous geometry add?"),
        "inputs": ("Cohorte V2, geometría preevento y perfil ENROUTE estático aproximado.", "V2 cohort, pre-event geometry and an approximate static ENROUTE profile."),
        "method": ("Tabular 5 NM/1000 ft, 7,5 NM/1200 ft, predicción geométrica, proxy estático y modelos de geometría continua, usando intervención observada como comparación.", "Tabulate 5 NM/1000 ft, 7.5 NM/1200 ft, predicted geometry, static proxy and continuous-geometry models against observed intervention."),
        "expected": ("Medir solapamiento y límites de las reglas, no sensibilidad para conflictos reales.", "Measure rule overlap and limitations, not sensitivity for actual conflicts."),
        "actual": (f"5 NM/1000 ft: {cell(rules, 'rule', '5NM_1000FT', 'intervention_rule_positive', 0)} intervenciones y {cell(rules, 'rule', '5NM_1000FT', 'control_rule_positive', 0)} controles positivos. 7,5 NM/1200 ft: {cell(rules, 'rule', '7_5NM_1200FT', 'intervention_rule_positive', 0)} y {cell(rules, 'rule', '7_5NM_1200FT', 'control_rule_positive', 0)}. Proxy estático: {cell(rules, 'rule', 'OPENSKY_STATIC_PROFILE_PROXY', 'intervention_rule_positive', 0)} y {cell(rules, 'rule', 'OPENSKY_STATIC_PROFILE_PROXY', 'control_rule_positive', 0)}.", f"5 NM/1000 ft: {cell(rules, 'rule', '5NM_1000FT', 'intervention_rule_positive', 0)} intervention and {cell(rules, 'rule', '5NM_1000FT', 'control_rule_positive', 0)} control positives. 7.5 NM/1200 ft: {cell(rules, 'rule', '7_5NM_1200FT', 'intervention_rule_positive', 0)} and {cell(rules, 'rule', '7_5NM_1200FT', 'control_rule_positive', 0)}. Static proxy: {cell(rules, 'rule', 'OPENSKY_STATIC_PROFILE_PROXY', 'intervention_rule_positive', 0)} and {cell(rules, 'rule', 'OPENSKY_STATIC_PROFILE_PROXY', 'control_rule_positive', 0)}."),
        "qa": ("Conteos positivo/negativo/desconocido separados; evaluación TEST congelada y sensibilidad de umbral.", "Positive/negative/unknown counts are separate; frozen TEST evaluation and threshold sensitivity were recorded."),
        "interpretation": ("La mayor parte de las intervenciones ocurre fuera de regiones estrictas; una regla positiva no prueba conflicto y una negativa no prueba seguridad.", "Most interventions occur outside strict regions; a positive rule does not prove conflict and a negative rule does not prove safety."),
        "limits": ("El proxy estático no ejecuta el pronóstico ni el separador activo del detector de producción.", "The static proxy does not execute the production detector's forecast or active-separator logic."),
        "decision": ("REVISE_RULE_COMPARISON: ejecutar el detector dinámico exacto antes de sacar conclusiones sobre reglas OpenSky.", "REVISE_RULE_COMPARISON: run the exact dynamic detector before drawing conclusions about OpenSky rules."),
    }, ["rule", "rules"])
    rule_rows = [["Regla", "Intervención +", "Control +"]] + [[str(r["rule"]), clean_number(r["intervention_rule_positive"], 0), clean_number(r["control_rule_positive"], 0)] for _, r in rules.head(8).iterrows()] if not rules.empty else [["Regla", "Intervención +", "Control +"], [NOT_FOUND] * 3]
    s.table(rule_rows, [["Rule", "Intervention +", "Control +"]] + rule_rows[1:], "rules")
    a.sections.append(s)

    s = phase(a, "15. Fase 10B — Detector dinámico OpenSky exacto", "15. Stage 10B — Exact dynamic OpenSky detector", {
        "objective": ("Comparar los estados del detector real de producción con intervención registrada.", "Compare actual production detector states with recorded intervention."),
        "question": ("¿Sigue la severidad dinámica OpenSky a ATC_INTERVENTION en pares emparejados?", "Does dynamic OpenSky severity track ATC_INTERVENTION in matched pairs?"),
        "inputs": ("Cohorte V2 y vigilancia SCAT nativa reconstruida as-of en cada ancla.", "V2 cohort and native SCAT surveillance reconstructed as-of at each anchor."),
        "method": ("Reutilizar sin cambios detect_opensky_conflicts.analyse_snapshot_fast, perfil ENROUTE, pronóstico cada 5 s hasta 5 min y lógica de separador activo. Instantánea causal con edad máxima 15 s; sensibilidad de 10 s.", "Reuse detect_opensky_conflicts.analyse_snapshot_fast unchanged, ENROUTE profile, 5-s forecast through 5 min and active-separator logic. Causal as-of snapshot with 15-s maximum age; 10-s sensitivity."),
        "expected": ("Medir relación real entre color dinámico e intervención y comprobar si el proxy estático era equivalente.", "Measure the real relation between dynamic colour and intervention and test whether the static proxy was equivalent."),
        "actual": (f"8.195/8.196 ventanas evaluables. OR condicional de no GREEN {n['or']} (IC 95% {n['ci_low']}–{n['ci_high']}); OR del proxy estático {n['static_or']}. Concordancia cruda {n['agreement']}%, kappa ponderada {n['kappa']}; el signo de asociación se invirtió. TEST balanced accuracy de regla dinámica {n['dynamic_bacc']}.", f"8,195/8,196 windows evaluable. Non-GREEN conditional OR {n['or']} (95% CI {n['ci_low']}–{n['ci_high']}); static-proxy OR {n['static_or']}. Raw agreement {n['agreement']}%, weighted kappa {n['kappa']}; association direction reversed. Dynamic-rule TEST balanced accuracy {n['dynamic_bacc']}."),
        "qa": ("Mapeo de unidades SCAT, replay determinista de casos, equivalencia de llamada directa, cobertura de snapshots y 100% de acuerdo de color con sensibilidad de 10 s.", "SCAT unit mapping, deterministic case replay, direct-call equivalence, snapshot coverage and 100% colour agreement under 10-s sensitivity."),
        "interpretation": ("La severidad dinámica no sigue positivamente la intervención en esta muestra. Estado de seguridad basado en reglas e intervención ATC son conceptos operativos distintos; el resultado no implica defecto del detector.", "Dynamic severity does not positively track intervention in this sample. Rule-based safety state and ATC intervention are distinct operational concepts; the result does not imply a detector defect."),
        "limits": ("Reconstrucción as-of SCAT distinta de la rejilla OpenSky; altitud barométrica proxy, muestra de una semana y celdas no GREEN pequeñas. RED no confirma conflicto; GREEN no certifica seguridad.", "SCAT as-of reconstruction differs from OpenSky's grid; barometric-altitude proxy, one-week sample and small non-GREEN cells. RED does not confirm conflict; GREEN does not certify safety."),
        "decision": (f"{n['gate']}: piloto OpenSky acotado de calidad y rendimiento de variables en FL290–FL410.", f"{n['gate']}: bounded FL290–FL410 OpenSky data-quality and feature-yield pilot."),
    }, ["dynamic", "states10", "association", "staticdynamic", "testdynamic"])
    state_rows = [["Estado", "Intervención", "Control"]] + [[str(r["dynamic_state"]), clean_number(r["n_intervention"], 0), clean_number(r["n_control"], 0)] for _, r in states10.head(10).iterrows()] if not states10.empty else [["Estado", "Intervención", "Control"], [NOT_FOUND] * 3]
    s.table(state_rows, [["State", "Intervention", "Control"]] + state_rows[1:], "states10")
    a.sections.append(s)

    s = Section("16. Síntesis y cronología científica", "16. Cross-stage synthesis and scientific timeline")
    s.p("El objetivo inicial era caracterizar estados asociados con la intervención. La agrupación no aportó clases naturales robustas; la revisión de altitud y emparejamiento hizo defendible una comparación de crucero. El modelo supervisado encontró señal moderada en variables continuas, reforzada por TopSky. La vía contrafactual no identificó conflictos evitados con suficiente certeza. Los umbrales fijos cubrieron pocas intervenciones y el detector dinámico exacto mostró una relación inversa débil en esta muestra. Así, intervención y estado reglado de seguridad deben estudiarse como constructos relacionados pero diferentes.",
        "The initial goal was to characterise states associated with intervention. Clustering did not yield robust natural classes; altitude review and rematching supported a defensible cruise comparison. Supervised modelling found moderate signal in continuous features, strengthened by TopSky. The counterfactual route did not identify avoided conflicts with sufficient confidence. Fixed thresholds covered few interventions, and the exact dynamic detector showed a weak inverse relation in this sample. Intervention and rule-based safety state should therefore be studied as related but distinct constructs.")
    timeline = [
        ["01", "Inventario SCAT", f"{n['archives']} archivos", "Extraer tráfico", "SCAT inventory", f"{n['archives']} archives", "Extract traffic"],
        ["03", "Intervención observable", f"{n['actionable']} autorizaciones", "Diseñar controles", "Observable intervention", f"{n['actionable']} clearances", "Design controls"],
        ["04/05", "Comparabilidad y física", f"{n['v2pairs']} pares V2; auditoría 30 NM", "Congelar crucero", "Comparability and physics", f"{n['v2pairs']} V2 pairs; 30-NM audit", "Freeze cruise"],
        ["06", "Clases naturales", f"Silueta {n['silhouette']}", "No imponer clases", "Natural classes", f"Silhouette {n['silhouette']}", "Do not force classes"],
        ["07", "Predicción de intervención", f"AUC {n['aucfull']}", "Explorar contrafactual", "Intervention prediction", f"AUC {n['aucfull']}", "Explore counterfactual"],
        ["09", "Verdad contrafactual", f"V2 UNKNOWN {cell(labels2, 'stage09_label', 'UNKNOWN', 'percent', 1)}%", "No usar como verdad", "Counterfactual truth", f"V2 UNKNOWN {cell(labels2, 'stage09_label', 'UNKNOWN', 'percent', 1)}%", "Do not use as truth"],
        ["10/10B", "Reglas vs intervención", f"OR dinámica {n['or']}", "Piloto acotado", "Rules vs intervention", f"Dynamic OR {n['or']}", "Bounded pilot"],
    ]
    s.table([["Fase", "Pregunta", "Resultado", "Decisión"]] + [[r[0], r[1], r[2], r[3]] for r in timeline],
            [["Stage", "Question", "Result", "Decision"]] + [[r[0], r[4], r[5], r[6]] for r in timeline])
    a.sections.append(s)

    s = Section("17. Limitaciones y reglas de consistencia", "17. Limitations and consistency rules")
    s.p("CLEARANCE != CONFLICT; ATC_INTERVENTION != CONFIRMED_CONFLICT; RULE_POSITIVE != CONFIRMED_CONFLICT; RULE_NEGATIVE != SAFE; RED != CONFIRMED_CONFLICT; GREEN != SAFE. Las etiquetas de fase 09 no son ground truth y la prevalencia 50/50 del diseño emparejado no es prevalencia poblacional.",
        "CLEARANCE != CONFLICT; ATC_INTERVENTION != CONFIRMED_CONFLICT; RULE_POSITIVE != CONFIRMED_CONFLICT; RULE_NEGATIVE != SAFE; RED != CONFIRMED_CONFLICT; GREEN != SAFE. Stage 09 labels are not ground truth, and the matched design's 50/50 prevalence is not population prevalence.")
    s.p("Persisten confusión operacional no medida, procedencia temporal TopSky incompleta, diferencias SCAT/OpenSky, una sola semana y banda, eventos no independientes por tráfico compartido y falta de verdad externa de conflicto. Los intervalos emparejados no representan incertidumbre entre semanas. Ninguna etapa estimó efecto causal de autorización ni calibración poblacional.",
        "Unmeasured operational confounding, incomplete TopSky publication timing, SCAT/OpenSky differences, one week and band, dependence through shared traffic and lack of external conflict truth remain. Matched intervals do not represent between-week uncertainty. No stage estimated a causal clearance effect or population calibration.")
    a.sections.append(s)

    s = Section("18. Conclusiones", "18. Conclusions")
    s.p(f"SCAT permite estudiar de forma trazable la intervención ATC registrada. Se construyó una cohorte de crucero emparejada de {n['v2pairs']} pares y la geometría continua ofreció discriminación moderada (AUC {n['aucfull']}). TopSky añadió {n['topsky_delta']} de ROC-AUC al XGBoost en esta muestra. La agrupación fue débil, y el contrafactual V2 dejó {cell(labels2, 'stage09_label', 'UNKNOWN', 'percent', 2)}% UNKNOWN. Las reglas fijas abarcaron una fracción pequeña de las intervenciones; la lógica dinámica de producción no mostró asociación positiva con ellas. La evidencia describe intervenciones, no conflictos confirmados ni seguridad certificada.",
        f"SCAT supports a traceable study of recorded ATC intervention. A {n['v2pairs']}-pair matched cruise cohort was built, and continuous geometry gave moderate discrimination (AUC {n['aucfull']}). TopSky added {n['topsky_delta']} ROC-AUC to XGBoost in this sample. Clustering was weak, and V2 counterfactual analysis left {cell(labels2, 'stage09_label', 'UNKNOWN', 'percent', 2)}% UNKNOWN. Fixed rules covered a small fraction of interventions; production dynamic logic showed no positive association with them. The evidence describes interventions, not confirmed conflicts or certified safety.")
    a.sections.append(s)

    s = Section("19. Siguiente paso", "19. Next step")
    s.p(f"Estado verificado: {n['gate']}. Ejecutar únicamente un piloto OpenSky FL290–FL410 acotado para medir calidad de estados, rendimiento de variables, cobertura de vecinos, sincronización y desplazamiento de dominio respecto a SCAT; revisar la interpretabilidad antes de ampliar escala. Esta recomendación no autoriza procesamiento histórico completo ni entrenamiento automático de un modelo de conflictos.",
        f"Verified gate: {n['gate']}. Run only a bounded FL290–FL410 OpenSky pilot to measure state quality, feature yield, neighbour coverage, synchronisation and domain shift relative to SCAT; review interpretability before expanding scale. This recommendation does not authorise full historical processing or automatic conflict-model training.")
    a.sections.append(s)

    s = Section("20. Apéndice — Inventario de evidencia", "20. Appendix — Evidence inventory")
    s.p("La tabla muestra los artefactos científicos principales realmente utilizados. El CSV adjunto conserva rutas completas, estado de uso y ausencias. Los Parquet extensos sólo se consultan por metadatos.",
        "The table shows the main scientific artifacts actually used. The companion CSV retains full paths, usage and missing items. Large Parquet files are inspected through metadata only.")
    a.sections.append(s)

    # Selected images: scientific relevance is fixed by stage/keyword preference,
    # then boosted when a final report links to the image. No blanket PNG inclusion.
    figure_specs = [
        ("fig_band", "06", "Altitude-band sample sizes", "06_exploration/figures/altitude_bands/*band_sample_sizes.png", "Distribución de pares por banda de altitud; motiva el dominio principal de crucero.", "Matched pairs by altitude band; motivates the primary cruise domain."),
        ("fig_cluster", "06", "Global versus cruise clustering", "06_exploration/figures/altitude_bands/*global_vs_cruise_clustering_quality.png", "Calidad del agrupamiento global y de crucero; muestra la debilidad de clases naturales.", "Global and cruise clustering quality; shows weak evidence for natural classes."),
        ("fig_separation", "06", "Bandwise horizontal separation", "06_exploration/figures/altitude_bands/*horizontal_separation_by_band.png", "Separación horizontal por banda; ilustra heterogeneidad operacional.", "Horizontal separation by band; illustrates operational heterogeneity."),
        ("fig_tree", "07", "Interpretable Stage07 tree", "07_explainability_cruise/tree/scat_full/tree.png", "Árbol poco profundo del modelo SCAT_FULL; muestra reglas interpretables de intervención, no de conflicto.", "Shallow SCAT_FULL tree; shows interpretable intervention rules, not conflict rules."),
        ("fig_rule", "10", "Horizontal separation comparison", "10_rule_geometry_comparison_cruise/figures/*horizontal_separation.png", "Separación horizontal y umbrales fijos; explica por qué una sola regla cubre pocas intervenciones.", "Horizontal separation and fixed thresholds; explains why one rule covers few interventions."),
        ("fig_dynamic", "10B", "Dynamic state distribution", "10_dynamic_rule_cruise/figures/dynamic_state_distribution.png", "Estados dinámicos por intervención/control; el predominio GREEN contextualiza la asociación.", "Dynamic states by intervention/control; the GREEN majority contextualises association."),
        ("fig_confusion", "10B", "Static versus dynamic confusion", "10_dynamic_rule_cruise/figures/static_vs_dynamic_confusion.png", "Proxy estático frente a detector dinámico; la concordancia bruta oculta cambios importantes entre clases.", "Static proxy versus dynamic detector; raw agreement masks important cross-class changes."),
    ]
    stage_to_section = {"06": 9, "07": 11, "10": 15, "10B": 16}
    for key, st, purpose, pattern, es, en in figure_specs:
        p = a.figure(key, st, purpose, pattern)
        if not p:
            continue
        report_key = {"06": "strat", "07": "model", "10": "rule", "10B": "dynamic"}[st]
        report_text = a.md(report_key)
        linked = p.name in report_text
        if not linked and st not in {"07", "10"}:
            a.warnings.append(f"Figure selected by stage/keyword, not explicit final-report link: {rel(p)}")
        a.figures.append((st, p, "report-link" if linked else "stage-keyword"))
        a.sections[stage_to_section[st]].fig(p, es, en)

    return n["gate"]


def font_setup() -> str:
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("SCATArial", str(regular)))
        pdfmetrics.registerFont(TTFont("SCATArial-Bold", str(bold)))
        pdfmetrics.registerFontFamily("SCATArial", normal="SCATArial", bold="SCATArial-Bold")
        return "SCATArial"
    return "Helvetica"


class PipelineDiagram(Flowable):
    def __init__(self, lang: str, font: str):
        super().__init__()
        self.lang, self.font = lang, font
        self.width, self.height = 470, 440

    def draw(self) -> None:
        labels = (["Archivo SCAT", "01 Auditoría", "02 Vigilancia", "03 Autorizaciones", "04 Pares intervención/control", "05 Variables físicas", "06 Exploración", "07 Modelo explicable", "09 Contrafactual", "10/10B Reglas y detector", "Piloto OpenSky acotado"] if self.lang == "es" else
                  ["SCAT archive", "01 Audit", "02 Surveillance", "03 Clearances", "04 Matched intervention/control", "05 Physical features", "06 Exploration", "07 Explainable model", "09 Counterfactual", "10/10B Rules and detector", "Bounded OpenSky pilot"])
        c = self.canv
        box_h, gap, x, w = 30, 9, 35, 400
        c.setFont(self.font, 9)
        for i, label in enumerate(labels):
            y = self.height - 35 - i * (box_h + gap)
            c.setFillColor(PALE if i < len(labels)-1 else colors.HexColor("#D2EDEB"))
            c.setStrokeColor(TEAL if i == len(labels)-1 else colors.HexColor("#A9C3CD"))
            c.roundRect(x, y, w, box_h, 5, stroke=1, fill=1)
            c.setFillColor(BLUE)
            c.drawCentredString(x + w/2, y + 10, label)
            if i < len(labels)-1:
                c.setStrokeColor(TEAL)
                c.line(x+w/2, y-1, x+w/2, y-gap+1)


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, font: str, lang: str):
        super().__init__(filename, pagesize=A4, leftMargin=48, rightMargin=48,
                         topMargin=62, bottomMargin=58, title="SCAT pipeline report")
        self.font, self.lang = font, lang
        self.addPageTemplates(PageTemplate(id="normal", frames=[Frame(48, 58, A4[0]-96, A4[1]-120, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decor))

    def decor(self, canv, doc) -> None:
        canv.saveState()
        canv.setStrokeColor(colors.HexColor("#B7CCD3"))
        canv.line(48, A4[1]-44, A4[0]-48, A4[1]-44)
        canv.setFont(self.font, 7.6)
        canv.setFillColor(GRAY)
        canv.drawString(48, A4[1]-35, "UNED · TFM · SCAT")
        canv.drawRightString(A4[0]-48, 37, f"{doc.page}")
        canv.drawString(48, 37, "ATC_INTERVENTION · FL290–FL410")
        canv.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "heading":
            text = flowable.getPlainText()
            key = f"section-{self.seq.nextf('section')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, False)
            self.notify("TOCEntry", (0, text, self.page, key))


def styles(font: str) -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("title", fontName=font, fontSize=21, leading=27, textColor=BLUE, alignment=TA_CENTER, spaceAfter=16),
        "subtitle": ParagraphStyle("subtitle", fontName=font, fontSize=12, leading=16, textColor=TEAL, alignment=TA_CENTER, spaceAfter=28),
        "heading": ParagraphStyle("heading", fontName=font, fontSize=14, leading=19, textColor=BLUE, spaceBefore=12, spaceAfter=9, keepWithNext=True),
        "body": ParagraphStyle("body", fontName=font, fontSize=9.3, leading=14, textColor=colors.HexColor("#26333D"), spaceAfter=8),
        "small": ParagraphStyle("small", fontName=font, fontSize=7.2, leading=10, textColor=GRAY, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=7.5, leading=10, textColor=colors.HexColor("#25343E")),
        "cellhead": ParagraphStyle("cellhead", fontName=font, fontSize=7.6, leading=10, textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName=font, fontSize=8.1, leading=11, textColor=GRAY, spaceBefore=5, spaceAfter=12),
    }


def safe_markup(value: str) -> str:
    # Content strings are owned by this script; escape only field/source paths.
    return value.replace("&", "&amp;")


def table_flow(rows: list[list[str]], sty: dict, available: float) -> LongTable:
    width = max(len(row) for row in rows)
    fractions = ([0.67, 0.33] if width == 2 else [0.36, 0.32, 0.32] if width == 3 else [0.25, 0.32, 0.22, 0.21] if width == 4 else [0.07, 0.32, 0.10, 0.41, 0.10])
    if width not in (2, 3, 4, 5):
        fractions = [1/width] * width
    data = []
    for i, row in enumerate(rows[:40]):
        vals = list(row) + [""] * (width-len(row))
        data.append([Paragraph(escape(str(v)), sty["cellhead" if i == 0 else "cell"]) for v in vals])
    t = LongTable(data, colWidths=[available*f for f in fractions], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#C4D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def inventory_rows(a: Audit, lang: str, for_pdf: bool = False) -> list[list[str]]:
    head = ["Fase", "Archivo", "Tipo", "Finalidad", "Uso"] if lang == "es" else ["Stage", "File", "Type", "Purpose", "Use"]
    rows = [head]
    keys = list(a.evidence)
    if for_pdf:
        keys = [key for key in ("audit", "sampling", "traffic", "states", "clearance", "clearance_types", "windows", "cruise", "balance", "feature", "featurev2", "explore", "strat", "bands", "model", "models", "ablation", "cf1", "labels1", "cf2", "labels2", "rule", "rules", "dynamic", "states10", "association", "staticdynamic") if key in a.evidence]
    for key in keys:
        e = a.evidence[key]
        if not e.used:
            continue
        name = e.path.name if e.path else NOT_FOUND
        rows.append([e.stage, name, e.path.suffix.lower().lstrip(".") if e.path else "-", e.purpose, ("Sí" if lang == "es" else "Yes")])
    return rows


def write_markdown(a: Audit, lang: str, target: Path) -> None:
    title = "Análisis del Pipeline SCAT para la Caracterización de Intervenciones ATC" if lang == "es" else "SCAT Pipeline Analysis for the Characterization of ATC Interventions"
    subtitle = "Auditoría metodológica y resultados del piloto FL290–FL410" if lang == "es" else "Methodological Audit and FL290–FL410 Pilot Results"
    lines = [f"# {title}", "", f"*{subtitle}*", "", "## Contenido" if lang == "es" else "## Contents", ""]
    lines += [f"- {s.title_es if lang == 'es' else s.title_en}" for s in a.sections[1:]]
    lines += [""]
    for index, sec in enumerate(a.sections):
        if index:
            lines += [f"## {sec.title_es if lang == 'es' else sec.title_en}", ""]
        for block in sec.blocks:
            value = block.es if lang == "es" else block.en
            if block.kind == "p":
                text = re.sub(r"</?b>", "**", str(value))
                lines += [text, ""]
            elif block.kind == "table":
                rows = value
                if rows:
                    lines += ["| " + " | ".join(str(x) for x in rows[0]) + " |",
                              "| " + " | ".join("---" for _ in rows[0]) + " |"]
                    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows[1:40]]
                    lines += [""]
            elif block.kind == "figure":
                path, caption = value
                lines += [f"![{caption}]({rel(path)})", "", f"*{caption}*", ""]
            elif block.kind == "diagram":
                labels = (["Archivo SCAT", "01 Auditoría", "02 Vigilancia", "03 Autorizaciones", "04 Pares", "05 Variables físicas", "06 Exploración", "07 Modelo explicable", "09 Contrafactual", "10/10B Reglas", "Piloto OpenSky acotado"] if lang == "es" else
                          ["SCAT archive", "01 Audit", "02 Surveillance", "03 Clearances", "04 Matched pairs", "05 Physical features", "06 Exploration", "07 Explainable model", "09 Counterfactual", "10/10B Rules", "Bounded OpenSky pilot"])
                lines += ["```text", " → ".join(labels), "```", ""]
    lines += ["### " + ("Archivos principales" if lang == "es" else "Main files"), ""]
    rows = inventory_rows(a, lang)
    lines += ["| " + " | ".join(rows[0]) + " |", "| --- | --- | --- | --- | --- |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    lines += [""]
    target.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(a: Audit, lang: str, target: Path, font: str) -> int:
    sty = styles(font)
    title = "Análisis del Pipeline SCAT para la Caracterización de Intervenciones ATC" if lang == "es" else "SCAT Pipeline Analysis for the Characterization of ATC Interventions"
    subtitle = "Auditoría metodológica y resultados del piloto FL290–FL410" if lang == "es" else "Methodological Audit and FL290–FL410 Pilot Results"
    story: list[Flowable] = [Spacer(1, 120), Paragraph(escape(title), sty["title"]), Paragraph(escape(subtitle), sty["subtitle"])]
    for block in a.sections[0].blocks:
        story.append(Paragraph(safe_markup(block.es if lang == "es" else block.en), sty["body"]))
    story += [PageBreak(), Paragraph("Índice" if lang == "es" else "Contents", sty["heading"])]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc", fontName=font, fontSize=9, leading=14, leftIndent=12, firstLineIndent=-12, textColor=BLUE, spaceBefore=4)]
    story += [toc, PageBreak()]
    available = A4[0] - 96
    for sec in a.sections[1:]:
        story.append(Paragraph(escape(sec.title_es if lang == "es" else sec.title_en), sty["heading"]))
        for block in sec.blocks:
            value = block.es if lang == "es" else block.en
            if block.kind == "p":
                paragraph = str(value)
                if paragraph.startswith("<b>Fuentes.</b>") or paragraph.startswith("<b>Sources.</b>"):
                    story.append(Paragraph(safe_markup(paragraph), sty["small"]))
                else:
                    story.append(Paragraph(safe_markup(paragraph), sty["body"]))
            elif block.kind == "table":
                story += [Spacer(1, 4), table_flow(value, sty, available), Spacer(1, 11)]
            elif block.kind == "figure":
                path, caption = value
                if path.exists():
                    with PILImage.open(path) as im:
                        w, h = im.size
                    scale = min(available/w, 275/h)
                    image = Image(str(path), width=w*scale, height=h*scale)
                    image.hAlign = "CENTER"
                    story.append(KeepTogether([image, Paragraph(escape(caption), sty["caption"])]))
            elif block.kind == "diagram":
                story += [Spacer(1, 10), PipelineDiagram(lang, font), Spacer(1, 12)]
    story += [Spacer(1, 5), table_flow(inventory_rows(a, lang, for_pdf=True), sty, available)]
    doc = ReportDoc(str(target), font, lang)
    doc.multiBuild(story)
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"PDF absent or empty: {target}")
    # Each page dictionary is marked /Type /Page; /Pages does not match the boundary.
    pages = len(re.findall(rb"/Type\s*/Page\b", target.read_bytes()))
    if pages == 0:
        raise RuntimeError(f"PDF has no pages: {target}")
    return pages


def validate(a: Audit, paths: dict[str, Path], page_counts: dict[str, int]) -> None:
    for lang in ("ES", "EN"):
        content = paths[f"md_{lang}"].read_text(encoding="utf-8")
        if re.search(r"(?:^|\n)##\s*(?:\d+\.\s*)?(?:Stage|Fase)\s*0?8\b", content, re.I):
            raise RuntimeError(f"Stage 08 section found in {lang} report")
        for sec in a.sections[1:]:
            title = sec.title_es if lang == "ES" else sec.title_en
            if title not in content or not sec.blocks:
                raise RuntimeError(f"Empty mandatory section: {title}")
        if re.search(r"\b(?:TODO|XXXXX|NaN)\b", content):
            a.warnings.append(f"Unresolved placeholder-like text in {lang} Markdown; inspect context")
        if page_counts[lang] <= 0:
            raise RuntimeError(f"No pages in {lang} PDF")
    for _, path, _ in a.figures:
        if not path.exists():
            raise RuntimeError(f"Selected figure disappeared: {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = Audit()
    gate = make_content(a)
    font = font_setup()
    paths = {
        "pdf_ES": OUT / "TFM_SCAT_PIPELINE_REPORT_ES.pdf",
        "pdf_EN": OUT / "TFM_SCAT_PIPELINE_REPORT_EN.pdf",
        "md_ES": OUT / "TFM_SCAT_PIPELINE_REPORT_ES.md",
        "md_EN": OUT / "TFM_SCAT_PIPELINE_REPORT_EN.md",
    }
    for lang in ("ES", "EN"):
        write_markdown(a, lang.lower(), paths[f"md_{lang}"])
    pages = {lang: write_pdf(a, lang.lower(), paths[f"pdf_{lang}"], font) for lang in ("ES", "EN")}
    validate(a, paths, pages)
    with (OUT / "evidence_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "key", "file", "type", "purpose", "used_in_report", "metadata"])
        for key, e in a.evidence.items():
            writer.writerow([e.stage, key, rel(e.path) if e.path else NOT_FOUND, e.path.suffix.lstrip(".") if e.path else "", e.purpose, "YES" if e.used else "NO", e.detail])
    with (OUT / "selected_figures_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "file", "selection_reason"])
        writer.writerows((st, rel(path), reason) for st, path, reason in a.figures)
    missing = [key for key, e in a.evidence.items() if e.path is None]
    log = [f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}", f"Project: {ROOT}",
           f"Discovered eligible files: {len(a.discovered)}", f"Evidence used: {sum(e.used for e in a.evidence.values())}",
           f"Other files ignored: {len(a.ignored_other)}", f"Stage08 files ignored: {len(a.ignored08)}",
           "", "DISCOVERED FILES:"]
    log.extend(rel(p) for p in a.discovered)
    log += ["", "USED FILES:"]
    log.extend(rel(e.path) for e in a.evidence.values() if e.used and e.path)
    log += ["", "STAGE08 FILES IGNORED:"]
    log.extend(rel(p) for p in a.ignored08)
    log += ["", "MISSING EXPECTED ARTIFACTS:"]
    log.extend(missing or ["none"])
    log += ["", "FIGURES INCLUDED:"]
    log.extend(rel(p) for _, p, _ in a.figures)
    log += ["", "WARNINGS:"]
    log.extend(a.warnings or ["none"])
    log += ["", "PDF PATHS:", f"ES: {paths['pdf_ES']} ({pages['ES']} pages; {paths['pdf_ES'].stat().st_size} bytes)",
            f"EN: {paths['pdf_EN']} ({pages['EN']} pages; {paths['pdf_EN'].stat().st_size} bytes)"]
    (OUT / "report_generation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("=" * 60)
    print("SCAT FULL PIPELINE REPORT")
    print("=" * 60)
    print(f"Project: {ROOT}")
    print("Stages reviewed: 01, 02, 03, 04, 05, 06, 07, 09, 10/10B")
    print("Stage 08: EXCLUDED")
    print("Revisions: CRUISE_MATCHING_V2; Stage05 V2; Stage06 altitude stratification; Stage09 V1/V2; Stage10/10B")
    for lang in ("ES", "EN"):
        path = paths[f"pdf_{lang}"]
        print(f"{lang} PDF: {path} | {pages[lang]} pages | {path.stat().st_size} bytes")
    print(f"Evidence files used: {sum(e.used for e in a.evidence.values())}")
    print(f"Figures included: {len(a.figures)}")
    print(f"Missing expected evidence: {', '.join(missing) if missing else 'none'}")
    print(f"Warnings: {len(a.warnings)} (see report_generation_log.txt)")
    print(f"Latest scientific gate: {gate}")
    print("=" * 60)


if __name__ == "__main__":
    main()

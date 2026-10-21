#!/usr/bin/env python3
"""Extract native-resolution SCAT surveillance states into Parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("02_extract_scat_traffic.py requires pyarrow for Parquet output") from exc

from common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    archive_paths,
    extract_traffic_rows,
    finite_number,
    iter_flights,
    update_read_me_first,
)


NUMERIC_COLUMNS = {
    "time_epoch",
    "latitude",
    "longitude",
    "altitude_ft",
    "flight_level",
    "selected_altitude_ft",
    "final_selected_altitude_ft",
    "ground_speed_ms",
    "ground_speed_kt",
    "track_deg",
    "heading_deg",
    "vertical_rate_fpm",
    "vertical_rate_ms",
    "fpl_copn_pel",
    "fpl_copx_pel",
    "clearance_cfl",
    "clearance_heading_deg",
    "clearance_speed",
    "context_age_seconds",
    "base_context_age_seconds",
    "plan_context_age_seconds",
    "clearance_context_age_seconds",
    "arrival_context_age_seconds",
    "departure_context_age_seconds",
    "holding_context_age_seconds",
    "centre_context_age_seconds",
    "prediction_context_age_seconds",
}


def normalise_state_row(row: dict) -> dict:
    result = dict(row)
    for key in NUMERIC_COLUMNS:
        result[key] = finite_number(result.get(key))
    for key, value in list(result.items()):
        if key in NUMERIC_COLUMNS or key in {"plot_index", "on_ground"}:
            continue
        if value is None:
            result[key] = None
        elif not isinstance(value, str):
            result[key] = str(value)
    return result


def normalise_prediction_row(row: dict) -> dict:
    result = dict(row)
    result["prediction_index"] = int(result["prediction_index"])
    result["prediction_epoch"] = finite_number(result.get("prediction_epoch"))
    for key, value in list(result.items()):
        if key in {"prediction_index", "prediction_epoch"}:
            continue
        if value is None:
            result[key] = None
        elif not isinstance(value, str):
            result[key] = str(value)
    return result


class ParquetSink:
    def __init__(self, path: Path, batch_size: int = 5_000, kind: str = "state"):
        self.path = path
        self.batch_size = batch_size
        self.kind = kind
        self.rows: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows_written = 0

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        if self.writer is None:
            sample = self.rows[0]
            fields = []
            for key in sample:
                if self.kind == "prediction":
                    arrow_type = pa.int64() if key == "prediction_index" else pa.float64() if key == "prediction_epoch" else pa.string()
                else:
                    arrow_type = pa.int64() if key == "plot_index" else pa.float64() if key in NUMERIC_COLUMNS else pa.bool_() if key == "on_ground" else pa.string()
                fields.append(pa.field(key, arrow_type))
            self.schema = pa.schema(fields)
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows_written += table.num_rows
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        elif not self.path.exists():
            empty = pa.Table.from_pandas(pd.DataFrame(), preserve_index=False)
            pq.write_table(empty, self.path, compression="zstd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "02_traffic")
    parser.add_argument("--weeks", nargs="*", help="Optional ZIP names or week tokens")
    parser.add_argument("--max-flights-per-archive", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--include-raw-plot-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    states_path = args.output_root / "scat_states.parquet"
    prediction_path = args.output_root / "scat_predicted_trajectories.parquet"
    config_path = args.output_root / "run_config.json"
    if not args.overwrite and (states_path.exists() or prediction_path.exists()):
        raise SystemExit(
            f"Output already exists under {args.output_root}; use --overwrite to replace it."
        )
    for path in (states_path, prediction_path, config_path):
        if path.exists() and args.overwrite:
            path.unlink()

    archives = archive_paths(args.input_root, args.weeks)
    state_sink = ParquetSink(states_path, args.batch_size)
    prediction_sink = ParquetSink(prediction_path, args.batch_size, kind="prediction")
    archive_rows = []
    try:
        for archive_index, archive in enumerate(archives, start=1):
            print(f"[{archive_index}/{len(archives)}] Extracting {archive.name}", flush=True)
            stats = {
                "archive": archive.name,
                "flights_seen": 0,
                "flights_with_plots": 0,
                "plots_written": 0,
                "prediction_updates_written": 0,
            }
            for current_archive, member, flight in iter_flights(
                [archive], args.max_flights_per_archive
            ):
                stats["flights_seen"] += 1
                rows, predictions = extract_traffic_rows(
                    current_archive,
                    member,
                    flight,
                    include_raw_plot_json=args.include_raw_plot_json,
                )
                if rows:
                    stats["flights_with_plots"] += 1
                for row in rows:
                    state_sink.add(normalise_state_row(row))
                for row in predictions:
                    prediction_sink.add(normalise_prediction_row(row))
                stats["plots_written"] += len(rows)
                stats["prediction_updates_written"] += len(predictions)
            archive_rows.append(stats)
            print(
                f"  flights={stats['flights_seen']} plots={stats['plots_written']} "
                f"predictions={stats['prediction_updates_written']}",
                flush=True,
            )
    finally:
        state_sink.close()
        prediction_sink.close()

    config = {
        "dataset": "SCAT",
        "input_root": str(args.input_root),
        "archives": [path.name for path in archives],
        "max_flights_per_archive": args.max_flights_per_archive,
        "native_surveillance_resolution": True,
        "include_raw_plot_json": args.include_raw_plot_json,
        "states_path": str(states_path),
        "predictions_path": str(prediction_path),
        "rows_written": state_sink.rows_written,
        "prediction_rows_written": prediction_sink.rows_written,
        "archive_stats": archive_rows,
        "target_semantics": "No conflict label is created; this is traffic extraction only.",
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(archive_rows).to_csv(args.output_root / "archive_extraction_stats.csv", index=False)
    report = [
        "# SCAT native-resolution traffic extraction",
        "",
        "The output has one row per valid surveillance plot. The raw ZIP archives were not modified or extracted.",
        "",
        f"- Archives: **{len(archives)}**",
        f"- Native-resolution surveillance rows: **{state_sink.rows_written}**",
        f"- Predicted-trajectory updates in sidecar: **{prediction_sink.rows_written}**",
        f"- States: `{states_path}`",
        f"- Predictions: `{prediction_path}`",
        "",
        "Each state retains `source_archive`, `source_member`, `plot_index`, `flight_key` and the source plot timestamp.",
        "The attached flight-plan, clearance and prediction fields are the latest records available at or before the plot time; missing as-of records remain empty.",
        "Context age fields record the age of each source record and `context_age_seconds` is the oldest available context used for the row.",
        "No field named `conflict` is created and no GREEN/YELLOW/ORANGE/RED rule is applied.",
    ]
    (args.output_root / "traffic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Traffic extraction complete: {args.output_root}")


if __name__ == "__main__":
    main()


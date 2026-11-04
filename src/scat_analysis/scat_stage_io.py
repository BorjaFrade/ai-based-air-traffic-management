#!/usr/bin/env python3
"""Utilidades compartidas para fases 20-21 de auditoría SCAT.

Los nombres de columnas NO se dan por supuestos: --inspect identifica el esquema.
Las marcas de tiempo se convierten a nanosegundos UTC solo cuando contienen
fecha absoluta o epoch reconocible. Una hora relativa se rechaza.
"""
from __future__ import annotations
import glob
import json
import re
import sys
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

ALIASES = {
    'pairs': {
        'time': ['obs_time', 'observation_time', 'snapshot_time', 'time_utc', 'timestamp_utc', 't_utc', 'timestamp', 'time_stamp', 'time', 't', 'snapshot_ts', 'epoch_s'],
        'flight1': ['flight_key_a', 'flight_key_1', 'flight_1', 'flight1', 'flight_a', 'a_flight_key', 'flight_id_a', 'flight_id_1', 'acid1', 'id1', 'aircraft_1', 'aircraft_a', 'a_id', 'flight_i', 'id_a'],
        'flight2': ['flight_key_b', 'flight_key_2', 'flight_2', 'flight2', 'flight_b', 'b_flight_key', 'flight_id_b', 'flight_id_2', 'acid2', 'id2', 'aircraft_2', 'aircraft_b', 'b_id', 'flight_j', 'id_b'],
        'color': ['color', 'colour', 'risk_color', 'risk_colour', 'risk_level', 'state', 'label_color', 'color_4', 'color_name'],
        'mmin': ['m_min', 'mmin', 'm_min_route', 'min_m', 'min_margin', 'predicted_m_min', 'm_min_pred', 'm_min_20m'],
        'tau': ['tau', 'tau_min', 'tau_minutes', 'time_to_min', 'tcpa_min', 't_min', 'time_to_min_minutes', 'tau_s', 'tau_sec'],
        'existing_label': ['stage16_label_300', 'has_clearance', 'has_order', 'order_5m', 'any_order_next5m', 'with_order', 'clearance_5m'],
    },
    'clearances': {
        'time': ['clearance_epoch', 'time_stamp', 'clearance_time', 'event_time', 'timestamp_utc', 'time_utc', 'timestamp', 'time', 't'],
        'flight': ['flight_key', 'flight_id', 'flight', 'aircraft_id', 'acid', 'callsign_key'],
        'kind': ['clearance_type', 'type', 'order_type', 'action_type', 'kind', 'intervention_type'],
        'eventid': ['clearance_id', 'event_id', 'intervention_id', 'id'],
    },
    'tracks': {
        'time': ['time_stamp', 'timestamp', 'time_utc', 'timestamp_utc', 'time', 'plot_time', 't'],
        'flight': ['flight_key', 'flight_id', 'flight', 'aircraft_id', 'acid'],
        'lat': ['lat', 'latitude', 'lat_deg', 'latitude_deg', 'plot_lat'],
        'lon': ['lon', 'longitude', 'long', 'lon_deg', 'longitude_deg', 'plot_lon'],
        'alt': ['alt_ft', 'altitude_ft', 'baro_alt_ft', 'pressure_altitude_ft', 'altitude', 'alt', 'flight_level', 'fl'],
    },
}


def input_files(spec: str) -> list[Path]:
    p = Path(spec).expanduser()
    if p.exists() and p.is_file():
        return [p]
    if p.exists() and p.is_dir():
        paths = sorted(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in {'.csv', '.parquet', '.gz'})
    else:
        paths = sorted(Path(x) for x in glob.glob(spec, recursive=True) if Path(x).is_file())
    paths = [x for x in paths if x.name.endswith(('.csv', '.csv.gz', '.parquet'))]
    if not paths:
        raise FileNotFoundError(f'No se encontraron CSV/CSV.GZ/Parquet en: {spec}')
    types = {('parquet' if x.suffix == '.parquet' else 'csv') for x in paths}
    if len(types)>1:
        raise ValueError(f'La ruta mezcla CSV y Parquet ({len(paths)} archivos). Especifica un glob más preciso.')
    return paths


def header(files: list[Path]) -> list[str]:
    p = files[0]
    if p.suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit('Falta pyarrow: python3 -m pip install pyarrow') from e
        return pq.ParquetFile(p).schema_arrow.names
    return list(pd.read_csv(p, nrows=0).columns)


def load_mapping(path: str | None) -> dict:
    return json.loads(Path(path).read_text(encoding='utf8')) if path else {}


def resolve(kind: str, columns: list[str], overrides: dict, required: list[str]) -> dict:
    lowered = {str(c).lower():c for c in columns}
    found = {}
    for key, aliases in ALIASES[kind].items():
        requested = overrides.get(kind,{}).get(key)
        if requested:
            if requested not in columns:
                raise ValueError(f'La columna {requested!r} solicitada para {kind}.{key} no existe. Disponibles: {columns}')
            found[key]=requested
            continue
        opts=[x for x in aliases if x.lower() in lowered]
        if opts:
            found[key]=lowered[opts[0].lower()]
    missing=[x for x in required if x not in found]
    if missing:
        raise ValueError(f'No puedo reconocer columnas {missing} en {kind}. Columnas disponibles: {columns}. Usa --inspect y --mapping MAP.json.')
    return found


def describe(path: str, kind: str, overrides: dict):
    fs=input_files(path)
    cols=header(fs)
    found=resolve(kind, cols, overrides, []);
    print(f'\n[{kind}] {len(fs)} archivos; primero={fs[0]}')
    print('  columnas:',', '.join(cols))
    print('  reconocidas:', json.dumps(found, ensure_ascii=False))
    print('  muestra:')
    for rec in next(read_batches(fs, min(3, 3), cols), pd.DataFrame()).head(3).to_dict(orient='records'):
        print('  ', str(rec)[:600])
    return found


def read_batches(files: list[Path], batch: int = 100000, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    if files[0].suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit('Parquet requiere pyarrow: python3 -m pip install pyarrow') from e
        for p in files:
            pf=pq.ParquetFile(p)
            cols=columns or pf.schema_arrow.names
            for b in pf.iter_batches(batch_size=batch, columns=cols):
                yield b.to_pandas()
    else:
        for p in files:
            for chunk in pd.read_csv(p, usecols=columns, chunksize=batch, low_memory=False):
                yield chunk


def to_ns(values: pd.Series, name: str) -> np.ndarray:
    """epoch seconds/millis/micros/nanos o ISO con zona UTC, no tiempo relativo."""
    s=values
    if pd.api.types.is_numeric_dtype(s):
        nums=pd.to_numeric(s, errors='coerce')
        v=nums.dropna()
        if v.empty:
            return np.full(len(s), np.iinfo(np.int64).min, dtype='int64')
        med=float(v.abs().median())
        if med > 1e17: unit='ns'
        elif med > 1e14: unit='us'
        elif med > 1e11: unit='ms'
        elif med > 1e8: unit='s'
        else:
            raise ValueError(f'{name}: tiempos numéricos relativos ({med:g}), sin fecha absoluta. Se necesita columna UTC/epoch.')
        dt=pd.to_datetime(nums,unit=unit,utc=True,errors='coerce')
    else:
        raw=s.astype('string').str.strip()
        try:
            dt=pd.to_datetime(raw, utc=True, errors='coerce',format='mixed')
        except TypeError:
            dt=pd.to_datetime(raw, utc=True, errors='coerce')
        valid=raw.notna().sum()
        if valid and dt.notna().sum() < valid*0.8:
            n=pd.to_numeric(raw,errors='coerce')
            if n.notna().sum()>valid*.8:
                return to_ns(n,name)
    # Pandas may preserve a seconds-resolution dtype for integral epoch values.
    # Force nanoseconds before converting, so integer pair timestamps and
    # fractional clearance timestamps share the same unit.
    return dt.dt.as_unit('ns').astype('int64').to_numpy()


def ns_iso(ns: int) -> str:
    if ns == np.iinfo(np.int64).min or ns < 0:
        return ''
    return pd.Timestamp(int(ns),unit='ns',tz='UTC').isoformat()


def flight_norm(s: pd.Series) -> pd.Series:
    # Se exige el mismo flight_key en ambas tablas; nunca se hace join por callsign salvo que sea explícito.
    return s.astype('string').str.strip().fillna('')


def save_json(path: Path, data: dict):
    def norm(x):
        if isinstance(x, (np.integer,)):return int(x)
        if isinstance(x, (np.floating,)):return float(x)
        if isinstance(x, Path):return str(x)
        raise TypeError(str(type(x)))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=norm),encoding='utf8')


def mapping_template(path: Path):
    template={
      'pairs': {'time':'CAMBIAR_fecha_UTC','flight1':'CAMBIAR_id_avion_A','flight2':'CAMBIAR_id_avion_B','color':'CAMBIAR_color','mmin':'CAMBIAR_mmin','tau':'CAMBIAR_tau'},
      'clearances': {'time':'CAMBIAR_fecha_UTC','flight':'flight_key','kind':'clearance_type'},
      'tracks': {'time':'time_stamp','flight':'flight_key','lat':'latitude','lon':'longitude','alt':'altitude_ft'},
    }
    save_json(path,template)

# AI-Based Air Traffic Management

Master's Thesis at the **[Universidad Nacional de Educación a Distancia (UNED)](https://www.uned.es/)** focused on the application of Artificial Intelligence (AI) models to Air Traffic Management (ATM).

The project investigates the use of historical and real-world aircraft trajectory data to analyse traffic situations, detect potential conflicts, study aircraft separation, and develop data-driven methods for supporting tactical air traffic conflict detection and resolution.

## 🎓 Master's Thesis

**Title:** Air Traffic Management Using Artificial Intelligence Models  
**Degree:** Master's Degree  
**University:** [Universidad Nacional de Educación a Distancia (UNED)](https://www.uned.es/)  
**Author:** [Borja Rodríguez Frade](https://orcid.org/0000-0002-9097-1323) - [ORCID: 0000-0002-9097-1323](https://orcid.org/0000-0002-9097-1323)  
**Supervisors:**  
- [Félix Hernández del Olmo](https://orcid.org/0000-0002-0567-9572) — [ORCID: 0000-0002-0567-9572](https://orcid.org/0000-0002-0567-9572)  
- [Elena Gaudioso Vázquez](https://orcid.org/0000-0003-2258-6623) — [ORCID: 0000-0003-2258-6623](https://orcid.org/0000-0003-2258-6623)  

**Status:** Work in progress  
**Academic Year:** 2026–2027
## ✈️ Project Overview

The main objective of this project is to investigate how Artificial Intelligence and data-driven models can support Air Traffic Management, particularly in scenarios involving multiple aircraft trajectories and potential loss of separation.

The project combines:

- historical aircraft trajectory data;
- geospatial air traffic analysis;
- aircraft state-vector processing;
- horizontal and vertical separation analysis;
- conflict detection;
- Closest Point of Approach (CPA) analysis;
- trajectory prediction;
- machine-learning methods for conflict detection and resolution;
- visualisation of air traffic scenarios.

The work is intended exclusively for **academic and research purposes** and is not designed for operational or safety-critical Air Traffic Control applications.

## 🛰️ Data Sources

The project may use data obtained from different aviation data providers, including:

- OpenSky Network;
- EUROCONTROL;
- Flightradar24;
- other publicly available or research-access aviation datasets.

For the historical part of the thesis, the repository also uses OpenSky's
[Weekly 24 Hours of State Vector Data 2017–2022](https://opensky-network.org/data/scientific#d1)
dataset. It contains complete Monday state-vector data, sampled at 10-second
update intervals and distributed as hourly files. The download and geographic
filtering workflow is implemented in
[`src/acquisition/download_opensky_monday_states.py`](src/acquisition/download_opensky_monday_states.py).

Third-party datasets are **not distributed under the license of this repository**.

Each dataset remains subject to the terms, conditions, attribution requirements, and licensing restrictions imposed by its original provider.

Raw datasets should therefore not be uploaded to this repository unless their respective licenses explicitly allow redistribution.

## 🗺️ Current Development

The initial development focuses on processing aircraft state vectors and visualising air traffic within configurable geographical regions.

Current functionality includes:

- configurable geographic bounding boxes;
- aircraft position visualisation;
- satellite basemap integration;
- barometric and geometric altitude processing;
- ground speed and vertical speed;
- trajectory-vector estimation;
- ground/airborne aircraft differentiation;
- configurable aircraft separation criteria;
- pairwise aircraft proximity analysis;
- conflict and proximity-warning visualisation.

The project will progressively incorporate historical trajectory processing and AI-based conflict analysis.

## Repository structure

```text
data/
├── raw/              # Original OpenSky data and aeronautical source documents
├── processed/        # Reusable cleaned and transformed datasets
└── samples/          # Small datasets for examples and tests
src/
├── acquisition/      # Data collection and provider-specific clients
├── preprocessing/    # Cleaning and feature preparation
├── trajectories/     # Trajectory handling and analysis
├── conflict_detection/
├── visualization/
└── models/
notebooks/            # Exploratory and reproducible research notebooks
experiments/          # Experiment configurations and runs
results/              # Generated figures and tables
docs/                 # Thesis and project documentation
tests/                # Automated tests
```

The folders have the following responsibilities:

- `data/`: datasets. `raw/` preserves source data without modification, `processed/`
  is for cleaned or transformed data, and `samples/` is for small datasets used in
  examples and tests. The original ILS documents are under `data/raw/ils/` and
  OpenSky snapshots under `data/raw/opensky/`. Historical regional Parquet output
  is generated under `data/raw/opensky_weekly/` and is intentionally not committed as
  part of the source repository.
- `src/`: reusable project code, organised by research stage. See
  [`src/README.md`](src/README.md).
- `notebooks/`: exploratory analyses, visual investigations and reproducible
  research notebooks.
- `experiments/`: experiment definitions, configurations and run-specific material.
- `results/`: generated outputs such as figures and tables. See
  [`results/README.md`](results/README.md).
- `docs/`: thesis-related and technical documentation.
- `tests/`: automated tests for the source code and data-processing pipeline.
- `README.md`, `LICENSE` and `requirements.txt`: project entry-point documentation,
  licensing information and Python dependencies respectively.

## ⚠️ Conflict Detection Model

A simplified separation model is currently used for research and experimentation.

Aircraft encounters can be classified according to configurable horizontal and vertical separation thresholds.

The thresholds implemented in this repository are modelling parameters and **must not be interpreted as universally applicable operational ATC separation minima**.

Actual separation requirements depend on airspace, surveillance capabilities, flight level, operational procedures, wake turbulence categories, and applicable aviation regulations.

## 🧠 Planned AI Pipeline

The expected research workflow is:

```text
Historical flight data
        ↓
Trajectory preprocessing
        ↓
Aircraft encounter extraction
        ↓
Separation / CPA / TCPA analysis
        ↓
Conflict scenario labelling
        ↓
Feature engineering
        ↓
Machine-learning models
        ↓
Conflict detection / prediction
        ↓
Resolution advisory research

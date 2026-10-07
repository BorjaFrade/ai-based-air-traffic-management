# Source code

The `src/` directory contains the reusable code of the research pipeline. The
folders are separated by responsibility so that data collection, analysis and
modelling can evolve independently:

- `acquisition/`: obtains data from external providers and creates initial
  snapshots or trajectory collections. See [`acquisition/README.md`](acquisition/README.md).
- `preprocessing/`: cleaning, validation, normalisation and feature preparation.
- `trajectories/`: trajectory representation, reconstruction and trajectory-based
  calculations.
- `conflict_detection/`: separation, CPA/TCPA and conflict-identification logic.
- `visualization/`: reusable plotting and map-rendering functions. The current
  OpenSky renderer is available at
  [`acquisition/opensky_plot.py`](acquisition/opensky_plot.py) and exposes
  `render_map()` for reuse by other acquisition or analysis scripts.
- `models/`: machine-learning models, training code and prediction utilities.

At present, the implemented functionality is concentrated in `acquisition/`.
The other directories are reserved for the subsequent stages of the thesis and
should contain code that can be imported or executed reproducibly rather than
one-off generated outputs.

## Data flow

```text
src/acquisition/
        ↓
data/raw/
        ↓
src/preprocessing/
        ↓
data/processed/
        ↓
src/trajectories/ and src/conflict_detection/
        ↓
src/models/ and src/visualization/
        ↓
results/
```

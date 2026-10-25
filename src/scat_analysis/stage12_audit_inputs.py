"""Read-only audit of Stage 12 input schemas, week coverage and identity provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage11_core import BASE, _week_groups

OUT=BASE/"12_pairwise_encounters"
WEEKS=("20161015_20161021","20161112_20161118")


def main():
    for d in ("tables","episodes","diagnostics","examples","reports",
              "figures/01_trajectories","figures/02_temporal_profiles",
              "figures/03_geometry_distributions","figures/04_cpa_tcpa",
              "figures/05_duration_persistence","figures/06_week_comparison",
              "figures/07_structure_discovery","figures/08_operational_comparison"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    states=pq.ParquetFile(BASE/"02_traffic/scat_states.parquet")
    clear=pq.ParquetFile(BASE/"03_clearances/scat_clearances.parquet")
    needed={"flight_key","flight_id","archive_week","source_archive","source_member","plot_index",
            "time_epoch","latitude","longitude","altitude_ft","ground_speed_kt","track_deg",
            "vertical_rate_fpm","on_ground","flight_phase","active_centre_id","callsign"}
    assert needed.issubset(states.schema.names)
    assert {"clearance_id","flight_key","archive_week","clearance_epoch","clearance_type"}.issubset(clear.schema.names)
    rows=[]
    for week in WEEKS:
        groups=list(_week_groups(states,week))
        native=sum(states.metadata.row_group(i).num_rows for i in groups)
        sample_indices=np.linspace(0,len(groups)-1,min(30,len(groups)),dtype=int)
        sample=pd.concat([states.read_row_group(groups[j],columns=["flight_key","flight_id","archive_week",
            "source_archive","source_member","time_epoch","latitude","longitude","altitude_ft",
            "ground_speed_kt","track_deg","vertical_rate_fpm","on_ground"]).to_pandas() for j in sample_indices],ignore_index=True)
        sample=sample.loc[sample.archive_week.eq(week)]
        identity=sample.groupby("flight_key").agg(members=("source_member","nunique"),
                      archives=("source_archive","nunique"),flight_ids=("flight_id","nunique"))
        rows.append(dict(week=week,native_row_groups=len(groups),native_rows_from_metadata=native,
            sampled_rows=len(sample),sampled_unique_flights=int(sample.flight_key.nunique()),
            sampled_id_collision_keys=int(((identity.members>1)|(identity.archives>1)|(identity.flight_ids>1)).sum()),
            sampled_missing_position=int((~sample.latitude.between(-90,90)|~sample.longitude.between(-180,180)).sum()),
            sampled_ground_true=int(sample.on_ground.fillna(False).sum()),
            sampled_missing_kinematics=int((sample.ground_speed_kt.isna()|sample.track_deg.isna()).sum()),
            stage11d_all_altitude_snapshots=json.loads((BASE/"11d_validation/diagnostics"/f"{week}_coverage.json").read_text())["snapshots"]))
    pd.DataFrame(rows).to_csv(OUT/"diagnostics/input_coverage_audit.csv",index=False)
    inventory={"states_path":str(BASE/"02_traffic/scat_states.parquet"),
       "states_rows_all_weeks":states.metadata.num_rows,"states_columns":states.schema.names,
       "clearance_path":str(BASE/"03_clearances/scat_clearances.parquet"),
       "clearance_rows_all_weeks":clear.metadata.num_rows,"clearance_columns":clear.schema.names,
       "stage05_geometry_source":"src/scat_analysis/05_build_interaction_features.py:horizontal_geometry,pair_features",
       "stage10b_detector_adapter":"src/scat_analysis/10b_exact_dynamic_rule_cruise.py:detector_frame,execute",
       "stage11d_snapshot_reference":"src/scat_analysis/stage11d_core.py:load_snapshots,select_neighbors",
       "note":"Metadata counts are exact for week row groups if each group has one archive_week; sampled quality is not a population estimate."}
    (OUT/"diagnostics/input_schema_inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__":main()

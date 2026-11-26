#!/usr/bin/env python3
"""Run the paired lift ladder at alternative gaps and on calibration weeks."""
from __future__ import annotations

import json
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import units_ladder as u

OUT = u.ROOT / "results/benchmarks_units/robustness"
ORIGINAL = u.ROOT / "results/benchmarks_units/ladder"
SCRIPT = u.ROOT / "paper/figure_scripts/units_ladder.py"
STEPS = ("0", "0b", "1", "2", "3")
COMPARISONS = (("0b","0"),("1","0"),("2","0"),("3","0"),
               ("3","2"),("1","0b"),("2","1"))
RUNS = (("test_gap360",360,"test"),("gap240",240,"test"),
        ("gap480",480,"test"),("calibration_gap360",360,"calib"))


def compare_csv(name: str, candidate: Path) -> float:
    old = pd.read_csv(ORIGINAL/name)
    new = pd.read_csv(candidate/name)
    u.check(old.columns.equals(new.columns) and len(old)==len(new), f"B0 columns or rows differ: {name}")
    u.check(old.select_dtypes(exclude="number").equals(new.select_dtypes(exclude="number")),
            f"B0 text columns differ: {name}")
    difference = float((old.select_dtypes("number")-new.select_dtypes("number")).abs().max().max())
    u.check(difference<=1e-12, f"B0 {name} differs by {difference}")
    return difference


def fmt(row: pd.Series, prefix: str) -> str:
    return f"{row[prefix]:.2f} ({row[prefix+'_lo']:.2f}–{row[prefix+'_hi']:.2f})"


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true",help="Reuse complete run folders from this invocation after an interruption")
    args=parser.parse_args()
    start = time.perf_counter()
    u.check(not OUT.exists() or (args.resume and not (OUT/"summary.csv").exists()),
            f"Output already exists: {OUT}")
    audit = json.loads((u.ROOT/"data/processed/scat/19_calib_selection/tables/audit.json").read_text())
    u.check(len(audit["test"])==4 and len(audit["calib"])==9 and
            set(audit["test"]).isdisjoint(audit["calib"]) and
            set(audit["test"]+audit["calib"])==set(audit["weeks"]),
            "Stage 20/19 split is not 4 test plus 9 calibration weeks")
    with tempfile.TemporaryDirectory(prefix="units_ladder_B0_") as directory:
        command=[sys.executable,str(SCRIPT),"--out",directory]
        print("B0:"," ".join(command),flush=True)
        subprocess.run(command,check=True,cwd=u.ROOT)
        b0={name:compare_csv(name,Path(directory)) for name in
            ("ladder_rates_lifts.csv","ladder_ratios.csv")}
    OUT.mkdir(parents=True,exist_ok=args.resume)
    for name,gap,weeks in RUNS[1:]:
        destination=OUT/name
        if args.resume and destination.exists():
            u.check(all((destination/file).is_file() for file in
                        ("ladder_rates_lifts.csv","ladder_ratios.csv","diagnostics.json")),
                    f"Incomplete run folder: {destination}")
            print(f"Reusing {name}",flush=True)
            continue
        u.check(not destination.exists(),f"Output already exists: {destination}")
        command=[sys.executable,str(SCRIPT),"--gap",str(gap),"--weeks",weeks,
                 "--out",str(destination),"--instant-only","--main-schemes"]
        print(name,": "," ".join(command),sep="",flush=True)
        subprocess.run(command,check=True,cwd=u.ROOT)
    summary=[]
    for name,gap,weeks in RUNS:
        source=ORIGINAL if name=="test_gap360" else OUT/name
        rates=pd.read_csv(source/"ladder_rates_lifts.csv")
        ratios=pd.read_csv(source/"ladder_ratios.csv")
        diag=json.loads((source/"diagnostics.json").read_text())
        weekly=diag["weekly"]
        totals={"observations":sum(v["observations"] for v in weekly.values()),
                "green_episodes":sum(v["green_episodes"] for v in weekly.values()),
                "non_green_episodes":sum(v["non_green_episodes"] for v in weekly.values())}
        totals["red_observations"]=int(rates.loc[(rates.scheme=="four_class")&
                                                   (rates.stratifier=="instant")&
                                                   (rates.step=="0")&
                                                   (rates["class"]=="rojo"),"n_obs"].iloc[0])
        totals["red_episodes"]=int(rates.loc[(rates.scheme=="four_class")&
                                               (rates.stratifier=="instant")&
                                               (rates.step=="3")&
                                               (rates["class"]=="rojo"),"n_episodes"].iloc[0])
        for scheme,classes in (("four_class",u.COLORS[1:]),("binary_1",("positive",))):
            for cls in classes:
                row={"execution":name,"gap":gap,"weeks":";".join(audit[weeks]),"scheme":scheme,"class":cls,**totals}
                for step in STEPS:
                    r=rates.loc[(rates.scheme==scheme)&(rates.stratifier=="instant")&
                                (rates.step==step)&(rates["class"]==cls)].iloc[0]
                    for key,col in (("lift","lift"),("lo","lift_lo"),("hi","lift_hi")):
                        row[f"{col}_{step}"]=r[key]
                for numerator,denominator in COMPARISONS:
                    r=ratios.loc[(ratios.scheme==scheme)&(ratios.stratifier=="instant")&
                                 (ratios.comparison==f"{numerator}_vs_{denominator}")&
                                 (ratios["class"]==cls)].iloc[0]
                    prefix=f"ratio_{numerator}_{denominator}"
                    row[prefix],row[prefix+"_lo"],row[prefix+"_hi"]=r["ratio"],r["lo"],r["hi"]
                    row[f"logdiff_{numerator}_{denominator}"]=r["logdiff"]
                    row[f"logdiff_{numerator}_{denominator}_lo"]=r["logdiff_lo"]
                    row[f"logdiff_{numerator}_{denominator}_hi"]=r["logdiff_hi"]
                summary.append(row)
    table=pd.DataFrame(summary)
    table.to_csv(OUT/"summary.csv",index=False)
    elapsed=round(time.perf_counter()-start,1)
    lines=["# Robustness of the lift ladder", "",
           "Command for a new run: `python3 paper/figure_scripts/units_robustness.py`. After an interruption: `python3 paper/figure_scripts/units_robustness.py --resume`.",
           f"Total measured time: {elapsed} s. Seed {u.SEED}; {u.B} whole-day bootstrap replicates in each run.",
           f"B0: maximum differences with respect to the published CSV files: rates/lifts {b0['ladder_rates_lifts.csv']:.3g}; ratios {b0['ladder_ratios.csv']:.3g} (tolerance 1e-12).",
           "", "The calibration weeks are read from `data/processed/scat/19_calib_selection/tables/audit.json`. The thresholds of the four levels were fitted on these weeks; the binary screen m<1 has no fitted parameters.",
           "The green/negative reference is re-estimated on the weeks of each run. The instantaneous FL290–FL410 strata [0,15), [15,25), [25,35), [35,∞) are used. Ratios and differences of logarithms are computed within each common replicate.",
           "On test with gap 360, the frozen checks of Stage 20/21, Table 6 and the population of 6,856,880 are applied. In the other scenarios, the checks are: unique assignment of observations, total weight of Step 0b equal to the number of episodes, and first red maximum / first m<1 equal to k_cls=1. The stratum counts are checked against Stage 16 in each week.",
           "", "Individual commands run from the repository root:",
           "- B0: `python3 paper/figure_scripts/units_ladder.py --out /private/tmp/units_ladder_B0` (the orchestrator uses a unique temporary folder).",
           "- Gap 240: `python3 paper/figure_scripts/units_ladder.py --gap 240 --weeks test --out results/benchmarks_units/robustness/gap240 --instant-only --main-schemes`.",
           "- Gap 480: `python3 paper/figure_scripts/units_ladder.py --gap 480 --weeks test --out results/benchmarks_units/robustness/gap480 --instant-only --main-schemes`.",
           "- Calibration: `python3 paper/figure_scripts/units_ladder.py --gap 360 --weeks calib --out results/benchmarks_units/robustness/calibration_gap360 --instant-only --main-schemes`.",
           "", "Computation times per run (s): " + ", ".join(
               f"{name}={json.loads(((ORIGINAL if name=='test_gap360' else OUT/name)/'diagnostics.json').read_text())['elapsed_seconds']}"
               for name,_,_ in RUNS) + ".",
           "", "| Run | Class | 0 | 1 | 2 | 3 | 1/0 | 2/0 |", "|---|---|---|---|---|---|---|---|"]
    for name,_,_ in RUNS:
        for scheme,cls in (("four_class","rojo"),("binary_1","positive")):
            r=table.loc[(table.execution==name)&(table.scheme==scheme)&(table["class"]==cls)].iloc[0]
            cells=[]
            for step in ("0","1","2","3"):
                cells.append(f"{r[f'lift_{step}']:.2f} ({r[f'lift_lo_{step}']:.2f}–{r[f'lift_hi_{step}']:.2f})")
            for num in ("1","2"):
                prefix=f"ratio_{num}_0"
                cells.append(f"{r[prefix]:.2f} ({r[prefix+'_lo']:.2f}–{r[prefix+'_hi']:.2f})")
            lines.append(f"| {name} | {'red' if cls=='rojo' else cls} | "+" | ".join(cells)+" |")
    lines += ["", "| Run | Observations | Green episodes | Non-green episodes | Red obs. | Red episodes |",
              "|---|---:|---:|---:|---:|---:|"]
    for name,_,_ in RUNS:
        r=table.loc[table.execution==name].iloc[0]
        lines.append(f"| {name} | {r.observations:,} | {r.green_episodes:,} | {r.non_green_episodes:,} | {r.red_observations:,} | {r.red_episodes:,} |")
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    print(f"Robustness complete in {elapsed} s; B0={b0}",flush=True)


if __name__=="__main__":
    main()

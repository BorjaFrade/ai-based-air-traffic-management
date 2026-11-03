import sys, pathlib
root = pathlib.Path(sys.argv[1])
p = root / "stage16_report.py"; s = p.read_text()
if "nominal_days_only" not in s:
    s = s.replace('''def load_tables(tab_dir: Path):
    cells =''', '''def nominal_days_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop border days: snapshots whose UTC day falls outside the nominal 7-day SCAT week of the shard.
    (16 such days, 803 of 21.6 M pair-snapshots; flights crossing midnight.)"""
    t0 = pd.Timestamp("1970-01-01")
    d0 = (pd.to_datetime(df.week.str.slice(0, 8), format="%Y%m%d") - t0).dt.days
    d1 = (pd.to_datetime(df.week.str.slice(9, 17), format="%Y%m%d") - t0).dt.days
    return df[(df.day >= d0) & (df.day <= d1)].reset_index(drop=True)


def load_tables(tab_dir: Path, nominal_only: bool = True):
    cells =''')
    s = s.replace('''    for name, df in (("cells", cells), ("colour_states", states)):''', '''    if nominal_only:
        cells, states = nominal_days_only(cells), nominal_days_only(states)
    for name, df in (("cells", cells), ("colour_states", states)):''')
    p.write_text(s)
p = root / "stage18_optimal_k.py"; s = p.read_text()
if "weeks=None" not in s:
    for fn in ("structured_main", "region_main", "named_main"):
        s = s.replace(f'def {fn}(base_dir="data/processed/scat/16_pair_hazard", out_dir="data/processed/scat/18_optimal_k"):',
                      f'def {fn}(base_dir="data/processed/scat/16_pair_hazard", out_dir="data/processed/scat/18_optimal_k", weeks=None):')
    # inside those three functions replace the weeks line
    parts = s.split("\n"); out = []; infn = False
    for ln in parts:
        if ln.startswith("def "): infn = any(ln.startswith(f"def {f}(") for f in ("structured_main", "region_main", "named_main"))
        if infn and ln.strip() == "weeks = sorted(cells.week.unique())":
            ln = "    weeks = weeks or sorted(cells.week.unique())"
        out.append(ln)
    p.write_text("\n".join(out))
print("patched", root)

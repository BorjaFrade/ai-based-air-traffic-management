"""Build the LaTeX table rows used in paper/MDPI/main.tex from the saved CSV outputs.

Inputs (read-only): results/benchmarks_units/ladder/{ladder_rates_lifts,ladder_ratios}.csv and
results/benchmarks_units/mechanism/position_rates_lifts.csv.
Outputs: results/benchmarks_units/paper_tables/{table_units,table_binary,table_position}.tex
(rows only; the table environment is in main.tex).

Run from the repository root:  python3 paper/figure_scripts/units_paper_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "results/benchmarks_units/ladder"
MECH = ROOT / "results/benchmarks_units/mechanism"
OUT = ROOT / "results/benchmarks_units/paper_tables"


def f2(x):
    return f"{x:.2f}"


def lift_ci(r):
    return f"{f2(r.lift)} ({f2(r.lo)}--{f2(r.hi)})"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rates = pd.read_csv(LADDER / "ladder_rates_lifts.csv")
    ratios = pd.read_csv(LADDER / "ladder_ratios.csv")
    inst = rates[rates.stratifier.eq("instant")]

    def pick(scheme, step, cls):
        return inst[inst.scheme.eq(scheme) & inst.step.eq(step) & inst["class"].eq(cls)].iloc[0]

    def ratio(scheme, comp, cls):
        r = ratios[ratios.scheme.eq(scheme) & ratios.stratifier.eq("instant") &
                   ratios.comparison.eq(comp) & ratios["class"].eq(cls)].iloc[0]
        return f"{f2(r.ratio)} ({f2(r.lo)}--{f2(r.hi)})"

    # Table 6: units of evaluation, four levels (class & rate & lift for each of three columns).
    names = {"verde": "Green", "amarillo": "Yellow", "naranja": "Orange", "rojo": "Red"}
    lines = []
    for cls, label in names.items():
        parts = []
        for step in ("0", "2", "2_any"):
            r = pick("four_class", step, cls)
            parts.append(f"{100 * r.rate:.1f} & " + ("1.00 (ref.)" if cls == "verde" else lift_ci(r)))
        lines.append(f"{label} & " + " & & ".join(parts) + r"\\")
    (OUT / "table_units.tex").write_text("\n".join(lines) + "\n")

    # Table 7: binary screens m_min < c.
    lines = []
    for c, scheme in ((1, "binary_1"), (1.5, "binary_1.5"), (2, "binary_2")):
        p0, p2, pa = (pick(scheme, s, "positive") for s in ("0", "2", "2_any"))
        lines.append(f"{c:g} & {f2(p0.lift)} & {f2(p2.lift)} & {ratio(scheme, '2_vs_0', 'positive')} & "
                     f"{f2(pa.lift)} & {ratio(scheme, '2_any_vs_0', 'positive')}" + r"\\")
    (OUT / "table_binary.tex").write_text("\n".join(lines) + "\n")

    # Position table: red and binary m<1, position of the observation within its episode.
    pos = pd.read_csv(MECH / "position_rates_lifts.csv")
    pos = pos[pos.variant.eq("k_ep")]
    lines = []
    for b in ("1", "2", "3", "4+"):
        cells = [b if b != "4+" else r"$\geq 4$"]
        for scheme, cls in (("four_class", "rojo"), ("binary_1", "positive")):
            r = pos[pos.scheme.eq(scheme) & pos["class"].eq(cls) & pos.bin.astype(str).eq(b)].iloc[0]
            cells += [f"{int(r.n_obs):,}", f"{100 * r.rate:.1f}",
                      f"{f2(r.lift_ref0)} ({f2(r.lift_ref0_lo)}--{f2(r.lift_ref0_hi)})"]
        lines.append(" & ".join(cells) + r"\\")
    (OUT / "table_position.tex").write_text("\n".join(lines) + "\n")
    print("Wrote", ", ".join(p.name for p in sorted(OUT.glob("*.tex"))))


if __name__ == "__main__":
    main()

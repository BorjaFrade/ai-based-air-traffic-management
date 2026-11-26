#!/usr/bin/env python3
"""Paired, one-change-at-a-time clearance-lift ladder on the four test weeks.

Green and non-green runs are formed separately with the same 360-second gap.
Unlike the green_all benchmark, nearby green observations remain green runs so
every canonical pair observation belongs to exactly one episode.
"""
from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/benchmarks_units/ladder"
sys.path.insert(0, str(ROOT / "src/scat_analysis"))
import stage15_predicted_separation as s15  # noqa: E402

SEED, B, GAP = 20261112, 400, 360
COLORS = ("verde", "amarillo", "naranja", "rojo")
STEPS = ("0", "0b", "1", "2", "3", "2_any")
STRATA = ("instant", "episode_hour")
EDGES_EP = np.array([2012, 2939])  # frozen calibration cutpoints from green_all


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def preflight() -> None:
    """Read existing outputs; never call the writers of the earlier unit analysis."""
    a = pd.read_csv(ROOT / "results/benchmarks_units/table6_reproduction.csv")
    wanted = {"pair_instant": [1, 1.53, 2.23, 3.11],
              "episode_start": [1, 1.37, 1.75, 1.96],
              "episode_any": [1, .89, 1.19, 1.33]}
    for window, lift in wanted.items():
        x = a.loc[a.window.eq(window)].set_index("color").loc[list(COLORS)]
        check(np.array_equal(np.round(x.lift.to_numpy(float), 2), lift),
              f"Table 6 preflight changed: {window}: {x.lift.to_list()}")
    b = pd.read_csv(ROOT / "results/benchmarks_units/binary_rates_lifts.csv")
    check({1., 1.5, 2.} <= set(b.threshold) and b.binary_class.nunique() == 2,
          "Existing binary unit analysis is incomplete")
    obs = pd.read_csv(ROOT / "results/stage20/stage20_non_green_observations.csv.gz",
                      usecols=["episode_id", "time_ns", "color", "split"])
    episodes = pd.read_csv(ROOT / "results/stage20/stage20_episodes.csv",
                           usecols=["episode_id", "highest_color"])
    obs = obs.loc[obs.split.eq("test")].merge(episodes, on="episode_id", validate="many_to_one")
    first_max = obs.loc[obs.color.eq(obs.highest_color)].groupby("episode_id").time_ns.min()
    stage21 = pd.read_csv(ROOT / "results/stage21/stage21_encounters.csv",
                          usecols=["episode_id", "split", "anchor_time_utc"])
    stage21 = stage21.loc[stage21.split.eq("test")].set_index("episode_id")
    anchor = pd.to_datetime(stage21.anchor_time_utc, utc=True).dt.as_unit("ns").astype("int64")
    check(len(first_max) == 17338 and first_max.sort_index().equals(anchor.sort_index().rename("time_ns")),
          "The first maximum-colour observation differs from Stage 21 anchor_time_utc")


def test_days(weeks: list[str]) -> tuple[list[str], dict[int, int]]:
    days = sorted(d for w in weeks for d in pd.date_range(w[:8], periods=7).strftime("%Y-%m-%d"))
    ordinal = [int(pd.Timestamp(d, tz="UTC").timestamp() // 86400) for d in days]
    return days, dict(zip(ordinal, range(len(days))))


def snapshot_strata(week: str, times: np.ndarray, pf: pq.ParquetFile) -> np.ndarray:
    """Replay Stage 16's instantaneous count of eligible aircraft at each T."""
    states, _ = s15.load_week_states_lean(pf, week)
    st = states.time_epoch.to_numpy(float)
    entity, _ = pd.factorize(states.entity_key, sort=False)
    kin_ok = np.isfinite(states.ground_speed_kt.to_numpy(float)) & np.isfinite(states.track_deg.to_numpy(float))
    del states
    lo = np.searchsorted(st, times - s15.MAX_AGE_S, side="left")
    hi = np.searchsorted(st, times, side="right")
    result = np.empty(len(times), dtype=np.int8)
    for i, (a, b) in enumerate(zip(lo, hi)):
        pos = np.arange(a, b)
        if len(pos) < 2:
            n = 0
        else:
            _, first_rev = np.unique(entity[pos][::-1], return_index=True)
            rows = pos[-1-first_rev]
            n = int(kin_ok[rows].sum())
        result[i] = np.searchsorted([15, 25, 35], n, side="right")
    return result


def make_episodes(t: np.ndarray, pair_id: np.ndarray, rank: np.ndarray,
                  label: np.ndarray, m: np.ndarray) -> dict:
    """Stable 360-s runs, separately for green and non-green observations."""
    n = len(t)
    ep_id = np.empty(n, np.int32)
    assigned = np.zeros(n, bool)
    start_ix, max_ix, first_rank, max_rank, size, any_y, minimum = [], [], [], [], [], [], []
    offset = 0
    for green in (False, True):
        picked = np.flatnonzero((rank == 0) if green else (rank > 0))
        ordered = picked[np.lexsort((picked, t[picked], pair_id[picked]))]
        cut = np.empty(len(ordered), bool)
        cut[0] = True
        cut[1:] = (pair_id[ordered[1:]] != pair_id[ordered[:-1]]) | (t[ordered[1:]]-t[ordered[:-1]] > GAP)
        starts = np.flatnonzero(cut)
        runs = np.cumsum(cut).astype(np.int32)-1 + offset
        ep_id[ordered] = runs
        assigned[ordered] = True
        maxes = np.maximum.reduceat(rank[ordered], starts)
        sizes = np.diff(np.r_[starts, len(ordered)])
        first = ordered[starts]
        at_max = rank[ordered] == maxes[runs-offset]
        # The first maximum may occur later than the first row of the run.
        max_start = np.minimum.reduceat(np.where(at_max, np.arange(len(ordered)), len(ordered)), starts)
        highest = ordered[max_start]
        start_ix.append(first)
        max_ix.append(highest)
        first_rank.append(rank[first])
        max_rank.append(maxes)
        size.append(sizes)
        any_y.append(np.maximum.reduceat(label[ordered].astype(np.int8), starts).astype(bool))
        minimum.append(np.minimum.reduceat(m[ordered], starts))
        offset += len(starts)
    check(offset > 0 and assigned.all() and np.bincount(ep_id, minlength=offset).sum() == n,
          "Not every observation was assigned to one episode")
    return {"id": ep_id, "first": np.concatenate(start_ix), "max": np.concatenate(max_ix),
            "first_rank": np.concatenate(first_rank), "max_rank": np.concatenate(max_rank),
            "size": np.concatenate(size), "any_y": np.concatenate(any_y),
            "min_m": np.concatenate(minimum), "count": offset}


def first_positive_indices(ep_id: np.ndarray, t: np.ndarray, m: np.ndarray,
                           n_ep: int, threshold: float) -> np.ndarray:
    """First m<c; ties use canonical Parquet row order."""
    selected = np.flatnonzero(m < threshold)
    ordered = selected[np.lexsort((selected, t[selected], ep_id[selected]))]
    ids, first_pos = np.unique(ep_id[ordered], return_index=True)
    result = np.full(n_ep, -1, np.int32)
    result[ids] = ordered[first_pos]
    return result


def add(acc: dict, key: tuple, day: np.ndarray, stratum: np.ndarray, y: np.ndarray,
        weight: np.ndarray, episode: np.ndarray, underlying_obs: int,
        n_days: int, n_strata: int) -> None:
    """Accumulate daily-stratum sufficient statistics and descriptive counts."""
    if len(day) == 0:
        return
    flat = day * n_strata + stratum
    rec = acc[key]
    rec["den"] += np.bincount(flat, weights=weight, minlength=n_days*n_strata).reshape(n_days, n_strata)
    rec["num"] += np.bincount(flat, weights=weight*y, minlength=n_days*n_strata).reshape(n_days, n_strata)
    rec["n_obs"] += len(day)
    rec["underlying_obs"] += underlying_obs
    rec["positive_rows"] += int(np.count_nonzero(y))
    rec["n_episodes"] += len(np.unique(episode))


def write_tex(rates: pd.DataFrame) -> None:
    """Booktabs fragment for the four-class comparison under both strata."""
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Traffic & Step & Yellow & Orange & Red \\", r"\midrule"]
    labels = {"0": "0: pair-instant", "0b": "0b: equal weight", "1": "1: first obs.",
              "2": "2: episode, start", "2_any": "2: any observation",
              "3": "3: episode, maximum"}
    for strat, heading in (("instant", "instantaneous aircraft"),
                           ("episode_hour", "episode starts/h")):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{{heading}}} \\")
        for step in ("0", "0b", "1", "2", "2_any", "3"):
            values = []
            for cls in COLORS[1:]:
                row = rates.loc[rates.scheme.eq("four_class") & rates.stratifier.eq(strat) &
                                rates.step.eq(step) & rates["class"].eq(cls)].iloc[0]
                values.append(f"{row.lift:.2f} ({row.lo:.2f}--{row.hi:.2f})")
            lines.append(f" & {labels[step]} & " + " & ".join(values) + r" \\")
        if strat == "instant":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUT / "ladder_table.tex").write_text("\n".join(lines))


def main() -> None:
    global GAP, OUT, STRATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=int, default=360)
    parser.add_argument("--weeks", choices=("test", "calib"), default="test")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--instant-only", action="store_true")
    parser.add_argument("--main-schemes", action="store_true")
    args = parser.parse_args()
    check(args.gap > 0, "Episode gap must be positive")
    GAP, OUT = args.gap, args.out
    if args.instant_only:
        STRATA = ("instant",)
    frozen_test = args.gap == 360 and args.weeks == "test"
    started = time.perf_counter()
    if frozen_test:
        preflight()
    audit = json.loads((ROOT / "data/processed/scat/19_calib_selection/tables/audit.json").read_text())
    weeks = audit[args.weeks]
    days, day_lookup = test_days(weeks)
    weights = np.random.default_rng(SEED).multinomial(len(days), np.ones(len(days))/len(days), size=B)
    n_days = len(days)
    acc = {}
    def record(key, n_strata):
        if key not in acc:
            acc[key] = {"den": np.zeros((n_days, n_strata)), "num": np.zeros((n_days, n_strata)),
                        "n_obs": 0, "underlying_obs": 0, "positive_rows": 0, "n_episodes": 0}
    diagnostics = {"test_weeks": weeks, "days": days, "weekly": {}, "seed": SEED, "replicates": B} if frozen_test else {"weeks": weeks, "gap": GAP, "days": days, "weekly": {}, "seed": SEED, "replicates": B}
    pf = pq.ParquetFile(ROOT / "data/processed/scat/02_traffic/scat_states.parquet")
    if frozen_test:
        frozen = pd.read_parquet(ROOT / "results/benchmarks_green_all/episodes.parquet",
                                 columns=["week", "highest_color", "observations"])
        frozen = frozen.loc[frozen.week.isin(weeks) & frozen.highest_color.eq("verde")]
        frozen_green = frozen.groupby("week").observations.agg(["size", "sum"])
        del frozen
        stage20 = pd.read_csv(ROOT / "results/stage20/stage20_episodes.csv", usecols=["week", "split"])
        non_green_counts = stage20.loc[stage20.split.eq("test")].week.value_counts()
    global_episode_offset = 0
    for week in weeks:
        path = ROOT / "results/stage20/pairs" / f"pairs_{week}.parquet"
        cols = ["T", "flight_key_a", "flight_key_b", "source_member_a", "source_member_b",
                "m_min", "color", "stage16_label_300"]
        frame = pd.read_parquet(path, columns=cols)
        t = frame["T"].to_numpy(float)
        m = frame.m_min.to_numpy(float)
        y = frame.stage16_label_300.to_numpy(bool)
        rank = frame.color.map({c: i for i, c in enumerate(COLORS)}).to_numpy(np.int8)
        a = frame.flight_key_a.astype(str).str.cat(frame.source_member_a.astype(str), sep="\x1f")
        b = frame.flight_key_b.astype(str).str.cat(frame.source_member_b.astype(str), sep="\x1f")
        keys = pd.Index(pd.concat([a, b], ignore_index=True).unique())
        ac, bc = keys.get_indexer(a), keys.get_indexer(b)
        pair = np.minimum(ac, bc).astype(np.int64)*len(keys) + np.maximum(ac, bc)
        del a, b, keys, ac, bc, frame
        unique_t, t_code = np.unique(t, return_inverse=True)
        inst_t = snapshot_strata(week, unique_t, pf)
        inst = inst_t[t_code]
        day = np.array([day_lookup[int(v)] for v in (t//86400).astype(int)], dtype=np.int16)
        ep = make_episodes(t, pair, rank, y, m)
        e_id = ep["id"] + global_episode_offset
        first, maxi = ep["first"], ep["max"]
        ep_start = t[first]
        starts_sorted = np.sort(ep_start)
        def hour_stratum(query):
            density = (np.searchsorted(starts_sorted, query, side="right") -
                       np.searchsorted(starts_sorted, query-3600, side="left"))
            return np.searchsorted(EDGES_EP, density, side="right").astype(np.int8)
        ep_h_start, ep_h_max, obs_h = hour_stratum(ep_start), hour_stratum(t[maxi]), hour_stratum(t)
        # Check instantaneous density against Stage 16's frozen daily tensor.
        expected_file = ROOT / "data/processed/scat/16_pair_hazard/tables" / f"cells_{week}.parquet"
        cells = pd.read_parquet(expected_file, columns=["day", "predictor", "dens_bin", "n_pairs"])
        nominal = (cells.day >= min((unique_t//86400).astype(int))) & (cells.day <= max((unique_t//86400).astype(int)))
        expected = cells.loc[nominal & cells.predictor.eq("R")].groupby(["day", "dens_bin"]).n_pairs.sum()
        got = pd.DataFrame({"day": (t//86400).astype(int), "dens_bin": inst}).groupby(["day", "dens_bin"]).size()
        check(np.array_equal(expected.reindex(got.index, fill_value=0).to_numpy(), got.to_numpy()) and
              int(expected.sum()) == len(t),
              f"Instantaneous traffic strata differ from Stage 16 in {week}")
        n_non_green = int((ep["max_rank"] > 0).sum())
        if frozen_test:
            check(n_non_green == int(non_green_counts[week]), f"Non-green episode count differs in {week}")
        if frozen_test:
            diagnostics["weekly"][week] = {"observations": len(t), "non_green_episodes": n_non_green,
                                            "green_episodes": int((ep["max_rank"] == 0).sum()),
                                            "frozen_green_episodes": int(frozen_green.loc[week, "size"]),
                                            "frozen_green_observations": int(frozen_green.loc[week, "sum"]),
                                            "near_non_green_green_observations": int((rank == 0).sum()-frozen_green.loc[week, "sum"]),
                                            "tmax_equals_start": int((t[maxi] == ep_start).sum())}
        else:
            diagnostics["weekly"][week] = {"observations": len(t), "non_green_episodes": n_non_green,
                                            "green_episodes": int((ep["max_rank"] == 0).sum()),
                                            "tmax_equals_start": int((t[maxi] == ep_start).sum()),
                                            "red_observations": int((rank == 3).sum()),
                                            "red_episodes": int((ep["max_rank"] == 3).sum())}
        check(np.array_equal(np.sort(maxi[ep["max_rank"] == 3]),
                             np.sort(first_positive_indices(ep["id"], t, np.where(rank == 3, 0., 1.), ep["count"], .5)[ep["max_rank"] == 3])),
              f"First red occurrence differs from T_max in {week}")
        configurations = [("four_class", rank, ep["first_rank"], ep["max_rank"], maxi)]
        for c in ((1.,) if args.main_schemes else (1., 1.5, 2.)):
            p = m < c
            emax = ep["min_m"] < c
            pmax = first_positive_indices(ep["id"], t, m, ep["count"], c)
            pmax[~emax] = first[~emax]
            if c == 1.:
                check(np.array_equal(pmax[emax], first_positive_indices(ep["id"], t, m, ep["count"], 1.)[emax]),
                      f"First binary positive differs from T_max in {week}")
            configurations.append((f"binary_{c:g}", p.astype(np.int8), p[first].astype(np.int8),
                                   emax.astype(np.int8), pmax))
        for scheme, obs_class, first_class, max_class, max_index in configurations:
            for strat in STRATA:
                n_strata = 4 if strat == "instant" else 3
                obs_s = inst if strat == "instant" else obs_h
                start_s = inst[first] if strat == "instant" else ep_h_start
                max_s = inst[max_index] if strat == "instant" else hour_stratum(t[max_index])
                labels = COLORS if scheme == "four_class" else ("negative", "positive")
                for cls_i, cls in enumerate(labels):
                    obs_sel = obs_class == cls_i
                    first_sel = first_class == cls_i
                    max_sel = max_class == cls_i
                    for step, idx, strata, outcome, wt, ids in (
                        ("0", np.flatnonzero(obs_sel), obs_s[obs_sel], y[obs_sel],
                         np.ones(obs_sel.sum()), e_id[obs_sel]),
                        ("0b", np.flatnonzero(obs_sel), obs_s[obs_sel], y[obs_sel],
                         1/ep["size"][ep["id"][obs_sel]], e_id[obs_sel]),
                        ("1", first[first_sel], start_s[first_sel], y[first[first_sel]],
                         np.ones(first_sel.sum()), e_id[first[first_sel]]),
                        ("2", first[max_sel], start_s[max_sel], y[first[max_sel]],
                         np.ones(max_sel.sum()), e_id[first[max_sel]]),
                        ("3", max_index[max_sel], max_s[max_sel], y[max_index[max_sel]],
                         np.ones(max_sel.sum()), e_id[max_index[max_sel]]),
                        ("2_any", first[max_sel], start_s[max_sel], ep["any_y"][max_sel],
                         np.ones(max_sel.sum()), e_id[first[max_sel]])):
                        key = (scheme, strat, step, cls)
                        record(key, n_strata)
                        underlying = (int(ep["size"][ep["id"][idx]].sum())
                                      if step in ("1", "2", "3", "2_any") else len(idx))
                        add(acc, key, day[idx], strata, outcome, wt, ids,
                            underlying, n_days, n_strata)
        global_episode_offset += ep["count"]
        print(f"Ladder: {week}: {len(t):,} observations, {ep['count']:,} episodes", flush=True)
    # Point and paired-bootstrap lifts share the same day weights throughout.
    rows, boot_lifts = [], {}
    for (scheme, strat, step, cls), rec in acc.items():
        base_name = "verde" if scheme == "four_class" else "negative"
        base = acc[(scheme, strat, step, base_name)]
        n, z = rec["den"], rec["num"]
        nb, zb = base["den"], base["num"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.divide(zb.sum(0), nb.sum(0), out=np.zeros(nb.shape[1]), where=nb.sum(0) > 0)
            lift = z.sum() / (n.sum(0)*rb).sum()
            n_boot, z_boot = weights @ n, weights @ z
            nb_boot, zb_boot = weights @ nb, weights @ zb
            rb_boot = np.divide(zb_boot, nb_boot, out=np.zeros_like(zb_boot), where=nb_boot > 0)
            bs = z_boot.sum(1) / (n_boot*rb_boot).sum(1)
        check(np.isfinite(bs).all(), f"Non-finite bootstrap lift: {scheme}/{strat}/{step}/{cls}")
        boot_lifts[(scheme, strat, step, cls)] = bs
        rows.append(dict(step=step, scheme=scheme, stratifier=strat, **{"class": cls},
                         n_obs=rec["n_obs"], n_episodes=rec["n_episodes"],
                         underlying_obs=rec["underlying_obs"],
                         weight_sum=n.sum(), positives=z.sum(), positive_rows=rec["positive_rows"],
                         rate=z.sum()/n.sum(), lift=lift,
                         lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    rates = pd.DataFrame(rows)
    if frozen_test:
        reference = pd.read_csv(ROOT / "results/benchmarks_units/table6_reproduction.csv")
        for color in COLORS:
            for step, window in (("0", "pair_instant"), ("2", "episode_start"), ("2_any", "episode_any")):
                now = rates.loc[rates.scheme.eq("four_class") & rates.stratifier.eq("instant") &
                                rates.step.eq(step) & rates["class"].eq(color)].iloc[0]
                old = reference.loc[reference.window.eq(window) & reference.color.eq(color)].iloc[0]
                if step == "0" or color != "verde":
                    check(int(now.n_obs) == int(old.n) and int(now.positive_rows) == int(old.positive),
                          f"Ladder {step}/{color} does not retain Table 6's observed numerator/denominator")
                if step == "0":
                    check(round(now.lift, 2) == round(old.lift, 2),
                          f"Pair-instant lift differs from Table 6: {color}")
    for strat in STRATA:
        for scheme in (("four_class", "binary_1") if args.main_schemes else ("four_class", "binary_1", "binary_1.5", "binary_2")):
            x = rates.loc[rates.stratifier.eq(strat) & rates.scheme.eq(scheme)]
            for step in ("0b", "2", "3"):
                part = x.loc[x.step.eq(step)]
                check(abs(part.weight_sum.sum()-global_episode_offset) < 1e-6,
                      f"Episode weight total differs in {scheme}/{strat}/{step}")
            if frozen_test:
                check(int(x.loc[x.step.eq("2"), "underlying_obs"].sum()) == 6856880,
                      f"Episode population differs in {scheme}/{strat}")
    if frozen_test:
        red_now = rates.loc[rates.scheme.eq("four_class") & rates.stratifier.eq("instant") &
                            rates.step.eq("2") & rates["class"].eq("rojo")].iloc[0]
        red_old = reference.loc[reference.window.eq("episode_start") & reference.color.eq("rojo")].iloc[0]
        diagnostics["red_step2_vs_table6"] = {"common_lift": float(red_now.lift),
                                               "published_lift": float(red_old.lift),
                                               "difference": float(red_now.lift-red_old.lift)}
    comparisons = []
    for scheme in (("four_class", "binary_1") if args.main_schemes else ("four_class", "binary_1", "binary_1.5", "binary_2")):
        labels = COLORS[1:] if scheme == "four_class" else ("positive",)
        for strat in STRATA:
            for a, b in (("0", "0b"), ("0", "1"), ("0", "2"), ("2", "3"), ("0", "3"), ("0b", "1"), ("1", "2"), ("0", "2_any")):
                for cls in labels:
                    A = rates.loc[rates.scheme.eq(scheme) & rates.stratifier.eq(strat) &
                                  rates.step.eq(a) & rates["class"].eq(cls)].iloc[0]
                    B_ = rates.loc[rates.scheme.eq(scheme) & rates.stratifier.eq(strat) &
                                   rates.step.eq(b) & rates["class"].eq(cls)].iloc[0]
                    ba = boot_lifts[(scheme, strat, a, cls)]
                    bb = boot_lifts[(scheme, strat, b, cls)]
                    ratio = bb/ba
                    logdiff = np.log(bb)-np.log(ba)
                    comparisons.append(dict(comparison=f"{b}_vs_{a}", scheme=scheme,
                                            stratifier=strat, **{"class": cls},
                                            ratio=B_.lift/A.lift, lo=np.quantile(ratio, .025),
                                            hi=np.quantile(ratio, .975),
                                            logdiff=np.log(B_.lift)-np.log(A.lift),
                                            logdiff_lo=np.quantile(logdiff, .025),
                                            logdiff_hi=np.quantile(logdiff, .975)))
    ratios = pd.DataFrame(comparisons)
    if not frozen_test:
        rates["gap"], rates["weeks"] = GAP, ";".join(weeks)
        ratios["gap"], ratios["weeks"] = GAP, ";".join(weeks)
    OUT.mkdir(parents=True, exist_ok=True)
    rates.to_csv(OUT / "ladder_rates_lifts.csv", index=False)
    ratios.to_csv(OUT / "ladder_ratios.csv", index=False)
    if not args.instant_only:
        write_tex(rates)
    diagnostics["elapsed_seconds"] = round(time.perf_counter()-started, 1)
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(rates.loc[rates.scheme.eq("four_class") & rates.stratifier.eq("instant") &
                    rates["class"].eq("rojo"), ["step", "n_obs", "n_episodes", "rate", "lift", "lo", "hi"]].to_string(index=False))
    print(f"Runtime: {diagnostics['elapsed_seconds']} s")


if __name__ == "__main__":
    main()

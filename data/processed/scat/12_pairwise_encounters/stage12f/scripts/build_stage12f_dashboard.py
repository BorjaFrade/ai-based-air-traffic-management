"""Build the updated Stage 12 dashboard from Stage 12F diagnostics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2]
TABLES = OUT / "stage12f" / "tables"
DEST = OUT / "figures" / "STAGE12_DASHBOARD.png"
WEEKS = ["20161015_20161021", "20161112_20161118"]
COLORS = ["#155e75", "#a21caf"]


def main():
    repro = pd.read_csv(TABLES / "ward_k2_reproduction.csv")
    sens = pd.read_csv(TABLES / "ward_feature_scaling_sensitivity.csv")
    gmm = pd.read_csv(TABLES / "flight_disjoint_gmm_k1_k12.csv")
    corr = pd.read_csv(TABLES / "duration_sampling_correlations.csv")
    censor = pd.read_csv(TABLES / "censoring_effect_sizes.csv")
    dep = pd.read_csv(TABLES / "flight_dependence.csv")
    flight = pd.read_csv(TABLES / "flight_disjoint_ward_validation.csv")
    boundary_counts = censor[censor.metric.eq("n_samples")].groupby("week").first()

    fig = plt.figure(figsize=(17, 10), facecolor="#f5f8fb")
    grid = fig.add_gridspec(3, 4, height_ratios=[.85, 2, 2], hspace=.48, wspace=.32,
                           top=.85, bottom=.11, left=.05, right=.97)
    fig.text(.04, .96, "SCAT STAGE 12  |  Robustez de estructura", fontsize=23,
             color="#153d59", weight="bold")
    fig.text(.04, .922, "Stage 12F · cohortes completas de estructura · análisis físico sin ATC ni colores",
             fontsize=11, color="#4c6271")

    cards = [
        ("CRUCES 100 NM", " / ".join(f"{100*boundary_counts.loc[w, 'bracketed_n']/(boundary_counts.loc[w, 'bracketed_n']+boundary_counts.loc[w, 'censored_n']):.1f}%" for w in WEEKS), "episodios acotados en ambas semanas"),
        ("VUELOS REPETIDOS", f"{100*dep.endpoint_observation_share_from_repeated_flights.iloc[0]:.1f}% / {100*dep.endpoint_observation_share_from_repeated_flights.iloc[1]:.1f}%", "observaciones A/B ligadas a vuelo repetido"),
        ("WARD K=2", f"{repro.silhouette.iloc[0]:.3f} / {repro.silhouette.iloc[1]:.3f}", f"minoría {100*repro.minority_share.iloc[0]:.1f}% / {100*repro.minority_share.iloc[1]:.1f}%"),
        ("WARD POR VUELO", f"{flight.groupby('week').evaluation_nearest_centroid_silhouette.median().iloc[0]:.3f} / {flight.groupby('week').evaluation_nearest_centroid_silhouette.median().iloc[1]:.3f}", "silhouette mediana de 3 cortes sin fuga de vuelo"),
    ]
    for i, (title, value, sub) in enumerate(cards):
        ax = fig.add_subplot(grid[0, i]); ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color("#cbd5e1")
        ax.text(.05, .78, title, transform=ax.transAxes, color="#526b7b", fontsize=9, weight="bold")
        ax.text(.05, .40, value, transform=ax.transAxes, color="#155e75", fontsize=19, weight="bold")
        ax.text(.05, .12, sub, transform=ax.transAxes, color="#526b7b", fontsize=8)

    ax = fig.add_subplot(grid[1, :2])
    for week, label, color in zip(WEEKS, ["Pilot", "Replica"], COLORS):
        q = sens[(sens.week == week) & (sens.scaling == "robust")]
        ax.plot(["Full", "Solo\ngeometría", "Geometría\n+ tasas"], q.silhouette, "o-", label=label, color=color, lw=2)
    ax.set_ylim(0, .72); ax.set_ylabel("Silhouette Ward K=2")
    ax.set_title("La partición depende de variables")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[1, 2:])
    for week, label, color in zip(WEEKS, ["Pilot", "Replica"], COLORS):
        q = gmm[gmm.week == week]
        ax.plot(q.k, q.evaluation_mean_log_likelihood, "o-", label=label, color=color, lw=2)
    ax.set_xlabel("K del GMM diagonal"); ax.set_ylabel("Log-verosimilitud media fuera de muestra")
    ax.set_title("GMM: mejora hasta K=12 en el corte agrupado")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, :2])
    focus = corr[corr.feature.isin(["track_change_deg", "altitude_change_ft", "track_rate_deg_min", "altitude_rate_ft_min"])]
    short = {"track_change_deg": "rumbo acumulado", "altitude_change_ft": "altitud acumulada",
             "track_rate_deg_min": "tasa de rumbo", "altitude_rate_ft_min": "tasa vertical"}
    width = .36
    for j, (week, label, color) in enumerate(zip(WEEKS, ["Pilot", "Replica"], COLORS)):
        q = focus[focus.week == week]
        ax.bar([i + (j-.5)*width for i in range(4)],
               [q.loc[q.feature.eq(f), "spearman_rho"].iloc[0] for f in short],
               width=width, label=label, color=color)
    ax.set_xticks(range(4), [short[x] for x in short], rotation=15, ha="right")
    ax.axhline(0, color="#526b7b", lw=.8); ax.set_ylabel("Spearman con duración")
    ax.set_title("La normalización reduce la asociación con duración")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, 2:])
    focus = censor[censor.metric.isin(["observed_duration_s", "track_change_deg", "altitude_change_ft", "vertical_at_min_horizontal_ft"])]
    labels = {"observed_duration_s": "duración", "track_change_deg": "cambio rumbo",
              "altitude_change_ft": "cambio altitud", "vertical_at_min_horizontal_ft": "V simultánea"}
    for j, (week, label, color) in enumerate(zip(WEEKS, ["Pilot", "Replica"], COLORS)):
        q = focus[focus.week == week]
        ax.barh([i + (j-.5)*width for i in range(4)],
                [q.loc[q.metric.eq(f), "standardized_mean_difference"].iloc[0] for f in labels],
                height=width, label=label, color=color)
    ax.set_yticks(range(4), [labels[x] for x in labels]); ax.axvline(0, color="#526b7b", lw=.8)
    ax.set_xlabel("SMD: completos menos censurados"); ax.set_title("La selección por censura cambia los perfiles")
    ax.legend(frameon=False)

    fig.text(.04, .035, "Decisión Stage 12F: HYBRID_INTERACTION_STRUCTURE. Cola pequeña con más cambio de rumbo, "
             "pero la frontera de clusters depende del escalado; sin etiquetas de conflicto ni severidad.",
             fontsize=11, color="#153d59", weight="bold")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(DEST)


if __name__ == "__main__":
    main()

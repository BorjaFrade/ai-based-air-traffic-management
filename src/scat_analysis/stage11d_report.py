"""Scientific interpretation of saved Stage 11D tables and figures."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from stage11d_core import OUT, PILOT, REPLICATION, CRITERIA, PARAMS


def t(week,name):return pd.read_csv(OUT/"tables"/f"{week}_{name}.csv")
def fmt(x,d=1):return f"{x:,.{d}f}".replace(",","X").replace(".",",").replace("X",".")
def integer(x):return fmt(x,0)
def row(d,x,value):
    a=d.loc[d[x].eq(value)]
    return a.iloc[0] if len(a) else None
def fig(n,folder,slug,caption):return f"![Figura {n}. {caption}](../figures/{folder}/fig{n:02d}_{slug}.png)"
def ratio(d,criterion):
    s=d.loc[d.criterion.eq(criterion)]
    a=row(s,"h_bin_nm",6);b=row(s,"h_bin_nm",24)
    if a is None or b is None or not (a.supported and b.supported):return "datos no apoyados en ambos bins"
    return f"{fmt(a.rate_per_1000h)} frente a {fmt(b.rate_per_1000h)} eventos/1.000 h (razón {fmt(a.rate_per_1000h/b.rate_per_1000h,2)})"


def build():
    cov1=json.loads((OUT/"diagnostics"/f"{PILOT}_coverage.json").read_text())
    cov2=json.loads((OUT/"diagnostics"/f"{REPLICATION}_coverage.json").read_text())
    audit=json.loads((OUT/"diagnostics"/"11c_episode_audit.json").read_text())
    sel1=t(PILOT,"selection_coverage");sel2=t(REPLICATION,"selection_coverage")
    h1=t(PILOT,"horizontal_rates");h2=t(REPLICATION,"horizontal_rates")
    v1=t(PILOT,"vertical_rates");v2=t(REPLICATION,"vertical_rates")
    phase=t(PILOT,"periodicity_decomposition");phase2=t(REPLICATION,"periodicity_decomposition")
    groups=t(PILOT,"pair_group_summary");groups2=t(REPLICATION,"pair_group_summary")
    c=pd.read_csv(OUT/"tables"/"neighbor_concordance.csv")
    rec=t(PILOT,"recurrence_windows");episodes=t(PILOT,"clearance_episode_sensitivity")
    censor=t(PILOT,"binary_horizon_censoring").iloc[0]
    flights=t(PILOT,"events_per_flight");e=pd.read_parquet(OUT/"tables"/f"{PILOT}_eligible_events.parquet")
    seg=pd.read_csv(OUT/"tables"/"11c_segment_gap_sensitivity.csv")
    d=pd.read_parquet(OUT/"tables"/"pilot_nearest_vs_future_cpa.parquet",
        columns=["same_neighbor","delta_horizontal_nm","delta_cpa_nm","event_count"])
    changed=d.loc[~d.same_neighbor]
    first=sel1.set_index("criterion");second=sel2.set_index("criterion")
    both=groups.loc[groups.pair_group.eq("BOTH_LEVEL")].iloc[0]
    level=phase.loc[phase.motion_group.eq("BOTH_LEVEL")]
    near=level.loc[level.phase_group.eq("NEAR_1000_LEVEL")].iloc[0]
    offset=level.loc[level.phase_group.eq("NEAR_500_OFFSET")].iloc[0]
    near_all=phase.loc[phase.phase_group.eq("NEAR_1000_LEVEL")]
    ph2=phase2.loc[phase2.phase_group.eq("NEAR_1000_LEVEL")]
    week_h=[]
    for week,d0 in ((PILOT,h1),(REPLICATION,h2)):
        near6=row(d0.loc[d0.criterion.eq("NEAREST_HORIZONTAL")],"h_bin_nm",6)
        far24=row(d0.loc[d0.criterion.eq("NEAREST_HORIZONTAL")],"h_bin_nm",24)
        week_h.append((week,near6,far24))
    corr=[]
    for name,a,b,key in (("horizontal",h1,h2,"h_bin_nm"),("vertical",v1,v2,"v_bin_ft")):
        a=a.loc[a.criterion.eq("NEAREST_HORIZONTAL") & a.supported,[key,"rate_per_1000h"]]
        b=b.loc[b.criterion.eq("NEAREST_HORIZONTAL") & b.supported,[key,"rate_per_1000h"]]
        merged=a.merge(b,on=key,suffixes=("_pilot","_rep"))
        corr.append((name,len(merged),float(merged.rate_per_1000h_pilot.corr(merged.rate_per_1000h_rep))))
    types=e.clearance_type.value_counts()
    epi120=row(episodes,"gap_seconds",120)
    epi300=row(episodes,"gap_seconds",300)
    within120=row(rec,"window_seconds",120)
    lines=[]
    def add(s=""):lines.append(s)
    add("# STAGE 11D — Validación de patrones de intervención ATC y geometría de encuentros")
    add("## 1. Resumen ejecutivo")
    add(f"En la semana piloto {PILOT} se observaron {integer(cov1['risk_ticks'])} instantes focales de crucero, {fmt(cov1['focal_hours'])} aeronaves-hora y {integer(cov1['eligible_events'])} autorizaciones elegibles. La segunda semana {REPLICATION} aportó {integer(cov2['risk_ticks'])} instantes, {fmt(cov2['focal_hours'])} aeronaves-hora y {integer(cov2['eligible_events'])} autorizaciones. Todos los criterios de vecino comparten exactamente los instantes focales y los eventos de su semana. Las tasas describen intensidad de autorizaciones ATC, no probabilidad de conflicto ni severidad física validada.")
    add("El resultado principal de H1 es que la concentración de observaciones en múltiplos de 1.000 ft refleja fuertemente la estructura de niveles: en parejas niveladas casi toda la exposición vertical de 0–6.000 ft cae cerca de esos múltiplos. La tasa normalizada suele ser menor allí que en separaciones intermedias. Por tanto, los máximos de los histogramas de eventos no deben confundirse con máximos de riesgo. H2 muestra que la identidad de la pareja cambia mucho con el criterio físico, aunque la asociación descriptiva con distancia horizontal corta persiste en el criterio original. La recurrencia reduce el conteo al formar episodios; la segunda semana sirve como réplica descriptiva, no como validación temporal definitiva.")
    add("![Dashboard de cinco paneles](../figures/stage11d_five_panel_dashboard.png)")
    add("## 2. Hipótesis y expectativas")
    add("H1: si la periodicidad se origina en niveles de vuelo y operaciones verticales, la exposición y los eventos deben concentrarse en separaciones cercanas a múltiplos de 1.000 ft, principalmente cuando ambas aeronaves están niveladas; al estratificar, la forma de la tasa debe cambiar. H2: el vecino horizontal más cercano puede diferir de la pareja que converge o alcanza el menor CPA futuro. Esperábamos concordancia incompleta y cambios en la geometría, pero no presumimos que la pareja con mayor asociación ATC sea un detector de conflicto superior.")
    add("## 3. Datos, población y decisiones fijadas antes de la réplica")
    add("Se reutilizaron los estados nativos de Stage 02, las autorizaciones de Stage 03 y la población 11B congelada para la semana piloto. Stage 11C solo se leyó para auditoría. El código no escribe en esos directorios. El instante focal es una cuadrícula de 30 s; se asigna el estado nativo más reciente que no sea posterior al instante y que tenga edad ≤15 s. El foco está entre FL290 y FL410; los vecinos pueden estar en cualquier altitud válida. Una autorización pertenece a (tick, tick+30 s]. La exposición es 30/3.600 h por instante focal, también cuando no hay vecino válido. El mismo universo de candidatos de ≤100 NM alimenta los cinco criterios. CPA futuro usa 1 s a 30 min; 3D convierte pies a NM con 6.076,12 ft/NM. La velocidad vertical preevento define nivelado si |V/S|≤200 ft/min, con sensibilidad 100 y 300. Las curvas usan bins de 2 NM y 250 ft; los mapas de visualización agregan a 5 NM y 1.000 ft, conservando las tablas finas. Los parámetros del cálculo se almacenaron antes de leer la segunda semana.")
    add(f"La selección de estados nativos produjo {integer(cov1['snapshots'])} instantáneas de todas las altitudes en la semana piloto y {integer(cov2['snapshots'])} en la réplica. Los eventos se asignaron solo si existía un foco elegible. Hubo {integer(cov1['multiple_events_same_tick'])} autorizaciones adicionales dentro de ticks que ya tenían otra autorización en el piloto; la tasa de eventos y la fracción de ticks con evento no son la misma magnitud.")
    add("## 4. Auditoría de cobertura y calidad")
    for criterion in CRITERIA:
        a=first.loc[criterion];b=second.loc[criterion]
        add(f"{criterion}: seleccionó vecino en {integer(a.selected)}/{integer(a.snapshots)} ticks del piloto ({fmt(100*a.selected/a.snapshots)} %) y {integer(b.selected)}/{integer(b.snapshots)} de la réplica ({fmt(100*b.selected/b.snapshots)} %).")
    add("Los registros UNKNOWN permanecen en la población elegible y no entran en un bin geométrico. Cada criterio usa su propio vecino tanto en el numerador como en la exposición del mismo bin. Los intervalos de las curvas son una envolvente conservadora entre Garwood para conteos y un error de influencia agrupado por vuelo focal; las celdas de mapas usan Garwood y se dibujan solo con ≥10 h, ≥3 eventos y ≥20 vuelos. Los intervalos no corrigen la selección de semana ni todos los factores de confusión.")
    add("## 5. H1 — Periodicidad vertical y estado preevento")
    add(f"Con tolerancia de 200 ft/min, las parejas ambas niveladas aportaron {integer(both.snapshots)} instantáneas, {fmt(both.hours)} h y {integer(both.events)} eventos. La tasa global fue {fmt(both.rate_per_1000h)} eventos/1.000 h. Las siete combinaciones se conservan en `tables/{PILOT}_pair_group_summary.csv`; ningún grupo se fusionó en ese cálculo. Para visualizar se muestran además BOTH_LEVEL, MOVING y UNKNOWN. Los grupos de descenso y movimientos opuestos tienen tasas más altas, pero pueden mezclar contextos operativos distintos y no prueban proximidad peligrosa.")
    add(fig(1,"01_vertical_periodicity","vertical_motion","Tasa por estado vertical de la pareja"))
    add(fig(2,"01_vertical_periodicity","types_exposure","Autorizaciones por tipo y exposición vertical"))
    add(f"En 0–6.000 ft, la categoría BOTH_LEVEL cerca de ±125 ft de un múltiplo de 1.000 reúne {fmt(100*near.exposure_hours/level.exposure_hours.sum())} % de su exposición y {fmt(100*near.events/level.events.sum())} % de sus eventos. En el offset de 500 ft solo hubo {fmt(offset.exposure_hours)} h y {int(offset.events)} eventos; esa comparación de tasas no está apoyada. Al incluir parejas en movimiento, las separaciones fuera de niveles discretos tienen exposición y eventos suficientes. En la semana piloto, {fmt(100*near_all.exposure_hours.sum()/phase.exposure_hours.sum())} % de la exposición 0–6.000 ft se encuentra cerca de múltiplos de 1.000 ft; en la réplica, {fmt(100*ph2.exposure_hours.sum()/phase2.exposure_hours.sum())} %.")
    add("El tipo LEVEL predomina en los eventos elegibles ("+fmt(100*types.get("LEVEL",0)/len(e))+" %); HEADING y SPEED representan menos eventos. No se usó ese tipo futuro para clasificar el estado de vuelo. En ambas aeronaves niveladas, LEVEL aparece sobre todo en bins de 0, 1.000 y 2.000 ft, pero la exposición en bins intermedios es demasiado pequeña para comparar tasas con confianza. En parejas con movimiento, la tasa de la fase cercana a 1.000 ft es similar a la de la fase desplazada 500 ft. La periodicidad está en ambos componentes, eventos y exposición, pero la masa de exposición en niveles discretos es decisiva: el bin 1.000–1.250 ft tiene una tasa inferior a bins intermedios próximos. La frase 'pico de intervenciones' se refiere al histograma bruto y no necesariamente a la tasa normalizada.")
    add(fig(3,"01_vertical_periodicity","both_level_map","Mapa H–V de parejas niveladas"))
    add(fig(4,"01_vertical_periodicity","moving_map","Mapa H–V de parejas con movimiento vertical"))
    add(fig(5,"01_vertical_periodicity","phase_decomposition","Eventos y exposición por fase vertical"))
    add("La sensibilidad a 100/200/300 ft/min cambia poco la proporción BOTH_LEVEL, pero los bins de 100 ft dejan muchas celdas con <10 h. Los bins de 250 y 500 ft reducen esa inestabilidad y también suavizan la forma. Una explicación física plausible es la cuantización de altitudes de crucero y la actividad de cambio de nivel; alternativas son redondeo de altitud, selección de autorización y diferencias en densidad o rutas. Se necesitarían validación de altitud y velocidad vertical, control por región/sector y modelos que ajusten distancia horizontal y fase de vuelo. La periodicidad no identifica conflictos.")
    add("## 6. H2 — Criterios de vecino")
    nearest=c.loc[c.criterion_a.eq("NEAREST_HORIZONTAL")]
    for criterion in CRITERIA[1:]:
        q=nearest.loc[nearest.criterion_b.eq(criterion)].iloc[0]
        add(f"Coincidencia NEAREST_HORIZONTAL con {criterion}: {fmt(100*q.concordance)} % ({integer(q.same)}/{integer(q.both_selected)} ticks con ambas selecciones).")
    add(f"Al sustituirlo por MINIMUM_FUTURE_CPA hubo {integer(len(changed))} ticks con pareja diferente. En esos ticks, el cambio mediano fue +{fmt(changed.delta_horizontal_nm.median())} NM de distancia actual y {fmt(changed.delta_cpa_nm.median())} NM de CPA. La nueva pareja suele estar más lejos en el instante y acercarse más bajo velocidad constante. Los casos extremos auditables se guardan en `examples/cpa_selection_discrepancies.csv`.")
    add(fig(6,"02_neighbour_selection","concordance","Coincidencia entre criterios"))
    add(fig(7,"02_neighbour_selection","horizontal_criteria","Tasas horizontales por criterio"))
    add(fig(8,"02_neighbour_selection","vertical_criteria","Tasas verticales por criterio"))
    add(fig(9,"02_neighbour_selection","rate_map_comparison","Mapas con numerador y exposición por criterio"))
    add(fig(10,"02_neighbour_selection","geometry_change","Cambio de H y CPA al cambiar de pareja"))
    add("Comparación descriptiva 6–8 NM frente a 24–26 NM, cada criterio con su propia pareja y exposición:")
    for criterion in CRITERIA:add(f"{criterion}: {ratio(h1,criterion)}.")
    add("La asociación de 6–8 frente a 24–26 NM no es universal: MAXIMUM_CLOSING invierte el contraste en el piloto y vuelve a favorecer la distancia corta en la réplica. Su forma es inestable. MINIMUM_FUTURE_CPA se define con extrapolación rectilínea durante 30 min y puede favorecer parejas lejanas con trayectorias nominales que cambiarán. MAXIMUM_CLOSING puede elegir distancias actuales muy grandes. NEAREST_VERTICAL con radio amplio puede escoger una aeronave en otro contexto operacional. La distancia 3D es solo comparación de unidades geométricas. La persistencia de una asociación bajo algunas reglas no demuestra causalidad ni seguridad. La futura prueba requiere continuidad de pareja, predicción de maniobras, márgenes verticales y revisión independiente de casos convergentes.")
    add("## 7. Recurrencia de autorizaciones y unidades de conteo")
    add(f"De {integer(len(e))} autorizaciones elegibles, {integer(types.get('LEVEL',0))} fueron LEVEL, {integer(types.get('SPEED',0))} SPEED y {integer(types.get('HEADING',0))} HEADING; hay {integer(len(e)-types.get('LEVEL',0)-types.get('SPEED',0)-types.get('HEADING',0))} tipo combinado. Hubo {integer(len(flights))} vuelos con al menos una autorización elegible; {integer(flights.events.gt(1).sum())} ({fmt(100*flights.events.gt(1).mean())} %) recibieron más de una. {integer(within120.events_with_prior)} eventos ({fmt(100*within120.fraction_with_prior)} %) tenían otra autorización del mismo vuelo en los 120 s anteriores.")
    add(f"Agrupando autorizaciones consecutivas por vuelo con hueco ≤120 s quedan {integer(epi120.episodes)} episodios de actividad ATC frente a {integer(len(e))} eventos (−{fmt(100*(1-epi120.episodes/len(e)))} %). Con 300 s quedan {integer(epi300.episodes)}. Estos grupos no son encuentros ni conflictos. El indicador binario usa un tick focal de 30 s con ≥1 autorización en (tick,tick+30]. Para comparar las tres curvas se exige que tick+30 no supere el último tiempo nativo o de autorización observado en la semana: se excluyeron {integer(censor.right_censored_ticks)} ticks del extremo derecho y {integer(censor.eligible_events_on_right_censored_ticks)} eventos en ellos. Quedan {integer(censor.complete_30s_ticks)} ticks con horizonte completo. Los eventos sin estado preevento elegible están fuera de esta población, y no se extrapola a intervalos sin estado. La salida real de cobertura de cada vuelo dentro de un tick no es identificable aquí y puede dejar censura residual. Las tres curvas tienen unidades diferentes: eventos, comienzos de episodio o ticks positivos por 1.000 aeronaves-hora.")
    add(fig(11,"03_clearance_recurrence","counts_per_flight","Autorizaciones por vuelo"))
    add(fig(12,"03_clearance_recurrence","interclearance_gaps","Intervalos entre autorizaciones"))
    add(fig(13,"03_clearance_recurrence","event_episode_curves","Eventos, episodios y ticks binarios"))
    add("## 8. Reproducibilidad en la segunda semana")
    add(f"La réplica utiliza exclusivamente {REPLICATION}, otra de las semanas disponibles; no se procesaron las otras once. Se aplicaron los mismos filtros, unidades, bins, criterios de vecino, radio y horizonte. Su tasa total elegible fue {fmt(1000*cov2['eligible_events']/cov2['focal_hours'])} eventos/1.000 h, frente a {fmt(1000*cov1['eligible_events']/cov1['focal_hours'])} en el piloto. La diferencia de tasas semanales puede reflejar mezcla de vuelos, sectores y volumen, además de patrones de encuentro.")
    for week,a,b in week_h:
        add(f"{week}: 6–8 NM {fmt(a.rate_per_1000h)} (IC 95 % {fmt(a.ci_low_per_1000h)}–{fmt(a.ci_high_per_1000h)}), 24–26 NM {fmt(b.rate_per_1000h)} (IC {fmt(b.ci_low_per_1000h)}–{fmt(b.ci_high_per_1000h)}); razón {fmt(a.rate_per_1000h/b.rate_per_1000h,2)}.")
    for name,n,r in corr:add(f"Correlación descriptiva de tasas por bin {name} con apoyo en ambas semanas: r={fmt(r,2)} en {n} bins; no equivale a validación externa.")
    add(fig(14,"04_week_reproducibility","horizontal_weeks","Tasas horizontales por semana"))
    add(fig(15,"04_week_reproducibility","vertical_weeks","Tasas verticales por semana"))
    add(fig(16,"04_week_reproducibility","rate_maps_weeks","Mapas comparables por semana"))
    add(fig(17,"04_week_reproducibility","type_level_weeks","Tipos de autorización en parejas niveladas"))
    add(fig(18,"04_week_reproducibility","criteria_weeks","Criterios de vecino en dos semanas"))
    add("La asociación y la periodicidad deben juzgarse con las bandas de incertidumbre y los bins apoyados, no solo por la similitud visual. Dos semanas observacionales no cubren cambios estacionales ni permiten calibrar una tasa de conflicto.")
    add("## 9. Auditoría de episodios de pareja 11C")
    add(f"11C contiene {integer(audit['trace_rows'])} filas de pareja fija en {integer(audit['windows'])} ventanas de ±300 s alrededor de autorizaciones o controles emparejados. Tras ordenar por tiempo, la mediana del hueco es {fmt(audit['gap_median_s'])} s y el percentil 95 es {fmt(audit['gap_p95_s'],2)} s. Hay {integer(audit['gaps_gt30_le60'])} huecos >30 y ≤60 s, frente a {integer(audit['gaps_gt60'])} >60 s. Una pequeña fluctuación alrededor de 30 s rompe un segmento bajo el umbral estricto de 30; 60 s lo reúne. Esta es una causa directa de la caída de conteos, más compatible con fragmentación de muestreo que con nuevos encuentros físicos independientes.")
    for _,x in seg.iterrows():add(f"Hueco máximo {int(x.gap_rule_s)} s: {integer(x.segments)} segmentos; mediana de duración {fmt(x.median_duration_s)} s; {integer(x.single_sample)} segmentos de una sola muestra; {integer(x.any_edge_censored)} tocan límites de ventana.")
    add(f"No hay duplicados por (ventana,bin) ni por (ventana,pareja,tiempo) en la traza ya deduplicada. La pareja es fija dentro de cada ventana por diseño; {integer(audit['focal_with_multiple_selected_neighbors_across_windows'])} focos tienen más de un vecino seleccionado en distintas ventanas. Hay {integer(audit['traces_over_50nm'])} puntos a >50 NM y {integer(audit['traces_over_100nm'])} a >100 NM. El archivo de Stage 11 ya aplicó una selección por bin, de modo que esta auditoría no demuestra ausencia de duplicados en datos crudos anteriores. Los límites ±300 s censuran la duración y la traza condicionada por evento no representa encuentros completos independientes de la autorización.")
    add(fig(19,"05_episode_audit","observed_gaps","Huecos reales de la traza"))
    add(fig(20,"05_episode_audit","segment_duration","Duración y muestras por segmento"))
    add(fig(21,"05_episode_audit","fragmentation_examples","Ejemplos de fragmentación 30→60 s"))
    add("## 10. Discusión científica y pruebas siguientes")
    add("H1, esperado: niveles discretos y cambios de nivel explicarían los máximos de frecuencia. Observado: la mayoría de la exposición de parejas niveladas cae cerca de múltiplos de 1.000 ft, con eventos también concentrados allí; la tasa en esos bins no es necesariamente máxima. Explicación operacional plausible: crucero nivelado y autorizaciones LEVEL. Alternativas: redondeo de altitud, selección de autorización, mezcla de sectores, densidad de rutas y fase de vuelo. Evidencia: descomposición numerador/denominador, estados verticales nativos y réplica. Pendiente: verificar resolución de altitud, continuidad temporal y ajuste multivariable. No se concluye un patrón de riesgo físico.")
    add("H2, esperado: otros criterios seleccionarían parejas distintas. Observado: el CPA futuro y el máximo cierre coinciden poco con el vecino horizontal, y el CPA futuro reduce el CPA predicho al coste de mayor distancia presente. Explicación física plausible: encuentros convergentes no siempre son los más próximos. Alternativas: extrapolación lineal, maniobras ATC y múltiples aeronaves en el mismo espacio. Evidencia: identidad y cambios geométricos por tick con un universo candidato común; curvas con numerador y denominador reconstruidos. Pendiente: validación de trayectoria posterior a la predicción y juicio independiente de encuentros. No se elige el criterio con mayor asociación ATC como 'mejor detector'.")
    add("La recurrencia muestra dependencia entre eventos del mismo vuelo. Los intervalos por vuelo focal de las curvas atienden parte de esa dependencia; el conteo de episodios reduce repetición, pero su hueco es una decisión operacional y no define conflictos. Las curvas de ticks binarios son probabilidades de evento por intervalo observado solo si se divide por ticks, sin extrapolación a áreas sin cobertura.")
    add("## 11. Decisión A–E y severidad")
    add("A. Sí, la estructura de niveles explica gran parte de la periodicidad de las frecuencias; LEVEL domina los eventos. No está demostrado que explique por sí sola toda la forma de la tasa ni un riesgo mayor cerca de los niveles.")
    add("B. Para el vecino más cercano, el contraste 6–8 frente a 24–26 NM permanece en las dos semanas; también aparece con CPA futuro, aunque más atenuado. MAXIMUM_CLOSING no lo reproduce consistentemente. La pareja convergente puede ser otra aeronave. Ninguna regla quedó validada como etiqueta de conflicto.")
    add(f"C. Sí cambia el nivel absoluto: agrupar a 120 s reduce {len(e):,} eventos a {int(epi120.episodes):,} comienzos de episodio. La comparación de forma se muestra en la figura 13 y no autoriza tratarlos como la misma medida.")
    add("D. Los rasgos principales se contrastaron en una segunda semana con parámetros idénticos. La semejanza o diferencia observada queda condicionada por dos semanas y sus IC; no es validación temporal definitiva.")
    add("E. Hay señales compatibles con estructura operacional estable y con parejas convergentes distintas del vecino más cercano. La evidencia sigue siendo insuficiente para derivar familias naturales de conflicto o una escala de severidad física.")
    add("Recomendación: mantener GREEN–YELLOW–ORANGE–RED solo como estados operacionales del detector existente; no modificar sus umbrales ahora. Investigar una representación continua de severidad física y trayectorias de encuentros en trabajo posterior, con etiquetas independientes de autorizaciones ATC. Explorar familias temporales solo tras reconstruir encuentros completos fuera de ventanas condicionadas por evento. Aplazar cualquier escala de riesgo hasta disponer de evidencia de separación futura real, consecuencias y validación externa.")
    add("## 12. Limitaciones")
    add("El dataset registra autorizaciones, no conflictos confirmados. Las trayectorias usadas para CPA suponen velocidad y rumbo constantes. Las instantáneas cubren solo estados de edad ≤15 s y los vecinos dentro de 100 NM; UNKNOWN no recibe geometría. Las diferencias de cobertura entre criterios generan poblaciones geométricas condicionadas distintas aunque la población focal sea idéntica. El universo de eventos elegibles omite autorizaciones sin estado de crucero reciente. Los intervalos Garwood suponen conteos Poisson, y la envolvente por clúster focal no captura correlaciones entre vuelos, sectores o semanas. Los mapas contienen celdas escasas; solo las apoyadas son visibles. Los grupos verticales y tipos futuros son descriptivos. La auditoría 11C observa ventanas y una traza ya deduplicada, no encuentros completos. Se analizaron dos semanas y no una partición aleatoria ni validación prospectiva.")
    add("## 13. Conclusiones y reproducibilidad")
    add("La periodicidad bruta está estrechamente ligada a la exposición en niveles discretos; el criterio de vecino cambia la identidad y la geometría con fuerza; agrupar autorizaciones cambia el nivel de intensidad; y la segmentación 11C a 30 s es muy sensible a pequeños huecos de muestreo. Estos hallazgos preparan pruebas físicas más directas sin convertir autorizaciones en clases de conflicto. Las tablas CSV/Parquet de cada figura, parámetros en `diagnostics`, código en `src/scat_analysis/stage11d_*.py`, pruebas sintéticas y manifest permiten repetir el análisis.")
    path=OUT/"reports"/"STAGE11D_VALIDATION_REPORT.md"
    path.write_text("\n\n".join(lines)+"\n",encoding="utf-8")
    print(path,flush=True)


if __name__=="__main__":build()

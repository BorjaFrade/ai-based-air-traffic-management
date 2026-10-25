"""Build the complete evidence-backed Stage 12 Markdown report and manifest."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from stage12_audit_inputs import OUT,WEEKS

TITLE='STAGE12_PAIRWISE_ENCOUNTER_DISCOVERY_REPORT'

def main():
    a,b=WEEKS
    e={w:pd.read_parquet(OUT/'episodes'/f'{w}_episode_summary.parquet') for w in WEEKS}
    s={w:pd.read_csv(OUT/'tables'/f'{w}_radius_gap_sensitivity.csv') for w in WEEKS}
    v={w:pd.read_csv(OUT/'tables'/f'{w}_prediction_error_summary.csv') for w in WEEKS}
    o={w:pd.read_parquet(OUT/'tables'/f'{w}_operational_enrichment.parquet') for w in WEEKS}
    build={w:json.loads((OUT/'diagnostics'/f'{w}_pair_build_audit.json').read_text()) for w in WEEKS}
    structure=json.loads((OUT/'diagnostics'/'structure_discovery.json').read_text())
    temporal=json.loads((OUT/'diagnostics'/'pilot_15s_vs_30s_sensitivity.json').read_text())
    sensitivity=pd.read_csv(OUT/'tables'/'structure_scaling_feature_sensitivity.csv')
    density=pd.read_csv(OUT/'tables'/'structure_density_sensitivity.csv')
    gmm=pd.read_csv(OUT/'tables'/'structure_gmm_metrics.csv')
    cases=pd.read_csv(OUT/'examples'/'case_gallery.csv')
    opcases=pd.read_csv(OUT/'examples'/'operational_case_gallery.csv')
    ctx=pd.read_parquet(OUT/'tables'/'pilot_detector_context.parquet')
    def n(x):return f'{int(x):,}'.replace(',','.')
    def pct(x):return f'{100*x:.1f}'.replace('.',',')+' %'
    def dec(x,places=2):return f'{x:.{places}f}'.replace('.',',')
    def val(w,h,col):return float(v[w].loc[(v[w].horizon_min==h)&v[w].validation_status.eq('VALIDATED'),col].iloc[0])
    def figure(folder,file,caption):return f'![{caption}](../figures/{folder}/{file})\n'
    def ex(name,caption):return f'![{caption}](../examples/{name}.png)\n'
    p=e[a];r=e[b]
    comparison=p[['episode_id','observed_duration_s','start_a_alt_ft','a_track_change_deg','b_track_change_deg',
                  'a_altitude_change_ft','b_altitude_change_ft']].merge(o[a][['episode_id','atc_context']],on='episode_id')
    comparison['max_abs_turn_deg']=np.maximum(comparison.a_track_change_deg.abs(),comparison.b_track_change_deg.abs())
    comparison['max_abs_alt_change_ft']=np.maximum(comparison.a_altitude_change_ft.abs(),comparison.b_altitude_change_ft.abs())
    compared=comparison.groupby('atc_context')[['observed_duration_s','start_a_alt_ft','max_abs_turn_deg','max_abs_alt_change_ft']].median()
    txt=f'''# Stage 12 — Descubrimiento de encuentros entre parejas de aeronaves

**TFM:** Air Traffic Management Using Artificial Intelligence Models. **Datos:** SCAT, semana piloto {a}, réplica {b}. **Versión:** 25 de octubre de 2026. **Decisión:** `CONTINUOUS_INTERACTION_STRUCTURE`, limitada a la interacción observada dentro de 100 NM. Todos los conteos y figuras proceden de los artefactos de esta etapa; no se modificaron las etapas 01–11D.

## 1. Resumen ejecutivo

Se reconstruyeron {n(len(p))} episodios piloto entre {n(p.pair_id.nunique())} parejas distintas y {n(len(r))} episodios réplica entre {n(r.pair_id.nunique())} parejas. Son segmentos de coobservación dentro de 100 NM, no conflictos confirmados. La calidad cinemática mínima para exploración se cumple en {pct(p.quality_for_structure.mean())} y {pct(r.quality_for_structure.mean())}; solo {pct(p.fully_bracketed_at_100nm.mean())} y {pct(r.fully_bracketed_at_100nm.mean())} muestran entrada y salida observadas del radio. La censura temporal, por tanto, condiciona cualquier distribución de duración y cualquier interpretación de familias.

Las distribuciones geométricas y los errores del modelo de velocidad constante se reproducen aproximadamente entre semanas. No aparece una partición discreta estable: el BIC del GMM continúa mejorando hasta K=6, su silhouette baja a {dec(gmm.loc[gmm.k.eq(6),'silhouette'].iloc[0],3)} y HDBSCAN cambia drásticamente con los parámetros. La evidencia respalda un espacio de interacción principalmente continuo, con geometrías descriptivas locales. No valida categorías de conflicto ni una escala de severidad.

![Resumen visual de las dos semanas](../figures/STAGE12_DASHBOARD.png)

## 2. Motivación y relación con Stage 11D

Stage 11D mostró que el vecino horizontal más próximo no representa necesariamente la pareja de menor CPA o mayor cierre, y que el muestreo fragmenta las trazas. Esperábamos que reconstruir identidades A–B durante todo el intervalo permitiera distinguir cruce, seguimiento y maniobra sin partir de una autorización. La prueba aquí conserva todas las parejas simultáneas y permite revisar una misma identidad a lo largo de la trayectoria. La alternativa explicativa más fuerte de muchos segmentos sigue siendo la pérdida de coobservación, no un cambio físico; por ello cada episodio lleva diagnóstico de frontera.

## 3. Preguntas y alcance científico

Se pregunta si existen encuentros observables y geometrías reproducibles; si CPA/TCPA es predictivo a horizontes cortos; si hay familias discretas o un continuo; y qué información añaden después ATC y el detector. El radio de 100 NM selecciona candidatos por eficiencia y visibilidad. No define riesgo. Tampoco se deduce una infracción de separación del par H/V observado: se requieren reglas operacionales, incertidumbre, espacio aéreo y contexto causal.

## 4. Datos, procedencia y cobertura

Stage 02 aporta estados SCAT nativos, identificador de vuelo por archivo, miembro de origen, tiempo, posición, altitud, velocidad, rumbo, velocidad vertical, fase y sector. Stage 03 aporta autorizaciones únicamente para el enriquecimiento posterior; Stage 10B aporta colores focales de ventanas emparejadas del piloto. Las rutas y esquemas reales están inventariados en `diagnostics/input_schema_inventory.json` y `diagnostics/input_coverage_audit.csv`. Las lecturas completas contienen {n(build[a]['native_rows'])} y {n(build[b]['native_rows'])} estados de las dos semanas. Tras sincronizar a 30 s quedan {n(build[a]['snapshots'])} y {n(build[b]['snapshots'])} estados as-of, con {n(build[a]['unique_entities'])} y {n(build[b]['unique_entities'])} identidades. No se detectaron colisiones de identidad en las dos semanas procesadas; el código conserva la procedencia y separa miembros si una colisión aparece en futuros datos. La auditoría de entrada muestreó calidad, mientras los conteos de reconstrucción proceden de la lectura completa.

## 5. Reconstrucción de parejas y eficiencia

En cada instante se usa un índice KD sobre la esfera unitaria para encontrar pares dentro de 150 NM, seguido de distancia de círculo máximo exacta y filtros de 50, 100 y 150 NM. Las claves de pareja se ordenan canónicamente e incluyen las longitudes de ambas identidades para evitar ambigüedad textual. Se guardan {n(build[a]['candidate_rows_within_nm']['150'])} y {n(build[b]['candidate_rows_within_nm']['150'])} observaciones de parejas dentro de 150 NM; dentro de 100 NM quedan {n(build[a]['candidate_rows_within_nm']['100'])} y {n(build[b]['candidate_rows_within_nm']['100'])}. No hubo duplicados A–B/B–A al mismo tick. La búsqueda no aplica un filtro vertical ni usa autorización o color.

## 6. Definición observacional de episodio

Un episodio une muestras consecutivas de la misma pareja dentro de 100 NM con hueco máximo de 60 s y descarta saltos horizontales imposibles superiores a 30 NM por cada 30 s de separación. El tiempo de inicio/fin corresponde a la primera/última muestra del segmento; la duración observada es su diferencia. No se extrapolan posiciones ni se presentan estados interpolados como mediciones. Esta definición es una ventana de interacción, no el inicio causal ni el final de un conflicto. Una pareja puede originar más de un episodio si sale y vuelve a entrar o se pierde la observación.

## 7. Calidad, censura y sensibilidad de reconstrucción

En el piloto {n(p.fully_bracketed_at_100nm.sum())} episodios quedan acotados por cruces observados de 100 NM en ambos extremos; {n(p.temporally_censored.sum())} están censurados temporalmente. En la réplica son {n(r.fully_bracketed_at_100nm.sum())} y {n(r.temporally_censored.sum())}. El test de frontera busca ambos vuelos a ±30 s y, si falta coincidencia, a ±60 s. La censura es alta porque la sincronización simultánea de dos vuelos es exigente; puede acortar encuentros, aunque los saltos físicos imposibles detectados son cero. No se debe llamar encuentro físico completo a los {pct(p.temporally_censored.mean())} segmentos piloto censurados.

Con radio 100 NM, los huecos 45/60/90 s producen {n(s[a].query('radius_nm==100 and gap_s==45').episodes.iloc[0])}/{n(s[a].query('radius_nm==100 and gap_s==60').episodes.iloc[0])}/{n(s[a].query('radius_nm==100 and gap_s==90').episodes.iloc[0])} episodios piloto: la diferencia es pequeña. Cambiar radio a 50 o 150 NM produce {n(s[a].query('radius_nm==50 and gap_s==60').episodes.iloc[0])} o {n(s[a].query('radius_nm==150 and gap_s==60').episodes.iloc[0])}, mostrando que la escala espacial sí define la población observada. En seis horas fijas, la cuadrícula de 15 s produce {n(temporal[0]['episodes_by_gap_s']['60'])} episodios, frente a {n(temporal[1]['episodes_by_gap_s']['60'])} a 30 s; la mediana de intervalo nativo observado es {dec(temporal[0]['median_observed_sample_interval_s'],1)} s. La sensibilidad temporal se limita a ese tramo, no a toda la semana.

{figure('05_duration_persistence','duration_persistence.png','Panel 05. Persistencia, censura y sensibilidad a huecos')}

## 8. Variables físicas por episodio

Se conservan series temporales completas y una fila por episodio. Se calculan H inicial, mínima, máxima y final; V inicial, mínima y final; signo de la diferencia de altitud; tiempo de H mínima; V simultánea en ese instante; tiempo de V mínima y H simultánea allí; velocidades relativas, cierre, ángulo de cruce, fracción convergente, cambios de tendencia, rumbo, velocidad y altitud, fase, sector y vecinos concurrentes. Las unidades son NM, ft, kt, fpm, minutos y segundos según el nombre de columna. Los máximos de vecinos se calculan dentro del radio amplio de 150 NM y por ello son contexto denso, no una alerta.

## 9. CPA/TCPA y validación as-of

El modelo de velocidad constante usa exclusivamente posición, track, velocidad y tasa vertical conocidos al comienzo del episodio. TCPA negativo o menor de 1 s queda inválido como CPA futuro; para cada horizonte el tiempo se limita al intervalo de 2/5/10/15 min y 30 min de sensibilidad. La vertical prevista se evalúa en el instante del CPA horizontal, nunca se equipara automáticamente a la vertical mínima. La validación retrospectiva exige muestra de la misma pareja exactamente al tick futuro dentro del episodio; esa exigencia selecciona episodios persistentes y reduce mucho la cobertura a horizontes largos.

En el piloto, la mediana del error absoluto horizontal final de predicciones evaluables crece de {dec(val(a,2,'median_abs_horizontal_end_error_nm'))} NM a 2 min, a {dec(val(a,5,'median_abs_horizontal_end_error_nm'))} NM a 5 min y {dec(val(a,15,'median_abs_horizontal_end_error_nm'))} NM a 15 min. En la réplica los valores son {dec(val(b,2,'median_abs_horizontal_end_error_nm'))}, {dec(val(b,5,'median_abs_horizontal_end_error_nm'))} y {dec(val(b,15,'median_abs_horizontal_end_error_nm'))} NM. El error vertical mediano piloto pasa de {n(val(a,2,'median_abs_vertical_end_error_ft'))} ft a {n(val(a,15,'median_abs_vertical_end_error_ft'))} ft. El crecimiento es compatible con maniobras y con una velocidad vertical instantánea inadecuada para extrapolar 15 min; también podría reflejar errores del estado SCAT. A 30 min solo {n(v[a].query('horizon_min==30 and validation_status=="VALIDATED"').n.iloc[0])} episodios piloto permiten esta validación, con error horizontal mediano {dec(val(a,30,'median_abs_horizontal_end_error_nm'))} NM: no se trata como predicción fiable por defecto.

{figure('04_cpa_tcpa','cpa_prediction_validation.png','Panel 04. Error predictivo, cobertura y CPA/TCPA')}

## 10. Atlas de trayectorias y perfiles temporales

El atlas utiliza la misma identidad A–B en todos los subgráficos. Se seleccionaron ejemplos mediante criterios físicos explícitos de aproximación, divergencia, cruce, paralelo, cambio vertical, giro, duración y entorno multiaeronave. Algunos ejemplos de convergencia o paralelo están censurados: se muestran como morfologías parciales. En las trayectorias, el círculo marca comienzo, el cuadrado fin y la X la separación horizontal mínima. La expectativa de cruces claros aparece en algunos casos; alternativas como cambio de nivel o seguimiento paralelo generan perfiles distintos con separaciones horizontales parecidas. Una galería visual verifica directamente continuidad y el orden temporal, pero no valida reglas operativas de separación.

{figure('01_trajectories','trajectory_atlas.png','Panel 01. Atlas de trayectorias de parejas')}

{figure('02_temporal_profiles','temporal_profiles.png','Panel 02. Evolución temporal de separación, cierre, ángulo y TCPA')}

## 11. Distribuciones geométricas

La mediana de H mínima es {dec(p.min_horizontal_nm.median())} NM en el piloto y {dec(r.min_horizontal_nm.median())} NM en la réplica; la V simultánea mediana es {n(p.vertical_at_min_horizontal_ft.median())} y {n(r.vertical_at_min_horizontal_ft.median())} ft. Hay {n(((p.min_horizontal_nm<5)&(p.vertical_at_min_horizontal_ft<1000)).sum())} episodios piloto con H mínima <5 NM y V simultánea <1.000 ft, frente a {n(((p.min_horizontal_nm<5)&(p.min_vertical_ft<1000)).sum())} si se combinan mínimos independientes ocurridos en momentos diferentes. Esa diferencia revela por qué la simultaneidad es esencial. Las bandas verticales cercanas a niveles de vuelo pueden explicar picos; Stage 11D ya vinculó esa periodicidad con altitudes discretas. Los ejemplos de proximidad no equivalen a infracción ni a riesgo validado.

{figure('03_geometry_distributions','geometry_distributions.png','Panel 03. Distribuciones y mapa de calor de H mínima frente a V simultánea')}

## 12. Duración, persistencia y encuentros simultáneos

La mediana de duración observada es {n(p.observed_duration_s.median())} s en ambas semanas. La persistencia convergente se calcula como fracción de muestras con cierre positivo y tiempo observado convergente; los cambios de tendencia se cuentan dentro de la misma pareja. Las formas de duración muestran tramos cortos y prolongados, pero una gran parte de los extremos carece de coobservación, así que la cola no estima la distribución física completa. Se conservan todas las parejas cercanas simultáneas y se evita elegir un único vecino focal. El número de vecinos dentro de 150 NM es alto para la mayoría de los vuelos; separar contexto de densidad local de riesgo exige análisis posterior.

## 13. Estructura discreta frente a continua

Se eligió una muestra reproducible de {n(structure['pilot_n'])} episodios piloto y {n(structure['replica_n'])} de réplica con cuatro muestras o más, duración ≥90 s, edad ≤15 s y ambas fronteras de 100 NM observadas. Se limitaron dos episodios por pareja y se fijó la semilla {structure['seed']}. Ocho variables físicas excluyen duración, número de observaciones, ATC y color: log H mínima, log V simultánea, cierre inicial, ángulo, fracción convergente, velocidad relativa, cambio máximo de rumbo y altitud. Se escala robustamente y se limita el efecto de valores extremos. PCA explica {pct(sum(structure['pca_variance_ratio'][:2]))} en los dos primeros ejes; la proyección no prueba separación.

El GMM diagonal mejora BIC de K=1 a K=6, sin óptimo interior en el rango probado. La silhouette a K=2 es {dec(gmm.loc[gmm.k.eq(2),'silhouette'].iloc[0],3)} y a K=6 {dec(gmm.loc[gmm.k.eq(6),'silhouette'].iloc[0],3)}: los componentes se solapan. El remuestreo dentro del piloto arroja ARI {', '.join(dec(x,2) for x in structure['gmm_resample_ari'])}; sin embargo, reajustar seis componentes en la réplica da ARI {dec(sensitivity.query('feature_set=="full" and scaling=="robust"').replica_refit_ari.iloc[0],2)} frente al modelo piloto. Quitar variables dinámicas reduce la concordancia con la solución completa a {dec(sensitivity.query('feature_set=="geometry_only" and scaling=="robust"').agreement_with_full_robust_on_replica.iloc[0],2)}. HDBSCAN varía de cero a dos grupos y su cobertura varía entre {pct(density.coverage.min())} y {pct(density.coverage.max())} según densidad mínima. Por ello los componentes son una aproximación estadística a distribuciones de geometría y maniobra, no especies físicas discretas. Un posible mecanismo de aparente separación es la cuantización de altitudes; otra alternativa es la mezcla de fases de vuelo.

{figure('07_structure_discovery','structure_discovery.png','PCA exploratorio y BIC del GMM')}

## 14. Estabilidad temporal entre semanas

La réplica aplica sin cambios radio, cuadrícula, hueco, reglas de identidad, variables y muestras de calidad. Las medianas de H, V y duración son cercanas; las curvas de error cinemático también. La cuota máxima de un componente GMM piloto al aplicarlo a la réplica cambia {pct(structure['max_replication_share_shift'])}, lo cual muestra estabilidad descriptiva de la partición fija. El ARI de reajuste moderado y la sensibilidad al conjunto de variables indican que esa estabilidad de cuotas no demuestra fronteras de clase. Las semanas no son una muestra aleatoria independiente de condiciones meteorológicas y operacionales; la conclusión se refiere a estas dos semanas.

{figure('06_week_comparison','week_comparison.png','Panel 06. Replicación de geometrías y completitud')}

## 15. Interpretación operacional posterior

Los archivos físicos de ambas semanas se congelaron mediante SHA-256 en `diagnostics/structure_discovery.json` antes del enriquecimiento. Las autorizaciones Stage 03 se enlazan por vuelo participante durante [inicio−300 s, fin+300 s]. En el piloto {n(o[a].atc_context.sum())} episodios ({pct(o[a].atc_context.mean())}) tienen al menos una autorización en ese intervalo; en la réplica {n(o[b].atc_context.sum())} ({pct(o[b].atc_context.mean())}). El enlace amplio produce {n((o[a].ambiguous_pair_context_count>0).sum())} episodios con asociación ambigua en el piloto. Cada autorización puede coexistir con varias parejas; no se atribuye causalmente a ninguna. Se registran LEVEL, HEADING y SPEED, incluso tipos combinados, y la distancia temporal al punto medio.

Contando enlaces episodio–autorización, aparecen {n(o[a].level_count.sum())} contextos LEVEL, {n(o[a].heading_count.sum())} HEADING y {n(o[a].speed_count.sum())} SPEED en el piloto; una autorización puede contarse muchas veces. La mediana del mayor cambio de rumbo entre A y B es {dec(compared.loc[True,'max_abs_turn_deg'])}° con contexto ATC y {dec(compared.loc[False,'max_abs_turn_deg'])}° sin él; para cambio de altitud son {n(compared.loc[True,'max_abs_alt_change_ft'])} y {n(compared.loc[False,'max_abs_alt_change_ft'])} ft. También difieren la duración ({n(compared.loc[True,'observed_duration_s'])} frente a {n(compared.loc[False,'observed_duration_s'])} s) y la altitud inicial mediana de A ({n(compared.loc[True,'start_a_alt_ft'])} frente a {n(compared.loc[False,'start_a_alt_ft'])} ft). Son diferencias descriptivas probablemente influidas por fase de vuelo, duración y exposición a autorizaciones; no estiman un efecto causal.

Los colores Stage 10B solo cubren {n(ctx.window_id.nunique())} ventanas piloto enlazadas de {n(json.loads((OUT/'diagnostics'/'operational_enrichment.json').read_text())[-1]['pilot_detector_windows'])} ventanas disponibles, con {n(len(ctx))} enlaces ventana–episodio. En esos enlaces hay {n(ctx.focal_dynamic_state.eq('GREEN').sum())} GREEN, {n(ctx.focal_dynamic_state.eq('YELLOW').sum())} YELLOW, {n(ctx.focal_dynamic_state.eq('ORANGE').sum())} ORANGE y {n(ctx.focal_dynamic_state.eq('RED').sum())} RED. Son estados **focales** derivados de ventanas emparejadas, no colores de la pareja ni una cobertura del universo SCAT. No existen colores de réplica en este enriquecimiento. Por eso GREEN con geometría cercana o RED sin autorización focal identifican discrepancias para examen, no aciertos o fallos del detector a nivel pareja.

{figure('08_operational_comparison','operational_comparison.png','Contexto ATC posterior y ambigüedad de atribución')}

## 16. Casos representativos y contradictorios

El caso `{cases.query('case=="close_simultaneous"').episode_id.iloc[0]}` tiene H mínima {dec(cases.query('case=="close_simultaneous"').min_horizontal_nm.iloc[0])} NM y V simultánea {n(cases.query('case=="close_simultaneous"').vertical_at_min_horizontal_ft.iloc[0])} ft, con {n(cases.query('case=="close_simultaneous"').duration_s.iloc[0])} s de traza. Se esperaba una aproximación y separación visibles; el perfil confirma el orden temporal en esta pareja, pero no establece responsabilidad ATC ni pérdida operacional de separación. El caso paralelo muestra que una H pequeña puede persistir con V mayor, y el caso de cambio vertical ilustra por qué H y V mínimas no deben combinarse sin tiempo común.

{ex('close_simultaneous','Caso físico con proximidad horizontal y vertical simultánea')}

El caso GREEN focal cercano `{opcases.query('case=="green_close"').episode_id.iloc[0]}` tiene H mínima {dec(opcases.query('case=="green_close"').min_horizontal_nm.iloc[0])} NM y V simultánea {n(opcases.query('case=="green_close"').vertical_at_min_horizontal_ft.iloc[0])} ft. La hipótesis de que GREEN descarta todas las geometrías interesantes no se sostiene para este enlace; la alternativa es que el color describe otra pareja o un resumen focal en otra ventana. El caso RED sin autorización focal `{opcases.query('case=="red_without_focal_atc"').episode_id.iloc[0]}` tampoco demuestra omisión de intervención: el color no es etiqueta de conflicto y una maniobra puede responder a otros factores. El caso convergente sin contexto ATC asociado mantiene H mínima muy pequeña, pero una V simultánea mayor; muestra que no basta la dimensión horizontal.

{ex('green_close','Ejemplo GREEN focal con geometría de pareja cercana; color no asignado a la pareja')}

{ex('red_without_focal_atc','Ejemplo RED focal sin autorización temporal asociada; interpretación abierta')}

{ex('convergent_no_atc_context','Ejemplo convergente sin autorización temporal asociada')}

La galería completa se enumera en `examples/case_gallery.csv` y `examples/operational_case_gallery.csv`; cada figura conserva trayectoria, sentido, duración y perfiles H/V.

## 17. Limitaciones

La principal limitación es la censura temporal de ~77 %: un segmento de observación puede ser solo una fracción de una interacción física. El radio de 100 NM determina inclusión; un cambio a 50/150 NM modifica notablemente conteos y distribuciones. La sincronización de 30 s admite estados hasta 15 s antiguos, sin interpolación; el contraste de 15 s es de seis horas. Los criterios de salto físico no garantizan ausencia de fusiones sutiles. Los vuelos pueden repetir identificadores en otros archivos; se preserva `source_member`, pero la auditoría no prueba que la procedencia SCAT esté libre de errores. Las tasas verticales instantáneas generan gran error a medio plazo. La validación del CPA está condicionada a trazas supervivientes; no estima exactitud en las censuradas. El muestreo de estructura restringe a fronteras completas, de modo que sus resultados no representan sin sesgo a todos los episodios. Los colores y TopSky no ofrecen una señal por pareja con procedencia temporal suficiente para afirmar validez de conflicto; no se infiere información TopSky en esta etapa.

## 18. Discusión científica

Esperábamos que la identidad de pareja resolviera las contradicciones de Stage 11D. La reconstrucción demuestra que puede seguirse la geometría A–B y que la distinción entre H mínima y V simultánea cambia radicalmente el conteo de geometrías próximas. La casi igualdad entre semanas fortalece la reproducibilidad de medidas descriptivas. Sin embargo, la censura y la ausencia de agrupaciones robustas impiden convertir estas geometrías en taxonomía de conflicto. La aparente estabilidad de GMM al trasladarlo a otra semana puede resultar de distribuciones marginales parecidas, mientras sus fronteras cambian con escalado y variables. De forma similar, muchas autorizaciones se asocian por la gran exposición temporal y el número de parejas, no necesariamente por causalidad. La conclusión defendible es que las interacciones observadas admiten descripción física continua y análisis de casos, no una escala ordinal validada.

## 19. Implicaciones para una futura severidad

Una investigación posterior debería conservar dimensiones separadas: H y V actuales simultáneas, H/V previstas al mismo TCPA, tiempo de aproximación, cierre y persistencia, cambios verticales, incertidumbre según horizonte y contexto operacional. Las medidas geométricas pueden describir proximidad, pero una severidad de conflicto requiere definir un evento operacional verificable, revisar reglas de separación según fase y sector, y calibrar predicciones bajo maniobra y censura. Una variable continua multicomponente parece más investigable que cuatro colores preimpuestos; tampoco se justifica un score 0–100 ni nuevos umbrales con estos datos.

## 20. Conclusión, decisión y siguiente paso

**Decisión: `CONTINUOUS_INTERACTION_STRUCTURE`.** La distribución de H/V/cierre es amplia y reproducible; el GMM no alcanza un K natural dentro del rango probado, su separación es baja, HDBSCAN es sensible a parámetros y los componentes cambian al modificar las variables. La decisión describe únicamente el espacio de episodios observados bajo 100 NM. Queda evidencia insuficiente para afirmar familias discretas de conflicto, para validar un detector o para entrenar severidad física.

El siguiente paso científico es auditar manualmente los casos cercanos con trazas nativas y contexto de sector, reducir la censura con un esquema explícito de continuidad sin mirar al futuro, y definir un resultado operacional independiente de ATC y color. Solo entonces se podrá evaluar una métrica multivariante calibrada de riesgo.

## Apéndice de reproducibilidad

Scripts en `src/scat_analysis/stage12_*.py`; datos procesados, episodios, predicciones, diagnósticos y figuras en esta carpeta. Los parámetros están en `stage12_build_encounters.py` y las semillas en `stage12_explore_structure.py`. La configuración primaria es 30 s, estado ≤15 s, radio 100 NM, hueco 60 s y sin interpolación. Las sensibilidades son 50/150 NM, 45/90 s, un tramo de 15 s y horizonte de 30 min. `manifest.csv` contiene tamaño y SHA-256 de cada salida. Los tests sintéticos cubren identidad, geometría, unidades, discontinuidades, censura, CPA futuro y simultaneidad; la galería ofrece auditoría visual adicional.
'''
    path=OUT/'reports'/f'{TITLE}.md';path.write_text(txt,encoding='utf-8')
    print(path,'chars',len(txt))

if __name__=='__main__':main()

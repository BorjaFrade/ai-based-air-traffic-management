# Tareas pendientes para completar el paper (para Codex)

Repositorio: https://github.com/BorjaFrade/ai-based-air-traffic-management
Manuscrito: `paper/ceas_manuscript_figures.tex` (sección «Encounter-level comparison against reference models», `sec:benchmarks`, y sección de sensibilidad).
Figuras: `paper/figure_scripts/generate.py` (solo PNG, un archivo por panel; ver `paper/figure_scripts/README.md`).
Entradas congeladas: resultados de Stage 16–21 (no se modifican). Calibración = 9 semanas, test = 4 semanas (se puntúa UNA sola vez).

## Reglas generales
- Nada se ajusta con las semanas de test: umbrales y parámetros se fijan en calibración (leave-one-week-out para hiperparámetros).
- Todos los modelos usan la MISMA cohorte de episodios, las mismas semanas y el mismo resultado Y.
- No inventar valores: cada cifra del paper debe salir de un script reproducible. Los scripts nuevos van en `paper/figure_scripts/` o `src/`, con `python3`, y escriben en carpetas de resultados nuevas.
- Cada script debe incluir comprobaciones (`check`) de conteos, como `generate.py`.

## Prioridad CRÍTICA

### T1. Decidir y fijar tres parámetros ANTES de ejecutar nada (requieren OK de los autores)
1. Umbral de inclusión de episodios verdes (propuesta del paper: m_min < 2). Hacer además una variante sin restringir y otra con m_min < 3 como sensibilidad.
2. Modelo de error de M6 (o decidir recortarlo: ver T9).
3. Umbral de la regla de decisión (diferencia de Brier de M0 frente al mejor de M2–M4, respecto a su error estándar bootstrap).
Entregable: sección `docs/decisiones_protocolo.md` con las decisiones y fecha, antes de puntuar el test.

### T2. Construir los episodios verdes y la cohorte común
- Aplicar la regla de gap de 360 s a los pair-instants verdes (restringidos según T1). Una racha toda verde = episodio verde; una racha con algún no verde conserva su color máximo.
- Salida: tabla de episodios (todas las clases) con: id, semana, split, color máximo, m_min, τ, inicio/fin, duración, estrato de tráfico, indicador de orden previa, Y (orden posterior).
- Comprobaciones: el conteo de episodios no verdes debe reproducir 54.441 (39.351 amarillos, 9.039 naranjas, 6.051 rojos) y los 1.863 rojos del test con 642/213/187/821.

### T3. Igualar las ventanas de asociación (exposición del episodio)
Problema: hoy una orden cuenta si cae en los 300 s posteriores a CUALQUIER observación del episodio, así que un episodio largo tiene más oportunidades de recibir una orden.
- Definir Y_fija: orden en una ventana fija (p. ej. (T0, T0+300 s]) respecto al INICIO del episodio.
- Comparar Y_actual frente a Y_fija para cada color: tasa por clase, y correlación entre duración del episodio y Y.
- Incluir la duración (o el log de la duración) como covariable/offset en M2–M4 como alternativa.
- Entregable: tabla y figura con las dos definiciones y su efecto en el orden de las clases (¿se mantiene la monotonía verde < amarillo < naranja < rojo?).

### T4. Experimento M0–M6 (resultado principal)
Modelos (según la tabla del paper):
- M0: partición de cuatro niveles congelada → probabilidad = tasa empírica de Y por clase estimada SOLO en calibración.
- M1: pantalla binaria m_min < 1 → misma conversión (tasa empírica por clase binaria). No se compara un indicador binario con probabilidades sin esta conversión.
- M2: logística en m_min. M3: logística con términos suaves en (m_min, τ). M4: M3 + estrato de tráfico + indicador de orden previa.
- M5: particiones monótonas con K = 3, 4, 5 niveles (ajustadas en calibración).
- M6: probabilidad de conflicto con error gaussiano creciente con el horizonte (si se mantiene; ver T9).
Métricas en test: Brier, log loss, intercepto y pendiente de calibración, curvas de fiabilidad por deciles, AUC (secundaria). Incertidumbre: bootstrap por días (≥ 400 réplicas) y diferencias pareadas por semana (descriptivas, 4 semanas). Alertas: episodios no verdes DEDUPLICADOS por hora, no pair-instants.
Informar con las dos ponderaciones: por pair-instant y un-episodio-un-voto.
Salidas: `results/benchmarks/*.csv`, tabla `tab:bench` para el .tex, y Fig. 8 (nueva función `fig8()` en `generate.py`: PNG, sin título A/B, subfiguras si hay varios paneles).

## Prioridad ALTA

### T5. Auditar la selección de las 51.950 autorizaciones
- Cuantificar el embudo: 1.019.799 registros accionables → 908.165 con rumbo/nivel → 51.950 elegibles (50.774 nivel + 1.176 rumbo): cuántos se pierden por cada filtro (FL290–FL410, nivel numérico y distinto, estado de vigilancia ≤ 10 s, etc.).
- Comparar el conjunto elegible con el ampliado (908.165): distribución por semana, nivel de vuelo, tipo de orden y color.
- Entregable: tabla de embudo + párrafo para la sección de datos.

### T6. Causas de la cobertura incompleta (solo 1.744 de 6.051 rojos evaluables)
- Clasificar los episodios no evaluables en: salida del área estudiada, fin de registro, hueco de vigilancia, fallo de reconstrucción/predicción, otros (usar última posición, límites geográficos, nivel de vuelo y cinemática; aprovechar el trabajo de continuidad individual existente).
- No mezclarlo con el modelo de predicción: es caracterización de calidad de datos para justificar las exclusiones.
- Entregable: resumen cuantitativo por color (tabla; figura opcional como panel nuevo de la Fig. 7).

### T7. Sensibilidad a umbrales y ventanas
Repetir T2–T4 con: gap de episodio 240/360/480 s; ventana posterior 120/300/600 s; ventana previa 300/600 s; conjunto de órdenes ampliado (908.165); umbral de episodios verdes (T1). Informar solo si cambia el orden de las clases y la comparación M0 vs M2–M4.
Entregable: tabla resumen y apartado «Sensitivity analyses» con resultados.

## Prioridad MEDIA

### T8. Integrar resultados en el manuscrito
- Rellenar tabla de resultados y Fig. 8 (`fig:benchmarks`, hoy `\figplaceholder`).
- Sustituir los `\tbc` de la sección de comparación y reescribir la conclusión con los hallazgos reales (no con los esperados).
- Si se recorta M6, actualizar de forma consistente: tabla de modelos, introducción («M1–M6»), protocolo, Fig. 1 y resumen.

### T9. Decisión de alcance (con los tutores)
Propuesta: reducir M6 (probabilidad de conflicto con error gaussiano) a trabajo futuro, porque su probabilidad de conflicto no equivale a la de autorización y exige mucha modelización. Si se acepta: eliminar M6 y las citas asociadas solo si ya no se usan, y renombrar «M0–M6» a «M0–M5».

### T10. Reproducibilidad
- README con el orden de ejecución, parámetros y semillas; etiquetar un release y archivar en Zenodo (DOI para «Code availability»).
- Comprobar que las figuras se regeneran desde cero con `python3 paper/figure_scripts/generate.py --figure all` y revisar visualmente figs. 3, 4 y 5 con datos reales.

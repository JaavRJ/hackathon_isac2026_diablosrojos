# Plan de Desarrollo — Rediseño Visual de Diablos AeroVision (HACK-019)

> **Estado:** propuesto, pendiente de aprobación del mockup (Fase 0).
> **Base:** [DISENO_UI_NIELSEN.md](DISENO_UI_NIELSEN.md) (usabilidad, se adopta casi completa) + auditoría visual del dashboard actual (2026-10-20).
> **No reabre HACK-015:** se registra como tarea nueva HACK-019 con AllowedPaths propios.

---

## 1. Diagnóstico

### 1.1 Correcciones al documento de Nielsen
| # | Problema en el doc | Corrección |
|---|---|---|
| 1 | Dice ρ̂ CDMX ≈ 0.78 | El modelo usa **0.852**. Leer siempre de `stuff_model.rho_extreme_median_`, nunca hardcodear. |
| 2 | Cifras de la descomposición escritas a mano en el glosario | Leerlas de `reports/altitude/channel_decomposition.csv`. |
| 3 | Semáforo verde/azul/gris/ámbar/coral no es perceptualmente ordenado | Escala **divergente ordenada** coral → gris → turquesa, apta para daltonismo, con íconos ▲ ● ▼. |
| 4 | Razones de contraste afirmadas, no medidas | Medirlas (WCAG) en la Fase 4 y ajustar tokens. |

### 1.2 Problemas del dashboard actual
- **Rojo de marca usado como dato:** el número Stuff+ CDMX, las barras SHAP negativas y el mapa de movimiento (todo rojo, ilegible).
- **Sin jerarquía:** el resultado queda debajo de ~20 controles; hay que hacer scroll para ver ΔStuff+.
- **Mapa y Simulador separados:** el pitcheo nunca se ve contra la liga.
- **Ruido de Streamlit:** botón "Deploy", aviso "Help agents", chrome por defecto.
- **Leaderboard plano:** sin barras, sin color semántico, sin tarjeta de lanzador atractiva.
- **Sculptor vacío** hasta pulsar "Buscar".

### 1.3 Hallazgo del modelo (fuera de diseño, a investigar)
Con el pitcheo por defecto, una Four-Seam que pierde *ride* en CDMX (IVB 16 → 13 in) **sube** su Stuff+ (86.0 → 87.2, Δ +1.21). Puede ser real (túnel o diferencial con la recta), pero es contraintuitivo y un juez lo preguntará. Investigar antes de presentarlo; la UI no lo esconde.

---

## 2. Dirección visual: "Laboratorio aerodinámico nocturno"

- **Superficies:** fondo carbón `#0D0D0D` con capas `#141414` / `#1C1C1C`, bordes finos `#2A2A2A`.
- **Marca:** el rojo Diablos `#CE1126` solo en barra superior, pestaña activa y botón primario. **Nunca como dato ni como "malo".**
- **Color de datos (divergente, centro = 100 o Δ = 0):** coral (pierde) → gris (promedio) → turquesa (gana), siempre con ▲ ● ▼. Escala secuencial neutra para niveles en el mapa.
- **Tipografía:** condensada deportiva (tipo marcador) para números grandes; sans limpia para texto; `tabular-nums` en todo número.
- **Gráficas:** plantilla Plotly propia aplicada a todas.

---

## 3. Estructura nueva

1. **Shell**
   - Encabezado compacto: wordmark + chips de contexto (ρ̂ CDMX, temporada, "2026 fuera de muestra").
   - Barra lateral fija: pitcheo activo con badge (Preset / Pitcher real / Editado), presets, Revertir, Promedio LMB, glosario.
   - Ocultar chrome de Streamlit.
2. **Simulador** (pieza estrella)
   - Dos zonas: riel de controles a la izquierda, lienzo de resultados a la derecha, visible sin scroll.
   - Arriba: **duelo** Nivel del Mar vs Harp Helú con **ΔStuff+ como número héroe** al centro.
   - Medio: plano de movimiento **sobre la densidad de la liga** con vector ○ → ★, y SHAP con colores semánticos.
   - Abajo: descomposición de 3 términos como barra apilada; residuo rayado con su etiqueta propia, nunca sumado al parque.
3. **Mapa de movimiento:** secuencial neutra para niveles, divergente para Δ, pitcheo activo superpuesto.
4. **Leaderboard + tarjeta de lanzador:** barras en celda y Δ coloreado; al seleccionar, tarjeta tipo estampa de béisbol con arsenal en chips, mini-barras nivel del mar vs CDMX, mini-mapa y botón "Llevar al Simulador".
5. **Sculptor:** contador de pares, barra p25–p75 con *tu* pitcheo marcado, comparables como tarjetas, guía de coaching en bloque punteado; se autollena con el pitcheo activo.
6. **VeloVision:** pestaña visible en estado "En espera" mientras HACK-017 está pausada.
7. **Modo Presentación** (entrevista de 10 min): oculta controles y agranda cifras para proyector.

Se conservan literales todos los textos de honestidad ya aprobados (HACK-015, HACK-016).

### 3.1 Ajustes por la actualización de la competencia (`COMPETENCIA.md`, 20-oct-2026)
8. **Franja de cifras ancla:** mejora de Pitching+ vs baseline nulo en cada holdout (RMSE `xrv_pitcher`) + estabilidad año a año, desde `reports/validation/`.
9. **Badges de procedencia** en cada pantalla: modelo, CSV de origen, ρ̂ usado, dentro/fuera de muestra.
10. **Desglose del resultado en lenguaje llano** (bola, strike cantado, whiff, contacto) junto al SHAP, desde las probabilidades `exp_*` de `predict_outcomes`.
11. **Pestaña "Escenarios"** con los 3 casos de entrevista (fichaje, rediseño con Sculptor, casa vs gira).
12. **Pestaña "Metodología"** con el pipeline en 5 pasos.
13. **Incertidumbre (±) en leaderboard:** columna prevista; requiere calcular IC por pitcher (pendiente, no inventar).

### 3.2 Riesgo científico detectado (fuera de diseño)
La competencia reporta ΔStuff+ CDMX ≈ −12.7 (medio −11.6); el nuestro es −0.06 de media. Antes de presentar hay que medir la **sensibilidad de Stuff+ al movimiento** (puntos por pulgada de IVB/HB) y explicar la diferencia (posible inclusión del *carry* del batazo en su Stuff+). Ligado al hallazgo §1.3.

---

## 4. Fases de ejecución

| Fase | Entregable | Responsable | Criterio de salida |
|---|---|---|---|
| 0 | Mockup visual (Simulador, Leaderboard + tarjeta, Sculptor) | Orquestador (Opus) | Aprobación del usuario |
| 1 | Sistema de diseño: `.streamlit/config.toml`, tokens CSS, plantilla Plotly, `dashboard/components.py` | Orquestador | Tokens y componentes con test |
| 2 | Shell, barra lateral con estado (`st.session_state` según §3.3 de Nielsen), Simulador | Orquestador | Simulador sin scroll a 1440×900 |
| 3 | Mapa, Leaderboard + tarjeta, Sculptor | Workers Sonnet | Siguen el sistema de diseño |
| 4 | Verificación | Revisor independiente | Ver §5 |

---

## 5. Criterios de aceptación (HACK-019)

1. `tests/test_dashboard_smoke.py` y `tests/test_sculptor.py` siguen en verde; carga en frío < 20 s y predicción en caliente < 3 s.
2. El rojo `#CE1126` no aparece en ningún valor numérico, barra ni escala de color de datos (test por inspección de tokens/plantilla).
3. Contraste medido ≥ 4.5:1 para texto normal y ≥ 3:1 para texto grande y elementos gráficos.
4. ρ̂ CDMX y la descomposición se leen de modelo/CSV, sin cifras hardcodeadas (test por grep).
5. Capturas en 1440×900 y 1920×1080 (proyector) de las 5 pestañas, revisadas por el usuario.
6. Toda la UI en español; textos de honestidad de HACK-015/016 intactos.

**AllowedPaths propuestos:** `dashboard/**`, `.streamlit/config.toml`, `tests/test_dashboard_smoke.py`, `tests/test_dashboard_design.py`, `docs/PLAN_REDISENO_UI.md`.

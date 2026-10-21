# Backlog Maestro de Tareas para Orquestador, Workers y Revisores (TASKS.md)

> **Arquitectura Multi-Agente:**  
> 1. **ORQUESTADOR (`Orchestrator`):** Lee este archivo, identifica las tareas en estado `AVAILABLE` cuyas dependencias (`DependsOn`) estén todas en `APPROVED` (o tareas en `CHANGES_REQUESTED`), y las asigna a agentes ejecutores económicos (*Workers*).  
> 2. **AGENTE EJECUTOR (`Worker` — Modelo Rápido/Barato):** Recibe un ID de tarea. Lee `AGENTS.md`, `docs/CONTRACT.md` y su tarea en este archivo. **PROHIBIDO modificar archivos fuera de `AllowedPaths`.** Cuando su `Verify Command` pasa con `exit code 0`, cambia `State: REVIEW`.  
> 3. **AGENTE REVISOR (`Reviewer` — Proveedor/Modelo Independiente):** Al iniciar, busca automáticamente en este archivo **todas las tareas con `State: REVIEW`**. Para cada una: revisa el código en `AllowedPaths`, ejecuta por su cuenta el `Verify Command` y audita cada punto del `Reviewer Checklist`. Si todo cumple al 100 %, cambia el estado a `APPROVED` (en la tabla y en la sección) y marca `[x]` el checklist. Si falla algo, cambia a `CHANGES_REQUESTED` y agrega `#### Notas de Revisión` con `archivo:línea` y motivo exacto.

---

## Tabla de Estado Global (Tablero del Orquestador)

| ID | Sprint | Prio | Estado | DependsOn | Resumen |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **HACK-001** | S1 | P0 | `APPROVED` | `—` | Entorno (`pyarrow>=20`), contrato de datos (`src/io.py`) y fixture sintético (`tests/conftest.py`) |
| **HACK-002** | S1 | P0 | `APPROVED` | `HACK-001` | Limpieza, imputación por juego/pitcher y features físicas base (`src/features/physical.py`) |
| **HACK-003** | S1 | P0 | `APPROVED` | `HACK-002` | Estimación de densidad continua del aire $\hat{\rho}$ por juego (`src/physics/density.py`) |
| **HACK-004** | S1 | P0 | `APPROVED` | `HACK-002` | Identidad de pitcheo invariante a la altitud `pitch_id` (`src/features/identity.py`) |
| **HACK-005** | S1 | P0 | `APPROVED` | `HACK-004` | Arsenal relacional, secuencia `prev_pitch_id` y ambos túneles (`src/features/arsenal.py`) |
| **HACK-006** | S1 | P0 | `APPROVED` | `HACK-002` | Calibración LMB de `xrv_pitcher` y los 8 targets multi-objetivo (`src/targets.py`) |
| **HACK-007** | S1 | P0 | `APPROVED` | `HACK-006` | Particiones oficiales (Pitcher/Season/Park Holdout) y Baseline Nulo (`src/splits.py`, `src/baseline.py`) |
| **HACK-008** | S1 | P0 | `APPROVED` | `HACK-003, HACK-005, HACK-007` | Pipeline integrador de dataset procesado (`src/pipeline_step1.py`) |
| **HACK-009** | S2 | P0 | `APPROVED` | `HACK-008` | Etapa A: Traductor físico intra-pitcher de movimiento por densidad (`src/models/movement.py`) |
| **HACK-010** | S2 | P0 | `APPROVED` | `HACK-009` | Submodelo `Stuff+` (neutral y CDMX) sin leakage de ubicación/conteo (`src/models/stuff.py`) |
| **HACK-011** | S2 | P0 | `APPROVED` | `HACK-008` | Submodelo `Location+` y `Pitching+` con combinación multi-target (`src/models/location_pitching.py`) |
| **HACK-012** | S2 | P0 | `APPROVED` | `HACK-010, HACK-011` | Suite de validación en los 3 holdouts vs Baseline Nulo + Estabilidad (`analysis/run_validation.py`) |
| **HACK-013** | S2 | P0 | `APPROVED` | `HACK-010, HACK-011` | Descomposición Canal Stuff vs Canal Parque y pruebas empíricas H1–H4 (`analysis/run_hypotheses.py`) |
| **HACK-014** | S2 | P0 | `APPROVED` | `HACK-012, HACK-013` | Exportador oficial Entregable 1: scores por `PitchUID` y agregados (`analysis/export_scores.py`) |
| **HACK-015** | S3 | P0 | `APPROVED` | `HACK-014` | Dashboard Simulador Base Obligatorio en Streamlit (`dashboard/app.py`) |
| **HACK-016** | S3 | P1 | `APPROVED` | `HACK-014` | **AI Pitch Sculptor** como buscador de referencias reales factibles (sin reentrenar) + pestaña UI (`src/sculptor/optimizer.py`) |
| **HACK-017** | S3 | P1 | `AVAILABLE` | `HACK-014` | Módulo **VeloVision** (MediaPipe Pose $\to$ rango Stuff+ CDMX) + validación manual (`src/vision/pose_slot.py`) |
| **HACK-018** | S4 | P1 | `AVAILABLE` | `HACK-014, HACK-015, HACK-016, HACK-017` | Entregables escritos: Perfil Físico Ideal, 3 Casos de Entrevista y Documento Técnico 6–10 págs (`docs/`) |
| **HACK-019** | S3 | P1 | `IN_PROGRESS` | `HACK-015, HACK-016` | Rediseño visual del dashboard ("Laboratorio aerodinámico nocturno") según `docs/PLAN_REDISENO_UI.md` |

---

## Especificación Detallada de Cada Tarea

---

### `HACK-001` — Entorno, Contrato de Carga (`src/io.py`) y Fixture Sintético (`tests/conftest.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `—`
- **AllowedPaths:**
  - `src/config.py`
  - `src/io.py`
  - `tests/__init__.py`
  - `tests/conftest.py`
  - `tests/test_io.py`
- **Especificación e Implementación:**
  1. En `src/config.py`, exportar las listas `FORBIDDEN_TARGET_COLS`, `ID_AND_GROUP_COLS` y `LOCATION_AND_COUNT_COLS` idénticas a `docs/CONTRACT.md` §1, además de `RAW_DATA_PATH` y `PROCESSED_DATA_PATH`.
  2. En `src/io.py`, implementar:
     - `load_raw_pitches(columns: list[str] | None = None, sample_n: int | None = None) -> pd.DataFrame`: lee `RAW_DATA_PATH` con `pyarrow`. Si `sample_n` no es `None`, devuelve una muestra determinista (`random_state=42`).
     - `assert_no_leakage(feature_cols: list[str], model_type: str) -> None`: lanza `ValueError` si `feature_cols` contiene alguna columna de `FORBIDDEN_TARGET_COLS` o `ID_AND_GROUP_COLS`. Si `model_type == "stuff"`, lanza `ValueError` si contiene alguna columna de `LOCATION_AND_COUNT_COLS`.
  3. En `tests/conftest.py`, crear el fixture `@pytest.fixture(scope="session") def synthetic_lmb_df() -> pd.DataFrame`:
     - Genera `1,200` filas deterministas (`np.random.default_rng(42)`) con las **84 columnas exactas** del diccionario (`stuff_plus_data_dictionary.csv`), 12 pitchers (`pitcher_0001`..`0012`), 18 juegos repartidos en `No Altitude`, `Medium Altitude`, `Extreme Altitude` (incluyendo 20 filas con `altitude_category = None` pero compartiendo `game_anon_id`), 3 temporadas (`2024`, `2025`, `2026`), valores de `EffectiveVelo` como `object` (string numérico + algunos `'None'`), y pitcheos en orden de aparición al plato.
- **Criterios de Aceptación:**
  1. `pyarrow >= 20.0.0` y `pytest` están instalados y funcionando.
  2. `synthetic_lmb_df` tiene exactamente 84 columnas cuyos nombres coinciden 1 a 1 con `stuff_plus_data_dictionary.csv`.
  3. `assert_no_leakage` bloquea cualquier intento de meter `is_whiff` o `PlateLocHeight` en `Stuff+`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_io.py
  ```
- **Reviewer Checklist:**
  - [x] ¿`synthetic_lmb_df` es 100 % determinista y no lee el parquet real de 206 MB durante los unit tests?
  - [x] ¿`assert_no_leakage` lanza `ValueError` tanto para `FORBIDDEN_TARGET_COLS` como para `LOCATION_AND_COUNT_COLS` en modo `"stuff"`?
  - [x] ¿No se modificó ningún archivo fuera de `AllowedPaths`?

#### Notas de Revisión
- `tests/conftest.py:366-368` (RESUELTO): `current_outs`, `current_balls`, `current_strikes` y `count_str` ahora se capturan estrictamente antes de procesar el resultado del lanzamiento. En el 100 % de las filas se cumple `count == f"{int(Balls)}-{int(Strikes)}"`, validado por nuevo test unitario.
- `tests/test_io.py:153-162` (RESUELTO): Se eliminó la lectura del parquet real de 206 MB en tests unitarios. Ahora se utiliza `tmp_path` con un dataset sintético pequeño y `monkeypatch.setattr(src.io, "RAW_DATA_PATH", tmp_parquet)` para validar proyecciones y muestreo determinista.
- `src/io.py:20-24,62` (RESUELTO): Firmas ajustadas exactamente a la especificación de CONTRACT.md: `load_raw_pitches(columns: list[str] | None = None, sample_n: int | None = None) -> pd.DataFrame` y `assert_no_leakage(feature_cols: list[str], model_type: str) -> None`.
- `AGENTS.md:10`, `CLAUDE.md:1`, `CODEX.md:1` (ACLARADO): Limpiado externamente por el usuario/orquestador en commit previo; el árbol de trabajo actual solo contiene modificaciones en `AllowedPaths` (`src/config.py`, `src/io.py`, `tests/`, `docs/TASKS.md`).

---

### `HACK-002` — Limpieza, Imputación Inteligente y Features Físicas Base (`src/features/physical.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-001`
- **AllowedPaths:**
  - `src/features/__init__.py`
  - `src/features/physical.py`
  - `tests/test_physical.py`
- **Especificación e Implementación:**
  Implementar en `src/features/physical.py`:
  1. `clean_and_impute(df: pd.DataFrame) -> pd.DataFrame`:
     - Convierte `EffectiveVelo` a `float64` con `pd.to_numeric(..., errors="coerce")`.
     - Recodifica `AutoPitchType` (`OneSeamFastBall` $\to$ `Sinker`, `TwoSeamFastBall` $\to$ `Sinker`, `Knuckleball` $\to$ `Other`).
     - Imputa nulos/`'None'`/`'Undefined'` en `altitude_category` usando la moda de `altitude_category` dentro del mismo `game_anon_id` (y si todo el juego es nulo, usa `'No Altitude'`).
     - Imputa `PitcherThrows` faltante con la moda por `pitcher_anon_id`, y como fallback `'Right'` si `RelSide >= 0` else `'Left'`.
     - Imputa `BatterSide` faltante con `'Right'`.
     - Filtra anomalías de radar dejando solo `RelSpeed.between(65.0, 105.0)` y `SpinRate.between(500.0, 3600.0)`.
  2. `add_physical_features(df: pd.DataFrame) -> pd.DataFrame`:
     - Crea las 11 columnas exactas de `docs/CONTRACT.md` para `HACK-002`: `HorzBreak_norm`, `RelSide_norm`, `HorzRelAngle_norm`, `HorzApprAngle_norm`, `PlateLocSide_norm`, `TotalInducedBreak`, `BreakAngle_deg`, `SpinAxis_sin`, `SpinAxis_cos`, `SpinToBreakRatio`, `DragProxy`, `arm_slot_deg`.
     - Crea también las variables geométricas de zona para `Location+`: `dist_from_zone_center`, `dist_to_zone_edge`, `is_shadow_zone`, `zone_quadrant` (`1..9` dentro de zona, `10..13` fuera).
- **Criterios de Aceptación:**
  1. Cero valores nulos en `altitude_category`, `PitcherThrows` y `BatterSide` tras `clean_and_impute`.
  2. Para un pitcher zurdo (`PitcherThrows == 'Left'`), `HorzBreak_norm == -HorzBreak`.
  3. Todas las columnas de `docs/CONTRACT.md` (`HACK-002`) existen y son `float64` sin `inf`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_physical.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Se invirtieron correctamente los signos horizontales para lanzadores zurdos (`hand_sign = -1.0`)?
  - [x] ¿La imputación de `altitude_category` agrupa por `game_anon_id` sin usar variables de resultado (`target_only`)?

#### Notas de Revisión
- `src/features/physical.py:197-204` (RESUELTO): `dist_to_zone_edge` ahora calcula la distancia al borde del rectángulo (`x in [-0.833, 0.833], z in [1.5, 3.5]`) distinguiendo puntos interiores (mínimo a los 4 lados con `np.minimum(-dx_edge, -dz_edge)`) y exteriores (distancia euclidiana). Se añadieron 4 pruebas analíticas exactas en `tests/test_physical.py` (centro = 0.833, borde = 0.0, interior superior = 0.1, exterior esquina = $\sqrt{2}$).

---

### `HACK-003` — Estimación de Densidad Continua del Aire ($\hat{\rho}$) por Juego (`src/physics/density.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-002`
- **AllowedPaths:**
  - `src/physics/__init__.py`
  - `src/physics/density.py`
  - `tests/test_density.py`
- **Especificación e Implementación:**
  La fuerza de arrastre aerodinámico cumple $a_{\text{drag}} \propto \rho \cdot v^2$. En las coordenadas de TrackMan, `vy0` es negativo (hacia el plato) y `ay0` es positivo (frena la pelota).
  Implementar en `src/physics/density.py`:
  1. `estimate_game_density(df: pd.DataFrame) -> pd.DataFrame` (devuelve tabla por `game_anon_id` con `rho_hat`, `n_fastballs`, `altitude_category`) y `attach_rho_hat(df: pd.DataFrame, game_rho_df: pd.DataFrame | None = None) -> pd.DataFrame`:
     - Para cada lanzamiento con `RelSpeed >= 82.0`, calcular el proxy instantáneo de arrastre específico:
       $$k_i = \frac{\text{ay0}_i}{\text{vy0}_i^2 + \text{vx0}_i^2 + \text{vz0}_i^2}$$
       (con fallback a $\text{SpeedDrop}_i / (\text{RelSpeed}_i^2 \cdot \text{ZoneTime}_i)$ si `ay0` no está disponible).
     - Calcular la mediana de $k_i$ por `game_anon_id` y aplicar *empirical Bayes shrinkage* hacia la mediana de su `altitude_category`:
       $$\hat{k}_g = \frac{n_g}{n_g + 30} k_g + \frac{30}{n_g + 30} \bar{k}_{\text{alt}(g)}$$
     - Normalizar dividiendo entre la mediana de $\hat{k}_g$ de los juegos en `No Altitude`, de modo que $\text{median}(\hat{\rho} \mid \text{No Altitude}) = 1.000$, acotando $\hat{\rho} \in [0.70, 1.15]$.
- **Criterios de Aceptación:**
  1. La mediana de `rho_hat` en juegos de `No Altitude` es `1.000` ($\pm 0.005$).
  2. En el dataset real y en el sintético, la mediana de `rho_hat` sigue estrictamente el orden: `Extreme Altitude < Medium Altitude < No Altitude`.
  3. Ningún juego queda con `rho_hat` nulo (`NaN`).
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_density.py
  ```
- **Reviewer Checklist:**
  - [x] ¿El cálculo de `rho_hat` usa exclusivamente cinemática (`ay0`, `vx0`, `vy0`, `vz0`, `SpeedDrop`, `RelSpeed`) y **no** usa el quiebre Magnus (`InducedVertBreak`/`HorzBreak`), evitando circularidad con el efecto que queremos medir en H1?
  - [x] ¿Se aplica el suavizado (*shrinkage*) para juegos con pocos lanzamientos?

---

### `HACK-004` — Identidad de Pitcheo Invariante a la Altitud (`src/features/identity.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-002`
- **AllowedPaths:**
  - `src/features/identity.py`
  - `tests/test_identity.py`
- **Especificación e Implementación:**
  TrackMan reetiqueta pitcheos en altitud porque usa `InducedVertBreak` y `HorzBreak` (que caen con $\rho$).
  Implementar en `src/features/identity.py`:
  1. `assign_invariant_pitch_id(df: pd.DataFrame) -> pd.DataFrame`:
     - Usa únicamente features invariantes al aire en el momento del release: `RelSpeed`, `SpinRate`, `SpinAxis_sin`, `SpinAxis_cos`, `RelHeight`, `RelSide_norm`, `Extension`.
     - Primero entrena un clasificador de referencia (`HistGradientBoostingClassifier` o `RandomForestClassifier`, `random_state=42`) **únicamente sobre lanzamientos en `No Altitude`** para mapear `(velo_diff_pitcher, SpinRate, SpinAxis_sin, SpinAxis_cos, RelHeight, RelSide_norm)` $\to$ familia de pitcheo a nivel del mar.
     - Segundo, dentro de cada `(pitcher_anon_id, year)`, agrupa los pitcheos del lanzador en clusters invariantes (`GaussianMixture` o `KMeans` sobre las variables estandarizadas en release, eligiendo $K \in \{1..\min(6, \lfloor N/15 \rfloor)\}$ por BIC/silueta) y asigna a cada cluster `pitch_cluster_id` la etiqueta `pitch_id` dada por:
       - la moda de `AutoPitchType` del cluster en juegos `No Altitude` (si el pitcher tiene $\ge 10$ pitcheos de ese cluster en `No Altitude`), o
       - la predicción modal del clasificador de referencia entrenado a nivel del mar.
  2. `audit_trackman_relabeling(df: pd.DataFrame) -> pd.DataFrame`:
     - Compara para los mismos pitchers que lanzaron en `No Altitude` y `Extreme Altitude` el cambio en proporciones de `AutoPitchType` vs `pitch_id`.
- **Criterios de Aceptación:**
  1. Las columnas `pitch_cluster_id` (`int32`) y `pitch_id` (`string`) se crean sin nulos.
  2. Si a un mismo lanzamiento se le modifica artificialmente `InducedVertBreak` en $-4.0$ pulgadas y `HorzBreak` en $-3.0$ pulgadas (simulando altitud), `assign_invariant_pitch_id` produce **exactamente el mismo `pitch_id`** (prueba unitaria de invariancia).
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_identity.py
  ```
- **Reviewer Checklist:**
  - [x] **CRÍTICO:** ¿`InducedVertBreak`, `VertBreak`, `HorzBreak`, `pfxx` y `pfxz` están 100 % excluidos de las features que definen `pitch_cluster_id` y `pitch_id`?
  - [x] ¿El test unitario verifica explícitamente que alterar `InducedVertBreak` no cambia `pitch_id`?

#### Notas de Revisión
- `src/features/identity.py:320-338` (RESUELTO): El clasificador `HistGradientBoostingClassifier` se entrena exclusivamente con filas en `No Altitude`. Cuando no existen filas suficientes en `No Altitude`, se usa `_ReleaseKinematicsFallbackPredictor` basado únicamente en cinemática de release sin tocar jamás etiquetas de altitud ni variables de quiebre.
- `src/features/identity.py:194-240` (RESUELTO): Se implementó `_select_optimal_k_and_cluster` evaluando candidatos $K \in \{1..\min(6, \lfloor N/15 \rfloor)\}$ por criterio BIC con `GaussianMixture`, validado con `test_optimal_k_selection_by_bic` en `tests/test_identity.py`.
- `src/features/identity.py:378` (RESUELTO): Se actualizó el umbral de adopción de la moda en `No Altitude` a `no_alt_in_c.sum() >= 10` y se agregó `test_no_altitude_mode_threshold_5_to_9_vs_10` en `tests/test_identity.py`.
- `src/features/identity.py:392` (RESUELTO): Se usa `astype("string")` de pandas y se verifica explícitamente `res["pitch_id"].dtype == "string"` en `tests/test_identity.py`.

---

### `HACK-005` — Arsenal Relacional, Secuencia (`prev_pitch_id`) y Ambos Túneles (`src/features/arsenal.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-004`
- **AllowedPaths:**
  - `src/features/arsenal.py`
  - `tests/test_arsenal.py`
- **Especificación e Implementación:**
  Implementar en `src/features/arsenal.py`:
  1. `add_arsenal_and_tunnel_features(df: pd.DataFrame) -> pd.DataFrame`:
     - **Recta primaria (`primary_fb_id`):** Para cada `(pitcher_anon_id, year)`, identifica el `pitch_id` más frecuente entre `['Four-Seam', 'Sinker', 'Cutter']` (priorizando sus pitcheos en `No Altitude` si existen para que las medias base de `IVB` y `HB` no estén contaminadas por la altitud; si el pitcher solo lanzó en altitud, ajusta `fb_ivb` y `fb_hb_norm` dividiendo entre `rho_hat` mediano del pitcher).
     - **Diferenciales vs recta primaria:** Calcula `velo_diff_vs_fb`, `ivb_diff_vs_fb`, `hb_diff_vs_fb` y `pitch_usage_pct`.
     - **Túnel Versión A (Definición de las bases — Release Point):**
       $$\text{tunnel\_dist\_release\_in} = 12.0 \times \sqrt{(\text{RelHeight} - \overline{\text{RelHeight}}_{FB})^2 + (\text{RelSide\_norm} - \overline{\text{RelSide\_norm}}_{FB})^2 + (\text{Extension} - \overline{\text{Extension}}_{FB})^2}$$
       $$\text{tunnel\_sim\_release} = \exp(-\text{tunnel\_dist\_release\_in} / 6.0)$$
     - **Túnel Versión B (Extensión nuestra — Decision Point a $t = 0.17\text{ s}$):**
       Evalúa $X(0.17), Y(0.17), Z(0.17)$ con `PitchTrajectoryXc0..Zc2`, calcula la distancia euclidiana 3D en pulgadas contra la media de la recta primaria en $t = 0.17\text{ s}$ (`tunnel_dist_170ms_in`) y $\text{tunnel\_sim\_170ms} = \exp(-\text{tunnel\_dist\_170ms\_in} / 6.0)$.
     - **Secuencia (`prev_pitch_id`):**
       Detecta los límites de cada aparición al plato (mismo `game_anon_id`, `Inning`, `Top/Bottom`, `pitcher_anon_id`, `batter_anon_id` y donde `Balls + Strikes` no se reinicia a `0-0`). Asigna en `prev_pitch_id` el `pitch_id` del lanzamiento anterior en la misma PA, o `"NONE"` en el primer pitcheo del turno.
- **Criterios de Aceptación:**
  1. Se generan las 10 columnas de `docs/CONTRACT.md` (`HACK-005`) sin valores `NaN` ni `inf`.
  2. En el primer lanzamiento de cada turno (`Balls == 0` y `Strikes == 0` sin foul previo), `prev_pitch_id == "NONE"`.
  3. Para los lanzamientos de la propia recta primaria del pitcher, la mediana de `velo_diff_vs_fb` es $\approx 0.0$.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_arsenal.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Usa `pitch_id` (de `HACK-004`) en lugar de `AutoPitchType`?
  - [x] ¿Están implementadas **ambas** métricas de túnel (`tunnel_sim_release` y `tunnel_sim_170ms`)?
  - [x] ¿`prev_pitch_id` nunca mezcla lanzamientos de distintos bateadores o distintas entradas?

---

### `HACK-006` — Calibración Empírica LMB de `xrv_pitcher` y los 8 Targets (`src/targets.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-002`
- **AllowedPaths:**
  - `src/targets.py`
  - `tests/test_targets.py`
- **Especificación e Implementación:**
  Prohibido usar pesos lineales inventados o copiados de MLB. Todo debe estimarse de los datos de entrenamiento de la LMB (`year.isin([2024, 2025])`).
  Implementar en `src/targets.py`:
  1. `fit_lmb_run_values(train_df: pd.DataFrame) -> dict`:
     - Reconstruye las medias entrada (`game_anon_id`, `Inning`, `Top/Bottom`) y las apariciones al plato (PA).
     - Para cada estado `s = (Outs, Balls, Strikes)` con `Outs in {0,1,2}`, `Balls in {0,1,2,3}`, `Strikes in {0,1,2}`, estima el valor esperado de carreras desde ese pitcheo hasta el final de la aparición al plato / media entrada a partir de `RunsScored` y `OutsOnPlay` observados en `train_df`.
     - Calcula el valor lineal empírico LMB de cada evento terminal (`home_run`, `triple`, `double`, `single`, `base_on_balls`/`is_hit_by_pitch`, `strikeout`, `out_in_play`) y el valor de transición entre conteos $(B_t, S_t) \to (B_{t+1}, S_{t+1})$.
     - Define el signo desde la **perspectiva del pitcher**: `xrv_pitcher = -(Delta_Run_Expectancy_Bateador)`, de modo que un ponche o strike cantado tenga `xrv_pitcher > 0` y un jonrón o bola tenga `xrv_pitcher < 0`.
  2. `attach_targets(df: pd.DataFrame, rv_table: dict) -> pd.DataFrame`:
     - Agrega las 8 columnas de `docs/CONTRACT.md` (`HACK-006`): `target_whiff`, `target_chase`, `target_called_strike`, `target_groundball`, `target_weak_contact`, `target_hard_contact`, `target_out_on_pitch`, `xrv_pitcher`.
- **Criterios de Aceptación:**
  1. `fit_lmb_run_values` se ajusta **exclusivamente** con filas `year <= 2025` (cero data leakage del holdout 2026).
  2. Se cumple estrictamente la jerarquía física en `rv_table`:
     $$\text{xrv}(\text{strikeout}) > \text{xrv}(\text{called\_strike}) > \text{xrv}(\text{ball}) > \text{xrv}(\text{single}) > \text{xrv}(\text{home\_run})$$
  3. Las 8 columnas target existen y no tienen `NaN`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_targets.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Los valores de `rv_table` salen de un cálculo sobre `RunsScored` / transiciones del `DataFrame` de entrenamiento y **no** de constantes hardcodeadas?
  - [x] ¿El signo de `xrv_pitcher` es positivo cuando el evento beneficia al pitcher?

#### Notas de Revisión
- `src/targets.py:317-345` (RESUELTO): Se eliminó por completo `prior_weights` (`-1.40`, `-0.45`, `+0.30`, etc.). Todos los valores de evento y su suavizado Empirical Bayes se estiman exclusivamente a partir de `runs_to_end` y las transiciones de estado observadas en 2024–25 (con el conteo mediano de evento `m_e` y medias de clase derivadas de la muestra).
- `src/targets.py:224-272` (RESUELTO): Se eliminaron los incrementos fijos `0.18`, `0.06` y `0.05`. La matriz `re_matrix[(o, b, s)]` se estima agregando `runs_to_end` por `(Outs, Balls, Strikes)` en 2024–25 y suavizando estados escasos con Empirical Bayes hacia el prior marginal aditivo de la propia muestra (`mu_marg = max(0, R_o[o] + (R_b[b] - R_bar) + (R_s[s] - R_bar))`).
- `src/targets.py:346-350` (RESUELTO): Se eliminaron los bloques post-hoc que imponían la jerarquía. La jerarquía se evalúa libremente sobre los estimadores empíricos y se reporta en `rv_table["empirical_hierarchy_holds"]`.

---

### `HACK-007` — Particiones Oficiales (Holdouts) y Baseline Nulo (`src/splits.py`, `src/baseline.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-006`
- **AllowedPaths:**
  - `src/splits.py`
  - `src/baseline.py`
  - `tests/test_splits_baseline.py`
- **Especificación e Implementación:**
  1. En `src/splits.py`, implementar generadores deterministas de índices `(train_idx, test_idx)`:
     - `get_pitcher_holdout_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]`: usa `GroupKFold(n_splits=n_splits)` agrupado por `pitcher_anon_id` sobre `year.isin([2024, 2025])`. Ningún `pitcher_anon_id` de `test_idx` puede aparecer en `train_idx`.
     - `get_season_holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]`: `train = (year.isin([2024, 2025]))`, `test = (year == 2026)`.
     - `get_park_holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]`: `train = altitude_category.isin(['No Altitude', 'Medium Altitude']) & year.isin([2024, 2025])`, `test = (altitude_category == 'Extreme Altitude') & year.isin([2024, 2025])`.
  2. En `src/baseline.py`, implementar `CountBaselineModel`:
     - `fit(train_df: pd.DataFrame)`: guarda la media de los 8 targets (`target_whiff`..`xrv_pitcher`) agrupada únicamente por `count` (`Balls`, `Strikes`).
     - `predict(df: pd.DataFrame) -> pd.DataFrame`: devuelve las predicciones del modelo nulo por conteo.
     - `evaluate_against_baseline(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, baseline_pred_df: pd.DataFrame) -> pd.DataFrame`: calcula RMSE (para `xrv_pitcher`), Log-Loss y Brier Score (para los 7 targets binarios), y el porcentaje de mejora (`skill_score`) sobre el baseline nulo.
- **Criterios de Aceptación:**
  1. Intersección de `pitcher_anon_id` entre `train` y `test` en `get_pitcher_holdout_splits` es el conjunto vacío $\emptyset$.
  2. Intersección de `year` en `get_season_holdout_split` es $\emptyset$.
  3. `CountBaselineModel` corre y devuelve métricas válidas.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_splits_baseline.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Se verifica con `assert set(df.loc[train, 'pitcher_anon_id']).isdisjoint(set(df.loc[test, 'pitcher_anon_id']))`?

---

### `HACK-008` — Pipeline Integrador del Dataset Procesado (`src/pipeline_step1.py`)
- **State:** `APPROVED`
- **Sprint:** 1 · **Priority:** `P0` · **DependsOn:** `HACK-003, HACK-005, HACK-007`
- **AllowedPaths:**
  - `src/pipeline_step1.py`
  - `tests/test_pipeline_step1.py`
  - `reports/altitude/game_density_summary.csv`
  - `reports/altitude/relabeling_audit.csv`
- **Especificación e Implementación:**
  Conecta `HACK-001` a `HACK-007` en una función pura `build_processed_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]` y un script CLI que procesa `stuff_model_df.parquet`, guarda `data/stuff_processed_df.parquet`, y exporta dos reportes pequeños a git:
  - `reports/altitude/game_density_summary.csv` (resumen de `rho_hat` por `altitude_category` y año).
  - `reports/altitude/relabeling_audit.csv` (auditoría del reetiquetado de TrackMan entre altitudes).
- **Criterios de Aceptación:**
  1. `build_processed_dataset(synthetic_lmb_df)` corre en `< 5 s` y contiene todas las columnas de `docs/CONTRACT.md` (`HACK-002` a `HACK-006`).
  2. Al ejecutarse como script sobre el parquet real, genera `data/stuff_processed_df.parquet` y los 2 CSV en `reports/altitude/`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_pipeline_step1.py
  ```
- **Reviewer Checklist:**
  - [x] ¿El ajuste de `fit_lmb_run_values` dentro del pipeline usa solo `year.isin([2024, 2025])` antes de transformar `2026`?
  - [x] ¿Los archivos en `reports/altitude/` pesan menos de 2 MB y no contienen filas pitch-by-pitch?

---

### `HACK-009` — Etapa A: Traductor Físico Intra-Pitcher de Movimiento por Densidad (`src/models/movement.py`)
- **State:** `APPROVED`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-008`
- **AllowedPaths:**
  - `src/models/__init__.py`
  - `src/models/movement.py`
  - `tests/test_movement.py`
- **Especificación e Implementación:**
  Para responder *"¿cómo se movería este mismo lanzamiento en el aire del Harp Helú vs nivel del mar?"*, implementamos la **Etapa A (Física de Movimiento)**:
  1. Identificación intra-pitcher: Para cada `(pitcher_anon_id, year, pitch_id)` que tenga lanzamientos en distintas densidades $\hat{\rho}$, calculamos las diferencias centradas en la media del pitcher-pitcheo ($\Delta IVB_{ij}, \Delta HB_{ij}, \Delta \hat{\rho}_{ij}$), eliminando el efecto fijo del pitcher.
  2. Entrenar un modelo `PhysicsMovementTranslator` (`LightGBMRegressor` con restricciones de monotonía / interacción entre `rho_hat` y el quiebre base):
     - **Inputs permitidos:** `RelSpeed`, `SpinRate`, `SpinAxis_sin`, `SpinAxis_cos`, `RelHeight`, `RelSide_norm`, `Extension`, `VertRelAngle`, `HorzRelAngle_norm`, `arm_slot_deg`, `rho_hat`.
     - **Método clave:** `translate_movement(df: pd.DataFrame, target_rho: float) -> pd.DataFrame`:
       Dado un lanzamiento observado en densidad $\hat{\rho}_{\text{obs}}$ con $(\text{IVB}_{\text{obs}}, \text{HB}_{\text{obs}}, \text{ZoneSpeed}_{\text{obs}})$, devuelve el lanzamiento con su movimiento y velocidad de llegada ajustados a `target_rho`:
       $$\text{IVB}(\rho^*) = \text{IVB}_{\text{obs}} + \left[\widehat{\text{IVB}}(X, \rho^*) - \widehat{\text{IVB}}(X, \hat{\rho}_{\text{obs}})\right]$$
       (análogo para `HorzBreak_norm`, `TotalInducedBreak`, `SpinToBreakRatio` y `ZoneSpeed`).
- **Criterios de Aceptación:**
  1. Si `target_rho == df['rho_hat']`, `translate_movement` devuelve exactamente el movimiento original ($\Delta = 0.0$).
  2. Si `target_rho < df['rho_hat']` (pasando de nivel del mar a CDMX), para rectas de alto IVB y curvas, la magnitud media de `TotalInducedBreak` disminuye (`< 0`), mientras que `ZoneSpeed` aumenta ligeramente por menor arrastre.
  3. `assert_no_leakage(feature_cols, "stuff")` pasa sin errores.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_movement.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Se conserva el residuo individual del pitcheo ($\text{IVB}_{\text{obs}} + \hat{f}(\rho^*) - \hat{f}(\hat{\rho}_{\text{obs}})$) en vez de reemplazar todo el movimiento por una media suave?
  - [x] ¿Ninguna variable de ubicación ni resultado entra en `PhysicsMovementTranslator`?

#### Notas de Revisión
- `src/models/movement.py:255-315` (RESUELTO): Los regresores de magnitud de IVB y HB (`_model_mag_ivb`, `_model_mag_hb`) ahora se ajustan condicionados por el quiebre base en release (`_model_base_ivb`, `_model_base_hb`), incorporando la interacción explícita `rho_x_base_mag = rho_hat * abs(base_break)` y `monotone_constraints` (`+1` en `rho_hat` y `+1` en `rho_x_base_mag`), preservando el residuo individual observado. Validado en `test_signed_break_response_across_densities` para IVB positivo, IVB negativo y HB de ambos signos.

---

### `HACK-010` — Submodelo `Stuff+` en Sabores Neutral y CDMX (`src/models/stuff.py`)
- **State:** `APPROVED`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-009`
- **AllowedPaths:**
  - `src/models/stuff.py`
  - `tests/test_stuff.py`
- **Especificación e Implementación:**
  Implementar la clase `StuffPlusModel` en `src/models/stuff.py`:
  1. **Features autorizadas (`STUFF_FEATURES`):**
     - Físicas de release y movimiento: `RelSpeed`, `EffectiveVelo`, `ZoneSpeed`, `SpinRate`, `SpinAxis_sin`, `SpinAxis_cos`, `RelHeight`, `RelSide_norm`, `Extension`, `InducedVertBreak`, `HorzBreak_norm`, `TotalInducedBreak`, `BreakAngle_deg`, `VertRelAngle`, `HorzRelAngle_norm`, `SpinToBreakRatio`, `arm_slot_deg`, `PitcherThrows`.
     - Arsenal y túneles: `velo_diff_vs_fb`, `ivb_diff_vs_fb`, `hb_diff_vs_fb`, `tunnel_dist_release_in`, `tunnel_sim_release`, `tunnel_dist_170ms_in`, `tunnel_sim_170ms`, `pitch_usage_pct`.
     - **Verificación obligatoria:** Llamar a `assert_no_leakage(STUFF_FEATURES, "stuff")` en `__init__` y `fit`. (`VertApprAngle` y `HorzApprAngle` se excluyen de `Stuff+` puro o se evalúan en su versión neutralizada de altura media para no filtrar `PlateLocHeight`).
  2. **Arquitectura Multi-Target $\to$ Score+:**
     - Entrena modelos LightGBM para los targets físicos y un regresor de combinación (`RidgeCV` no negativo sobre predicciones *out-of-fold*) hacia `xrv_pitcher` evaluado en el **canal Stuff** (sin premiar ni castigar al pitcher por el *carry* extra del estadio en bolas bateadas).
     - Guarda la media $\mu_{\text{neutral}}$ y desviación estándar $\sigma_{\text{neutral}}$ de las predicciones evaluadas a $\hat{\rho}_0 = 1.000$ sobre el set de referencia LMB.
  3. **Dos sabores con la misma regla de escala:**
     - `predict_scores(df: pd.DataFrame, movement_translator: PhysicsMovementTranslator, rho_cdmx: float = 0.82) -> pd.DataFrame`:
       - **Sabor Neutral (`stuff_plus_neutral`):** traduce cada lanzamiento a $\hat{\rho}_0 = 1.000$ con `movement_translator`, recalcula diferenciales de movimiento y evalúa `100 + 10 * (pred_neutral - mu_neutral) / sigma_neutral`.
       - **Sabor CDMX (`stuff_plus_cdmx`):** traduce el mismo lanzamiento a `rho_cdmx`, recalcula diferenciales y evalúa con **la misma $\mu_{\text{neutral}}$ y $\sigma_{\text{neutral}}$** para que la pérdida o ganancia en CDMX sea directamente medible en puntos de la escala Plus.
       - **`delta_stuff_cdmx`:** `stuff_plus_cdmx - stuff_plus_neutral`.
  4. **Explicabilidad SHAP:** método `explain_pitch(df_row: pd.DataFrame) -> dict` que devuelve las contribuciones SHAP por feature.
- **Criterios de Aceptación:**
  1. Sobre el set de referencia evaluado en sabor neutral, `mean(stuff_plus_neutral) == 100.0` ($\pm 0.05$) y `std(stuff_plus_neutral) == 10.0` ($\pm 0.05$).
  2. `assert_no_leakage` garantiza que ninguna columna de `LOCATION_AND_COUNT_COLS` ni `FORBIDDEN_TARGET_COLS` entró al modelo.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_stuff.py
  ```
- **Reviewer Checklist:**
  - [x] **CRÍTICO:** ¿`stuff_plus_cdmx` se normaliza usando $(\mu_{\text{neutral}}, \sigma_{\text{neutral}})$ de nivel del mar? (Si se re-estandarizara a media 100 dentro de CDMX, el promedio de `delta_stuff_cdmx` sería artificialmente 0 y borraría el efecto de la altitud).
  - [x] ¿Se excluyeron todas las variables de `PlateLoc*`, `count`, `Balls`, `Strikes` y `BatterSide` de `Stuff+` puro?

#### Notas de Revisión
- `src/models/stuff.py:328`: `target_preds_dict` se llena con predicciones sobre las mismas filas usadas para ajustar cada submodelo; en `src/models/stuff.py:377` se entrena `Ridge(alpha=1.0, positive=True)` sobre ellas. La especificación exige predicciones *out-of-fold* y selección de regularización por validación cruzada. Genera predicciones OOF por `pitcher_anon_id` para los 8 submodelos, ajusta el combinador con esas predicciones y selecciona `alpha` mediante CV manteniendo coeficientes no negativos; después reajusta los submodelos para inferencia. Agrega una prueba que demuestre que ninguna fila del meta-entrenamiento fue predicha por un submodelo entrenado con ella.
- `src/models/stuff.py:343`: El regresor directo de `xrv_pitcher` se ajusta con todas las altitudes, aunque su etiqueta incluye el efecto de parque en bolas bateadas; limitar solo el ajuste del combinador a `No Altitude` en la línea 364 no elimina esa contaminación de las predicciones base. Entrena ese componente con una etiqueta de canal Stuff neutralizada respecto al parque, o ajústalo exclusivamente con observaciones `No Altitude` y documenta la limitación. Verifica con una prueba que alterar los resultados de bolas bateadas solo en `Extreme Altitude` no cambie el ajuste del canal Stuff.
- `src/models/stuff.py:470`: Los dos sabores usan `ivb_diff_vs_fb` y `hb_diff_vs_fb` tal como salen del traductor, que suma el cambio del pitcheo sin traducir la recta primaria de referencia. Recalcula ambos diferenciales contra la recta primaria traducida a la misma densidad antes de evaluar cada sabor, como exige la especificación, y prueba la consistencia del diferencial para un pitcher con recta y pitcheo secundario.

**Segunda revisión (51 pruebas pasan):**
- `src/models/stuff.py:417`: El filtro nuevo excluye solo `Extreme Altitude`; `Medium Altitude` sigue aportando etiquetas de `xrv_pitcher` y bolas bateadas afectadas por el carry del parque al canal Stuff. Ajusta los submodelos de resultado con observaciones `No Altitude` o con etiquetas neutralizadas por parque antes del ajuste OOF y final. Amplía la prueba de invariancia para alterar también los resultados de `Medium Altitude` y comprobar que el canal Stuff no cambia.
- `src/models/stuff.py:309`: `_recompute_arsenal_diffs_at_density` toma la media de las rectas presentes en el lote de predicción. Así, el mismo slider puede tener otro `ivb_diff_vs_fb` y otro score al predecirse solo o junto con más rectas del mismo pitcher. Guarda durante `fit` la referencia de la recta primaria por `(pitcher_anon_id, year)` y tradúcela a la densidad objetivo para todos los lotes; no derives la referencia de las filas que casualmente acompañan al lanzamiento. Agrega una prueba de invariancia de `predict_scores` entre una fila aislada y esa misma fila dentro de un lote.

**Tercera revisión (52 pruebas pasan):**
- `src/models/stuff.py:782`: `explain_pitch` calcula TreeSHAP solo para `self.booster_`, el regresor directo de `xrv_pitcher`, pero llama a esas contribuciones puntos de `Stuff+` en las líneas 786–789. El score real sale de `_predict_stuff_value`, que combina ocho submodelos con `self.meta_ridge_`; por tanto, `base_value + sum(shap_values)` no reconstruye el score que se muestra. Calcula atribuciones del modelo combinado en la misma escala neutral, o presenta estas contribuciones explícitamente como explicación exclusiva del regresor directo y agrega una explicación del score final. Añade una prueba de aditividad que compare las contribuciones reportadas con el `Stuff+` correspondiente al mismo lanzamiento.

**Cuarta revisión (53 pruebas pasan):**
- `src/models/stuff.py:617`: `mu_neutral_` y `sigma_neutral_` se calculan con `df_stuff_train`, que desde la línea 464 contiene solo lanzamientos de `No Altitude`. `docs/CONTRACT.md` §3 y el criterio de HACK-010 definen la referencia como los pitcheos LMB de la temporada evaluados a densidad neutral, incluidos los observados en `Medium` y `Extreme Altitude`. Mantén el entrenamiento de resultados en `No Altitude` para evitar carry, pero calibra la escala con todos los lanzamientos del conjunto de referencia `df_train` traducidos a `rho_hat=1.000` y con los diferenciales de arsenal recalculados. Cambia la prueba de media 100/desviación 10 para evaluar ese conjunto LMB completo, no solo `No Altitude`.

**Quinta revisión (APROBADA, 55 pruebas pasan, exit code 0 con `.venv\Scripts\python.exe -m pytest -q tests/test_stuff.py`):**
- AllowedPaths: el diff de HACK-010 solo toca `src/models/stuff.py` y `tests/test_stuff.py`. `STUFF_FEATURES` y `assert_no_leakage` no cambian.
- El conteo entra solo como offset: `src/models/stuff.py:495-501` (residuo `y - baseline_conteo`). `_predict_submodels` (`:614-625`) y `predict_scores` no leen `Balls`/`Strikes`; `tests/test_stuff.py:623` prueba que alterarlos no cambia `stuff_plus_neutral` y sí `predict_outcomes`. mu/sigma siguen calibradas a `rho=1.000` sobre `df_train` LMB completo (`:582-590`); se conservan OOF, entrenamiento en `No Altitude`, referencia de recta guardada en `fit`, SHAP aditivo (ahora sobre submodelos de residuo, `:823-842`).
- (a) `predict` -> alias de `predict_outcomes` (`:727`). Consumidores: `analysis/run_validation.py:46,120` solo usa `predict` en frames de validación con `Balls`/`Strikes`/`count`; `analysis/run_hypotheses.py` solo usa `predict_scores`. No rompe a nadie.
- (b) Baseline ajustado una vez fuera del OOF: es una media por conteo (pocas celdas, independiente de pitcher y features físicas), por lo que su influencia de una fila de validación es O(1/N_celda); fuga despreciable para el meta-entrenamiento.
- (c) Regresor LightGBM sobre residuo para binarios es coherente (modelo lineal de probabilidad sobre offset, recorte `[1e-6, 1-1e-6]` en `predict_outcomes`).
- Observaciones NO bloqueantes: (d) el docstring del módulo (`src/models/stuff.py:1-62`) no menciona el residuo/offset de conteo, `predict_outcomes` ni que los binarios son regresores; el §2 y §5 siguen describiendo clasificadores y xRV directo. Conviene actualizarlo en la próxima edición. (e) El offset usa el baseline ajustado en `No Altitude` (`df_stuff_train`), no el de todo el entrenamiento; HACK-012 debe documentar que su baseline de comparación puede diferir en cortes con altitud.

---

#### Reapertura del Orquestador (2026-10-20) — Stuff+ como residuo sobre el conteo
- **Motivo:** en HACK-012, Stuff+ pierde contra `CountBaselineModel` en `target_called_strike` y `target_out_on_pitch` en los 7 cortes, y por un margen mínimo en el RMSE de `xrv_pitcher` en Park. Esto pasa porque el baseline *es* el conteo y Stuff+ tiene prohibido usarlo como feature.
- **Cambio requerido:** los submodelos de resultado se entrenan sobre el **residuo** `y − baseline_conteo(y)`. Para eso se usa un `CountBaselineModel` (`src/baseline.py`, solo lectura) ajustado dentro de `fit` con el mismo conjunto de entrenamiento. El conteo entra **solo como offset** y nunca como feature: `STUFF_FEATURES` y `assert_no_leakage` no cambian.
  - El **score** `stuff_plus_neutral` / `stuff_plus_cdmx` sale únicamente de la parte residual (independiente del conteo). Conserva toda la escala 100/10, la calibración con $(\mu_{\text{neutral}}, \sigma_{\text{neutral}})$ y los arreglos de las cuatro revisiones previas (OOF, entrenamiento de resultados en `No Altitude`, referencia de recta guardada en `fit`, SHAP aditivo, escala sobre LMB completo).
  - Nuevo método `predict_outcomes(df) -> pd.DataFrame` con una columna por cada target de `TARGET_COLS` (mismos nombres que `CountBaselineModel.predict`): `baseline_conteo + residuo_stuff`, con los binarios recortados a `[1e-6, 1−1e-6]`. HACK-012 usará este método para comparar contra el baseline.
- **Criterios adicionales:** (3) los tests prueban que alterar `Balls`/`Strikes` no cambia `stuff_plus_neutral` pero sí `predict_outcomes`; (4) `predict_outcomes` en el fixture no produce NaN ni probabilidades fuera de `[0, 1]`.
- **Prohibido** ajustar hiperparámetros mirando los holdouts de HACK-012.

#### Segunda reapertura del Orquestador (2026-10-20) — quitar los proxies de ubicación
- **Motivo, de una investigación de sensibilidad de solo lectura:**
  - Con `STUFF_FEATURES` se reconstruye `PlateLocHeight` / `PlateLocSide` con R² fuera de muestra ≈ 0.995 / 0.994.
  - Los ángulos de soltura y los túneles a 170 ms concentran ~65 % del |SHAP|, y el IVB solo ~3 %.
  - La sensibilidad mediana de Stuff+ en Four-Seam es +0.09 puntos por pulgada de IVB.
  - La física es determinista: soltura + ángulos de soltura + velocidad + movimiento ⇒ ubicación en el plato. Es el mismo motivo por el que la especificación ya excluía `VertApprAngle`.
- **Cambio requerido:**
  1. Sacar de `STUFF_FEATURES`: `VertRelAngle`, `HorzRelAngle_norm`, `tunnel_dist_170ms_in` y `tunnel_sim_170ms`.
     - Siguen disponibles para Location+ y Pitching+ (HACK-011), donde la ubicación sí está permitida. Así se cumple AGENTS.md §3.6, que pide que ambos túneles existan como features del proyecto.
     - Se conservan `tunnel_dist_release_in` y `tunnel_sim_release`, que solo usan el punto de soltura.
     - Ajusta también la reconstrucción de la recta primaria (`_fit_primary_fb_reference`, `_recompute_arsenal_diffs_at_density`) para que no dependa de esos campos.
  2. Agrega a `assert_valid_stuff_features` una lista explícita `TRAJECTORY_PROXY_COLS` (`VertRelAngle`, `HorzRelAngle`, `HorzRelAngle_norm`, `VertApprAngle`, `HorzApprAngle`, `tunnel_dist_170ms_in`, `tunnel_sim_170ms`) que lance error si alguna entra a Stuff+.
  3. **Regularización:** elige hiperparámetros de LightGBM (`min_child_samples`, `num_leaves`, `reg_lambda`, `learning_rate` / `n_estimators`) con CV por `pitcher_anon_id` **solo dentro de No Altitude, year ≤ 2025**, nunca mirando los holdouts de HACK-012. Documenta la rejilla y la elección.
  4. Conserva todo lo demás aprobado: residuo sobre el conteo, `predict_outcomes`, OOF, entrenamiento de resultados en No Altitude, referencia de recta guardada en `fit`, SHAP aditivo, escala calibrada sobre el LMB completo con μ/σ neutrales.
- **Criterios adicionales:**
  - **(5)** Ninguna columna de `TRAJECTORY_PROXY_COLS` está en `feature_cols`. Lo verifica un test.
  - **(6)** Un script de diagnóstico, `analysis/stuff_diagnostics.py` (nuevo en AllowedPaths), reporta sobre datos reales en `reports/validation/stuff_diagnostics.csv`:
    - el R² fuera de muestra de reconstruir `PlateLocHeight` y `PlateLocSide` desde las features finales de Stuff+. **(Redefinido por el Orquestador con el usuario, 2026-10-20.)** El umbral fijo de 0.30 no tenía referencia. En la primera corrida dio 0.312 en altura, y la ablación lo atribuye a la velocidad y al tipo de pitcheo, no a la trayectoria. El criterio ahora es **R²_completo − R²_referencia ≤ 0.20** en cada eje, donde R²_referencia es el R² medido con solo las features de velocidad de Stuff+ (`RelSpeed`, `EffectiveVelo`, `ZoneSpeed`, `velo_diff_vs_fb`) más `pitch_id`. Ambas cifras se reportan tal cual en el CSV;
    - la sensibilidad mediana de Stuff+ por pulgada de IVB y HB y por mph, por `pitch_id`, con los derivados recalculados;
    - el ΔStuff+ CDMX medio por `pitch_id` con IC bootstrap por pitcher.
    Estas cifras se reportan tal cual; no hay umbral de sensibilidad que forzar.
- **Efecto aguas abajo:** HACK-012, HACK-013 y HACK-014 vuelven a `AVAILABLE` y deben re-correrse con el modelo nuevo. El criterio de HACK-012 se revisará con los resultados reales.
- **AllowedPaths de esta reapertura:** `src/models/stuff.py`, `tests/test_stuff.py`, `analysis/stuff_diagnostics.py`, `reports/validation/stuff_diagnostics.csv`.

**Sexta revisión (APROBADA, 66 pruebas pasan, exit code 0 con `.venv\Scripts\python.exe -m pytest -q tests/test_stuff.py`):**
- AllowedPaths: el diff de código solo toca `src/models/stuff.py`, `tests/test_stuff.py`, `analysis/stuff_diagnostics.py` y `reports/validation/stuff_diagnostics.csv`. (`.claude/` sin seguimiento es del entorno, no del modelo.)
- Criterio 5: `STUFF_FEATURES` ya no contiene `VertRelAngle`, `HorzRelAngle_norm`, `tunnel_dist_170ms_in` ni `tunnel_sim_170ms` (`src/models/stuff.py:114-126`). `TRAJECTORY_PROXY_COLS` (`:134-140`) la valida `assert_valid_stuff_features` (`:185-191`, ValueError) antes de la lista blanca. Un grep de las columnas prohibidas en `stuff.py` solo da docstrings y la lista de proxies: se eliminaron la normalización en `_prepare_stuff_features`, `num_cols` de `_fit_primary_fb_reference` y `fb_row` de `_recompute_arsenal_diffs_at_density`. Las medianas salen de `feature_cols`. Los tests (`tests/test_stuff.py:677-700`) cubren `feature_cols`, SHAP y el rechazo de cada proxy.
- Criterio 6 (redefinido, no reabierto): se re-corrió `python -m analysis.stuff_diagnostics` con una salida temporal y el CSV resultó **idéntico byte a byte** al versionado. Exceso sobre la referencia: altura 0.1275 (0.3123 − 0.1848), lado 0.1117 (0.1378 − 0.0261); `PlateLocSide_norm` 0.0917. Máximo 0.1275 ≤ 0.20. En el código (`analysis/stuff_diagnostics.py:90-120`), el modelo completo y la referencia usan la misma muestra `d`, los mismos `folds` (un `GroupKFold` por `pitcher_anon_id`, calculado una sola vez) y el mismo `_lgbm_loc()`. No hay fuga entre folds.
- Regularización: `_select_lgbm_params` (`src/models/stuff.py:651-713`) recibe `X_train`, que viene de `df_train` filtrado a `year <= 2025` (`:477`) y a `No Altitude` (`:510-513`). Usa `GroupKFold` por pitcher y no recibe holdouts de HACK-012. La rejilla de 12 combinaciones y la elección (`nl=7, mc=1000, rl=300`, RMSE 0.21393) están documentadas en el docstring (sección 5) y en el CSV. La diferencia entre combinaciones es ~1e-4: la elección es estable.
- Se conserva lo aprobado: residuo sobre el conteo y `predict_outcomes`, OOF, entrenamiento en No Altitude, referencia de recta guardada en `fit`, SHAP aditivo y escala 100/10 con μ/σ neutrales sobre el LMB completo (las pruebas heredadas pasan). Checklist original: CDMX con μ/σ neutrales y sin `PlateLoc`/conteo/`BatterSide`.
- Plausibilidad: Four-Seam −4.57 en CDMX (IC −4.84/−4.28) y sensibilidad IVB +2.29 pts/in son coherentes. La menor densidad reduce Magnus y arrastre, el IVB baja unas pulgadas y el modelo lo premia con signo positivo. Los signos mixtos en los demás pitcheos tienen IC que cruzan 0 o son pequeños (|s| < 0.7). Son una no linealidad aceptable (quiebre negativo del Curveball/Slider, interacción velocidad-movimiento), no un defecto. Aviso no bloqueante: el sinker tiene sensibilidad en HB positiva y el Changeup tiene IVB negativa; HACK-013 debe presentarlo como resultado, no forzarlo.
- Observación no bloqueante: la sensibilidad de `ci_low`/`ci_high` son cuartiles Q25/Q75 y no un IC (está documentado en el docstring, `analysis/stuff_diagnostics.py:34-36`).

### `HACK-011` — Submodelos `Location+` y `Pitching+` Multi-Target (`src/models/location_pitching.py`)
- **State:** `APPROVED`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-008`
- **AllowedPaths:**
  - `src/models/location_pitching.py`
  - `tests/test_location_pitching.py`
- **Especificación e Implementación:**
  Implementar en `src/models/location_pitching.py`:
  1. **`LocationPlusModel`:**
     - **Features autorizadas:** `PlateLocHeight`, `PlateLocSide_norm`, `in_strike_zone`, `dist_from_zone_center`, `dist_to_zone_edge`, `is_shadow_zone`, `zone_quadrant`, `Balls`, `Strikes`, `Outs`, `BatterSide`, `PitcherThrows`, `is_same_hand`, `pitch_id` (el cluster invariante del pitcheo).
     - **Prohibidas:** Velocidad, spin rate, IVB, HB (mide comando puro condicionado al tipo de pitcheo y conteo).
     - Escala a `location_plus_neutral` y `location_plus_cdmx` (`100 + 10·z`).
  2. **`PitchingPlusModel`:**
     - **Features autorizadas:** Unión de `STUFF_FEATURES` + `LOCATION_FEATURES` + `VertApprAngle` + `HorzApprAngle_norm` + `Inning` + `prev_pitch_id` + `rho_hat`.
     - Predice explícitamente los **8 targets multi-objetivo** pedidos por las bases (`P(whiff)`, `P(chase)`, `P(called_strike)`, `P(weak_contact)`, `P(groundball)`, `P(hard_contact)`, `P(out_on_pitch)`, `xrv_pitcher`), los combina mediante regresión sobre `xrv_pitcher` y produce `pitching_plus_neutral` y `pitching_plus_cdmx`.
- **Criterios de Aceptación:**
  1. `LocationPlusModel` rechaza cualquier columna de velocidad o movimiento físico.
  2. `PitchingPlusModel.predict_all_targets(df)` devuelve las 8 probabilidades/valores calibrados además de `pitching_plus_neutral` y `pitching_plus_cdmx`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_location_pitching.py
  ```
- **Reviewer Checklist:**
  - [x] ¿`LocationPlusModel` no tiene acceso a `RelSpeed`, `SpinRate`, `InducedVertBreak` ni `HorzBreak`?
  - [x] ¿`PitchingPlusModel` predice los 8 targets completos de la tabla de las bases?

#### Notas de Revisión
- `src/models/location_pitching.py:85-185` (RESUELTO): Se eliminó `LOCATION_MODEL_FEATURES` y se retiró `rho_hat` de `LocationPlusModel`. `assert_valid_location_features` valida mediante whitelist estricta contra `LOCATION_FEATURES`, rechazando cualquier otra variable física no enumerada, y `location_plus_neutral` y `location_plus_cdmx` se evalúan en la misma escala neutral.
- `src/models/location_pitching.py:375-420` (RESUELTO): Se eliminó el valor fijo `0.82` y la afirmación en docstrings; ahora `PitchingPlusModel.fit` calcula empíricamente `self.rho_extreme_median_` a partir de `Extreme Altitude` en los datos de entrenamiento ($\approx 0.852$ en LMB 2024–25, documentando que agrupa varios parques de altura).

---

### `HACK-012` — Suite de Validación en los 3 Holdouts vs Baseline Nulo (`analysis/run_validation.py`)
- **State:** `APPROVED` (re-evaluado 2026-10-21 con el Criterio 1 actualizado)
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-010, HACK-011`
- **AllowedPaths:**
  - `analysis/__init__.py`
  - `analysis/run_validation.py`
  - `tests/test_validation_suite.py`
  - `reports/validation/holdout_metrics.csv`
  - `reports/validation/calibration_deciles.csv`
  - `reports/validation/year_to_year_stability.csv`
  - `reports/validation/holdout_metrics_partial.csv`
  - `reports/validation/calibration_deciles_partial.csv`
  - `reports/validation/year_to_year_stability_partial.csv`
  - `reports/validation/run_status.json`
  - `reports/validation/partition_timing.json`
- **Especificación e Implementación:**
  Script y módulo que ejecuta la validación obligatoria (Criterios 1 y 2 de la rúbrica = 50 % del puntaje total):
  1. Corre los 3 esquemas de `src/splits.py`:
     - **Pitcher Holdout** (5-fold `GroupKFold` por `pitcher_anon_id` en 2024–2025).
     - **Season Holdout** (Entrena en 2024–2025 $\to$ evalúa en 2026).
     - **Park Holdout** (Entrena en `No + Medium` $\to$ evalúa en `Extreme`).
  2. En cada holdout compara `Stuff+`, `Location+` y `Pitching+` contra `CountBaselineModel` (`xrv_pitcher` medio por conteo), reportando RMSE, MAE, Log-Loss de cada sub-target, correlación con `xrv_pitcher` futuro y tabla de calibración por deciles.
  3. Calcula la **estabilidad año-a-año** (correlación de Pearson y Spearman entre el promedio por pitcher en 2024 vs 2025 para lanzadores con $\ge 150$ pitcheos en ambos años).
  4. Guarda resultados agregados en:
     - `reports/validation/holdout_metrics.csv`
     - `reports/validation/calibration_deciles.csv`
     - `reports/validation/year_to_year_stability.csv`
- **Criterios de Aceptación:**
  1. **(Actualizado por el usuario y el Orquestador, 2026-10-21, tras quitar a Stuff+ los proxies de ubicación en la segunda reapertura de HACK-010.)** En los 7 cortes de los 3 holdouts:
     - **`Pitching+`** supera (menor RMSE y menor Log-Loss) al `CountBaselineModel` en **todos** los targets.
     - **`Stuff+`** se juzga solo en `xrv_pitcher` y en los targets de *stuff*: `target_whiff`, `target_chase`, `target_hard_contact`, `target_weak_contact`.
       - En los 4 targets binarios: menor RMSE y menor Log-Loss que el baseline.
       - En `xrv_pitcher`: RMSE ≤ RMSE del baseline + **0.0001** (≈ 0.05 % del RMSE). Si Stuff+ no es menor, el corte se reporta como **empate**, nunca como victoria.
     - `target_called_strike`, `target_out_on_pitch` y `target_groundball` de Stuff+ se **reportan sin exigir**. Dependen sobre todo de ubicación y conteo, que Stuff+ tiene prohibido usar (AGENTS.md §3.3). Que Stuff+ no les gane al conteo es lo esperado y se documenta tal cual.
  2. Los 3 archivos CSV existen en `reports/validation/` con cifras reales calculadas.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_validation_suite.py
  ```
- **Reviewer Checklist:**
  - [x] ¿En cada fold/holdout tanto `PhysicsMovementTranslator` como los modelos se ajustan **únicamente con `train_idx`** antes de predecir en `test_idx`?

#### Nota de ejecución
- Auditoría del revisor: corrida completa sobre 634,624 filas; `run_status.json` confirma los cinco folds de pitcher, season, park y estabilidad completa. Los tres CSV agregados están presentes. Verify: 5 passed, exit code 0. Estabilidad 2024–2025 es in-sample y está etiquetada así en el CSV.
- La aceptación sigue incumplida. En cada uno de los siete cortes, Stuff+ tiene mayor RMSE y Log-Loss que el baseline para `target_called_strike` y `target_out_on_pitch`; en Park, también tiene mayor RMSE de `xrv_pitcher` (0.211702 vs 0.211461). Las cifras completas permanecen en `reports/validation/holdout_metrics.csv`; no se alteraron los resultados.

#### Notas de Revisión
- `reports/validation/holdout_metrics.csv` (criterio de aceptación 1): los registros de Stuff+ muestran RMSE y Log-Loss mayores que CountBaselineModel para `target_called_strike` y `target_out_on_pitch` en los siete holdouts; Park también empeora RMSE de `xrv_pitcher`. Corrige el modelo/pipeline dentro de sus tareas y AllowedPaths autorizados y vuelve a ejecutar la validación hasta que Stuff+ y Pitching+ cumplan el criterio en los tres esquemas; no cambies ni filtres las métricas observadas para aparentar superioridad.
- `analysis/run_validation.py:69-104`: revisión actual confirma que el traductor y los modelos se ajustan desde `train` formado con `train_idx`, y el Verify Command pasa (5 pruebas). Sin embargo, el criterio de aceptación 1 sigue fallando en el CSV actual: Stuff+ empeora RMSE y Log-Loss de `target_called_strike` y `target_out_on_pitch` en los siete holdouts, y también RMSE de `xrv_pitcher` en Park. Corrige el modelo/pipeline y vuelve a correr el análisis; conserva todas las métricas observadas.
- `reports/validation/holdout_metrics.csv`, `reports/validation/calibration_deciles.csv`, `reports/validation/year_to_year_stability.csv` (AllowedPaths): estos archivos aparecen modificados y los artefactos `*_partial.csv`, `run_status.json` y `partition_timing.json` aparecen nuevos, pero `reports/` no está en `AllowedPaths`. El orquestador debe reconciliar la lista `AllowedPaths` con los reportes agregados requeridos antes de reentregar; limitar la autorización a esos artefactos de salida agregados y no incluir datos pitch-by-pitch.

#### Veredicto del Revisor (2026-10-20): APPROVED
- Verify: `.venv\Scripts\python.exe -m pytest -q tests/test_validation_suite.py` -> 5 passed, exit 0. `run_status.json`: complete=true, 7 cortes + estabilidad; 634,624 filas.
- Alcance: `analysis/run_validation.py`, `tests/test_validation_suite.py` y `analysis/__init__.py` sin cambios; solo cambian reportes de `reports/validation/` (AllowedPaths). Los cambios en `src/models/stuff.py`, `analysis/run_hypotheses.py`, `reports/altitude/` pertenecen a HACK-010/013.
- Criterio 1 (programatico sobre `holdout_metrics.csv`, 7 cortes x 8 targets): Stuff+ y Pitching+ tienen menor RMSE y menor Log-Loss que CountBaselineModel en 100 % de las celdas (0 fallas). Margen minimo Stuff+: dRMSE 0.001853, dLogLoss 0.009173 (`target_hard_contact`, pitcher_fold_1). Pitching+: dRMSE 0.005429, dLogLoss 0.025133 (`target_hard_contact`, pitcher_fold_2). Minimo en `xrv_pitcher` (RMSE): Stuff+ 0.002712, Pitching+ 0.008752. Location+ no es parte del criterio (pierde en groundball/out_on_pitch en algunos cortes; se conserva sin filtrar).
- Criterio 2: los 3 CSV existen con cifras reales (224 filas de metricas, 280 de calibracion sin NaN, 3 filas de estabilidad). Estabilidad: >=150 pitcheos en ambos anios (`run_validation.py:136`), n=202 pitchers, `score_basis=in_sample_train_2024_2025`.
- Checklist: `_score_split` (`run_validation.py:90-101`) construye traductor, Stuff+, Location+, Pitching+ y baseline solo con `train = df.loc[train_idx]`; el `CountBaselineModel` interno de Stuff+ se crea dentro de `fit` (`src/models/stuff.py:495`) sobre train.
- Sin fuga por `fit_lmb_run_values`: se ajusta con `train` (`run_validation.py:96`) y su tabla solo se aplica a test via `attach_targets`; no usa filas de test.
- Observacion heredada (offset de Stuff+ ajustado solo con `No Altitude`, `stuff.py:465,495`, vs baseline de comparacion con todo el train): no invalida la comparacion; es un desajuste que perjudica a Stuff+ (menos datos/distinto parque) y aun asi gana con margen. Solo requiere documentarse en el informe final; en Park es equivalente al no haber Extreme en train.

#### Notas de Revisión (re-corrida tras la segunda reapertura de HACK-010)
- Verify: `.venv\Scripts\python.exe -m pytest -q tests/test_validation_suite.py` -> 5 passed, exit 0. `run_status.json`: complete=true, 7 cortes + estabilidad.
- Alcance OK: `analysis/run_validation.py`, `tests/test_validation_suite.py` y `analysis/__init__.py` sin cambios; solo cambian CSV/JSON de `reports/validation/` (AllowedPaths). El resto del working tree es de HACK-010/013 (`reports/validation/stuff_diagnostics.csv` es de HACK-010).
- Criterio 2 OK: 224 filas de métricas, 280 de calibración, 3 de estabilidad con cifras reales; estabilidad etiquetada `in_sample_train_2024_2025` (Stuff+ r = 0.7415, Spearman 0.7554, n = 202).
- Checklist OK: `run_validation.py:37-41` y `:90-101` ajustan traductor, modelos y baseline solo con `train = df.loc[train_idx]`.
- **Criterio 1 FALLA (evaluado tal como está escrito, sin reinterpretar).** Pitching+: 0 fallas en 105 comparaciones (7 cortes x 8 targets + xrv_pitcher, RMSE y Log-Loss). Stuff+: 8 fallas (margen = Stuff+ - baseline; positivo = peor):

| Corte | Target | Métrica | Stuff+ | Baseline | Margen |
|---|---|---|---|---|---|
| pitcher_fold_1 | target_called_strike | Log-Loss | 0.377204 | 0.376496 | +0.000708 |
| pitcher_fold_1 | xrv_pitcher | RMSE | 0.209156 | 0.209139 | +0.000017 |
| pitcher_fold_2 | target_called_strike | Log-Loss | 0.376461 | 0.374278 | +0.002182 |
| pitcher_fold_4 | target_called_strike | Log-Loss | 0.376730 | 0.376517 | +0.000213 |
| pitcher_fold_5 | target_called_strike | Log-Loss | 0.377230 | 0.377011 | +0.000219 |
| season | target_called_strike | Log-Loss | 0.404960 | 0.402812 | +0.002148 |
| park | target_called_strike | Log-Loss | 0.369356 | 0.369261 | +0.000094 |
| park | xrv_pitcher | RMSE | 0.211470 | 0.211461 | +0.000009 |

  Se confirma lo informado por el Orquestador: Log-Loss de `target_called_strike` falla en 6 de 7 cortes (pasa pitcher_fold_3), el RMSE de `target_called_strike` pasa en los 7, y el RMSE de `xrv_pitcher` falla en pitcher_fold_1 y park. El resto de targets y cortes pasa. Location+ (fuera del criterio) sigue perdiendo en groundball y out_on_pitch.
- Para cerrar: la corrección requiere una decisión del usuario sobre el criterio de aceptación o sobre el modelo (HACK-010). No se modificaron métricas, código ni reportes.

#### Veredicto del Revisor (2026-10-21, criterio 1 actualizado): APPROVED
- Verify: `pytest -q tests/test_validation_suite.py` -> 5 passed, exit 0 (corrida anterior; ni `holdout_metrics.csv` ni el código cambiaron desde entonces). Sin cambios a código ni a métricas en esta evaluación.
- Pitching+ (7 cortes, todos los targets, RMSE y Log-Loss): 0 fallas. Margen mínimo (baseline - modelo): RMSE 0.005429, Log-Loss 0.025133. Log-Loss de `xrv_pitcher` es NaN por ser continuo y no aplica.
- Stuff+ binarios de stuff (7 cortes, RMSE y Log-Loss): 0 fallas. Márgenes mínimos: whiff RMSE 0.002233 / LL 0.005269; chase RMSE 0.000241 / LL 0.000430; hard_contact RMSE 0.000193 / LL 0.000744; weak_contact RMSE 0.000178 / LL 0.000488.
- Stuff+ `xrv_pitcher` (RMSE <= baseline + 0.0001): 7/7 dentro de tolerancia. Empates (no victorias):

| Corte | Stuff+ RMSE | Baseline RMSE | Delta | Resultado |
|---|---|---|---|---|
| pitcher_fold_1 | 0.209156 | 0.209139 | +0.000017 | EMPATE |
| pitcher_fold_2 | 0.207804 | 0.207827 | -0.000023 | gana |
| pitcher_fold_3 | 0.207315 | 0.207348 | -0.000033 | gana |
| pitcher_fold_4 | 0.208430 | 0.208516 | -0.000086 | gana |
| pitcher_fold_5 | 0.207861 | 0.207897 | -0.000036 | gana |
| season | 0.217513 | 0.217557 | -0.000044 | gana |
| park | 0.211470 | 0.211461 | +0.000009 | EMPATE |

- Reportados sin exigir (Stuff+): `called_strike` pierde en Log-Loss en 6 de 7 cortes (hasta +0.002182 en pitcher_fold_2); `out_on_pitch` y `groundball` no se exigen. Se documentan tal cual.
- Criterio 2 y checklist: cumplen (ver notas previas); traductor, modelos y baseline se ajustan solo con `train_idx` (`run_validation.py:37-41, 90-101`). Las cifras de xRV en empate deben presentarse como empate en el informe final.

---

### `HACK-013` — Descomposición Canal Stuff vs. Canal Parque y Pruebas H1–H4 (`analysis/run_hypotheses.py`)
- **State:** `APPROVED`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-010, HACK-011`
- **AllowedPaths:**
  - `analysis/run_hypotheses.py`
  - `tests/test_hypotheses_channels.py`
  - `reports/altitude/channel_decomposition.csv`
  - `reports/altitude/delta_by_pitch_shape.csv`
  - `reports/altitude/hypotheses_h1_h4_summary.csv`
- **Especificación e Implementación:**
  Responde al Criterio 3 de la rúbrica (25 %) y a las 4 hipótesis de las bases:
  1. **Delta Intra-Pitcher por Pitch Shape:**
     Para los pitchers que lanzaron el mismo `pitch_id` tanto en `No Altitude` como en `Extreme Altitude`, calcula el cambio real intra-pitcher en `InducedVertBreak`, `HorzBreak_norm`, `TotalInducedBreak`, `ZoneSpeed`, `target_whiff`, `home_run` y `delta_stuff_cdmx`, con intervalos de confianza al 95 % por bootstrap de cluster (`pitcher_anon_id`).
  2. **Descomposición Mediacional (Canal Stuff vs. Canal Parque):**
     Descompone la diferencia total de `xrv_pitcher` en `Extreme Altitude` ($\Delta \text{xRV}_{\text{total}}$) en:
     - **Canal Stuff ($\Delta \text{xRV}_{\text{stuff}}$):** el cambio en `xRV` esperado causado únicamente por la pérdida de movimiento físico $(\Delta \text{IVB}, \Delta \text{HB})$ manteniendo constante la función de pago de nivel del mar.
     - **Canal Parque ($\Delta \text{xRV}_{\text{parque}}$):** el cambio adicional en `xRV` en pelotas puestas en juego (mayor tasa de HR y extrabases por menor resistencia del aire durante el vuelo del batazo) para un mismo ángulo y velocidad de salida.
  3. **Pruebas Estadísticas Formales de H1, H2, H3 y H4:**
     - **H1:** Caída de movimiento inducido por spin en cada `pitch_id`.
     - **H2:** Cambio relativo de `Stuff+` de `Changeup`/`Splitter` frente a `Four-Seam`/`Curveball`/`Slider`.
     - **H3:** Interacción entre tasa de `GroundBall` del pitcher y prevención de carreras/HR en `Extreme Altitude` vs `No Altitude`.
     - **H4:** Regresión de `xrv_pitcher` y `whiff` sobre `SpinRate` bruto vs `SpinToBreakRatio` y movimiento observado en altitud.
  4. Exporta tablas limpias a `reports/altitude/`:
     - `delta_by_pitch_shape.csv`
     - `channel_decomposition.csv`
     - `hypotheses_h1_h4_summary.csv`
- **Criterios de Aceptación:**
  1. Los 3 archivos CSV se generan en `reports/altitude/` con intervalos de confianza al 95 % (`ci_low`, `ci_high`, `p_value`, `n_pitchers`).
  2. **(Actualizado por el Orquestador, 2026-10-20)** Descomposición de **3 términos**: `canal_stuff + canal_parque_bip + residuo_no_atribuido = total` dentro de tolerancia numérica (`atol=1e-10`). El residuo es un término propio, con su CI bootstrap por pitcher, y su etiqueta dice que no se atribuye a ningún canal. Nunca se suma al canal parque ni se presenta como tal. `channel_decomposition.csv` reporta además `share_of_total` para cada término. La etiqueta `NOT_ESTIMABLE` de dos canales se elimina; queda `decomposition_type = "three_term"`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_hypotheses_channels.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Se usó `pitch_id` invariante (y no solo `AutoPitchType`) para no contaminar H1 y H2 con el reetiquetado de TrackMan?
  - [x] ¿Los errores estándar están agrupados por `pitcher_anon_id`?

#### Nota de ejecución
- Auditoría del revisor: los tres reportes actuales existen en `reports/altitude/` y contienen los campos de intervalo requeridos. Verify: 6 passed, exit code 0. El análisis usa `pitch_id` y bootstrap por pitcher; H3 restringe la comparación a pitchers observados en ambas altitudes. La aceptación aditiva sigue fallando: total = −0.0141315904, Stuff = −0.0011652544, parque BIP = −0.0060007047 y residuo no atribuido = −0.0069656314; Stuff + parque = −0.0071659590, que no reconstruye el total.

#### Notas de Revisión
- `analysis/run_hypotheses.py:200-217`: aunque Stuff y parque BIP se estiman por separado y se informa `unallocated`, la especificación exige que **Stuff + parque** reconstruya el efecto total. El resultado actual deja un residuo −0.0069656314 xRV por pitcheo; Stuff + parque = −0.0071659590 frente a total = −0.0141315904. Estima un canal Stuff y un canal parque que reconstruyan el total según el criterio, o marca la descomposición como no estimable y solicita al orquestador ampliar AllowedPaths si la solución requiere cambios fuera de `analysis/run_hypotheses.py` y `tests/test_hypotheses_channels.py`. Conserva el canal parque condicionado a BIP, ExitSpeed y Angle y reporta el residuo explícitamente; no lo renombres ni lo presentes como canal parque.
- `analysis/run_hypotheses.py:129-165`: el problema previo de H3 (incluir pitchers no pareados) está resuelto en el código actual. Mantén la comparación pareada por `pitcher_anon_id`, el GB basal de No Altitude y el bootstrap agrupado al corregir el incumplimiento de descomposición.
- `analysis/run_hypotheses.py:129-146, 238-263` y `reports/altitude/channel_decomposition.csv`: el cambio nuevo etiqueta la descomposición `NOT_ESTIMABLE`, conserva el residuo separado y el Verify Command pasa (6 pruebas); los CSV requeridos están actualizados. Sin embargo, el criterio de aceptación vigente exige que Stuff + parque reconstruya el total dentro de tolerancia, y el reporte actual aún muestra un residuo medio de −0.006965631382 xRV/pitcheo. La etiqueta honesta resuelve el problema de interpretación, pero no satisface ese criterio. No aprobar hasta que se obtenga una descomposición aditiva válida o el orquestador actualice explícitamente el criterio de aceptación para permitir una salida `NOT_ESTIMABLE` cuando los canales no se identifican; no reasignar el residuo al canal parque.
- `analysis/run_hypotheses.py:200-217`: persiste el incumplimiento del criterio de aceptación 2. En los resultados actuales, total = −0.0141315904, Stuff = −0.0011652544 y parque = −0.0060007047; Stuff + parque = −0.0071659590, diferencia −0.0069656314 xRV/pitcheo respecto al total. El residuo está etiquetado explícitamente como no atribuido, pero la aceptación exige que los dos canales reconstruyan el total dentro de tolerancia; determina una descomposición válida o marca el análisis como no estimable y coordina una corrección de especificación sin presentar el residuo como canal parque. Verify pasa (6 pruebas); el checklist de `pitch_id` invariante y bootstrap por pitcher se cumple.
- `reports/altitude/channel_decomposition.csv`, `reports/altitude/delta_by_pitch_shape.csv`, `reports/altitude/hypotheses_h1_h4_summary.csv` (AllowedPaths): son salidas requeridas y aparecen sin seguimiento, pero `reports/` no está en `AllowedPaths`. El orquestador debe reconciliar la lista para autorizar solo estos reportes agregados requeridos antes de reentregar.

- **Revisión independiente 2026-10-20 (APPROVED):** Verify `python -m pytest -q tests/test_hypotheses_channels.py` -> 7 passed, exit 0. Diff limitado a `analysis/run_hypotheses.py` y `tests/test_hypotheses_channels.py`; CSV de `reports/altitude/` solo los 3 de AllowedPaths. CSV reales: total = -0.0141316, stuff = -0.0003433, parque BIP = -0.0060007, residuo no atribuido = -0.0077876; suma - total = 9.9e-17 (atol 1e-10 OK); share_of_total suma 1.0 (0.024/0.425/0.551); `decomposition_type = three_term`, sin `NOT_ESTIMABLE`; residuo con CI/p/n_pitchers=778 propios y etiqueta `residuo_no_atribuido (no se atribuye a ningún canal)`. `pitch_id` invariante (run_hypotheses.py:169-170) y bootstrap/agrupación por `pitcher_anon_id` (:72, :195). Conversión Score+ -> xRV (:204-205) correcta: `stuff.py:684-692` define Score+ = 100 + 10*(pred-mu)/sigma y `pred` es la salida del meta-ridge entrenado sobre el residuo de `xrv_pitcher` (+ = bueno para el pitcher, CONTRACT), así que Δpred = (Δ Score+/10)*sigma_neutral_ está en unidades xRV con signo coherente. La caída de Stuff de -0.00117 a -0.00034 es plausible: al ser residuo sobre el conteo, el modelo ya no absorbe varianza de conteo y el meta-ridge contrae más; el orden de magnitud sigue siendo pequeño y el CI excluye 0 (no verificado con reentrenamiento independiente). Honestidad: parque descrito como contraste descriptivo en BIP, "no causal" (:231, :345); el 55 % residual queda sin atribuir; sin cifras inventadas. Observación no bloqueante: Stuff explica solo ~2 % del total y el residuo es el término mayor; conviene decirlo así en el informe final.
- **Revisión independiente de la re-corrida (2026-10-20, tras segunda reapertura de HACK-010): APPROVED.** Verify `python -m pytest -q tests/test_hypotheses_channels.py` -> 7 passed, exit 0. La re-corrida solo cambió los 3 CSV de `reports/altitude/`; `analysis/run_hypotheses.py` y su test sin cambios (el resto del working tree es HACK-010). Cifras nuevas: total = -0.014132; Stuff = -0.001278 (9.0 %, IC -0.001339/-0.001214); parque BIP = -0.006001 (42.5 %); residuo no atribuido = -0.006853 (48.5 %); suma - total = 1.0e-16 (atol 1e-10 OK), shares suman 1.0, `decomposition_type = three_term`, residuo con etiqueta propia, CI y n_pitchers=778. Checklist: `pitch_id` (run_hypotheses.py:169-170) y bootstrap por `pitcher_anon_id`. Conversión Score+ -> xRV (:204-205) correcta con sigma_neutral_ = 0.0430. H1-H4 bien planteadas; Sinker (HB) y Changeup se reportan tal cual, sin forzar. Parque descrito como no causal. Observaciones no bloqueantes: (a) Stuff pasó de 2.4 % a 9.0 % pero sigue siendo pequeño; el residuo (48.5 %) es el mayor término junto con el parque, y el informe debe decirlo. (b) Discrepancia a explicar, no bloqueante: reproduje el contrafactual de `run_hypotheses.py` con `models/stuff_model.joblib` y rho = mediana Extreme. El delta medio por fila es -0.206 Score+ en No Altitude (296,675 filas) y +0.078 en Extreme (182,474 filas), muy por debajo del ΔStuff+ CDMX medio de -1.52 de `stuff_diagnostics.csv`. El canal Stuff (-0.00128 xRV) es coherente con ese contrafactual de la re-corrida pero no con el -1.52 (que implicaría -0.0066 xRV); probablemente se deba a muestra/rho/modelo distintos en el diagnóstico, sin confirmar. El canal Stuff debe leerse como cota baja y el informe no debe citar -1.52 y 9.0 % como si fueran la misma medida. (c) En `delta_by_pitch_shape.csv` `delta_stuff_cdmx` es la diferencia Extreme-vs-Neutral del delta contrafactual por fila (Four-Seam +2.19), no el ΔStuff+ de HACK-010 (-4.57); no deben compararse directamente.

---

### `HACK-014` — Exportador Oficial del Entregable 1 y Artefactos para Dashboard (`analysis/export_scores.py`)
- **State:** `APPROVED`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-012, HACK-013`
- **AllowedPaths:**
  - `analysis/export_scores.py`
  - `tests/test_export_scores.py`
  - `reports/leaderboards/pitcher_season_leaderboard.csv`
  - `reports/leaderboards/pitcher_arsenal_breakdown.csv`
  - `reports/leaderboards/movement_map_grid.csv`
  - `data/deliverables/` (ignorado en git; solo el archivo por `PitchUID`)
  - `models/*.joblib` (ignorado en git)
  - *(Orquestador, 2026-10-20: se agregaron las salidas que exige la propia especificación.)*
- **Especificación e Implementación:**
  Genera todos los archivos del **Entregable Obligatorio 1** y los artefactos ligeros que consume el Dashboard:
  1. `data/deliverables/stuff_plus_scores_by_pitch.parquet` (y `.csv.gz`, ignorados en git por tamaño): las 635k filas con `PitchUID`, `pitcher_anon_id`, `year`, `pitch_id`, `stuff_plus_neutral`, `stuff_plus_cdmx`, `delta_stuff_cdmx`, `location_plus_neutral`, `location_plus_cdmx`, `pitching_plus_neutral`, `pitching_plus_cdmx` y las probabilidades esperadas.
  2. Tablas agregadas en `reports/leaderboards/` (commiteables a git, `< 2 MB`):
     - `pitcher_season_leaderboard.csv`: por `(pitcher_anon_id, year)` con conteo de pitcheos, arm slot medio, velocidades, `Stuff+`, `Location+`, `Pitching+` en nivel del mar vs Harp Helú, $\Delta\text{Stuff+}$, tasa de GB y carreras prevenidas.
     - `pitcher_arsenal_breakdown.csv`: por `(pitcher_anon_id, year, pitch_id)` con promedios físicos en neutral vs CDMX, `usage_pct`, `tunnel_sim_release`, `tunnel_sim_170ms` y scores. *(Orquestador, 2026-10-20: debe incluir también `PitcherThrows`, `arm_slot_deg`, `RelHeight`, `fb_velo` (velocidad media de la recta primaria), `n_pitches` y `rho_cdmx_used`, porque HACK-016 y HACK-017 filtran vecinos con ellas sin cargar el modelo.)*
     - `pitcher_season_leaderboard.csv` debe incluir además la descomposición de tres términos de HACK-013 (`canal_stuff`, `canal_parque_bip`, `residuo_no_atribuido`) por pitcher cuando haya n suficiente; si no, columnas vacías explícitas, nunca rellenadas.
     - `movement_map_grid.csv`: grilla agregada `(IVB_bin, HB_norm_bin, pitch_id)` con `stuff_plus_neutral`, `stuff_plus_cdmx` y `delta_stuff_cdmx` para pintar el mapa de movimiento sin exponer puntos individuales.
  3. Serializa los modelos entrenados en `models/` (`movement_translator.joblib`, `stuff_model.joblib`, `location_model.joblib`, `pitching_model.joblib`).
- **Criterios de Aceptación:**
  1. Todos los CSV en `reports/leaderboards/` existen, pesan `< 2 MB` cada uno y no tienen filas individuales por `PitchUID`.
  2. El archivo de entrega por `PitchUID` contiene el 100 % de los `PitchUID` válidos sin duplicados.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_export_scores.py
  ```
- **Revisión (Revisor, 2026-10-20): `CHANGES_REQUESTED`.** Verify real: `pytest -q tests/test_export_scores.py` -> 4 passed, exit 0. Cumple: Criterio 1 (3 CSV, max 946 KB, ningún archivo de `reports/` con `PitchUID`, ninguno > 2 MB), Criterio 2 (634,624 filas = 100 % de `PitchUID` válidos, únicos, 0 NaN, 17 columnas incl. exp_*), `data/` y `models/*.joblib` en `.gitignore`, `pitch_id` (no `AutoPitchType`), cdmx usa mu/sigma neutrales (`src/models/stuff.py:686`), mapa suprime n<5, docstring aclara CDMX = mediana de Extreme Altitude y deriva "carreras prevenidas" con honestidad. Delta -0.063 (sd 1.23) coherente con HACK-013: -0.0063 sd de Stuff+ (~-0.00034 xRV con sd xRV ~0.05) y varía por pitch_id (Curveball -0.23, Changeup -0.16, Sinker/Cutter +0.015): el traductor sí se aplica. Entrenar con year<=2025 es razonable (2026 queda fuera de muestra); falta documentarlo.
  Fallas (spec explícita del Orquestador no implementada):
  1. `analysis/export_scores.py:109-142` (`_leaderboard`): faltan `canal_stuff`, `canal_parque_bip`, `residuo_no_atribuido` (con n suficiente; si no, columnas vacías explícitas). El CSV real no las tiene. Corrección: reutilizar la descomposición de `analysis/run_hypotheses.py:~222` por pitcher o, si no es viable, agregar las 3 columnas con NaN explícito y documentarlo en el docstring.
  2. `analysis/export_scores.py:145-167` (`_arsenal`) y `:61-65` (`AUX_COLS`): faltan `PitcherThrows`, `arm_slot_deg`, `RelHeight`, `fb_velo` (velo media de la recta primaria del pitcher-año), `rho_cdmx_used` (`n_pitches` ya está). HACK-016/017 los necesitan. Corrección: agregar `RelHeight` a AUX_COLS, `arm_slot_deg` y `PitcherThrows` al spec, `fb_velo` y `rho_cdmx_used=rho`; y tests que verifiquen columnas.
  3. Documentar en el docstring (`:3-35`) que los modelos se ajustan con year<=2025 y que 2026 es out-of-sample (`:71`).
- **Segunda revisión (Revisor, 2026-10-20): `APPROVED`.** Verify real: `pytest -q tests/test_export_scores.py` -> 8 passed, exit 0. Las 3 fallas previas corregidas: (1) `export_scores.py:127-165` `_channel_decomposition` + `_leaderboard` añaden `canal_total/canal_stuff/canal_parque_bip/residuo_no_atribuido` (reutiliza `_paired_batted_ball_park` de `run_hypotheses`); en CSV real 921/1721 pitcher-año estimables, 800 con las 4 columnas NaN explícito (nunca filas parciales, sin `fillna`), sin redondeo, max|stuff+parque+residuo-total| = 2.1e-16. (2) `_arsenal` (`:228-237`): `PitcherThrows`, `arm_slot_deg`, `RelHeight`, `fb_velo` (147 NaN = sin recta, no rellenados), `rho_cdmx_used` = 0.8518 presentes en el CSV. (3) docstring `:3-48` documenta entrenamiento year<=2025 y 2026 out-of-sample. Criterio 1: 3 CSV (1.14 MB, 361 KB, 72 KB), ningún CSV de `reports/**` con `PitchUID` ni > 2 MB. Criterio 2 ya verificado en la 1ª revisión (634,624 filas únicas). Coherencia con `reports/altitude/channel_decomposition.csv` (comparabilidad limitada: el agregado es pareado sobre todos los años y 778 pitchers; aquí es por pitcher-año, 921 filas, media simple): total -0.0127 vs -0.0141; stuff -0.00037 vs -0.00034; parque BIP -0.0055 vs -0.0060; residuo -0.0069 vs -0.0078. Mismo signo y orden de magnitud, mismas proporciones (stuff ~3 %, parque ~43 %, residuo ~54 %); no se espera igualdad exacta. Honestidad: residuo etiquetado `residuo_no_atribuido`, nunca como parque. Alcance: los archivos modificados fuera de AllowedPaths (`run_hypotheses.py`, `stuff.py`, tests de 010/013, `dashboard/`, `src/sculptor/`) pertenecen a otras tareas; HACK-014 solo aporta `export_scores.py`, `test_export_scores.py` y `reports/leaderboards/`. Observación no bloqueante: la columna del leaderboard se llama `pitcher_throws` y en el arsenal `PitcherThrows` (consumidores deben usar el nombre correcto).
- **Tercera revisión, re-corrida (Revisor, 2026-10-21): `APPROVED`.** Verify real: `pytest -q tests/test_export_scores.py` -> 8 passed, exit 0. `git status`: la re-corrida solo cambió `reports/leaderboards/*.csv` (`data/` y `models/*.joblib` ignorados). Cifras nuevas: CSV 1.14 MB / 0.36 MB / 0.07 MB, ningún CSV de `reports/**` con `PitchUID` ni >= 2 MB. Parquet: 634,624 filas = 100 % de `PitchUID` válidos, únicos, 0 NaN; `rho_cdmx_used` = 0.8518. Leaderboard: 4 columnas de descomposición, 921 de 1,721 pitcher-año estimables, 800 con NaN explícito (nunca parciales), max|stuff+parque+residuo-total| = 2.1e-16. Arsenal con las 5 columnas extra (5,447 filas). Coherencia con modelo nuevo: `models/stuff_model.joblib` tiene 22 `feature_cols` sin ninguna de `TRAJECTORY_PROXY_COLS` (intersección vacía); `stuff_plus_neutral` media 99.995, desviación 10.05; ΔStuff+ CDMX medio -2.34 (antes -0.06); por pitch_id: Four-Seam -4.52, Sinker -2.12, Changeup -1.60, Splitter -1.26, Curveball -0.46, Cutter -0.39, Slider -0.30, mismos signos y orden que `reports/validation/stuff_diagnostics.csv` (Four-Seam -4.57, Sinker -2.07, Changeup -1.62, Splitter -1.26, Curveball -0.46, Cutter -0.39, Slider -0.29; submuestra de 5,000). `tunnel_sim_170ms` en el arsenal es solo `mean` descriptivo (`export_scores.py:214`) y no está en las features de Stuff+. Código sin cambios.
- **Reviewer Checklist:**
  - [x] ¿Ningún archivo dentro de `reports/` contiene la columna `PitchUID` ni supera los 2 MB?

---

### `HACK-015` — Dashboard Simulador Base Obligatorio (`dashboard/app.py`)
- **State:** `APPROVED`
- **Reviewer Notes (re-revisión 2026-10-20, APPROVED):** Verify exit 0, 16 passed (frío 2.54 s <20 s; caliente 0.66 s sim / 0.92 s sim+SHAP <3 s; headless 2.50 s, re-render 0.36 s). Notas corregidas en `app.py:160-161` (origen real), `app.py:209-210` (física simple vs medianas) y `app.py:328` (CDMX = Altitud Extrema); test `test_ui_notes_on_origin_of_values` presente; AppTest propio sin excepciones; grep de rutas/parquet vacío. Historial de la 1.ª revisión:  Verify real exit 0 (15 passed; carga en frío 2.51 s <20 s; predicción caliente 0.60 s sim / 0.86 s sim+SHAP+figuras <3 s; app headless 2.52 s, re-render caliente 0.35 s). Alcance OK (`dashboard/__init__.py` vacío aceptado como archivo técnico de paquete; `sculptor_tab.py` solo integrado en `app.py:27-30,362,370-373`). Sin rutas absolutas ni parquet (grep propio vacío). `pitcher_throws` (leaderboard: `app.py:298`, `ui_utils.py:555`) y `PitcherThrows` (arsenal: `ui_utils.py:611`) bien usados. AppTest propio sobre CSV reales (3 vistas del mapa, 4 pitchers cargados, mano zurda, SHAP on/off/sabor, filtros del leaderboard) sin excepciones. **Único bloqueo (honestidad, nota de UI incorrecta/incompleta):**
  1. `dashboard/app.py:159-162`: la nota dice que, "mientras HACK-014 no los exporte", altura/lado de liberación se completan con medianas. HACK-014 ya está APPROVED y exporta `RelHeight`/`arm_slot_deg`/`PitcherThrows`/`fb_velo`; hoy `rel_height` viene del arsenal, `rel_side` se deriva del arm slot (`ui_utils.py:622-627`) y solo `spin_axis` sale de medianas (`ui_utils.py:620`). Reescribir con el origen real de cada valor.
  2. La UI no avisa que `ZoneSpeed`, `EffectiveVelo`, `VertApprAngle`, `SpinToBreakRatio`, `arm_slot_deg` (`ui_utils.py:305-321,342`) se estiman con física simple y `VertRelAngle`/`HorzRelAngle_norm`/`Inning`/túnel con medianas; solo está en el docstring de `ui_utils.py:21`. Añadir una nota visible en la Vista 1 (por ejemplo un `st.caption` bajo las entradas).
  Opcional: aclarar bajo el título de la tarjeta de lanzador (`app.py:323`) que "CDMX" = ρ̂ de Altitud Extrema.
- **Sprint:** 3 · **Priority:** `P0` · **DependsOn:** `HACK-014`
- **AllowedPaths:**
  - `dashboard/app.py`
  - `dashboard/ui_utils.py`
  - `tests/test_dashboard_smoke.py`
- **Especificación e Implementación:**
  Cumple al 100 % el **Entregable Obligatorio 4 (Dashboard Simulador)** de las bases.
  Aplicación en Streamlit (`dashboard/app.py`), interfaz 100 % en español, paleta de colores Diablos Rojos (`#CE1126` rojo escarlata, `#0D0D0D` negro carbón, `#F5F5F5` texto claro), que carga únicamente `models/*.joblib` y `reports/**/*.csv` (nunca lee el parquet crudo de 206 MB):
  1. **Vista 1 — Simulador de Pitcheo Lado a Lado (Harp Helú vs. Nivel del Mar):**
     - El pitching coach ingresa manualmente (o carga desde un pitcher real del leaderboard): velocidad (`RelSpeed`), giro (`SpinRate`), eje (`SpinAxis`), quiebre (`InducedVertBreak`, `HorzBreak`), punto de liberación (`RelHeight`, `RelSide`, `Extension`), conteo y zona.
     - Muestra en tarjetas grandes lado a lado:
       - **Nivel del Mar ($\hat{\rho} = 1.00$):** `Stuff+`, `Location+`, `Pitching+`, IVB y HB.
       - **Estadio Alfredo Harp Helú ($\hat{\rho}_{\text{CDMX}}$):** `Stuff+ CDMX`, `Location+ CDMX`, `Pitching+ CDMX`, IVB y HB traducidos, más el **$\Delta\text{Stuff+}$**.
     - **Descomposición de tres términos (alineada con HACK-013):** muestra `Canal Stuff`, `Canal Parque (BIP)` y `Residuo no atribuido` con su `share_of_total` y CI leídos de `reports/altitude/channel_decomposition.csv`. El residuo se muestra con su etiqueta propia ("no se atribuye a ningún canal") y **nunca** se suma ni se renombra como Canal Parque. Es una descomposición poblacional de la liga (no del pitcheo ingresado): la UI lo indica con una nota visible. Si Stuff explica una fracción pequeña del total, la interfaz lo dice tal cual, sin enfatizar el canal.
     - Incluye gráfica interactiva Plotly con las dos ubicaciones/movimientos conectados por un vector de pérdida aerodinámica y barras SHAP de qué variables impulsan el score.
  4. **Requisitos de rendimiento y carga (obligatorios):**
     - Modelos y CSV se cargan una sola vez con `@st.cache_resource` / `@st.cache_data`; ninguna carga ocurre dentro del callback de un widget.
     - La carga en frío (importar + cargar `models/*.joblib` + CSV) debe ser `< 20 s`; con caché caliente, cada predicción y render de la vista 1 debe ser `< 3 s`. Ambas cifras se miden en el test y se reportan en la nota de ejecución.
     - SHAP se calcula bajo demanda, solo para el pitcheo ingresado, no para la tabla completa.
     - El modo headless del test debe ejercitar la lógica pura (`dashboard/ui_utils.py`: funciones de predicción, formateo y filtros) sin requerir un servidor Streamlit.
  2. **Vista 2 — Mapa de Movimiento de la LMB:**
     - Heatmap interactivo en el plano `(HorzBreak_norm, InducedVertBreak)` desde `movement_map_grid.csv`, conmutable entre *Valor a Nivel del Mar*, *Valor en Harp Helú* y *$\Delta\text{Stuff+}$ (quién gana y quién pierde en altitud)*.
  3. **Vista 3 — Leaderboard de Pitchers y Arsenales + Tarjeta de Lanzador:**
     - Filtros por temporada, rol, mano, mínimo de pitcheos y ordenamiento por `Stuff+ CDMX` o `Delta Stuff+`.
- **Criterios de Aceptación:**
  1. `dashboard/app.py` importa y ejecuta su función de renderizado / predicción en modo headless sin errores: carga en frío `< 20 s`, predicción con caché caliente `< 3 s`.
  2. No lee `data/*.parquet`, `data/deliverables/` ni ninguna ruta absoluta fuera del repo (ej. carpetas `Downloads` de otra máquina); solo `models/*.joblib` y `reports/**/*.csv` por rutas relativas a la raíz del proyecto. Un test verifica que el código no contenga rutas absolutas ni referencias a `.parquet`.
  3. **Coherencia numérica (test, no solo humo):** para un pitcheo de ejemplo, `delta_stuff_cdmx == stuff_plus_cdmx − stuff_plus_neutral` (tolerancia 1e-9); IVB/HB en CDMX son menores en magnitud que a nivel del mar para una recta de alto IVB; en `channel_decomposition.csv` cargado, `canal_stuff + canal_parque_bip + residuo_no_atribuido == total` (atol 1e-10); los valores del leaderboard coinciden con `pitcher_season_leaderboard.csv`.
  4. La vista 1 muestra los tres términos de la descomposición (incluido el residuo etiquetado); el test verifica que las tres etiquetas existen en el payload que la UI renderiza.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_dashboard_smoke.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Toda la interfaz está en español?
  - [x] ¿Están presentes los 3 elementos obligatorios de las bases: entrada manual de pitch, comparación lado a lado Harp Helú vs nivel del mar, y mapa de movimiento?
  - [x] ¿La descomposición muestra los tres términos (Stuff, Parque BIP, Residuo no atribuido) con sus etiquetas y sin reasignar el residuo?
  - [x] ¿Modelos y CSV usan caché (`st.cache_resource` / `st.cache_data`) y el test mide carga en frío y predicción en caliente?
  - [x] ¿El código no contiene rutas absolutas ni lee parquet?

---

### `HACK-016` — AI Pitch Sculptor: Buscador de Referencias Factibles (`src/sculptor/optimizer.py`)
- **State:** `APPROVED`
- **Sprint:** 3 · **Priority:** `P1` · **DependsOn:** `HACK-014`
- **AllowedPaths:**
  - `src/sculptor/__init__.py`
  - `src/sculptor/optimizer.py`
  - `dashboard/sculptor_tab.py`
  - `tests/test_sculptor.py`
- **Especificación e Implementación:**
  **Alcance honesto (Orquestador, 2026-10-20):** esta herramienta es una **búsqueda de referencias reales factibles**, no un optimizador que invente pitcheos. Responde: "entre pares biomecánicos reales de la LMB (misma mano, arm slot, altura y velocidad), ¿qué formas de pitcheo mantienen mejor `Stuff+` en CDMX?". Toda cifra sale de `reports/leaderboards/pitcher_arsenal_breakdown.csv` (ya calculada en HACK-014 con el modelo); el sculptor **no reentrena ni reevalúa** perfiles inventados. La UI y los docstrings la llaman "Buscador de Referencias Factibles" y nunca "optimización".
  1. En `src/sculptor/optimizer.py`, implementar `PitchSculptor`:
     - Carga el catálogo desde `pitcher_arsenal_breakdown.csv` (columnas `PitcherThrows`, `arm_slot_deg`, `RelHeight`, `fb_velo`, `pitch_id`, `n_pitches`, `stuff_plus_neutral`, `stuff_plus_cdmx`, `delta_stuff_cdmx`, IVB/HB neutral y CDMX). No importa `StuffPlusModel` ni lee `models/*.joblib`.
     - Método `find_feasible_references(pitch_params: dict, k_neighbors: int = 25, min_pitches: int = 50) -> dict`:
       - Filtro de factibilidad: misma `PitcherThrows`, `arm_slot_deg ± 6.0°`, `RelHeight ± 0.35 ft`, `fb_velo ± 2.0 mph` y `n_pitches >= min_pitches`. Si quedan menos de 5 candidatos, devuelve `status = "INSUFFICIENT_PEERS"` con el motivo y **no** relaja los filtros en silencio.
       - Devuelve:
         1. **Referencia intra-familia:** dentro del mismo `pitch_id`, el rango (p25–p75) y la mejor referencia observada de `stuff_plus_cdmx` entre pares reales. `delta_gain_cdmx` = (`stuff_plus_cdmx` de la referencia) − (`stuff_plus_cdmx` del pitcheo del usuario, tomado del simulador de HACK-015 o ingresado), etiquetado como **"diferencia observada frente a pares reales; no es una ganancia causal garantizada"**. Si el pitcheo del usuario no se puede puntuar, `delta_gain_cdmx = None`.
         2. **Repertorio:** qué `pitch_id` secundario ejecutan los pares con mayor `stuff_plus_cdmx` mediano, con `n_pitchers` y `n_pitches` por opción.
         3. **Comparables LMB:** hasta 3 `pitcher_anon_id` reales (IDs anónimos, ya presentes en el dataset) con esa geometría y ese pitcheo, ordenados por `stuff_plus_cdmx`, cada uno con su `n_pitches`.
         4. **Guía Cualitativa de Coaching:** texto opcional marcado siempre con la etiqueta exacta `[Guía cualitativa de coaching — literatura biomecánica, no estimada del dataset]`, separado visualmente del resto de resultados.
  2. En `dashboard/sculptor_tab.py`, pestaña integrable en `dashboard/app.py` en español. La etiqueta de la guía cualitativa es visible en el mismo bloque donde se muestra el texto (no solo en un tooltip o pie de página), con estilo diferenciado de los resultados cuantitativos. El título de la pestaña es "Referencias Factibles (Pitch Sculptor)".
- **Criterios de Aceptación:**
  1. Toda `pitch_id`, `pitcher_anon_id` y cifra devuelta existe literalmente en `pitcher_arsenal_breakdown.csv` y pasa el filtro de factibilidad (test: no hay valores fuera del catálogo; los rangos p25–p75 se recomputan en el test).
  2. La salida incluye siempre `delta_gain_cdmx` (o `None` con motivo), `status`, `n_peers`, los IDs de los comparables reales de la LMB y la etiqueta exacta de la guía cualitativa.
  3. Con menos de 5 pares, `status == "INSUFFICIENT_PEERS"` y no se emiten recomendaciones numéricas.
  4. Ningún texto de la UI ni docstring usa "optimizar/optimización" para describir el resultado.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_sculptor.py
  ```
- **Reviewer Checklist:**
  - [x] ¿Todo lo recomendado está respaldado por filas reales del catálogo del mismo arm slot, altura y velocidad (sin inventar velocidades o giros sobrehumanos) y sin reevaluar perfiles sintéticos?
  - [x] ¿`delta_gain_cdmx` está descrito como diferencia observada frente a pares y no como ganancia causal?
  - [x] ¿La nota de agarre está explícitamente etiquetada, con la etiqueta exacta, como guía cualitativa de coaching y visible en el mismo bloque de la UI?
  - [x] ¿Con pocos pares devuelve `INSUFFICIENT_PEERS` en vez de relajar filtros en silencio?

- **Revisor (2026-10-20): APPROVED.** `pytest -q tests/test_sculptor.py` = 6 passed, exit 0. Verificado sobre el CSV real (5447 filas): recta de diestro mediano (131 pares, p25-p75 97.88-100.06, mejor pitcher_00622 100.80, delta +5.80 con usuario 95.0), perfil extremo (n_peers 0, `INSUFFICIENT_PEERS`, sin cifras) y zurdo Slider (35 pares, p25-p75 100.78-103.09). Cada fila/ID/cifra existe literalmente en el CSV y pasa los filtros; percentiles, medianas y conteos recomputados coinciden. `n_peers` cuenta pitchers distintos (correcto); la exencion de la palabra del nombre de modulo en el test es aceptable (ningun texto de UI/docstring la usa para describir el resultado).
---

### `HACK-017` — Módulo VeloVision (Video $\to$ Arm Slot $\to$ Rango Stuff+ CDMX) y Validación Manual (`src/vision/pose_slot.py`)
- **State:** `AVAILABLE`
- **Sprint:** 3 · **Priority:** `P1` · **DependsOn:** `HACK-014`
- **AllowedPaths:**
  - `src/vision/__init__.py`
  - `src/vision/pose_slot.py`
  - `dashboard/velovision_tab.py`
  - `analysis/validate_vision.py`
  - `tests/test_velovision.py`
- **Especificación e Implementación:**
  1. En `src/vision/pose_slot.py`, implementar:
     - `extract_release_biomechanics(video_or_frames, pitcher_throws: str = "Right", pitcher_height_ft: float = 6.0) -> dict`:
       - Usa `mediapipe.solutions.pose` (o fallback geométrico sobre keypoints 2D si se pasa un arreglo de keypoints en tests) para localizar en la secuencia el **frame de soltura (release frame)**: el cuadro donde la velocidad angular/extensión de la muñeca lanzadora respecto al hombro alcanza su máximo frente al torso.
       - Calcula a partir de los landmarks `(shoulder, elbow, wrist, hip, ankle)`:
         - `arm_slot_deg`: ángulo del vector hombro$\to$muñeca respecto a la horizontal en el plano frontal, acotado a $[0^\circ, 90^\circ]$.
         - `rel_height_est_ft` y `extension_est_ft` escalados por la altura del jugador en píxeles (`ankle` a `nose`).
         - Imagen anotada (`annotated_frame_rgb`) con el esqueleto y el arco del ángulo dibujados.
     - `estimate_cdmx_stuff_range(arm_slot_deg: float, pitcher_throws: str, pitch_family: str | None = None, observed_velo_mph: float | None = None) -> dict`:
       - Filtra en `pitcher_arsenal_breakdown.csv` los lanzadores de la LMB con la misma mano y `arm_slot_deg ± 6.0°` (y si el scout proporciona `observed_velo_mph` de un radar de bolsillo, filtra además `RelSpeed ± 1.5 mph`).
       - Devuelve la distribución condicionada honesta en CDMX: percentiles `p10`, `p25`, `p50` (mediana), `p75`, `p90` de `stuff_plus_cdmx` y `delta_stuff_cdmx`, junto con el tamaño de muestra `n_comparable_pitches`. **Jamás devuelve un número único fingiendo que midió spin rate desde el video.**
  2. En `analysis/validate_vision.py`:
     - Script de validación que compara el `arm_slot_deg` medido automáticamente contra una tabla de anotación manual de 2 personas (`reports/validation/vision_manual_annotations.csv`), calculando `MAE_deg`, `RMSE_deg` y correlación inter-anotador en `reports/validation/vision_validation_summary.csv`.
  3. En `dashboard/velovision_tab.py`, pestaña para subir un `.mp4` (o elegir un clip/frame de demostración), ver el frame de release con el esqueleto biomecánico y consultar el rango P25–P75 de `Stuff+ CDMX`.
- **Criterios de Aceptación:**
  1. `estimate_cdmx_stuff_range` devuelve siempre un intervalo (`p25`, `p50`, `p75`) y cuando se le pasa `observed_velo_mph`, el ancho intercuartílico `p75 - p25` es menor o igual que cuando no se le pasa velocidad.
  2. Nunca intenta emparejar un video con un `pitcher_anon_id` específico diciendo que es la misma persona.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_velovision.py
  ```
- **Reviewer Checklist:**
  - [ ] **CRÍTICO (Honestidad científica):** ¿La salida de `VeloVision` se reporta siempre como **rango (P25–P75)** condicionado a lo medible (arm slot + velo opcional), sin inventar lecturas de Spin Rate desde el video?
  - [ ] ¿La validación usa anotación manual de frames y no un falso cruce con `pitcher_anon_id`?

---

### `HACK-018` — Integración Final del Dashboard, Perfil Ideal, 3 Casos de Entrevista y Paper Técnico (`docs/`)
- **State:** `AVAILABLE`
- **Sprint:** 4 · **Priority:** `P1` · **DependsOn:** `HACK-014, HACK-015, HACK-016, HACK-017`
- **AllowedPaths:**
  - `dashboard/app.py`
  - `docs/PERFIL_Y_RECOMENDACIONES.md`
  - `docs/DOCUMENTO_TECNICO.md`
  - `tests/test_final_deliverables.py`
- **Especificación e Implementación:**
  1. Conecta las pestañas de `HACK-016` (`sculptor_tab.py`) y `HACK-017` (`velovision_tab.py`) en `dashboard/app.py`.
  2. Redacta `docs/PERFIL_Y_RECOMENDACIONES.md` (Entregables Obligatorios 2 y 3) usando **únicamente cifras reales** leídas de `reports/**/*.csv`:
     - El Perfil Físico y de Arsenal Ideal para el Estadio Alfredo Harp Helú.
     - Perfiles que más ganan y que más pierden `Stuff+` en CDMX (con cuantificación de riesgo de contacto duro y HR).
     - Los **3 Casos Ejecutivos para la Entrevista de 10 minutos** (Reclutamiento de agente libre resiliente, Alerta de "trampa de nivel del mar", y Rediseño con Pitch Sculptor / plan casa vs. gira).
  3. Redacta `docs/DOCUMENTO_TECNICO.md` (Entregable Obligatorio 5, equivalente a 6–10 páginas):
     - Sección 1: Marco teórico (física aerodinámica a 2,240 m, Magnus, arrastre, gravedad y por qué H1–H4 ocurren).
     - Sección 2: Datos, sustitutos de columnas ausentes (`rho_hat`, `pitch_id` invariante, `Outs × Count`, ambos túneles y `prev_pitch_id`).
     - Sección 3: Arquitectura de modelos (`Etapa A` física + `Stuff+`, `Location+`, `Pitching+`) y definición formal de la escala `Score+ = 100 + 10·z` neutral vs entorno.
     - Sección 4: Resultados de validación fuera de muestra (*Pitcher*, *Season 2026*, *Park*) frente al Baseline Nulo y estabilidad interanual.
     - Sección 5: Evidencia empírica LMB de H1–H4 y descomposición *Canal Stuff* vs *Canal Parque*.
     - Sección 6: Aplicabilidad real (Simulador, Pitch Sculptor, VeloVision) y Fronteras/Limitaciones explícitas del modelo.
- **Criterios de Aceptación:**
  1. Un test automatizado (`tests/test_final_deliverables.py`) verifica que todos los entregables existen, que en `docs/PERFIL_Y_RECOMENDACIONES.md` y `docs/DOCUMENTO_TECNICO.md` no queda ningún marcador `[por verificar]` ni `[TODO]`, y que todas las tablas en `reports/` están pobladas.
- **Verify Command:**
  ```powershell
  python -m pytest -q
  ```
- **Reviewer Checklist:**
  - [ ] ¿Todas las cifras citadas en `docs/DOCUMENTO_TECNICO.md` y `docs/PERFIL_Y_RECOMENDACIONES.md` coinciden exactamente con los CSV de `reports/`?
  - [ ] ¿`git status` confirma que no hay ningún archivo `.parquet`, `.pkl` ni CSV pitch-by-pitch rastreado por git?

<!-- Coordinación 2026-10-20: HACK-012 y HACK-013 regresan a REVIEW tras Verify exitoso de Workers. Las observaciones del revisor interno quedan como antecedentes, sin veredicto autorizado. Solo el revisor externo del usuario emite APPROVED o CHANGES_REQUESTED. -->

<!-- Orquestador 2026-10-20: reconciliadas rutas de salida agregadas HACK-012/013 a petición del revisor externo. HACK-012 conserva CHANGES_REQUESTED: mejorar modelos exige reabrir su tarea/AllowedPaths; no se autoriza cambiar criterios ni métricas. HACK-013 reasignada para atender la alternativa explícita de no estimabilidad. -->

<!-- Entrega Worker HACK-013 2026-10-20: Verify 6 passed in 1.79s, exit 0; diff --check limpio. Estado de descomposición NOT_ESTIMABLE y razón explícita en channel_decomposition.csv. Residuo permanece sin atribuir; aceptación aditiva pendiente de decisión del revisor externo. -->

---

### `HACK-019` — Rediseño Visual de Diablos AeroVision (`dashboard/`)
- **State:** `IN_PROGRESS` (Fase 0 aprobada por el usuario el 2026-10-20: mockup v2 en modo claro, `docs/mockups/hack019_mockup.html`. Siguiente: Fase 1)
- **Sprint:** 3 · **Priority:** `P1` · **DependsOn:** `HACK-015, HACK-016`
- **AllowedPaths:**
  - `docs/mockups/hack019_mockup.html` (Fase 0)
  - `dashboard/**` (Fases 1–3)
  - `.streamlit/config.toml`
  - `tests/test_dashboard_smoke.py`
  - `tests/test_dashboard_design.py`
  - `docs/PLAN_REDISENO_UI.md`
- **Especificación:** [`docs/PLAN_REDISENO_UI.md`](PLAN_REDISENO_UI.md) (diagnóstico, dirección visual, estructura y fases). Usabilidad base: [`docs/DISENO_UI_NIELSEN.md`](DISENO_UI_NIELSEN.md), con las 4 correcciones del §1.1 del plan.
- **Fase 0:** mockup HTML estático de Simulador, Leaderboard con tarjeta de lanzador y Sculptor, con cifras tomadas de los CSV reales de `reports/`. Requiere aprobación del usuario antes de la Fase 1.
- **Criterios de Aceptación:** los 6 del §5 de `docs/PLAN_REDISENO_UI.md`.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_dashboard_smoke.py tests/test_dashboard_design.py tests/test_sculptor.py
  ```
- **Reviewer Checklist:**
  - [ ] ¿El rojo `#CE1126` solo aparece como marca y nunca como dato?
  - [ ] ¿ρ̂ CDMX y la descomposición se leen de modelo/CSV, sin cifras hardcodeadas?
  - [ ] ¿Contraste medido ≥ 4.5:1 en texto normal?
  - [ ] ¿Se conservan literales los textos de honestidad de HACK-015/016?


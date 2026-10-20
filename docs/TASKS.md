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
| **HACK-012** | S2 | P0 | `IN_PROGRESS` | `HACK-010, HACK-011` | Suite de validación en los 3 holdouts vs Baseline Nulo + Estabilidad (`analysis/run_validation.py`) |
| **HACK-013** | S2 | P0 | `IN_PROGRESS` | `HACK-010, HACK-011` | Descomposición Canal Stuff vs Canal Parque y pruebas empíricas H1–H4 (`analysis/run_hypotheses.py`) |
| **HACK-014** | S2 | P0 | `AVAILABLE` | `HACK-012, HACK-013` | Exportador oficial Entregable 1: scores por `PitchUID` y agregados (`analysis/export_scores.py`) |
| **HACK-015** | S3 | P0 | `AVAILABLE` | `HACK-014` | Dashboard Simulador Base Obligatorio en Streamlit (`dashboard/app.py`) |
| **HACK-016** | S3 | P1 | `AVAILABLE` | `HACK-014` | Motor **AI Pitch Sculptor** con restricción de factibilidad + pestaña UI (`src/sculptor/optimizer.py`) |
| **HACK-017** | S3 | P1 | `AVAILABLE` | `HACK-014` | Módulo **VeloVision** (MediaPipe Pose $\to$ rango Stuff+ CDMX) + validación manual (`src/vision/pose_slot.py`) |
| **HACK-018** | S4 | P1 | `AVAILABLE` | `HACK-014, HACK-015, HACK-016, HACK-017` | Entregables escritos: Perfil Físico Ideal, 3 Casos de Entrevista y Documento Técnico 6–10 págs (`docs/`) |

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

---

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
- **State:** `IN_PROGRESS`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-010, HACK-011`
- **AllowedPaths:**
  - `analysis/__init__.py`
  - `analysis/run_validation.py`
  - `tests/test_validation_suite.py`
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
  1. `Pitching+` y `Stuff+` superan (menor RMSE y menor Log-Loss) al `CountBaselineModel` en los 3 holdouts.
  2. Los 3 archivos CSV existen en `reports/validation/` con cifras reales calculadas.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_validation_suite.py
  ```
- **Reviewer Checklist:**
  - [ ] ¿En cada fold/holdout tanto `PhysicsMovementTranslator` como los modelos se ajustan **únicamente con `train_idx`** antes de predecir en `test_idx`?

---

### `HACK-013` — Descomposición Canal Stuff vs. Canal Parque y Pruebas H1–H4 (`analysis/run_hypotheses.py`)
- **State:** `IN_PROGRESS`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-010, HACK-011`
- **AllowedPaths:**
  - `analysis/run_hypotheses.py`
  - `tests/test_hypotheses_channels.py`
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
  2. La suma `canal_stuff + canal_parque` reconstruye el efecto total dentro de tolerancia numérica.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_hypotheses_channels.py
  ```
- **Reviewer Checklist:**
  - [ ] ¿Se usó `pitch_id` invariante (y no solo `AutoPitchType`) para no contaminar H1 y H2 con el reetiquetado de TrackMan?
  - [ ] ¿Los errores estándar están agrupados por `pitcher_anon_id`?

---

### `HACK-014` — Exportador Oficial del Entregable 1 y Artefactos para Dashboard (`analysis/export_scores.py`)
- **State:** `AVAILABLE`
- **Sprint:** 2 · **Priority:** `P0` · **DependsOn:** `HACK-012, HACK-013`
- **AllowedPaths:**
  - `analysis/export_scores.py`
  - `tests/test_export_scores.py`
- **Especificación e Implementación:**
  Genera todos los archivos del **Entregable Obligatorio 1** y los artefactos ligeros que consume el Dashboard:
  1. `data/deliverables/stuff_plus_scores_by_pitch.parquet` (y `.csv.gz`, ignorados en git por tamaño): las 635k filas con `PitchUID`, `pitcher_anon_id`, `year`, `pitch_id`, `stuff_plus_neutral`, `stuff_plus_cdmx`, `delta_stuff_cdmx`, `location_plus_neutral`, `location_plus_cdmx`, `pitching_plus_neutral`, `pitching_plus_cdmx` y las probabilidades esperadas.
  2. Tablas agregadas en `reports/leaderboards/` (commiteables a git, `< 2 MB`):
     - `pitcher_season_leaderboard.csv`: por `(pitcher_anon_id, year)` con conteo de pitcheos, arm slot medio, velocidades, `Stuff+`, `Location+`, `Pitching+` en nivel del mar vs Harp Helú, $\Delta\text{Stuff+}$, tasa de GB y carreras prevenidas.
     - `pitcher_arsenal_breakdown.csv`: por `(pitcher_anon_id, year, pitch_id)` con promedios físicos en neutral vs CDMX, `usage_pct`, `tunnel_sim_release`, `tunnel_sim_170ms` y scores.
     - `movement_map_grid.csv`: grilla agregada `(IVB_bin, HB_norm_bin, pitch_id)` con `stuff_plus_neutral`, `stuff_plus_cdmx` y `delta_stuff_cdmx` para pintar el mapa de movimiento sin exponer puntos individuales.
  3. Serializa los modelos entrenados en `models/` (`movement_translator.joblib`, `stuff_model.joblib`, `location_model.joblib`, `pitching_model.joblib`).
- **Criterios de Aceptación:**
  1. Todos los CSV en `reports/leaderboards/` existen, pesan `< 2 MB` cada uno y no tienen filas individuales por `PitchUID`.
  2. El archivo de entrega por `PitchUID` contiene el 100 % de los `PitchUID` válidos sin duplicados.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_export_scores.py
  ```
- **Reviewer Checklist:**
  - [ ] ¿Ningún archivo dentro de `reports/` contiene la columna `PitchUID` ni supera los 2 MB?

---

### `HACK-015` — Dashboard Simulador Base Obligatorio (`dashboard/app.py`)
- **State:** `AVAILABLE`
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
       - **Estadio Alfredo Harp Helú ($\hat{\rho}_{\text{CDMX}}$):** `Stuff+ CDMX`, `Location+ CDMX`, `Pitching+ CDMX`, IVB y HB traducidos, más el **$\Delta\text{Stuff+}$** y la descomposición en *Canal Stuff* vs *Canal Parque*.
     - Incluye gráfica interactiva Plotly con las dos ubicaciones/movimientos conectados por un vector de pérdida aerodinámica y barras SHAP de qué variables impulsan el score.
  2. **Vista 2 — Mapa de Movimiento de la LMB:**
     - Heatmap interactivo en el plano `(HorzBreak_norm, InducedVertBreak)` desde `movement_map_grid.csv`, conmutable entre *Valor a Nivel del Mar*, *Valor en Harp Helú* y *$\Delta\text{Stuff+}$ (quién gana y quién pierde en altitud)*.
  3. **Vista 3 — Leaderboard de Pitchers y Arsenales + Tarjeta de Lanzador:**
     - Filtros por temporada, rol, mano, mínimo de pitcheos y ordenamiento por `Stuff+ CDMX` o `Delta Stuff+`.
- **Criterios de Aceptación:**
  1. `dashboard/app.py` importa y ejecuta su función de renderizado / predicción en modo headless sin errores y en `< 3 s`.
  2. No lee nada de `C:\Users\L13Yoga\Downloads` ni de `data/*.parquet` para poder desplegarse sin datos privados.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_dashboard_smoke.py
  ```
- **Reviewer Checklist:**
  - [ ] ¿Toda la interfaz está en español?
  - [ ] ¿Están presentes los 3 elementos obligatorios de las bases: entrada manual de pitch, comparación lado a lado Harp Helú vs nivel del mar, y mapa de movimiento?

---

### `HACK-016` — Motor AI Pitch Sculptor con Restricción de Factibilidad (`src/sculptor/optimizer.py`)
- **State:** `AVAILABLE`
- **Sprint:** 3 · **Priority:** `P1` · **DependsOn:** `HACK-014`
- **AllowedPaths:**
  - `src/sculptor/__init__.py`
  - `src/sculptor/optimizer.py`
  - `dashboard/sculptor_tab.py`
  - `tests/test_sculptor.py`
- **Especificación e Implementación:**
  Nuestro diferenciador prescriptivo sobre los simuladores pasivos:
  1. En `src/sculptor/optimizer.py`, implementar `PitchSculptor`:
     - Construye el catálogo empírico de arsenales reales a partir de `pitcher_arsenal_breakdown.csv` (en nivel del mar).
     - Método `sculpt_pitch(pitch_params: dict, k_neighbors: int = 25) -> dict`:
       - Restricción de factibilidad biomecánica: filtra únicamente los clusters de pitcheos reales lanzados por pitchers de la LMB de la misma mano (`PitcherThrows`) cuyo ángulo de brazo esté dentro de `arm_slot_deg ± 6.0°`, altura `RelHeight ± 0.35 ft` y velocidad de recta primaria dentro de `fb_velo ± 2.0 mph`.
       - Evalúa esos perfiles candidatos reales bajo el aire del Harp Helú (`stuff_plus_cdmx`) usando `StuffPlusModel` y `PhysicsMovementTranslator`.
       - Devuelve:
         1. **Optimización Intra-Pitcheo:** Qué ajuste factible de eje/eficiencia dentro de la misma familia maximiza `stuff_plus_cdmx` y cuántos puntos recupera (`delta_gain_cdmx`).
         2. **Recomendación de Repertorio (Adición/Sustitución):** Qué tipo de pitcheo secundario (ej. `Changeup` o `Splitter`) lanzado por sus pares biomecánicos reales logra el mayor `stuff_plus_cdmx`.
         3. **Pitchers Comparables LMB:** Lista de los 3 `pitcher_anon_id` reales de la liga con esa misma geometría de brazo que ya ejecutan ese pitcheo con éxito en altitud.
         4. **Guía Cualitativa de Coaching:** Nota claramente etiquetada como `[Guía cualitativa de coucheo — literatura biomecánica, no estimada del dataset]` sobre pronación/supinación o presión de dedos asociada a ese cambio de forma.
  2. En `dashboard/sculptor_tab.py`, componente visual integrable como pestaña en `dashboard/app.py`.
- **Criterios de Aceptación:**
  1. Ninguna forma recomendada por `sculpt_pitch` cae fuera del envolvente convexo / rango `[min, max]` de los vecinos reales con el mismo slot y velocidad.
  2. La salida incluye siempre `delta_gain_cdmx`, los IDs de los comparables reales de la LMB y la etiqueta explícita en la guía cualitativa.
- **Verify Command:**
  ```powershell
  python -m pytest -q tests/test_sculptor.py
  ```
- **Reviewer Checklist:**
  - [ ] ¿El optimizador propone **únicamente** formas respaldadas por vecinos reales del mismo arm slot y velocidad (sin inventar velocidades o giros sobrehumanos)?
  - [ ] ¿La nota de agarre está explícitamente etiquetada como guía cualitativa de coaching y no como predicción del modelo?

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

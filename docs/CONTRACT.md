# Contrato de Datos, Columnas y Firmas (CONTRACT.md)

> **Propósito:** Ningún agente ejecutor (*Worker*) puede inventar nombres de columnas, rutas o firmas de funciones fuera de este contrato. El agente revisor (*Reviewer*) rechazará cualquier tarea que viole estas firmas o nombres de columnas.

---

## 1. Regla de Separación de Columnas por Rol (Prohibición de Leakage)

El archivo `stuff_plus_data_dictionary.csv` define `feature_role`. Todo código debe respetar estos conjuntos disjuntos:

### 1.1 Columnas Prohibidas como Features en CUALQUIER Modelo (`FORBIDDEN_TARGET_COLS`)
Jamás pueden entrar en la matriz `X` de ningún modelo (`Stuff+`, `Location+`, `Pitching+`, `Movement`):
```python
FORBIDDEN_TARGET_COLS = [
    "PitchCall", "KorBB", "play_result", "hit_type",
    "ExitSpeed", "Angle", "Direction", "Distance",
    "is_swing", "is_whiff", "is_contact", "is_called_strike",
    "is_swinging_strike", "is_ball_in_play", "is_batted", "is_hit",
    "single", "double", "triple", "home_run",
    "base_on_balls", "strikeout", "is_hit_by_pitch",
    "RunsScored", "OutsOnPlay", "swung_outside_strike_zone",
    # Y todos los targets derivados en HACK-006:
    "target_whiff", "target_chase", "target_called_strike",
    "target_groundball", "target_weak_contact", "target_hard_contact",
    "target_out_on_pitch", "xrv_pitcher",
]
```

### 1.2 Columnas Prohibidas como Features Directas (`ID_AND_GROUP_COLS`)
Solo se usan para `groupby`, `merge`, `GroupKFold` o exportar resultados:
```python
ID_AND_GROUP_COLS = [
    "PitchUID", "game_anon_id", "pitcher_anon_id",
    "batter_anon_id", "catcher_anon_id", "year",
]
```

### 1.3 Columnas Prohibidas en `Stuff+` Puro (`LOCATION_AND_COUNT_COLS`)
Exclusivas para `Location+` y `Pitching+`:
```python
LOCATION_AND_COUNT_COLS = [
    "PlateLocHeight", "PlateLocSide", "PlateLocSide_norm",
    "in_strike_zone", "outside_strike_zone",
    "dist_from_zone_center", "dist_to_zone_edge", "is_shadow_zone", "zone_quadrant",
    "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "count",
    "BatterSide", "is_same_hand", "prev_pitch_id",
]
```

---

## 2. Contrato de Columnas Derivadas (Nombres Exactos)

Cada tarea agrega columnas con **estos nombres exactos** al `DataFrame`:

| Tarea | Columna | Tipo | Descripción y Unidad |
| :--- | :--- | :--- | :--- |
| **HACK-002** | `HorzBreak_norm` | `float64` | `HorzBreak * hand_sign` (`+1` si `PitcherThrows == 'Right'`, `-1` si `'Left'`). Positivo = *arm-side*, negativo = *glove-side* (pulgadas). |
| **HACK-002** | `RelSide_norm` | `float64` | `RelSide * hand_sign` (pies). |
| **HACK-002** | `HorzRelAngle_norm` | `float64` | `HorzRelAngle * hand_sign` (grados). |
| **HACK-002** | `HorzApprAngle_norm` | `float64` | `HorzApprAngle * hand_sign` (grados). |
| **HACK-002** | `PlateLocSide_norm` | `float64` | `PlateLocSide * hand_sign` (pies). |
| **HACK-002** | `TotalInducedBreak` | `float64` | $\sqrt{\text{InducedVertBreak}^2 + \text{HorzBreak}^2}$ (pulgadas). |
| **HACK-002** | `BreakAngle_deg` | `float64` | $\text{atan2}(\text{HorzBreak\_norm}, \text{InducedVertBreak})$ en grados $[-180, 180]$. |
| **HACK-002** | `SpinAxis_sin`, `SpinAxis_cos` | `float64` | $\sin(\text{SpinAxis})\cdot\text{hand\_sign}$ y $\cos(\text{SpinAxis})$. |
| **HACK-002** | `SpinToBreakRatio` | `float64` | $\text{TotalInducedBreak} / ((\text{SpinRate}/1000) \cdot \max(\text{ZoneTime}, 0.35) + 10^{-5})$. Proxy de eficiencia de giro (H4). |
| **HACK-002** | `DragProxy` | `float64` | $\text{SpeedDrop} / \max(\text{RelSpeed}, 60.0)$. |
| **HACK-002** | `arm_slot_deg` | `float64` | Ángulo de liberación estimado desde el montículo: $\text{ deg}(\text{atan2}(\text{RelHeight} - 4.0, |\text{RelSide}| + 0.5))$ acotado a $[0, 90]$. |
| **HACK-003** | `rho_hat` | `float64` | Índice continuo de densidad del aire por `game_anon_id`, normalizado tal que la mediana de juegos en `No Altitude` sea `1.000`. |
| **HACK-004** | `pitch_cluster_id` | `int32` | Cluster invariante a la altitud dentro de `(pitcher_anon_id, year)` ($0 \dots K-1$). |
| **HACK-004** | `pitch_id` | `string` | Etiqueta invariante del cluster (`Four-Seam`, `Sinker`, `Cutter`, `Slider`, `Sweeper`, `Curveball`, `Changeup`, `Splitter`, `Other`) asignada según el modo en `No Altitude` / modelo de referencia a nivel del mar. |
| **HACK-005** | `primary_fb_id` | `string` | El `pitch_id` de recta (`Four-Seam`, `Sinker`, `Cutter`) más frecuente del pitcher en esa temporada. |
| **HACK-005** | `velo_diff_vs_fb` | `float64` | `RelSpeed - fb_velo` (mph). |
| **HACK-005** | `ivb_diff_vs_fb` | `float64` | `InducedVertBreak - fb_ivb` (pulgadas). |
| **HACK-005** | `hb_diff_vs_fb` | `float64` | `HorzBreak_norm - fb_hb_norm` (pulgadas). |
| **HACK-005** | `tunnel_dist_release_in` | `float64` | Distancia 3D en el punto de release contra la recta primaria en pulgadas (definición de las bases). |
| **HACK-005** | `tunnel_sim_release` | `float64` | $\exp(-\text{tunnel\_dist\_release\_in} / 6.0) \in (0, 1]$. |
| **HACK-005** | `tunnel_dist_170ms_in` | `float64` | Distancia 3D en $t = 0.17\text{ s}$ usando `PitchTrajectoryXc0..Zc2` contra la recta primaria en pulgadas (extensión nuestra). |
| **HACK-005** | `tunnel_sim_170ms` | `float64` | $\exp(-\text{tunnel\_dist\_170ms\_in} / 6.0) \in (0, 1]$. |
| **HACK-005** | `pitch_usage_pct` | `float64` | Proporción de uso de `pitch_id` en el repertorio anual del pitcher $\in (0, 1]$. |
| **HACK-005** | `prev_pitch_id` | `string` | `pitch_id` del lanzamiento inmediato anterior dentro de la misma aparición al plato (`"NONE"` si es el primer pitcheo de la PA). |
| **HACK-006** | `target_whiff` | `float64` | `1.0` si `is_whiff == 1`, `0.0` en otro caso. |
| **HACK-006** | `target_chase` | `float64` | `1.0` si `swung_outside_strike_zone == 1`, `0.0` en otro caso. |
| **HACK-006** | `target_called_strike` | `float64` | `1.0` si `is_called_strike == 1`, `0.0` en otro caso. |
| **HACK-006** | `target_groundball` | `float64` | `1.0` si `hit_type == 'GroundBall'`, `0.0` en otro caso. |
| **HACK-006** | `target_weak_contact` | `float64` | `1.0` si `is_ball_in_play == 1` y (`ExitSpeed < 85.0` o `hit_type == 'Popup'`), `0.0` en otro caso. |
| **HACK-006** | `target_hard_contact` | `float64` | `1.0` si `is_ball_in_play == 1` y (`ExitSpeed >= 95.0` o `home_run == 1`), `0.0` en otro caso. |
| **HACK-006** | `target_out_on_pitch` | `float64` | `1.0` si `OutsOnPlay > 0` o `strikeout == 1`, `0.0` en otro caso. |
| **HACK-006** | `xrv_pitcher` | `float64` | Valor de carrera empírico LMB desde la perspectiva del pitcher (`+` = previene carreras / bueno para el pitcher). |

---

## 3. Contrato de Salida de Scores (`scores_by_pitch.parquet` y agregados)

Todo modelo de la familia `Score+` produce valores con media `100.0` y desviación estándar `10.0` sobre la distribución LMB evaluada a densidad neutral ($\hat{\rho}_0 = 1.0$), donde **mayor que 100 es mejor para el pitcher**:
- `stuff_plus_neutral`: evaluado con $\hat{\rho} = 1.000$ y movimiento esperado a nivel del mar.
- `stuff_plus_cdmx`: evaluado con $\hat{\rho}_{\text{CDMX}}$ (mediana de `Extreme Altitude`) y movimiento traducido por la Etapa A (*canal Stuff*).
- `delta_stuff_cdmx`: `stuff_plus_cdmx - stuff_plus_neutral`.
- `location_plus_neutral`, `location_plus_cdmx`.
- `pitching_plus_neutral`, `pitching_plus_cdmx`.

---

## 4. Fixture Sintético para Pruebas Unitarias (`tests/conftest.py`)

Para que cualquier agente ejecutor o revisor corra `pytest` en menos de 5 segundos sin requerir los 206 MB de datos reales:
- El fixture `synthetic_lmb_df` (definido en `HACK-001`) genera un `DataFrame` determinista (`np.random.default_rng(42)`) de `1,200` filas y las **84 columnas exactas** del diccionario con tipos idénticos al parquet real, conteniendo 12 pitchers, 18 juegos en las 3 altitudes (`No Altitude`, `Medium Altitude`, `Extreme Altitude`) y las 3 temporadas (`2024`, `2025`, `2026`).

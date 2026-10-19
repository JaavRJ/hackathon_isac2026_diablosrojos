# AGENTS.md — Reglas para agentes de IA

> **Proyecto:** Hackathon ISAC 2026 · Reto 2 Diablos Rojos — *Stuff Plus Model: calidad de pitcheo ajustada por entorno*.
> **Producto:** **Diablos AeroVision** (Stuff+ LMB · Pitch Sculptor · VeloVision).
> **Entrega:** 22-oct-2026, 23:59 CDMX (`itamsportsanalytics@gmail.com`). Sprint.
> **Lee después:** [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) (idea, física, métodos) · **[docs/TASKS.md](docs/TASKS.md)** (backlog especificado por tarea) · **[docs/CONTRACT.md](docs/CONTRACT.md)** (contrato de columnas y firmas) · [docs/PLAN.md](docs/PLAN.md) (protocolo Orquestador → Worker → Revisor). Benchmark del rival: [docs/COMPETENCIA.md](docs/COMPETENCIA.md) (solo si te lo piden).

---

## 0. Protocolo Autónomo por Rol (Léelo al iniciar sesión)

- **Si eres el AGENTE REVISOR (`Reviewer`):**
  1. Abre [`docs/TASKS.md`](docs/TASKS.md) y busca automáticamente **todas las tareas cuyo estado sea `REVIEW`**.
  2. Para cada tarea en `REVIEW`:
     - Verifica que solo se hayan tocado archivos dentro de su lista `AllowedPaths`.
     - Lee el código implementado y contrástalo contra [`docs/CONTRACT.md`](docs/CONTRACT.md) y la especificación de la tarea.
     - Ejecuta tú mismo en terminal el `Verify Command` de esa tarea y comprueba que pase con `exit code 0`.
     - Evalúa rigurosamente cada punto de su `Reviewer Checklist`.
  3. **Veredicto en [`docs/TASKS.md`](docs/TASKS.md):**
     - Si cumple el 100 % sin excepciones: cambia el estado de la tarea de `REVIEW` a `APPROVED` (tanto en la tabla global como en el encabezado de la tarea) y marca `[x]` las casillas de su `Reviewer Checklist`.
     - Si falla cualquier punto: cambia el estado a `CHANGES_REQUESTED` y agrega debajo de la tarea una subsección `#### Notas de Revisión` con `archivo:línea` y la corrección exacta requerida.
- **Si eres un AGENTE EJECUTOR (`Worker`):**
  Ejecuta únicamente el ID de tarea asignado, respeta `AllowedPaths`, haz pasar el `Verify Command` (`exit code 0`) y pon la tarea en `REVIEW` en [`docs/TASKS.md`](docs/TASKS.md).
- **Si eres el ORQUESTADOR (`Orchestrator`):**
  Despacha a Workers económicos las tareas en `AVAILABLE` cuyas dependencias (`DependsOn`) estén en `APPROVED`, o re-despacha las que estén en `CHANGES_REQUESTED`.

---

## 1. Decisiones cerradas (no reabrir)
1. Reto 2 (Diablos Rojos). El Reto 1 (Club América) está descartado.
2. El **núcleo obligatorio** (modelos + features de arsenal + delta por altitud + holdouts + dashboard simulador base) va antes que los extras (Sculptor, VeloVision). Si hay conflicto de tiempo, gana lo obligatorio.
3. Tres submodelos separados: **Stuff+**, **Location+**, **Pitching+**. Cada uno en dos sabores: **neutral** (densidad de nivel del mar) y **de entorno** (densidad del estadio, p. ej. CDMX). `ΔStuff+ CDMX = entorno − neutral`.
4. Escala: `Score+ = 100 + 10·z`, referencia = pitches LMB de la temporada evaluados en sabor neutral. Signo: más alto = mejor para el pitcher.
5. Para las bases, Stuff+ es un modelo de **predicción** y el Causal Forest es una capa táctica opcional. Nuestro análisis intra-pitcher alimenta la evidencia empírica de `ΔStuff+` por pitch shape, no el nombre del producto.

## 2. Reglas de honestidad científica
1. **Nunca inventes cifras.** Todo número en docs, UI o paper sale de un análisis que corrimos, o se marca `[por verificar]` / `[ejemplo ilustrativo]`.
2. **No copies cifras del equipo rival**: re-mídelas con nuestros datos.
3. **Física correcta:** Magnus, arrastre **y Seam-Shifted Wake (SSW)** son fuerzas aerodinámicas ∝ ρ·v²; **todas** se debilitan en altitud. Lo único que no cambia es la gravedad. Nunca afirmes que el SSW "no depende del aire".
4. El dataset **no tiene orientación de costura**. El SSW no se puede ajustar ni medir con estos datos: cualquier recomendación de agarre es **guía cualitativa de coaching**, etiquetada como tal, no salida del modelo.
5. Un video monocular **no mide** spin rate, eje, IVB/HB ni velocidad precisa. VeloVision entrega un **rango** de Stuff+ condicionado a lo que sí se mide (arm slot, release), nunca un valor puntual "sin TrackMan".
6. **No se puede emparejar un video con un pitcher del dataset** porque `pitcher_anon_id` es anónimo. VeloVision se valida contra **anotación manual de 2 personas** en los mismos frames (y opcionalmente clips públicos con Statcast si los organizadores lo confirman por correo).
7. La estrategia de identificación del efecto de altitud debe ser explícita (variación intra-pitcher, PROJECT_CONTEXT §5). No escribas "Double ML elimina el sesgo" sin decir qué confusores se controlan y cómo.

## 3. Reglas de datos
1. Dataset: `C:\Users\L13Yoga\Downloads\Diablos - Hackathon 2026\stuff_model_df.parquet` (635,002 × 84). Diccionario: `stuff_plus_data_dictionary.csv` en la misma carpeta. Requiere `pyarrow >= 20`.
2. Datos crudos y predicciones pitch-by-pitch **nunca** van a git ni al deploy. Solo agregados y figuras.
3. **Leakage:** columnas con `feature_role = target_only` jamás son features. `location_plus_only` (`PlateLocHeight`, `PlateLocSide`, `in_strike_zone`, `outside_strike_zone`) jamás entran en Stuff+.
4. IDs (`pitcher_anon_id`, `batter_anon_id`, `game_anon_id`, `catcher_anon_id`) solo para agrupar, partir y agregar; nunca como features.
5. **No uses `AutoPitchType` para comparar entre altitudes**: TrackMan reetiqueta pitcheos según el aire. Usa la identidad de pitcheo invariante (`pitch_id`, PROJECT_CONTEXT §4.3).
6. **Features pedidas que no existen en el dataset** y sus sustitutos obligatorios:
   - `stadium` → `altitude_category` + ρ̂ por juego (`Extreme Altitude` mezcla varios parques; no digas "CDMX" sin aclararlo).
   - `altitude_m` → ρ̂ continua por juego.
   - `runners_on_base` → estado reducido **outs × conteo** (documentar como limitación).
   - `prev_pitch_type` → construible desde el orden de juego/PA usando el `pitch_id` anterior.
   - `tunnel_similarity` → incluir **ambas** versiones: distancia de release point (definición de las bases) y distancia a 170 ms con polinomios (nuestra extensión).
7. Ningún modelo principal se entrena con datos externos (todo calibrado 100 % con LMB).

## 4. Validación obligatoria (todo modelo la reporta)
- **Baseline nulo:** xRV medio por conteo. Todo modelo se compara contra él.
- **Pitcher holdout:** `GroupKFold` por `pitcher_anon_id`.
- **Season holdout:** train 2024–25 → test 2026.
- **Park holdout:** train No + Medium → test Extreme.
- **Estabilidad:** correlación año-a-año del Stuff+ por pitcher.

## 5. Estructura del repo
```
src/            # librería (features, identidad, densidad, modelos)
src/vision/     # VeloVision
src/sculptor/   # Pitch Sculptor
analysis/       # corridas pesadas → reports/
reports/        # tablas y figuras agregadas (sí van a git, ≤ 2 MB)
models/         # modelos serializados (no van a git)
dashboard/      # app (UI en español, rojo Diablos + negro)
docs/           # paper, benchmark, decisiones
```

## 6. Prioridades del sprint
| Prio | Entregable |
|---|---|
| P0 | Features + `pitch_id` invariante + `prev_pitch_type` + ambos túneles + ρ̂ por juego + targets/xRV calibrados a LMB |
| P0 | Stuff+/Location+/Pitching+ (neutral y entorno) + validaciones + baseline nulo |
| P0 | Delta Stuff+ por pitch shape (intra-pitcher) + descomposición canal Stuff vs canal parque + H1–H4 |
| P0 | Archivo de scores por `PitchUID` y agregados por pitcher/arsenal |
| P0 | **Dashboard simulador obligatorio:** entrada manual de pitch, comparación Harp Helú vs nivel del mar, mapa de movimiento y leaderboard |
| P1 | Pitch Sculptor (optimización con restricción de factibilidad) + casos para la entrevista de 10 min |
| P1 | VeloVision mínimo (video → arm slot → rango de Stuff+ CDMX) + validación por anotación manual |
| P1 | Documento técnico (6–10 págs) + recomendaciones operativas + perfil físico ideal |
| P2 | Causal Forest táctico por conteo/secuencia, identificación de equipos por alineaciones |

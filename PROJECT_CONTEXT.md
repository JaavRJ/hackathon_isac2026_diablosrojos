# PROJECT_CONTEXT — Diablos AeroVision

> Hackathon ISAC 2026 · Reto 2 Diablos Rojos · entrega 22-oct-2026 por correo a `itamsportsanalytics@gmail.com` (incluir correos de todos los integrantes).
> Convención: cifras marcadas `[por verificar]` aún no las medimos nosotros; `[ejemplo]` es ilustrativo.

---

## 1. La idea en una línea
Para el **pitching coach y el director deportivo de Diablos Rojos**, **AeroVision** mide cuánto vale cada pitcheo en el aire del Harp Helú, **cuantifica el delta Stuff+ por pitch shape entre CDMX y nivel del mar separando qué parte es del pitcheo y qué parte es del parque**, y **recomienda qué forma de pitcheo factible recupera valor**. Funciona desde datos TrackMan o, con menor precisión, desde un video de bullpen.

## 2. Qué pide el reto vs qué agregamos
**Obligatorio según las bases** (va primero):
- Stuff+, Location+ y Pitching+ con targets multi-objetivo y escala `100 + 10·z`.
- Features de arsenal.
- Delta Stuff+ por pitch shape entre Harp Helú y nivel del mar.
- Holdout de pitcher, temporada y parque contra el baseline nulo.
- Dashboard simulador con entrada manual y comparación lado a lado.
- Perfil ideal, recomendaciones operativas y paper.

**Nuestros diferenciadores** (sobre lo obligatorio):
1. **Delta de altitud bien identificado y descompuesto:** comparación intra-pitcher con densidad continua ρ̂, separando el *canal Stuff* (el pitcheo se mueve menos) del *canal parque* (la bola bateada vuela más). Solo el primero debe castigar el Stuff+.
2. **Prescripción, no solo diagnóstico:** Pitch Sculptor busca la forma de pitcheo que maximiza el Stuff+ CDMX **dentro de lo que pitchers con el mismo slot y velocidad realmente lanzan**.
3. **Entrada desde video** (VeloVision) para prospectos sin TrackMan, con el error medido y reportado.

## 3. Física (versión correcta)
- **Densidad.** Atmósfera estándar a 2,240 m: ρ/ρ₀ ≈ 0.78 (teórico). Las bases dicen "~20 % menor". La densidad efectiva por juego la **estimamos del arrastre** (ρ̂, §4.3). `[por verificar: valor medido en Extreme]`.
- **Fuerzas aerodinámicas** (arrastre, Magnus y Seam-Shifted Wake) escalan ∝ ρ·v². **Todas** se debilitan en altitud. La **gravedad** no cambia.
- **Hipótesis del reto** (todas se cuantifican con datos LMB antes de usarse como supuesto):
  - **H1:** Los pitcheos que viven de movimiento aerodinámico (rectas de mucho IVB, sweepers, curvas de mucho quiebre) pierden más. Referencia: Nathan estima 18" → 14–15" en Coors.
  - **H2:** Los pitcheos que viven de gravedad y de diferencial de velocidad (splitter, changeup, sinker de bajo IVB, gyro-slider) son más resilientes.
  - **H3:** Por el mayor *carry*, en altitud valen más los roletazos.
  - **H4:** La eficiencia de giro y el movimiento observado importan más que el spin bruto ("shape over label").
- **Arrastre menor** ⇒ la recta llega con más velocidad al plato. Es un efecto compensatorio que hay que medir.

## 4. Datos y features

### 4.1 Dataset
`stuff_model_df.parquet`: 635,002 pitches × 84 columnas · 1,134 pitchers · 864 bateadores · 2,127 juegos.
| Temporada | Filas | Uso |
|---|---|---|
| 2024 | 243,254 | train |
| 2025 | 256,144 | train |
| 2026 | 135,604 | season holdout |

`altitude_category`: No 293,875 (46.3 %) · Extreme 183,267 (28.9 %) · Medium 154,840 (24.4 %) · nulos 3,020 (imputar por `game_anon_id`).

Roles del diccionario: `stuff_feature` 41 · `target_only` 26 · `context_only` 8 · `location_plus_only` 4 · `grouping_only` 4 · `id_only` 1.

**Uso de datos (bases):** los datos de ISAC solo se usan para la competencia. Se presentan resultados, nunca los datos.

### 4.2 Features que pide el reto y no existen en el dataset
| Feature pedida | Estado | Sustituto |
|---|---|---|
| `stadium` | no existe | `altitude_category` + ρ̂ por juego. Equipos o estadios por comunidades de alineaciones (P2) |
| `altitude_m` | no existe | ρ̂ continua (§4.3); altitud nominal solo para el Harp Helú (2,240 m) |
| `runners_on_base` (24 estados base-out) | no existe | estado reducido **outs × conteo**; se documenta como limitación |
| `prev_pitch_type` | construible | las filas vienen en orden de juego/PA `[por verificar orden]`. Se usa el `pitch_id` del pitch anterior en la misma PA |
| `zone` (9 + fuera) | construible | desde `PlateLocSide`/`PlateLocHeight` |
| `inning`, `count`, mano del pitcher y del bateador | existen | directo |

### 4.3 Identidad de pitcheo invariante a la altitud
TrackMan clasifica por movimiento, y el movimiento cambia con el aire, así que `AutoPitchType` se **reetiqueta** en altitud (`[por verificar]`: sinkers→cutters, changeups→sliders). Definimos `pitch_id` por pitcher-temporada a partir de rasgos que el aire no altera: velocidad, spin rate, release y, como desempate, el eje de giro. Se hace con clustering (GMM/KMeans) dentro de cada pitcher-temporada, nombrando cada cluster con la etiqueta mayoritaria **a nivel del mar**. Todo lo que compara altitudes usa `pitch_id`. También sirve como el "pitch cluster" que pide Location+.

### 4.4 Densidad continua ρ̂ por juego
A partir de la desaceleración en el eje y (`ay0`, `vy0`, `SpeedDrop`, `ZoneTime`) de rectas con velocidad y spin similares, estimamos un coeficiente de arrastre relativo por juego. Se suaviza por categoría y se normaliza a nivel del mar = 1.

### 4.5 Features de arsenal (≈25 % de la rúbrica)
- **Recta primaria** por `(pitcher, año)`: el `pitch_id` de tipo recta más usado, **definido con pitches a nivel del mar**.
- **Diferenciales vs la recta primaria:** `velo_diff_vs_fb`, `ivb_diff_vs_fb`, `hb_diff_vs_fb` (HB normalizado por mano).
- **`tunnel_similarity`, en dos versiones:**
  - **(a) Definición de las bases:** similitud del **release point** con la recta (distancia 3D de `RelHeight`, `RelSide`, `Extension`).
  - **(b) Extensión nuestra:** distancia en el punto de decisión (~170 ms, con los polinomios `PitchTrajectory*`); `exp(−dist/s)`, donde `s` es un hiperparámetro calibrado.
  - Se reportan ambas y su aporte por SHAP.
- **`pitch_usage_pct`** y **`prev_pitch_type`** (cluster del pitch anterior).

### 4.6 Features físicas
- `TotalBreak = √(IVB² + HB²)`.
- Eje de giro como seno y coseno, normalizado por mano.
- `SpinToBreak = TotalBreak / (SpinRate/1000 · ZoneTime)`: proxy de eficiencia de giro (H4).
- `DragProxy = SpeedDrop / RelSpeed`.
- Arm slot aproximado a partir de `RelHeight`, `RelSide` y `Extension`: es el puente con VeloVision.

### 4.7 Targets
- **xRV por pitch** desde la perspectiva del pitcher = valor de transición de conteo + linear weights de eventos, **estimados con datos LMB** (`RunsScored`, outs × conteo).
- **Multi-target de las bases:** P(whiff), P(chase), P(called strike), P(weak contact), P(GB), P(barrel/hard contact), xRV, P(out en la PA).
- **Pesos de combinación** (para obtener el score final): salen de una regresión del RV observado sobre las probabilidades.

## 5. Efecto de la altitud: estrategia de identificación
> Las bases dejan el Causal Forest como **opcional** y para preguntas tácticas. Stuff+ es ante todo un modelo de **predicción**. Lo de esta sección no reemplaza al modelo: es la **evidencia empírica del delta Stuff+ por pitch shape** que pide el criterio "Integración arsenal + altitud".

**Pregunta:** ¿cuánto cambia el movimiento y el valor de un *mismo* pitcheo por la densidad del aire?
- **Confusor principal:** la identidad del pitcher. Los pitchers de equipos de altura lanzan más ahí y tienen perfiles distintos.
- **Identificación:** variación **intra-pitcher** (el mismo `pitch_id` en distintas densidades), apoyada en los **visitantes**, cuya asignación a parques depende del calendario. Se implementa con efectos fijos de pitcher-temporada; opcionalmente con Double ML / Causal Forest (`econml`) para la heterogeneidad por forma (`T` = ρ̂, `X` = forma, `W` = efectos fijos + conteo + mano del bateador).
- **Etapa A (física):** `release + velo + spin + eje + ρ̂ → IVB, HB`. Da el efecto heterogéneo de ρ según la forma. Para el movimiento, el bateador no es confusor.
- **Etapa B (valor):** `forma → xRV` (el modelo Stuff+).
- **Descomposición de canales:**
  - El efecto total de CDMX sobre el xRV se divide en **canal Stuff** (pasa por el cambio de movimiento, A → B) y **canal parque** (residuo: valor de la bola en juego, *carry*).
  - El Stuff+ de entorno incorpora solo el canal Stuff. El canal parque se reporta como factor de parque y riesgo de HR.
- **Supuestos que se declaran en el paper:**
  - No hay confusión residual intra-pitcher (fatiga o cambio de mecánica entre parques).
  - Hay *overlap* (pitchers con lanzamientos en ambas densidades).
  - El reetiquetado está controlado vía `pitch_id`.
- **Capa táctica (P2, como sugieren las bases):** Causal Forest para preguntas del tipo "¿slider o cambio en este conteo contra este bateador?".

## 6. Modelos
| Modelo | Inputs | Prohibido |
|---|---|---|
| **Stuff+** | forma física en el release + arsenal + ρ (neutral = ρ₀; entorno = ρ del estadio) | ubicación, conteo, bateador, resultados |
| **Location+** | zona (9 + fuera), `PlateLoc*`, conteo, mano del bateador, condicionado a `pitch_id` | forma física del pitcheo |
| **Pitching+** | Stuff + ubicación + arsenal + contexto (outs, conteo, inning, `prev_pitch_type`, ρ) | resultados |

LightGBM principal + SHAP. Contrafactual = mismos inputs a ρ_CDMX vs ρ₀. Escala: 100 = promedio LMB ajustado por temporada y estadio (sabor neutral).

## 7. Producto

### 7.1 Dashboard simulador (P0, obligatorio por las bases)
1. **Simulador manual:** el coach captura velo, spin, eje, IVB/HB, release y extensión, y obtiene el Stuff+.
2. **Comparación lado a lado:** el mismo pitch en Harp Helú vs nivel del mar (ΔStuff+ y movimiento perdido).
3. **Mapa de movimiento** con los pitches más y menos efectivos.
4. Leaderboard por pitcher y arsenal (neutral vs CDMX) y ficha con SHAP.

UI en español, rojo Diablos + negro. Solo muestra agregados.

### 7.2 Pitch Sculptor (P1)
- **Entrada:** un pitcheo o el arsenal de un pitcher.
- **Búsqueda:** variaciones de forma (eje, eficiencia, Δvelo vs la recta, agregar o quitar un pitcheo) que maximizan el **Stuff+ CDMX**.
- **Restricción de factibilidad:** solo formas que aparecen en la distribución condicional de pitchers con slot y velocidad similares (vecinos cercanos o densidad condicional).
- **Salida:** "forma X → Y: +N Stuff+ CDMX `[calculado]`", con comparables reales que ya lanzan esa forma.
- **Capa de coaching** (cualitativa y etiquetada): agarre o pronación sugeridos para llegar a esa forma. No es salida del modelo.

### 7.3 VeloVision (P1, mínimo viable)
- **Entrada:** video `.mp4` de bullpen o transmisión.
- **Proceso:** MediaPipe Pose detecta el frame de release y estima arm slot y altura/extensión relativas. Las medidas absolutas requieren calibración de escala (altura del jugador o distancia del montículo).
- **Puente con los datos:** el slot define la distribución de formas que lanzan pitchers con slot similar, y de ahí sale un **rango** de Stuff+ CDMX. Si se captura velo o spin (radar de bolsillo), el rango se estrecha.
- **Validación:** los IDs del dataset son anónimos, así que **no se puede emparejar un video con un pitcher del dataset**. Plan:
  1. **Principal:** validar el arm slot contra **anotación manual de 2 personas** en los mismos frames. Se reporta el error y el acuerdo entre anotadores.
  2. **Opcional:** clips públicos con arm angle de Statcast. Las bases no prohíben ni autorizan datos externos, así que se usarán solo si los organizadores lo confirman por correo. Se presentarían como validación de la herramienta de visión, nunca como datos de entrenamiento del modelo.

## 8. Entregables (según las bases)
1. **Scores:** Stuff+, Location+ y Pitching+ por pitch (`PitchUID`), con agregados por pitcher y repertorio. Incluye qué arsenal rinde más en el Harp Helú y la comparación CDMX vs otros estadios.
2. **Perfil físico ideal en CDMX:** la combinación de forma y arsenal que maximiza el Stuff+ en el Harp Helú, con argumentación física y estadística (H1–H4).
3. **Recomendaciones operativas:**
   - Perfiles que ganan Stuff+ relativo, con su explicación física.
   - Perfiles que pierden, cuantificando el riesgo de contacto duro y HR.
   - Cómo perfilar a un prospecto o agente libre (Sculptor + VeloVision).
   - El staff de Diablos solo se puede identificar reconstruyendo equipos por alineaciones (P2). Si no, se reporta por "pitchers con más juegos en Extreme".
4. **Dashboard simulador** (§7.1).
5. **Documento técnico** de 6–10 páginas: marco teórico, decisiones de modelado (arquitectura, features, función de pérdida), definición formal de la escala, validación fuera de muestra, alcance y fronteras.

## 9. Rúbrica → evidencia
| Criterio | Peso | Evidencia |
|---|---:|---|
| Calidad | 25 | RMSE/log-loss/calibración en holdout vs baseline nulo (xRV medio por conteo) |
| Robustez | 25 | holdout de pitcher, temporada y parque; estabilidad año-a-año |
| Arsenal + altitud | 25 | features de arsenal y túnel, delta Stuff+ por pitch shape con IC (intra-pitcher, ρ̂), canales, H1–H4 |
| Aplicabilidad | 25 | **entrevista de 10 min**: ¿el jurado cambiaría sus decisiones de reclutamiento y táctica? Hay que llevar 3 casos concretos (fichaje, ajuste de pitch, uso en casa vs gira) |

## 10. Limitaciones conocidas (decirlas antes de que las diga el jurado)
- Sin orientación de costura: el SSW no es modelable con estos datos.
- Sin estadio ni altitud en metros: `Extreme Altitude` mezcla varios parques.
- Sin corredores: el valor de carrera es aproximado (outs × conteo).
- VeloVision no mide spin ni movimiento: por eso entrega rangos.

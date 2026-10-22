# Guion del video demo — Diablos AeroVision (máx. 5:00)

> **Meta:** 4:50 de contenido + 10 s de margen. A ~140 palabras por minuto, cada bloque tiene su presupuesto de palabras.
> **Voces (2 personas):**
> - **P1 = Diapositivas** (sin acceso a la demo): abre, explica el problema y la evidencia, introduce VeloVision con una lámina y cierra. Graba su voz y sus láminas por separado.
> - **P2 = Demo** (maneja la app): narra Simulador, Leaderboard y VeloVision → Sculptor sobre la pantalla en vivo.
> **Cifras:** todas salen de `reports/` y del modelo (ver `README.md`). No se dice ninguna cifra que no esté ahí.

## 1. Revisión de la idea original

| Idea original | Problema | Ajuste |
|---|---|---|
| 2:00 de elevator pitch | Un gancho de 2 min se vuelve discurso; la atención se gana o se pierde en los primeros 30 s | **0:35 de gancho** con una pregunta y un número |
| 1:00 de teoría | Bien, pero el 50 % de la rúbrica (calidad + robustez) es **validación**, y no aparecía | **1:00 de problema → solución → evidencia** (física en una frase y holdouts en otra) |
| 2:00 de demo | Corta para 5 herramientas; la aplicabilidad vale 25 % | **2:40 de demo** con un hilo narrativo (un mismo pitcher de punta a punta) |
| Sin cierre | El jurado se queda sin el "y entonces qué" | **0:35 de cierre**: qué decisión cambia, qué no afirmamos y la URL |

## 2. Estructura

| Tiempo | Bloque | Voz | En pantalla |
|---|---|---|---|
| 0:00–0:35 | Gancho (elevator pitch) | P1 | Lámina 1: pregunta + "15 % menos aire" y "−4.5 Stuff+" en grande |
| 0:35–1:35 | Problema, solución y evidencia | P1 | Láminas 2–4: física en una línea → tres modelos → holdouts y descomposición |
| 1:35–2:35 | Demo 1: Simulador | P2 | App: Simulador con el preset "Recta alta" |
| 2:35–3:05 | Demo 2: Leaderboard | P2 | App: Leaderboard ordenado por pérdida → tarjeta de lanzador |
| 3:05–3:20 | Puente a VeloVision (¿y sin TrackMan?) | P1 | Lámina 5: flujo video → cuerpo → soltura → mecánica → pares reales |
| 3:20–4:15 | Demo 3: VeloVision → Sculptor | P2 | App: video → frame de soltura → ficha → salto al Sculptor |
| 4:15–4:50 | Cierre | P1 | Lámina 6: tres decisiones, límites y URL en grande |

Reparto: P1 ≈ 130 s · P2 ≈ 165 s (el reparto es más parejo que antes y cada quien graba por separado).

## 3. Guion

### 0:00–0:35 · Gancho — P1 (≈ 80 palabras)
*Lámina 1.*
> "¿Ficharían a un cerrador por su recta de mucha subida… sin saber cuánta le queda en la Ciudad de México?
> En el Harp Helú hay **15 % menos aire** que a nivel del mar. Menos aire, menos movimiento: una recta 4 costuras pierde en promedio **4.5 puntos de Stuff+** al llegar aquí.
> Somos [equipo] y construimos **Diablos AeroVision**: la herramienta que le dice al cuerpo técnico de los Diablos cuánto valor pierde cada pitcheo por la altitud y qué referencias reales seguir para recuperarlo."

*Fuentes: ρ̂ = 0.852 del modelo; Four-Seam −4.52 de `pitcher_arsenal_breakdown.csv`.*

### 0:35–1:35 · Problema, solución y evidencia — P1 (≈ 140 palabras)
*Láminas 2 (física), 3 (tres modelos), 4 (holdouts + descomposición). Cambiar de lámina en cada párrafo.*
> "La física es simple: el efecto Magnus, el arrastre y el *seam-shifted wake* dependen de la densidad del aire; solo la gravedad no cambia. Así que en CDMX las pelotas **se mueven menos**.
> Con **634 mil pitcheos** de la LMB construimos tres modelos: **Stuff+** para la calidad física del pitcheo, **Location+** para la ubicación y **Pitching+** para los dos juntos. Cada uno se calcula a nivel del mar y en el aire del Harp Helú.
> ¿Por qué creernos? Validamos en tres escenarios que el modelo no vio: pitchers nuevos, la temporada 2026 y los parques de altura. **Pitching+ reduce entre 4 y 5 % el error frente al baseline** en los tres. Stuff+ es muy estable de un año a otro (r = 0.74).
> Y separamos la causa: del efecto total de la altitud, cerca del **9 % es pérdida de movimiento**, el **42 % son batazos que vuelan más** y el resto no lo atribuimos a nadie. Lo decimos tal cual."

*Última frase de P1 antes de la demo (≈ 3 s): "Y ahora, lo vemos funcionando." → corte a la pantalla de P2.*

### 1:35–2:35 · Demo 1: Simulador — P2 (≈ 140 palabras)
*Clic en "Abrir el simulador" → preset "Recta alta".*
> "Este es el simulador. Cargo una recta de mucha subida.
> A la izquierda, a nivel del mar; a la derecha, en el Harp Helú. El veredicto lo dice en español: **pierde casi 3 pulgadas de subida** y su Stuff+ cae diez puntos. En el diagnóstico se ve la flecha: el aire delgado la aplana.
> Si muevo el giro o la velocidad, todo se recalcula en menos de un segundo. Así el coach prueba ajustes antes de pedírselos al pitcher.
> Y abajo, en modo analista, está el porqué: qué variables empujan el Stuff+ y la evidencia estadística, para quien quiera auditar."

*Mostrar: veredicto → tarjetas → flecha del diagnóstico → mover el slider de giro → abrir y cerrar el Modo Analista.*

### 2:35–3:05 · Demo 2: Leaderboard — P2 (≈ 70 palabras)
> "Para fichar, el leaderboard: toda la liga ordenada por quién pierde más en CDMX. Abro la tarjeta de un lanzador: su arsenal, qué pitcheo sufre y cuál aguanta la altitud. Y con un clic, lo llevo al simulador."

*Mostrar: orden por ΔStuff+ → tarjeta → mapa del arsenal → botón "Llevar al Simulador".*

### 3:05–3:20 · Puente a VeloVision — P1 (≈ 35 palabras)
*Lámina 5: flujo de cinco pasos (video de celular → detección del cuerpo → momento de soltura → mecánica con margen de error → pares reales de la LMB).*
> "¿Y un prospecto sin TrackMan? VeloVision parte de un video de celular: detecta el cuerpo, encuentra la soltura y mide la mecánica del brazo. Lo vemos con un clip real."

*Corte a la pantalla de P2 con la pestaña VeloVision abierta.*

### 3:20–4:15 · Demo 3: VeloVision → Sculptor — P2 (≈ 130 palabras)
*Pestaña VeloVision con un clip ya cargado (procesado antes de grabar).*
> "Aquí está el clip. VeloVision encuentra el momento de soltura, y el coach lo confirma aquí. Mide altura, lado y ángulo del brazo, **con la misma fórmula que los datos de la liga y con su margen de error**.
> No inventamos su spin ni su velocidad. La ficha dice qué logran en el Harp Helú **los pitchers LMB con una mecánica parecida**, por tipo de pitcheo: con este brazo, el slider aguanta y la recta sufre. Si el video es malo, la app se niega a dar resultados y explica por qué.
> Un clic y pasamos al **Sculptor**: los pares reales con esa mecánica, el mejor de ellos y qué pitcheo secundario les funciona. Es un plan de trabajo con referencias reales, no una receta mágica."

*Mostrar: guía de captura → barra de frames → confirmar → tarjetas de medidas → ficha por familia → botón → salto al Sculptor → barra p25–p75 y comparables.*

### 4:15–4:50 · Cierre — P1 (≈ 80 palabras)
*Lámina 6.*
> "Tres decisiones que AeroVision cambia: **a quién fichar** para jugar en casa, **qué pitcheo desarrollar** a cada brazo y **cómo planear** la rotación entre casa y gira.
> Y lo que no afirmamos: Stuff+ describe el pitcheo, no predice cada lanzamiento, y VeloVision entrega rangos, no lecturas de TrackMan.
> Diablos AeroVision ya está en línea: **diablos-aerovision.streamlit.app**. Gracias."

*Pantalla final: URL grande + logo + "Hackathon ISAC 2026 · Reto 2".*

## 4. Láminas de P1 (6)

| # | Contenido | Se usa en |
|---|---|---|
| 1 | Pregunta del gancho + dos cifras grandes: **15 % menos aire**, **−4.5 Stuff+** | 0:00–0:35 |
| 2 | Física en una línea: Magnus, arrastre y *seam-shifted wake* ∝ densidad del aire; la gravedad no cambia | 0:35–1:00 |
| 3 | Tres modelos: Stuff+, Location+, Pitching+ (nivel del mar vs. Harp Helú), 634 mil pitcheos | 1:00–1:15 |
| 4 | Holdouts (pitchers nuevos, 2026, parques de altura): −4 a −5 % de error; r = 0.74; descomposición 9 % / 42 % / resto sin atribuir | 1:15–1:35 |
| 5 | Flujo de VeloVision en cinco pasos (sin cifras nuevas) | 3:05–3:20 |
| 6 | Tres decisiones, límites del modelo y URL grande + logo | 4:15–4:50 |

Las láminas de P1 son las únicas con cifras: pueden vivir en un deck (PowerPoint o Google Slides) y exportarse a imagen o grabarse en pantalla.

## 5. Producción (dos personas, uno solo con la demo)

1. **P2 graba la pantalla** a 1920×1080, con el zoom del navegador en 100 % y Sculptor, VeloVision y Modo anotación ya al 125 % por la app. Usar OBS o la grabadora de Windows (Win+Alt+R). P2 graba **tres tomas de pantalla**, una por bloque (Simulador, Leaderboard, VeloVision → Sculptor), para poder cortar errores.
2. **P1 prepara las láminas y graba su voz** (idealmente también con el deck en pantalla). No necesita la app.
3. **Edición:** montar las tomas en el orden de la tabla de la sección 2. Los cortes entre P1 y P2 son directos; no hace falta que las voces se escuchen juntas.
4. **Audio aparte:** ambas personas graban sus voces con buen micrófono, en un cuarto sin eco, al mismo nivel. Así se pueden recortar silencios y empatar niveles.
5. **P2 usa la app local**, más estable y sin arranque en frío, pero la URL pública se muestra en el cierre (lámina 6 de P1). Abrirla 2 minutos antes para calentar la caché.
6. **Pre-cargar** el clip de VeloVision y confirmar el frame ANTES de grabar, para no esperar el procesamiento en cámara.
7. **El clip de la demo debe ser propio o con licencia de uso público.** Los previews de iStock con marca de agua no se deben mostrar en el video.
8. **Subtítulos** con las cifras clave (15 % menos aire, −4.5 Stuff+, 4–5 % menos error, 9 % / 42 %).
9. **Ensayo con cronómetro:** cada bloque en su tiempo; si se pasa, recortar del bloque de ciencia (P1) antes que de la demo.
10. **No leer en pantalla:** cada quien se aprende sus frases clave y no el guion palabra por palabra.
11. **Como P2 es quien más habla sobre la pantalla**, P1 debe revisar al menos una vez la demo grabada para confirmar que las cifras que P2 dice coinciden con las de las láminas.

## 6. Checklist antes de grabar

**P2 (demo)**
- [ ] La app local corre con la última versión (`git pull`) y se reinició Streamlit.
- [ ] El preset "Recta alta" da el veredicto esperado; anotar sus cifras reales del día, porque el guion cita "casi 3 pulgadas" y "diez puntos".
- [ ] El Leaderboard está ordenado por pérdida.
- [ ] El clip propio está cargado en VeloVision, con el frame confirmado y la ficha visible.
- [ ] El salto VeloVision → Sculptor funciona.

**P1 (láminas)**
- [ ] Las 6 láminas están listas y las cifras coinciden con `reports/` y `README.md`.
- [ ] La URL pública abre y está escrita bien en la lámina 6.
- [ ] Nombre del equipo puesto en el gancho (reemplazar `[equipo]`).

**Ambos**
- [ ] Duración total ≤ 5:00.

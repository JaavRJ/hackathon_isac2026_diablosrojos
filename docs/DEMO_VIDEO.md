# Guion del video demo — Diablos AeroVision (máx. 5:00)

> **Meta:** 4:50 de contenido + 10 s de margen. A ~140 palabras por minuto, cada bloque tiene su presupuesto de palabras.
> **Voces:** **P1 = Negocio** (abre y cierra), **P2 = Ciencia** (problema, evidencia y VeloVision), **P3 = Demo** (maneja la pantalla y narra Simulador y Leaderboard).
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
| 0:00–0:35 | Gancho (elevator pitch) | P1 | Inicio: hero animado → zoom a las cifras ancla |
| 0:35–1:35 | Problema, solución y evidencia | P2 | Inicio: física en una línea → tarjeta de holdouts → descomposición |
| 1:35–2:35 | Demo 1: Simulador | P3 | Simulador con el preset "Recta alta" |
| 2:35–3:05 | Demo 2: Leaderboard | P3 | Leaderboard ordenado por pérdida → tarjeta de lanzador |
| 3:05–4:15 | Demo 3: VeloVision → Sculptor | P2 | Video → frame de soltura → ficha → salto al Sculptor |
| 4:15–4:50 | Cierre | P1 | Inicio: herramientas → URL en grande |

Reparto: P1 ≈ 70 s · P2 ≈ 130 s · P3 ≈ 90 s.

## 3. Guion

### 0:00–0:35 · Gancho — P1 (≈ 80 palabras)
> "¿Ficharían a un cerrador por su recta de mucha subida… sin saber cuánta le queda en la Ciudad de México?
> En el Harp Helú hay **15 % menos aire** que a nivel del mar. Menos aire, menos movimiento: una recta 4 costuras pierde en promedio **4.5 puntos de Stuff+** al llegar aquí.
> Somos [equipo] y construimos **Diablos AeroVision**: la herramienta que le dice al cuerpo técnico de los Diablos cuánto valor pierde cada pitcheo por la altitud y qué referencias reales seguir para recuperarlo."

*Fuentes: ρ̂ = 0.852 del modelo; Four-Seam −4.52 de `pitcher_arsenal_breakdown.csv`.*

### 0:35–1:35 · Problema, solución y evidencia — P2 (≈ 140 palabras)
> "La física es simple: el efecto Magnus, el arrastre y el *seam-shifted wake* dependen de la densidad del aire; solo la gravedad no cambia. Así que en CDMX las pelotas **se mueven menos**.
> Con **634 mil pitcheos** de la LMB construimos tres modelos: **Stuff+** para la calidad física del pitcheo, **Location+** para la ubicación y **Pitching+** para los dos juntos. Cada uno se calcula a nivel del mar y en el aire del Harp Helú.
> ¿Por qué creernos? Validamos en tres escenarios que el modelo no vio: pitchers nuevos, la temporada 2026 y los parques de altura. **Pitching+ reduce entre 4 y 5 % el error frente al baseline** en los tres. Stuff+ es muy estable de un año a otro (r = 0.74).
> Y separamos la causa: del efecto total de la altitud, cerca del **9 % es pérdida de movimiento**, el **42 % son batazos que vuelan más** y el resto no lo atribuimos a nadie. Lo decimos tal cual."

### 1:35–2:35 · Demo 1: Simulador — P3 (≈ 140 palabras)
*Clic en "Abrir el simulador" → preset "Recta alta".*
> "Este es el simulador. Cargo una recta de mucha subida.
> A la izquierda, a nivel del mar; a la derecha, en el Harp Helú. El veredicto lo dice en español: **pierde casi 3 pulgadas de subida** y su Stuff+ cae diez puntos. En el diagnóstico se ve la flecha: el aire delgado la aplana.
> Si muevo el giro o la velocidad, todo se recalcula en menos de un segundo. Así el coach prueba ajustes antes de pedírselos al pitcher.
> Y abajo, en modo analista, está el porqué: qué variables empujan el Stuff+ y la evidencia estadística, para quien quiera auditar."

*Mostrar: veredicto → tarjetas → flecha del diagnóstico → mover el slider de giro → abrir y cerrar el Modo Analista.*

### 2:35–3:05 · Demo 2: Leaderboard — P3 (≈ 70 palabras)
> "Para fichar, el leaderboard: toda la liga ordenada por quién pierde más en CDMX. Abro la tarjeta de un lanzador: su arsenal, qué pitcheo sufre y cuál aguanta la altitud. Y con un clic, lo llevo al simulador."

*Mostrar: orden por ΔStuff+ → tarjeta → mapa del arsenal → botón "Llevar al Simulador".*

### 3:05–4:15 · Demo 3: VeloVision → Sculptor — P2 (≈ 160 palabras)
*Pestaña VeloVision con un clip ya cargado (procesado antes de grabar).*
> "¿Y un prospecto sin TrackMan? VeloVision trabaja con un video de celular.
> Detecta el cuerpo, encuentra el momento de soltura (el coach lo confirma aquí) y mide la mecánica: altura, lado y ángulo del brazo, **con la misma fórmula que los datos de la liga y con su margen de error**.
> No inventamos su spin ni su velocidad. La ficha dice qué logran en el Harp Helú **los pitchers LMB con una mecánica parecida**, por tipo de pitcheo: con este brazo, el slider aguanta y la recta sufre. Si el video es malo, la app se niega a dar resultados y explica por qué.
> Un clic y pasamos al **Sculptor**: los pares reales con esa mecánica, el mejor de ellos y qué pitcheo secundario les funciona. Es un plan de trabajo con referencias reales, no una receta mágica."

*Mostrar: guía de captura → barra de frames → confirmar → tarjetas de medidas → ficha por familia → botón → salto al Sculptor → barra p25–p75 y comparables.*

### 4:15–4:50 · Cierre — P1 (≈ 80 palabras)
> "Tres decisiones que AeroVision cambia: **a quién fichar** para jugar en casa, **qué pitcheo desarrollar** a cada brazo y **cómo planear** la rotación entre casa y gira.
> Y lo que no afirmamos: Stuff+ describe el pitcheo, no predice cada lanzamiento, y VeloVision entrega rangos, no lecturas de TrackMan.
> Diablos AeroVision ya está en línea: **diablos-aerovision.streamlit.app**. Gracias."

*Pantalla final: URL grande + logo + "Hackathon ISAC 2026 · Reto 2".*

## 4. Producción

1. **Grabación de pantalla** a 1920×1080, con el zoom del navegador en 100 % y Sculptor, VeloVision y Modo anotación ya al 125 % por la app. Usar OBS o la grabadora de Windows (Win+Alt+R).
2. **Usar la app local**, más estable y sin arranque en frío, pero mostrar la URL pública en el cierre. Abrirla 2 minutos antes para calentar la caché.
3. **Pre-cargar** el clip de VeloVision y confirmar el frame ANTES de grabar, para no esperar el procesamiento en cámara.
4. **El clip de la demo debe ser propio o con licencia de uso público.** Los previews de iStock con marca de agua no se deben mostrar en el video.
5. **Audio aparte:** grabar las voces con buen micrófono y montarlas sobre la pantalla. Así se pueden recortar silencios.
6. **Subtítulos** con las cifras clave (15 % menos aire, −4.5 Stuff+, 4–5 % menos error, 9 % / 42 %).
7. **Ensayo con cronómetro:** cada bloque en su tiempo; si se pasa, recortar del bloque de ciencia antes que de la demo.
8. **No leer en pantalla:** cada quien se aprende sus frases clave y no el guion palabra por palabra.

## 5. Checklist antes de grabar

- [ ] La app local corre con la última versión (`git pull`) y se reinició Streamlit.
- [ ] El preset "Recta alta" da el veredicto esperado; anotar sus cifras reales del día, porque el guion cita "casi 3 pulgadas" y "diez puntos".
- [ ] El Leaderboard está ordenado por pérdida.
- [ ] El clip propio está cargado en VeloVision, con el frame confirmado y la ficha visible.
- [ ] El salto VeloVision → Sculptor funciona.
- [ ] La URL pública abre.
- [ ] Duración total ≤ 5:00.

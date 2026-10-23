# Imágenes para el video demo

Seis láminas horizontales 16:9, en el orden de `docs/DEMO_VIDEO.md`.

| Imagen | Momento del video | Contenido |
|---|---|---|
| `01_gancho.png` | 0:00–0:35 | Pregunta inicial, densidad y pérdida media de la recta |
| `02_fisica.png` | 0:35–1:00 | Fuerzas aerodinámicas y gravedad |
| `03_modelos.png` | 1:00–1:15 | Stuff+, Location+ y Pitching+ |
| `04_evidencia.png` | 1:15–1:35 | Holdouts, estabilidad y descomposición descriptiva |
| `05_velovision.png` | 3:05–3:20 | Video, pose, soltura, mecánica y pares reales |
| `06_cierre.png` | 4:15–4:50 | Decisiones, límites y URL |

Inserta cada PNG como imagen fija en un proyecto de video 1920 × 1080. Conserva la proporción y evita recortar las notas inferiores. Los intervalos siguen el guion y pueden ajustarse al ritmo de la narración. Las secciones de demo utilizan la grabación de la app.

## Cifras y alcance

- El escenario llamado CDMX usa la mediana de la categoría Altitud Extrema (ρ̂ = 0.852), que mezcla parques. No es una medición aislada del Harp Helú.
- La pérdida media de la recta de cuatro costuras es ponderada por número de pitcheos: `reports/leaderboards/pitcher_arsenal_breakdown.csv`.
- Los porcentajes de validación son reducciones relativas de RMSE de Pitching+ en `xrv_pitcher` frente al baseline por conteo. Stuff+ empata con ese baseline en xRV: `reports/validation/holdout_metrics.csv`.
- La correlación de Stuff+ es en muestra 2024–25, con 202 pitchers: `reports/validation/year_to_year_stability.csv`.
- La descomposición es descriptiva. El canal parque no establece causalidad y el residuo queda sin atribuir: `reports/altitude/channel_decomposition.csv`.
- Las ilustraciones son generadas y esquemáticas. No representan trayectorias medidas, un lanzador identificado ni una fotografía del estadio.

## Identidad y generación

Referencia: página pública de AeroVision y `dashboard/theme.py`. Fondo claro, texto oscuro, cifras en Barlow Condensed e Inter como referencia. Rojo de marca en el encabezado, navy para contexto, coral para pérdidas, turquesa para mejoras, azul y morado para los canales.

Generación mediante la herramienta integrada ImageGen. `prompts.json` conserva las instrucciones de cada imagen.

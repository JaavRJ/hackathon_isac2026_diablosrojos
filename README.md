# Diablos AeroVision
**Stuff+ LMB · Pitch Sculptor · VeloVision**
ISAC Sports Analytics Hackathon 2026 · Reto Diablos Rojos — *calidad de pitcheo ajustada por entorno*.

AeroVision evalúa cada lanzamiento de la LMB con tres submodelos: **Stuff+** (calidad física + arsenal), **Location+** (comando) y **Pitching+** (combinado). Los reporta en dos sabores: **neutral** (nivel del mar) y **CDMX** (Harp Helú, ~2,240 m).

Además del simulador obligatorio (entrada manual de características, comparación lado a lado y mapa de movimiento):
- **Descompone** el efecto de la altitud en *canal Stuff* (el pitcheo se mueve menos) y *canal parque* (la bola bateada vuela más).
- **Recomienda** qué forma de pitcheo factible recupera valor en CDMX (**Pitch Sculptor**).
- **Acepta video** de bullpen para estimar un rango de Stuff+ en prospectos sin TrackMan (**VeloVision**).

Escala: `100` = promedio LMB, `110` = una desviación estándar mejor.

| Qué | Dónde |
|---|---|
| Reglas para agentes | [AGENTS.md](AGENTS.md) |
| Idea, física, métodos, entregables | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) |
| Benchmark del rival | [docs/COMPETENCIA.md](docs/COMPETENCIA.md) |

## Instalación
Python ≥ 3.11.
```bash
pip install -r requirements.txt
```
`pyarrow >= 20` es obligatorio para leer el parquet de ISAC.

## Datos
Los datos de ISAC **no se suben a git**. Ruta esperada:
`C:\Users\L13Yoga\Downloads\Diablos - Hackathon 2026\stuff_model_df.parquet`
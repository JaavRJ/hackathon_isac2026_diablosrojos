# Continuar el proyecto en otra computadora

Estado a la fecha (22-oct-2026): el código, los modelos entrenados (`models/*.joblib`), el modelo de pose (`models/pose_landmarker_full.task`) y los reportes agregados (`reports/**/*.csv`) están en git. **La app corre solo con un `git clone` y las dependencias.** Lo que no está en git son los datos y los clips de video. El tablero de tareas y su estado real están en [`TASKS.md`](TASKS.md); la app pública y su despliegue, en [`DEPLOY.md`](DEPLOY.md).

## 1. Clonar y preparar Python

Python 3.12 (lo probado es 3.12.1; 3.11 o superior debería servir). Desde la raíz:

```powershell
git clone https://github.com/JaavRJ/hackathon_isac2026_diablosrojos.git
cd hackathon_isac2026_diablosrojos
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt          # lo que importa la app desplegada (versiones fijadas)
pip install -r requirements-dev.txt      # análisis y tests: incluye requirements.txt + pytest, shap, xgboost, catboost, econml, etc.
```

`requirements-dev.txt` es necesario para correr `pytest` y los scripts de `analysis/`. `packages.txt` es solo para Streamlit Cloud (apt), no se instala en local.

## 2. Qué trae git y qué no

| En git | Fuera de git (cópialo a mano) |
| :--- | :--- |
| `dashboard/`, `src/`, `analysis/`, `tests/`, `docs/` | `data/stuff_model_df.parquet` (crudo) |
| `models/*.joblib` y `models/pose_landmarker_full.task` | `data/stuff_processed_df.parquet` (se regenera con `python -m src.pipeline_step1`) |
| `reports/**/*.csv` | `data/stuff_plus_data_dictionary.csv` (los tests que consultan el diccionario) |
| `.streamlit/config.toml`, `requirements*.txt`, `packages.txt` | `data/vision_clips/` (clips de VeloVision y del Modo anotación) |
| | `data/deliverables/` (scores por pitcheo; salida de `analysis.export_scores`) |

Los datos de ISAC son privados: no los publiques ni los subas al despliegue. `.gitignore` excluye `data/`, `*.parquet`, `*.mp4`, `*.mov`.

## 3. Verificar que todo funciona

```powershell
.venv\Scripts\python.exe -m pytest -q        # usa un fixture sintético; no necesita el parquet
streamlit run dashboard/app.py               # usa models/ y reports/
```

## 4. Regenerar reportes (solo si se autoriza otra corrida)

Requiere los parquets de `data/`. Orden y detalles en el README de la raíz (sección 7). Las corridas de validación y de hipótesis son pesadas; `analysis.run_validation` tiene `--resume` para reanudar desde checkpoints agregados (`reports/validation/*_partial.csv`, `run_status.json`). Confirma con `python -m analysis.run_validation --help` que la opción existe en tu copia.

## 5. Notas

- Los scripts resuelven el parquet crudo desde `data/` o desde las rutas definidas en `src/config.py`; si no lo encuentran, indícalo con `--raw-path` / `--input`.
- El Modo anotación (dos anotadores) y `analysis.validate_vision` solo funcionan en local, con los clips en `data/vision_clips/`.
- Ninguna tarea puede marcarse `APPROVED` sin su revisión; ver `AGENTS.md` y `TASKS.md`.

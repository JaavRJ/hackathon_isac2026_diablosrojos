# Continuar los análisis en otra computadora

Estado al cerrar este traslado (20-oct-2026): HACK-012 y HACK-013 están en REVIEW. La validación HACK-012 terminó sobre 634,624 lanzamientos y sus tres reportes están en `reports/validation/`; no hace falta repetir esa corrida para continuar la revisión. Stuff+ no supera al baseline en todos los criterios y la estabilidad es in-sample: consultar las notas de `docs/TASKS.md`.

HACK-013 tiene código corregido y tests, pero sus tablas anteriores fueron retiradas porque no corresponden al código final. No se volvió a entrenar ni se guardó el modelo previo. Su regeneración queda pendiente, sujeta a nueva instrucción del usuario. Los seguimientos automáticos fueron eliminados y no se inició revisión posterior ni otra tarea del sprint.

## Archivos necesarios

- Código completo, `requirements.txt`, `docs/`, `tests/` y reportes agregados del commit de traslado.
- `data/stuff_processed_df.parquet` (datos privados, ignorados en git).
- Para ejecutar los tests que consultan el diccionario: copiar también `stuff_plus_data_dictionary.csv` a `data/` desde la carpeta original del dataset.
- Si existen checkpoints: copiar juntos `reports/validation/run_status.json`, `holdout_metrics_partial.csv` y `calibration_deciles_partial.csv`.

El ZIP local de traslado contiene datos privados y una copia anterior del código. Usar Git para obtener el código final; del ZIP recuperar los datos necesarios sin sobrescribir ese código. No publicar el ZIP ni enviarlo al deploy.

## Preparar Python

Desde la raíz del proyecto, usar Python 3.11 o superior y crear un entorno virtual. Instalar `numpy pandas pyarrow scipy scikit-learn lightgbm shap joblib pytest`; para el proyecto completo, usar `python -m pip install -r requirements.txt` e instalar también `pytest`.

## Ejecutar solo si se autoriza otra corrida

Validación nueva con checkpoints por partición:

```powershell
python -m analysis.run_validation --input data/stuff_processed_df.parquet --reports-dir reports/validation
```

Reanudar una corrida que sí haya guardado checkpoints compatibles:

```powershell
python -m analysis.run_validation --input data/stuff_processed_df.parquet --reports-dir reports/validation --resume
```

La opción `--resume` no recupera los resultados en memoria de la corrida original. Antes de usarla, comprobar que el código instalado la muestra en `python -m analysis.run_validation --help`.

Análisis de altitud (entrena el modelo si no se proporciona uno ajustado):

```powershell
python -m analysis.run_hypotheses --input data/stuff_processed_df.parquet --output-dir reports/altitude
```

## Qué se conserva

Los checkpoints de validación preservan métricas agregadas de particiones terminadas. El entrenamiento de una partición a medias se vuelve a ejecutar. El análisis de altitud actual no ofrece reanudación del entrenamiento. Ninguna tarea puede marcarse APPROVED hasta completar sus verificaciones y la revisión independiente.

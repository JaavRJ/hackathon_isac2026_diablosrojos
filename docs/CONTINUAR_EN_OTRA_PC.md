# Continuar los análisis en otra computadora

Estado al preparar este traslado (20-oct-2026): HACK-012 y HACK-013 están IN_PROGRESS. La corrida original de validación completó dos particiones y empezó la tercera, pero no guardó sus resultados en disco. Ese avance en memoria no se transfiere copiando archivos. No apagar la computadora original si se desea conservar esas corridas.

## Archivos necesarios

- Código completo, `requirements.txt`, `docs/` y `tests/`, incluidos los archivos nuevos sin commit.
- `data/stuff_processed_df.parquet` (datos privados, ignorados en git).
- Para ejecutar los tests que consultan el diccionario: copiar también `stuff_plus_data_dictionary.csv` a `data/` desde la carpeta original del dataset.
- Si existen checkpoints: copiar juntos `reports/validation/run_status.json`, `holdout_metrics_partial.csv` y `calibration_deciles_partial.csv`.

El paquete local de traslado no es un artefacto para publicar ni enviar al deploy: contiene datos de la competencia.

## Preparar Python

Desde la raíz del proyecto, usar Python 3.11 o superior y crear un entorno virtual. Instalar `numpy pandas pyarrow scipy scikit-learn lightgbm shap joblib pytest`; para el proyecto completo, usar `python -m pip install -r requirements.txt` e instalar también `pytest`.

## Ejecutar

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

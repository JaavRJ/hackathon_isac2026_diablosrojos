# Despliegue publico de Diablos AeroVision (HACK-020)

Plataforma: **Streamlit Community Cloud** (despliega desde `main` de GitHub). Respaldo: Hugging Face Spaces.
Repo: https://github.com/JaavRJ/hackathon_isac2026_diablosrojos

## Que viaja al despliegue
- Codigo (`dashboard/`, `src/`), `models/*.joblib` (4 archivos) y `reports/**/*.csv`. Tamano total versionado: ~12 MB (`.git` ~5.7 MB).
- Nunca `data/`, parquet, videos ni predicciones pitch-by-pitch (AGENTS.md 3.2): `data/` y `*.parquet` estan en `.gitignore`; `git ls-files` no contiene ninguno.

## Archivos de configuracion
| Archivo | Para que |
|---|---|
| `requirements.txt` | Solo lo que la app desplegada importa, con versiones exactas (las de `.venv` de hoy). |
| `requirements-dev.txt` | Analisis/tests (`-r requirements.txt` + pytest, shap, xgboost, catboost, econml, matplotlib, seaborn, kaleido, fastapi, uvicorn). NO lo lee Streamlit Cloud. |
| `packages.txt` | apt: `libgl1`, `libglib2.0-0`, `libportaudio2` (mediapipe arrastra `opencv-contrib-python` no headless y `sounddevice`). |
| `.streamlit/config.toml` | Tema claro; sin cambios. |

Decision: se sacaron de `requirements.txt` xgboost, catboost, econml, shap, kaleido, seaborn, matplotlib, fastapi, uvicorn y pytest porque ningun import de `dashboard/` ni `src/` (ni los joblib, que cargan sin ellos) los usa. Hacen la instalacion mucho mas pesada y lenta.

## Python
Streamlit Community Cloud **no lee `runtime.txt` ni `.python-version`**: la version se elige en la UI (Advanced settings al desplegar, o Settings de la app). Elegir **Python 3.12** (el `.venv` local usa 3.12.1). mediapipe 1.1.1 es un wheel `py3-none-manylinux_2_28_x86_64`, valido para 3.12.

## Pasos (los hace el usuario)
1. Haz commit y push a `main` de: `requirements.txt`, `requirements-dev.txt`, `packages.txt`, `docs/DEPLOY.md` (y `models/*.joblib`, ya versionados).
2. Entra a https://share.streamlit.io con tu cuenta de GitHub (dale acceso al repo si lo pide).
3. **Create app** > "Deploy a public app from GitHub".
4. Repository: `JaavRJ/hackathon_isac2026_diablosrojos` · Branch: `main` · Main file path: `dashboard/app.py`.
5. **Advanced settings** > Python version: **3.12**. No hacen falta secrets.
6. (Opcional) elige un subdominio en App URL. Pulsa **Deploy**.

## Primer arranque
- Instalacion de dependencias: varios minutos (mediapipe ~36 MB, opencv, lightgbm, scikit-learn). Los logs aparecen en el panel "Manage app".
- Despues, carga de la app ~3 s en local (import + joblib + CSV); en frio en la nube debe quedar < 60 s. Las 5 pestanas: Simulador, Mapa de movimiento, Leaderboard, Sculptor, VeloVision.
- Si la app se duerme por inactividad, el primer visitante la despierta (unos segundos).

## Actualizar
`git push` a `main`: Streamlit Cloud redespliega solo. Cambios en `requirements.txt` reinstalan dependencias. "Reboot app" en Manage app si algo queda en cache.

## Plan B: Hugging Face Spaces
1. Crea un Space con SDK **Streamlit** (o Docker si MediaPipe da problemas) y sube/conecta el repo.
2. En el `README.md` del Space (front-matter): `sdk: streamlit`, `app_file: dashboard/app.py`, `python_version: 3.12`.
3. `requirements.txt` se usa igual; `packages.txt` tambien es soportado por Spaces.

## Prueba de humo (completar tras desplegar)
- URL publica: _pendiente_
- Tiempo de carga en frio: _pendiente_
- 5 pestanas sin error: _pendiente_
- VeloVision procesa clip de 5 s: _pendiente_

## Verificacion previa (2026-10-21, venv limpio con solo `requirements.txt`, Windows/Python 3.12.1)
- Instalacion OK, incluido mediapipe 1.1.1 y opencv-python-headless 5.0.0.93.
- Los 4 `models/*.joblib` cargan sin warnings de version (sklearn 1.9.1, lightgbm 4.7.0).
- `streamlit run dashboard/app.py --server.headless true --server.port 8504`: HTTP 200, `/_stcore/health` = ok.
- `AppTest` de `dashboard/app.py`: 0 excepciones, 0 errores.
- `tests/test_dashboard_smoke.py`: 16 passed (carga en frio 2.69 s).
- No se pudo probar Linux localmente; los wheels de Linux x86_64 se verificaron en PyPI.

## Riesgos
- mediapipe 1.x ya no tiene `mp.solutions` (solo `mediapipe.tasks`): VeloVision debe usar la API Tasks y un archivo de modelo `.task` versionado en el repo.
- mediapipe instala `opencv-contrib-python` junto a `opencv-python-headless`; ambos son 5.0.0.93, pero si hay error `libGL.so.1` en Linux, confirmar que `packages.txt` se aplico.
- numpy 2.5 / pandas 3.0 / scikit-learn 1.9 son versiones recientes: si Streamlit Cloud no resolviera alguna, bajar todas en bloque a las probadas en `.venv`.
- Limites de RAM (~2.7 GB) y CPU en Community Cloud: clips de video largos pueden agotarlos; limitar tamano/duracion del clip.

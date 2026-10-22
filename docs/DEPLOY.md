# Despliegue publico de Diablos AeroVision (HACK-020)

Plataforma: **Streamlit Community Cloud** (despliega desde `main` de GitHub). Respaldo: Hugging Face Spaces.
Repo (privado): https://github.com/JaavRJ/hackathon_isac2026_diablosrojos
**App pública: https://diablos-aerovision.streamlit.app** (publicada; ver "Prueba de humo" abajo).

## Que viaja al despliegue
- Codigo (`dashboard/`, `src/`), `models/*.joblib` (4 archivos) y `reports/**/*.csv`. Tamano total versionado: ~12 MB (`.git` ~5.7 MB).
- Nunca `data/`, parquet, videos ni predicciones pitch-by-pitch (AGENTS.md 3.2): `data/` y `*.parquet` estan en `.gitignore`; `git ls-files` no contiene ninguno.

## Archivos de configuracion
| Archivo | Para que |
|---|---|
| `requirements.txt` | Solo lo que la app desplegada importa, con versiones exactas (las de `.venv` de hoy). |
| `requirements-dev.txt` | Analisis/tests (`-r requirements.txt` + pytest, shap, xgboost, catboost, econml, matplotlib, seaborn, kaleido, fastapi, uvicorn). NO lo lee Streamlit Cloud. |
| `packages.txt` | apt: `libgl1`, `libglib2.0-0`, `libportaudio2` (mediapipe arrastra `opencv-contrib-python` no headless y `sounddevice`) y, añadidas después del primer despliegue (commit `cbc73bc`), `libegl1` y `libgles2`, que MediaPipe necesita en Linux. |
| `.streamlit/config.toml` | Tema claro; sin cambios. |

Decision: se sacaron de `requirements.txt` xgboost, catboost, econml, shap, kaleido, seaborn, matplotlib, fastapi, uvicorn y pytest porque ningun import de `dashboard/` ni `src/` (ni los joblib, que cargan sin ellos) los usa. Hacen la instalacion mucho mas pesada y lenta.

## Python
Streamlit Community Cloud **no lee `runtime.txt` ni `.python-version`**: la version se elige en la UI (Advanced settings al desplegar, o Settings de la app). Elegir **Python 3.12** (el `.venv` local usa 3.12.1). mediapipe 1.1.1 es un wheel `py3-none-manylinux_2_28_x86_64`, valido para 3.12.

## Pasos (los hace el usuario)
El repo es **privado** y la app es **pública**. Para que Streamlit Cloud lea un repo privado hay que autorizarlo; después se hace pública la app.
1. Haz commit y push a `main` de: `requirements.txt`, `requirements-dev.txt`, `packages.txt`, `docs/DEPLOY.md` (y `models/*.joblib`, ya versionados).
2. Entra a https://share.streamlit.io con tu cuenta de GitHub. Para repos privados: **Settings → Linked accounts** y autoriza el acceso a repositorios privados de GitHub.
3. **Create app** > "Deploy a public app from GitHub".
4. Repository: `JaavRJ/hackathon_isac2026_diablosrojos` · Branch: `main` · Main file path: `dashboard/app.py`.
5. **Advanced settings** > Python version: **3.12**. No hacen falta secrets.
6. Elige el subdominio en App URL (el usado es `diablos-aerovision`). Pulsa **Deploy**.
7. Con repo privado la app nace privada: **Share → "Make this app public"** para que cualquiera abra la URL.

## Primer arranque
- Instalacion de dependencias: varios minutos (mediapipe ~36 MB, opencv, lightgbm, scikit-learn). Los logs aparecen en el panel "Manage app".
- Despues, carga de la app ~3 s en local (import + joblib + CSV); en frio en la nube debe quedar < 60 s. Las 7 pestanas: Inicio, Simulador, Mapa de movimiento, Leaderboard, Sculptor, VeloVision y Modo anotacion.
- Si la app se duerme por inactividad, el primer visitante la despierta (unos segundos).

## Actualizar
`git push` a `main`: Streamlit Cloud redespliega solo. Cambios en `requirements.txt` reinstalan dependencias. **Tras cambiar `packages.txt`, hacer "Reboot app" en Manage app** para que se reinstalen las librerias de sistema. Tambien "Reboot app" si algo queda en cache.

## Plan B: Hugging Face Spaces
1. Crea un Space con SDK **Streamlit** (o Docker si MediaPipe da problemas) y sube/conecta el repo.
2. En el `README.md` del Space (front-matter): `sdk: streamlit`, `app_file: dashboard/app.py`, `python_version: 3.12`.
3. `requirements.txt` se usa igual; `packages.txt` tambien es soportado por Spaces.

## Prueba de humo
Estado a la fecha (22-oct-2026): **la app esta publicada** en https://diablos-aerovision.streamlit.app, con Python 3.12.
- URL publica: https://diablos-aerovision.streamlit.app (registrada).
- Tiempo de carga en frio: _sin medir en este repo_ (en local, la carga en frio de las pruebas es de unos 3 s; el criterio es < 60 s).
- 7 pestanas sin error: _pendiente de registrar por el usuario/Revisor_.
- VeloVision procesa un clip de 5 s en el servidor: _pendiente de registrar_. `cbc73bc` anadio `libegl1`/`libgles2` a `packages.txt` para MediaPipe en Linux; tras ese cambio hay que hacer Reboot y confirmar VeloVision en la nube.
- Limite conocido: el Modo anotacion escribe en un contenedor efimero y los clips de `data/vision_clips/` no existen en la nube; esa pestana se usa en local.
- HACK-020 sigue en `REVIEW` en `docs/TASKS.md` hasta registrar la prueba de humo (el estado lo cambia el Revisor o el usuario).

## Verificacion previa (2026-10-21, venv limpio con solo `requirements.txt`, Windows/Python 3.12.1)
- Instalacion OK, incluido mediapipe 1.1.1 y opencv-python-headless 5.0.0.93.
- Los 4 `models/*.joblib` cargan sin warnings de version (sklearn 1.9.1, lightgbm 4.7.0).
- `streamlit run dashboard/app.py --server.headless true --server.port 8504`: HTTP 200, `/_stcore/health` = ok.
- `AppTest` de `dashboard/app.py`: 0 excepciones, 0 errores.
- `tests/test_dashboard_smoke.py`: 16 passed (carga en frio 2.69 s).
- No se pudo probar Linux localmente; los wheels de Linux x86_64 se verificaron en PyPI.

## Riesgos
- mediapipe 1.x ya no tiene `mp.solutions` (solo `mediapipe.tasks`): VeloVision usa la API Tasks y el modelo `models/pose_landmarker_full.task`, que **ya esta versionado** en el repo.
- mediapipe instala `opencv-contrib-python` junto a `opencv-python-headless`; ambos son 5.0.0.93, pero si hay error `libGL.so.1` (o de EGL/GLES) en Linux, confirmar que `packages.txt` se aplico y hacer Reboot.
- numpy 2.5 / pandas 3.0 / scikit-learn 1.9 son versiones recientes: si Streamlit Cloud no resolviera alguna, bajar todas en bloque a las probadas en `.venv`.
- Limites de RAM (~2.7 GB) y CPU en Community Cloud: clips de video largos pueden agotarlos; limitar tamano/duracion del clip.

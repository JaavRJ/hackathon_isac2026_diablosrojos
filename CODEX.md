# CODEX.md

Lee primero **[AGENTS.md](AGENTS.md)** (contiene las reglas obligatorias del proyecto y el protocolo autónomo por rol), **[docs/TASKS.md](docs/TASKS.md)** (tablero de tareas) y **[docs/CONTRACT.md](docs/CONTRACT.md)** (contrato de columnas y firmas).

## Si entraste como Agente Revisor:
1. Revisa en **[docs/TASKS.md](docs/TASKS.md)** todas las tareas que estén en estado **`REVIEW`**.
2. Para cada una:
   - Inspecciona los archivos listados en `AllowedPaths` y verifica que cumplan `docs/CONTRACT.md` y la especificación de la tarea.
   - Corre en terminal su comando `Verify Command` (`python -m pytest -q ...`).
   - Audita cada punto de su `Reviewer Checklist`.
3. Actualiza `docs/TASKS.md`:
   - Si todo pasa al 100 %: cambia `REVIEW` $\to$ `APPROVED` (en la tabla global y en el encabezado de la tarea) y marca `[x]` en el `Reviewer Checklist`.
   - Si encuentras cualquier error o incumplimiento: cambia `REVIEW` $\to$ `CHANGES_REQUESTED` y anota `archivo:línea` con el problema exacto.

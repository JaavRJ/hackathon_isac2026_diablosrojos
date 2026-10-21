# Entregables escritos (HACK-018) en LaTeX

| Archivo | Entregable |
|---|---|
| `documento_tecnico.tex` | Entregable 5: documento técnico (6–10 págs.) |
| `perfil_recomendaciones.tex` | Entregables 2 y 3: perfil ideal, recomendaciones y 3 casos de entrevista |
| `preambulo.tex` | Estilo y macros compartidos (`\cf{clave}`, `\tabla{nombre}`) |
| `generated/` | **Generado.** Cifras y tablas leídas de `reports/`; no editar a mano |

## Regenerar cifras (si cambian los CSV de `reports/`)
```bash
python -m analysis.build_paper_numbers
```

## Compilar
Desde esta carpeta (`docs/latex/`), dos pasadas para que salgan las referencias:
```bash
pdflatex documento_tecnico.tex && pdflatex documento_tecnico.tex
pdflatex perfil_recomendaciones.tex && pdflatex perfil_recomendaciones.tex
```
En Overleaf: sube toda la carpeta `docs/latex/` (incluida `generated/`) y compila con pdfLaTeX.

`tests/test_final_deliverables.py` falla si un documento cita una cifra que no existe,
si quedan números escritos a mano o si `generated/` no coincide con `reports/`.

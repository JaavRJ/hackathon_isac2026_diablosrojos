# Entregables escritos (HACK-018) en LaTeX

| Archivo | Entregable |
|---|---|
| `documento_tecnico.tex` | **Documento principal y único PDF.** Cuerpo = Entregable 5 (6–10 págs.); al final, `\appendix` con el Anexo A (Entregables 2 y 3). El número de páginas del cuerpo se calcula con `\pageref{fin:cuerpo}` |
| `anexo_perfil.tex` | Cuerpo (sin preámbulo) del Anexo A: Entregables 2 y 3 (perfil ideal, recomendaciones, 3 casos). Se carga con `\input` desde `documento_tecnico.tex` |
| `preambulo.tex` | Estilo y macros compartidos (`\cf{clave}`, `\tabla{nombre}`) |
| `generated/` | **Generado.** Cifras y tablas leídas de `reports/`; no editar a mano |

## Regenerar cifras (si cambian los CSV de `reports/`)
```bash
python -m analysis.build_paper_numbers
```

## Compilar
Desde esta carpeta (`docs/latex/`), tres pasadas para que salgan las referencias y el `\pageref` del anexo:
```bash
pdflatex documento_tecnico.tex && pdflatex documento_tecnico.tex && pdflatex documento_tecnico.tex
```
El único entregable es `documento_tecnico.pdf`. Las etiquetas del anexo llevan prefijo `anx:` para no chocar con las del cuerpo.
En Overleaf: sube toda la carpeta `docs/latex/` (incluida `generated/`) y compila con pdfLaTeX.

`tests/test_final_deliverables.py` falla si un documento cita una cifra que no existe,
si quedan números escritos a mano o si `generated/` no coincide con `reports/`.

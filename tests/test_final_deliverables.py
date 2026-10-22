"""HACK-018: entregables escritos en LaTeX (docs/latex/).

Verifica que los documentos existen, que no quedan marcadores pendientes, que toda
cifra citada con ``\\cf{clave}`` y toda tabla ``\\tabla{nombre}`` existe en
``docs/latex/generated/``, y que esos archivos generados coinciden exactamente con lo
que produce hoy ``analysis/build_paper_numbers.py`` a partir de ``reports/``.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from analysis.build_paper_numbers import build

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "docs" / "latex"
GENERATED = LATEX / "generated"
DOCS = ["documento_tecnico.tex", "anexo_perfil.tex"]
PENDING = re.compile(r"\[(?:TODO|por verificar|ejemplo)\]", re.IGNORECASE)


def _defined_keys() -> set[str]:
    text = (GENERATED / "cifras.tex").read_text(encoding="utf-8")
    return set(re.findall(r"\\@namedef\{cf@([^}]+)\}", text))


@pytest.mark.parametrize("name", DOCS + ["preambulo.tex"])
def test_document_exists_and_has_no_pending_markers(name):
    path = LATEX / name
    assert path.is_file(), f"falta {path}"
    text = path.read_text(encoding="utf-8")
    assert not PENDING.search(text), f"marcador pendiente en {name}: {PENDING.search(text).group(0)}"


@pytest.mark.parametrize("name", DOCS)
def test_every_cited_number_and_table_is_generated(name):
    text = (LATEX / name).read_text(encoding="utf-8")
    keys = _defined_keys()
    missing = sorted(set(re.findall(r"\\cf\{([^}]+)\}", text)) - keys)
    assert not missing, f"{name} cita cifras sin generar: {missing}"
    tables = set(re.findall(r"\\tabla\{([^}]+)\}", text))
    assert tables, f"{name} no incluye ninguna tabla generada"
    absent = sorted(t for t in tables if not (GENERATED / f"{t}.tex").is_file())
    assert not absent, f"{name} incluye tablas inexistentes: {absent}"


def test_no_literal_numbers_outside_generated_macros():
    """Las cifras de resultados viven en generated/; en el texto solo quedan constantes de diseño."""
    allowed = {"0", "1", "2", "3", "5", "10", "17", "22", "24", "30", "100", "170", "2024", "2025", "2026",
               "1{,}000", "2{,}240", "0.80", "1.000", "0.01", "0.20", "95"}
    for name in DOCS:
        body = (LATEX / name).read_text(encoding="utf-8")
        body = re.sub(r"(?<!\\)%.*", "", body)
        body = re.sub(r"\\cf\{[^}]+\}|\\tabla\{[^}]+\}|\\(?:label|ref)\{[^}]+\}|\[[0-9.]+pt\]|[0-9.]+pt", "", body)
        found = {m for m in re.findall(r"(?<![\w.{])\d[\d{},.]*\d|(?<![\w.{])\d(?![\w}])", body)}
        extra = sorted(found - allowed)
        assert not extra, f"{name} contiene números escritos a mano: {extra}"


def _pdf_pages(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def test_single_pdf_with_annex():
    """Un solo PDF: cuerpo del documento tecnico (6-10 pags.) y anexo con los Entregables 2 y 3."""
    pdfs = sorted(p.name for p in LATEX.glob("*.pdf"))
    assert pdfs == ["documento_tecnico.pdf"], f"debe existir un solo PDF en docs/latex: {pdfs}"
    assert not (LATEX / "perfil_recomendaciones.tex").exists()
    main = (LATEX / "documento_tecnico.tex").read_text(encoding="utf-8")
    assert r"\appendix" in main and r"\input{anexo_perfil.tex}" in main
    assert r"\label{fin:cuerpo}" in main and r"\pageref{fin:cuerpo}" in main
    assert main.index(r"\label{fin:cuerpo}") < main.index(r"\appendix") < main.index(r"\input{anexo_perfil.tex}")
    assert "Entregables 2 y 3" in main
    aux = LATEX / "documento_tecnico.aux"
    if aux.is_file():  # el .aux es local (gitignored); si esta, valida el rango del cuerpo
        m = re.search(r"\\newlabel\{fin:cuerpo\}\{\{[^}]*\}\{(\d+)\}", aux.read_text(encoding="utf-8"))
        assert m, "falta la etiqueta fin:cuerpo en el .aux; recompila el documento"
        body_pages = int(m.group(1))
        assert 6 <= body_pages <= 10, f"el cuerpo ocupa {body_pages} paginas (limite 6-10)"
        assert _pdf_pages(LATEX / "documento_tecnico.pdf") > body_pages, "el anexo no aparece en el PDF"


def test_no_label_collisions_between_body_and_annex():
    labels = {}
    for name in DOCS:
        text = (LATEX / name).read_text(encoding="utf-8")
        for lab in re.findall(r"\\label\{([^}]+)\}", text):
            assert lab not in labels, f"etiqueta duplicada {lab!r} en {name} y {labels[lab]}"
            labels[lab] = name
    annex = (LATEX / "anexo_perfil.tex").read_text(encoding="utf-8")
    undefined = {r for r in re.findall(r"\\ref\{([^}]+)\}", annex)} - set(labels)
    assert not undefined, f"referencias sin etiqueta en el anexo: {sorted(undefined)}"


VV_KEYS = ["vv.n", "vv.icc", "vv.mae.ann", "vv.mae.ann.n", "vv.mae.det", "vv.mae.det.n", "vv.rel.ms",
           "vv.window"]


def test_velovision_macros_exist_and_are_used_not_hardcoded():
    assert set(VV_KEYS) <= _defined_keys()
    used = (LATEX / "documento_tecnico.tex").read_text(encoding="utf-8")
    for k in ("vv.n", "vv.icc", "vv.mae.ann", "vv.mae.det", "vv.rel.ms", "vv.window"):
        assert "\\cf{" + k + "}" in used, f"documento_tecnico.tex no usa cf{{{k}}}"
    summ = ROOT / "reports" / "validation" / "vision_validation_summary.csv"
    if summ.is_file():
        df = pd.read_csv(summ)
        body = re.sub(r"(?<!\\)%.*", "", used + (LATEX / "anexo_perfil.tex").read_text(encoding="utf-8"))
        for v in df["value"].dropna().abs():
            for nd in (1, 2, 3):
                lit = f"{v:.{nd}f}"
                if len(lit) >= 4:
                    assert not re.search(r"(?<![\d.])" + re.escape(lit) + r"(?!\d)", body), f"cifra de VeloVision escrita a mano: {lit}"


def test_generated_files_match_reports(tmp_path):
    fresh = build(out_dir=tmp_path)
    for fname, content in fresh.items():
        committed = GENERATED / fname
        assert committed.is_file(), f"falta {committed}; corre python -m analysis.build_paper_numbers"
        assert committed.read_text(encoding="utf-8") == content, (
            f"{fname} está desactualizado respecto de reports/; corre python -m analysis.build_paper_numbers")


def test_reports_tables_are_populated():
    csvs = sorted((ROOT / "reports").rglob("*.csv"))
    assert csvs
    for csv in csvs:
        assert len(pd.read_csv(csv)) > 0, f"{csv} está vacío"


def test_no_raw_data_tracked_by_git():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    banned = [f for f in out if f.endswith((".parquet", ".pkl")) or f.startswith("data/")]
    assert not banned, f"datos crudos rastreados por git: {banned}"
    for f in out:
        if f.startswith("reports/") and f.endswith(".csv"):
            cols = pd.read_csv(ROOT / f, nrows=0).columns
            assert "PitchUID" not in cols, f"{f} es pitch-by-pitch"

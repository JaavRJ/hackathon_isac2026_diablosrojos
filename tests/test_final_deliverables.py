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
DOCS = ["documento_tecnico.tex", "perfil_recomendaciones.tex"]
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

"""Plausibilidad de las medidas del video frente a la LMB (ajustes tras la primera prueba con video real)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.vision import prospect_card as PC


def _catalog(heights, slots, throws="Right"):
    rows = []
    for i, (h, s) in enumerate(zip(heights, slots)):
        rows.append({"pitcher_anon_id": f"p{i}_{throws}", "year": 2025, "pitch_id": "Four-Seam", "n_pitches": 100,
                     "PitcherThrows": throws, "arm_slot_deg": s, "RelHeight": h, "fb_velo": 92.0,
                     "stuff_plus_cdmx": 100.0, "delta_stuff_cdmx": 0.0})
    return pd.DataFrame(rows)


@pytest.fixture
def cat():
    rng = np.random.default_rng(3)
    right = _catalog(rng.uniform(5.0, 6.5, 400), rng.uniform(20, 60, 400))
    left = _catalog(rng.uniform(3.0, 4.0, 50), rng.uniform(5, 10, 50), "Left")     # otra mano: no debe contar
    return pd.concat([right, left], ignore_index=True)


def test_inside_has_percentile_and_text(cat):
    r = PC.plausibility_report(5.8, 1.0, 40.0, "Derecho", {"rel_height_ft": 0.4, "arm_slot_dataset_deg": 5.0}, cat)
    h = r["rel_height_ft"]
    assert h["status"] == "dentro" and 0 < h["percentile"] < 100
    assert "dentro del rango de la LMB" in h["text"] and "percentil" in h["text"]
    assert r["arm_slot_dataset_deg"]["status"] == "dentro"
    assert r["any_outside"] is False and r["outside"] == []


def test_outside_uses_catalog_numbers_and_same_hand_only(cat):
    r = PC.plausibility_report(7.8, 0.0, 40.0, "Right", {"rel_height_ft": 0.4}, cat)
    h = r["rel_height_ft"]
    right = cat[cat.PitcherThrows == "Right"].RelHeight
    assert h["status"] == "fuera" and r["any_outside"] and r["outside"] == ["rel_height_ft"]
    assert h["high"] == pytest.approx(np.percentile(right, 99.5))           # cifra del catálogo, misma mano
    assert f"{h['high']:.2f}" in h["text"]
    assert "por encima de cualquier lanzador LMB" in h["text"] and "revisa la estatura y la cámara" in h["text"]
    assert "mecánica atípica" in h["text"]


def test_sys_error_widens_the_range(cat):
    hi = float(np.percentile(cat[cat.PitcherThrows == "Right"].RelHeight, 99.5))
    v = hi + 0.3
    assert PC.plausibility_report(v, 0, 40, "R", {"rel_height_ft": 0.0}, cat)["rel_height_ft"]["status"] == "fuera"
    assert PC.plausibility_report(v, 0, 40, "R", {"rel_height_ft": 0.4}, cat)["rel_height_ft"]["status"] == "dentro"
    r = PC.plausibility_report(v, 0, 40, "R", {"rel_height_sys_err_ft": 0.4}, cat)     # alias de pose_slot
    assert r["rel_height_ft"]["status"] == "dentro"


def test_below_range_and_side_has_no_reference(cat):
    r = PC.plausibility_report(3.0, 1.0, 1.0, "R", {}, cat)
    assert r["rel_height_ft"]["status"] == "fuera" and "por debajo" in r["rel_height_ft"]["text"]
    assert r["arm_slot_dataset_deg"]["status"] == "fuera"
    side = r["rel_side_ft"]
    assert side["status"] == "sin referencia" and "rel_side_ft" not in r["outside"]


def test_real_catalog_flags_the_7_8_ft_case_and_not_a_normal_one():
    r = PC.plausibility_report(7.80, -1.97, 57.0, "Right", {"rel_height_ft": 0.4, "arm_slot_dataset_deg": 8.1})
    assert r["rel_height_ft"]["status"] == "fuera" and r["rel_height_ft"]["high"] < 7.3
    ok = PC.plausibility_report(5.9, 1.0, 40.0, "Right", {"rel_height_ft": 0.4, "arm_slot_dataset_deg": 8.1})
    assert not ok["any_outside"] and ok["rel_height_ft"]["status"] == "dentro"


def test_missing_measure_is_reported_not_crashing(cat):
    r = PC.plausibility_report(None, None, None, "R", {}, cat)
    assert r["rel_height_ft"]["status"] == "sin referencia" and not r["any_outside"]

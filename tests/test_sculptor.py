"""Pruebas de HACK-016: Buscador de Referencias Factibles (Pitch Sculptor)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sculptor.optimizer import COACHING_LABEL, PitchSculptor

ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = [
    ROOT / "src" / "sculptor" / "__init__.py",
    ROOT / "src" / "sculptor" / "optimizer.py",
    ROOT / "dashboard" / "sculptor_tab.py",
    ROOT / "tests" / "test_sculptor.py",
]
PARAMS = {"PitcherThrows": "Right", "arm_slot_deg": 45.0, "RelHeight": 5.8, "fb_velo": 93.0, "pitch_id": "Four-Seam"}


def _row(pid, year, pitch, n, throws, slot, height, velo, s_cdmx):
    return {
        "pitcher_anon_id": pid, "year": year, "pitch_id": pitch, "n_pitches": n,
        "PitcherThrows": throws, "arm_slot_deg": slot, "RelHeight": height, "fb_velo": velo,
        "stuff_plus_neutral": s_cdmx - 1.0, "stuff_plus_cdmx": s_cdmx, "delta_stuff_cdmx": 1.0,
        "ivb_neutral": 15.0, "ivb_cdmx": 12.0, "hb_norm_neutral": 8.0, "hb_norm_cdmx": 6.5,
        "rho_cdmx_used": 0.95,
    }


@pytest.fixture
def catalog_df():
    rows = []
    for i in range(8):  # pares factibles
        pid = f"pitcher_{i:03d}"
        rows.append(_row(pid, 2025, "Four-Seam", 120 + i, "Right", 44.0 + 0.5 * i, 5.7 + 0.02 * i, 92.0 + 0.2 * i, 95.0 + 2.0 * i))
        rows.append(_row(pid, 2025, "Slider", 80 + i, "Right", 44.0 + 0.5 * i, 5.7 + 0.02 * i, 92.0 + 0.2 * i, 100.0 + i * (1 if i % 2 else -1)))
    # distractores que deben quedar fuera
    rows.append(_row("pitcher_L", 2025, "Four-Seam", 200, "Left", 45.0, 5.8, 93.0, 150.0))
    rows.append(_row("pitcher_slot", 2025, "Four-Seam", 200, "Right", 60.0, 5.8, 93.0, 151.0))
    rows.append(_row("pitcher_few", 2025, "Four-Seam", 10, "Right", 45.0, 5.8, 93.0, 152.0))
    rows.append(_row("pitcher_fast", 2025, "Four-Seam", 200, "Right", 45.0, 5.8, 99.0, 153.0))
    rows.append(_row("pitcher_tall", 2025, "Four-Seam", 200, "Right", 45.0, 6.5, 93.0, 154.0))
    return pd.DataFrame(rows)


def _passes(df, p=PARAMS, min_pitches=50):
    return df[
        (df.PitcherThrows == p["PitcherThrows"])
        & ((df.arm_slot_deg - p["arm_slot_deg"]).abs() <= 6.0)
        & ((df.RelHeight - p["RelHeight"]).abs() <= 0.35)
        & ((df.fb_velo - p["fb_velo"]).abs() <= 2.0)
        & (df.n_pitches >= min_pitches)
    ]


def test_values_exist_in_catalog_and_pass_filters(catalog_df, tmp_path):
    path = tmp_path / "pitcher_arsenal_breakdown.csv"
    catalog_df.to_csv(path, index=False)
    res = PitchSculptor(path).find_feasible_references({**PARAMS, "user_stuff_plus_cdmx": 90.0})
    feas = _passes(catalog_df)
    fam = feas[feas.pitch_id == "Four-Seam"]

    assert res["status"] == "OK"
    assert res["n_peers"] == feas.pitcher_anon_id.nunique() == 8
    intra = res["intra_family"]
    s = fam.stuff_plus_cdmx
    assert intra["p25_stuff_plus_cdmx"] == pytest.approx(np.percentile(s, 25))
    assert intra["p75_stuff_plus_cdmx"] == pytest.approx(np.percentile(s, 75))
    best = intra["best_reference"]
    match = fam[(fam.pitcher_anon_id == best["pitcher_anon_id"]) & (fam.stuff_plus_cdmx == best["stuff_plus_cdmx"])]
    assert len(match) == 1 and best["stuff_plus_cdmx"] == s.max()
    assert res["delta_gain_cdmx"] == pytest.approx(s.max() - 90.0)

    for comp in res["comparables"]:
        row = fam[(fam.pitcher_anon_id == comp["pitcher_anon_id"]) & (fam.stuff_plus_cdmx == comp["stuff_plus_cdmx"])]
        assert len(row) == 1 and int(row.n_pitches.iloc[0]) == comp["n_pitches"]
    for opt in res["repertoire"]:
        g = feas[feas.pitch_id == opt["pitch_id"]]
        assert opt["pitch_id"] in set(catalog_df.pitch_id) and opt["pitch_id"] != "Four-Seam"
        assert opt["median_stuff_plus_cdmx"] == pytest.approx(g.stuff_plus_cdmx.median())
        assert opt["n_pitchers"] == g.pitcher_anon_id.nunique()
        assert opt["n_pitches"] == g.n_pitches.sum()
    used = {c["pitcher_anon_id"] for c in res["comparables"]} | {best["pitcher_anon_id"]}
    assert used.isdisjoint({"pitcher_L", "pitcher_slot", "pitcher_few", "pitcher_fast", "pitcher_tall"})


def test_output_is_complete(catalog_df):
    res = PitchSculptor(catalog_df).find_feasible_references(PARAMS)
    for key in ("status", "n_peers", "delta_gain_cdmx", "delta_gain_reason", "comparables",
                "coaching_guide_label", "repertoire", "intra_family"):
        assert key in res
    assert res["delta_gain_cdmx"] is None and res["delta_gain_reason"]
    assert 1 <= len(res["comparables"]) <= 3
    assert len({c["pitcher_anon_id"] for c in res["comparables"]}) == len(res["comparables"])
    sc = [c["stuff_plus_cdmx"] for c in res["comparables"]]
    assert sc == sorted(sc, reverse=True)
    assert res["coaching_guide_label"] == COACHING_LABEL
    assert COACHING_LABEL == "[Guía cualitativa de coaching — literatura biomecánica, no estimada del dataset]"
    assert "no es una ganancia causal garantizada" in res["delta_gain_note"]


def test_insufficient_peers_no_numbers(catalog_df):
    small = catalog_df[catalog_df.pitcher_anon_id.isin([f"pitcher_{i:03d}" for i in range(4)])]
    res = PitchSculptor(small).find_feasible_references(PARAMS)
    assert res["status"] == "INSUFFICIENT_PEERS"
    assert res["n_peers"] == 4 and res["reason"]
    assert res["intra_family"] is None and res["delta_gain_cdmx"] is None
    assert res["repertoire"] == [] and res["comparables"] == []
    assert res["coaching_guide_label"] == COACHING_LABEL


def test_missing_columns_raise_clear_error(catalog_df):
    with pytest.raises(ValueError, match="PitcherThrows.*fb_velo|fb_velo"):
        PitchSculptor(catalog_df.drop(columns=["PitcherThrows", "fb_velo"]))


def test_coaching_block_shows_label_with_text():
    from dashboard.sculptor_tab import TAB_TITLE, coaching_block_html, render_sculptor_tab

    assert TAB_TITLE == "Referencias Factibles (Pitch Sculptor)" and callable(render_sculptor_tab)
    html = coaching_block_html("texto de prueba")
    assert COACHING_LABEL in html and "texto de prueba" in html


def test_no_forbidden_word_in_task_files():
    word = "opti" + "miz"
    module = "opti" + "mizer"  # nombre de módulo/archivo exigido por la tarea; solo se tolera como identificador
    for f in TASK_FILES:
        text = f.read_text(encoding="utf-8").lower().replace(module, "")
        assert word not in text, f



# --------------------------------------------------------------------------
# Tolerancias opcionales (ajustes tras la primera prueba con video real)
# --------------------------------------------------------------------------
def test_default_tolerances_are_identical_to_before(catalog_df):
    sc = PitchSculptor(catalog_df)
    a = sc.find_feasible_references(PARAMS)
    b = sc.find_feasible_references(PARAMS, tolerances={"arm_slot_deg": 6.0, "RelHeight": 0.35, "fb_velo": 2.0})
    assert a["n_peers"] == b["n_peers"] == _passes(catalog_df).pitcher_anon_id.nunique()
    assert a["tolerances_used"] == {"arm_slot_deg": 6.0, "RelHeight": 0.35, "fb_velo": 2.0}
    assert a["tolerance_source"] == b["tolerance_source"] == "estándar"
    assert a["filters"]["arm_slot_tol_deg"] == 6.0 and a["filters"]["rel_height_tol_ft"] == 0.35
    assert a["filters"]["fb_velo_tol_mph"] == 2.0
    assert a["intra_family"] == b["intra_family"] and a["comparables"] == b["comparables"]


def test_widened_tolerances_find_more_peers_and_are_declared(catalog_df):
    sc = PitchSculptor(catalog_df)
    std = sc.find_feasible_references(PARAMS)
    wide = sc.find_feasible_references(PARAMS, tolerances={"RelHeight": 0.75})
    assert wide["n_peers"] == std["n_peers"] + 1
    assert wide["tolerance_source"] == "ampliadas por el error del video"
    assert wide["tolerances_used"]["RelHeight"] == 0.75 and wide["tolerances_used"]["arm_slot_deg"] == 6.0
    assert wide["filters"]["rel_height_tol_ft"] == 0.75
    assert wide["coaching_guide_label"] == COACHING_LABEL and "no es una ganancia causal" in wide["delta_gain_note"]
    assert sc.find_feasible_references(PARAMS, tolerances={"arm_slot_deg": 16.0})["n_peers"] == std["n_peers"] + 1


def test_bad_tolerances_are_rejected(catalog_df):
    sc = PitchSculptor(catalog_df)
    with pytest.raises(ValueError):
        sc.find_feasible_references(PARAMS, tolerances={"velocidad": 3.0})
    with pytest.raises(ValueError):
        sc.find_feasible_references(PARAMS, tolerances={"RelHeight": 0.0})


def test_insufficient_peers_message_names_the_responsible_filter(catalog_df):
    df = catalog_df.copy()
    df.loc[df.pitcher_anon_id.str.startswith("pitcher_0"), "RelHeight"] = 6.6
    res = PitchSculptor(df).find_feasible_references(PARAMS)
    assert res["status"] == "INSUFFICIENT_PEERS"
    counts = {c["filter"]: c["n_pitchers"] for c in res["filter_counts"]}
    assert counts["misma mano"] >= 8 and counts["arm slot ±6.0°"] >= 8
    assert counts["altura ±0.35 ft"] < 5 and res["n_peers"] == counts["≥ 50 pitcheos"]
    assert "altura ±0.35 ft" in res["reason"] and "dejó fuera" in res["reason"] and "no se relajan" in res["reason"]
    assert res["intra_family"] is None and res["delta_gain_cdmx"] is None
    wide = PitchSculptor(df).find_feasible_references(PARAMS, tolerances={"RelHeight": 1.0})
    assert wide["status"] == "OK" and wide["tolerance_source"] == "ampliadas por el error del video"
    still = PitchSculptor(df).find_feasible_references({**PARAMS, "fb_velo": 80.0}, tolerances={"RelHeight": 1.0})
    assert still["status"] == "INSUFFICIENT_PEERS" and "velocidad" in still["reason"]
    assert "ampliadas por el error del video" in still["reason"]

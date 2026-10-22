"""Humo y coherencia del dashboard (HACK-015).

Ejercita la logica pura de ``dashboard/ui_utils.py`` sin servidor Streamlit, mas una corrida
headless de ``dashboard/app.py`` con ``streamlit.testing``. Mide y reporta:

* carga en frio (proceso nuevo: importar + ``models/*.joblib`` + CSV) < 20 s;
* prediccion en caliente (simulacion + SHAP + figuras de la Vista 1) < 3 s.

Usa los modelos y CSV reales del repositorio (no el crudo de pitcheos).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from dashboard import ui_utils as ui  # noqa: E402

APP_PATH = ROOT / "dashboard" / "app.py"
UI_PATH = ROOT / "dashboard" / "ui_utils.py"
COLD_LIMIT_S = 20.0
WARM_LIMIT_S = 3.0
HACK014_COLS = ("PitcherThrows", "arm_slot_deg", "RelHeight", "fb_velo", "rho_cdmx_used")
CHANNEL_COLS = ("canal_stuff", "canal_parque_bip", "residuo_no_atribuido")

FASTBALL_HIGH_IVB = {"pitch_id": "Four-Seam", "rel_speed": 94.0, "spin_rate": 2450.0, "ivb": 19.0, "hb_norm": 9.0}


@pytest.fixture(scope="module")
def bundle():
    return ui.load_all()


# ----------------------------------------------------------------------------
# Criterio 2: sin rutas absolutas, sin datos crudos; solo models/*.joblib y reports/**/*.csv
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("path", [APP_PATH, UI_PATH])
def test_no_absolute_paths_or_raw_data(path):
    text = path.read_text(encoding="utf-8")
    assert ".parquet" not in text.lower() and "read_parquet" not in text
    assert "deliverables" not in text
    assert not re.search(r"[A-Za-z]:[\\/]", text), "ruta absoluta de Windows detectada"
    assert not re.search(r"(?<![\w.])/(Users|home|mnt|var|opt)/", text), "ruta absoluta POSIX detectada"
    assert "Downloads" not in text


def test_paths_are_repo_relative():
    assert ui.PROJECT_ROOT == ROOT
    assert ui.MODELS_DIR == ROOT / "models" and ui.REPORTS_DIR == ROOT / "reports"
    for rel in ui.REPORT_FILES.values():
        assert not rel.is_absolute() and rel.suffix == ".csv"
    assert all(f.endswith(".joblib") for f in ui.MODEL_FILES.values())


def test_app_uses_streamlit_caches():
    text = APP_PATH.read_text(encoding="utf-8")
    assert "st.cache_resource" in text and "st.cache_data" in text
    assert "joblib.load" not in text and "read_csv(" not in text, "la carga vive en ui_utils y se cachea en app.py"


# ----------------------------------------------------------------------------
# Criterio 3: coherencia numerica
# ----------------------------------------------------------------------------
def test_delta_equals_cdmx_minus_neutral(bundle):
    res = ui.simulate_pitch(bundle["models"], FASTBALL_HIGH_IVB, bundle["meta"])
    assert res["delta_stuff"] == pytest.approx(res["cdmx"]["stuff"] - res["neutral"]["stuff"], abs=1e-9)
    direct = bundle["models"]["stuff_model"].predict_scores(res["frame"], rho_cdmx=res["rho_cdmx"])
    assert res["delta_stuff"] == pytest.approx(float(direct["delta_stuff_cdmx"].iloc[0]), abs=1e-9)
    assert res["delta_pitching"] == pytest.approx(res["cdmx"]["pitching"] - res["neutral"]["pitching"], abs=1e-9)
    for flavor in ("neutral", "cdmx"):
        assert all(np.isfinite(res[flavor][k]) for k in ("stuff", "location", "pitching", "ivb", "hb"))
    assert res["neutral"]["rho"] == 1.0 and 0.5 < res["rho_cdmx"] < 1.0


def test_cdmx_movement_is_smaller_for_high_ivb_fastball(bundle):
    res = ui.simulate_pitch(bundle["models"], FASTBALL_HIGH_IVB, bundle["meta"])
    assert res["neutral"]["ivb"] == pytest.approx(FASTBALL_HIGH_IVB["ivb"], abs=1e-9)
    assert abs(res["cdmx"]["ivb"]) < abs(res["neutral"]["ivb"])
    assert abs(res["cdmx"]["hb"]) < abs(res["neutral"]["hb"])


def test_shap_is_additive_and_matches_score(bundle):
    res = ui.simulate_pitch(bundle["models"], FASTBALL_HIGH_IVB, bundle["meta"])
    expl = ui.explain_pitch_shap(bundle["models"], res["frame"], res["rho_cdmx"])
    total = expl["base_value"] + sum(expl["shap_values"].values())
    assert total == pytest.approx(expl["predicted_stuff_plus"], abs=1e-6)
    assert expl["predicted_stuff_plus"] == pytest.approx(res["cdmx"]["stuff"], abs=1e-6)
    table = ui.shap_table(expl, 8)
    assert len(table) == 8 and {"variable", "contribucion"} <= set(table.columns)


def test_channel_decomposition_sums_to_total(bundle):
    decomp = bundle["reports"]["decomposition"].set_index("channel")
    est = decomp["estimate"]
    assert est["stuff"] + est["park"] + est["unallocated"] == pytest.approx(est["total"], rel=0, abs=1e-10)
    share = decomp["share_of_total"]
    assert share["stuff"] + share["park"] + share["unallocated"] == pytest.approx(1.0, abs=1e-9)
    assert set(decomp["decomposition_type"]) == {"three_term"}


def test_leaderboard_matches_csv(bundle):
    raw = pd.read_csv(ROOT / "reports" / "leaderboards" / "pitcher_season_leaderboard.csv")
    lb = bundle["reports"]["leaderboard"]
    assert len(lb) == len(raw)
    merged = raw.merge(lb, on=["pitcher_anon_id", "year"], suffixes=("_raw", ""))
    for col in ("stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx", "pitching_plus_cdmx"):
        np.testing.assert_allclose(merged[col + "_raw"], merged[col], rtol=0, atol=0, equal_nan=True)
    view = ui.filter_leaderboard(lb, years=[2025], min_pitches=100, sort_by="stuff_plus_cdmx")
    assert len(view) > 0 and (view["year"] == 2025).all() and (view["n_pitches"] >= 100).all()
    assert view["stuff_plus_cdmx"].is_monotonic_decreasing
    first = view.iloc[0]
    csv_row = raw[(raw["pitcher_anon_id"] == first["pitcher_anon_id"]) & (raw["year"] == first["year"])].iloc[0]
    assert first["stuff_plus_cdmx"] == csv_row["stuff_plus_cdmx"]
    by_delta = ui.filter_leaderboard(lb, sort_by="delta_stuff_cdmx", ascending=True)
    assert by_delta["delta_stuff_cdmx"].dropna().is_monotonic_increasing
    card = ui.pitcher_card(lb, bundle["reports"]["arsenal"], first["pitcher_anon_id"], int(first["year"]))
    assert card["summary"]["stuff_plus_cdmx"] == csv_row["stuff_plus_cdmx"]


# ----------------------------------------------------------------------------
# Criterio 4: los tres terminos (con etiqueta propia del residuo) en el payload de la UI
# ----------------------------------------------------------------------------
def test_payload_has_three_labelled_terms(bundle):
    decomp = bundle["reports"]["decomposition"]
    payload = ui.decomposition_payload(decomp)
    assert "Canal Stuff" in payload["labels"]
    assert "Canal Parque (BIP)" in payload["labels"]
    assert "Residuo no atribuido (no se atribuye a ningún canal)" in payload["labels"]
    by = {r["channel"]: r for r in payload["rows"]}
    assert set(by) == {"total", "stuff", "park", "unallocated"}
    # El residuo conserva su etiqueta y no se suma al Canal Parque.
    assert by["park"]["label"] != by["unallocated"]["label"]
    assert by["park"]["estimate"] == decomp.set_index("channel").loc["park", "estimate"]
    assert all(np.isfinite(r["share_of_total"]) and r["ci_low"] <= r["estimate"] <= r["ci_high"] for r in payload["rows"])
    assert "poblacional" in payload["population_note"]
    if by["stuff"]["share_of_total"] < 0.25:  # la UI debe decirlo tal cual
        assert f"{100 * by['stuff']['share_of_total']:.1f}" in payload["stuff_note"]


# ----------------------------------------------------------------------------
# Tolerancia a columnas pendientes de HACK-014
# ----------------------------------------------------------------------------
def test_tolerates_missing_and_present_hack014_columns(bundle):
    reports, meta = bundle["reports"], bundle["meta"]
    arsenal, lb = reports["arsenal"], reports["leaderboard"]
    base = arsenal.iloc[0]
    slim = base.drop([c for c in HACK014_COLS if c in base.index])
    inputs = ui.inputs_from_arsenal_row(slim, lb.iloc[0].to_dict(), meta)
    assert np.isfinite(ui.simulate_pitch(bundle["models"], inputs, meta)["delta_stuff"])

    full = slim.copy()
    full["PitcherThrows"], full["arm_slot_deg"], full["RelHeight"] = "Left", 40.0, 5.9
    full["fb_velo"], full["rho_cdmx_used"] = 92.0, meta["rho_cdmx"]
    inputs2 = ui.inputs_from_arsenal_row(full, lb.iloc[0].to_dict(), meta)
    assert inputs2["pitcher_throws"] == "Left" and inputs2["rel_height"] == pytest.approx(5.9)
    assert np.isfinite(ui.simulate_pitch(bundle["models"], inputs2, meta)["delta_stuff"])

    lb_no = lb.drop(columns=[c for c in CHANNEL_COLS if c in lb.columns])
    row = lb_no.iloc[0]
    card = ui.pitcher_card(lb_no, arsenal, row["pitcher_anon_id"], int(row["year"]))
    assert all(v is None for v in card["channels"].values())
    lb_yes = lb_no.assign(canal_stuff=-0.001, canal_parque_bip=-0.002, residuo_no_atribuido=-0.003)
    card2 = ui.pitcher_card(lb_yes, arsenal, row["pitcher_anon_id"], int(row["year"]))
    assert card2["channels"]["residuo_no_atribuido"] == pytest.approx(-0.003)
    table = ui.display_table(lb_yes, ui.LEADERBOARD_COLUMNS + ["canal_stuff"])
    assert "Stuff+ CDMX" in table.columns and "Canal Stuff (xRV/pitcheo)" in table.columns


def test_movement_heatmap_matrix(bundle):
    grid = bundle["reports"]["movement_grid"]
    for label, key in ui.MAP_METRICS.items():
        mat = ui.movement_heatmap_matrix(grid, key, "Todos")
        assert not mat.empty and mat.shape[0] > 3, label
    one = ui.movement_heatmap_matrix(grid, "delta_stuff_cdmx", "Four-Seam")
    assert not one.empty
    assert ui.fig_movement_heatmap(one, "delta_stuff_cdmx", "t").data[0].zmid == 0


# ----------------------------------------------------------------------------
# Rendimiento: carga en frio (< 20 s) y prediccion en caliente (< 3 s)
# ----------------------------------------------------------------------------
def test_cold_load_under_20s(capsys):
    code = (
        "import sys, time\n"
        "t0 = time.perf_counter()\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from dashboard import ui_utils as u\n"
        "b = u.load_all()\n"
        "print('COLD_SECONDS=%.3f' % (time.perf_counter() - t0))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=120, check=True).stdout
    cold = float(re.search(r"COLD_SECONDS=([\d.]+)", out).group(1))
    with capsys.disabled():
        print(f"\n[HACK-015] carga en frio (proceso nuevo: importar + joblib + CSV): {cold:.2f} s (limite {COLD_LIMIT_S:.0f} s)")
    assert cold < COLD_LIMIT_S


def test_warm_prediction_under_3s(bundle, capsys):
    models, meta, reports = bundle["models"], bundle["meta"], bundle["reports"]
    ui.simulate_pitch(models, FASTBALL_HIGH_IVB, meta)  # calentamiento (modelos ya cargados)

    t0 = time.perf_counter()
    res = ui.simulate_pitch(models, {**FASTBALL_HIGH_IVB, "spin_rate": 2500.0}, meta)
    sim_s = time.perf_counter() - t0
    expl = ui.explain_pitch_shap(models, res["frame"], res["rho_cdmx"])
    ui.fig_movement_vector(res)
    ui.fig_shap(ui.shap_table(expl, 10))
    ui.fig_decomposition(ui.decomposition_payload(reports["decomposition"]))
    total_s = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[HACK-015] prediccion en caliente: {sim_s:.2f} s (solo simulacion) / "
              f"{total_s:.2f} s (simulacion + SHAP + figuras de la Vista 1); limite {WARM_LIMIT_S:.0f} s")
    assert sim_s < WARM_LIMIT_S and total_s < WARM_LIMIT_S


# ----------------------------------------------------------------------------
# Corrida headless de dashboard/app.py (sin servidor)
# ----------------------------------------------------------------------------
def test_app_runs_headless_in_spanish(bundle, capsys):
    from streamlit.testing.v1 import AppTest

    t0 = time.perf_counter()
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    run_s = time.perf_counter() - t0
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs][:4] == ["Inicio", "Simulador", "Mapa de movimiento", "Leaderboard"]

    shown = " ".join(" ".join(map(str, df.value["Término"])) for df in at.dataframe if "Término" in df.value.columns)
    for label in ("Canal Stuff", "Canal Parque (BIP)", "Residuo no atribuido (no se atribuye a ningún canal)"):
        assert label in shown
    html = " ".join(m.value for m in at.markdown)
    assert "nivel del mar" in html.lower() and "Estadio Alfredo Harp Helú" in html and "ΔStuff+" in html
    with capsys.disabled():
        print(f"\n[HACK-015] corrida headless completa de app.py (primera ejecucion): {run_s:.2f} s")

    t1 = time.perf_counter()
    at.run()
    warm_s = time.perf_counter() - t1
    assert not at.exception
    with capsys.disabled():
        print(f"[HACK-015] re-render completo de app.py con cache caliente: {warm_s:.2f} s")
    assert warm_s < WARM_LIMIT_S * 2


def test_ui_notes_on_origin_of_values():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    text = " ".join([c.value for c in at.caption] + [m.value for m in at.markdown])
    assert "física simple" in text
    for name in ("ZoneSpeed", "EffectiveVelo", "VertApprAngle", "SpinToBreakRatio", "arm_slot_deg",
                 "VertRelAngle", "HorzRelAngle_norm", "Inning", "medianas de entrenamiento"):
        assert name in text, name
    assert "mientras HACK-014" not in text
    assert "mientras HACK-014" not in APP_PATH.read_text(encoding="utf-8")
    assert "se deriva de su arm slot" in text and "sale del arsenal" in text
    assert "Altitud Extrema" in text

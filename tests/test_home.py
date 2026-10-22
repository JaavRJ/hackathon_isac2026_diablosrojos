"""Portada "Inicio" (HACK-021): render, primera pestaña, cifras = CSV, botones, sin cifras hardcodeadas, degradación."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")

from dashboard import home_tab as H  # noqa: E402
from dashboard import ui_utils as ui  # noqa: E402

APP = ROOT / "dashboard" / "app.py"
HOME_SRC = ROOT / "dashboard" / "home_tab.py"
REPORTS = ROOT / "reports"
TOOL_BUTTONS = {"home_go_sim": "Simulador", "home_go_map": "Mapa de movimiento", "home_go_lb": "Leaderboard",
                "home_go_scu": "Sculptor", "home_go_vv": "VeloVision"}


def _home_app():
    from dashboard import home_tab, theme, ui_utils as u
    theme.begin_run()
    theme.inject_global_css()
    reports = u.load_reports()
    home_tab.render_home_tab(reports, {"rho_cdmx": 0.8517956137362525})


def _degraded_app():
    import pandas as pd
    from dashboard import home_tab, ui_utils as u
    reports = u.load_reports()
    reports["holdout_metrics"] = pd.DataFrame()
    reports["stability"] = pd.DataFrame()
    home_tab.render_home_tab(reports, None)


def _empty_app():
    from dashboard import home_tab
    home_tab.render_home_tab({}, None)


def _run(fn):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_function(fn, default_timeout=120).run()


def test_home_renders_without_exceptions():
    at = _run(_home_app)
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Harp Helú" in html and "AeroVision te dice cuánto pierde y qué referencias reales seguir" in html
    assert "Diablos Rojos del México · Hackathon ISAC 2026 · Reto 2" in html


def test_home_is_first_and_default_tab():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.tabs[0].label == "Inicio"
    assert "home_hero_sim" in {b.key for b in at.button}
    assert at.button(key="home_hero_sim").proto.type == "primary"


def test_buttons_to_five_tools_and_hero():
    at = _run(_home_app)
    keys = {b.key for b in at.button}
    assert set(TOOL_BUTTONS) <= keys and {"home_hero_sim", "home_hero_vv"} <= keys
    for key, tab in TOOL_BUTTONS.items():
        at.button(key=key).click().run()
        req = at.session_state["_tab_jump"]
        assert req["label"] == tab and req["done"] is False


def test_figures_match_csv():
    rep = ui.load_reports()
    fig = H.home_figures(rep, {"rho_cdmx": 0.5})
    assert fig["rho"] == 0.5 and fig["rho_src"] == H.SRC_MODEL
    # ΔStuff+ ponderado, recalculado
    a = pd.read_csv(REPORTS / "leaderboards" / "pitcher_arsenal_breakdown.csv")
    fs = a[a["pitch_id"] == "Four-Seam"]
    exp = (fs["delta_stuff_cdmx"] * fs["n_pitches"]).sum() / fs["n_pitches"].sum()
    assert fig["delta"]["pitch_id"] == "Four-Seam" and fig["delta"]["delta"] == pytest.approx(exp)
    assert fig["delta"]["n"] == int(fs["n_pitches"].sum())
    # holdouts
    h = pd.read_csv(REPORTS / "validation" / "holdout_metrics.csv")
    h = h[h["target"] == "xrv_pitcher"]
    def red(name):
        p = h[(h["holdout"] == name) & (h["model"] == "Pitching+")]["rmse"].iloc[0]
        b = h[(h["holdout"] == name) & (h["model"] == "CountBaselineModel")]["rmse"].iloc[0]
        return 1 - p / b
    got = dict(fig["holdouts"])
    assert got["Season"] == pytest.approx(red("season")) and got["Park"] == pytest.approx(red("park"))
    folds = sorted(x for x in h["holdout"].unique() if x.startswith("pitcher_fold"))
    assert got["Pitcher"] == pytest.approx(sum(red(f) for f in folds) / len(folds))
    # estabilidad y alcance
    s = pd.read_csv(REPORTS / "validation" / "year_to_year_stability.csv").set_index("model")
    assert fig["stability"]["stuff_r"] == pytest.approx(s.loc["Stuff+", "pearson_r"])
    lb = pd.read_csv(REPORTS / "leaderboards" / "pitcher_season_leaderboard.csv")
    assert fig["scope"]["pitches"] == int(lb["n_pitches"].sum())
    assert fig["scope"]["pitchers"] == lb["pitcher_anon_id"].nunique()
    # las cifras mostradas salen de esos valores
    cards = H.anchor_cards(fig)
    assert len(cards) == 5 and all(c["src"] for c in cards)
    hold = [c for c in cards if c["src"] == H.SRC_HOLDOUT][0]
    assert "Pitching+" in hold["label"] and "Stuff+ solo empata" in hold["sub"]
    sv = []
    for name in ("season", "park"):
        b = h[(h["holdout"] == name) & (h["model"] == "CountBaselineModel")]["rmse"].iloc[0]
        sv.append(1 - h[(h["holdout"] == name) & (h["model"] == "Stuff+")]["rmse"].iloc[0] / b)
    assert H._signed(100 * max(sv)) in hold["sub"] or H._signed(100 * min(sv)) in hold["sub"]
    assert "en muestra" in [c for c in cards if "Estabilidad" in c["label"]][0]["label"]
    shown = " ".join(c["value"] for c in cards)
    assert f"{fig['rho']:.3f}" in shown and f"{fig['stability']['stuff_r']:.2f}" in shown


def test_no_hardcoded_result_numbers():
    src = HOME_SRC.read_text(encoding="utf-8")
    assert ".parquet" not in src.lower() and "read_parquet" not in src
    assert not re.search(r"(?<![\w])[A-Za-z]:[\/]", src)
    # ninguna cifra de resultado (calculada hoy) aparece escrita en el código
    fig = H.home_figures(ui.load_reports(), {"rho_cdmx": 0.8517956137362525})
    needles = {f"{fig['rho']:.3f}", f"{abs(fig['delta']['delta']):.2f}", f"{fig['stability']['stuff_r']:.2f}",
               f"{fig['stability']['pitching_r']:.2f}", str(fig["scope"]["pitches"]), f"{fig['scope']['pitches']:,}",
               str(fig["scope"]["pitchers"])}
    needles |= {f"{100 * v:.2f}" for _, v in fig["holdouts"]}
    for n in needles:
        assert n not in src, f"cifra hardcodeada: {n}"
    assert not re.search(r"\b\d{1,3}(,\d{3})+\b", re.sub(r"M[\dMCLQVHZ ,.\-]+|rgba?\([^)]*\)", "", src))  # sin SVG ni colores
    assert "optimiz" not in src.lower()
    assert "entrena con evidencia" not in src and "qué hacer" not in src


def test_degrades_without_csv():
    at = _run(_degraded_app)
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "pitcher_arsenal_breakdown.csv" in html and "holdout_metrics.csv" not in html
    at = _run(_empty_app)
    assert not at.exception, [e.value for e in at.exception]
    assert "home_hero_sim" in {b.key for b in at.button}
    fig = H.home_figures(None, None)
    assert H.anchor_cards(fig) == [] and H.physics_bars(None) == ""

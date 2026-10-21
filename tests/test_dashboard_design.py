"""Pruebas del rediseno visual (HACK-019, Fases 1 y 2): tokens, contraste, honestidad y render headless."""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from dashboard import components as C  # noqa: E402
from dashboard import theme as T  # noqa: E402
from dashboard import ui_utils as ui  # noqa: E402

APP = ROOT / "dashboard" / "app.py"
BRAND = "#CE1126"


def test_brand_red_not_in_plotly_template():
    T.register_plotly_template()
    import plotly.io as pio
    assert pio.templates.default == T.TEMPLATE_NAME
    assert BRAND.lower() not in pio.templates[T.TEMPLATE_NAME].to_json().lower()
    for name, val in T.DATA_COLORS.items():
        assert val.upper() != BRAND, name
    for scale in (T.DIVERGING_SCALE, T.SEQUENTIAL_SCALE):
        assert all(c.upper() != BRAND for _, c in scale)


def test_brand_red_not_in_data_color_functions():
    for fn in (ui.fig_movement_vector, ui.fig_shap, ui.fig_decomposition, ui.fig_movement_heatmap,
               ui.fig_league_plane, ui.fig_arsenal):
        src = inspect.getsource(fn).lower()
        assert "ce1126" not in src and "theme.brand" not in src and "brand" not in src, fn.__name__
    for fn in (C.band_bar, C.delta_badge, C.diagnostic_svg, C.decomposition_stack, C.kpi_card):
        assert "brand" not in inspect.getsource(fn).lower(), fn.__name__
    assert "ce1126" not in (ROOT / "dashboard" / "ui_utils.py").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("name,hex_", sorted(T.TEXT_TOKENS.items()))
def test_semantic_tokens_contrast_on_white(name, hex_):
    assert T.contrast_ratio(hex_, "#FFFFFF") >= 4.5, name


def test_tinted_pairs_and_graphic_contrast():
    for fg, bg in ((T.LOSS, T.LOSS_BG), (T.GAIN, T.GAIN_BG), (T.NEUTRAL, T.NEU_BG), (T.WARN_TEXT, T.WARN_BG)):
        assert T.contrast_ratio(fg, bg) >= 4.5
    assert T.contrast_ratio(T.LVL_SEA, "#FFFFFF") >= 3.0


def test_no_hardcoded_result_figures():
    files = [ROOT / "dashboard" / n for n in ("app.py", "components.py", "theme.py", "ui_utils.py", "sculptor_tab.py")]
    banned = [r"0\.852\b", r"0\.8518", r"[−-]0\.014", r"102\.1", r"100\.9", r"4\.50\s*%", r"4\.77", r"4\.14\s*%", r"0\.0141"]
    for f in files:
        text = f.read_text(encoding="utf-8")
        for pat in banned:
            hits = [m.group(0) for m in re.finditer(pat, text)]
            # 0.852 solo se tolera como respaldo documentado de getattr en ui_utils
            if f.name == "ui_utils.py" and pat == banned[0]:
                continue
            assert not hits, (f.name, pat)


def test_rho_comes_from_model_and_no_forbidden_word():
    assert "rho_extreme_median_" in (ROOT / "dashboard" / "ui_utils.py").read_text(encoding="utf-8")
    word = "opti" + "miz"
    for f in (ROOT / "dashboard").glob("*.py"):
        assert word not in f.read_text(encoding="utf-8").lower().replace("sculptor.opti" + "mizer", ""), f


def test_components_escape_html():
    assert "<script>" not in C.provenance_pill("<script>x</script>")
    assert "&lt;" in C.breadcrumb(["<b>"], "<i>")
    assert "▼" in C.delta_badge(-2.0) and "▲" in C.delta_badge(2.0) and "●" in C.delta_badge(0.0)


def test_bands_are_real_weighted_percentiles():
    b = ui.load_all()
    ars = b["reports"]["arsenal"]
    info = ui.league_bands(ars, "Four-Seam")
    rows = ars[(ars.pitch_id == "Four-Seam") & (ars.n_pitches >= 50)]
    assert info["n_rows"] == len(rows)
    lo, hi = info["bands"]["rel_speed"]
    assert rows.rel_speed_mean.min() <= lo < hi <= rows.rel_speed_mean.max()
    presets = ui.compute_presets(ars, b["meta"])
    assert {"alta", "sweeper"} <= set(presets)
    a = presets["alta"]["inputs"]
    assert a["pitch_id"] == "Four-Seam" and rows.ivb_neutral.min() <= a["ivb"] <= rows.ivb_neutral.max()
    anchors = ui.validation_anchors(b["reports"])
    assert len(anchors) == 4 and all(x["value"] for x in anchors)


def test_simulator_renders_headless_with_verdict_and_cards():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Veredicto en CDMX" in html and "dr-verdict" in html
    assert html.count("dr-big") >= 2 and "Diagnóstico visual" in html
    assert "Nivel del mar" in html and "Harp Helú" in html
    assert at.session_state["source_mode"] == "PRESET"
    for k in ("active_pitch_params", "loaded_pitcher_snapshot", "last_sim_results"):
        assert k in at.session_state
    # editar un slider pasa a MANUAL_EDIT y revertir regresa al preset
    at.slider(key="in_ivb").set_value(10.0).run()
    assert not at.exception and at.session_state["source_mode"] == "MANUAL_EDIT"
    at.button(key="btn_revert").click().run()
    assert at.session_state["source_mode"] == "PRESET"


# ----------------------------------------------------------------------------
# Pulido del Simulador + Fase 3
# ----------------------------------------------------------------------------
def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for sb in list(at.selectbox) + list(at.multiselect) + list(at.radio):
        parts += [str(o) for o in sb.options]
    return " ".join(parts)


def _visible_text(at) -> str:
    return re.sub(r"<style>.*?</style>", " ", _texts(at), flags=re.S)


def test_css_hero_numbers_dark_and_chips_not_brand():
    css = T.build_css()
    assert ".dr-card{color:var(--text)" in css
    for m in re.finditer(r"([^{}]*)\{([^{}]*)\}", css):
        sel, body = m.group(1), m.group(2)
        if any(k in sel for k in ("tag", "pills", "segmented", "stSliderThumb", "stRadioOption", "stCheckbox")):
            assert "var(--brand)" not in body and BRAND.lower() not in body.lower(), sel
    assert "st.multiselect(" not in (ROOT / "dashboard" / "leaderboard_tab.py").read_text(encoding="utf-8")


def test_no_default_streamlit_boxes_and_english_in_ui():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.info) == 0 and len(at.warning) == 0
    txt = _visible_text(at)
    assert "dr-notebox" in txt and "Descomposición poblacional de la liga" in txt
    assert "El Canal Stuff explica solo" in txt and "dr-callout" in txt
    assert "Choose options" not in txt
    assert not re.search(r"\b(Right|Left)\b", txt)
    assert "Zurdo" in txt and "Derecho" in txt


def test_sculptor_autocomputed_and_peers_panel():
    from streamlit.testing.v1 import AppTest
    from dashboard.sculptor_tab import COACHING_LABEL, DELTA_DISCLAIMER

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    html = _texts(at)
    # Sculptor se calcula solo con el pitcheo activo, sin pulsar nada
    assert "Pares reales factibles" in html and "tu pitcheo" in html and "mejor par" in html
    assert COACHING_LABEL in html and DELTA_DISCLAIMER in html and "Top 3 comparables" in html
    assert "Pares reales para este pitcheo" not in " ".join(e.label for e in at.expander)
    at.button(key="btn_peers").click().run()
    assert not at.exception
    assert "Pares reales para este pitcheo (Pitch Sculptor)" in " ".join(e.label for e in at.expander)
    assert "dr-range" in _texts(at)


def test_peers_panel_shows_insufficient_diagnostic():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.slider(key="in_arm_slot").set_value(80.0).run()
    at.button(key="btn_peers").click().run()
    assert not at.exception
    assert "INSUFFICIENT_PEERS" in _texts(at)


def test_leaderboard_components_and_send_to_simulator():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    html = _texts(at)
    assert "dr-tbl" in html and "± pendiente" in html and "Distribución de ΔStuff+ CDMX" in html
    assert "no estimable" in html or "Residuo no atribuido" in html
    at.button(key="btn_to_sim").click().run()
    assert not at.exception
    assert at.session_state["source_mode"] == "REAL_PITCHER"
    assert at.session_state["loaded_pitcher_snapshot"]["pitcher_anon_id"].startswith("pitcher_")
    assert at.session_state["active_pitch_params"]["pitch_id"] == at.session_state["loaded_pitcher_snapshot"]["pitch_id"]


def test_map_overlay_uses_last_sim_results():
    import plotly.graph_objects as go
    b = ui.load_all()
    grid = b["reports"]["movement_grid"]
    m = ui.movement_heatmap_matrix(grid, "delta_stuff_cdmx", "Todos")
    active = {"hb_sea": 9.0, "ivb_sea": 18.0, "hb_cdmx": 7.0, "ivb_cdmx": 15.0}
    fig = ui.fig_movement_heatmap(m, "delta_stuff_cdmx", "t", active)
    assert len(fig.data) == 3 and fig.data[0].zmid == 0 and isinstance(fig, go.Figure)
    assert BRAND.lower() not in fig.to_json().lower()


def test_root_font_size_is_original_and_zoom_rule_for_three_tabs():
    """Raiz original (Simulador, Mapa y Leaderboard sin cambios) y zoom 1.25 solo en Sculptor, VeloVision y anotacion."""
    assert "html:root{font-size:clamp(9px,min(0.96vw,1.55vh),20px) !important}" in T._BASE_CSS
    m = re.search(r"((?:\.st-key-zoom_\w+,?\s*)+)\{zoom:1\.25\}", T._BASE_CSS)
    assert m, "falta la regla zoom:1.25"
    keys = set(re.findall(r"st-key-(zoom_\w+)", m.group(1)))
    assert keys == {"zoom_sculptor", "zoom_velovision", "zoom_annotation"}
    src = APP.read_text(encoding="utf-8")
    for k in keys:
        assert f'st.container(key="{k}")' in src

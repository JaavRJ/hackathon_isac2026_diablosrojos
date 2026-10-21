"""Pruebas de la interfaz de VeloVision y del Modo anotación (HACK-017, 17-C). Sin videos reales ni MediaPipe."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard import annotation_tab as AN  # noqa: E402
from dashboard import velovision_tab as VT  # noqa: E402
from src.vision import pose_slot as PS  # noqa: E402
from src.vision import prospect_card as PC  # noqa: E402

W, H = 640, 480
ANK_Y, NOSE_Y = 0.9, 0.2
HEIGHT_FT = 6.0
FORBIDDEN = [r"mide (el )?spin", r"mide (el )?quiebre", r"medimos (spin|quiebre|velocidad)", r"pitcher_\d+",
             r"mide la velocidad", r"estimamos (el )?spin", r"emparej\w+ (con|a) (el )?pitcher"]


def _skeleton(throws="R", wrist=None, vis=1.0):
    k = np.full((33, 3), np.nan)
    k[:, 2] = vis
    k[PS.NOSE, :2] = (0.5, NOSE_Y)
    k[PS.R_SHOULDER, :2], k[PS.L_SHOULDER, :2] = (0.42, 0.30), (0.58, 0.30)
    k[PS.R_ELBOW, :2], k[PS.L_ELBOW, :2] = (0.38, 0.38), (0.62, 0.38)
    k[PS.R_WRIST, :2], k[PS.L_WRIST, :2] = (0.36, 0.46), (0.64, 0.46)
    k[PS.R_HIP, :2], k[PS.L_HIP, :2] = (0.45, 0.55), (0.55, 0.55)
    k[PS.R_ANKLE, :2], k[PS.L_ANKLE, :2] = (0.45, ANK_Y), (0.55, ANK_Y)
    for i in range(33):
        if np.isnan(k[i, 0]):
            k[i, :2] = (0.5, 0.5)
    if wrist is not None:
        k[PS.R_WRIST if throws == "R" else PS.L_WRIST, :2] = wrist
    return k


def _wrist(height_ft, rel_h, rel_s):
    """Posición de la muñeca derecha que da (rel_h, rel_s) con el origen actual de pose_slot (lanzador derecho)."""
    ft_px = PS.ANTHROPOMETRIC_NOSE_FRACTION * height_ft / ((ANK_Y - NOSE_Y) * H)
    offset = PS.ANKLE_TO_SOLE_FT + PS.MOUND_HEIGHT_FT + PS.HAND_TO_BALL_FT
    return (0.55 - rel_s / ft_px / W, ANK_Y - (rel_h - offset) / ft_px / H)


def _synthetic_seq(vis=1.0, n=40):
    kp = np.stack([_skeleton("R", _wrist(HEIGHT_FT, 6.4, 1.2), vis) for _ in range(n)])
    return PS.PoseSequence.from_keypoints(kp, fps=30.0, width=W, height=H)


def _app():
    from dashboard import theme
    from dashboard.velovision_tab import render_velovision_tab

    theme.begin_run()
    theme.inject_global_css()
    render_velovision_tab()


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return re.sub(r"<style>.*?</style>", " ", " ".join(parts), flags=re.S)


@pytest.fixture
def fake_clip(tmp_path, monkeypatch):
    """Una carpeta de clips con un archivo falso; la extracción de pose se inyecta en cada prueba."""
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip_demo.mp4").write_bytes(b"no-es-un-video")
    monkeypatch.setattr(VT, "CLIPS_DIR", clips)
    return clips


def _inject(monkeypatch, seq, valid=True):
    """Inyecta la pose. Los skeletons sintéticos no son un lanzamiento real, así que por defecto la detección de
    soltura se da por válida; con ``valid=None`` se usa el detector real (que dirá "sin soltura clara")."""
    monkeypatch.setattr(PS, "extract_pose_sequence", lambda path, max_seconds=6.0, target_width=640, **kw: seq)
    if valid is not None:
        real = PS.detect_release_frame
        monkeypatch.setattr(PS, "detect_release_frame", lambda *a, **k: {**real(*a, **k), "valid": bool(valid)})


def _open_clip(at):
    at.selectbox(key="vv_clip_pick").set_value("clip_demo.mp4").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


# ----------------------------------------------------------------------------
def test_empty_state_renders_without_video(monkeypatch, tmp_path):
    monkeypatch.setattr(VT, "CLIPS_DIR", tmp_path / "no-existe")
    at = AppTest.from_function(_app, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "dr-guides" in txt and "Cámara de frente" in txt and "Estatura conocida" in txt
    assert "dr-empty" in txt and "Sube un video para empezar" in txt
    assert PC.DISCLAIMER in txt                                  # badge permanente
    assert "se borra al terminar" in txt                         # privacidad
    assert "en espera" not in txt.lower()
    assert len(at.warning) == 0 and len(at.info) == 0


def test_card_and_literal_disclaimer_with_injected_pose(fake_clip, monkeypatch):
    _inject(monkeypatch, _synthetic_seq())
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    txt = _text(at)
    assert "dr-fstrip" in txt and "Altura de soltura" in txt and "Ángulo del brazo (biomecánico)" in txt
    assert "informativo" in txt and "aún no medido (pendiente de validación)" in txt
    assert "error sistemático supuesto" in txt                   # viene de pose_slot (rel_*_sys_err)
    assert "Ficha de prospecto" not in txt                       # primero se confirma la soltura
    at.button(key="vv_confirm").click().run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "Soltura confirmada" in txt and "Ficha de prospecto" in txt
    assert "dr-fams" in txt and "Ventana de búsqueda" in txt and "sistemático supuesto" in txt
    assert "error de medición aún no medido" in txt
    assert "window_deg" not in txt and "error_source" not in txt and "medido (error_source)" not in txt
    assert txt.count(PC.DISCLAIMER) >= 2                          # arriba y junto a la ficha
    assert "Buscar referencias factibles" in " ".join(b.label for b in at.button)
    claims = txt.replace(PC.DISCLAIMER, " ")                      # el disclaimer literal NIEGA medir spin/quiebre/velocidad
    for pat in FORBIDDEN:
        assert not re.search(pat, claims, flags=re.I), pat
    assert "#CE1126" not in txt.upper()                           # el rojo de marca no es dato


def test_low_quality_blocks_ranges_and_shows_issues(fake_clip, monkeypatch):
    _inject(monkeypatch, _synthetic_seq(vis=0.1))
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    txt = _text(at)
    m = PS.measure_release(_synthetic_seq(vis=0.1), 0, "Right", HEIGHT_FT)
    assert m["quality_score"] < PS.QUALITY_MIN and m["quality_issues"]
    assert "no se generan rangos" in txt and "dr-callout" in txt
    for issue in m["quality_issues"]:
        assert issue in txt
    assert "dr-fams" not in txt and "Ficha de prospecto" not in txt
    assert "vv_confirm" in {b.key for b in at.button}             # el botón existe pero no produce ficha
    at.button(key="vv_confirm").click().run()
    assert "dr-fams" not in _text(at) and not at.exception


def test_unreadable_video_is_a_message_not_a_traceback(fake_clip, monkeypatch):
    def boom(*a, **k):
        raise ValueError("No se pudo abrir el video")
    monkeypatch.setattr(PS, "extract_pose_sequence", boom)
    at = AppTest.from_function(_app, default_timeout=60).run()
    _open_clip(at)
    assert "No se pudo leer el video" in _text(at)

    def nomodel(*a, **k):
        raise PS.PoseModelUnavailable("falta el modelo")
    monkeypatch.setattr(PS, "extract_pose_sequence", nomodel)
    at = AppTest.from_function(_app, default_timeout=60).run()
    _open_clip(at)
    assert "no encontró el modelo de pose" in _text(at)


def test_sculptor_button_requires_velocity_then_loads_active_params(fake_clip, monkeypatch):
    _inject(monkeypatch, _synthetic_seq())
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    at.button(key="vv_confirm").click().run()
    btn = at.button(key="vv_to_sculptor")
    assert btn.disabled                                           # sin velocidad no se puede continuar
    assert "Falta la altura o la velocidad" in _text(at)
    at.checkbox(key="vv_use_radar").check().run()
    at.number_input(key="vv_velo").set_value(91.5).run()
    assert not at.exception
    btn = at.button(key="vv_to_sculptor")
    assert not btn.disabled
    btn.click().run()
    p = at.session_state["active_pitch_params"]
    assert p["PitcherThrows"] == "Right" and p["fb_velo"] == pytest.approx(91.5) and p["RelHeight"] is not None
    assert 0.0 <= p["arm_slot_deg"] <= 90.0
    assert "Efecto del radar" in _text(at)                       # antes/después con radar


def test_override_survives_simulator_rewrite_and_drops_when_it_changes():
    class S(dict):
        pass
    import streamlit as st
    base = {"pitch_id": "Slider", "arm_slot_deg": 40.0}
    new = {**base, "arm_slot_deg": 52.0, "fb_velo": 91.0}
    st.session_state.clear()
    st.session_state.update(vv_override={"base": base, "params": new}, active_pitch_params=dict(base),
                            last_sim_results={"stuff_cdmx": 100.0, "ivb_cdmx": 1.0})
    VT.apply_velovision_override()
    assert st.session_state["active_pitch_params"] == new
    assert "stuff_cdmx" not in st.session_state["last_sim_results"]
    st.session_state["active_pitch_params"] = {**base, "arm_slot_deg": 41.0}      # el usuario movió el Simulador
    VT.apply_velovision_override()
    assert st.session_state["active_pitch_params"]["arm_slot_deg"] == 41.0 and st.session_state["vv_override"] is None
    st.session_state.clear()


# ----------------------------------------------------------------------------
# Modo anotación
# ----------------------------------------------------------------------------
def test_annotation_csv_columns_and_replacement(tmp_path):
    path = tmp_path / "vision_manual_annotations.csv"
    pts = [(0.1 * i + 0.05, 0.1 * i + 0.1) for i in range(6)]
    rows = AN.build_rows("C001", 12, "A", pts, "Right", 6.0, "2026-10-21T00:00:00+00:00")
    df = AN.save_annotation(rows, path)
    assert list(df.columns) == ["clip_id", "frame_idx", "annotator", "landmark", "x", "y", "throws", "height_ft", "timestamp"]
    assert list(df["landmark"]) == ["nariz", "hombro", "codo", "muneca", "cadera", "tobillo_apoyo"]
    assert df["x"].between(0, 1).all() and df["y"].between(0, 1).all() and set(df["throws"]) == {"R"}
    AN.save_annotation(AN.build_rows("C001", 12, "B", pts, "Right", 6.0), path)
    AN.save_annotation(AN.build_rows("C001", 12, "A", pts, "Right", 6.0), path)       # reemplaza, no duplica
    back = pd.read_csv(path)
    assert len(back) == 12 and list(back.columns) == AN.COLUMNS
    prog = AN.progress_by_annotator(AN.load_annotations(path))
    assert prog["por_anotador"] == {"A": 1, "B": 1} and prog["ambos"] == 1 and prog["clips"] == 1
    assert AN.next_clip_code(AN.load_annotations(path)) == "C002"


def test_same_video_same_clip_code_across_sessions(fake_clip, monkeypatch, tmp_path):
    reg = tmp_path / "reg.csv"
    a = AN.register_clip("a" * 40, reg)
    assert a == "C001" and AN.register_clip("b" * 40, reg) == "C002"
    assert AN.register_clip("a" * 40, reg) == "C001"                       # mismo contenido, mismo código
    assert list(pd.read_csv(reg).columns) == ["clip_code", "file_hash"] and len(pd.read_csv(reg)) == 2
    _inject(monkeypatch, PS.PoseSequence(_synthetic_seq().landmarks, 30.0, W, H,
                                         [np.full((270, 480, 3), 240, dtype=np.uint8)] * 40))
    codes = []
    for _ in range(2):                                                      # dos sesiones independientes
        at = AppTest.from_function(_ann_app, args=(str(tmp_path / "ann.csv"),), default_timeout=60).run()
        at.selectbox(key="an_clip_pick").set_value("clip_demo.mp4").run()
        assert not at.exception, [e.value for e in at.exception]
        codes.append([t.value for t in at.text_input][0])
    assert codes[0] == codes[1] and AN.CLIP_ID_RE.match(codes[0])


@pytest.mark.parametrize("bad", ["Juan Pérez", "", "a" * 30, "clip/1", "Juan", "Pedro_Lopez", "Maria", "C1", "c001", "C123456"])
def test_annotation_rejects_names_as_clip_id(bad):
    with pytest.raises(ValueError):
        AN.build_rows(bad, 0, "A", [(0.5, 0.5)] * 6, "R", 6.0)


def test_annotation_needs_all_points_and_valid_annotator():
    with pytest.raises(ValueError):
        AN.build_rows("C001", 0, "A", [(0.5, 0.5)] * 5, "R", 6.0)
    with pytest.raises(ValueError):
        AN.build_rows("C001", 0, "Z", [(0.5, 0.5)] * 6, "R", 6.0)


def test_draw_points_keeps_shape_and_avoids_brand_red():
    img = np.full((270, 480, 3), 255, dtype=np.uint8)
    out = AN.draw_points(img, [(0.5, 0.5), (0.2, 0.3)], width=720)
    assert out.shape == (405, 720, 3) and out.dtype == np.uint8
    assert not ((out[..., 0] == 0xCE) & (out[..., 1] == 0x11) & (out[..., 2] == 0x26)).any()


def _ann_app(path):
    from dashboard import theme
    from dashboard.annotation_tab import render_annotation_tab

    theme.begin_run()
    theme.inject_global_css()
    render_annotation_tab(path)


def test_annotation_tab_renders_empty_state(monkeypatch, tmp_path):
    monkeypatch.setattr(VT, "CLIPS_DIR", tmp_path / "no-existe")
    at = AppTest.from_function(_ann_app, args=(str(tmp_path / "ann.csv"),), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "Anotador A" in txt and "Anotador B" in txt and "Elige un clip para anotar" in txt
    assert "nunca un nombre" in txt or "nunca el nombre" in txt


def test_annotation_tab_with_clip_shows_prompt_for_first_point(fake_clip, monkeypatch, tmp_path):
    _inject(monkeypatch, PS.PoseSequence(_synthetic_seq().landmarks, 30.0, W, H,
                                         [np.full((270, 480, 3), 240, dtype=np.uint8)] * 40))
    at = AppTest.from_function(_ann_app, args=(str(tmp_path / "ann.csv"),), default_timeout=60).run()
    at.selectbox(key="an_clip_pick").set_value("clip_demo.mp4").run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "Haz clic en" in txt and "Nariz" in txt and "punto 1 de 6" in txt
    assert at.button(key="an_save").disabled and at.button(key="an_undo").disabled


# ----------------------------------------------------------------------------
# Reglas generales del módulo
# ----------------------------------------------------------------------------
def test_sources_have_no_brand_red_as_data_and_no_forbidden_claims():
    for f in ("velovision_tab.py", "annotation_tab.py"):
        src = (ROOT / "dashboard" / f).read_text(encoding="utf-8")
        assert "CE1126" not in src.upper()
        assert not re.search(r"[A-Za-z]:[\\/]", src)
        for pat in FORBIDDEN[:3]:
            assert not re.search(pat, src, flags=re.I)


def test_measure_card_always_shows_error_text():
    from dashboard import components as C
    html = C.measure_card("up", "Altura", 6.1234, "ft", VT.ERR_PENDING, decimals=2)
    assert "6.12" in html and VT.ERR_PENDING in html
    assert "—" in C.measure_card("up", "Altura", None, "ft", VT.ERR_PENDING)



# ----------------------------------------------------------------------------
# Ajustes tras la primera prueba con video real
# ----------------------------------------------------------------------------
REFERENCE_TEXT = "Referencia de lo que logran pitchers LMB con una mecánica parecida; no es el nivel de este lanzador."


def _seq_with(rel_h, rel_s=1.2, n=40):
    kp = np.stack([_skeleton("R", _wrist(HEIGHT_FT, rel_h, rel_s)) for _ in range(n)])
    return PS.PoseSequence.from_keypoints(kp, fps=30.0, width=W, height=H)


def test_card_shows_literal_reference_text_and_plausibility_inside(fake_clip, monkeypatch):
    _inject(monkeypatch, _synthetic_seq())
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    at.button(key="vv_confirm").click().run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert REFERENCE_TEXT in txt and PC.REFERENCE_NOTE == REFERENCE_TEXT
    assert "Plausibilidad frente a la LMB" in txt and "dentro del rango de la LMB" in txt
    assert not any("fuera del rango de la LMB" in c.label for c in at.checkbox)


def test_outside_range_needs_explicit_confirmation_and_sends_widened_tolerances(fake_clip, monkeypatch):
    _inject(monkeypatch, _seq_with(7.8))
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    at.button(key="vv_confirm").click().run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "por encima de cualquier lanzador LMB" in txt and "revisa la estatura y la cámara" in txt
    assert "Ficha de prospecto" in txt and REFERENCE_TEXT in txt
    at.checkbox(key="vv_use_radar").check().run()
    at.number_input(key="vv_velo").set_value(91.0).run()
    ack = [c for c in at.checkbox if "fuera del rango de la LMB" in c.label]
    assert len(ack) == 1 and ack[0].label == "Entiendo que la medida está fuera del rango de la LMB"
    assert at.button(key="vv_to_sculptor").disabled
    ack[0].check().run()
    assert not at.button(key="vv_to_sculptor").disabled
    at.button(key="vv_to_sculptor").click().run()
    tol = at.session_state["vv_tolerances"]
    assert tol["RelHeight"] == pytest.approx(max(0.35, PS.REL_HEIGHT_SYS_ERR_FT))
    assert tol["arm_slot_deg"] >= 6.0 and tol["fb_velo"] == 2.0


def test_blocking_issue_hides_ranges_even_with_passing_score(fake_clip, monkeypatch):
    kp = np.stack([_skeleton("R", _wrist(HEIGHT_FT, 6.4, 1.2)) for _ in range(40)])
    kp[:, PS.R_SHOULDER, 0], kp[:, PS.L_SHOULDER, 0] = 0.49, 0.51
    _inject(monkeypatch, PS.PoseSequence.from_keypoints(kp, fps=30.0, width=W, height=H))
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    txt = _text(at)
    assert "Bloquea:" in txt and "Vista de perfil" in txt and "aunque el puntaje pase" in txt
    assert "Ficha de prospecto" not in txt and "dr-fams" not in txt and "Plausibilidad" not in txt


def test_sculptor_tab_declares_widened_tolerances():
    def app():
        import streamlit as st
        from dashboard import theme
        from dashboard.sculptor_tab import render_sculptor_tab
        theme.begin_run()
        theme.inject_global_css()
        st.session_state["active_pitch_params"] = {"PitcherThrows": "Right", "arm_slot_deg": 40.0,
                                                   "RelHeight": 5.9, "fb_velo": 92.0, "pitch_id": "Four-Seam"}
        st.session_state["vv_override"] = {"base": {}, "params": {}}
        st.session_state["vv_tolerances"] = {"arm_slot_deg": 9.0, "RelHeight": 0.4, "fb_velo": 2.0}
        render_sculptor_tab()

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "Tolerancias ampliadas por el error del video" in txt and "±0.40 ft altura" in txt and "±9.0° arm slot" in txt


NO_RELEASE = "No se detectó una soltura clara en este tramo; probablemente es la preparación. Mueve el segundo de inicio o elige el frame a mano."


def test_no_clear_release_shows_callout_and_requires_checkbox(fake_clip, monkeypatch):
    kp = np.stack([_skeleton("R", _wrist(HEIGHT_FT, 6.4, 1.2)) for _ in range(40)])      # sin barrido: preparación
    _inject(monkeypatch, PS.PoseSequence.from_keypoints(kp, fps=30.0, width=W, height=H), valid=None)
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    assert NO_RELEASE in _text(at)
    at.button(key="vv_confirm").click().run()
    txt = _text(at)
    assert "Ficha de prospecto" not in txt and "dr-fams" not in txt           # sin la casilla no hay ficha
    ack = [c for c in at.checkbox if "no se detectó una soltura clara" in c.label]
    assert len(ack) == 1
    ack[0].check().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Ficha de prospecto" in _text(at)


def test_valid_release_has_no_callout_or_checkbox(fake_clip, monkeypatch):
    _inject(monkeypatch, _synthetic_seq())
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    assert NO_RELEASE not in _text(at)
    assert not [c for c in at.checkbox if "no se detectó una soltura clara" in c.label]


def test_start_second_slider_is_part_of_the_cache_key_and_shows_window(fake_clip, monkeypatch):
    calls = []

    def fake_extract(path, max_seconds=6.0, target_width=640, start_seconds=0.0, **kw):
        calls.append(float(start_seconds))
        return _synthetic_seq()

    monkeypatch.setattr(PS, "extract_pose_sequence", fake_extract)
    real = PS.detect_release_frame
    monkeypatch.setattr(PS, "detect_release_frame", lambda *a, **k: {**real(*a, **k), "valid": True})
    monkeypatch.setattr(VT, "video_seconds", lambda data, digest: 16.0)
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    assert "Se analizan" in _text(at) and "de un video de 16.0 s" in _text(at) and "de 0.0 a" in _text(at)
    sl = [x for x in at.slider if x.label == "Segundo de inicio"]
    assert len(sl) == 1 and sl[0].max == 15.0 and sl[0].min == 0.0
    sl[0].set_value(10.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls == [0.0, 10.0] and "de 10.0 a" in _text(at)
    [x for x in at.slider if x.label == "Segundo de inicio"][0].set_value(0.0).run()
    assert calls == [0.0, 10.0]                                                 # volver a 0 usa el caché


# ----------------------------------------------------------------------------
# Salto entre pestañas
# ----------------------------------------------------------------------------
def test_tab_jump_is_injected_once_per_click_with_toast_and_pill(monkeypatch):
    import streamlit.components.v1 as cv1
    injected = []
    monkeypatch.setattr(cv1, "html", lambda body, **kw: injected.append((body, kw)))

    def app():
        import streamlit as st
        from dashboard import components as C
        tabs = st.tabs(["Simulador", "Sculptor"])
        C.consume_tab_jump()
        with tabs[1]:
            C.context_pill("Sculptor")
            st.button("ir", key="go", on_click=C.request_tab_jump, args=("Sculptor", "VeloVision", "Parámetros cargados en Sculptor"))
        st.button("otro", key="other")

    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception and injected == []                       # sin clic no hay salto
    at.button(key="go").click().run()
    assert len(injected) == 1 and not at.exception
    body, kw = injected[0]
    assert kw.get("height") == 0 and '[role="tab"]' in body and '"Sculptor"' in body
    assert "textContent" in body and "setTimeout" in body and "scrollTo(0,0)" in body       # por texto, con reintentos, al inicio
    assert any("Parámetros cargados en Sculptor" in t.value for t in at.toast)
    assert "Cargado desde VeloVision" in _text(at)
    at.button(key="other").click().run()                              # un rerun cualquiera no vuelve a saltar
    at.run()
    assert len(injected) == 1
    at.button(key="go").click().run()                                 # un nuevo clic sí
    assert len(injected) == 2


def test_velovision_button_requests_jump_to_sculptor(fake_clip, monkeypatch):
    _inject(monkeypatch, _synthetic_seq())
    at = AppTest.from_function(_app, default_timeout=120).run()
    _open_clip(at)
    at.button(key="vv_confirm").click().run()
    at.checkbox(key="vv_use_radar").check().run()
    at.number_input(key="vv_velo").set_value(91.5).run()
    at.button(key="vv_to_sculptor").click().run()
    req = at.session_state["_tab_jump"]
    assert req["label"] == "Sculptor" and req["source"] == "VeloVision"
    assert at.session_state["_tab_context"]["Sculptor"] == "VeloVision"


def test_leaderboard_and_sculptor_load_buttons_use_the_jump_helper():
    import inspect
    from dashboard import leaderboard_tab, sculptor_tab
    for mod in (leaderboard_tab, sculptor_tab):
        assert "click_and_jump" in inspect.getsource(mod)
    assert "consume_tab_jump" in (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")

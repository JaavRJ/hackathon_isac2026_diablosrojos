"""Diablos AeroVision - Dashboard (HACK-015 base + rediseño visual HACK-019, Fases 1 y 2).

Lanzar desde la raiz del proyecto:

    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

Interfaz 100 % en espanol. Carga solo ``models/*.joblib`` y ``reports/**/*.csv`` (rutas
relativas a la raiz); la logica pura vive en ``dashboard/ui_utils.py``, el sistema de diseno en
``dashboard/theme.py`` y los componentes en ``dashboard/components.py``.

Vistas: (1) Simulador Nivel del Mar vs. Harp Helu, (2) Mapa de movimiento, (3) Leaderboard + tarjeta de
lanzador, (4) Pitch Sculptor (modulo externo opcional), (5) VeloVision, (6) Modo anotación.

Estado (``st.session_state``, segun docs/DISENO_UI_NIELSEN.md 3.3): ``active_pitch_params``,
``source_mode`` (PRESET | REAL_PITCHER | MANUAL_EDIT), ``loaded_pitcher_snapshot`` y ``last_sim_results``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from dashboard import components as C
from dashboard import theme
from dashboard import ui_utils as ui

from dashboard.annotation_tab import render_annotation_tab
from dashboard.leaderboard_tab import render_leaderboard
from dashboard.velovision_tab import apply_velovision_override, render_velovision_tab

try:  # HACK-016 entrega este modulo en paralelo
    from dashboard.sculptor_tab import compute_sculptor, render_sculptor_result, render_sculptor_tab
except ImportError:
    render_sculptor_tab = None

try:
    from src.sculptor.optimizer import PitchSculptor
except ImportError:  # pragma: no cover
    PitchSculptor = None

HAND_LABELS = {"Derecho": "Right", "Zurdo": "Left"}
HAND_NAMES = {v: k for k, v in HAND_LABELS.items()}
CDMX_NOTE = ("CDMX = mediana de la categoría Altitud Extrema de la LMB "
             "(agrupa varios parques de altura, no solo CDMX).")
LOADER_NOTE = (
    "Al cargar un pitcheo real: la altura de liberación (rel_height) sale del arsenal, el lado de "
    "liberación (rel_side) se deriva de su arm slot (arm_slot_deg) y solo el eje de giro usa la mediana "
    "de entrenamiento. Ajústalos a mano si los conoces."
)
COMPLETION_NOTE = (
    "Cómo se completa el pitcheo ingresado: se estiman con física simple ZoneSpeed, EffectiveVelo, "
    "VertApprAngle, SpinToBreakRatio y arm_slot_deg; usan medianas de entrenamiento VertRelAngle, "
    "HorzRelAngle_norm, Inning y los valores de túnel. No son mediciones."
)
NO_STUFF_NOTE = "No alteran Stuff+ (libre de proxies de ubicación)."
GAUGE_RANGE = 10.0  # medidor "vs promedio": +-10 puntos de Stuff+ (escala visual)
SOURCE_LABELS = {"PRESET": "Preset", "REAL_PITCHER": "Pitcher real", "MANUAL_EDIT": "Editado"}


# ----------------------------------------------------------------------------
# Carga en cache (una sola vez; nunca dentro de callbacks de widgets)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos...")
def get_models() -> dict:
    return ui.load_models()


@st.cache_data(show_spinner="Cargando reportes...")
def get_reports() -> dict:
    return ui.load_reports()


@st.cache_resource(show_spinner=False)
def get_meta() -> dict:
    return ui.build_meta(get_models(), get_reports())


@st.cache_data(show_spinner=False, max_entries=128)
def cached_simulation(inputs_key: tuple) -> dict:
    """Prediccion cacheada por entradas; los modelos y metadatos salen de los caches de arriba."""
    return ui.simulate_pitch(get_models(), dict(inputs_key), get_meta())


@st.cache_data(show_spinner=False, max_entries=64)
def cached_shap(inputs_key: tuple, flavor: str) -> dict:
    res = cached_simulation(inputs_key)
    rho = res["rho_cdmx"] if flavor == "cdmx" else ui.SEA_LEVEL_RHO
    return ui.explain_pitch_shap(get_models(), res["frame"], rho)


@st.cache_data(show_spinner=False)
def get_presets() -> dict:
    return ui.compute_presets(get_reports()["arsenal"], get_meta())


@st.cache_data(show_spinner=False)
def get_bands(pitch_id: str) -> dict:
    return ui.league_bands(get_reports()["arsenal"], pitch_id)


@st.cache_data(show_spinner=False)
def get_anchors() -> list:
    return ui.validation_anchors(get_reports())


@st.cache_resource(show_spinner=False)
def get_sculptor():
    if PitchSculptor is None:
        return None
    try:
        return PitchSculptor(get_reports()["arsenal"])
    except (ValueError, KeyError):
        return None


@st.cache_data(show_spinner=False, max_entries=128)
def cached_reference(params_key: tuple) -> dict:
    sculptor = get_sculptor()
    if sculptor is None:
        return {"status": "UNAVAILABLE", "n_peers": 0, "reason": "El catálogo de arsenal no está disponible."}
    return ui.reference_summary(sculptor.find_feasible_references(dict(params_key)))


# ----------------------------------------------------------------------------
# Estado y callbacks
# ----------------------------------------------------------------------------
def _inputs_key(inputs: dict) -> tuple:
    return tuple(sorted((k, round(v, 6) if isinstance(v, float) else v) for k, v in inputs.items()))


def _apply_inputs(values: dict) -> None:
    """Escribe un pitcheo (preset o pitcher real) en los widgets; lo no especificado vuelve al valor por defecto."""
    full = ui.clip_inputs({**ui.default_pitch_inputs(get_meta()), **values})
    for k, v in full.items():
        st.session_state[f"in_{k}"] = v
    st.session_state["hand_widget"] = HAND_NAMES[full["pitcher_throws"]]
    st.session_state["batter_widget"] = HAND_NAMES[full["batter_side"]]


def _on_preset(pid: str) -> None:
    preset = get_presets()[pid]
    _apply_inputs(preset["inputs"])
    st.session_state.update(source_mode="PRESET", active_preset=pid, loaded_pitcher_snapshot=None,
                            _base_inputs=dict(preset["inputs"]), _base_mode="PRESET", _base_label=preset["label"])


def _on_load_pitcher(values: dict, snapshot: dict) -> None:
    _apply_inputs(values)
    st.session_state["_loaded_flash"] = f"{snapshot['pitcher_anon_id']}-{snapshot['year']}"
    st.session_state.update(source_mode="REAL_PITCHER", active_preset=None, loaded_pitcher_snapshot=snapshot,
                            _base_inputs=dict(values), _base_mode="REAL_PITCHER",
                            _base_label=f"{snapshot['pitcher_anon_id']} · {snapshot['year']}")


def _mark_manual() -> None:
    st.session_state["source_mode"] = "MANUAL_EDIT"


def _on_revert() -> None:
    base = st.session_state.get("_base_inputs")
    if base:
        _apply_inputs(base)
        st.session_state["source_mode"] = st.session_state.get("_base_mode", "PRESET")


def _open_peers() -> None:
    st.session_state["peers_open"] = True


def _close_peers() -> None:
    st.session_state["peers_open"] = False


def _init_state() -> None:
    """Primera ejecucion: arranca con el preset ``alta`` (o con los valores por defecto si no hay catalogo)."""
    meta = get_meta()
    if "source_mode" in st.session_state:
        return
    presets = get_presets()
    st.session_state.setdefault("loaded_pitcher_snapshot", None)
    st.session_state.setdefault("last_sim_results", {})
    st.session_state.setdefault("active_pitch_params", {})
    if "alta" in presets:
        _on_preset("alta")
    else:
        _apply_inputs(ui.default_pitch_inputs(meta))
        st.session_state.update(source_mode="MANUAL_EDIT", active_preset=None, _base_inputs=None,
                                _base_mode="MANUAL_EDIT", _base_label="Pitcheo por defecto")


def _collect_inputs(defaults: dict) -> dict:
    inputs = {k: st.session_state[f"in_{k}"] for k in defaults}
    inputs["pitcher_throws"] = HAND_LABELS[st.session_state["hand_widget"]]
    inputs["batter_side"] = HAND_LABELS[st.session_state["batter_widget"]]
    inputs["rel_side"] = ui.rel_side_from(float(inputs["rel_height"]), float(inputs["arm_slot"]))
    return inputs


def _num(label: str, name: str, step: float, disabled: bool = False) -> None:
    lo, hi = ui.INPUT_BOUNDS[name]
    st.number_input(label, lo, hi, step=step, key=f"in_{name}", disabled=disabled, on_change=_mark_manual)


def _slider(label: str, name: str, step: float) -> None:
    lo, hi = ui.INPUT_BOUNDS[name]
    st.slider(label, lo, hi, step=step, key=f"in_{name}", on_change=_mark_manual)


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
def render_header(meta: dict) -> None:
    st.markdown(C.wordmark(), unsafe_allow_html=True)
    mode = st.session_state.get("source_mode", "PRESET")
    if mode == "MANUAL_EDIT":
        name = "Pitcheo editado"
    else:
        name = st.session_state.get("_base_label", "Pitcheo")
    hand = st.session_state.get("hand_widget", "Derecho")
    pitch = st.session_state.get("in_pitch_id", "Four-Seam")
    crumb = C.breadcrumb([name, hand, pitch], source=SOURCE_LABELS.get(mode, mode),
                         env=f"Harp Helú (ρ̂ {meta['rho_cdmx']:.3f})")
    c1, c2, c3, c4 = st.columns([4.2, 4.2, 1.5, 1.5], vertical_alignment="center")
    c1.markdown(crumb, unsafe_allow_html=True)
    c2.markdown(C.anchors_row(get_anchors()), unsafe_allow_html=True)
    c3.markdown(f'<div style="text-align:right">{C.pill("2026 fuera de muestra")}</div>', unsafe_allow_html=True)
    c4.button("↶ Revertir cambios", key="btn_revert", on_click=_on_revert, width="stretch",
              disabled=(mode != "MANUAL_EDIT" or not st.session_state.get("_base_inputs")))


# ----------------------------------------------------------------------------
# Vista 1: Simulador
# ----------------------------------------------------------------------------
def render_rail(reports: dict, meta: dict, defaults: dict) -> None:
    """Riel izquierdo: asistente con presets-icono, sliders con banda Zona LMB y contexto avanzado."""
    presets = get_presets()
    st.markdown(f'<div class="dr-rail-h">{C.icon("flask")}Asistente de Simulación</div>', unsafe_allow_html=True)
    if presets:
        st.markdown(C.preset_css([(pid, p["icon"]) for pid, p in presets.items()],
                                 st.session_state.get("active_preset") if st.session_state.get("source_mode") == "PRESET" else None),
                    unsafe_allow_html=True)
        with st.container(key="preset_row"):
            cols = st.columns(len(presets))
            for col, (pid, p) in zip(cols, presets.items()):
                col.button(p["label"], key=f"preset_{pid}", on_click=_on_preset, args=(pid,),
                           help=f"{p['description']} Fuente: pitcher_arsenal_breakdown.csv, {p['n_rows']} filas.",
                           width="stretch")

    k1, k2 = st.columns([3, 2])
    k1.selectbox("Tipo de pitcheo", ui.PITCH_TYPES, key="in_pitch_id", on_change=_mark_manual)
    k2.selectbox("Mano", list(HAND_LABELS), key="hand_widget", on_change=_mark_manual)

    info = get_bands(st.session_state["in_pitch_id"])
    note = ui.band_note(info)
    for key, ic, label, unit, step, dec, _col in ui.PRIMARY_SLIDERS:
        lo, hi = ui.INPUT_BOUNDS[key]
        val = float(st.session_state[f"in_{key}"])
        st.markdown(C.slider_header(ic, label, f"{val:.{dec}f}", unit), unsafe_allow_html=True)
        st.slider(label, lo, hi, step=step, key=f"in_{key}", label_visibility="collapsed", on_change=_mark_manual)
        st.markdown(C.band_bar(lo, hi, val, info["bands"].get(key), title=note), unsafe_allow_html=True)
    st.markdown('<div class="dr-zwrap">' + C.zone_legend(f"Zona LMB: p10–p90 real de la liga para {info['pitch_id']}")
                + f'<div class="note">{C.esc(note)}</div></div>', unsafe_allow_html=True)

    st.button("Buscar pares reales para este pitcheo", key="btn_peers", type="primary", width="stretch",
              on_click=_open_peers)

    with st.expander("Contexto del pitcheo y pitcher real", expanded=False):
        render_context_group(reports, meta, defaults)


def render_context_group(reports: dict, meta: dict, defaults: dict) -> None:
    lb, ars = reports["leaderboard"], reports["arsenal"]
    if len(lb) and len(ars):
        st.markdown("**Cargar desde un pitcher real del leaderboard**")
        c1, c2 = st.columns(2)
        pid = c1.selectbox("Pitcher", sorted(lb["pitcher_anon_id"].unique()), key="load_pid")
        years = sorted(lb.loc[lb["pitcher_anon_id"] == pid, "year"].unique(), reverse=True)
        year = c2.selectbox("Temporada", years, key="load_year")
        sub = ars[(ars["pitcher_anon_id"] == pid) & (ars["year"] == year)].sort_values("n_pitches", ascending=False)
        d1, d2 = st.columns([3, 2])
        pitch = d1.selectbox("Pitcheo", list(sub["pitch_id"]), key="load_pitch") if len(sub) else None
        if pitch is not None:
            row = sub[sub["pitch_id"] == pitch].iloc[0]
            summary = lb[(lb["pitcher_anon_id"] == pid) & (lb["year"] == year)].iloc[0].to_dict()
            values = ui.inputs_from_arsenal_row(row, summary, meta)
            snapshot = {"pitcher_anon_id": str(pid), "year": int(year), "pitch_id": str(pitch),
                        "n_pitches": int(row["n_pitches"]), "inputs": dict(values)}
            d2.button("Cargar", key="btn_load", on_click=_on_load_pitcher, args=(values, snapshot),
                      width="stretch")
        st.caption(LOADER_NOTE)

    st.markdown("**Punto de liberación**")
    r1, r2 = st.columns(2)
    with r1:
        _num("Altura de liberación (pies)", "rel_height", 0.05)
        _num("Extensión (pies)", "extension", 0.05)
    with r2:
        _num("Eje de giro (°, 0–360)", "spin_axis", 1.0)

    st.markdown("**Arsenal y túnel (afectan Stuff+)**")
    st.checkbox("Este pitcheo es la recta primaria", key="in_is_primary_fb", on_change=_mark_manual)
    off = bool(st.session_state["in_is_primary_fb"]) and st.session_state["in_pitch_id"] in ui.FASTBALL_IDS
    f1, f2, f3 = st.columns(3)
    with f1:
        _num("Velo recta primaria (mph)", "fb_velo", 0.5, off)
    with f2:
        _num("IVB recta primaria (in)", "fb_ivb", 0.5, off)
    with f3:
        _num("HB recta primaria (in)", "fb_hb_norm", 0.5, off)
    _slider("Similitud de túnel en la liberación", "tunnel_sim_release", 0.01)
    _slider("Uso del pitcheo en el repertorio", "pitch_usage_pct", 0.01)
    st.caption("Si es la recta primaria, las diferencias contra la recta valen 0; si no, se calculan contra los "
               "valores de la recta primaria.")

    st.markdown("**Contexto de Location+ / Pitching+**")
    st.caption(NO_STUFF_NOTE + " VertRelAngle, HorzRelAngle_norm y el túnel a 170 ms solo alimentan Location+ y Pitching+.")
    _slider("Similitud de túnel a 170 ms", "tunnel_sim_170ms", 0.01)
    g1, g2, g3 = st.columns(3)
    g1.selectbox("Bolas", [0, 1, 2, 3], key="in_balls", on_change=_mark_manual)
    g2.selectbox("Strikes", [0, 1, 2], key="in_strikes", on_change=_mark_manual)
    g3.selectbox("Outs", [0, 1, 2], key="in_outs", on_change=_mark_manual)
    st.selectbox("Bateador", list(HAND_LABELS), key="batter_widget", on_change=_mark_manual)
    _slider("Ubicación: lado (pies, vista del catcher; − = izquierda)", "plate_side", 0.05)
    _slider("Ubicación: altura (pies)", "plate_height", 0.05)
    geo = ui.zone_geometry(st.session_state["in_plate_side"], st.session_state["in_plate_height"])
    st.caption("Dentro de la zona de strike." if geo["in_strike_zone"] else "Fuera de la zona de strike.")
    st.caption(COMPLETION_NOTE)


def verdict_html(res: dict, inputs: dict) -> str:
    vp = ui.verdict_parts(res, inputs["pitch_id"])
    sentence = (
        f"Veredicto en CDMX: tu {C.esc(vp['pitch_name'])} {C.esc(vp['movement'])}; su Stuff+ "
        f"{C.esc(vp['stuff_verb'])} " + (
            f"de <b>{vp['stuff_sea']:.1f}</b> a <b>{vp['stuff_cdmx']:.1f}</b>." if vp["tone"] != "neu"
            else f"en <b>{vp['stuff_sea']:.1f}</b> (nivel del mar) y <b>{vp['stuff_cdmx']:.1f}</b> (CDMX)."))
    fb_velo = inputs["rel_speed"] if (inputs["is_primary_fb"] and inputs["pitch_id"] in ui.FASTBALL_IDS) else inputs["fb_velo"]
    params = (("PitcherThrows", inputs["pitcher_throws"]), ("arm_slot_deg", round(float(inputs["arm_slot"]), 1)),
              ("RelHeight", round(float(inputs["rel_height"]), 2)), ("fb_velo", round(float(fb_velo), 1)),
              ("pitch_id", inputs["pitch_id"]))
    ref = cached_reference(params)
    if ref["status"] == "OK":
        ref_html = (f"Referencia: {ref['n_peers']} pitchers reales con tu mecánica; su otra familia con mayor "
                    f"Stuff+ CDMX mediano es el <b>{C.esc(ref['pitch_id'])}</b> (<b>{ref['median']:.1f}</b>).")
    elif ref["status"] == "INSUFFICIENT_PEERS":
        ref_html = f"Referencia: INSUFFICIENT_PEERS. {C.esc(ref['reason'])}"
    else:
        ref_html = f"Referencia: {C.esc(ref.get('reason', 'no disponible'))}"
    return C.verdict_banner(sentence, vp["tone"], ref_html, "observado en pares reales, no causal")


def result_cards(res: dict, inputs: dict, meta: dict) -> tuple[str, str, str]:
    n, c = res["neutral"], res["cdmx"]
    vp = ui.verdict_parts(res, inputs["pitch_id"])
    pos = lambda s: 50.0 + 50.0 * max(-1.0, min(1.0, (s - 100.0) / GAUGE_RANGE))
    sea = C.kpi_card(
        "Nivel del mar", "home", "ρ̂ = 1.000", n["stuff"], "Stuff+", vs_avg=n["stuff"] - 100.0, avg_marker=pos(n["stuff"]),
        minis=[{"icon": "up", "label": "Subida (IVB)", "value": n["ivb"], "unit": "in"},
               {"icon": "right", "label": "Corrida (HB)", "value": n["hb"], "unit": "in"}],
        chips=[("Location+", C.fmt_num(n["location"])), ("Pitching+", C.fmt_num(n["pitching"]))])
    cdm = C.kpi_card(
        "Estadio Harp Helú", "mtn", f"ρ̂ = {res['rho_cdmx']:.4f}", c["stuff"], "Stuff+ CDMX", hero_delta=res["delta_stuff"],
        vs_avg=c["stuff"] - 100.0, avg_marker=pos(c["stuff"]),
        minis=[{"icon": "up", "label": "Subida (IVB)", "value": c["ivb"], "unit": "in", "delta": vp["d_ivb"]},
               {"icon": "right", "label": "Corrida (HB)", "value": c["hb"], "unit": "in", "delta": vp["d_hb"]}],
        chips=[("Location+", C.fmt_num(c["location"])), ("Pitching+", C.fmt_num(c["pitching"]))],
        note="Estadio Alfredo Harp Helú. " + CDMX_NOTE, title_attr="Estadio Alfredo Harp Helú")
    cap = (f"Cambio por aire delgado: {C.delta_badge(vp['d_ivb'], 1, 'in')} de subida y "
           f"{C.delta_badge(vp['d_hb'], 1, 'in')} de corrida. Retícula IVB × HB; ○ nivel del mar, ★ CDMX.")
    diag = C.diagnostic_card(n["hb"], n["ivb"], c["hb"], c["ivb"], cap, vp["tone"])
    return sea, cdm, diag


def render_analyst(res: dict, inputs: dict, reports: dict, meta: dict, ikey: tuple) -> None:
    """Modo Analista (cerrado por defecto): procedencia, mapa sobre la liga, SHAP y descomposicion de 3 terminos."""
    mode = st.session_state.get("source_mode", "PRESET")
    snap = st.session_state.get("loaded_pitcher_snapshot")
    if mode == "REAL_PITCHER" and snap:
        ctx = f"pitcher {snap['pitcher_anon_id']} · {snap['year']} · {snap['pitch_id']} · n = {snap['n_pitches']}"
    elif mode == "PRESET":
        p = get_presets().get(st.session_state.get("active_preset") or "", {})
        ctx = f"preset {p.get('label', '')} · percentiles de {p.get('n_rows', 0)} filas del catálogo"
    else:
        ctx = "pitcheo editado a mano"
    st.markdown(C.provenance_row([
        "modelo: StuffPlusModel (HACK-010)", "fuente: pitcher_arsenal_breakdown.csv",
        f"ρ̂ = {res['rho_cdmx']:.4f}", "2026 fuera de muestra", ctx]), unsafe_allow_html=True)

    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown(f'<div class="dr-h4">{C.icon("layers")}Movimiento sobre la liga</div>', unsafe_allow_html=True)
        st.plotly_chart(ui.fig_league_plane(reports["movement_grid"], inputs["pitch_id"], res), width="stretch")
        st.caption(f"Densidad: pitcheos {inputs['pitch_id']} por celda de 2 in (fuente: movement_map_grid.csv). "
                   "○ nivel del mar → ★ CDMX.")
    with m2:
        st.markdown(f'<div class="dr-h4">{C.icon("chart")}Qué empuja el Stuff+</div>', unsafe_allow_html=True)
        flavor = st.radio("Sabor a explicar", ["Harp Helú", "Nivel del Mar"], horizontal=True, key="shap_flavor")
        expl = cached_shap(ikey, "cdmx" if flavor == "Harp Helú" else "sea")
        st.plotly_chart(ui.fig_shap(ui.shap_table(expl, 10), title="", height=300), width="stretch")
        st.caption(
            f"Base {expl['base_value']:.1f} + suma de contribuciones = Stuff+ {expl['predicted_stuff_plus']:.1f}. "
            "▲ turquesa suma, ▼ coral resta (puntos de Stuff+)."
        )

    st.markdown(f'<div class="dr-h4">{C.icon("split")}Descomposición de altitud en 3 términos · población de la liga</div>',
                unsafe_allow_html=True)
    payload = ui.decomposition_payload(reports["decomposition"])
    st.markdown(C.note_box(payload["population_note"]), unsafe_allow_html=True)
    if payload["rows"]:
        rows = {r["channel"]: r for r in payload["rows"]}
        tot = rows.get("total")
        total_text = ""
        if tot:
            total_text = (f'<span class="mut note">Total observado (xRV/pitcheo)</span>'
                          f'<span class="num">{tot["estimate"]:+.5f}</span>'
                          f'<span class="mut note">IC95 [{tot["ci_low"]:+.5f}, {tot["ci_high"]:+.5f}]'
                          f'{" · n = " + str(tot["n_pitchers"]) + " pitchers pareados" if tot["n_pitchers"] else ""}</span>')
        st.markdown(C.provenance_row(["fuente: channel_decomposition.csv"]), unsafe_allow_html=True)
        st.markdown(C.decomposition_stack(payload["rows"], total_text), unsafe_allow_html=True)
        table = pd.DataFrame([{
            "Término": r["label"],
            "Estimación (xRV/pitcheo)": r["estimate"],
            "IC 95 % inferior": r["ci_low"],
            "IC 95 % superior": r["ci_high"],
            "Participación del total": ui.fmt_pct(r["share_of_total"]),
        } for r in payload["rows"]])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.5f")
                                    for c in ("Estimación (xRV/pitcheo)", "IC 95 % inferior", "IC 95 % superior")})
        if payload["stuff_note"]:
            st.markdown(C.callout(payload["stuff_note"]), unsafe_allow_html=True)
        st.caption("El residuo se muestra aparte: no se suma ni se renombra como Canal Parque.")
    else:
        st.warning("No se encontró reports/altitude/channel_decomposition.csv.")


def render_peers_panel(inputs: dict, res: dict, reports: dict, meta: dict) -> None:
    """Panel desplegable con el resultado REAL del Pitch Sculptor para el pitcheo activo."""
    fb_velo = inputs["rel_speed"] if (inputs["is_primary_fb"] and inputs["pitch_id"] in ui.FASTBALL_IDS) else inputs["fb_velo"]
    params = {"PitcherThrows": inputs["pitcher_throws"], "arm_slot_deg": float(inputs["arm_slot"]),
              "RelHeight": float(inputs["rel_height"]), "fb_velo": float(fb_velo), "pitch_id": inputs["pitch_id"]}
    score = float(res["cdmx"]["stuff"])
    sres = compute_sculptor({**params, "user_stuff_plus_cdmx": score})
    with st.expander("Pares reales para este pitcheo (Pitch Sculptor)", expanded=True):
        st.button("Cerrar panel", key="btn_close_peers", on_click=_close_peers)
        render_sculptor_result(sres, params, score, "peers", {"arsenal": reports["arsenal"], "meta": meta,
                                                              "load_cb": _on_load_pitcher},
                               st.session_state.get("loaded_pitcher_snapshot"))


def render_simulator(reports: dict, meta: dict) -> None:
    defaults = ui.default_pitch_inputs(meta)
    left, right = st.columns([0.24, 0.76], gap="medium")
    with left:
        with st.container(key="rail"):
            render_rail(reports, meta, defaults)

    inputs = _collect_inputs(defaults)
    ikey = _inputs_key(inputs)
    res = cached_simulation(ikey)
    # Estado compartido (docs/DISENO_UI_NIELSEN.md 3.3)
    st.session_state["active_pitch_params"] = ui.inputs_to_params(inputs)
    st.session_state["last_sim_results"] = {
        "stuff_sea": res["neutral"]["stuff"], "stuff_cdmx": res["cdmx"]["stuff"], "delta_stuff": res["delta_stuff"],
        "ivb_cdmx": res["cdmx"]["ivb"], "hb_cdmx": res["cdmx"]["hb"],
        "ivb_sea": res["neutral"]["ivb"], "hb_sea": res["neutral"]["hb"],
    }

    with right.container(key="simright"):
        st.markdown(verdict_html(res, inputs), unsafe_allow_html=True)
        sea, cdm, diag = result_cards(res, inputs, meta)
        with st.container(key="cards"):
            c1, c2, c3 = st.columns([1, 1, 1.1], gap="small")
            c1.markdown(sea, unsafe_allow_html=True)
            c2.markdown(cdm, unsafe_allow_html=True)
            c3.markdown(diag, unsafe_allow_html=True)
        st.caption(
            "“Harp Helú” usa ρ̂ = " f"{meta['rho_cdmx']:.3f}, la mediana de la categoría Altitud Extrema de la LMB "
            "(agrupa varios parques de altura, no solo CDMX). ΔStuff+ = Stuff+ Harp Helú − Stuff+ Nivel del Mar; "
            f"Location+ no depende de la densidad del aire (Δ = {ui.fmt_delta(res['delta_location'])}); "
            f"ΔPitching+ = {ui.fmt_delta(res['delta_pitching'])}."
        )
    if st.session_state.get("peers_open") and render_sculptor_tab is not None:
        render_peers_panel(inputs, res, reports, meta)
    with st.expander("Ver desglose estadístico avanzado y SHAP (Modo Analista)", expanded=False):
        render_analyst(res, inputs, reports, meta, ikey)


# ----------------------------------------------------------------------------
# Vista 2: Mapa de movimiento
# ----------------------------------------------------------------------------
def render_movement_map(reports: dict) -> None:
    st.subheader("Mapa de movimiento de la LMB")
    grid = reports["movement_grid"]
    if grid.empty:
        st.warning("No se encontró reports/leaderboards/movement_map_grid.csv.")
        return
    c1, c2 = st.columns([2, 1])
    metric_label = c1.radio("Valor a mostrar", list(ui.MAP_METRICS), horizontal=True, key="map_metric")
    pitch = c2.selectbox("Pitcheo", ["Todos"] + sorted(grid["pitch_id"].unique()), key="map_pitch")
    key = ui.MAP_METRICS[metric_label]
    matrix = ui.movement_heatmap_matrix(grid, key, pitch)
    if matrix.empty:
        st.info("Sin celdas para este filtro.")
        return
    active = st.session_state.get("last_sim_results") or None
    if active and not {"hb_sea", "ivb_sea", "hb_cdmx", "ivb_cdmx"} <= set(active):
        active = None
    st.plotly_chart(ui.fig_movement_heatmap(matrix, key, f"{metric_label} · {pitch}", active), width="stretch")
    if active:
        st.caption("○ → ★ azul marino: pitcheo activo del Simulador (nivel del mar → CDMX).")
    st.caption(
        "Cada celda promedia (ponderado por pitcheos) el score de los pitcheos observados en ese cuadrante de "
        "movimiento; se suprimen celdas con muy pocos pitcheos. En ΔStuff+, turquesa ▲ = gana en altitud y coral ▼ = pierde."
    )


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Diablos AeroVision", page_icon="⚾", layout="wide",
                       initial_sidebar_state="collapsed")
    theme.begin_run()
    theme.inject_global_css()

    models, reports, meta = get_models(), get_reports(), get_meta()
    if not models:
        st.error("No se pudieron cargar los modelos.")
        return
    _init_state()
    render_header(meta)

    tabs = st.tabs(["Simulador", "Mapa de movimiento", "Leaderboard", "Sculptor", "VeloVision", "Modo anotación"])
    C.consume_tab_jump()      # salto one-shot pedido por un botón "Llevar a…" (ver components.switch_to_tab)
    with tabs[0]:
        C.context_pill("Simulador", active=st.session_state.get("source_mode") == "REAL_PITCHER")
        render_simulator(reports, meta)
    with tabs[1]:
        render_movement_map(reports)
    with tabs[2]:
        render_leaderboard(reports, meta, _on_load_pitcher)
    with tabs[4], st.container(key="zoom_velovision"):
        render_velovision_tab()
    with tabs[3], st.container(key="zoom_sculptor"):
        # VeloVision (arriba) pudo cargar una mecánica medida; se reaplica tras el Simulador, que reescribe el estado
        apply_velovision_override()
        if render_sculptor_tab is None:
            st.info("Pestaña en construcción.")
        else:
            render_sculptor_tab({"arsenal": reports["arsenal"], "meta": meta, "load_cb": _on_load_pitcher})
    with tabs[5], st.container(key="zoom_annotation"):
        render_annotation_tab()


if __name__ == "__main__":
    main()

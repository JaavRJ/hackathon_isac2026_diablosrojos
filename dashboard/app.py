"""Diablos AeroVision - Dashboard Simulador Base (HACK-015).

Lanzar desde la raiz del proyecto:

    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

Interfaz 100 % en espanol. Carga solo ``models/*.joblib`` y ``reports/**/*.csv`` (rutas
relativas a la raiz); la logica pura vive en ``dashboard/ui_utils.py``.

Vistas: (1) Simulador lado a lado Nivel del Mar vs. Harp Helu, (2) Mapa de movimiento,
(3) Leaderboard + tarjeta de lanzador, (4) Pitch Sculptor (modulo externo opcional).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from dashboard import ui_utils as ui

try:  # HACK-016 entrega este modulo en paralelo
    from dashboard.sculptor_tab import render_sculptor_tab
except ImportError:
    render_sculptor_tab = None

HAND_LABELS = {"Derecho": "Right", "Izquierdo": "Left"}
HAND_NAMES = {v: k for k, v in HAND_LABELS.items()}

CSS = f"""
<style>
.stApp {{ background-color: {ui.BLACK}; color: {ui.LIGHT}; }}
h1, h2, h3, h4 {{ color: {ui.LIGHT}; }}
.dr-title {{ border-left: 8px solid {ui.RED}; padding-left: 14px; margin-bottom: 8px; }}
.dr-card {{ background: {ui.PANEL}; border: 1px solid #333; border-top: 4px solid {ui.RED};
           border-radius: 10px; padding: 16px 18px; height: 100%; }}
.dr-card.sea {{ border-top-color: {ui.LIGHT}; }}
.dr-card h4 {{ margin: 0 0 6px 0; font-size: 1.0rem; color: {ui.LIGHT}; }}
.dr-big {{ font-size: 2.6rem; font-weight: 800; line-height: 1.1; }}
.dr-lbl {{ color: {ui.GREY}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }}
.dr-row {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 10px; }}
.dr-row div {{ min-width: 90px; }}
.dr-val {{ font-size: 1.3rem; font-weight: 700; }}
.dr-red {{ color: {ui.RED}; }}
.dr-note {{ color: {ui.GREY}; font-size: 0.85rem; }}
.dr-tag {{ display:inline-block; background:{ui.RED}; color:{ui.LIGHT}; border-radius:4px;
          padding:1px 8px; font-size:.75rem; margin-left:6px; }}
</style>
"""


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


# ----------------------------------------------------------------------------
# Utilidades de UI
# ----------------------------------------------------------------------------
def score_card(title: str, res: dict, css: str, delta: float | None = None) -> str:
    badge = f'<span class="dr-tag">ΔStuff+ {ui.fmt_delta(delta)}</span>' if delta is not None else ""
    return f"""
<div class="dr-card {css}">
  <h4>{title}{badge}</h4>
  <div class="dr-lbl">Stuff+</div>
  <div class="dr-big {'dr-red' if css != 'sea' else ''}">{ui.fmt_score(res['stuff'])}</div>
  <div class="dr-row">
    <div><div class="dr-lbl">Location+</div><div class="dr-val">{ui.fmt_score(res['location'])}</div></div>
    <div><div class="dr-lbl">Pitching+</div><div class="dr-val">{ui.fmt_score(res['pitching'])}</div></div>
    <div><div class="dr-lbl">IVB (in)</div><div class="dr-val">{res['ivb']:.1f}</div></div>
    <div><div class="dr-lbl">HB (in)</div><div class="dr-val">{res['hb']:+.1f}</div></div>
  </div>
</div>"""


def _init_state(defaults: dict) -> None:
    for k, v in defaults.items():
        st.session_state.setdefault(f"in_{k}", v)


def _apply_inputs(values: dict) -> None:
    """Callback de boton: escribe los valores de un pitcheo real en los widgets."""
    for k, v in ui.clip_inputs(values).items():
        st.session_state[f"in_{k}"] = v
    if "pitcher_throws" in values:
        st.session_state["hand_widget"] = HAND_NAMES[values["pitcher_throws"]]


def _num(label: str, name: str, step: float, disabled: bool = False) -> None:
    lo, hi = ui.INPUT_BOUNDS[name]
    st.number_input(label, lo, hi, step=step, key=f"in_{name}", disabled=disabled)


def _slider(label: str, name: str, step: float) -> None:
    lo, hi = ui.INPUT_BOUNDS[name]
    st.slider(label, lo, hi, step=step, key=f"in_{name}")


def _collect_inputs(defaults: dict) -> dict:
    return {k: st.session_state[f"in_{k}"] for k in defaults}


def _inputs_key(inputs: dict) -> tuple:
    return tuple(sorted((k, round(v, 6) if isinstance(v, float) else v) for k, v in inputs.items()))


# ----------------------------------------------------------------------------
# Vista 1: Simulador
# ----------------------------------------------------------------------------
def render_simulator(reports: dict, meta: dict) -> None:
    st.subheader("Simulador de pitcheo: Nivel del Mar vs. Estadio Alfredo Harp Helú")
    st.caption(
        "Ingresa un pitcheo (valores medidos a nivel del mar, ρ̂ = 1.00) o cárgalo desde un pitcher real. "
        f"“Harp Helú” usa ρ̂ = {meta['rho_cdmx']:.3f}, la mediana de la categoría Altitud Extrema de la LMB "
        "(agrupa varios parques de altura, no solo CDMX)."
    )
    defaults = ui.default_pitch_inputs(meta)
    _init_state(defaults)

    # --- Carga desde un pitcher real ---------------------------------------------------
    lb, ars = reports["leaderboard"], reports["arsenal"]
    if len(lb) and len(ars):
        with st.expander("Cargar desde un pitcher real del leaderboard", expanded=False):
            c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
            pid = c1.selectbox("Pitcher", sorted(lb["pitcher_anon_id"].unique()), key="load_pid")
            years = sorted(lb.loc[lb["pitcher_anon_id"] == pid, "year"].unique(), reverse=True)
            year = c2.selectbox("Temporada", years, key="load_year")
            sub = ars[(ars["pitcher_anon_id"] == pid) & (ars["year"] == year)].sort_values("n_pitches", ascending=False)
            pitch = c3.selectbox("Pitcheo", list(sub["pitch_id"]), key="load_pitch") if len(sub) else None
            if pitch is not None:
                row = sub[sub["pitch_id"] == pitch].iloc[0]
                summary = lb[(lb["pitcher_anon_id"] == pid) & (lb["year"] == year)].iloc[0].to_dict()
                values = ui.inputs_from_arsenal_row(row, summary, meta)
                c4.button("Cargar", on_click=_apply_inputs, args=(values,), use_container_width=True)
            st.caption(
                "Al cargar un pitcheo real: la altura de liberación (rel_height) sale del arsenal, el lado de "
                "liberación (rel_side) se deriva de su arm slot (arm_slot_deg) y solo el eje de giro usa la mediana "
                "de entrenamiento. Ajústalos a mano si los conoces."
            )

    # --- Entradas ---------------------------------------------------------------------
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Pitcheo**")
        st.selectbox("Tipo de pitcheo", ui.PITCH_TYPES, key="in_pitch_id")
        hand_label = st.selectbox("Mano del lanzador", list(HAND_LABELS), index=list(HAND_LABELS.values()).index(
            st.session_state["in_pitcher_throws"]), key="hand_widget")
        st.session_state["in_pitcher_throws"] = HAND_LABELS[hand_label]
        _num("Velocidad de liberación (mph)", "rel_speed", 0.5)
        _num("Giro (rpm)", "spin_rate", 10.0)
        _num("Eje de giro (°, 0–360)", "spin_axis", 1.0)
        _num("Quiebre vertical inducido, IVB (in)", "ivb", 0.5)
        _num("Quiebre horizontal, HB (in; + lado brazo / − lado guante)", "hb_norm", 0.5)
    with col_b:
        st.markdown("**Punto de liberación**")
        _num("Altura de liberación (pies)", "rel_height", 0.05)
        _num("Lado de liberación (pies; + = lado del brazo)", "rel_side", 0.05)
        _num("Extensión (pies)", "extension", 0.05)
        st.markdown("**Conteo y zona**")
        k1, k2, k3 = st.columns(3)
        k1.selectbox("Bolas", [0, 1, 2, 3], key="in_balls")
        k2.selectbox("Strikes", [0, 1, 2], key="in_strikes")
        k3.selectbox("Outs", [0, 1, 2], key="in_outs")
        bl = st.selectbox("Bateador", list(HAND_LABELS), index=list(HAND_LABELS.values()).index(
            st.session_state["in_batter_side"]), key="batter_widget")
        st.session_state["in_batter_side"] = HAND_LABELS[bl]
    with col_c:
        st.markdown("**Ubicación en el plato**")
        _slider("Lado (pies, vista del catcher; − = izquierda)", "plate_side", 0.05)
        _slider("Altura (pies)", "plate_height", 0.05)
        geo = ui.zone_geometry(st.session_state["in_plate_side"], st.session_state["in_plate_height"])
        st.caption("Dentro de la zona de strike." if geo["in_strike_zone"] else "Fuera de la zona de strike.")
        with st.expander("Arsenal y túnel (avanzado)"):
            st.checkbox("Este pitcheo es la recta primaria", key="in_is_primary_fb")
            off = bool(st.session_state["in_is_primary_fb"]) and st.session_state["in_pitch_id"] in ui.FASTBALL_IDS
            _num("Velocidad de la recta primaria (mph)", "fb_velo", 0.5, off)
            _num("IVB de la recta primaria (in)", "fb_ivb", 0.5, off)
            _num("HB de la recta primaria (in)", "fb_hb_norm", 0.5, off)
            _slider("Similitud de túnel en la liberación", "tunnel_sim_release", 0.01)
            _slider("Similitud de túnel a 170 ms", "tunnel_sim_170ms", 0.01)
            _slider("Uso del pitcheo en el repertorio", "pitch_usage_pct", 0.01)
            st.caption("Si es la recta primaria, las diferencias contra la recta valen 0; si no, se calculan contra los valores de la recta primaria.")

    st.caption(
        "Cómo se completa el pitcheo ingresado: se estiman con física simple ZoneSpeed, EffectiveVelo, "
        "VertApprAngle, SpinToBreakRatio y arm_slot_deg; usan medianas de entrenamiento VertRelAngle, "
        "HorzRelAngle_norm, Inning y los valores de túnel. No son mediciones."
    )

    inputs = _collect_inputs(defaults)
    res = cached_simulation(_inputs_key(inputs))

    # --- Tarjetas lado a lado ----------------------------------------------------------
    st.markdown("### Resultado lado a lado")
    left, right = st.columns(2)
    left.markdown(score_card("Nivel del Mar (ρ̂ = 1.00)", res["neutral"], "sea"), unsafe_allow_html=True)
    right.markdown(
        score_card(f"Estadio Alfredo Harp Helú (ρ̂ = {res['rho_cdmx']:.3f})", res["cdmx"], "cdmx", res["delta_stuff"]),
        unsafe_allow_html=True,
    )
    st.caption(
        "ΔStuff+ = Stuff+ Harp Helú − Stuff+ Nivel del Mar. Location+ no depende de la densidad del aire "
        f"(Δ = {ui.fmt_delta(res['delta_location'])}); ΔPitching+ = {ui.fmt_delta(res['delta_pitching'])}. "
        "IVB y HB de Harp Helú son el movimiento traducido por la Etapa A (modelo físico)."
    )

    st.plotly_chart(ui.fig_movement_vector(res), use_container_width=True)

    # --- SHAP bajo demanda --------------------------------------------------------------
    st.markdown("### ¿Qué variables impulsan el Stuff+?")
    show_shap = st.checkbox("Calcular explicación SHAP (solo para este pitcheo)", value=True, key="show_shap")
    if show_shap:
        flavor = st.radio("Sabor a explicar", ["Harp Helú", "Nivel del Mar"], horizontal=True, key="shap_flavor")
        rho = res["rho_cdmx"] if flavor == "Harp Helú" else ui.SEA_LEVEL_RHO
        expl = ui.explain_pitch_shap(get_models(), res["frame"], rho)
        st.plotly_chart(ui.fig_shap(ui.shap_table(expl, 10)), use_container_width=True)
        st.caption(
            f"Base {expl['base_value']:.1f} + suma de contribuciones = Stuff+ {expl['predicted_stuff_plus']:.1f}. "
            "Verde suma, rojo resta (puntos de Stuff+)."
        )

    # --- Descomposicion de tres terminos -------------------------------------------------
    st.markdown("### Descomposición de la diferencia por altitud (tres términos)")
    payload = ui.decomposition_payload(reports["decomposition"])
    st.info(payload["population_note"])
    if payload["rows"]:
        table = pd.DataFrame([{
            "Término": r["label"],
            "Estimación (xRV/pitcheo)": r["estimate"],
            "IC 95 % inferior": r["ci_low"],
            "IC 95 % superior": r["ci_high"],
            "Participación del total": ui.fmt_pct(r["share_of_total"]),
        } for r in payload["rows"]])
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.5f")
                                    for c in ("Estimación (xRV/pitcheo)", "IC 95 % inferior", "IC 95 % superior")})
        st.plotly_chart(ui.fig_decomposition(payload), use_container_width=True)
        if payload["stuff_note"]:
            st.warning(payload["stuff_note"])
        st.caption("El residuo se muestra aparte: no se suma ni se renombra como Canal Parque.")
    else:
        st.warning("No se encontró reports/altitude/channel_decomposition.csv.")


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
    st.plotly_chart(ui.fig_movement_heatmap(matrix, key, f"{metric_label} · {pitch}"), use_container_width=True)
    st.caption(
        "Cada celda promedia (ponderado por pitcheos) el score de los pitcheos observados en ese cuadrante de "
        "movimiento; se suprimen celdas con muy pocos pitcheos. En ΔStuff+, verde = gana en altitud y rojo = pierde."
    )


# ----------------------------------------------------------------------------
# Vista 3: Leaderboard y tarjeta de lanzador
# ----------------------------------------------------------------------------
def render_leaderboard(reports: dict) -> None:
    st.subheader("Leaderboard de pitchers y arsenales")
    lb, ars = reports["leaderboard"], reports["arsenal"]
    if lb.empty:
        st.warning("No se encontró reports/leaderboards/pitcher_season_leaderboard.csv.")
        return
    f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1, 1.2, 1.4])
    years = f1.multiselect("Temporada", sorted(lb["year"].unique()), default=sorted(lb["year"].unique())[-1:], key="lb_years")
    roles = f2.multiselect("Rol (aprox.)", sorted(lb["rol"].unique()), key="lb_roles") if "rol" in lb else []
    hands = f3.multiselect("Mano", [HAND_NAMES.get(h, h) for h in sorted(lb["pitcher_throws"].dropna().unique())],
                           key="lb_hands") if "pitcher_throws" in lb else []
    min_p = f4.slider("Mínimo de pitcheos", 0, int(lb["n_pitches"].max()), 100, step=10, key="lb_min")
    sort_label = f5.selectbox("Ordenar por", list(ui.SORT_OPTIONS), key="lb_sort")
    asc = st.checkbox("Orden ascendente", value=False, key="lb_asc")
    throws = [HAND_LABELS[h] for h in hands]
    view = ui.filter_leaderboard(lb, years, roles, throws, min_p, ui.SORT_OPTIONS[sort_label], asc)
    st.caption(
        f"{len(view)} pitcher-temporadas. El rol es una heurística por repertorio (≥3 pitcheos con ≥10 % de uso), "
        "no un dato oficial. Los scores de 2024–2025 son dentro de muestra; 2026 es fuera de muestra."
    )
    table = ui.display_table(view, ui.LEADERBOARD_COLUMNS)
    st.dataframe(table, hide_index=True, use_container_width=True, height=380)

    st.markdown("### Tarjeta de lanzador")
    if view.empty:
        st.info("Ajusta los filtros para elegir un pitcher.")
        return
    labels = [f"{r.pitcher_anon_id} · {int(r.year)}" for r in view.itertuples()]
    choice = st.selectbox("Pitcher", labels, key="card_pick")
    pid, yr = choice.split(" · ")
    card = ui.pitcher_card(lb, ars, pid, int(yr))
    s = card["summary"]
    st.caption("CDMX = mediana de la categoría Altitud Extrema de la LMB (agrupa varios parques de altura, no solo CDMX).")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Stuff+ Nivel del Mar", ui.fmt_score(s.get("stuff_plus_neutral")))
    m2.metric("Stuff+ CDMX", ui.fmt_score(s.get("stuff_plus_cdmx")), ui.fmt_delta(s.get("delta_stuff_cdmx")))
    m3.metric("Location+", ui.fmt_score(s.get("location_plus_neutral")))
    m4.metric("Pitching+ CDMX", ui.fmt_score(s.get("pitching_plus_cdmx")))
    m5.metric("Pitcheos", f"{int(s.get('n_pitches', 0))}")
    if not card["arsenal"].empty:
        a = card["arsenal"]
        st.plotly_chart(ui.fig_arsenal(a), use_container_width=True)
        st.dataframe(ui.display_table(a, [
            "pitch_id", "n_pitches", "usage_pct", "rel_speed_mean", "spin_rate_mean", "ivb_neutral", "ivb_cdmx",
            "hb_norm_neutral", "hb_norm_cdmx", "stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx",
        ]), hide_index=True, use_container_width=True)
    ch = card["channels"]
    if any(v is not None for v in ch.values()):
        st.markdown("**Descomposición de tres términos de este lanzador (xRV/pitcheo)**")
        d1, d2, d3 = st.columns(3)
        d1.metric(ui.LABEL_STUFF, "—" if ch["canal_stuff"] is None else f"{ch['canal_stuff']:+.5f}")
        d2.metric(ui.LABEL_PARK, "—" if ch["canal_parque_bip"] is None else f"{ch['canal_parque_bip']:+.5f}")
        d3.metric("Residuo no atribuido", "—" if ch["residuo_no_atribuido"] is None else f"{ch['residuo_no_atribuido']:+.5f}")
        st.caption("Residuo: no se atribuye a ningún canal.")
    else:
        st.caption("Descomposición por lanzador no disponible (sin lanzamientos suficientes en ambas altitudes).")


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Diablos AeroVision", page_icon="⚾", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="dr-title"><h1>Diablos AeroVision</h1>'
                '<div class="dr-note">Stuff+ ajustado por entorno · Liga Mexicana de Beisbol · '
                'Diablos Rojos del México</div></div>', unsafe_allow_html=True)

    models, reports, meta = get_models(), get_reports(), get_meta()
    if not models:
        st.error("No se pudieron cargar los modelos.")
        return

    sculptor_title = "Referencias Factibles (Pitch Sculptor)"
    tabs = st.tabs(["Simulador", "Mapa de movimiento", "Leaderboard", sculptor_title])
    with tabs[0]:
        render_simulator(reports, meta)
    with tabs[1]:
        render_movement_map(reports)
    with tabs[2]:
        render_leaderboard(reports)
    with tabs[3]:
        if render_sculptor_tab is None:
            st.info("Pestaña en construcción.")
        else:
            render_sculptor_tab()


if __name__ == "__main__":
    main()

"""Pestaña "Referencias Factibles (Pitch Sculptor)" (HACK-016, rediseño HACK-019 Fase 3).

Expone ``render_sculptor_tab()`` para ``dashboard/app.py`` y ``render_sculptor_result()``, que reutiliza
también el panel de pares del Simulador. Es una búsqueda de pares reales de la LMB; no inventa ni
reevalúa pitcheos. Se calcula SOLO con el pitcheo activo (``active_pitch_params``); el formulario es un
ajuste opcional.
"""
from __future__ import annotations

from pathlib import Path

from dashboard import components as C
from dashboard import theme
from src.sculptor.optimizer import COACHING_LABEL, DELTA_DISCLAIMER, PitchSculptor

TAB_TITLE = "Referencias Factibles (Pitch Sculptor)"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "reports" / "leaderboards" / "pitcher_arsenal_breakdown.csv"
PITCH_IDS = ["Four-Seam", "Sinker", "Cutter", "Slider", "Sweeper", "Curveball", "Changeup", "Splitter"]
HANDS = {"Right": "Derecho", "Left": "Zurdo"}  # solo visualizacion


def coaching_block_html(text: str) -> str:
    """Bloque de la guía cualitativa: etiqueta y texto juntos, punteado y neutro (el rojo de marca no es dato)."""
    return (
        f'<div style="border:2px dotted #98A1AE;background:#FAFBFC;color:{theme.TEXT};'
        f'padding:12px;border-radius:{theme.R_PX}px;font-style:italic;">'
        f'<strong style="font-family:{theme.FONT_MONO};font-style:normal;">{COACHING_LABEL}</strong><br>{text}</div>'
    )


def _prefill() -> dict:
    """Pitcheo activo del Simulador (``active_pitch_params``) para autollenar el ajuste opcional."""
    import streamlit as st

    p = st.session_state.get("active_pitch_params") or {}
    last = st.session_state.get("last_sim_results") or {}
    return {
        "throws": p.get("PitcherThrows") if p.get("PitcherThrows") in HANDS else "Right",
        "pitch_id": p.get("pitch_id") if p.get("pitch_id") in PITCH_IDS else "Four-Seam",
        "arm_slot": min(90.0, max(0.0, float(p.get("arm_slot_deg") or 45.0))),
        "rel_height": min(8.0, max(3.0, float(p.get("RelHeight") or 5.8))),
        "fb_velo": min(105.0, max(60.0, float(p.get("fb_velo") or 93.0))),
        "score": last.get("stuff_cdmx"),
    }


def _sculptor():
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load():
        return PitchSculptor(CATALOG_PATH)

    return _load()


def compute_sculptor(params: dict) -> dict:
    """Resultado COMPLETO de ``PitchSculptor.find_feasible_references`` (cacheado por parametros)."""
    import streamlit as st

    @st.cache_data(show_spinner=False, max_entries=128)
    def _run(key: tuple) -> dict:
        return _sculptor().find_feasible_references(dict(key))

    return _run(tuple(sorted(params.items())))


def render_sculptor_result(res: dict, params: dict, user_score: float | None, kp: str, ctx: dict | None = None,
                           snapshot: dict | None = None) -> None:
    """Pinta el resultado del Sculptor con los componentes del sistema (tab y panel del Simulador)."""
    import streamlit as st

    f = res["filters"]
    chips = ["misma mano", f"±{f['arm_slot_tol_deg']:.1f}° arm slot", f"±{f['rel_height_tol_ft']:.2f} ft altura",
             f"±{f['fb_velo_tol_mph']:.1f} mph recta", f"≥ {f['min_pitches']} pitcheos"]
    st.markdown(C.provenance_row(["PitchSculptor (HACK-016)", "fuente: pitcher_arsenal_breakdown.csv",
                                  "2026 fuera de muestra"]), unsafe_allow_html=True)
    st.markdown(C.counter_card(res["n_peers"], f"en la LMB · {res['pitch_id']}", chips)
                if res["status"] == "INSUFFICIENT_PEERS" else "", unsafe_allow_html=True)
    if res["status"] == "INSUFFICIENT_PEERS":
        st.markdown(C.callout(f"INSUFFICIENT_PEERS: {res['reason']}"), unsafe_allow_html=True)
        st.markdown(f'<div class="note">{C.esc(res["delta_gain_reason"] or "")} Los filtros no se relajan.</div>',
                    unsafe_allow_html=True)
        return
    intra = res["intra_family"]
    top = st.columns([0.34, 0.66], gap="small")
    top[0].markdown(C.counter_card(res["n_peers"], f"en la LMB · {res['pitch_id']}", chips), unsafe_allow_html=True)
    if intra:
        best = intra["best_reference"]
        head = (f'<div class="dr-card"><div class="dr-h4">{C.icon("target")}Stuff+ CDMX de pares con {C.esc(res["pitch_id"])} '
                f'({intra["n_rows"]} filas) · franja p25–p75</div>')
        top[1].markdown(head + C.range_bar(intra["p25_stuff_plus_cdmx"], intra["p75_stuff_plus_cdmx"], user_score,
                                           best["stuff_plus_cdmx"]) + "</div>", unsafe_allow_html=True)
    else:
        top[1].markdown(C.callout(res["reason"] or "Sin pares de esta familia."), unsafe_allow_html=True)

    a, b, c = st.columns([1.25, 1, 1.1], gap="small")
    comps = "".join(C.comparable_card(x, bool(snapshot) and snapshot.get("pitcher_anon_id") == x["pitcher_anon_id"])
                    for x in res["comparables"])
    best_txt = (f'Mejor referencia: {C.esc(intra["best_reference"]["pitcher_anon_id"])} · {intra["best_reference"]["year"]}.'
                if intra else "")
    a.markdown(f'<div class="dr-card"><div class="dr-h4">{C.icon("users")}Top 3 comparables LMB</div>'
               f'<div style="margin-top:.5rem">{comps}</div><div class="note">{best_txt}</div></div>', unsafe_allow_html=True)

    if res["delta_gain_cdmx"] is not None:
        dg = (f'<div class="dr-dg"><span class="mut" style="font-size:.72rem">delta_gain_cdmx</span>'
              f'<span class="n dr-num">{C.delta_badge(res["delta_gain_cdmx"], 2)}</span></div>')
    else:
        dg = f'<div class="note">{C.esc(res["delta_gain_reason"] or "")}</div>'
    b.markdown(f'<div class="dr-card"><div class="dr-h4">{C.icon("chart")}Repertorio secundario sostenido por pares</div>'
               f'<div style="margin-top:.4rem">{C.repertoire_rows(res["repertoire"])}</div>'
               f'<div class="note">Mediana de Stuff+ CDMX entre pares factibles.</div>'
               f'<div style="margin-top:.7rem">{dg}</div><div class="note ill">{C.esc(DELTA_DISCLAIMER)}</div></div>',
               unsafe_allow_html=True)

    with c:
        if res["coaching_guide"]:
            st.markdown(coaching_block_html(res["coaching_guide"]), unsafe_allow_html=True)
        st.markdown(
            f'<div class="dr-card"><div class="dr-h4">{C.icon("ball")}Tu pitcheo</div><div class="dr-mini2" style="grid-template-columns:repeat(3,1fr);margin-top:.4rem">'
            f'<div class="dr-mini"><span class="t">{C.icon("arm")}Arm slot</span><span class="v dr-num" style="font-size:1.8rem">{params["arm_slot_deg"]:.1f}<small>°</small></span></div>'
            f'<div class="dr-mini"><span class="t">{C.icon("up")}Altura</span><span class="v dr-num" style="font-size:1.8rem">{params["RelHeight"]:.2f}<small>ft</small></span></div>'
            f'<div class="dr-mini"><span class="t">{C.icon("gauge")}Recta</span><span class="v dr-num" style="font-size:1.8rem">{params["fb_velo"]:.1f}<small>mph</small></span></div></div></div>',
            unsafe_allow_html=True)
        if ctx and intra and ctx.get("load_cb"):
            ref = intra["best_reference"]
            ars = ctx["arsenal"]
            row = ars[(ars["pitcher_anon_id"] == ref["pitcher_anon_id"]) & (ars["year"] == ref["year"])
                      & (ars["pitch_id"] == ref["pitch_id"])]
            if len(row):
                from dashboard import ui_utils as ui
                r = row.iloc[0]
                values = ui.inputs_from_arsenal_row(r, {}, ctx["meta"])
                snap = {"pitcher_anon_id": ref["pitcher_anon_id"], "year": int(ref["year"]), "pitch_id": ref["pitch_id"],
                        "n_pitches": int(ref["n_pitches"]), "inputs": dict(values)}
                st.button("Llevar referencia al Simulador", key=f"{kp}_load_ref", type="primary", width="stretch",
                          on_click=ctx["load_cb"], args=(values, snap))


def render_sculptor_tab(ctx: dict | None = None) -> None:
    import streamlit as st

    st.markdown(f"### {TAB_TITLE}")
    st.caption(
        "Búsqueda de referencias reales de la LMB con la misma mano, arm slot, altura de "
        "liberación y velocidad. No genera pitcheos nuevos ni garantiza resultados. "
        "Se calcula con el pitcheo activo del Simulador."
    )
    try:
        _sculptor()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"No se pudo cargar el catálogo de arsenal: {exc}")
        return
    pre = _prefill()
    with st.expander("Ajustar parámetros (opcional)", expanded=False):
        c1, c2 = st.columns(2)
        throws = c1.selectbox("Mano", list(HANDS), index=list(HANDS).index(pre["throws"]), format_func=HANDS.get)
        pitch_id = c2.selectbox("Familia de pitcheo", PITCH_IDS, index=PITCH_IDS.index(pre["pitch_id"]))
        arm_slot = c1.number_input("Arm slot (°)", 0.0, 90.0, round(pre["arm_slot"], 1))
        rel_height = c2.number_input("Altura de liberación (ft)", 3.0, 8.0, round(pre["rel_height"], 2))
        fb_velo = c1.number_input("Velocidad de la recta (mph)", 60.0, 105.0, round(pre["fb_velo"], 1))
        st.caption("El formulario parte del pitcheo activo; ajustarlo no cambia el Simulador.")
    params = {"PitcherThrows": throws, "arm_slot_deg": float(arm_slot), "RelHeight": float(rel_height),
              "fb_velo": float(fb_velo), "pitch_id": pitch_id}
    score = pre["score"] if pitch_id == pre["pitch_id"] and pre["score"] is not None else None
    res = compute_sculptor({**params, "user_stuff_plus_cdmx": score})
    snap = st.session_state.get("loaded_pitcher_snapshot")
    render_sculptor_result(res, params, score, "sc", ctx, snap)
    if res["coaching_guide"] is None and res["status"] == "INSUFFICIENT_PEERS":
        st.markdown(f'<div class="note">{C.esc(res["coaching_guide_label"])}</div>', unsafe_allow_html=True)

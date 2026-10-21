"""Pestaña "Referencias Factibles (Pitch Sculptor)" (HACK-016).

Expone ``render_sculptor_tab()`` para ``dashboard/app.py``. Es una búsqueda de
pares reales de la LMB; no inventa ni reevalúa pitcheos.
"""
from __future__ import annotations

from pathlib import Path

from src.sculptor.optimizer import COACHING_LABEL, DELTA_DISCLAIMER, PitchSculptor

TAB_TITLE = "Referencias Factibles (Pitch Sculptor)"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "reports" / "leaderboards" / "pitcher_arsenal_breakdown.csv"
PITCH_IDS = ["Four-Seam", "Sinker", "Cutter", "Slider", "Sweeper", "Curveball", "Changeup", "Splitter"]

RED, BLACK, LIGHT = "#CE1126", "#0D0D0D", "#F5F5F5"


def coaching_block_html(text: str) -> str:
    """Bloque de la guía cualitativa: etiqueta y texto juntos, con estilo diferenciado."""
    return (
        f'<div style="border:2px dashed {RED};background:#2a2a2a;color:{LIGHT};'
        f'padding:12px;border-radius:8px;font-style:italic;">'
        f'<strong style="color:{RED};">{COACHING_LABEL}</strong><br>{text}</div>'
    )


def render_sculptor_tab() -> None:
    import streamlit as st

    st.header(TAB_TITLE)
    st.caption(
        "Búsqueda de referencias reales de la LMB con la misma mano, arm slot, altura de "
        "liberación y velocidad. No genera pitcheos nuevos ni garantiza resultados."
    )
    try:
        sculptor = PitchSculptor(CATALOG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"No se pudo cargar el catálogo de arsenal: {exc}")
        return

    c1, c2 = st.columns(2)
    throws = c1.selectbox("Mano", ["Right", "Left"])
    pitch_id = c2.selectbox("Familia de pitcheo", PITCH_IDS)
    arm_slot = c1.number_input("Arm slot (°)", 0.0, 90.0, 45.0)
    rel_height = c2.number_input("Altura de liberación (ft)", 3.0, 8.0, 5.8)
    fb_velo = c1.number_input("Velocidad de la recta (mph)", 60.0, 105.0, 93.0)
    user_score = c2.number_input("Stuff+ CDMX del pitcheo (opcional, 0 = sin dato)", 0.0, 200.0, 0.0)

    if not st.button("Buscar referencias"):
        return
    res = sculptor.find_feasible_references({
        "PitcherThrows": throws, "arm_slot_deg": arm_slot, "RelHeight": rel_height,
        "fb_velo": fb_velo, "pitch_id": pitch_id,
        "user_stuff_plus_cdmx": user_score if user_score > 0 else None,
    })
    st.write(f"Estado: **{res['status']}** · pares reales: **{res['n_peers']}**")
    if res["status"] == "INSUFFICIENT_PEERS":
        st.warning(res["reason"])
    else:
        intra = res["intra_family"]
        if intra:
            st.subheader("Referencia intra-familia")
            st.write(f"Rango P25–P75 de Stuff+ CDMX: {intra['p25_stuff_plus_cdmx']:.1f} – {intra['p75_stuff_plus_cdmx']:.1f}")
            st.json(intra["best_reference"])
        if res["delta_gain_cdmx"] is not None:
            st.write(f"Diferencia: {res['delta_gain_cdmx']:+.1f} ({DELTA_DISCLAIMER})")
        else:
            st.info(res["delta_gain_reason"])
        st.subheader("Repertorio de los pares")
        st.table(res["repertoire"])
        st.subheader("Comparables LMB (IDs anónimos)")
        st.table(res["comparables"])
    if res["coaching_guide"]:
        st.markdown(coaching_block_html(res["coaching_guide"]), unsafe_allow_html=True)

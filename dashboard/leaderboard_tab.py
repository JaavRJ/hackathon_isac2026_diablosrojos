"""Pestaña Leaderboard (HACK-019, Fase 3): filtros en pastillas, tabla con barras en celda, histograma real
de dStuff+ CDMX y tarjeta de lanzador tipo estampa con "Llevar al Simulador"."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import components as C
from dashboard import ui_utils as ui

SORTS = {"Stuff+ nivel del mar": "stuff_plus_neutral", "Stuff+ CDMX": "stuff_plus_cdmx",
         "ΔStuff+ CDMX": "delta_stuff_cdmx"}
HAND_CHOICES = {"Ambas manos": None, "Derechos": "Right", "Zurdos": "Left"}
MAX_ROWS = 60


def _hand(h) -> str:
    return ui.HAND_DISPLAY.get(h, "—")


def _pill(label: str, options, key: str, default, selection_mode: str = "single"):
    val = st.pills(label, options, selection_mode=selection_mode, default=default, key=key, label_visibility="collapsed")
    return val


def render_leaderboard(reports: dict, meta: dict, load_cb=None) -> None:
    st.markdown("### Leaderboard de pitchers y arsenales")
    lb, ars = reports["leaderboard"], reports["arsenal"]
    if lb.empty:
        st.warning("No se encontró reports/leaderboards/pitcher_season_leaderboard.csv.")
        return
    years_all = sorted(int(y) for y in lb["year"].unique())

    with st.container(key="lbfilters"):
        r1 = st.columns([1.5, 1.25, 1.0], vertical_alignment="center")
        with r1[0]:
            sort_label = _pill("Ordenar por", list(SORTS), "lb_sort", "ΔStuff+ CDMX") or "ΔStuff+ CDMX"
        with r1[1]:
            year_sel = _pill("Temporada", ["Todas"] + [str(y) for y in years_all], "lb_year", "Todas") or "Todas"
        with r1[2]:
            hand_sel = _pill("Mano", list(HAND_CHOICES), "lb_hand", "Ambas manos") or "Ambas manos"
        f2 = st.columns([1.2, 1.2, 1.2, 1.4], vertical_alignment="bottom")
        with f2[0]:
            asc_sel = _pill("Orden", ["↓ Mayor primero", "↑ Menor primero"], "lb_asc", "↓ Mayor primero") or "↓ Mayor primero"
        max_n = int(lb["n_pitches"].max())
        min_p = f2[1].slider("Mínimo de pitcheos", 0, max_n, 100, step=10, key="lb_min")
        with f2[2]:
            st.markdown('<div class="note">Rol (aprox.): sin selección = todos</div>', unsafe_allow_html=True)
            roles = list(_pill("Rol (aprox.)", sorted(lb["rol"].unique()), "lb_roles", None, "multi") or []) if "rol" in lb else []
        query = f2[3].text_input("Buscar lanzador", key="lb_query", placeholder="p. ej. pitcher_00123")
        st.markdown(C.provenance_row(["fuente: pitcher_season_leaderboard.csv", "Δ = CDMX − mar",
                                      f"ρ̂ = {meta['rho_cdmx']:.4f}", "2026 fuera de muestra"]), unsafe_allow_html=True)

    years = None if year_sel == "Todas" else [int(year_sel)]
    throws = [h] if (h := HAND_CHOICES[hand_sel]) else None
    view = ui.filter_leaderboard(lb, years, roles, throws, min_p, SORTS[sort_label], asc_sel.startswith("↑"))
    if query.strip():
        view = view[view["pitcher_anon_id"].str.contains(query.strip(), case=False, regex=False)].reset_index(drop=True)
    st.caption(
        f"{len(view)} pitcher-temporadas. El rol es una heurística por repertorio (≥3 pitcheos con ≥10 % de uso), "
        "no un dato oficial. Los scores de 2024–2025 son dentro de muestra; 2026 es fuera de muestra."
    )
    if view.empty:
        st.markdown(C.callout("Ningún lanzador cumple los filtros. Ajusta la temporada, la mano o el mínimo de pitcheos."),
                    unsafe_allow_html=True)
        return

    shown = view.head(MAX_ROWS)
    labels = [f"{r.pitcher_anon_id} · {int(r.year)}" for r in shown.itertuples()]
    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        choice = st.selectbox("Ver tarjeta de lanzador", labels, key="card_pick", placeholder="Elegir lanzador")
        sel = labels.index(choice) if choice in labels else 0
        lo = float(np.floor(min(lb["stuff_plus_neutral"].min(), lb["stuff_plus_cdmx"].min())))
        hi = float(np.ceil(max(lb["stuff_plus_neutral"].max(), lb["stuff_plus_cdmx"].max())))
        dmax = float(lb["delta_stuff_cdmx"].abs().max())
        rows = [{"id": r.pitcher_anon_id, "year": int(r.year), "hand": _hand(r.pitcher_throws), "n": int(r.n_pitches),
                 "neutral": r.stuff_plus_neutral, "cdmx": r.stuff_plus_cdmx, "delta": r.delta_stuff_cdmx,
                 "pitching": r.pitching_plus_cdmx, "rp100": r.runs_prevented_per_100} for r in shown.itertuples()]
        with st.container(key="lbtable"):
            st.markdown(C.lb_table(rows, lo, hi, dmax, sel), unsafe_allow_html=True)
            if len(view) > MAX_ROWS:
                st.caption(f"Se muestran las primeras {MAX_ROWS} de {len(view)} filas; usa los filtros para acotar.")
        pid, yr = labels[sel].split(" · ")
        srow = view[(view["pitcher_anon_id"] == pid) & (view["year"] == int(yr))].iloc[0]
        with st.container(key="lbhist"):
            st.markdown(f'<div class="dr-h4">{C.icon("chart")}Distribución de ΔStuff+ CDMX en la liga</div>'
                        + C.provenance_row([f"{len(view)} temporadas-lanzador", "fuente: pitcher_season_leaderboard.csv"]),
                        unsafe_allow_html=True)
            st.plotly_chart(ui.fig_delta_hist(view["delta_stuff_cdmx"], srow["delta_stuff_cdmx"], pid), width="stretch")
    with right:
        _stamp(lb, ars, pid, int(yr), meta, load_cb)


def _stamp(lb, ars, pid: str, yr: int, meta: dict, load_cb) -> None:
    card = ui.pitcher_card(lb, ars, pid, yr)
    s = card["summary"]
    a = card["arsenal"]
    with st.container(key="stamp"):
        st.markdown(C.stamp_head(pid, f"Temporada {yr} · {_hand(s.get('pitcher_throws'))} · {int(s.get('n_pitches', 0))} pitcheos",
                                 s.get("delta_stuff_cdmx")), unsafe_allow_html=True)
        st.markdown(f'<div class="note">Stuff+ {ui.fmt_score(s.get("stuff_plus_neutral"))} → {ui.fmt_score(s.get("stuff_plus_cdmx"))} '
                    f'(mar → CDMX) · Pitching+ CDMX {ui.fmt_score(s.get("pitching_plus_cdmx"))}. '
                    "CDMX = mediana de la categoría Altitud Extrema de la LMB (agrupa varios parques de altura, no solo CDMX).</div>",
                    unsafe_allow_html=True)
        if len(a):
            st.markdown(C.arsenal_chips([(r.pitch_id, r.usage_pct) for r in a.itertuples()]), unsafe_allow_html=True)
            st.markdown(f'<div class="dr-h4">{C.icon("ball")}Mapa del arsenal · mar ○ → CDMX ★</div>', unsafe_allow_html=True)
            st.plotly_chart(ui.fig_arsenal_map(a), width="stretch")
            allv = pd.concat([ars["stuff_plus_neutral"], ars["stuff_plus_cdmx"]])
            lo, hi = float(np.floor(allv.min())), float(np.ceil(allv.max()))
            st.markdown(f'<div class="dr-h4">{C.icon("chart")}Stuff+ por pitcheo</div>'
                        + C.arsenal_bars([{"pitch": r.pitch_id, "sea": r.stuff_plus_neutral, "cdmx": r.stuff_plus_cdmx,
                                           "delta": r.delta_stuff_cdmx} for r in a.itertuples()], lo, hi),
                        unsafe_allow_html=True)
        else:
            st.markdown(C.callout("Este lanzador no tiene arsenal en el catálogo."), unsafe_allow_html=True)
        has_cols = all(c in lb.columns for c in ("canal_stuff", "canal_parque_bip", "residuo_no_atribuido"))
        st.markdown(f'<div class="dr-h4">{C.icon("split")}Descomposición de tres términos (xRV/pitcheo)</div>',
                    unsafe_allow_html=True)
        if has_cols:
            st.markdown(C.channel_rows(card["channels"], {
                "canal_stuff": ui.LABEL_STUFF, "canal_parque_bip": ui.LABEL_PARK,
                "residuo_no_atribuido": "Residuo no atribuido"}), unsafe_allow_html=True)
            st.caption("Residuo: no se atribuye a ningún canal. «No estimable» = sin lanzamientos suficientes en ambas altitudes.")
        else:
            st.caption("Descomposición por lanzador no disponible en este reporte.")
        st.markdown(C.provenance_row(["pitcher_arsenal_breakdown.csv", "pitcher_season_leaderboard.csv"]), unsafe_allow_html=True)
        if load_cb is not None and len(a):
            top = a.iloc[0]
            values = ui.inputs_from_arsenal_row(top, s, meta)
            snap = {"pitcher_anon_id": pid, "year": yr, "pitch_id": str(top["pitch_id"]),
                    "n_pitches": int(top["n_pitches"]), "inputs": dict(values)}
            st.button(f"Llevar al Simulador ({top['pitch_id']})", key="btn_to_sim", type="primary", width="stretch",
                      on_click=load_cb, args=(values, snap))
            if st.session_state.get("_loaded_flash") == f"{pid}-{yr}":
                st.caption("Cargado: abre la pestaña Simulador.")

"""Componentes de UI de Diablos AeroVision (HACK-019, Fase 1).

Funciones que devuelven HTML seguro (todo texto dinamico pasa por ``html.escape``) o que
renderizan componentes de Streamlit. Los estilos viven en ``dashboard/theme.py``.
Ninguna cifra de resultados esta escrita aqui: todo llega por argumento.
"""
from __future__ import annotations

import base64
import html
import math
from typing import Iterable, Sequence

from dashboard import theme as T

# ----------------------------------------------------------------------------
# Iconos SVG inline (trazo 24x24, mismos del mockup)
# ----------------------------------------------------------------------------
ICONS: dict[str, str] = {
    "ball": '<circle cx="12" cy="12" r="9"/><path d="M6.5 5.5c2.5 3.5 2.5 9.5 0 13M17.5 5.5c-2.5 3.5-2.5 9.5 0 13"/>',
    "fall": '<path d="M6 6l12 12M18 8v10H8"/>',
    "gauge": '<path d="M4 17a8 8 0 1 1 16 0M12 17l4-5"/>',
    "spin": '<path d="M20 12a8 8 0 1 1-3-6.2M20 4v4h-4"/>',
    "mtn": '<path d="M3 19l6-11 4 7 3-4 5 8z"/>',
    "shield": '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "undo": '<path d="M9 5L4 10l5 5M4 10h9a6 6 0 0 1 0 11h-4"/>',
    "warn": '<path d="M12 3l10 18H2zM12 10v5M12 18v.5"/>',
    "up": '<path d="M12 20V5M6 11l6-6 6 6"/>',
    "right": '<path d="M4 12h15M13 6l6 6-6 6"/>',
    "arm": '<path d="M4 20h16M4 20L16 7M11 20a7 7 0 0 0-2.2-5"/>',
    "book": '<path d="M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM5 17a3 3 0 0 1 3-3h10"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M16 16l5 5"/>',
    "flask": '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/>',
    "chart": '<path d="M5 20V11M11 20V4M17 20v-7M2 20h20"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>',
    "wave": '<path d="M3 12c3-7 6-7 9 0s6 7 9 0"/>',
    "split": '<path d="M12 3v8M12 11l-6 10M12 11l6 10"/>',
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.5 2.5-5.5 6-5.5s6 2 6 5.5M16 5.5a3 3 0 0 1 0 5.5M18 14.7c2 .6 3 2.3 3 5.3"/>',
    "chev": '<path d="M6 9l6 6 6-6"/>',
    "bulb": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.8.8 1 1.5 1 2.5h6c0-1 .2-1.7 1-2.5A6 6 0 0 0 12 3z"/>',
    "home": '<path d="M3 11l9-7 9 7M5 10v10h14V10"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5"/>',
}


def esc(text: object) -> str:
    """Escapa texto dinamico para insertarlo en HTML."""
    return html.escape(str(text), quote=True)


def icon(name: str, cls: str = "") -> str:
    """SVG inline de un icono del registro (``cls`` opcional, p. ej. ``'lg'``)."""
    inner = ICONS.get(name, ICONS["ball"])
    return f'<svg class="ic {esc(cls)}" viewBox="0 0 24 24" aria-hidden="true">{inner}</svg>'


def icon_data_uri(name: str, stroke: str = "#000000", width: float = 1.9) -> str:
    """Icono como ``data:`` URI (para usarlo como mascara CSS en botones nativos)."""
    inner = ICONS.get(name, ICONS["ball"])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
           f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# ----------------------------------------------------------------------------
# Numeros, deltas y procedencia
# ----------------------------------------------------------------------------
def fmt_num(x: float | None, decimals: int = 1, signed: bool = False) -> str:
    if x is None or x != x or x in (float("inf"), float("-inf")):
        return "—"
    return f"{x:+.{decimals}f}" if signed else f"{x:.{decimals}f}"


def delta_badge(value: float | None, decimals: int = 2, unit: str = "", cls: str = "", title: str = "") -> str:
    """Badge de delta con simbolo (▲ ● ▼) y color semantico (pierde coral / gana turquesa / neutro gris)."""
    tone = T.delta_tone(value)
    txt = fmt_num(value, decimals, signed=True).replace("-", "−")
    unit_html = f" {esc(unit)}" if unit and txt != "—" else ""
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="dpill {tone} {esc(cls)}"{t}>{T.tone_symbol(tone)} {txt}{unit_html}</span>'


def provenance_pill(text: str, title: str = "") -> str:
    """Pastilla de procedencia (modelo, CSV de origen, rho usado, dentro/fuera de muestra)."""
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="bd"{t}>{esc(text)}</span>'


def provenance_row(items: Iterable[str]) -> str:
    return '<div class="bds">' + "".join(provenance_pill(i) for i in items) + "</div>"


def anchor_pill(label: str, value: str, tooltip: str, icon_name: str = "shield") -> str:
    return (f'<span class="dr-ap" title="{esc(tooltip)}">{icon(icon_name)}{esc(label)} <b>{esc(value)}</b></span>')


def pill(text: str) -> str:
    return f'<span class="dr-pill">{esc(text)}</span>'


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
def wordmark() -> str:
    return '<div class="dr-wm"><div class="d">D</div>Diablos <span>AeroVision</span></div>'


def breadcrumb(parts: Sequence[str], source: str = "", env: str = "") -> str:
    """Pastilla de contexto del pitcheo activo: ``parts`` (negritas en el primero) + entorno."""
    segs = []
    for i, p in enumerate(parts):
        segs.append(f"<b>{esc(p)}</b>" if i == 0 else esc(p))
    body = "<i></i>".join(segs)
    src = f'<span class="src">{esc(source)}</span>' if source else ""
    envh = f'<i></i>{icon("mtn")}Entorno: <b>{esc(env)}</b>' if env else ""
    return f'<div class="dr-crumb">{icon("ball")}{body}{envh}{src}</div>'


def anchors_row(anchors: Sequence[dict]) -> str:
    """Pastillas compactas de cifras ancla de holdouts (ya calculadas desde ``reports/validation``)."""
    if not anchors:
        return ""
    pills = "".join(anchor_pill(a["label"], a["value"], a.get("title", ""), a.get("icon", "shield")) for a in anchors)
    return f'<div class="dr-anc">{pills}</div>'


# ----------------------------------------------------------------------------
# Veredicto
# ----------------------------------------------------------------------------
def verdict_banner(sentence_html: str, tone: str = "loss", reference_html: str = "", tag: str = "") -> str:
    """Banner de veredicto en lenguaje llano. ``sentence_html`` y ``reference_html`` deben venir ya escapados."""
    cls = {"loss": "", "gain": "gain"}.get(tone, "neu")
    ic = {"loss": "warn", "gain": "shield"}.get(tone, "bulb")
    ref = ""
    if reference_html:
        tg = f'<span class="tag">{esc(tag)}</span>' if tag else ""
        ref = f'<div class="ref">{icon("bulb")}<span>{reference_html}</span>{tg}</div>'
    return (f'<div class="dr-verdict {cls}" role="status">{icon(ic)}'
            f'<div><p>{sentence_html}</p>{ref}</div></div>')


# ----------------------------------------------------------------------------
# Tarjeta KPI con numero heroe
# ----------------------------------------------------------------------------
def kpi_card(
    title: str,
    icon_name: str,
    pill_text: str,
    hero: float | None,
    hero_label: str,
    hero_delta: float | None = None,
    vs_avg: float | None = None,
    avg_marker: float | None = None,
    minis: Sequence[dict] = (),
    chips: Sequence[tuple[str, str]] = (),
    note: str = "",
    title_attr: str = "",
) -> str:
    """Tarjeta grande: numero heroe (Barlow Condensed), medidor vs promedio, 2 minis y chips.

    ``minis``: ``[{'icon','label','value','unit','delta'}]``; ``avg_marker`` en 0..100 (% del medidor).
    """
    dh = delta_badge(hero_delta, 2, title="ΔStuff+ = Stuff+ Harp Helú − Stuff+ Nivel del Mar") if hero_delta is not None else ""
    hero_html = (f'<div class="dr-stf"><div class="n dr-num">{fmt_num(hero)}</div>'
                 f'<div>{dh}<div class="l" style="margin:.3rem 0 0">{esc(hero_label)}</div></div></div>'
                 if dh else
                 f'<div class="dr-stf"><div class="n dr-num">{fmt_num(hero)}</div><div class="l">{esc(hero_label)}</div></div>')
    gauge = ""
    if vs_avg is not None:
        tone = T.delta_tone(vs_avg)
        pos = 50.0 if avg_marker is None else max(2.0, min(98.0, avg_marker))
        gauge = (f'<div class="dr-gauge"><div class="dr-gt"><span>vs promedio LMB (100)</span>'
                 f'<b class="{tone}">{T.tone_symbol(tone)} {fmt_num(vs_avg, 1, signed=True).replace("-", "−")}</b></div>'
                 f'<div class="dr-gtrack"><i style="left:50%"></i><u style="left:{pos:.1f}%"></u></div></div>')
    mini_html = ""
    if minis:
        cells = []
        for m in minis:
            d = delta_badge(m["delta"], 1) if m.get("delta") is not None else ""
            d = d.replace('class="dpill', 'style="align-self:flex-start" class="dpill', 1) if d else ""
            cells.append(f'<div class="dr-mini"><span class="t">{icon(m.get("icon", "up"))}{esc(m["label"])}</span>'
                         f'<span class="v dr-num">{fmt_num(m["value"])}<small>{esc(m.get("unit", ""))}</small></span>{d}</div>')
        mini_html = '<div class="dr-mini2">' + "".join(cells) + "</div>"
    chips_html = ""
    if chips:
        chips_html = '<div class="dr-chips2">' + "".join(
            f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in chips) + "</div>"
    note_html = f'<div class="note">{esc(note)}</div>' if note else ""
    ttl = f' title="{esc(title_attr)}"' if title_attr else ""
    return (f'<div class="dr-card dr-big"><div class="hd"{ttl}>{icon(icon_name)}{esc(title)}'
            f'<span class="dr-pill">{esc(pill_text)}</span></div>{hero_html}{gauge}{mini_html}{chips_html}{note_html}</div>')


# ----------------------------------------------------------------------------
# Slider con banda "Zona LMB"
# ----------------------------------------------------------------------------
def slider_header(icon_name: str, label: str, value_text: str, unit: str) -> str:
    return (f'<div class="dr-slh">{icon(icon_name)}<span>{esc(label)}</span>'
            f'<b class="dr-num">{esc(value_text)}<small>{esc(unit)}</small></b></div>')


def band_bar(lo: float, hi: float, value: float, band: tuple[float, float] | None, title: str = "") -> str:
    """Pista HTML bajo el slider nativo: banda p10-p90 de la liga (verde translucido). El marcador del valor actual
    es el pulgar navy del slider nativo, superpuesto exactamente sobre esta pista."""
    span = (hi - lo) or 1.0

    def pct(v: float) -> float:
        return max(0.0, min(100.0, 100.0 * (v - lo) / span))

    zona = ""
    if band is not None:
        l, r = pct(band[0]), pct(band[1])
        zona = f'<i class="zona" style="left:{l:.1f}%;width:{max(r - l, 0.8):.1f}%"></i>'
    t = f' title="{esc(title)}"' if title else ""
    return f'<div class="dr-track"{t}>{zona}</div>'


def zone_legend(text: str) -> str:
    return f'<div class="dr-zleg"><i></i>{esc(text)}</div>'


# ----------------------------------------------------------------------------
# Diagnostico visual: vector ○ -> ★ sobre una reticula IVB x HB
# ----------------------------------------------------------------------------
def _nice_ticks(lo: float, hi: float, step: float = 2.0) -> list[float]:
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def diagnostic_svg(sea_hb: float, sea_ivb: float, cdmx_hb: float, cdmx_ivb: float, tone: str = "loss") -> str:
    """Reticula IVB x HB con el vector ○ (nivel del mar) -> ★ (CDMX). Ejes y escala salen de los datos."""
    W, H = 320.0, 340.0
    x0, x1, y0, y1 = 40.0, 306.0, 14.0, 300.0
    pad = 2.5
    hb_lo, hb_hi = min(sea_hb, cdmx_hb) - pad, max(sea_hb, cdmx_hb) + pad
    iv_lo, iv_hi = min(sea_ivb, cdmx_ivb) - pad, max(sea_ivb, cdmx_ivb) + pad
    # escala isotropica minima de 8 in por eje para que el vector no se deforme
    if hb_hi - hb_lo < 9:
        c = (hb_hi + hb_lo) / 2; hb_lo, hb_hi = c - 4.5, c + 4.5
    if iv_hi - iv_lo < 9:
        c = (iv_hi + iv_lo) / 2; iv_lo, iv_hi = c - 4.5, c + 4.5

    def X(v: float) -> float:
        return x0 + (v - hb_lo) / (hb_hi - hb_lo) * (x1 - x0)

    def Y(v: float) -> float:
        return y1 - (v - iv_lo) / (iv_hi - iv_lo) * (y1 - y0)

    col = T.tone_color(tone)
    parts: list[str] = []
    for v in _nice_ticks(hb_lo, hb_hi):
        parts.append(f'<line x1="{X(v):.1f}" y1="{y0}" x2="{X(v):.1f}" y2="{y1}" class="gl"/>'
                     f'<text x="{X(v):.1f}" y="{y1 + 13}" class="ax" text-anchor="middle">{v:g}</text>')
    for v in _nice_ticks(iv_lo, iv_hi):
        parts.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" class="gl"/>'
                     f'<text x="{x0 - 6}" y="{Y(v) + 3:.1f}" class="ax" text-anchor="end">{v:g}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2}" y="{H - 6}" class="ax2" text-anchor="middle">Corrida HB (in) →</text>')
    parts.append(f'<text transform="translate(10 {(y0 + y1) / 2}) rotate(-90)" class="ax2" text-anchor="middle">Subida IVB (in) →</text>')
    sx, sy, cx, cy = X(sea_hb), Y(sea_ivb), X(cdmx_hb), Y(cdmx_ivb)
    # lineas de componentes (cuanto se pierde en cada eje)
    parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{cy:.1f}" class="comp"/>'
                 f'<line x1="{sx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" class="comp"/>')
    parts.append(f'<defs><marker id="ahd" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
                 f'<path d="M0 0L8 4L0 8z" fill="{col}"/></marker></defs>')
    dx, dy = cx - sx, cy - sy
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ax1, ay1 = sx + dx / n * 12, sy + dy / n * 12
    ax2, ay2 = cx - dx / n * 14, cy - dy / n * 14
    if n > 30:
        parts.append(f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" stroke="{col}" '
                     f'stroke-width="2.4" stroke-dasharray="1 5" stroke-linecap="round" marker-end="url(#ahd)"/>')
    # ○ nivel del mar (marcador "tu pitcheo", navy) y ★ CDMX
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="10" fill="#fff" stroke="{T.NAVY}" stroke-width="2.6"/>')
    pts = []
    for k in range(10):
        r = 13 if k % 2 == 0 else 5.5
        a = -math.pi / 2 + k * math.pi / 5
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    parts.append(f'<polygon points="{" ".join(pts)}" fill="{col}" stroke="#fff" stroke-width="1.2"/>')
    parts.append(f'<text x="{sx + 14:.1f}" y="{sy - 12:.1f}" class="lb" style="fill:{T.NAVY}">○ Nivel del mar</text>')
    side = "end" if cx > (x0 + x1) / 2 else "start"
    off = -16 if side == "end" else 16
    parts.append(f'<text x="{cx + off:.1f}" y="{cy + 5:.1f}" class="lb" text-anchor="{side}" style="fill:{col}">★ CDMX</text>')
    return (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" class="dg" role="img" '
            f'aria-label="Vector de movimiento de nivel del mar a CDMX sobre una reticula IVB por HB">{"".join(parts)}</svg>')


def diagnostic_card(sea_hb: float, sea_ivb: float, cdmx_hb: float, cdmx_ivb: float, caption: str, tone: str = "loss") -> str:
    return (f'<div class="dr-card dr-diag"><div class="dr-h4">{icon("fall")}Diagnóstico visual</div>'
            f'{diagnostic_svg(sea_hb, sea_ivb, cdmx_hb, cdmx_ivb, tone)}<div class="note">{caption}</div></div>')


# ----------------------------------------------------------------------------
# Modo Analista: descomposicion de tres terminos
# ----------------------------------------------------------------------------
def decomposition_stack(rows: Sequence[dict], total_text: str = "") -> str:
    """Barra apilada de los 3 terminos (residuo rayado, con su propia etiqueta; nunca sumado al parque).

    ``rows``: filas del payload (channel in stuff/park/unallocated con estimate/ci/share_of_total/label).
    """
    by = {r["channel"]: r for r in rows}
    segs, legend = [], []
    for ch, cls in (("stuff", "dr-s1"), ("park", "dr-s2"), ("unallocated", "dr-s3")):
        r = by.get(ch)
        if r is None:
            continue
        share = r["share_of_total"]
        w = max(0.0, min(100.0, 100.0 * share)) if share == share else 0.0
        segs.append(f'<b class="{cls}" style="width:{w:.2f}%"></b>')
        legend.append(
            f'<div><i class="dr-sw {cls}"></i><span><strong>{esc(r["label"])}</strong> '
            f'<em>{r["estimate"]:+.5f}</em> xRV/pitcheo · {100 * share:.1f} % · '
            f'IC95 [{r["ci_low"]:+.5f}, {r["ci_high"]:+.5f}]</span></div>')
    tot = f'<div class="dr-dtot">{total_text}</div>' if total_text else ""
    return f'{tot}<div class="dr-stack">{"".join(segs)}</div><div class="dr-dleg">{"".join(legend)}</div>'


# ----------------------------------------------------------------------------
# Pantalla "En espera"
# ----------------------------------------------------------------------------
def wait_screen(title: str, badge: str, text: str = "") -> str:
    tx = f'<p class="mut">{esc(text)}</p>' if text else ""
    return f'<div class="dr-wait">{icon("clock")}<h3>{esc(title)}</h3><span class="bd">{esc(badge)}</span>{tx}</div>'


# ----------------------------------------------------------------------------
# CSS de botones-preset (nativos) con icono SVG como mascara
# ----------------------------------------------------------------------------
def preset_css(presets: Sequence[tuple[str, str]], active: str | None) -> str:
    """CSS para convertir ``st.button`` con ``key='preset_<id>'`` en tarjetas-icono.

    ``presets``: ``[(id, icon_name)]``. El preset activo (y solo ese) usa el rojo de marca.
    """
    css = [
        ".st-key-preset_row button{display:flex;flex-direction:column;align-items:center;gap:.35rem;height:auto;"
        "padding:.6rem .3rem !important;min-height:4.6rem;border:1.5px solid var(--line) !important;border-radius:12px !important;"
        "width:100%;background:var(--card) !important}",
        ".st-key-preset_row button p{font-size:.72rem !important;font-weight:700 !important;text-align:center;line-height:1.15}",
        ".st-key-preset_row button::before{content:'';display:block;width:1.7rem;height:1.7rem;background:var(--muted);"
        "-webkit-mask:var(--ic) center/contain no-repeat;mask:var(--ic) center/contain no-repeat}",
        ".st-key-preset_row button p, .st-key-preset_row button div{white-space:normal !important;overflow:visible !important;"
        "text-overflow:clip !important}",
    ]
    for pid, ic in presets:
        css.append(f".st-key-preset_{pid} button::before{{--ic:url('{icon_data_uri(ic)}')}}")
    if active:
        css.append(
            f".st-key-preset_{active} button{{border-color:var(--brand) !important;background:var(--brand-tint) !important;"
            "box-shadow:0 3px 10px rgba(206,17,38,.15) !important}"
            f".st-key-preset_{active} button::before{{background:var(--brand)}}")
    return "<style>" + "".join(css) + "</style>"


# ----------------------------------------------------------------------------
# Notas y callouts propios (sustituyen st.info / st.warning)
# ----------------------------------------------------------------------------
def note_box(text: str) -> str:
    """Nota neutra con borde izquierdo (texto literal)."""
    return f'<div class="dr-notebox">{esc(text)}</div>'


def callout(text: str) -> str:
    """Callout ambar suave, como el banner de veredicto (texto literal)."""
    return f'<div class="dr-callout">{icon("warn")}<span>{esc(text)}</span></div>'


# ----------------------------------------------------------------------------
# Leaderboard: tabla con barras en celda, histograma y tarjeta estampa
# ----------------------------------------------------------------------------
def lb_table(rows: Sequence[dict], lo: float, hi: float, dmax: float, selected: int | None = None) -> str:
    """Tabla HTML: Stuff+ con barra en celda (niveles neutros), Δ con ▲ ● ▼ y columna ± prevista como pendiente."""
    span = (hi - lo) or 1.0
    dmax = dmax or 1.0

    def pc(v: float) -> float:
        return max(2.0, min(100.0, 100.0 * (v - lo) / span))

    body = []
    for i, r in enumerate(rows):
        d = r.get("delta")
        tone = T.delta_tone(d)
        half = min(50.0, 50.0 * abs(d) / dmax) if d is not None and d == d else 0.0
        left = 50.0 if (d or 0) >= 0 else 50.0 - half
        dbar = f'<span class="dr-dbar"><i></i><b class="{tone}" style="left:{left:.1f}%;width:{half:.1f}%"></b></span>'
        cls = ' class="sel"' if selected == i else ""
        body.append(
            f"<tr{cls}><td class='mut'>{i + 1}</td><td><b>{esc(r['id'])}</b> <small>{esc(r['year'])} · {esc(r['hand'])}</small></td>"
            f"<td>{r['n']}</td>"
            f"<td class='dr-bc'><span class='v'>{fmt_num(r['neutral'])}</span><span class='dr-cellbar'><b class='dr-lvl' style='width:{pc(r['neutral']):.1f}%'></b></span></td>"
            f"<td class='dr-bc'><span class='v'>{fmt_num(r['cdmx'])}</span><span class='dr-cellbar'><b class='dr-lvl2' style='width:{pc(r['cdmx']):.1f}%'></b></span></td>"
            f"<td>{delta_badge(d, 2)}{dbar}</td>"
            f"<td><span class='dr-pend' title='Sin IC por pitcher hoy'>± pendiente</span></td>"
            f"<td>{fmt_num(r.get('pitching'))}</td><td>{fmt_num(r.get('rp100'), 2)}</td></tr>")
    head = ("<tr><th>#</th><th>Lanzador</th><th>N</th><th>Stuff+ mar</th><th>Stuff+ CDMX</th><th>ΔStuff+</th>"
            "<th>±</th><th>Pitching+ CDMX</th><th>Carr./100</th></tr>")
    legend = (f'<div class="dr-legend"><span class="loss">▼ pierde</span><span class="neu">● sin cambio '
              f'(|Δ| &lt; {T.DELTA_NEUTRAL_BAND:g})</span><span class="gain">▲ gana</span>'
              f'<span>± pendiente: aún no hay IC por pitcher</span></div>')
    return f'<div class="dr-tblwrap"><table class="dr-tbl"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>{legend}'


def stamp_head(name: str, sub: str, delta: float | None) -> str:
    return (f'<div class="dr-stamp-top"><div><div class="nm">{esc(name)}</div><div class="note">{esc(sub)}</div></div>'
            f'{delta_badge(delta, 2)}</div>')


def arsenal_chips(items: Sequence[tuple[str, float | None]]) -> str:
    chips = "".join(
        f'<span class="dr-chip">{icon("ball")}{esc(n)}'
        + (f'<small>{100 * u:.0f}%</small>' if u is not None and u == u else "") + "</span>"
        for n, u in items)
    return f'<div class="dr-chiprow">{chips}</div>'


def arsenal_bars(rows: Sequence[dict], lo: float, hi: float) -> str:
    """Mini-barras Stuff+ nivel del mar -> CDMX por pitcheo (dos barras por fila + simbolo semantico)."""
    span = (hi - lo) or 1.0

    def pc(v: float) -> float:
        return max(2.0, min(100.0, 100.0 * (v - lo) / span))

    out = []
    for r in rows:
        tone = T.delta_tone(r["delta"])
        out.append(
            f'<div class="dr-mbrow"><span class="dr-mbl">{esc(r["pitch"])}</span><span class="dr-mbt">'
            f'<b class="dr-lvl" style="width:{pc(r["sea"]):.1f}%"></b><b class="dr-lvl2" style="width:{pc(r["cdmx"]):.1f}%"></b></span>'
            f'<span class="dr-mbv">{fmt_num(r["sea"])} → {fmt_num(r["cdmx"])}</span>'
            f'<span class="dpill {tone}">{T.tone_symbol(tone)}</span></div>')
    leg = (f'<div class="dr-mmleg"><span><b style="color:{T.LVL_SEA}">━</b> mar</span>'
           f'<span><b style="color:{T.LVL_CDMX}">━</b> CDMX</span></div>')
    return "".join(out) + leg


def channel_rows(ch: dict, labels: dict) -> str:
    """Descomposicion por pitcher; None se muestra como "no estimable"."""
    rows = []
    for key, lab in labels.items():
        v = ch.get(key)
        txt = "no estimable" if v is None else f"{v:+.5f}".replace("-", "−")
        rows.append(f'<div class="dr-chrow"><span>{esc(lab)}</span><b>{txt}</b></div>')
    return "".join(rows)


# ----------------------------------------------------------------------------
# Sculptor
# ----------------------------------------------------------------------------
def counter_card(n: int, caption: str, chips: Sequence[str]) -> str:
    fl = "".join(f"<span>{esc(c)}</span>" for c in chips)
    return (f'<div class="dr-card"><div class="dr-h4">{icon("users")}Pares reales factibles</div>'
            f'<div class="dr-counter"><div class="n dr-num">{n}</div><div class="note" style="font-size:.8rem">{esc(caption)}</div></div>'
            f'<div class="dr-filt">{fl}</div></div>')


def range_bar(p25: float, p75: float, me: float | None, best: float | None) -> str:
    """Barra p25-p75 protagonista: "tu pitcheo" (navy, arriba) y "mejor par" (turquesa, abajo)."""
    vals = [v for v in (p25, p75, me, best) if v is not None]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.25, 1.0)
    lo, hi = lo - pad, hi + pad

    def pct(v: float) -> float:
        return 100.0 * (v - lo) / (hi - lo)

    parts = ['<div class="axis"></div>',
             f'<div class="iqr" style="left:{pct(p25):.1f}%;width:{pct(p75) - pct(p25):.1f}%"></div>']
    if me is not None:
        parts.append(f'<div class="mk me" style="left:{pct(me):.1f}%"><b class="dr-num">{me:.1f}</b>tu pitcheo</div>')
    if best is not None:
        parts.append(f'<div class="mk ref" style="left:{pct(best):.1f}%"><b class="dr-num">{best:.1f}</b>mejor par</div>')
    ticks = "".join(f"<span>{lo + (hi - lo) * k / 5:.1f}</span>" for k in range(6))
    return (f'<div class="dr-range">{"".join(parts)}<div class="ticks">{ticks}</div></div>'
            f'<div class="note" style="text-align:center">Franja p25 {p25:.1f} – p75 {p75:.1f} (Stuff+ CDMX de los pares)</div>')


def comparable_card(c: dict, same_pitcher: bool = False) -> str:
    tag = '<span class="tag">mismo lanzador, otra temporada</span>' if same_pitcher else ""
    return (f'<div class="dr-comp"><div><strong>{esc(c["pitcher_anon_id"])}</strong> <span class="mut">{esc(c["year"])}</span></div>'
            f'<div class="cbig dr-num">{c["stuff_plus_cdmx"]:.1f}</div>'
            f'<div class="mut" style="font-size:.72rem">Stuff+ CDMX · {c["n_pitches"]} pitcheos · {esc(c["pitch_id"])}</div>{tag}</div>')


def repertoire_rows(rep: Sequence[dict]) -> str:
    if not rep:
        return ""
    vals = [r["median_stuff_plus_cdmx"] for r in rep]
    lo, hi = min(vals) - 2, max(vals)
    out = []
    for r in rep[:6]:
        w = max(4.0, 100.0 * (r["median_stuff_plus_cdmx"] - lo) / ((hi - lo) or 1.0))
        out.append(f'<div class="dr-reprow"><span>{esc(r["pitch_id"])}</span><span class="dr-rt"><b style="width:{w:.1f}%"></b></span>'
                   f'<span class="dr-num" style="font-weight:800">{r["median_stuff_plus_cdmx"]:.1f}</span>'
                   f'<small>{r["n_pitches"]} lanz.</small></div>')
    return "".join(out)


# ----------------------------------------------------------------------------
# VeloVision (HACK-017, 17-C)
# ----------------------------------------------------------------------------
ICONS.update({
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "person": '<circle cx="12" cy="5" r="2.5"/><path d="M12 8v7M7 10l5-2 5 2M12 15l-3 6M12 15l3 6"/>',
    "ruler": '<path d="M3 15L15 3l6 6L9 21zM7 11l2 2M10 8l2 2M13 5l2 2"/>',
    "film": '<path d="M4 4h16v16H4zM8 4v16M16 4v16M4 9h4M4 15h4M16 9h4M16 15h4"/>',
    "lock": '<path d="M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3"/>',
    "check": '<path d="M4 12l5 5L20 6"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>',
    "frames": '<path d="M4 6h16M4 12h16M4 18h16M8 3v6M16 9v6M10 15v6"/>',
})


def disclaimer_badge(text: str) -> str:
    """Badge permanente de honestidad (texto literal)."""
    return f'<div class="dr-disclaimer" role="note">{icon("lock")}<span>{esc(text)}</span></div>'


def capture_guide(items: Sequence[tuple[str, str, str]]) -> str:
    """Guia de captura con iconos: ``[(icono, titulo, detalle)]``."""
    cells = "".join(
        f'<div class="dr-guide"><div class="dr-guide-ic">{icon(ic)}</div><div><b>{esc(t)}</b>'
        f'<div class="note">{esc(d)}</div></div></div>' for ic, t, d in items)
    return f'<div class="dr-guides">{cells}</div>'


def frame_strip(n_frames: int, suggested: int, candidates: Sequence[int], selected: int, confirmed: int | None = None) -> str:
    """Barra de frames: release sugerido (turquesa), candidatos (navy tenue), seleccionado y confirmado."""
    span = max(n_frames - 1, 1)
    pct = lambda i: 100.0 * i / span
    marks = "".join(f'<i class="cand" style="left:{pct(c):.2f}%" title="candidato: frame {c}"></i>' for c in candidates)
    marks += f'<i class="sug" style="left:{pct(suggested):.2f}%" title="sugerido: frame {suggested}"></i>'
    if confirmed is not None:
        marks += f'<i class="conf" style="left:{pct(confirmed):.2f}%" title="confirmado: frame {confirmed}"></i>'
    marks += f'<u style="left:{pct(selected):.2f}%"></u>'
    return (f'<div class="dr-fstrip">{marks}</div><div class="dr-zleg"><i style="background:rgba(11,122,102,.35)"></i>sugerido '
            f'<i style="background:rgba(31,58,95,.35)"></i>candidatos <i style="background:#1F3A5F"></i>seleccionado</div>')


def measure_card(icon_name: str, title: str, value: float | None, unit: str, error_text: str, badge: str = "",
                 decimals: int = 1) -> str:
    """Tarjeta de una medida del video; ``error_text`` va completo (sin prefijo), nunca se oculta el error."""
    v = "—" if value is None else f"{value:.{decimals}f}"
    b = f'<span class="tag">{esc(badge)}</span>' if badge else ""
    return (f'<div class="dr-card dr-measure"><div class="dr-h4">{icon(icon_name)}{esc(title)}</div>'
            f'<div class="v dr-num">{v}<small>{esc(unit)}</small></div>'
            f'<div class="note">{esc(error_text)}</div>{b}</div>')


def empty_state(icon_name: str, title: str, text: str = "") -> str:
    """Estado vacio elegante (sin la etiqueta 'en espera')."""
    tx = f'<p class="mut">{esc(text)}</p>' if text else ""
    return f'<div class="dr-empty">{icon(icon_name)}<h4>{esc(title)}</h4>{tx}</div>'


def family_ranges(families: Sequence[dict], min_pitchers: int) -> str:
    """Una fila por familia: barra p25-p75 (solida), p10-p90 (tenue), mediana, delta con simbolo y n_pitchers.

    Las familias en INSUFFICIENT_PEERS salen atenuadas y sin cifras.
    """
    ok = [f for f in families if f["status"] == "OK"]
    if ok:
        lo = min(f["stuff_plus_cdmx"]["p10"] for f in ok)
        hi = max(f["stuff_plus_cdmx"]["p90"] for f in ok)
        pad = max((hi - lo) * 0.08, 0.5)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = 0.0, 1.0
    span = (hi - lo) or 1.0
    pc = lambda v: max(0.0, min(100.0, 100.0 * (v - lo) / span))
    rows = []
    for f in families:
        if f["status"] != "OK":
            rows.append(f'<div class="dr-fam dim"><span class="nm">{esc(f["pitch_id"])}</span>'
                        f'<span class="mut">INSUFFICIENT_PEERS · {f["n_pitchers"]} pitchers (mínimo {min_pitchers}): sin cifras</span></div>')
            continue
        s, d = f["stuff_plus_cdmx"], f["delta_stuff_cdmx"]
        rows.append(
            f'<div class="dr-fam"><span class="nm">{esc(f["pitch_id"])}</span>'
            f'<span class="dr-frange"><b class="wide" style="left:{pc(s["p10"]):.1f}%;width:{pc(s["p90"]) - pc(s["p10"]):.1f}%"></b>'
            f'<b class="iqr" style="left:{pc(s["p25"]):.1f}%;width:{pc(s["p75"]) - pc(s["p25"]):.1f}%"></b>'
            f'<i style="left:{pc(s["p50"]):.1f}%"></i></span>'
            f'<span class="dr-num rng">{s["p50"]:.1f} <small>[{s["p25"]:.1f}–{s["p75"]:.1f}]</small></span>'
            f'<span>{delta_badge(d["p50"], 2)}</span><span class="mut n">{f["n_pitchers"]} pitchers</span></div>')
    leg = ('<div class="dr-legend"><span>barra sólida = p25–p75</span><span>franja tenue = p10–p90</span>'
           '<span>marca = mediana de Stuff+ CDMX</span><span>Δ = ΔStuff+ CDMX mediano (CDMX − mar)</span></div>')
    return f'<div class="dr-fams">{"".join(rows)}</div>{leg}'


def narrowing_table(rows: Sequence[dict]) -> str:
    """Antes/despues del ancho p25-p75 al agregar el radar (se reporta, no se exige mejora)."""
    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    body = []
    for r in rows:
        ch = r["change"]
        chs = "—" if ch is None else f"{ch:+.2f}".replace("-", "−")
        body.append(f'<tr><td><b>{esc(r["pitch_id"])}</b></td><td>{fmt(r["width_p25_p75_without_velo"])}</td>'
                    f'<td>{fmt(r["width_p25_p75_with_velo"])}</td><td>{chs}</td>'
                    f'<td>{r["n_pitchers_without_velo"]} → {r["n_pitchers_with_velo"]}</td></tr>')
    return ('<table class="dr-tbl"><thead><tr><th>Familia</th><th>Ancho p25–p75 sin radar</th><th>con radar</th>'
            f'<th>Cambio</th><th>Pitchers</th></tr></thead><tbody>{"".join(body)}</tbody></table>')

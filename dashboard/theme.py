"""Sistema de diseno de Diablos AeroVision (HACK-019, Fase 1).

Fuente unica de tokens de color, CSS global y plantilla Plotly. Referencia visual:
``docs/mockups/hack019_mockup.html`` (mockup v2, modo claro sport-tech, aprobado).

Reglas de color (no negociables)
--------------------------------
* ``BRAND`` (#CE1126) va SOLO en la marca: logo, barra superior, pestana activa, boton primario y
  preset activo. Nunca en datos, barras, celdas, escalas ni SHAP.
* Semanticos de datos: pierde ``LOSS``, gana ``GAIN``, neutro ``NEUTRAL``, marcador "tu pitcheo" ``NAVY``.
* Cada estado semantico va acompanado de un simbolo (▲ ● ▼) para no depender solo del color.

Contraste WCAG 2.x medido sobre ``CARD`` (#FFFFFF) (se recalcula en ``tests/test_dashboard_design.py``):

====================  =========  ========  ======================================
Token                 Hex        Contraste Uso
====================  =========  ========  ======================================
TEXT                  #14181F    17.79:1   texto principal
MUTED / NEUTRAL       #5B6470     6.00:1   texto secundario, estado neutro
LOSS                  #B83A28     5.72:1   pierde (texto)
GAIN                  #0B7A66     5.26:1   gana (texto)
NAVY                  #1F3A5F    11.48:1   marcador "tu pitcheo"
LVL_CDMX              #3E5C7A     6.96:1   nivel en CDMX
CAT_STUFF             #2F6FB5     5.17:1   categoria Canal Stuff
CAT_PARK              #7A52C7     5.44:1   categoria Canal Parque
LVL_SEA               #7C8999     3.56:1   SOLO elemento grafico (>= 3:1), nunca texto
BRAND                 #CE1126     5.63:1   marca (no es dato)
====================  =========  ========  ======================================

Los pares texto/fondo tintado (LOSS sobre LOSS_BG, GAIN sobre GAIN_BG, NEUTRAL sobre NEU_BG,
WARN_TEXT sobre WARN_BG) tambien superan 4.5:1.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Tokens (mismo hex que el mockup)
# ----------------------------------------------------------------------------
PAGE = "#F6F7F9"
CARD = "#FFFFFF"
SOFT = "#F0F2F5"
LINE = "#E3E6EB"
TEXT = "#14181F"
MUTED = "#5B6470"

BRAND = "#CE1126"          # SOLO marca (logo, pestana activa, boton primario, preset activo)
BRAND_TINT = "#FFF3F4"     # fondo del preset activo

LOSS = "#B83A28"           # pierde (coral)
NEUTRAL = "#5B6470"        # promedio / sin cambio (gris)
GAIN = "#0B7A66"           # gana (turquesa)
LOSS_BG = "#FCEBE7"
GAIN_BG = "#E3F5F1"
NEU_BG = "#EEF0F3"

NAVY = "#1F3A5F"           # marcador "tu pitcheo"
LVL_SEA = "#7C8999"        # nivel del mar (elemento grafico)
LVL_CDMX = "#3E5C7A"       # nivel CDMX / densidad
DENSITY = "#3E5C7A"
CAT_STUFF = "#2F6FB5"      # categoria Canal Stuff
CAT_PARK = "#7A52C7"       # categoria Canal Parque

WARN_BG = "#FFF6E0"
WARN_LINE = "#F2C14E"
WARN_TEXT = "#5C3D00"

R_PX = 14
FONT_TEXT = "'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_NUM = "'Barlow Condensed', 'Inter', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Consolas, monospace"

#: Tokens que se usan como TEXTO y deben cumplir >= 4.5:1 sobre CARD (verificado por test).
TEXT_TOKENS = {"TEXT": TEXT, "MUTED": MUTED, "LOSS": LOSS, "GAIN": GAIN, "NEUTRAL": NEUTRAL,
               "NAVY": NAVY, "LVL_CDMX": LVL_CDMX, "CAT_STUFF": CAT_STUFF, "CAT_PARK": CAT_PARK}
#: Tokens semanticos de DATOS (nunca el rojo de marca).
DATA_COLORS = {"LOSS": LOSS, "GAIN": GAIN, "NEUTRAL": NEUTRAL, "NAVY": NAVY, "LVL_SEA": LVL_SEA,
               "LVL_CDMX": LVL_CDMX, "CAT_STUFF": CAT_STUFF, "CAT_PARK": CAT_PARK}

#: Umbral VISUAL para pintar un delta como neutro (●). No es un umbral estadistico ni una conclusion.
DELTA_NEUTRAL_BAND = 0.25


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str = CARD) -> float:
    """Razon de contraste WCAG 2.x entre dos colores hex."""
    hi, lo = sorted((relative_luminance(fg), relative_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def delta_tone(x: float | None, band: float = DELTA_NEUTRAL_BAND) -> str:
    """'gain' | 'loss' | 'neu' para un delta (mas alto = mejor para el pitcher)."""
    if x is None or x != x:
        return "neu"
    return "gain" if x > band else "loss" if x < -band else "neu"


def tone_color(tone: str) -> str:
    return {"gain": GAIN, "loss": LOSS}.get(tone, NEUTRAL)


def tone_symbol(tone: str) -> str:
    return {"gain": "▲", "loss": "▼"}.get(tone, "●")


# ----------------------------------------------------------------------------
# Escalas de color de datos (sin rojo de marca)
# ----------------------------------------------------------------------------
DIVERGING_SCALE = [[0.0, LOSS], [0.5, "#F2F3F5"], [1.0, GAIN]]
SEQUENTIAL_SCALE = [[0.0, "#F2F5F8"], [0.5, "#9FB1C5"], [1.0, LVL_CDMX]]


# ----------------------------------------------------------------------------
# Plantilla Plotly (registrada como default)
# ----------------------------------------------------------------------------
TEMPLATE_NAME = "diablos"


def plotly_template():
    import plotly.graph_objects as go

    return go.layout.Template(layout=dict(
        font=dict(family=FONT_TEXT, color=TEXT, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD,
        colorway=[LVL_CDMX, LVL_SEA, CAT_STUFF, CAT_PARK, NEUTRAL, GAIN, NAVY],
        margin=dict(l=44, r=16, t=44, b=40),
        title=dict(font=dict(size=14, color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LVL_SEA, linecolor=LINE, tickfont=dict(color=MUTED, size=11),
                   title=dict(font=dict(color=MUTED, size=11))),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LVL_SEA, linecolor=LINE, tickfont=dict(color=MUTED, size=11),
                   title=dict(font=dict(color=MUTED, size=11))),
        hoverlabel=dict(bgcolor=CARD, font=dict(color=TEXT, family=FONT_TEXT), bordercolor=LINE),
        colorscale=dict(diverging=DIVERGING_SCALE, sequential=SEQUENTIAL_SCALE),
    ))


def register_plotly_template() -> None:
    """Registra la plantilla propia y la deja como default de Plotly (idempotente)."""
    import plotly.io as pio

    pio.templates[TEMPLATE_NAME] = plotly_template()
    pio.templates.default = TEMPLATE_NAME


# ----------------------------------------------------------------------------
# CSS global (se inyecta una sola vez por ejecucion desde app.py)
# ----------------------------------------------------------------------------
FONTS_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700"
    "&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');"
)


def _tokens_css() -> str:
    return f"""
:root{{
  --page:{PAGE}; --card:{CARD}; --soft:{SOFT}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
  --brand:{BRAND}; --brand-tint:{BRAND_TINT};
  --loss:{LOSS}; --neutral:{NEUTRAL}; --gain:{GAIN};
  --loss-bg:{LOSS_BG}; --gain-bg:{GAIN_BG}; --neu-bg:{NEU_BG};
  --navy:{NAVY}; --lvl-sea:{LVL_SEA}; --lvl-cdmx:{LVL_CDMX}; --density:{DENSITY};
  --cat-stuff:{CAT_STUFF}; --cat-park:{CAT_PARK};
  --warn-bg:{WARN_BG}; --warn-line:{WARN_LINE}; --warn-text:{WARN_TEXT};
  --f-txt:{FONT_TEXT}; --f-num:{FONT_NUM}; --f-mono:{FONT_MONO};
  --sh:0 1px 2px rgba(16,24,40,.06),0 6px 18px rgba(16,24,40,.06); --r:14px;
}}
"""


_CHROME_CSS = """
/* ---- ocultar el chrome de Streamlit (Deploy, menu, footer, toasts) ---- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
[data-testid="stAppDeployButton"], [data-testid="stToastContainer"], [data-testid="stToast"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{display:none !important}
"""

_BASE_CSS = """
html:root{font-size:clamp(9px,min(0.96vw,1.55vh),20px) !important}
html,body,.stApp{background:var(--page) !important;color:var(--text);font-family:var(--f-txt) !important}
.stApp{border-top:.25rem solid var(--brand)}
.stApp *{font-variant-numeric:tabular-nums}
[data-testid="stMainBlockContainer"], .block-container{
  max-width:none !important;padding:.35rem 1.4rem .8rem !important;position:relative}
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"]{background:transparent}
h1,h2,h3,h4{font-family:var(--f-txt) !important;letter-spacing:-.01em;color:var(--text)}
[data-testid="stVerticalBlock"]{gap:.6rem}
[data-testid="stMarkdownContainer"] p{margin:0}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] :last-child{margin-bottom:0 !important}
.stMarkdown, [data-testid="stCaptionContainer"]{color:var(--muted)}
[data-testid="stCaptionContainer"]{font-size:.72rem;line-height:1.4}
.dr-num{font-family:var(--f-num);font-variant-numeric:tabular-nums}
.mut{color:var(--muted)} .gain{color:var(--gain)} .loss{color:var(--loss)} .neu{color:var(--neutral)}
small{font-size:.75em;color:var(--muted);font-weight:500;margin-left:.2em}
.ic{width:1.15em;height:1.15em;fill:none;stroke:currentColor;stroke-width:1.9;stroke-linecap:round;
  stroke-linejoin:round;flex:none;vertical-align:-.2em}

/* ---- pestanas (nav): la lista se eleva a la fila del wordmark ---- */
[data-testid="stTabs"] [role="tablist"]{gap:.2rem !important;background:transparent !important;border:0 !important;
  box-shadow:none !important;position:absolute;top:.75rem;left:17.5rem;z-index:5;width:auto !important;
  overflow:visible !important}
.react-aria-SelectionIndicator{display:none !important}
[data-testid="stTab"]{height:auto !important;padding:.45rem .85rem !important;border-radius:99px !important;
  background:transparent !important;color:var(--muted) !important;border:0 !important;cursor:pointer}
[data-testid="stTab"] p{font-size:.88rem !important;font-weight:600 !important;color:inherit !important}
[data-testid="stTab"]:hover{background:var(--soft) !important;color:var(--text) !important}
[data-testid="stTab"][aria-selected="true"]{background:var(--card) !important;color:var(--text) !important;
  box-shadow:var(--sh);outline:2px solid var(--brand);outline-offset:-2px}
[data-testid="stTabs"] > div{border-bottom:0 !important}
[data-testid="stTabs"]{margin-top:-.2rem}

/* ---- encabezado ---- */
.dr-wm{display:flex;align-items:center;gap:.6rem;font-weight:800;font-size:1.35rem;letter-spacing:-.01em;
  color:var(--text);height:2.6rem}
.dr-wm .d{width:2.1rem;height:2.1rem;border-radius:.6rem;background:var(--brand);color:#fff;display:grid;
  place-items:center;font-weight:800;font-size:1.3rem;box-shadow:0 3px 8px rgba(206,17,38,.3)}
.dr-wm span{font-weight:500;color:var(--muted)}
.dr-crumb{display:inline-flex;align-items:center;gap:.5rem;background:var(--card);border:1px solid var(--line);
  border-radius:99px;padding:.32rem .9rem;font-size:.82rem;box-shadow:var(--sh);color:var(--text);white-space:nowrap}
.dr-crumb b{font-weight:700} .dr-crumb i{width:1px;height:1rem;background:var(--line)}
.dr-crumb .src{font-size:.62rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
  background:var(--soft);border-radius:99px;padding:.05rem .45rem}
.dr-anc{display:flex;gap:.4rem;align-items:center;flex-wrap:nowrap}
.dr-ap{display:inline-flex;align-items:center;gap:.35rem;font-size:.76rem;background:var(--card);
  border:1px solid var(--line);border-radius:99px;padding:.28rem .7rem;font-weight:600;white-space:nowrap}
.dr-ap b{font-weight:800;font-size:.86rem}
.dr-ap .ic{color:var(--gain)}
.dr-pill{display:inline-flex;align-items:center;gap:.3rem;font-size:.7rem;font-weight:700;border-radius:99px;
  padding:.15rem .6rem;background:var(--soft);color:var(--muted);white-space:nowrap}

/* ---- botones ---- */
.stButton > button, [data-testid="stBaseButton-secondary"]{border-radius:12px !important;border:1px solid var(--line) !important;
  background:var(--card) !important;color:var(--text) !important;font-weight:700 !important;font-size:.84rem !important;
  min-height:0 !important;padding:.5rem .9rem !important;box-shadow:none}
.stButton > button:hover{border-color:var(--lvl-sea) !important;color:var(--text) !important}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{background:var(--brand) !important;
  border-color:var(--brand) !important;color:#fff !important;box-shadow:0 4px 12px rgba(206,17,38,.28)}
.stButton > button p{font-size:.84rem !important;font-weight:700 !important}

/* ---- tarjetas ---- */
.dr-card{color:var(--text);background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh);
  padding:.9rem 1.1rem;min-width:0}
.dr-h4{display:flex;align-items:center;gap:.4rem;font-size:.78rem;font-weight:700;color:var(--muted);
  letter-spacing:.04em;text-transform:uppercase}
.bd{font-family:var(--f-mono);font-size:.6rem;color:var(--muted);border:1px solid var(--line);background:var(--soft);
  padding:.1rem .45rem;border-radius:.3rem;white-space:nowrap;display:inline-block}
.bds{display:flex;gap:.35rem;flex-wrap:wrap}
.tag{font-size:.6rem;font-weight:600;color:var(--text);border:1px solid var(--line);background:var(--soft);
  padding:.05rem .4rem;border-radius:99px;display:inline-block}
.note{font-size:.7rem;color:var(--muted);line-height:1.4}
.note.ill{border-left:3px solid var(--warn-line);padding-left:.5rem}
.dpill{display:inline-flex;align-items:center;gap:.25rem;font-weight:800;font-size:.9rem;padding:.2rem .65rem;
  border-radius:99px}
.dpill.loss{background:var(--loss-bg);color:var(--loss)} .dpill.gain{background:var(--gain-bg);color:var(--gain)}
.dpill.neu{background:var(--neu-bg);color:var(--neutral)}

/* ---- veredicto ---- */
.dr-verdict{border:1px solid var(--warn-line);background:var(--warn-bg);color:var(--warn-text);border-radius:var(--r);
  padding:.7rem 1.1rem;display:flex;gap:.8rem;align-items:flex-start}
.dr-verdict.gain{background:var(--gain-bg);border-color:#7CCFBE;color:#0B4A3E}
.dr-verdict.neu{background:var(--neu-bg);border-color:var(--line);color:var(--text)}
.dr-verdict .ic{width:1.6rem;height:1.6rem;margin-top:.1rem;flex:none}
.dr-verdict p{font-size:.95rem;line-height:1.4;margin:0}
.dr-verdict .ref{margin-top:.35rem;font-size:.88rem;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.dr-verdict .tag{background:rgba(255,255,255,.7);border-color:var(--warn-line);color:var(--warn-text)}

/* ---- tarjetas grandes del Simulador ---- */
.dr-big{display:flex;flex-direction:column;gap:.6rem;justify-content:space-between;height:100%}
.dr-big .hd{display:flex;align-items:center;gap:.5rem;font-weight:800;font-size:1rem}
.dr-big .hd .dr-pill{margin-left:auto}
.dr-stf{display:flex;align-items:flex-end;gap:.4rem .8rem;flex-wrap:wrap}
.dr-stf .n{font-family:var(--f-num);font-weight:700;font-size:6.6rem;line-height:.85;letter-spacing:-.01em}
.dr-stf .l{font-size:.78rem;font-weight:700;color:var(--muted);margin-bottom:.4rem}
.dr-stf .dpill{font-size:1.3rem;margin-bottom:.35rem}
.dr-mini2{display:grid;grid-template-columns:1fr 1fr;gap:.6rem}
.dr-mini{background:var(--soft);border-radius:12px;padding:.7rem .9rem;display:flex;flex-direction:column;gap:.2rem}
.dr-mini .t{font-size:.82rem;font-weight:700;color:var(--muted);display:flex;gap:.35rem;align-items:center}
.dr-mini .v{font-size:2.4rem;font-weight:800;letter-spacing:-.02em;line-height:1}
.dr-mini .v small{font-size:.45em}
.dr-gauge{display:flex;flex-direction:column;gap:.4rem}
.dr-gt{display:flex;justify-content:space-between;font-size:.82rem;font-weight:600;color:var(--muted)}
.dr-gt b{font-weight:800}
.dr-gtrack{position:relative;height:.7rem;background:linear-gradient(90deg,var(--loss-bg),var(--neu-bg) 50%,var(--gain-bg));
  border-radius:99px;border:1px solid var(--line)}
.dr-gtrack i{position:absolute;top:-.25rem;bottom:-.25rem;width:2px;background:var(--muted)}
.dr-gtrack u{position:absolute;top:-.3rem;width:1.1rem;height:1.1rem;margin-left:-.55rem;border-radius:50%;
  background:var(--navy);border:3px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.35)}
.dr-chips2{display:flex;gap:.5rem;flex-wrap:wrap}
.dr-chips2 span{font-size:.85rem;font-weight:600;color:var(--muted);border:1px solid var(--line);border-radius:99px;
  padding:.15rem .65rem}
.dr-chips2 b{color:var(--text)}
.dr-diag{display:flex;flex-direction:column;gap:.3rem;height:100%}
.dr-diag svg.dg{flex:1;min-height:0;width:100%}
svg .gl{stroke:#E3E6EB;stroke-width:.7} svg .ax{fill:var(--muted);font-size:10px}
svg .ax2{fill:var(--muted);font-size:10.5px;font-weight:600}
svg .lb{fill:var(--text);font-size:12px;font-weight:700}
svg .lb2{font-size:12.5px;font-weight:800}
svg .comp{stroke:#B9C0CA;stroke-width:1.2;stroke-dasharray:3 3}

/* ---- slider + banda Zona LMB ---- */
.dr-slh{display:flex;align-items:center;gap:.4rem;font-size:.9rem;font-weight:600;color:var(--muted);margin-bottom:-1rem}
.dr-slh span{flex:1}
.dr-slh b{font-size:1.4rem;color:var(--text);font-weight:800}
.dr-slh b small{margin-left:.2rem}
.dr-track{height:.5rem;background:#E7EAEF;border-radius:99px;position:relative;margin:-1.85rem .375rem .45rem;
  pointer-events:none;z-index:0}
.dr-track .zona{position:absolute;top:-.2rem;bottom:-.2rem;background:rgba(11,122,102,.16);
  border:1px solid rgba(11,122,102,.55);border-radius:.3rem}
.dr-zleg{display:flex;align-items:center;gap:.4rem;font-size:.68rem;color:var(--muted);margin-bottom:.25rem}
.dr-zleg i{width:1.4rem;height:.6rem;background:rgba(11,122,102,.16);border:1px solid rgba(11,122,102,.55);
  border-radius:.2rem;display:inline-block}

/* ---- barras de analista ---- */
.dr-stack{display:flex;height:1.5rem;border-radius:.4rem;overflow:hidden;border:1px solid var(--line);background:var(--soft)}
.dr-stack b{display:block;height:100%}
.dr-s1{background:var(--cat-stuff)} .dr-s2{background:var(--cat-park)}
.dr-s3{background:repeating-linear-gradient(135deg,var(--neutral) 0 .28rem,#fff .28rem .56rem);border-left:1px solid var(--neutral)}
.dr-sw{display:inline-block;width:.9rem;height:.9rem;border-radius:.25rem;flex:none;border:1px solid var(--line)}
.dr-dleg{display:flex;flex-direction:column;gap:.2rem;font-size:.78rem;margin-top:.4rem}
.dr-dleg div{display:flex;gap:.5rem;align-items:center}
.dr-dleg em{font-style:normal;font-weight:800}
.dr-dtot{display:flex;align-items:baseline;gap:.7rem;margin-bottom:.4rem}
.dr-dtot .num{font-size:1.5rem;font-weight:800}
.dr-wait{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.8rem;color:var(--muted);padding:6rem 0}
.dr-wait h3{font-size:2.4rem;color:var(--text);font-weight:800}
.dr-wait .ic{width:3.5rem;height:3.5rem}


/* ---- riel del Simulador ---- */
.st-key-rail{background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh);
  padding:.7rem 1rem !important;gap:.2rem !important}
.dr-zwrap{display:flex;flex-direction:column;gap:.2rem;padding-bottom:.6rem}
.dr-rail-h{display:flex;align-items:center;gap:.5rem;font-size:1.05rem;font-weight:800;color:var(--text)}
.st-key-in_rel_speed [data-testid="stSliderThumbValue"], .st-key-in_rel_speed [data-testid="stSliderTickBar"],
.st-key-in_spin_rate [data-testid="stSliderThumbValue"], .st-key-in_spin_rate [data-testid="stSliderTickBar"],
.st-key-in_ivb [data-testid="stSliderThumbValue"], .st-key-in_ivb [data-testid="stSliderTickBar"],
.st-key-in_hb_norm [data-testid="stSliderThumbValue"], .st-key-in_hb_norm [data-testid="stSliderTickBar"],
.st-key-in_arm_slot [data-testid="stSliderThumbValue"], .st-key-in_arm_slot [data-testid="stSliderTickBar"]{display:none !important}
[data-testid="stSlider"]{padding-bottom:0 !important}
.st-key-rail [data-testid="stSlider"] > div{padding-top:0 !important}
.st-key-btn_peers button{width:100%}


/* ---- tarjetas del Simulador a la misma altura ---- */
.st-key-cards [data-testid="stHorizontalBlock"]{align-items:stretch}
.st-key-cards [data-testid="stColumn"], .st-key-cards [data-testid="stColumn"] > div,
.st-key-cards [data-testid="stElementContainer"], .st-key-cards [data-testid="stMarkdown"],
.st-key-cards [data-testid="stMarkdownContainer"]{height:100%}
.st-key-cards .dr-card{height:100%;min-height:34rem}
.st-key-simright{gap:.8rem !important}

/* ---- Fase 3: notas, callouts, Leaderboard, Sculptor ---- */
.dr-notebox{border-left:3px solid var(--lvl-sea);background:var(--soft);padding:.6rem .9rem;
  border-radius:0 12px 12px 0;font-size:.85rem;color:var(--text);line-height:1.45}
.dr-callout{border:1px solid var(--warn-line);background:var(--warn-bg);color:var(--warn-text);border-radius:var(--r);
  padding:.6rem 1rem;display:flex;gap:.7rem;align-items:flex-start;font-size:.88rem;line-height:1.45}
.dr-callout .ic{flex:none;width:1.3rem;height:1.3rem}
.dr-dg{display:flex;align-items:center;gap:.7rem}
.dr-dg .n{font-size:3rem;font-weight:800;line-height:1}
.st-key-lbfilters, .st-key-lbtable, .st-key-lbhist, .st-key-stamp, .st-key-peers_panel{background:var(--card);
  border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh);padding:.8rem 1rem !important}
.st-key-stamp{background:linear-gradient(180deg,#fff,#F3F5F8)}
.dr-tblwrap{max-height:30rem;overflow:auto}
table.dr-tbl{width:100%;border-collapse:collapse;font-size:.9rem}
table.dr-tbl th{position:sticky;top:0;background:var(--card);font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);font-weight:700;text-align:left;padding:.3rem .4rem;border-bottom:1px solid var(--line);white-space:nowrap}
table.dr-tbl td{padding:.5rem .4rem;border-bottom:1px solid var(--line);vertical-align:middle;color:var(--text)}
table.dr-tbl tr.sel td{background:#EEF3FA} table.dr-tbl tr.sel td:first-child{box-shadow:inset .25rem 0 0 var(--navy)}
.dr-bc .v{display:inline-block;width:3.2rem;font-weight:700}
.dr-cellbar{display:inline-block;width:4.2rem;height:.55rem;background:var(--soft);border-radius:3px;vertical-align:middle}
.dr-cellbar b{display:block;height:100%;border-radius:3px}
.dr-lvl{background:var(--lvl-sea)} .dr-lvl2{background:var(--lvl-cdmx)}
.dr-dbar{display:inline-block;position:relative;width:4.2rem;height:.55rem;background:var(--soft);border-radius:3px;
  vertical-align:middle;margin-left:.5rem}
.dr-dbar i{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)}
.dr-dbar b{position:absolute;top:0;bottom:0;border-radius:3px}
.dr-dbar b.loss{background:var(--loss)} .dr-dbar b.gain{background:var(--gain)} .dr-dbar b.neu{background:var(--neutral)}
.dr-pend{font-family:var(--f-mono);font-size:.6rem;color:var(--muted);text-align:center;border:1px dashed #B9C0CA;
  border-radius:.3rem;white-space:nowrap;padding:.05rem .3rem}
.dr-legend{display:flex;gap:1rem;font-size:.72rem;color:var(--muted);flex-wrap:wrap;margin-top:.4rem}
.dr-stamp-top{display:flex;justify-content:space-between;align-items:center;gap:.5rem}
.dr-stamp-top .nm{font-weight:800;font-size:1.45rem;letter-spacing:-.02em;color:var(--text)}
.dr-chiprow{display:flex;gap:.35rem;flex-wrap:wrap;margin:.5rem 0}
.dr-chip{font-size:.76rem;font-weight:700;border:1px solid var(--line);border-radius:99px;padding:.2rem .65rem;background:#fff;
  display:inline-flex;align-items:center;gap:.3rem;color:var(--text)}
.dr-mbrow{display:grid;grid-template-columns:5.2rem 1fr 6.6rem 2.6rem;gap:.4rem;align-items:center;font-size:.85rem;padding:.3rem 0}
.dr-mbl{font-weight:700;color:var(--text)}
.dr-mbt{position:relative;height:.8rem;display:block}
.dr-mbt b{position:absolute;left:0;height:.34rem;border-radius:3px;display:block}
.dr-mbt b.dr-lvl{top:0} .dr-mbt b.dr-lvl2{top:.42rem}
.dr-mbv{font-weight:800;text-align:right;color:var(--text)}
.dr-mmleg{display:flex;gap:.8rem;font-size:.7rem;color:var(--muted);align-items:center}
.dr-chrow{display:flex;justify-content:space-between;font-size:.85rem;padding:.25rem 0;border-bottom:1px dashed var(--line);color:var(--text)}
.dr-chrow b{font-weight:800}
.dr-counter{display:flex;align-items:center;gap:.9rem;margin:.4rem 0}
.dr-counter .n{font-weight:800;font-size:6rem;line-height:.9;letter-spacing:-.03em;color:var(--navy)}
.dr-filt{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.5rem}
.dr-filt span{font-size:.8rem;font-weight:600;border:1px solid var(--line);background:var(--soft);border-radius:99px;padding:.15rem .6rem;color:var(--text)}
.dr-range{position:relative;height:14rem;margin:.5rem 1.2rem 0}
.dr-range .axis{position:absolute;left:0;right:0;top:6.1rem;height:.5rem;background:#E7EAEF;border-radius:9px}
.dr-range .iqr{position:absolute;top:5.5rem;height:1.7rem;background:var(--lvl-cdmx);border-radius:.5rem;opacity:.9}
.dr-range .mk{position:absolute;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;
  font-size:.9rem;font-weight:700;white-space:nowrap}
.dr-range .mk b{font-size:2.8rem;font-weight:800;line-height:1}
.dr-range .mk.me{top:0;color:var(--navy);height:6.3rem;justify-content:flex-end}
.dr-range .mk.me::after{content:"";width:3px;height:1.2rem;background:currentColor;border-radius:2px;margin-top:.1rem}
.dr-range .mk.ref{top:6.7rem;color:var(--gain)}
.dr-range .mk.ref::before{content:"";width:3px;height:1.2rem;background:currentColor;border-radius:2px;margin-bottom:.1rem}
.dr-range .ticks{position:absolute;left:0;right:0;bottom:0;display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted)}
.dr-comp{background:var(--soft);border-radius:12px;padding:.8rem 1rem;display:grid;grid-template-columns:1fr auto;
  align-items:center;gap:0 .6rem;margin-bottom:.5rem;color:var(--text)}
.dr-comp .cbig{font-size:2.6rem;font-weight:800;grid-row:span 3;color:var(--text)}
.dr-reprow{display:grid;grid-template-columns:5rem 1fr 3rem 4.4rem;gap:.5rem;align-items:center;font-size:.92rem;font-weight:600;
  padding:.35rem 0;color:var(--text)}
.dr-rt{height:.8rem;background:var(--soft);border-radius:3px;display:block}.dr-rt b{display:block;height:100%;background:var(--lvl-cdmx);border-radius:3px}
.dr-coach{border:2px dotted #98A1AE;border-radius:var(--r);padding:.8rem 1rem;background:#FAFBFC;color:var(--text)}

/* ---- pastillas (st.pills / segmented): el rojo primario no se usa; activo = navy ---- */
button[data-variant="pills"], button[data-variant="segmented_control"]{border-radius:99px !important;
  border:1px solid var(--line) !important;background:var(--card) !important;color:var(--text) !important;
  min-height:0 !important;padding:.25rem .75rem !important}
button[data-variant="pills"] p, button[data-variant="segmented_control"] p{font-size:.78rem !important;font-weight:600 !important}
button[data-variant="pills"][aria-checked="true"], button[data-variant="pills"][aria-pressed="true"],
button[data-variant="segmented_control"][aria-checked="true"]{border-color:var(--navy) !important;color:var(--navy) !important;
  background:#EEF3FA !important}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] *{color:var(--text) !important;font-weight:700}
table.dr-tbl td small{white-space:nowrap}
.dr-legend{margin-bottom:.5rem}

[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child{background:var(--navy) !important;
  border-color:var(--navy) !important}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div{background:#fff !important}
[data-testid="stCheckbox"][data-selected="true"] label > div:not([data-testid]):first-of-type,
[data-testid="stCheckbox"][data-selected="true"] label > span + div{background:var(--navy) !important;border-color:var(--navy) !important}
h1,h2,h3,h4,[data-testid="stMarkdownContainer"] h3{color:var(--text) !important}

/* ---- widgets nativos: neutralizar el primario (rojo = solo marca) ---- */
[data-testid="stSlider"] [data-orientation="horizontal"][style*="relative"] > div:first-child{
  background:#DDE1E7 !important;height:.45rem !important;border-radius:99px !important}
.st-key-in_rel_speed [data-orientation="horizontal"][style*="relative"] > div:first-child,
.st-key-in_spin_rate [data-orientation="horizontal"][style*="relative"] > div:first-child,
.st-key-in_ivb [data-orientation="horizontal"][style*="relative"] > div:first-child,
.st-key-in_hb_norm [data-orientation="horizontal"][style*="relative"] > div:first-child,
.st-key-in_arm_slot [data-orientation="horizontal"][style*="relative"] > div:first-child{
  background:transparent !important}
[data-testid="stSlider"] [data-orientation="horizontal"][style*="relative"] > div:nth-child(2){
  background:#fff !important;border:3px solid var(--navy) !important;width:1.1rem !important;height:1.1rem !important;
  box-sizing:border-box;box-shadow:0 2px 5px rgba(0,0,0,.25) !important;z-index:2}
[data-testid="stSlider"]{position:relative;z-index:1}
[data-baseweb="tag"]{background:var(--navy) !important}
[data-baseweb="checkbox"] > div:first-child{border-color:var(--navy) !important}
[data-baseweb="checkbox"][aria-checked="true"] > div:first-child, label[data-baseweb="checkbox"] input:checked + div{
  background:var(--navy) !important;border-color:var(--navy) !important}
[data-baseweb="radio"] input:checked + div{border-color:var(--navy) !important;background:var(--navy) !important}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"]{border-radius:10px !important}
[data-testid="stExpander"] details{border:1px solid var(--line);border-radius:var(--r);background:var(--card);box-shadow:var(--sh)}
[data-testid="stExpander"] summary{font-weight:700;font-size:.9rem}
[data-testid="stExpander"] summary p{font-weight:700;font-size:.9rem}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.6rem .9rem}
[data-testid="stMetricValue"]{font-weight:800}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stAlert"]{border-radius:12px}
label, [data-testid="stWidgetLabel"] p{color:var(--muted);font-size:.8rem;font-weight:600}
"""


def build_css() -> str:
    """CSS global completo (tokens + chrome + base)."""
    return f"<style>{FONTS_IMPORT}{_tokens_css()}{_CHROME_CSS}{_BASE_CSS}</style>"


def inject_global_css() -> None:
    """Inyecta el CSS global UNA sola vez por ejecucion del script y registra la plantilla Plotly."""
    import streamlit as st

    register_plotly_template()
    run = st.session_state.get("_dr_run_counter", 0)
    if st.session_state.get("_dr_css_done") == run and run:
        return
    st.session_state["_dr_css_done"] = run
    st.markdown(build_css(), unsafe_allow_html=True)


def begin_run() -> None:
    """Marca el inicio de una ejecucion; con ``inject_global_css`` garantiza una inyeccion por run."""
    import streamlit as st

    st.session_state["_dr_run_counter"] = st.session_state.get("_dr_run_counter", 0) + 1

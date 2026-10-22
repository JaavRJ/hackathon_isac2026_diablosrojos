"""Portada "Inicio" de Diablos AeroVision (HACK-021).

Primera pestana de la app. Explica que es el producto, por que es una app y no solo un modelo, y
lleva con botones a las 5 herramientas. Reglas:

* Ninguna cifra de resultados esta escrita aqui: todo se calcula en ``home_figures`` a partir de
  ``reports/**/*.csv`` y ``models/*.joblib`` (rutas relativas, sin parquet).
* Si falta un CSV, la cifra correspondiente se omite y la portada sigue funcionando.
* Rojo de marca solo en el logo y el boton primario; los datos usan navy, turquesa y gris.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard import components as C

# ----------------------------------------------------------------------------
# Cifras ancla (todas derivadas de los CSV / modelos)
# ----------------------------------------------------------------------------
SRC_ARSENAL = "pitcher_arsenal_breakdown.csv"
SRC_LEADERBOARD = "pitcher_season_leaderboard.csv"
SRC_HOLDOUT = "holdout_metrics.csv"
SRC_STABILITY = "year_to_year_stability.csv"
SRC_MODEL = "stuff_model.joblib"

#: Familias que se consideran para "la que mas pierde" cuando la Four-Seam no esta en el catalogo.
PRIMARY_FAMILY = "Four-Seam"


def _frame(reports: dict | None, key: str) -> pd.DataFrame:
    df = (reports or {}).get(key)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _rho(reports: dict | None, meta: dict | None) -> tuple[float | None, str]:
    """rho de CDMX: la del modelo (``rho_extreme_median_``) o, si no hay metadatos, ``rho_cdmx_used`` del arsenal."""
    if meta and meta.get("rho_cdmx") is not None:
        return float(meta["rho_cdmx"]), SRC_MODEL
    ars = _frame(reports, "arsenal")
    if "rho_cdmx_used" in ars.columns and ars["rho_cdmx_used"].notna().any():
        return float(ars["rho_cdmx_used"].median()), SRC_ARSENAL
    return None, ""


def _family_delta(ars: pd.DataFrame) -> dict | None:
    """ΔStuff+ CDMX medio (ponderado por ``n_pitches``) de la Four-Seam, o de la familia que mas pierde."""
    need = {"pitch_id", "n_pitches", "delta_stuff_cdmx"}
    if not need <= set(ars.columns) or ars.empty:
        return None
    d = ars.dropna(subset=list(need)).copy()
    d = d[d["n_pitches"] > 0]
    if d.empty:
        return None
    d["w"] = d["delta_stuff_cdmx"] * d["n_pitches"]
    g = d.groupby("pitch_id").agg(w=("w", "sum"), n=("n_pitches", "sum"))
    g["delta"] = g["w"] / g["n"]
    pid = PRIMARY_FAMILY if PRIMARY_FAMILY in g.index else str(g["delta"].idxmin())
    return {"pitch_id": pid, "delta": float(g.loc[pid, "delta"]), "n": int(g.loc[pid, "n"])}


def _holdouts(h: pd.DataFrame, model: str = "Pitching+") -> list[tuple[str, float]]:
    """Reduccion relativa de RMSE de ``xrv_pitcher`` (``model``) contra el baseline nulo, por holdout."""
    if h.empty or not {"holdout", "model", "target", "rmse"} <= set(h.columns):
        return []
    x = h[h["target"] == "xrv_pitcher"]
    pit = x[x["model"] == model].set_index("holdout")["rmse"]
    base = x[x["model"] == "CountBaselineModel"].set_index("holdout")["rmse"]
    red = (1.0 - pit / base).dropna()
    folds = [i for i in red.index if str(i).startswith("pitcher_fold")]
    out: list[tuple[str, float]] = []
    if folds:
        out.append(("Pitcher", float(red[folds].mean())))
    for key, label in (("season", "Season"), ("park", "Park")):
        if key in red.index:
            out.append((label, float(red[key])))
    return out


def _stability(s: pd.DataFrame) -> dict | None:
    if s.empty or not {"model", "pearson_r"} <= set(s.columns):
        return None
    by = s.set_index("model")
    if "Stuff+" not in by.index:
        return None
    row = by.loc["Stuff+"]
    out = {"stuff_r": float(row["pearson_r"]), "n": int(row.get("n_pitchers", 0) or 0),
           "basis": str(row.get("score_basis", "")).replace("in_sample_train_2024_2025", "en muestra 2024–25")}
    if "Pitching+" in by.index:
        out["pitching_r"] = float(by.loc["Pitching+", "pearson_r"])
    return out


def _scope(lb: pd.DataFrame) -> dict | None:
    if lb.empty or not {"n_pitches", "pitcher_anon_id"} <= set(lb.columns):
        return None
    out = {"pitches": int(lb["n_pitches"].sum()), "pitchers": int(lb["pitcher_anon_id"].nunique())}
    if "year" in lb.columns and lb["year"].notna().any():
        out["y0"], out["y1"] = int(lb["year"].min()), int(lb["year"].max())
    return out


def home_figures(reports: dict | None, meta: dict | None = None) -> dict[str, Any]:
    """Cifras de la portada. Cada clave puede ser ``None``/vacia si el CSV de origen no existe."""
    rho, rho_src = _rho(reports, meta)
    return {
        "rho": rho, "rho_src": rho_src,
        "delta": _family_delta(_frame(reports, "arsenal")),
        "holdouts": _holdouts(_frame(reports, "holdout_metrics")),
        "stuff_holdouts": _holdouts(_frame(reports, "holdout_metrics"), "Stuff+"),
        "stability": _stability(_frame(reports, "stability")),
        "scope": _scope(_frame(reports, "leaderboard")),
    }


def _signed(x: float, decimals: int = 2) -> str:
    return f"{x:+.{decimals}f}".replace("-", "−")


def _stuff_note(fig: dict[str, Any]) -> str:
    """Aviso honesto: Stuff+ solo empata con el baseline en ``xrv_pitcher`` (rango calculado de los holdouts)."""
    sv = [v for _, v in (fig.get("stuff_holdouts") or [])]
    if not sv:
        return ""
    return f" · Stuff+ solo empata con el baseline ({_signed(100 * min(sv))} a {_signed(100 * max(sv))} %)"


def anchor_cards(fig: dict[str, Any]) -> list[dict[str, str]]:
    """Tarjetas de cifras ancla (solo las que tienen datos)."""
    cards: list[dict[str, str]] = []
    if fig.get("rho") is not None:
        less = 100.0 * (1.0 - fig["rho"])
        cards.append({"icon": "mtn", "label": "Densidad del aire ρ̂ en el Harp Helú (mediana de Altitud Extrema)", "value": f"{fig['rho']:.3f}",
                      "unit": "", "sub": f"{less:.0f} % menos aire que a nivel del mar",
                      "src": fig["rho_src"], "tone": "navy"})
    d = fig.get("delta")
    if d:
        cards.append({"icon": "fall", "label": f"ΔStuff+ medio de la {d['pitch_id']} en CDMX",
                      "value": _signed(d["delta"]), "unit": "puntos",
                      "sub": f"Ponderado por {d['n']:,} pitcheos reales".replace(",", " "), "src": SRC_ARSENAL,
                      "tone": "loss" if d["delta"] < 0 else "gain"})
    hs = fig.get("holdouts") or []
    if hs:
        vals = [v for _, v in hs]
        cards.append({"icon": "shield", "label": "Pitching+: menos error que el baseline nulo, en holdout",
                      "value": f"{100 * min(vals):.1f}–{100 * max(vals):.1f}", "unit": "%",
                      "sub": " · ".join(f"{k} {100 * v:.2f} %" for k, v in hs) + _stuff_note(fig), "src": SRC_HOLDOUT, "tone": "gain"})
    st_ = fig.get("stability")
    if st_:
        extra = f" · Pitching+ r {st_['pitching_r']:.2f}" if "pitching_r" in st_ else ""
        cards.append({"icon": "spin", "label": "Estabilidad año a año de Stuff+ (en muestra)", "value": f"{st_['stuff_r']:.2f}",
                      "unit": "r", "sub": f"{st_['n']} pitchers, {st_['basis']}{extra}", "src": SRC_STABILITY,
                      "tone": "navy"})
    sc = fig.get("scope")
    if sc:
        yrs = f" · temporadas {sc['y0']}–{sc['y1']}" if "y0" in sc else ""
        cards.append({"icon": "ball", "label": "Pitcheos analizados", "value": f"{sc['pitches']:,}".replace(",", " "),
                      "unit": "", "sub": f"{sc['pitchers']:,} pitchers{yrs}".replace(",", " "), "src": SRC_LEADERBOARD,
                      "tone": "navy"})
    return cards


# ----------------------------------------------------------------------------
# Contenido estatico (sin cifras de resultados)
# ----------------------------------------------------------------------------
TOOLS: list[dict[str, str]] = [
    {"id": "sim", "tab": "Simulador", "icon": "flask", "name": "Simulador",
     "q": "¿Cuánto pierde este pitcheo en el Harp Helú?",
     "d": "Ajusta velocidad, giro y liberación y compara nivel del mar contra CDMX.", "btn": "Abrir el simulador"},
    {"id": "map", "tab": "Mapa de movimiento", "icon": "layers", "name": "Mapa de movimiento",
     "q": "¿Qué formas ganan o pierden en altitud?",
     "d": "Todo el movimiento de la LMB coloreado por ΔStuff+.", "btn": "Ver el mapa"},
    {"id": "lb", "tab": "Leaderboard", "icon": "chart", "name": "Leaderboard",
     "q": "¿Quién de la liga rinde mejor en CDMX?",
     "d": "Ranking de pitchers con tarjeta y arsenal.", "btn": "Ver el ranking"},
    {"id": "scu", "tab": "Sculptor", "icon": "target", "name": "Sculptor",
     "q": "¿Qué pares reales con su mecánica sostienen mejor su arsenal?",
     "d": "Referencias factibles de la liga, no una receta causal.", "btn": "Abrir Sculptor"},
    {"id": "vv", "tab": "VeloVision", "icon": "video", "name": "VeloVision",
     "q": "¿Qué arsenal le conviene a un prospecto, desde un video de celular?",
     "d": "Mide la mecánica del clip y entrega un rango, no un valor puntual.", "btn": "Analizar un video"},
]

WHY: list[tuple[str, str, str]] = [
    ("bulb", "Decidir en el dugout", "Simula un pitcheo y míralo en el Harp Helú contra nivel del mar, antes de lanzarlo."),
    ("search", "Fichar y desarrollar", "Evalúa prospectos desde un video y compáralos con referencias reales de la liga."),
    ("shield", "Evidencia honesta", "Rangos, intervalos de confianza, holdouts y una descomposición de 3 términos."),
]

FLOW: list[tuple[str, str, str]] = [
    ("video", "Video o datos", "Un clip de celular o un pitcheo cargado a mano."),
    ("arm", "Mecánica", "Brazo, liberación y eje de giro del lanzador."),
    ("mtn", "Arsenal en CDMX", "Stuff+ y movimiento en el aire del Harp Helú."),
    ("users", "Referencias para entrenar", "Pares reales con mecánica parecida."),
]

HONESTY: list[tuple[str, str]] = [
    ("target", "Stuff+ es un descriptor físico del pitcheo, no un predictor pitcheo a pitcheo."),
    ("mtn", "CDMX es la mediana de la categoría Altitud Extrema, que agrupa varios parques de altura."),
    ("video", "VeloVision entrega rangos: un video no mide giro, quiebre ni velocidad."),
    ("clock", "La temporada 2026 está fuera de muestra: no entrenó ningún modelo."),
]

CREDITS = "Diablos Rojos del México · Hackathon ISAC 2026 · Reto 2"


# ----------------------------------------------------------------------------
# Hero: escena SVG/CSS propia (sin video de stock)
# ----------------------------------------------------------------------------
PATH_SEA = "M70,168 C330,138 590,196 800,300"
PATH_CDMX = "M70,168 C330,152 590,232 800,372"


def _particles(n: int = 34) -> str:
    """Particulas de aire en posiciones deterministas (sin azar): derivan de derecha a izquierda."""
    out, x = [], 0.37
    for i in range(n):
        x = (x * 9301 + 49297) % 233280 / 233280.0
        y = (x * 7919 + i * 0.173) % 1.0
        cx, cy = 40 + 820 * ((i * 0.6180339) % 1.0), 40 + 330 * y
        r = 1.6 + 1.8 * ((i * 0.37) % 1.0)
        out.append(f'<circle class="hm-p" cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.1f}" style="animation-delay:-{(i * 0.83) % 9:.1f}s;'
                   f'animation-duration:{8 + (i % 5) * 1.6:.1f}s"/>')
    return "".join(out)


def hero_scene() -> str:
    """SVG de fondo: montanas, silueta de estadio, particulas y dos trayectorias (nivel del mar vs Harp Helu)."""
    return (
        '<div class="hm-bg" aria-hidden="true"><svg viewBox="0 0 900 470" preserveAspectRatio="xMaxYMid meet" '
        'xmlns="http://www.w3.org/2000/svg" role="img">'
        '<defs><linearGradient id="hmsky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#EAF0F6"/>'
        '<stop offset="1" stop-color="#F8FAFC"/></linearGradient></defs>'
        '<rect width="900" height="470" fill="url(#hmsky)"/>'
        '<path d="M0,380 L120,300 L190,340 L300,230 L380,320 L470,270 L560,350 L640,300 L760,360 L900,310 L900,470 L0,470Z" '
        'fill="#DCE5EE"/>'
        '<path d="M0,410 L90,360 L170,395 L260,340 L360,400 L470,350 L580,405 L700,365 L810,400 L900,370 L900,470 L0,470Z" '
        'fill="#C9D5E2"/>'
        # estadio: gradas, torres de luz y campo
        '<path d="M560,430 Q700,330 860,430Z" fill="#B3C2D3"/>'
        '<path d="M585,430 Q705,352 835,430Z" fill="#9FB1C5"/>'
        '<g stroke="#9FB1C5" stroke-width="3" stroke-linecap="round"><path d="M575,430V340M845,430V340"/></g>'
        '<g fill="#9FB1C5"><rect x="562" y="326" width="26" height="12" rx="2"/><rect x="832" y="326" width="26" height="12" rx="2"/></g>'
        '<rect x="0" y="430" width="900" height="40" fill="#B3C2D3"/>'
        + _particles() +
        # plato / zona de strike (referencia)
        '<g fill="none" stroke="#9FB1C5" stroke-width="1.5" stroke-dasharray="4 4"><path d="M800,276V388"/></g>'
        # trayectorias
        f'<path class="hm-tr hm-t-sea" d="{PATH_SEA}" pathLength="1" fill="none" stroke="#7C8999" stroke-width="3.2" stroke-linecap="round"/>'
        f'<path class="hm-tr hm-t-cdmx" d="{PATH_CDMX}" pathLength="1" fill="none" stroke="#1F3A5F" stroke-width="3.6" stroke-linecap="round"/>'
        '<circle cx="70" cy="168" r="6" fill="#fff" stroke="#1F3A5F" stroke-width="2.4"/>'
        f'<circle class="hm-ball hm-b-sea" cx="0" cy="0" r="8.5" fill="#fff" stroke="#7C8999" stroke-width="2.6" '
        f'style="offset-path:path(\'{PATH_SEA}\')"/>'
        f'<circle class="hm-ball hm-b-cdmx" cx="0" cy="0" r="9.5" fill="#fff" stroke="#1F3A5F" stroke-width="3" '
        f'style="offset-path:path(\'{PATH_CDMX}\')"/>'
        # cota de caida
        '<g class="hm-drop"><path d="M826,300V372" stroke="#1F3A5F" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M820,300h12M820,372h12" stroke="#1F3A5F" stroke-width="2" stroke-linecap="round"/></g>'
        '</svg></div>'
    )


def hero_legend() -> str:
    return (
        '<div class="hm-legend"><span><i class="sea"></i>Nivel del mar</span><span><i class="mx"></i>Harp Helú: cae más</span>'
        '<em>Ilustración esquemática, no es una medición</em></div>'
    )


def physics_bars(rho: float | None) -> str:
    """Mini diagrama: fuerzas aerodinamicas escalan con rho (ancho = rho); la gravedad no cambia."""
    if rho is None:
        return ""
    rows = [("Magnus", rho), ("Arrastre", rho), ("Seam-Shifted Wake", rho), ("Gravedad", 1.0)]
    body = []
    for name, k in rows:
        cls = "g" if name == "Gravedad" else "a"
        body.append(
            f'<div class="r"><span class="n">{C.esc(name)}</span><span class="tk">'
            f'<i class="sea" style="width:100%"></i><i class="mx {cls}" style="width:{100 * k:.1f}%"></i></span></div>')
    return ('<div class="hm-phys-d" role="img" aria-label="Fuerzas aerodinámicas proporcionales a la densidad del aire">'
            + "".join(body)
            + '<div class="lg"><span><i class="sea"></i>Nivel del mar</span><span><i class="mx a"></i>Harp Helú</span></div></div>')


# ----------------------------------------------------------------------------
# CSS propio de la portada (tokens de theme.py; el rojo solo en logo/boton primario)
# ----------------------------------------------------------------------------
HOME_CSS = """<style>
/* oculta miga y pastillas del Simulador solo cuando la pestaña activa es la primera (Inicio) */
.stApp:has([data-testid="stTab"]:first-of-type[aria-selected="true"]) .st-key-app_header_ctx{display:none !important}
.hm-eyebrow{display:flex;align-items:center;gap:.4rem;font-size:.78rem;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:var(--lvl-cdmx)}
.hm-h2{font-size:1.75rem;font-weight:800;letter-spacing:-.02em;line-height:1.15;color:var(--text);margin:.15rem 0 0}
.hm-lead{font-size:1rem;color:var(--muted);line-height:1.5;max-width:46rem;margin-top:.35rem}
.hm-sec{margin-top:2.6rem;margin-bottom:.4rem}

/* ---- hero ---- */
.st-key-home_hero{position:relative;overflow:hidden;border-radius:20px;border:1px solid var(--line);
  background:linear-gradient(100deg,#FFFFFF 0%,#FFFFFF 36%,#EEF3F8 100%);box-shadow:var(--sh);
  padding:2.4rem 2.8rem 2.4rem !important;min-height:31rem;margin-top:.4rem;justify-content:center}
.st-key-home_hero [data-testid="stElementContainer"]{position:relative;z-index:2}
.st-key-home_hero [data-testid="stElementContainer"]:has(.hm-bg),
.st-key-home_hero [data-testid="stElementContainer"]:has(.hm-legend){position:absolute !important;z-index:1;margin:0}
.st-key-home_hero [data-testid="stElementContainer"]:has(.hm-bg){right:0;top:0;bottom:0;width:68%}
.st-key-home_hero [data-testid="stElementContainer"]:has(.hm-legend){right:1.4rem;bottom:1rem;width:auto;z-index:3}
.hm-bg{position:absolute;inset:0;pointer-events:none;
  -webkit-mask-image:linear-gradient(90deg,transparent 0%,#000 26%);mask-image:linear-gradient(90deg,transparent 0%,#000 26%)}
.hm-bg::after{content:"";position:absolute;inset:0;z-index:1;background:linear-gradient(90deg,#fff 0%,rgba(255,255,255,.85) 14%,rgba(255,255,255,0) 46%)}
.hm-bg svg{width:100%;height:100%;display:block}
.hm-legend{display:flex;gap:.9rem;align-items:center;flex-wrap:wrap;font-size:.74rem;font-weight:700;color:var(--text);
  background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:99px;padding:.3rem .85rem}
.hm-legend span{display:inline-flex;align-items:center;gap:.35rem}
.hm-legend i{width:.7rem;height:.7rem;border-radius:50%;display:inline-block;border:2px solid var(--lvl-sea);background:#fff}
.hm-legend i.mx{border-color:var(--navy)}
.hm-legend em{font-style:normal;font-weight:500;color:var(--muted)}
.st-key-home_hero .hm-title{font-size:3.3rem;font-weight:800;letter-spacing:-.03em;line-height:1.06;color:var(--text);max-width:50rem;margin:1rem 0 0}
.hm-title b{color:var(--lvl-cdmx);font-weight:800}
.hm-title span{display:block;margin-top:.5rem;font-size:.54em;font-weight:700;color:var(--gain);letter-spacing:-.02em}
.st-key-home_hero .hm-sub{font-size:1.15rem;color:var(--muted);margin:1.3rem 0 .6rem;max-width:42rem;line-height:1.45}
.hm-tag{display:inline-flex;align-items:center;gap:.5rem;font-size:.82rem;font-weight:700;color:var(--text);
  background:var(--card);border:1px solid var(--line);border-radius:99px;padding:.3rem .85rem;box-shadow:var(--sh)}
.hm-tag .ic{color:var(--lvl-cdmx)}
.st-key-home_hero .stButton > button{padding:.75rem 1.1rem !important}
.st-key-home_hero .stButton > button p{font-size:.95rem !important}
.hm-tr{stroke-dasharray:1;stroke-dashoffset:1;animation:hmdraw 7s linear infinite}
.hm-t-cdmx{animation-delay:.15s}
.hm-ball{offset-distance:0%;opacity:0;animation:hmfly 7s linear infinite}
.hm-b-cdmx{animation-delay:.15s}
.hm-drop{animation:hmdrop 7s linear infinite}
.hm-p{fill:#3E5C7A;opacity:.0;animation:hmdrift 10s linear infinite}
@keyframes hmdraw{0%{stroke-dashoffset:1;opacity:1}62%{stroke-dashoffset:0;opacity:1}90%{stroke-dashoffset:0;opacity:1}100%{stroke-dashoffset:0;opacity:0}}
@keyframes hmfly{0%{offset-distance:0%;opacity:1}62%{offset-distance:100%;opacity:1}90%{offset-distance:100%;opacity:1}100%{offset-distance:100%;opacity:0}}
@keyframes hmdrop{0%,60%{opacity:0}72%,90%{opacity:1}100%{opacity:0}}
@keyframes hmdrift{0%{transform:translateX(0);opacity:0}15%{opacity:.34}85%{opacity:.34}100%{transform:translateX(-120px);opacity:0}}
@media (prefers-reduced-motion:reduce){
  .hm-tr,.hm-drop,.hm-ball,.hm-p{animation:none !important}
  .hm-tr{stroke-dashoffset:0;opacity:1}.hm-ball{offset-distance:100%;opacity:1}.hm-p{opacity:.3}
}

/* ---- cifras ancla ---- */
.hm-figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(15.5rem,1fr));gap:1rem;margin-top:1.1rem}
.hm-fig{background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh);
  padding:1.1rem 1.2rem 1rem;display:flex;flex-direction:column;gap:.35rem;position:relative;overflow:hidden}
.hm-fig::before{content:"";position:absolute;left:0;top:0;bottom:0;width:.28rem;background:var(--lvl-cdmx)}
.hm-fig.loss::before{background:var(--loss)} .hm-fig.gain::before{background:var(--gain)}
.hm-fig .t{display:flex;gap:.4rem;align-items:center;font-size:.8rem;font-weight:700;color:var(--muted);line-height:1.25}
.hm-fig .v{font-family:var(--f-num);font-weight:700;font-size:3.5rem;line-height:.95;letter-spacing:-.01em;color:var(--navy)}
.hm-fig.loss .v{color:var(--loss)} .hm-fig.gain .v{color:var(--gain)}
.hm-fig .v small{font-family:var(--f-txt);font-size:.34em;font-weight:700;margin-left:.3rem;color:var(--muted)}
.hm-fig .s{font-size:.78rem;color:var(--muted);line-height:1.4;min-height:2.2rem}

/* ---- tarjetas por que una app ---- */
.hm-why{display:grid;grid-template-columns:repeat(auto-fit,minmax(18rem,1fr));gap:1rem;margin-top:1rem}
.hm-wc{background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--sh);padding:1.4rem 1.4rem 1.3rem}
.hm-ib{width:3rem;height:3rem;border-radius:.9rem;display:grid;place-items:center;background:#E8EEF5;color:var(--navy);margin-bottom:.9rem}
.hm-ib .ic{width:1.5rem;height:1.5rem}
.hm-wc h3{font-size:1.15rem;font-weight:800;margin:0 0 .35rem;letter-spacing:-.01em}
.hm-wc p{font-size:.92rem;color:var(--muted);line-height:1.5}

/* ---- herramientas ---- */
[class*="st-key-home_tool_"]{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--sh);padding:1.15rem 1.1rem 1.1rem !important;height:100%;gap:.8rem !important}
.hm-tool .ic-wrap{width:2.7rem;height:2.7rem;border-radius:.8rem;display:grid;place-items:center;background:#E8EEF5;color:var(--navy)}
.hm-tool .ic-wrap .ic{width:1.35rem;height:1.35rem}
.hm-tool .nm{font-size:.74rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--lvl-cdmx);margin-top:.7rem}
.hm-tool .q{font-size:1.02rem;font-weight:800;line-height:1.3;letter-spacing:-.01em;margin-top:.25rem;min-height:4rem}
.hm-tool .d{font-size:.82rem;color:var(--muted);line-height:1.45;margin-top:.4rem;min-height:3.6rem}

/* ---- flujo ---- */
.hm-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin-top:1rem;position:relative}
.hm-st{position:relative;padding:0 1.1rem 0 0}
.hm-st .top{display:flex;align-items:center;margin-bottom:.8rem}
.hm-st .n{width:3.3rem;height:3.3rem;border-radius:50%;background:var(--card);border:2px solid var(--lvl-cdmx);color:var(--navy);
  display:grid;place-items:center;box-shadow:var(--sh);flex:none}
.hm-st .n .ic{width:1.5rem;height:1.5rem}
.hm-st .ln{flex:1;height:2px;background:repeating-linear-gradient(90deg,var(--lvl-sea) 0 .45rem,transparent .45rem .8rem);margin:0 .5rem}
.hm-st:last-child .ln{display:none}
.hm-st .k{font-size:.72rem;font-weight:800;color:var(--lvl-cdmx);letter-spacing:.06em;text-transform:uppercase}
.hm-st h4{font-size:1.05rem;font-weight:800;margin:.15rem 0 .25rem}
.hm-st p{font-size:.86rem;color:var(--muted);line-height:1.45}

/* ---- fisica ---- */
.hm-phys{display:grid;grid-template-columns:1.1fr 1fr;gap:2rem;align-items:center;background:var(--card);border:1px solid var(--line);
  border-radius:20px;box-shadow:var(--sh);padding:1.8rem 2rem;margin-top:1rem}
.hm-eq{font-family:var(--f-num);font-weight:700;font-size:3.4rem;letter-spacing:-.01em;color:var(--navy);line-height:1}
.hm-eq small{font-family:var(--f-txt);font-size:.3em;font-weight:600;color:var(--muted);margin-left:.5rem;letter-spacing:0}
.hm-phys p{font-size:1rem;line-height:1.55;color:var(--text);margin-top:.8rem}
.hm-phys p b{color:var(--navy)}
.hm-phys-d{display:flex;flex-direction:column;gap:.7rem}
.hm-phys-d .r{display:grid;grid-template-columns:9.5rem 1fr;gap:.8rem;align-items:center;font-size:.82rem;font-weight:700}
.hm-phys-d .tk{position:relative;height:1.1rem}
.hm-phys-d .tk i{position:absolute;left:0;border-radius:99px;display:block}
.hm-phys-d .tk .sea{top:0;height:.45rem;background:var(--lvl-sea)}
.hm-phys-d .tk .mx{top:.55rem;height:.45rem;background:var(--navy)}
.hm-phys-d .lg{display:flex;gap:1rem;font-size:.74rem;color:var(--muted);font-weight:600;margin-top:.2rem}
.hm-phys-d .lg i{display:inline-block;width:.8rem;height:.45rem;border-radius:99px;margin-right:.35rem;vertical-align:middle;background:var(--lvl-sea)}
.hm-phys-d .lg i.mx{background:var(--navy)}

/* ---- honestidad ---- */
.hm-hon{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:.9rem;margin-top:1rem}
.hm-hc{display:flex;gap:.7rem;align-items:flex-start;background:var(--soft);border:1px solid var(--line);border-radius:12px;
  padding:.9rem 1rem;font-size:.86rem;line-height:1.45;color:var(--text)}
.hm-hc .ic{width:1.3rem;height:1.3rem;color:var(--lvl-cdmx);margin-top:.05rem}
.hm-credit{display:flex;align-items:center;gap:.6rem;margin:1.6rem 0 .4rem;font-size:.85rem;font-weight:700;color:var(--muted)}
.hm-credit .d{width:1.6rem;height:1.6rem;border-radius:.5rem;background:var(--brand);color:#fff;display:grid;place-items:center;
  font-weight:800;font-size:.95rem}
@media (max-width:900px){
  .hm-flow{grid-template-columns:1fr 1fr;row-gap:1.4rem}.hm-phys{grid-template-columns:1fr}
  .hm-title{font-size:2.1rem}
  .st-key-home_hero{padding:1.6rem 1.3rem !important}
}
</style>"""


# ----------------------------------------------------------------------------
# Secciones
# ----------------------------------------------------------------------------
def _eyebrow(icon_name: str, text: str) -> str:
    return f'<div class="hm-eyebrow">{C.icon(icon_name)}{C.esc(text)}</div>'


def _section(icon_name: str, eyebrow: str, title: str, lead: str = "") -> None:
    lead_html = f'<div class="hm-lead">{C.esc(lead)}</div>' if lead else ""
    st.markdown(f'<div class="hm-sec">{_eyebrow(icon_name, eyebrow)}<h2 class="hm-h2">{C.esc(title)}</h2>{lead_html}</div>',
                unsafe_allow_html=True)


def _hero() -> None:
    with st.container(key="home_hero"):
        st.markdown(hero_scene(), unsafe_allow_html=True)
        st.markdown(hero_legend(), unsafe_allow_html=True)
        st.markdown(
            f'<span class="hm-tag">{C.icon("ball")}Diablos AeroVision · Stuff+ LMB ajustado por altitud</span>'
            '<div class="hm-title" role="heading" aria-level="1">El aire del <b>Harp Helú</b> le quita movimiento a tu pitcheo.'
            '<span>AeroVision te dice cuánto pierde y qué referencias reales seguir.</span></div>'
            '<p class="hm-sub">Simula, compara y encuentra referencias con datos de la LMB: del dugout al prospecto.</p>',
            unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.15, 1.15, 2.4], gap="small")
        c1.button("Abrir el simulador", key="home_hero_sim", type="primary", width="stretch",
                  on_click=C.jump_to_tab, args=("Simulador", "Abriendo el Simulador"))
        c2.button("Analizar un video", key="home_hero_vv", width="stretch",
                  on_click=C.jump_to_tab, args=("VeloVision", "Abriendo VeloVision"))


def _anchors(fig: dict[str, Any]) -> None:
    cards = anchor_cards(fig)
    if not cards:
        st.markdown(C.note_box("Las cifras de respaldo no están disponibles: faltan los reportes de reports/."),
                    unsafe_allow_html=True)
        return
    html = []
    for c in cards:
        unit = f"<small>{C.esc(c['unit'])}</small>" if c["unit"] else ""
        html.append(
            f'<div class="hm-fig {C.esc(c["tone"])}"><div class="t">{C.icon(c["icon"])}{C.esc(c["label"])}</div>'
            f'<div class="v">{C.esc(c["value"])}{unit}</div><div class="s">{C.esc(c["sub"])}</div>'
            f'<div class="bds">{C.provenance_pill("fuente: " + c["src"])}</div></div>')
    st.markdown(f'<div class="hm-figs">{"".join(html)}</div>', unsafe_allow_html=True)


def _why() -> None:
    _section("layers", "Más que un modelo", "Por qué una app y no solo un modelo",
             "Un Stuff+ en una tabla no cambia una decisión. Aquí cada resultado se vuelve una herramienta para el cuerpo técnico.")
    cards = "".join(
        f'<div class="hm-wc"><div class="hm-ib">{C.icon(ic)}</div><h3>{C.esc(t)}</h3><p>{C.esc(d)}</p></div>'
        for ic, t, d in WHY)
    st.markdown(f'<div class="hm-why">{cards}</div>', unsafe_allow_html=True)


def _tools() -> None:
    _section("target", "Cinco herramientas", "Una pregunta de negocio por pestaña")
    st.write("")
    cols = st.columns(len(TOOLS), gap="small")
    for col, t in zip(cols, TOOLS):
        with col, st.container(key=f"home_tool_{t['id']}"):
            st.markdown(
                f'<div class="hm-tool"><div class="ic-wrap">{C.icon(t["icon"])}</div><div class="nm">{C.esc(t["name"])}</div>'
                f'<div class="q">{C.esc(t["q"])}</div><div class="d">{C.esc(t["d"])}</div></div>', unsafe_allow_html=True)
            st.button(t["btn"], key=f"home_go_{t['id']}", width="stretch", on_click=C.jump_to_tab,
                      args=(t["tab"], f"Abriendo {t['name']}"))


def _flow() -> None:
    _section("right", "Flujo del coach", "De un video a un plan de entrenamiento")
    steps = "".join(
        f'<div class="hm-st"><div class="top"><div class="n">{C.icon(ic)}</div><div class="ln"></div></div>'
        f'<div class="k">Paso {i}</div><h4>{C.esc(t)}</h4><p>{C.esc(d)}</p></div>'
        for i, (ic, t, d) in enumerate(FLOW, 1))
    st.markdown(f'<div class="hm-flow">{steps}</div>', unsafe_allow_html=True)


def _physics(fig: dict[str, Any]) -> None:
    _section("wave", "La física en una línea", "Menos aire, menos movimiento")
    st.markdown(
        '<div class="hm-phys"><div><div class="hm-eq">F ∝ ρ · v²<small>Magnus · arrastre · Seam-Shifted Wake</small></div>'
        '<p>Magnus, arrastre y Seam-Shifted Wake escalan con <b>ρ · v²</b>: <b>menos aire implica menos movimiento</b>, '
        'y solo la gravedad no cambia.</p></div>' + physics_bars(fig.get("rho")) + '</div>', unsafe_allow_html=True)


def _honesty() -> None:
    _section("shield", "Honestidad del producto", "Lo que esta app dice y lo que no")
    cells = "".join(f'<div class="hm-hc">{C.icon(ic)}<span>{C.esc(t)}</span></div>' for ic, t in HONESTY)
    st.markdown(f'<div class="hm-hon">{cells}</div>'
                f'<div class="hm-credit"><span class="d">D</span>{C.esc(CREDITS)}</div>', unsafe_allow_html=True)


def render_home_tab(reports: dict | None = None, meta: dict | None = None) -> None:
    """Portada. ``reports`` y ``meta`` llegan de ``app.py``; si faltan, la portada se degrada sin truene."""
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    try:
        fig = home_figures(reports, meta)
    except Exception:  # noqa: BLE001 - la portada nunca debe tumbar la app
        fig = {}
    _hero()
    _anchors(fig)
    _why()
    _tools()
    _flow()
    _physics(fig)
    _honesty()

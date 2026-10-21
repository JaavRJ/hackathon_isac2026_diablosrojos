"""Puente VeloVision -> catalogo LMB (HACK-017, subtarea 17-B).

Dado el arm slot medido en video (misma formula que el dataset,
``src/features/physical.py``), devuelve por familia de pitcheo el RANGO de
Stuff+ CDMX y de delta Stuff+ CDMX que tienen lanzadores reales de la LMB con
esa mecanica. Nunca devuelve un valor puntual del prospecto ni lo empareja con
un ``pitcher_anon_id``; toda cifra sale de percentiles del catalogo
``reports/leaderboards/pitcher_arsenal_breakdown.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "reports" / "leaderboards" / "pitcher_arsenal_breakdown.csv"

DISCLAIMER = (
    "Rango condicionado a lo medible en 2D; el video no mide spin, quiebre ni "
    "velocidad y no se empareja con pitchers del dataset."
)

MIN_WINDOW_DEG = 6.0
REL_HEIGHT_TOL_FT = 0.5   # filtro OPCIONAL: solo si se pasa rel_height_ft
FB_VELO_TOL_MPH = 1.5     # filtro OPCIONAL: solo si se pasa observed_velo_mph
PERCENTILES = (10, 25, 50, 75, 90)
METRICS = ("stuff_plus_cdmx", "delta_stuff_cdmx")

REQUIRED_COLUMNS = [
    "pitcher_anon_id", "pitch_id", "n_pitches", "PitcherThrows",
    "arm_slot_deg", "RelHeight", "fb_velo", "stuff_plus_cdmx", "delta_stuff_cdmx",
]

_THROWS = {
    "right": "Right", "r": "Right", "derecho": "Right", "diestro": "Right",
    "left": "Left", "l": "Left", "izquierdo": "Left", "zurdo": "Left",
}


def normalize_throws(throws: str) -> str:
    key = str(throws).strip().lower()
    if key not in _THROWS:
        raise ValueError(f"Mano no reconocida: {throws!r} (use Right/Left o Derecho/Zurdo)")
    return _THROWS[key]


def weighted_percentile(values, weights, q) -> np.ndarray:
    """Percentiles ponderados (q en 0-100).

    Cada observacion ocupa la posicion ``(cum_w - w/2) / sum_w`` en la CDF
    ponderada y se interpola linealmente entre posiciones; fuera de
    [primera, ultima] se satura al minimo/maximo.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return np.full(np.atleast_1d(q).shape, np.nan)
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    pos = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(q, dtype=float) / 100.0, pos, v)


def _load_catalog(catalog) -> pd.DataFrame:
    if catalog is None:
        catalog = DEFAULT_CATALOG
    df = catalog.copy() if isinstance(catalog, pd.DataFrame) else pd.read_csv(catalog)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "El catálogo de arsenal no tiene las columnas requeridas: " + ", ".join(missing)
        )
    return df.reset_index(drop=True)


def _filter_catalog(df, throws, slot, window_deg, rel_height_ft, observed_velo_mph) -> pd.DataFrame:
    mask = (df["PitcherThrows"] == throws) & ((df["arm_slot_deg"] - slot).abs() <= window_deg)
    if rel_height_ft is not None:
        mask &= (df["RelHeight"] - float(rel_height_ft)).abs() <= REL_HEIGHT_TOL_FT
    if observed_velo_mph is not None:
        mask &= (df["fb_velo"] - float(observed_velo_mph)).abs() <= FB_VELO_TOL_MPH
    return df[mask]


def _family_stats(sub: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        s = sub.dropna(subset=[metric])
        vals = weighted_percentile(s[metric], s["n_pitches"], PERCENTILES)
        out[metric] = {f"p{p}": float(x) for p, x in zip(PERCENTILES, vals)}
    return out


def _families(sub: pd.DataFrame, min_pitchers: int) -> list[dict[str, Any]]:
    fams: list[dict[str, Any]] = []
    for pitch_id, g in sub.groupby("pitch_id"):
        entry: dict[str, Any] = {
            "pitch_id": pitch_id,
            "n_pitchers": int(g["pitcher_anon_id"].nunique()),
            "n_pitches": int(g["n_pitches"].sum()),
        }
        if entry["n_pitchers"] < min_pitchers:
            entry["status"] = "INSUFFICIENT_PEERS"
        else:
            entry["status"] = "OK"
            entry.update(_family_stats(g))
        fams.append(entry)

    def sort_key(f):
        if f["status"] != "OK":
            return (1, 0.0)
        return (0, -f["stuff_plus_cdmx"]["p50"])

    fams.sort(key=sort_key)
    return fams


def build_prospect_card(
    arm_slot_dataset_deg: float,
    throws: str,
    rel_height_ft: float | None = None,
    slot_error_deg: float | None = None,
    observed_velo_mph: float | None = None,
    min_pitchers: int = 8,
    catalog=None,
    slot_error_kind: str | None = None,
) -> dict[str, Any]:
    """Ficha de prospecto: rangos de Stuff+ CDMX por familia, sin valor puntual.

    Ventana de slot: ``max(6, slot_error_deg)``. ``slot_error_deg`` ya es una
    cota (desviación máxima por esquinas, no una desviación estándar), por eso
    no se duplica. Si es None, negativo o NaN se trata como "aún no medido"
    (ventana 6). Lanza ValueError si ``arm_slot_dataset_deg`` es NaN.
    """
    hand = normalize_throws(throws)
    slot = float(arm_slot_dataset_deg)
    if not np.isfinite(slot):
        raise ValueError("arm_slot_dataset_deg no es un número finito (NaN/inf)")
    has_error = (
        slot_error_deg is not None
        and np.isfinite(float(slot_error_deg))
        and float(slot_error_deg) >= 0
    )
    window = max(MIN_WINDOW_DEG, float(slot_error_deg)) if has_error else MIN_WINDOW_DEG

    df = _load_catalog(catalog)
    sub = _filter_catalog(df, hand, slot, window, rel_height_ft, observed_velo_mph)

    filters = {
        "PitcherThrows": hand,
        "arm_slot_window_deg": window,
        "rel_height_ft": (
            {"center": float(rel_height_ft), "tol": REL_HEIGHT_TOL_FT, "optional": True}
            if rel_height_ft is not None else None
        ),
        "fb_velo_mph": (
            {"center": float(observed_velo_mph), "tol": FB_VELO_TOL_MPH, "optional": True}
            if observed_velo_mph is not None else None
        ),
        "min_pitchers": int(min_pitchers),
    }
    return {
        "families": _families(sub, min_pitchers),
        "window_deg": window,
        # slot_error_kind (p. ej. "sistemático supuesto", "medido (MAE de validación)",
        # "combinado (MAE + sistemático)") etiqueta de dónde sale el error; sin él, la lógica original.
        "error_source": slot_error_kind or ("medido" if has_error else "aún no medido"),
        "filters_applied": filters,
        "n_pitchers_total": int(sub["pitcher_anon_id"].nunique()),
        "sculptor_params": {
            "PitcherThrows": hand,
            "arm_slot_deg": slot,
            "fb_velo": float(observed_velo_mph) if observed_velo_mph is not None else None,
            "RelHeight": float(rel_height_ft) if rel_height_ft is not None else None,
        },
        "disclaimer": DISCLAIMER,
    }


def velo_narrowing_report(
    arm_slot_dataset_deg: float,
    throws: str,
    observed_velo_mph: float,
    rel_height_ft: float | None = None,
    slot_error_deg: float | None = None,
    min_pitchers: int = 8,
    catalog=None,
    slot_error_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Reporta (sin exigir mejora) el ancho p25-p75 de Stuff+ CDMX por familia
    sin y con ``observed_velo_mph``."""
    kw = dict(rel_height_ft=rel_height_ft, slot_error_deg=slot_error_deg,
              min_pitchers=min_pitchers, catalog=catalog, slot_error_kind=slot_error_kind)
    base = build_prospect_card(arm_slot_dataset_deg, throws, **kw)
    with_v = build_prospect_card(arm_slot_dataset_deg, throws,
                                 observed_velo_mph=observed_velo_mph, **kw)

    def widths(card):
        res = {}
        for f in card["families"]:
            if f["status"] == "OK":
                m = f["stuff_plus_cdmx"]
                res[f["pitch_id"]] = (float(m["p75"] - m["p25"]), f["n_pitchers"])
            else:
                res[f["pitch_id"]] = (None, f["n_pitchers"])
        return res

    wb, wv = widths(base), widths(with_v)
    rows = []
    for pid in sorted(set(wb) | set(wv)):
        b, nb = wb.get(pid, (None, 0))
        v, nv = wv.get(pid, (None, 0))
        rows.append({
            "pitch_id": pid,
            "width_p25_p75_without_velo": b,
            "width_p25_p75_with_velo": v,
            "change": (v - b) if (b is not None and v is not None) else None,
            "n_pitchers_without_velo": nb,
            "n_pitchers_with_velo": nv,
        })
    return rows


# ------------------------------------------------------------------ plausibilidad
PLAUS_LOW_Q, PLAUS_HIGH_Q = 0.5, 99.5      # percentiles del rango "plausible" (supuesto de diseño)
REFERENCE_NOTE = ("Referencia de lo que logran pitchers LMB con una mecánica parecida; "
                  "no es el nivel de este lanzador.")

# medida -> (columna del catálogo, etiqueta, unidad, decimales, claves aceptadas del error sistemático)
_PLAUS_MEASURES = {
    "rel_height_ft": ("RelHeight", "Altura de soltura", "ft", 2, ("rel_height_ft", "rel_height_sys_err_ft")),
    "arm_slot_dataset_deg": ("arm_slot_deg", "Arm slot", "°", 1,
                             ("arm_slot_dataset_deg", "arm_slot_deg", "slot_sys_err_deg")),
}


def _sys_err(sys_errs: dict | None, keys) -> float:
    for k in keys:
        v = (sys_errs or {}).get(k)
        if v is not None and np.isfinite(float(v)):
            return abs(float(v))
    return 0.0


def plausibility_report(
    rel_height_ft: float | None,
    rel_side_ft: float | None,
    arm_slot_dataset_deg: float | None,
    throws: str,
    sys_errs: dict | None = None,
    catalog=None,
) -> dict[str, Any]:
    """Compara cada medida del video con la LMB de la MISMA mano (un registro por pitcher-año).

    Rango plausible = [p0.5, p99.5] del catálogo ``pitcher_arsenal_breakdown.csv``, ampliado por el error
    sistemático de la medida (``sys_errs``: ``rel_height_ft``/``rel_height_sys_err_ft``,
    ``arm_slot_dataset_deg``/``slot_sys_err_deg``). Todas las cifras salen del catálogo.

    Por medida devuelve ``status`` ("dentro" | "fuera" | "sin referencia"), ``percentile`` (aproximado,
    0-100, en la liga), ``low``/``high`` (límites del catálogo, sin ampliar), ``err`` y ``text`` (español).
    El catálogo NO trae el lado de soltura (``RelSide``): esa medida queda "sin referencia" y no cuenta
    como fuera. No bloquea nada: solo informa; ``any_outside`` alimenta la confirmación hacia el Sculptor.
    """
    hand = normalize_throws(throws)
    df = _load_catalog(catalog)
    df = df[df["PitcherThrows"].map(normalize_throws) == hand]
    unit_df = df.groupby(["pitcher_anon_id", "year"] if "year" in df.columns else ["pitcher_anon_id"],
                         as_index=False)[["RelHeight", "arm_slot_deg"]].mean()

    values = {"rel_height_ft": rel_height_ft, "arm_slot_dataset_deg": arm_slot_dataset_deg}
    report: dict[str, Any] = {}
    for key, (col, label, unit, dec, err_keys) in _PLAUS_MEASURES.items():
        v = values[key]
        ref = unit_df[col].dropna().to_numpy(dtype=float)
        if v is None or not np.isfinite(float(v)) or ref.size == 0:
            report[key] = {"status": "sin referencia", "value": None, "percentile": None, "low": None,
                           "high": None, "err": 0.0, "n_reference": int(ref.size),
                           "text": f"{label}: sin medida para comparar con la LMB."}
            continue
        v = float(v)
        lo, hi = (float(x) for x in np.percentile(ref, [PLAUS_LOW_Q, PLAUS_HIGH_Q]))
        err = _sys_err(sys_errs, err_keys)
        pct = float(100.0 * np.mean(ref <= v))
        f = lambda x: f"{x:.{dec}f}"
        measured = f"{label} {'medido' if key == 'arm_slot_dataset_deg' else 'medida'} {f(v)} {unit}"
        if v > hi + err:
            status = "fuera"
            text = (f"{measured}, por encima de cualquier lanzador LMB (p99.5 de la LMB ~{f(hi)} {unit}): revisa la "
                    "estatura y la cámara, o es una mecánica atípica")
        elif v < lo - err:
            status = "fuera"
            text = (f"{measured}, por debajo de cualquier lanzador LMB (p0.5 de la LMB ~{f(lo)} {unit}): revisa la "
                    "estatura y la cámara, o es una mecánica atípica")
        else:
            status = "dentro"
            text = f"{measured}, dentro del rango de la LMB (percentil ~{pct:.0f} de la liga)"
        report[key] = {"status": status, "value": v, "percentile": pct, "low": lo, "high": hi, "err": err,
                       "n_reference": int(ref.size), "text": text}
    report["rel_side_ft"] = {
        "status": "sin referencia", "value": None if rel_side_ft is None else float(rel_side_ft),
        "percentile": None, "low": None, "high": None,
        "err": _sys_err(sys_errs, ("rel_side_ft", "rel_side_sys_err_ft")), "n_reference": 0,
        "text": ("Lado de soltura: el catálogo de la LMB no incluye el lado, así que no hay con qué compararlo; "
                 "queda solo con su error sistemático."),
    }
    outside = [k for k, r in report.items() if r["status"] == "fuera"]
    report["outside"] = outside
    report["any_outside"] = bool(outside)
    return report

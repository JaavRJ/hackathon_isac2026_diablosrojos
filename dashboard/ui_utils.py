"""Logica pura del dashboard Diablos AeroVision (HACK-015).

Este modulo NO importa Streamlit: se ejercita en modo headless (tests) y desde
``dashboard/app.py``. Carga unicamente ``models/*.joblib`` y ``reports/**/*.csv`` por rutas
relativas a la raiz del proyecto (nunca datos crudos de pitcheo).

Contenido
---------
* Carga: :func:`load_models`, :func:`load_reports`, :func:`load_all`.
* Simulador (Vista 1): :func:`build_pitch_frame`, :func:`simulate_pitch`,
  :func:`explain_pitch_shap`, :func:`decomposition_payload`.
* Mapa de movimiento (Vista 2): :func:`movement_heatmap_matrix`.
* Leaderboard y tarjeta de lanzador (Vista 3): :func:`attach_role`,
  :func:`filter_leaderboard`, :func:`pitcher_card`.
* Figuras Plotly: ``fig_*``.

Supuestos y limitaciones (se muestran tambien en la UI)
-------------------------------------------------------
* Los valores ingresados se interpretan como medidos a nivel del mar (rho = 1.00).
* ``EffectiveVelo``, ``ZoneSpeed``, ``ZoneTime``, ``VertApprAngle`` son ESTIMACIONES
  fisicas a partir de lo ingresado; ``VertRelAngle`` y ``HorzRelAngle_norm`` usan la mediana
  de entrenamiento del modelo Stuff+. No son mediciones.
* Location+ no depende de la densidad del aire (ver ``src/models/location_pitching.py``).
* Los entrenamientos usan year <= 2025; 2026 es fuera de muestra.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from dashboard import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_FILES = {
    "movement_translator": "movement_translator.joblib",
    "stuff_model": "stuff_model.joblib",
    "location_model": "location_model.joblib",
    "pitching_model": "pitching_model.joblib",
}
REPORT_FILES = {
    "leaderboard": Path("leaderboards") / "pitcher_season_leaderboard.csv",
    "arsenal": Path("leaderboards") / "pitcher_arsenal_breakdown.csv",
    "movement_grid": Path("leaderboards") / "movement_map_grid.csv",
    "decomposition": Path("altitude") / "channel_decomposition.csv",
    "holdout_metrics": Path("validation") / "holdout_metrics.csv",
    "stability": Path("validation") / "year_to_year_stability.csv",
}

# Los colores viven en dashboard/theme.py (el rojo de marca nunca se usa como color de datos).

SEA_LEVEL_RHO = 1.0
PITCH_TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
FASTBALL_IDS = ("Four-Seam", "Sinker", "Cutter")
DEFAULT_ZONE_SPEED_RATIO = 0.92  # respaldo si no se puede estimar desde el arsenal
ROLE_MIN_PITCH_TYPES = 3  # heuristica de rol por repertorio
ROLE_MIN_USAGE = 0.10

# Etiquetas de la descomposicion de tres terminos (HACK-013)
LABEL_TOTAL = "Total observado (intra-pitcher, Altitud Extrema - Nivel del Mar)"
LABEL_STUFF = "Canal Stuff"
LABEL_PARK = "Canal Parque (BIP)"
LABEL_RESIDUAL = "Residuo no atribuido (no se atribuye a ningún canal)"
CHANNEL_LABELS = {
    "total": LABEL_TOTAL,
    "stuff": LABEL_STUFF,
    "park": LABEL_PARK,
    "unallocated": LABEL_RESIDUAL,
}
CHANNEL_ORDER = ["total", "stuff", "park", "unallocated"]

FEATURE_LABELS = {
    "RelSpeed": "Velocidad de liberación",
    "EffectiveVelo": "Velocidad efectiva",
    "ZoneSpeed": "Velocidad en la zona",
    "SpinRate": "Giro (rpm)",
    "SpinAxis_sin": "Eje de giro (seno)",
    "SpinAxis_cos": "Eje de giro (coseno)",
    "RelHeight": "Altura de liberación",
    "RelSide_norm": "Lado de liberación",
    "Extension": "Extensión",
    "InducedVertBreak": "Quiebre vertical inducido (IVB)",
    "HorzBreak_norm": "Quiebre horizontal (HB)",
    "TotalInducedBreak": "Quiebre total inducido",
    "BreakAngle_deg": "Ángulo de quiebre",
    "VertRelAngle": "Ángulo vertical de liberación",
    "HorzRelAngle_norm": "Ángulo horizontal de liberación",
    "SpinToBreakRatio": "Eficiencia de giro (proxy)",
    "arm_slot_deg": "Ángulo del brazo (arm slot)",
    "PitcherThrows": "Mano del lanzador",
    "velo_diff_vs_fb": "Diferencia de velocidad vs. recta",
    "ivb_diff_vs_fb": "Diferencia de IVB vs. recta",
    "hb_diff_vs_fb": "Diferencia de HB vs. recta",
    "tunnel_dist_release_in": "Distancia de túnel (liberación)",
    "tunnel_sim_release": "Similitud de túnel (liberación)",
    "tunnel_dist_170ms_in": "Distancia de túnel (170 ms)",
    "tunnel_sim_170ms": "Similitud de túnel (170 ms)",
    "pitch_usage_pct": "Uso del pitcheo en el repertorio",
}

COLUMN_LABELS = {
    "pitcher_anon_id": "Pitcher",
    "year": "Temporada",
    "n_pitches": "Pitcheos",
    "arm_slot_deg_mean": "Arm slot (°)",
    "rel_speed_mean": "Velocidad (mph)",
    "effective_velo_mean": "Velocidad efectiva (mph)",
    "stuff_plus_neutral": "Stuff+ Nivel del Mar",
    "stuff_plus_cdmx": "Stuff+ CDMX",
    "delta_stuff_cdmx": "ΔStuff+ CDMX",
    "location_plus_neutral": "Location+ Nivel del Mar",
    "location_plus_cdmx": "Location+ CDMX",
    "pitching_plus_neutral": "Pitching+ Nivel del Mar",
    "pitching_plus_cdmx": "Pitching+ CDMX",
    "gb_pct_pitches": "% rodado (por pitcheo)",
    "gb_rate_bip": "% rodado (en juego)",
    "runs_prevented_xrv": "Carreras prevenidas (xRV)",
    "runs_prevented_per_100": "Carreras prevenidas / 100",
    "exp_xrv_per_100": "xRV esperado / 100",
    "pitcher_throws": "Mano",
    "rol": "Rol (aprox.)",
    "pitch_id": "Pitcheo",
    "usage_pct": "Uso",
    "spin_rate_mean": "Giro (rpm)",
    "ivb_neutral": "IVB Nivel del Mar (in)",
    "ivb_cdmx": "IVB CDMX (in)",
    "hb_norm_neutral": "HB Nivel del Mar (in)",
    "hb_norm_cdmx": "HB CDMX (in)",
    "canal_stuff": "Canal Stuff (xRV/pitcheo)",
    "canal_parque_bip": "Canal Parque BIP (xRV/pitcheo)",
    "residuo_no_atribuido": "Residuo no atribuido (xRV/pitcheo)",
}

HAND_DISPLAY = {"Right": "Derecho", "Left": "Zurdo"}  # solo visualizacion; los datos conservan Right/Left

SORT_OPTIONS = {"Stuff+ CDMX": "stuff_plus_cdmx", "ΔStuff+ CDMX": "delta_stuff_cdmx"}


# ----------------------------------------------------------------------------
# Carga (una sola vez; app.py la envuelve en st.cache_resource / st.cache_data)
# ----------------------------------------------------------------------------
def ensure_import_path() -> None:
    """Garantiza que ``src`` sea importable (necesario para deserializar los joblib)."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_models(models_dir: Path | str | None = None) -> dict[str, Any]:
    """Carga ``models/*.joblib`` (traductor, Stuff+, Location+, Pitching+)."""
    ensure_import_path()
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    return {name: joblib.load(base / fname) for name, fname in MODEL_FILES.items()}


def load_reports(reports_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Carga los CSV agregados necesarios. Un archivo ausente produce un DataFrame vacio."""
    base = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    out: dict[str, pd.DataFrame] = {}
    for key, rel in REPORT_FILES.items():
        path = base / rel
        out[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    out["leaderboard"] = attach_role(out["leaderboard"], out["arsenal"])
    return out


def estimate_zone_speed_ratio(arsenal: pd.DataFrame) -> float:
    """Mediana de zone_speed_neutral / rel_speed_mean del arsenal (estimacion de ZoneSpeed)."""
    if {"zone_speed_neutral", "rel_speed_mean"} <= set(arsenal.columns) and len(arsenal):
        ratio = (arsenal["zone_speed_neutral"] / arsenal["rel_speed_mean"]).replace([np.inf, -np.inf], np.nan)
        med = ratio.median()
        if pd.notna(med) and med > 0:
            return float(med)
    return DEFAULT_ZONE_SPEED_RATIO


def build_meta(models: dict[str, Any], reports: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Metadatos derivados: rho de CDMX (mediana de Altitud Extrema), razon de ZoneSpeed, medianas."""
    return {
        "rho_cdmx": float(getattr(models["stuff_model"], "rho_extreme_median_", 0.852)),
        "zone_speed_ratio": estimate_zone_speed_ratio(reports["arsenal"]),
        "feature_medians": dict(getattr(models["stuff_model"], "feature_medians_", {})),
    }


def load_all(models_dir: Path | str | None = None, reports_dir: Path | str | None = None) -> dict[str, Any]:
    """Carga modelos + reportes y metadatos derivados (rho de CDMX, razon de ZoneSpeed)."""
    models = load_models(models_dir)
    reports = load_reports(reports_dir)
    return {"models": models, "reports": reports, "meta": build_meta(models, reports)}


# ----------------------------------------------------------------------------
# Simulador de pitcheo (Vista 1)
# ----------------------------------------------------------------------------
def hand_sign(throws: str) -> float:
    return -1.0 if str(throws).upper().startswith("L") else 1.0


def zone_geometry(plate_side: float, plate_height: float) -> dict[str, float]:
    """Geometria de zona (misma formula que ``src/features/physical.py``)."""
    dx_edge = abs(plate_side) - 0.833
    dz_edge = max(1.5 - plate_height, plate_height - 3.5)
    outside = float(np.hypot(max(0.0, dx_edge), max(0.0, dz_edge)))
    inside_rect = dx_edge <= 0.0 and dz_edge <= 0.0
    dist_edge = min(-dx_edge, -dz_edge) if inside_rect else outside
    shadow = (abs(dx_edge) <= 0.25 and 1.25 <= plate_height <= 3.75) or (
        abs(dz_edge) <= 0.25 and -1.08 <= plate_side <= 1.08
    )
    in_zone = int(inside_rect)
    if in_zone:
        col = int(np.clip(np.floor((plate_side + 0.833) / (1.666 / 3.0)), 0, 2))
        row = int(np.clip(np.floor((plate_height - 1.5) / (2.0 / 3.0)), 0, 2))
        quad = row * 3 + col + 1
    else:
        quad = (11 if plate_side >= 0.0 else 10) if plate_height >= 2.5 else (13 if plate_side >= 0.0 else 12)
    return {
        "in_strike_zone": in_zone,
        "outside_strike_zone": 1 - in_zone,
        "dist_from_zone_center": float(np.hypot(plate_side, plate_height - 2.5)),
        "dist_to_zone_edge": float(dist_edge),
        "is_shadow_zone": int(shadow),
        "zone_quadrant": int(quad),
    }


INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "rel_speed": (60.0, 105.0), "spin_rate": (800.0, 3400.0), "spin_axis": (0.0, 360.0),
    "ivb": (-25.0, 25.0), "hb_norm": (-25.0, 25.0), "arm_slot": (15.0, 80.0), "rel_height": (3.0, 8.0), "rel_side": (0.0, 5.0),
    "extension": (4.0, 8.0), "fb_velo": (55.0, 110.0), "fb_ivb": (-30.0, 30.0), "fb_hb_norm": (-30.0, 30.0),
    "tunnel_sim_release": (0.0, 1.0), "tunnel_sim_170ms": (0.0, 1.0), "pitch_usage_pct": (0.01, 1.0),
    "plate_side": (-2.0, 2.0), "plate_height": (0.0, 5.0),
}


def clip_inputs(values: dict[str, Any]) -> dict[str, Any]:
    """Acota las entradas numericas a los limites de los widgets (evita valores fuera de rango)."""
    out = dict(values)
    for k, (lo, hi) in INPUT_BOUNDS.items():
        if k in out and out[k] is not None:
            out[k] = float(np.clip(float(out[k]), lo, hi))
    return out


def default_spin_axis(medians: dict[str, float], throws: str) -> float:
    """Eje de giro (grados, convencion bruta) a partir de la mediana de seno/coseno normalizados."""
    s = float(medians.get("SpinAxis_sin", -0.532)) * hand_sign(throws)
    c = float(medians.get("SpinAxis_cos", -0.68))
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def default_pitch_inputs(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Entrada por defecto: una recta de cuatro costuras tipica de la LMB."""
    d = _default_pitch_inputs_raw(meta)
    d["arm_slot"] = round(arm_slot_from(d["rel_height"], d["rel_side"]), 1)
    return d


def _default_pitch_inputs_raw(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    med = (meta or {}).get("feature_medians", {})
    throws = "Right"
    return {
        "pitch_id": "Four-Seam",
        "pitcher_throws": throws,
        "rel_speed": round(float(med.get("RelSpeed", 90.0)), 1),
        "spin_rate": float(round(med.get("SpinRate", 2300.0), -1)),
        "spin_axis": round(default_spin_axis(med, throws), 0),
        "ivb": 16.0,
        "hb_norm": 8.0,
        "rel_height": round(float(med.get("RelHeight", 5.8)), 2),
        "rel_side": round(float(med.get("RelSide_norm", 1.75)), 2),
        "extension": round(float(med.get("Extension", 5.8)), 2),
        "arm_slot": 0.0,  # se completa abajo con la relacion altura/lado
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "batter_side": "Right",
        "plate_side": 0.0,
        "plate_height": 2.5,
        "is_primary_fb": True,
        "fb_velo": round(float(med.get("RelSpeed", 90.0)), 1),
        "fb_ivb": 16.0,
        "fb_hb_norm": 8.0,
        "tunnel_sim_release": round(float(med.get("tunnel_sim_release", 0.46)), 3),
        "tunnel_sim_170ms": round(float(med.get("tunnel_sim_170ms", 0.15)), 3),
        "pitch_usage_pct": round(float(med.get("pitch_usage_pct", 0.38)), 3),
    }


def build_pitch_frame(inputs: dict[str, Any], meta: dict[str, Any] | None = None) -> pd.DataFrame:
    """Construye la fila de una fila con todas las columnas que requieren los cuatro modelos."""
    meta = meta or {}
    med = meta.get("feature_medians", {})
    zs_ratio = float(meta.get("zone_speed_ratio", DEFAULT_ZONE_SPEED_RATIO))
    d = {**default_pitch_inputs(meta), **inputs}

    throws = str(d["pitcher_throws"])
    sign = hand_sign(throws)
    speed = float(d["rel_speed"])
    ext = float(d["extension"])
    ivb = float(d["ivb"])
    hb_norm = float(d["hb_norm"])
    rel_height = float(d["rel_height"])
    rel_side_norm = float(d["rel_side"])
    rel_side_raw = rel_side_norm * sign
    plate_side = float(d["plate_side"])
    plate_height = float(d["plate_height"])

    zone_speed = speed * zs_ratio
    effective = speed + (ext - 6.0) * 1.7  # estimacion: 1.7 mph por pie de extension
    mean_fps = 1.4667 * 0.5 * (speed + zone_speed)
    zone_time = max((60.5 - ext) / mean_fps, 0.35)
    spin_rad = np.radians(float(d["spin_axis"]))
    total_break = float(np.hypot(ivb, hb_norm))
    arm_slot = float(np.clip(np.degrees(np.arctan2(rel_height - 4.0, abs(rel_side_raw) + 0.5)), 0.0, 90.0))

    # Angulo de aproximacion vertical (estimacion): vz en el plato con gravedad e IVB.
    t = (60.5 - ext) / (1.4667 * speed * 0.96)
    vz_plate = (plate_height - rel_height) / t + (-16.1 * t + ivb / (12.0 * t))
    vy_plate = 1.4667 * zone_speed
    vert_appr = float(np.degrees(np.arctan2(vz_plate, vy_plate)))

    if bool(d["is_primary_fb"]) and str(d["pitch_id"]) in FASTBALL_IDS:
        fb_velo, fb_ivb, fb_hb = speed, ivb, hb_norm
    else:
        fb_velo, fb_ivb, fb_hb = float(d["fb_velo"]), float(d["fb_ivb"]), float(d["fb_hb_norm"])
    tun_rel = float(d["tunnel_sim_release"])
    tun_170 = float(d["tunnel_sim_170ms"])

    balls, strikes = int(d["balls"]), int(d["strikes"])
    geo = zone_geometry(plate_side, plate_height)
    row = {
        "PitchUID": "SIMULADOR",
        "year": 2026,
        "pitch_id": str(d["pitch_id"]),
        "prev_pitch_id": "NONE",
        "PitcherThrows": throws,
        "BatterSide": str(d["batter_side"]),
        "is_same_hand": int(str(d["batter_side"]) == throws),
        "rho_hat": SEA_LEVEL_RHO,
        "RelSpeed": speed,
        "EffectiveVelo": effective,
        "ZoneSpeed": zone_speed,
        "SpinRate": float(d["spin_rate"]),
        "SpinAxis": float(d["spin_axis"]),
        "SpinAxis_sin": float(np.sin(spin_rad) * sign),
        "SpinAxis_cos": float(np.cos(spin_rad)),
        "RelHeight": rel_height,
        "RelSide": rel_side_raw,
        "RelSide_norm": rel_side_norm,
        "Extension": ext,
        "InducedVertBreak": ivb,
        "HorzBreak": hb_norm * sign,
        "HorzBreak_norm": hb_norm,
        "TotalInducedBreak": total_break,
        "BreakAngle_deg": float(np.degrees(np.arctan2(hb_norm, ivb))),
        "VertRelAngle": float(med.get("VertRelAngle", -1.2)),
        "HorzRelAngle_norm": float(med.get("HorzRelAngle_norm", -2.5)),
        "VertApprAngle": vert_appr,
        "HorzApprAngle_norm": 0.0,
        "SpinToBreakRatio": total_break / ((float(d["spin_rate"]) / 1000.0) * zone_time + 1e-5),
        "arm_slot_deg": arm_slot,
        "velo_diff_vs_fb": speed - fb_velo,
        "ivb_diff_vs_fb": ivb - fb_ivb,
        "hb_diff_vs_fb": hb_norm - fb_hb,
        "tunnel_dist_release_in": float(-6.0 * np.log(max(tun_rel, 1e-6))),
        "tunnel_sim_release": tun_rel,
        "tunnel_dist_170ms_in": float(-6.0 * np.log(max(tun_170, 1e-6))),
        "tunnel_sim_170ms": tun_170,
        "pitch_usage_pct": float(d["pitch_usage_pct"]),
        "PlateLocSide": plate_side,
        "PlateLocSide_norm": plate_side * sign,
        "PlateLocHeight": plate_height,
        "Balls": balls,
        "Strikes": strikes,
        "Outs": int(d["outs"]),
        "Inning": 5,
        "count": f"{balls}-{strikes}",
        **geo,
    }
    return pd.DataFrame([row])


def _scalar(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].iloc[0])


def simulate_pitch(models: dict[str, Any], inputs: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Puntua un pitcheo en Nivel del Mar (rho=1.00) y en CDMX (rho = mediana de Altitud Extrema).

    No calcula SHAP (ver :func:`explain_pitch_shap`). ``delta_*`` = CDMX - Nivel del Mar.
    """
    meta = meta or {}
    stuff, loc, pit, trans = (models["stuff_model"], models["location_model"],
                              models["pitching_model"], models["movement_translator"])
    rho = float(meta.get("rho_cdmx", getattr(stuff, "rho_extreme_median_", 0.852)))
    frame = build_pitch_frame(inputs, meta)

    s = stuff.predict_scores(frame, rho_cdmx=rho)
    l = loc.predict_scores(frame, rho_cdmx=rho)
    p = pit.predict_all_targets(frame, rho_cdmx=rho)
    mv_sea = trans.translate_movement(frame, target_rho=SEA_LEVEL_RHO)
    mv_cdmx = trans.translate_movement(frame, target_rho=rho)

    neutral = {
        "stuff": _scalar(s, "stuff_plus_neutral"),
        "location": _scalar(l, "location_plus_neutral"),
        "pitching": _scalar(p, "pitching_plus_neutral"),
        "ivb": _scalar(mv_sea, "InducedVertBreak"),
        "hb": _scalar(mv_sea, "HorzBreak_norm"),
        "zone_speed": _scalar(mv_sea, "ZoneSpeed"),
        "rho": SEA_LEVEL_RHO,
    }
    cdmx = {
        "stuff": _scalar(s, "stuff_plus_cdmx"),
        "location": _scalar(l, "location_plus_cdmx"),
        "pitching": _scalar(p, "pitching_plus_cdmx"),
        "ivb": _scalar(mv_cdmx, "InducedVertBreak"),
        "hb": _scalar(mv_cdmx, "HorzBreak_norm"),
        "zone_speed": _scalar(mv_cdmx, "ZoneSpeed"),
        "rho": rho,
    }
    return {
        "frame": frame,
        "neutral": neutral,
        "cdmx": cdmx,
        "delta_stuff": cdmx["stuff"] - neutral["stuff"],
        "delta_location": cdmx["location"] - neutral["location"],
        "delta_pitching": cdmx["pitching"] - neutral["pitching"],
        "rho_cdmx": rho,
    }


def explain_pitch_shap(models: dict[str, Any], frame: pd.DataFrame, target_rho: float) -> dict[str, Any]:
    """SHAP aditivo (TreeSHAP) bajo demanda, solo para el pitcheo ingresado."""
    return models["stuff_model"].explain_pitch(frame, target_rho=float(target_rho))


def shap_table(explanation: dict[str, Any], top_n: int = 10) -> pd.DataFrame:
    """Tabla (variable en espanol, contribucion en puntos de Stuff+) ordenada por |impacto|."""
    items = explanation["shap_values"]
    df = pd.DataFrame({"feature": list(items.keys()), "contribucion": list(items.values())})
    df["variable"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    df = df.reindex(df["contribucion"].abs().sort_values(ascending=False).index).head(top_n)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Descomposicion de tres terminos (poblacional, HACK-013)
# ----------------------------------------------------------------------------
def decomposition_payload(decomp: pd.DataFrame) -> dict[str, Any]:
    """Payload que renderiza la UI: filas total / stuff / park / unallocated con share y CI.

    El residuo conserva su propia etiqueta y nunca se suma al Canal Parque.
    """
    rows: list[dict[str, Any]] = []
    if len(decomp) and "channel" in decomp.columns:
        by_channel = decomp.set_index("channel")
        for ch in CHANNEL_ORDER:
            if ch not in by_channel.index:
                continue
            r = by_channel.loc[ch]
            rows.append({
                "channel": ch,
                "label": CHANNEL_LABELS[ch],
                "estimate": float(r["estimate"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "share_of_total": float(r["share_of_total"]) if "share_of_total" in r and pd.notna(r["share_of_total"]) else float("nan"),
                "n_pitchers": int(r["n_pitchers"]) if "n_pitchers" in r and pd.notna(r["n_pitchers"]) else None,
            })
    stuff = next((r for r in rows if r["channel"] == "stuff"), None)
    stuff_note = ""
    if stuff is not None and np.isfinite(stuff["share_of_total"]) and stuff["share_of_total"] < 0.25:
        stuff_note = (
            f"El Canal Stuff explica solo {stuff['share_of_total']:.1%} del total observado; "
            "la mayor parte de la diferencia queda en el Canal Parque (BIP) y en el residuo no atribuido."
        )
    return {
        "rows": rows,
        "labels": [r["label"] for r in rows],
        "stuff_note": stuff_note,
        "population_note": (
            "Descomposición poblacional de la liga (unidad: xRV por pitcheo, Altitud Extrema menos Nivel del Mar, "
            "intra-pitcher). No corresponde al pitcheo ingresado ni es una atribución causal."
        ),
    }


# ----------------------------------------------------------------------------
# Mapa de movimiento (Vista 2)
# ----------------------------------------------------------------------------
MAP_METRICS = {
    "Valor a Nivel del Mar": "stuff_plus_neutral",
    "Valor en Harp Helú": "stuff_plus_cdmx",
    "ΔStuff+ (quién gana y quién pierde en altitud)": "delta_stuff_cdmx",
}


def movement_heatmap_matrix(grid: pd.DataFrame, metric: str, pitch_id: str | None = None) -> pd.DataFrame:
    """Matriz IVB (filas) x HB_norm (columnas) del metrico, promedio ponderado por n_pitches."""
    if grid.empty or metric not in grid.columns:
        return pd.DataFrame()
    g = grid if not pitch_id or pitch_id == "Todos" else grid[grid["pitch_id"] == pitch_id]
    if g.empty:
        return pd.DataFrame()
    w = g["n_pitches"].astype(float)
    tmp = g.assign(_wv=g[metric] * w, _w=w)
    agg = tmp.groupby(["IVB_bin", "HB_norm_bin"], observed=True)[["_wv", "_w"]].sum()
    agg["v"] = agg["_wv"] / agg["_w"]
    return agg["v"].unstack("HB_norm_bin").sort_index(ascending=True)


# ----------------------------------------------------------------------------
# Leaderboard y tarjeta de lanzador (Vista 3)
# ----------------------------------------------------------------------------
def attach_role(leaderboard: pd.DataFrame, arsenal: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``rol``. Usa ``role`` si existe; si no, heuristica por repertorio (aproximada).

    Abridor (aprox.) = al menos 3 pitcheos con uso >= 10 % en la temporada. No es un dato oficial.
    """
    if leaderboard.empty:
        return leaderboard
    lb = leaderboard.copy()
    if "role" in lb.columns:
        lb["rol"] = lb["role"]
        return lb
    if arsenal.empty or "usage_pct" not in arsenal.columns:
        lb["rol"] = "Sin dato"
        return lb
    n_types = (arsenal[arsenal["usage_pct"] >= ROLE_MIN_USAGE]
               .groupby(["pitcher_anon_id", "year"]).size().rename("_n_types").reset_index())
    lb = lb.merge(n_types, on=["pitcher_anon_id", "year"], how="left")
    lb["rol"] = np.where(lb["_n_types"].fillna(0) >= ROLE_MIN_PITCH_TYPES, "Abridor (aprox.)", "Relevista (aprox.)")
    return lb.drop(columns=["_n_types"])


def filter_leaderboard(
    lb: pd.DataFrame,
    years: Iterable[int] | None = None,
    roles: Iterable[str] | None = None,
    throws: Iterable[str] | None = None,
    min_pitches: int = 0,
    sort_by: str = "stuff_plus_cdmx",
    ascending: bool = False,
) -> pd.DataFrame:
    """Filtra por temporada, rol, mano y minimo de pitcheos; ordena por Stuff+ CDMX o Delta."""
    if lb.empty:
        return lb
    out = lb
    if years:
        out = out[out["year"].isin(list(years))]
    if roles and "rol" in out.columns:
        out = out[out["rol"].isin(list(roles))]
    if throws and "pitcher_throws" in out.columns:
        out = out[out["pitcher_throws"].isin(list(throws))]
    out = out[out["n_pitches"] >= int(min_pitches)]
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending, kind="mergesort")
    return out.reset_index(drop=True)


def display_table(df: pd.DataFrame, columns: list[str] | None = None, decimals: int = 2) -> pd.DataFrame:
    """Renombra a espanol y redondea para mostrar (sin alterar los valores del original)."""
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    out = df[cols].copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(decimals)
    return out.rename(columns={c: COLUMN_LABELS.get(c, c) for c in cols})


LEADERBOARD_COLUMNS = [
    "pitcher_anon_id", "year", "rol", "pitcher_throws", "n_pitches",
    "stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx",
    "location_plus_cdmx", "pitching_plus_neutral", "pitching_plus_cdmx",
    "runs_prevented_per_100", "gb_rate_bip",
]


def pitcher_card(lb: pd.DataFrame, arsenal: pd.DataFrame, pitcher_id: str, year: int) -> dict[str, Any]:
    """Tarjeta de lanzador: fila del leaderboard + arsenal + (si existe) tres terminos del pitcher."""
    row = lb[(lb["pitcher_anon_id"] == pitcher_id) & (lb["year"] == year)]
    ars = arsenal[(arsenal["pitcher_anon_id"] == pitcher_id) & (arsenal["year"] == year)] if len(arsenal) else arsenal
    summary = row.iloc[0].to_dict() if len(row) else {}
    channels = {}
    for col in ("canal_stuff", "canal_parque_bip", "residuo_no_atribuido"):
        val = summary.get(col)
        channels[col] = float(val) if val is not None and pd.notna(val) else None
    return {
        "summary": summary,
        "arsenal": ars.sort_values("n_pitches", ascending=False).reset_index(drop=True) if len(ars) else ars,
        "channels": channels,
    }


def inputs_from_arsenal_row(
    arsenal_row: pd.Series, summary: dict[str, Any] | None, meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Pre-llena el simulador con un pitcheo real del arsenal (valores a nivel del mar).

    Tolera la ausencia de ``PitcherThrows``, ``arm_slot_deg``, ``RelHeight`` y ``fb_velo``
    (HACK-014 pendiente): en ese caso usa la mano del leaderboard y medianas de entrenamiento.
    """
    base = default_pitch_inputs(meta)
    summary = summary or {}
    r = arsenal_row

    def has(col: str) -> bool:
        return col in r.index and pd.notna(r[col])

    throws = r["PitcherThrows"] if has("PitcherThrows") else summary.get("pitcher_throws", "Right")
    throws = "Left" if str(throws).upper().startswith("L") else "Right"
    base.update(
        pitch_id=str(r["pitch_id"]) if str(r["pitch_id"]) in PITCH_TYPES else base["pitch_id"],
        pitcher_throws=throws,
        rel_speed=round(float(r["rel_speed_mean"]), 1) if has("rel_speed_mean") else base["rel_speed"],
        spin_rate=float(round(r["spin_rate_mean"], 0)) if has("spin_rate_mean") else base["spin_rate"],
        ivb=round(float(r["ivb_neutral"]), 1) if has("ivb_neutral") else base["ivb"],
        hb_norm=round(float(r["hb_norm_neutral"]), 1) if has("hb_norm_neutral") else base["hb_norm"],
        spin_axis=round(default_spin_axis((meta or {}).get("feature_medians", {}), throws), 0),
    )
    if has("RelHeight"):
        base["rel_height"] = round(float(r["RelHeight"]), 2)
        arm = r["arm_slot_deg"] if has("arm_slot_deg") else summary.get("arm_slot_deg_mean")
        if arm is not None and pd.notna(arm) and 5.0 < float(arm) < 85.0:
            side = (float(r["RelHeight"]) - 4.0) / np.tan(np.radians(float(arm))) - 0.5
            base["rel_side"] = round(float(np.clip(side, 0.3, 4.0)), 2)
    if has("tunnel_sim_release"):
        base["tunnel_sim_release"] = round(float(r["tunnel_sim_release"]), 3)
    if has("tunnel_sim_170ms"):
        base["tunnel_sim_170ms"] = round(float(r["tunnel_sim_170ms"]), 3)
    if has("usage_pct"):
        base["pitch_usage_pct"] = round(float(r["usage_pct"]), 3)
    is_fb = base["pitch_id"] in FASTBALL_IDS
    base["is_primary_fb"] = bool(is_fb)
    if not is_fb and has("fb_velo"):
        base["fb_velo"] = round(float(r["fb_velo"]), 1)
    base["fb_ivb"], base["fb_hb_norm"] = base["ivb"], base["hb_norm"]
    arm_col = r["arm_slot_deg"] if has("arm_slot_deg") else summary.get("arm_slot_deg_mean")
    if arm_col is not None and pd.notna(arm_col) and 5.0 < float(arm_col) < 85.0:
        base["arm_slot"] = float(arm_col)
    else:
        base["arm_slot"] = arm_slot_from(float(base["rel_height"]), float(base["rel_side"]))
    return clip_inputs(base)


# ----------------------------------------------------------------------------
# Asistente de simulacion: bandas "Zona LMB", presets y contexto (HACK-019)
# ----------------------------------------------------------------------------
#: Sliders principales del Simulador: (clave, icono, etiqueta, unidad, paso, decimales, columna del catalogo).
PRIMARY_SLIDERS: list[tuple[str, str, str, str, float, int, str]] = [
    ("rel_speed", "gauge", "Velocidad", "mph", 0.1, 1, "rel_speed_mean"),
    ("spin_rate", "spin", "Giro", "rpm", 10.0, 0, "spin_rate_mean"),
    ("ivb", "up", "Subida (IVB)", "in", 0.1, 1, "ivb_neutral"),
    ("hb_norm", "right", "Corrida (HB)", "in", 0.1, 1, "hb_norm_neutral"),
    ("arm_slot", "arm", "Arm slot", "°", 0.5, 1, "arm_slot_deg"),
]
BAND_MIN_PITCHES = 50     # filas del catalogo con al menos 50 pitcheos
BAND_MIN_ROWS = 10        # si hay menos filas, se baja el minimo a BAND_FALLBACK_PITCHES
BAND_FALLBACK_PITCHES = 20
BAND_LOW_Q, BAND_HIGH_Q = 0.10, 0.90

#: Nombres en lenguaje llano por pitch_id.
PITCH_NAMES_ES = {
    "Four-Seam": "recta", "Sinker": "sinker", "Cutter": "cutter", "Slider": "slider",
    "Curveball": "curva", "Changeup": "cambio de velocidad", "Splitter": "splitter",
}

#: Mapeo de las claves de ``active_pitch_params`` (docs/DISENO_UI_NIELSEN.md 3.3) a las entradas del simulador.
PARAM_KEYS = {
    "pitch_id": "pitch_id", "PitcherThrows": "pitcher_throws", "RelSpeed": "rel_speed", "SpinRate": "spin_rate",
    "SpinAxis": "spin_axis", "InducedVertBreak": "ivb", "HorzBreak_norm": "hb_norm", "RelHeight": "rel_height",
    "RelSide": "rel_side", "Extension": "extension", "arm_slot_deg": "arm_slot", "fb_velo": "fb_velo",
    "PlateLocSide": "plate_side", "PlateLocHeight": "plate_height", "Balls": "balls", "Strikes": "strikes",
    "Outs": "outs", "BatterSide": "batter_side",
}


def weighted_percentile(values: Iterable[float], weights: Iterable[float], q: float) -> float:
    """Percentil ponderado (interpolacion lineal sobre la CDF acumulada centrada); ``q`` en [0, 1]."""
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cdf, v))


def _catalog_rows(arsenal: pd.DataFrame, pitch_id: str) -> tuple[pd.DataFrame, int]:
    """Filas del catalogo para un pitch_id con n_pitches >= 50 (o >= 20 si hay muy pocas)."""
    if arsenal.empty or "pitch_id" not in arsenal.columns:
        return pd.DataFrame(), BAND_MIN_PITCHES
    g = arsenal[arsenal["pitch_id"] == pitch_id]
    sub = g[g["n_pitches"] >= BAND_MIN_PITCHES]
    if len(sub) >= BAND_MIN_ROWS:
        return sub, BAND_MIN_PITCHES
    return g[g["n_pitches"] >= BAND_FALLBACK_PITCHES], BAND_FALLBACK_PITCHES


def league_bands(arsenal: pd.DataFrame, pitch_id: str) -> dict[str, Any]:
    """Banda "Zona LMB" p10-p90 REAL por ``pitch_id``, ponderada por ``n_pitches``, desde el catalogo."""
    rows, min_p = _catalog_rows(arsenal, pitch_id)
    bands: dict[str, tuple[float, float]] = {}
    for key, _ic, _lab, _unit, _step, _dec, col in PRIMARY_SLIDERS:
        if col in rows.columns and len(rows):
            lo = weighted_percentile(rows[col], rows["n_pitches"], BAND_LOW_Q)
            hi = weighted_percentile(rows[col], rows["n_pitches"], BAND_HIGH_Q)
            if np.isfinite(lo) and np.isfinite(hi):
                bands[key] = (lo, hi)
    return {"pitch_id": pitch_id, "n_rows": int(len(rows)), "min_pitches": min_p, "bands": bands}


def band_note(info: dict[str, Any]) -> str:
    """Texto de procedencia de la banda (tooltip y nota), con el n real."""
    return (f"Zona LMB = p10–p90 real de {info['n_rows']} filas {info['pitch_id']} "
            f"(n_pitches ≥ {info['min_pitches']}), ponderado por n_pitches. Fuente: pitcher_arsenal_breakdown.csv.")


def arm_slot_from(rel_height: float, rel_side: float) -> float:
    """Arm slot (grados) a partir de altura y lado de liberacion (misma relacion que ``build_pitch_frame``)."""
    return float(np.clip(np.degrees(np.arctan2(rel_height - 4.0, abs(rel_side) + 0.5)), 0.0, 90.0))


def rel_side_from(rel_height: float, arm_slot: float) -> float:
    """Lado de liberacion (pies) que reproduce un arm slot dado a una altura de liberacion."""
    arm = float(np.clip(arm_slot, 5.0, 85.0))
    side = (float(rel_height) - 4.0) / np.tan(np.radians(arm)) - 0.5
    return float(np.clip(side, 0.3, 4.0))


def _step_round(x: float, step: float) -> float:
    return float(round(round(x / step) * step, 6))


def compute_presets(arsenal: pd.DataFrame, meta: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Presets del asistente calculados con percentiles REALES del catalogo (nada inventado).

    * ``alta``: Four-Seam; velocidad y giro en p75, IVB en p90, HB y arm slot en p50.
    * ``sweeper``: Slider; HB en p10 (mayor corrida hacia el lado del guante), el resto en p50.
    * ``splitter``: Splitter "resiliente"; mediana de las filas Splitter cuyo dStuff+ CDMX esta en el cuartil
      superior (las que mejor sostienen su Stuff+ en altitud).

    Devuelve ``{id: {label, icon, pitch_id, inputs, description, n_rows}}`` (vacio si no hay catalogo).
    """
    if arsenal.empty:
        return {}
    base = default_pitch_inputs(meta)
    fb_rows, _ = _catalog_rows(arsenal, "Four-Seam")
    fb_ref: dict[str, float] = {}
    if len(fb_rows):
        for key, col in (("fb_velo", "rel_speed_mean"), ("fb_ivb", "ivb_neutral"), ("fb_hb_norm", "hb_norm_neutral")):
            fb_ref[key] = weighted_percentile(fb_rows[col], fb_rows["n_pitches"], 0.5)

    def build(pitch_id: str, quant: dict[str, float], rows: pd.DataFrame | None = None) -> tuple[dict[str, Any], int]:
        r = _catalog_rows(arsenal, pitch_id)[0] if rows is None else rows
        vals: dict[str, Any] = {}
        for key, _ic, _lab, _unit, step, _dec, col in PRIMARY_SLIDERS:
            if len(r) and col in r.columns:
                vals[key] = _step_round(weighted_percentile(r[col], r["n_pitches"], quant.get(key, 0.5)), step)
        inp = {**base, "pitch_id": pitch_id, **vals}
        if pitch_id in FASTBALL_IDS:
            inp.update(is_primary_fb=True, fb_velo=inp["rel_speed"], fb_ivb=inp["ivb"], fb_hb_norm=inp["hb_norm"])
        else:
            inp["is_primary_fb"] = False
            for k, v in fb_ref.items():
                inp[k] = _step_round(v, 0.1)
        if "arm_slot" not in inp:
            inp["arm_slot"] = arm_slot_from(float(inp["rel_height"]), float(inp["rel_side"]))
        return clip_inputs(inp), int(len(r))

    out: dict[str, dict[str, Any]] = {}
    inp, n = build("Four-Seam", {"rel_speed": .75, "spin_rate": .75, "ivb": .90, "hb_norm": .5, "arm_slot": .5})
    out["alta"] = {"label": "Recta alta", "icon": "up", "pitch_id": "Four-Seam", "inputs": inp, "n_rows": n,
                   "description": "Four-Seam: velocidad y giro en p75, IVB en p90, HB y arm slot en p50 del catálogo."}
    inp, n = build("Slider", {"hb_norm": .10})
    out["sweeper"] = {"label": "Sweeper", "icon": "wave", "pitch_id": "Slider", "inputs": inp, "n_rows": n,
                      "description": "Slider con la mayor corrida hacia el lado del guante (HB en p10); resto en p50."}
    rows, _ = _catalog_rows(arsenal, "Splitter")
    if len(rows) >= 4:
        top = rows[rows["delta_stuff_cdmx"] >= rows["delta_stuff_cdmx"].quantile(0.75)]
        inp, n = build("Splitter", {}, top if len(top) else rows)
        out["splitter"] = {"label": "Splitter resiliente", "icon": "split", "pitch_id": "Splitter", "inputs": inp,
                           "n_rows": n, "description": "Mediana de los Splitter cuyo ΔStuff+ CDMX está en el cuartil superior."}
    return out


def inputs_to_params(inputs: dict[str, Any]) -> dict[str, Any]:
    """Entradas del simulador -> ``active_pitch_params`` (claves de docs/DISENO_UI_NIELSEN.md 3.3)."""
    return {nk: inputs.get(uk) for nk, uk in PARAM_KEYS.items()}


def validation_anchors(reports: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    """Cifras ancla para el encabezado, leidas de ``reports/validation/*.csv`` (nunca escritas a mano).

    Mejora de Pitching+ frente a ``CountBaselineModel`` en RMSE de ``xrv_pitcher`` por holdout
    (pitcher = media de folds; season y park directos) y estabilidad anio a anio (Pearson).
    """
    out: list[dict[str, str]] = []
    h = reports.get("holdout_metrics", pd.DataFrame())
    if len(h) and {"holdout", "model", "target", "rmse"} <= set(h.columns):
        x = h[h["target"] == "xrv_pitcher"]
        pit = x[x["model"] == "Pitching+"].set_index("holdout")["rmse"]
        base = x[x["model"] == "CountBaselineModel"].set_index("holdout")["rmse"]
        red = (1.0 - pit / base).dropna()
        folds = [i for i in red.index if str(i).startswith("pitcher_fold")]
        spec = [
            ("Pitcher", float(red[folds].mean()) if folds else None,
             f"Holdout Pitcher: reducción de RMSE de xrv_pitcher (Pitching+) vs CountBaselineModel, media de {len(folds)} folds"),
            ("Season", float(red["season"]) if "season" in red.index else None,
             "Holdout Season: reducción de RMSE de xrv_pitcher (Pitching+) vs CountBaselineModel"),
            ("Park", float(red["park"]) if "park" in red.index else None,
             "Holdout Park: reducción de RMSE de xrv_pitcher (Pitching+) vs CountBaselineModel"),
        ]
        for label, val, title in spec:
            if val is not None:
                out.append({"label": label, "value": f"{100 * val:.2f}%", "title": title, "icon": "shield"})
    s = reports.get("stability", pd.DataFrame())
    if len(s) and {"model", "pearson_r"} <= set(s.columns):
        row = s[s["model"] == "Pitching+"]
        if len(row):
            r = row.iloc[0]
            basis = str(r.get("score_basis", "")).replace("in_sample_train_2024_2025", "en muestra 2024–25")
            out.append({"label": "Año a año", "value": f"r {float(r['pearson_r']):.2f}", "icon": "spin",
                        "title": f"Estabilidad año a año de Pitching+ (Pearson), n = {int(r.get('n_pitchers', 0))}, {basis}"})
    return out


# ----------------------------------------------------------------------------
# Veredicto en lenguaje llano (generado a partir de la simulacion real)
# ----------------------------------------------------------------------------
def verdict_parts(res: dict[str, Any], pitch_id: str) -> dict[str, Any]:
    """Cifras y frases del veredicto: dIVB, dHB y Stuff+ de nivel del mar -> CDMX (todo sale de ``res``)."""
    n, c = res["neutral"], res["cdmx"]
    d_ivb = abs(c["ivb"]) - abs(n["ivb"])       # < 0: pierde magnitud de quiebre vertical
    d_hb = abs(c["hb"]) - abs(n["hb"])
    ivb_word = "subida" if n["ivb"] >= 0 else "caída"
    tol = 0.05

    def clause(delta: float, word: str) -> tuple[str, str]:
        if delta < -tol:
            return "loss", f"pierde {abs(delta):.1f} in de {word}"
        if delta > tol:
            return "gain", f"gana {delta:.1f} in de {word}"
        return "neu", f"conserva su {word}"

    k1, t1 = clause(d_ivb, ivb_word)
    k2, t2 = clause(d_hb, "corrida")
    any_loss = "loss" in (k1, k2)
    if k1 == k2 and k1 in ("loss", "gain"):  # no repetir el verbo
        verb = "pierde" if k1 == "loss" else "gana"
        a, b = abs(d_ivb), abs(d_hb)
        movement = f"{verb} {a:.1f} in de {ivb_word} y {b:.1f} in de corrida"
    else:
        movement = f"{t1} y {t2}"
    movement += " por el aire delgado" if any_loss else ""
    ds = res["delta_stuff"]
    tone = theme.delta_tone(ds)
    stuff = {"loss": "baja", "gain": "sube"}.get(tone, "se mantiene")
    return {
        "pitch_name": PITCH_NAMES_ES.get(pitch_id, pitch_id.lower()),
        "d_ivb": d_ivb, "d_hb": d_hb, "movement": movement, "stuff_verb": stuff,
        "stuff_sea": n["stuff"], "stuff_cdmx": c["stuff"], "delta_stuff": ds, "tone": tone,
    }


def reference_summary(sculptor_res: dict[str, Any]) -> dict[str, Any]:
    """Resumen de la linea "Referencia" a partir de ``PitchSculptor.find_feasible_references``."""
    status = sculptor_res.get("status")
    n = int(sculptor_res.get("n_peers", 0))
    if status == "INSUFFICIENT_PEERS":
        return {"status": status, "n_peers": n, "reason": sculptor_res.get("reason") or ""}
    rep = sculptor_res.get("repertoire") or []
    if not rep:
        return {"status": "NO_OTHER_FAMILY", "n_peers": n,
                "reason": "Los pares reales no muestran otra familia de pitcheo con muestra suficiente."}
    top = rep[0]
    return {"status": "OK", "n_peers": n, "pitch_id": top["pitch_id"],
            "median": float(top["median_stuff_plus_cdmx"]), "n_pitchers": int(top["n_pitchers"])}


# ----------------------------------------------------------------------------
# Figuras Plotly (plantilla propia de dashboard/theme.py; el rojo de marca nunca es color de datos)
# ----------------------------------------------------------------------------
def _layout(fig, title: str = "", height: int = 420):
    fig.update_layout(template=theme.TEMPLATE_NAME, title=dict(text=title), height=height)
    return fig


def fig_movement_vector(result: dict[str, Any]):
    """Movimiento y ubicacion: Nivel del Mar -> Harp Helu unidos por el vector de perdida aerodinamica."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    theme.register_plotly_template()
    n, c = result["neutral"], result["cdmx"]
    frame = result["frame"]
    tone_col = theme.tone_color(theme.delta_tone(result["delta_stuff"]))
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=("Plano de movimiento (pulgadas)", "Ubicación en la zona (vista del catcher)"),
    )
    fig.add_trace(go.Scatter(
        x=[n["hb"]], y=[n["ivb"]], mode="markers+text", name="Nivel del Mar",
        text=["Nivel del Mar"], textposition="top center",
        marker=dict(size=16, color="#FFFFFF", line=dict(color=theme.NAVY, width=3)),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[c["hb"]], y=[c["ivb"]], mode="markers+text", name="Harp Helú",
        text=["Harp Helú"], textposition="bottom center",
        marker=dict(size=18, color=tone_col, symbol="star", line=dict(color="#FFFFFF", width=1)),
    ), row=1, col=1)
    fig.add_annotation(
        x=c["hb"], y=c["ivb"], ax=n["hb"], ay=n["ivb"], xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5, arrowcolor=tone_col,
    )
    fig.update_xaxes(title_text="HB (+ lado brazo / − lado guante)", row=1, col=1)
    fig.update_yaxes(title_text="IVB", row=1, col=1)

    side, height = float(frame["PlateLocSide"].iloc[0]), float(frame["PlateLocHeight"].iloc[0])
    fig.add_shape(type="rect", x0=-0.833, x1=0.833, y0=1.5, y1=3.5, xref="x2", yref="y2",
                  line=dict(color=theme.MUTED, width=2))
    fig.add_trace(go.Scatter(
        x=[side], y=[height], mode="markers", name="Ubicación",
        marker=dict(size=14, color=theme.NAVY, symbol="circle", line=dict(color="#FFFFFF", width=1)),
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Lado (pies)", range=[-2, 2], row=1, col=2)
    fig.update_yaxes(title_text="Altura (pies)", range=[0.5, 4.5], row=1, col=2)
    fig.update_layout(showlegend=True)
    return _layout(fig, "Pérdida aerodinámica: Nivel del Mar → Harp Helú", 400)


def fig_shap(table: pd.DataFrame, title: str = "Qué variables impulsan el Stuff+", height: int = 380):
    """Barras horizontales de contribucion SHAP (puntos de Stuff+): turquesa suma, coral resta (con ▲/▼)."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    t = table.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["contribucion"], y=t["variable"], orientation="h",
        marker_color=[theme.GAIN if v >= 0 else theme.LOSS for v in t["contribucion"]],
        text=[f"{'▲' if v >= 0 else '▼'} {v:+.2f}".replace("-", "−") for v in t["contribucion"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.3f} pts<extra></extra>",
    ))
    vals = [float(v) for v in t["contribucion"]]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = (hi - lo) or 1.0
    fig.update_xaxes(title_text="Contribución (puntos de Stuff+)", range=[lo - 0.30 * span, hi + 0.30 * span])
    fig.update_yaxes(automargin=True)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=40))
    return _layout(fig, title, height)


def fig_decomposition(payload: dict[str, Any]):
    """Barras de los tres terminos con CI (xRV por pitcheo); el residuo conserva su etiqueta y va rayado."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    rows = [r for r in payload["rows"] if r["channel"] != "total"]
    colors = {"stuff": theme.CAT_STUFF, "park": theme.CAT_PARK, "unallocated": theme.NEUTRAL}
    fig = go.Figure(go.Bar(
        x=[r["label"] for r in rows],
        y=[r["estimate"] for r in rows],
        marker=dict(color=[colors[r["channel"]] for r in rows],
                    pattern=dict(shape=["" if r["channel"] != "unallocated" else "/" for r in rows],
                                 bgcolor="#FFFFFF", fgcolor=theme.NEUTRAL)),
        error_y=dict(
            type="data", symmetric=False,
            array=[r["ci_high"] - r["estimate"] for r in rows],
            arrayminus=[r["estimate"] - r["ci_low"] for r in rows],
            color=theme.MUTED,
        ),
        hovertemplate="%{x}<br>%{y:.5f} xRV/pitcheo<extra></extra>",
    ))
    fig.update_yaxes(title_text="xRV por pitcheo")
    return _layout(fig, "Descomposición de tres términos (liga)", 360)


def fig_movement_heatmap(matrix: pd.DataFrame, metric_key: str, title: str, active: dict[str, float] | None = None):
    """Heatmap interactivo IVB x HB_norm. El delta usa escala divergente centrada en 0 (coral/turquesa)."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    is_delta = metric_key == "delta_stuff_cdmx"
    kwargs: dict[str, Any] = {}
    if is_delta:
        lim = float(np.nanmax(np.abs(matrix.to_numpy()))) if matrix.size else 1.0
        kwargs.update(zmid=0, zmin=-lim, zmax=lim, colorscale=theme.DIVERGING_SCALE)
        bar = "ΔStuff+"
    else:
        kwargs.update(colorscale=theme.SEQUENTIAL_SCALE)
        bar = "Stuff+"
    fig = go.Figure(go.Heatmap(
        x=matrix.columns, y=matrix.index, z=matrix.to_numpy(),
        colorbar=dict(title=bar), hovertemplate="HB %{x} in · IVB %{y} in<br>%{z:.2f}<extra></extra>",
        **kwargs,
    ))
    if active:
        fig.add_annotation(x=active["hb_cdmx"], y=active["ivb_cdmx"], ax=active["hb_sea"], ay=active["ivb_sea"],
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2.4,
                           arrowcolor=theme.NAVY)
        fig.add_trace(go.Scatter(x=[active["hb_sea"]], y=[active["ivb_sea"]], mode="markers", name="○ Tu pitcheo, nivel del mar",
                                 marker=dict(size=14, color="#FFFFFF", line=dict(color=theme.NAVY, width=3))))
        fig.add_trace(go.Scatter(x=[active["hb_cdmx"]], y=[active["ivb_cdmx"]], mode="markers", name="★ Tu pitcheo, CDMX",
                                 marker=dict(size=18, symbol="star", color=theme.NAVY, line=dict(color="#FFFFFF", width=1))))
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(title_text="HB normalizado (+ lado brazo)")
    fig.update_yaxes(title_text="IVB (pulgadas)")
    return _layout(fig, title, 560)


def fig_league_plane(grid: pd.DataFrame, pitch_id: str, result: dict[str, Any], height: int = 330):
    """Plano IVB x HB con la densidad de la liga (n_pitches por celda de la familia) y el vector ○ -> ★."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    n, c = result["neutral"], result["cdmx"]
    tone_col = theme.tone_color(theme.delta_tone(result["delta_stuff"]))
    fig = go.Figure()
    g = grid[grid["pitch_id"] == pitch_id] if len(grid) else grid
    if len(g):
        dens = g.pivot_table(index="IVB_bin", columns="HB_norm_bin", values="n_pitches", aggfunc="sum").sort_index()
        fig.add_trace(go.Heatmap(
            x=dens.columns, y=dens.index, z=dens.to_numpy(), colorscale=theme.SEQUENTIAL_SCALE, showscale=False,
            hovertemplate="HB %{x} in · IVB %{y} in<br>%{z} pitcheos<extra></extra>", name="Liga",
        ))
    fig.add_annotation(x=c["hb"], y=c["ivb"], ax=n["hb"], ay=n["ivb"], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowwidth=2.2, arrowcolor=tone_col)
    fig.add_trace(go.Scatter(x=[n["hb"]], y=[n["ivb"]], mode="markers", name="○ Nivel del mar",
                             marker=dict(size=13, color="#FFFFFF", line=dict(color=theme.NAVY, width=3))))
    fig.add_trace(go.Scatter(x=[c["hb"]], y=[c["ivb"]], mode="markers", name="★ Harp Helú",
                             marker=dict(size=17, symbol="star", color=tone_col, line=dict(color="#FFFFFF", width=1))))
    fig.update_xaxes(title_text="HB normalizado (in)")
    fig.update_yaxes(title_text="IVB (in)")
    fig.update_layout(legend=dict(orientation="h", y=-0.22), margin=dict(l=44, r=10, t=10, b=70))
    return _layout(fig, "", height)


def fig_arsenal(arsenal: pd.DataFrame):
    """Barras Stuff+ Nivel del Mar vs. CDMX por pitcheo del arsenal."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    fig = go.Figure()
    fig.add_bar(x=arsenal["pitch_id"], y=arsenal["stuff_plus_neutral"], name="Nivel del Mar", marker_color=theme.LVL_SEA)
    fig.add_bar(x=arsenal["pitch_id"], y=arsenal["stuff_plus_cdmx"], name="Harp Helú", marker_color=theme.LVL_CDMX)
    fig.add_hline(y=100, line_dash="dot", line_color=theme.MUTED)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Stuff+", range=[min(90, float(arsenal[["stuff_plus_neutral", "stuff_plus_cdmx"]].min().min()) - 2), None])
    return _layout(fig, "Stuff+ por pitcheo del arsenal", 340)


def fig_arsenal_map(arsenal: pd.DataFrame, height: int = 360):
    """Mini mapa IVB x HB del arsenal: ○ nivel del mar -> ★ CDMX por pitcheo, con una etiqueta por pitcheo (sin encimar)."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    fig = go.Figure()
    placed: list[tuple[float, float]] = []
    for i, r in enumerate(arsenal.itertuples()):
        near = [q for q in placed if abs(q[0] - r.hb_norm_neutral) < 5 and abs(q[1] - r.ivb_neutral) < 5]
        pos = "top center" if len(near) % 2 == 0 else "bottom center"
        placed.append((r.hb_norm_neutral, r.ivb_neutral))
        fig.add_annotation(x=r.hb_norm_cdmx, y=r.ivb_cdmx, ax=r.hb_norm_neutral, ay=r.ivb_neutral, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=1.8, arrowcolor=theme.LVL_SEA)
        tone = theme.tone_color(theme.delta_tone(r.delta_stuff_cdmx))
        fig.add_trace(go.Scatter(x=[r.hb_norm_neutral], y=[r.ivb_neutral], mode="markers+text", text=[r.pitch_id],
                                 textposition=pos, textfont=dict(size=12, color=theme.TEXT), showlegend=False,
                                 marker=dict(size=13, color="#FFFFFF", line=dict(color=theme.NAVY, width=3)),
                                 hovertemplate=f"{r.pitch_id} (nivel del mar)<br>HB %{{x:.1f}} · IVB %{{y:.1f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[r.hb_norm_cdmx], y=[r.ivb_cdmx], mode="markers", showlegend=False,
                                 marker=dict(size=16, symbol="star", color=tone, line=dict(color="#FFFFFF", width=1)),
                                 hovertemplate=f"{r.pitch_id} (CDMX)<br>HB %{{x:.1f}} · IVB %{{y:.1f}}<extra></extra>"))
    fig.add_hline(y=0, line_color=theme.LINE)
    fig.add_vline(x=0, line_color=theme.LINE)
    fig.update_xaxes(title_text="HB normalizado (in)  ○ mar → ★ CDMX", automargin=True)
    fig.update_yaxes(title_text="IVB (in)", automargin=True)
    fig.update_layout(margin=dict(l=44, r=24, t=10, b=44))
    return _layout(fig, "", height)


def fig_delta_hist(deltas: pd.Series, selected: float | None = None, label: str = "", height: int = 240):
    """Histograma REAL de dStuff+ CDMX (barras coral/gris/turquesa por signo) con el lanzador elegido marcado."""
    import plotly.graph_objects as go

    theme.register_plotly_template()
    d = pd.Series(deltas).dropna().astype(float)
    fig = go.Figure()
    if len(d):
        counts, edges = np.histogram(d, bins=24)
        centers = (edges[:-1] + edges[1:]) / 2.0
        colors = [theme.tone_color(theme.delta_tone(c)) for c in centers]
        fig.add_trace(go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]) * 0.92, marker_color=colors,
                             hovertemplate="ΔStuff+ %{x:.2f}<br>%{y} temporadas-lanzador<extra></extra>"))
        if selected is not None and np.isfinite(selected):
            fig.add_vline(x=float(selected), line_color=theme.NAVY, line_width=3)
            fig.add_annotation(x=float(selected), y=float(counts.max()), text=f"{label or 'seleccionado'}: {selected:+.2f}".replace("-", "−"),
                               showarrow=False, yshift=12, xanchor="right", xshift=-6, font=dict(color=theme.NAVY, size=12))
    fig.update_xaxes(title_text="ΔStuff+ CDMX (CDMX − nivel del mar)")
    fig.update_yaxes(title_text="Temporadas-lanzador")
    fig.update_layout(margin=dict(l=44, r=16, t=28, b=44), showlegend=False)
    return _layout(fig, "", height)


# ----------------------------------------------------------------------------
# Formateo
# ----------------------------------------------------------------------------
def fmt_score(x: float | None, decimals: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{decimals}f}"


def fmt_delta(x: float | None, decimals: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{decimals}f}"


def fmt_pct(x: float | None, decimals: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100.0 * x:.{decimals}f} %"

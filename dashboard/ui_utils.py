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
}

# Paleta Diablos Rojos
RED = "#CE1126"
BLACK = "#0D0D0D"
LIGHT = "#F5F5F5"
GREY = "#8A8A8A"
PANEL = "#1A1A1A"
GREEN = "#3FB950"

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
    "rel_speed": (55.0, 110.0), "spin_rate": (500.0, 3800.0), "spin_axis": (0.0, 360.0),
    "ivb": (-30.0, 30.0), "hb_norm": (-30.0, 30.0), "rel_height": (3.0, 8.0), "rel_side": (0.0, 5.0),
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
    return clip_inputs(base)


# ----------------------------------------------------------------------------
# Figuras Plotly
# ----------------------------------------------------------------------------
def _dark_layout(fig, title: str = "", height: int = 420):
    fig.update_layout(
        title=dict(text=title, font=dict(color=LIGHT, size=15)),
        paper_bgcolor=BLACK,
        plot_bgcolor=PANEL,
        font=dict(color=LIGHT),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#333333", zerolinecolor="#555555")
    fig.update_yaxes(gridcolor="#333333", zerolinecolor="#555555")
    return fig


def fig_movement_vector(result: dict[str, Any]):
    """Movimiento y ubicacion: Nivel del Mar -> Harp Helu unidos por el vector de perdida aerodinamica."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n, c = result["neutral"], result["cdmx"]
    frame = result["frame"]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=("Plano de movimiento (pulgadas)", "Ubicación en la zona (vista del catcher)"),
    )
    fig.add_trace(go.Scatter(
        x=[n["hb"]], y=[n["ivb"]], mode="markers+text", name="Nivel del Mar",
        text=["Nivel del Mar"], textposition="top center",
        marker=dict(size=16, color=LIGHT, line=dict(color=LIGHT, width=1)),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[c["hb"]], y=[c["ivb"]], mode="markers+text", name="Harp Helú",
        text=["Harp Helú"], textposition="bottom center",
        marker=dict(size=16, color=RED, line=dict(color=LIGHT, width=1)),
    ), row=1, col=1)
    fig.add_annotation(
        x=c["hb"], y=c["ivb"], ax=n["hb"], ay=n["ivb"], xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=3, arrowcolor=RED,
    )
    fig.update_xaxes(title_text="HB (+ lado brazo / − lado guante)", row=1, col=1)
    fig.update_yaxes(title_text="IVB", row=1, col=1)

    side, height = float(frame["PlateLocSide"].iloc[0]), float(frame["PlateLocHeight"].iloc[0])
    fig.add_shape(type="rect", x0=-0.833, x1=0.833, y0=1.5, y1=3.5, xref="x2", yref="y2",
                  line=dict(color=LIGHT, width=2))
    fig.add_trace(go.Scatter(
        x=[side], y=[height], mode="markers", name="Ubicación",
        marker=dict(size=14, color=RED, symbol="circle", line=dict(color=LIGHT, width=1)),
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Lado (pies)", range=[-2, 2], row=1, col=2)
    fig.update_yaxes(title_text="Altura (pies)", range=[0.5, 4.5], row=1, col=2)
    fig.update_layout(showlegend=True)
    return _dark_layout(fig, "Pérdida aerodinámica: Nivel del Mar → Harp Helú", 400)


def fig_shap(table: pd.DataFrame, title: str = "Qué variables impulsan el Stuff+"):
    """Barras horizontales de contribucion SHAP (puntos de Stuff+)."""
    import plotly.graph_objects as go

    t = table.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["contribucion"], y=t["variable"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in t["contribucion"]],
        hovertemplate="%{y}: %{x:+.3f} pts<extra></extra>",
    ))
    fig.update_xaxes(title_text="Contribución (puntos de Stuff+)")
    return _dark_layout(fig, title, 380)


def fig_decomposition(payload: dict[str, Any]):
    """Barras de los tres terminos con CI (xRV por pitcheo); el residuo conserva su etiqueta."""
    import plotly.graph_objects as go

    rows = [r for r in payload["rows"] if r["channel"] != "total"]
    colors = {"stuff": RED, "park": GREY, "unallocated": "#5A5A5A"}
    fig = go.Figure(go.Bar(
        x=[r["label"] for r in rows],
        y=[r["estimate"] for r in rows],
        marker_color=[colors[r["channel"]] for r in rows],
        error_y=dict(
            type="data", symmetric=False,
            array=[r["ci_high"] - r["estimate"] for r in rows],
            arrayminus=[r["estimate"] - r["ci_low"] for r in rows],
            color=LIGHT,
        ),
        hovertemplate="%{x}<br>%{y:.5f} xRV/pitcheo<extra></extra>",
    ))
    fig.update_yaxes(title_text="xRV por pitcheo")
    return _dark_layout(fig, "Descomposición de tres términos (liga)", 360)


def fig_movement_heatmap(matrix: pd.DataFrame, metric_key: str, title: str):
    """Heatmap interactivo IVB x HB_norm. El delta usa escala divergente centrada en 0."""
    import plotly.graph_objects as go

    is_delta = metric_key == "delta_stuff_cdmx"
    kwargs: dict[str, Any] = {}
    if is_delta:
        lim = float(np.nanmax(np.abs(matrix.to_numpy()))) if matrix.size else 1.0
        kwargs.update(zmid=0, zmin=-lim, zmax=lim,
                      colorscale=[[0, RED], [0.5, "#2A2A2A"], [1, GREEN]])
        bar = "ΔStuff+"
    else:
        kwargs.update(colorscale=[[0, "#1F1F1F"], [0.5, "#8C1020"], [1, RED]])
        bar = "Stuff+"
    fig = go.Figure(go.Heatmap(
        x=matrix.columns, y=matrix.index, z=matrix.to_numpy(),
        colorbar=dict(title=bar), hovertemplate="HB %{x} in · IVB %{y} in<br>%{z:.2f}<extra></extra>",
        **kwargs,
    ))
    fig.update_xaxes(title_text="HB normalizado (+ lado brazo)")
    fig.update_yaxes(title_text="IVB (pulgadas)")
    return _dark_layout(fig, title, 560)


def fig_arsenal(arsenal: pd.DataFrame):
    """Barras Stuff+ Nivel del Mar vs. CDMX por pitcheo del arsenal."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_bar(x=arsenal["pitch_id"], y=arsenal["stuff_plus_neutral"], name="Nivel del Mar", marker_color=LIGHT)
    fig.add_bar(x=arsenal["pitch_id"], y=arsenal["stuff_plus_cdmx"], name="Harp Helú", marker_color=RED)
    fig.add_hline(y=100, line_dash="dot", line_color=GREY)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Stuff+", range=[min(90, float(arsenal[["stuff_plus_neutral", "stuff_plus_cdmx"]].min().min()) - 2), None])
    return _dark_layout(fig, "Stuff+ por pitcheo del arsenal", 340)


# ----------------------------------------------------------------------------
# Formateo
# ----------------------------------------------------------------------------
def fmt_score(x: float | None, decimals: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{decimals}f}"


def fmt_delta(x: float | None, decimals: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{decimals}f}"


def fmt_pct(x: float | None, decimals: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100.0 * x:.{decimals}f} %"

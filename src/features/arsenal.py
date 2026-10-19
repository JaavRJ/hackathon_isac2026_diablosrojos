"""
src/features/arsenal.py
=======================
Módulo de características del arsenal relacional, dinámica de secuencia (prev_pitch_id)
y métricas de túnel de pitcheo (versión release y versión punto de decisión a 170 ms)
para Diablos AeroVision (Reto 2 Diablos Rojos - Hackathon ISAC 2026).

Cumple estrictamente con docs/CONTRACT.md (HACK-005), AGENTS.md y docs/TASKS.md:
- Uso EXCLUSIVO de pitch_id (HACK-004) como identidad invariante del pitcheo (PROHIBIDO AutoPitchType).
- Determinación de la recta primaria (primary_fb_id) por (pitcher_anon_id, year) priorizando No Altitude.
- Corrección de las medias de quiebre (InducedVertBreak, HorzBreak_norm) por rho_hat si el pitcher
  solo lanzó en altitud extrema o media.
- Diferenciales de velocidad y movimiento contra la recta primaria del pitcher (velo_diff_vs_fb,
  ivb_diff_vs_fb, hb_diff_vs_fb) y uso de lanzamiento (pitch_usage_pct).
- Túnel Versión A (Release Point - definición de las bases): tunnel_dist_release_in y tunnel_sim_release.
- Túnel Versión B (Decision Point a t = 0.17 s - extensión Diablos AeroVision): tunnel_dist_170ms_in
  y tunnel_sim_170ms evaluados con polinomios de TrackMan.
- Secuencia (prev_pitch_id): detección rigurosa de inicios de aparición al plato (PA) respetando
  el orden de filas original, sin mezclar bateadores, lanzadores, entradas ni juegos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.identity import assign_invariant_pitch_id
from src.features.physical import add_physical_features, clean_and_impute

# Tipos de recta elegibles para primary_fb_id en orden de precedencia ante empates
FASTBALL_TYPES: list[str] = ["Four-Seam", "Sinker", "Cutter"]

# Tiempo al punto de decisión visual del bateador (170 milisegundos = 0.17 segundos)
DECISION_POINT_TIME: float = 0.17

# Escala de decaimiento exponencial para similitud de túnel (pulgadas)
TUNNEL_DECAY_SCALE_IN: float = 6.0

# 10 columnas oficiales de CONTRACT.md (HACK-005)
ARSENAL_FEATURE_COLS: list[str] = [
    "primary_fb_id",
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "tunnel_dist_release_in",
    "tunnel_sim_release",
    "tunnel_dist_170ms_in",
    "tunnel_sim_170ms",
    "pitch_usage_pct",
    "prev_pitch_id",
]


def _compute_decision_point_3d(
    df: pd.DataFrame,
    t: float = DECISION_POINT_TIME,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Evalúa la posición 3D (X, Y, Z en pies) en el punto de decisión (t segundos)
    utilizando los polinomios de trayectoria de TrackMan:
        X(t) = PitchTrajectoryXc0 + PitchTrajectoryXc1*t + PitchTrajectoryXc2*t^2
        Y(t) = PitchTrajectoryYc0 + PitchTrajectoryYc1*t + PitchTrajectoryYc2*t^2
        Z(t) = PitchTrajectoryZc0 + PitchTrajectoryZc1*t + PitchTrajectoryZc2*t^2

    Si faltan coeficientes o existen valores nulos, aplica fallbacks cinemáticos
    basados en x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0 o datos de release.
    """
    t2 = t * t

    # Coordenada X (lateral)
    if "PitchTrajectoryXc0" in df.columns:
        x_poly = (
            pd.to_numeric(df["PitchTrajectoryXc0"], errors="coerce")
            + pd.to_numeric(df.get("PitchTrajectoryXc1", 0.0), errors="coerce").fillna(0.0) * t
            + pd.to_numeric(df.get("PitchTrajectoryXc2", 0.0), errors="coerce").fillna(0.0) * t2
        )
    else:
        x_poly = pd.Series(np.nan, index=df.index, dtype="float64")

    # Coordenada Y (distancia hacia home plate)
    if "PitchTrajectoryYc0" in df.columns:
        y_poly = (
            pd.to_numeric(df["PitchTrajectoryYc0"], errors="coerce")
            + pd.to_numeric(df.get("PitchTrajectoryYc1", 0.0), errors="coerce").fillna(0.0) * t
            + pd.to_numeric(df.get("PitchTrajectoryYc2", 0.0), errors="coerce").fillna(0.0) * t2
        )
    else:
        y_poly = pd.Series(np.nan, index=df.index, dtype="float64")

    # Coordenada Z (vertical / altura)
    if "PitchTrajectoryZc0" in df.columns:
        z_poly = (
            pd.to_numeric(df["PitchTrajectoryZc0"], errors="coerce")
            + pd.to_numeric(df.get("PitchTrajectoryZc1", 0.0), errors="coerce").fillna(0.0) * t
            + pd.to_numeric(df.get("PitchTrajectoryZc2", 0.0), errors="coerce").fillna(0.0) * t2
        )
    else:
        z_poly = pd.Series(np.nan, index=df.index, dtype="float64")

    # Fallbacks cinemáticos si hay nulos
    if x_poly.isna().any():
        if "x0" in df.columns and "vx0" in df.columns and "ax0" in df.columns:
            x_kin = (
                pd.to_numeric(df["x0"], errors="coerce").fillna(0.0)
                + pd.to_numeric(df["vx0"], errors="coerce").fillna(0.0) * t
                + 0.5 * pd.to_numeric(df["ax0"], errors="coerce").fillna(0.0) * t2
            )
        else:
            x_kin = pd.to_numeric(df.get("RelSide_norm", 0.0), errors="coerce").fillna(0.0)
        x_poly = x_poly.fillna(x_kin).fillna(0.0)

    if y_poly.isna().any():
        if "y0" in df.columns and "vy0" in df.columns and "ay0" in df.columns:
            y_kin = (
                pd.to_numeric(df["y0"], errors="coerce").fillna(50.0)
                + pd.to_numeric(df["vy0"], errors="coerce").fillna(-130.0) * t
                + 0.5 * pd.to_numeric(df["ay0"], errors="coerce").fillna(25.0) * t2
            )
        else:
            rel_sp = pd.to_numeric(df.get("RelSpeed", 90.0), errors="coerce").fillna(90.0)
            y_kin = 50.0 - rel_sp * 1.467 * t
        y_poly = y_poly.fillna(y_kin).fillna(30.0)

    if z_poly.isna().any():
        if "z0" in df.columns and "vz0" in df.columns and "az0" in df.columns:
            z_kin = (
                pd.to_numeric(df["z0"], errors="coerce").fillna(5.8)
                + pd.to_numeric(df["vz0"], errors="coerce").fillna(-3.0) * t
                + 0.5 * pd.to_numeric(df["az0"], errors="coerce").fillna(-20.0) * t2
            )
        else:
            z_kin = pd.to_numeric(df.get("RelHeight", 5.8), errors="coerce").fillna(5.8)
        z_poly = z_poly.fillna(z_kin).fillna(5.5)

    return (
        x_poly.astype("float64"),
        y_poly.astype("float64"),
        z_poly.astype("float64"),
    )


def _detect_new_pa(df: pd.DataFrame) -> pd.Series:
    """
    Detecta el inicio de cada aparición al plato (PA) respetando el orden de filas original.
    Una fila es el primer lanzamiento de una PA (is_new_pa = True) si:
    1. Es la primera fila del DataFrame, O
    2. Cambia cualquiera de (game_anon_id, Inning, Top/Bottom, pitcher_anon_id, batter_anon_id)
       respecto a la fila anterior, O
    3. Balls == 0 y Strikes == 0 (en béisbol ningún lanzamiento posterior dentro de una misma PA
       puede tener conteo 0-0), O
    4. Balls < prev_Balls o Strikes < prev_Strikes, O
    5. Balls == prev_Balls y Strikes == prev_Strikes y Strikes < 2 (con menos de 2 strikes
       el conteo no puede repetirse dentro de la misma PA).
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=bool)

    # Columnas de contexto
    game_s = df["game_anon_id"].astype(str) if "game_anon_id" in df.columns else pd.Series("g", index=df.index)
    inn_s = df["Inning"].astype(str) if "Inning" in df.columns else pd.Series("1", index=df.index)
    half_s = df["Top/Bottom"].astype(str) if "Top/Bottom" in df.columns else pd.Series("Top", index=df.index)
    pitcher_s = df["pitcher_anon_id"].astype(str) if "pitcher_anon_id" in df.columns else pd.Series("p", index=df.index)
    batter_s = df["batter_anon_id"].astype(str) if "batter_anon_id" in df.columns else pd.Series("b", index=df.index)

    balls_s = pd.to_numeric(df.get("Balls", 0), errors="coerce").fillna(0).astype(int)
    strikes_s = pd.to_numeric(df.get("Strikes", 0), errors="coerce").fillna(0).astype(int)

    # Cambios respecto a la fila anterior
    prev_game = game_s.shift(1)
    prev_inn = inn_s.shift(1)
    prev_half = half_s.shift(1)
    prev_pitcher = pitcher_s.shift(1)
    prev_batter = batter_s.shift(1)
    prev_balls = balls_s.shift(1)
    prev_strikes = strikes_s.shift(1)

    diff_context = (
        prev_game.isna()
        | (game_s != prev_game)
        | (inn_s != prev_inn)
        | (half_s != prev_half)
        | (pitcher_s != prev_pitcher)
        | (batter_s != prev_batter)
    )

    is_0_0 = (balls_s == 0) & (strikes_s == 0)
    is_count_drop = (balls_s < prev_balls) | (strikes_s < prev_strikes)
    is_repeat_sub2 = (balls_s == prev_balls) & (strikes_s == prev_strikes) & (strikes_s < 2)

    is_new = diff_context | is_0_0 | is_count_drop | is_repeat_sub2
    # La primera fila es siempre inicio de PA
    is_new.iloc[0] = True

    return is_new


def add_arsenal_and_tunnel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las 10 características relacionales de arsenal, secuencia de pitcheos
    y túneles en release y decisión (docs/CONTRACT.md HACK-005).

    Trabaja estrictamente sobre una copia (df.copy()) sin mutar df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos con columnas de TrackMan LMB.

    Returns
    -------
    pd.DataFrame
        Copia de df con las 10 columnas exactas de HACK-005 agregadas:
        - primary_fb_id (string)
        - velo_diff_vs_fb (float64)
        - ivb_diff_vs_fb (float64)
        - hb_diff_vs_fb (float64)
        - tunnel_dist_release_in (float64)
        - tunnel_sim_release (float64)
        - tunnel_dist_170ms_in (float64)
        - tunnel_sim_170ms (float64)
        - pitch_usage_pct (float64)
        - prev_pitch_id (string)
    """
    out = df.copy()

    # Manejo de DataFrame vacío
    if out.empty:
        for col in ARSENAL_FEATURE_COLS:
            if col in ("primary_fb_id", "prev_pitch_id"):
                out[col] = pd.Series(dtype="string")
            else:
                out[col] = pd.Series(dtype="float64")
        return out

    # Verificación de dependencias previas sin mutar el DataFrame de entrada
    if "HorzBreak_norm" not in out.columns or "RelSide_norm" not in out.columns:
        if "PitcherThrows" not in out.columns:
            out = clean_and_impute(out)
        out = add_physical_features(out)

    if "pitch_id" not in out.columns:
        out = assign_invariant_pitch_id(out)

    # Identificación segura de pitcher, año y altitud
    if "pitcher_anon_id" not in out.columns:
        out["pitcher_anon_id"] = "pitcher_default"
    if "year" not in out.columns:
        out["year"] = 2024
    if "altitude_category" not in out.columns:
        out["altitude_category"] = "No Altitude"

    # Evaluación de la posición 3D al punto de decisión (t = 0.17 s)
    x170, y170, z170 = _compute_decision_point_3d(out, t=DECISION_POINT_TIME)

    # Proporción de uso de cada pitch_id por (pitcher_anon_id, year)
    py_total = out.groupby(["pitcher_anon_id", "year"])["pitch_id"].transform("count")
    pyp_count = out.groupby(["pitcher_anon_id", "year", "pitch_id"])["pitch_id"].transform("count")
    out["pitch_usage_pct"] = (pyp_count / py_total).fillna(1.0).astype("float64")

    # Mapeo de referencia de la recta primaria por (pitcher_anon_id, year)
    # Tupla: (primary_fb_id, fb_velo, fb_ivb, fb_hb_norm, fb_rel_h, fb_rel_s_norm, fb_ext, fb_x, fb_y, fb_z)
    ref_map: dict[tuple[str, int], tuple[str, float, float, float, float, float, float, float, float, float]] = {}

    grouped = out.groupby(["pitcher_anon_id", "year"], sort=False)
    for (pitcher, yr), group in grouped:
        group_no_alt = group[group["altitude_category"] == "No Altitude"]

        # 1. Selección de primary_fb_id
        fb_no_alt = group_no_alt[group_no_alt["pitch_id"].isin(FASTBALL_TYPES)]
        if len(fb_no_alt) > 0:
            c = fb_no_alt["pitch_id"].value_counts()
            best_fb = sorted(
                c.index,
                key=lambda pid: (-c[pid], FASTBALL_TYPES.index(pid) if pid in FASTBALL_TYPES else 999),
            )[0]
        else:
            fb_all = group[group["pitch_id"].isin(FASTBALL_TYPES)]
            if len(fb_all) > 0:
                c = fb_all["pitch_id"].value_counts()
                best_fb = sorted(
                    c.index,
                    key=lambda pid: (-c[pid], FASTBALL_TYPES.index(pid) if pid in FASTBALL_TYPES else 999),
                )[0]
            else:
                c = group["pitch_id"].value_counts()
                best_fb = sorted(c.index, key=lambda pid: -c[pid])[0]

        primary_fb = str(best_fb)

        # 2. Pitcheos de la recta primaria para cálculo de medias de referencia
        fb_subset = group[group["pitch_id"] == primary_fb]
        fb_subset_no_alt = fb_subset[fb_subset["altitude_category"] == "No Altitude"]

        # Velocidad de referencia (priorizando No Altitude si existen, sino todos)
        if len(fb_subset_no_alt) > 0:
            fb_velo = float(fb_subset_no_alt["RelSpeed"].mean())
            fb_ivb = float(fb_subset_no_alt["InducedVertBreak"].mean())
            fb_hb_norm = float(fb_subset_no_alt["HorzBreak_norm"].mean())
        else:
            fb_velo = float(fb_subset["RelSpeed"].mean())
            raw_ivb = float(fb_subset["InducedVertBreak"].mean())
            raw_hb_norm = float(fb_subset["HorzBreak_norm"].mean())

            # Si solo lanzó en altitud, ajustar dividiendo por rho_hat mediano
            if "rho_hat" in group.columns:
                rho_med = group["rho_hat"].median()
                if pd.isna(rho_med) or rho_med < 0.70:
                    rho_adj = 0.70 if not pd.isna(rho_med) and rho_med > 0 else 1.0
                else:
                    rho_adj = float(rho_med)
            else:
                rho_adj = 1.0

            fb_ivb = raw_ivb / rho_adj
            fb_hb_norm = raw_hb_norm / rho_adj

        # Geometría de release de referencia
        fb_rel_h = float(fb_subset["RelHeight"].mean())
        fb_rel_s_norm = float(fb_subset["RelSide_norm"].mean())
        fb_ext = float(fb_subset["Extension"].mean())

        # Posición 3D en punto de decisión de referencia
        fb_x = float(x170.loc[fb_subset.index].mean())
        fb_y = float(y170.loc[fb_subset.index].mean())
        fb_z = float(z170.loc[fb_subset.index].mean())

        # Fallbacks seguros ante nulos inesperados
        if np.isnan(fb_velo):
            fb_velo = float(group["RelSpeed"].mean()) if not group["RelSpeed"].isna().all() else 93.0
        if np.isnan(fb_ivb):
            fb_ivb = float(group["InducedVertBreak"].mean()) if not group["InducedVertBreak"].isna().all() else 10.0
        if np.isnan(fb_hb_norm):
            fb_hb_norm = float(group["HorzBreak_norm"].mean()) if not group["HorzBreak_norm"].isna().all() else 5.0
        if np.isnan(fb_rel_h):
            fb_rel_h = float(group["RelHeight"].mean()) if not group["RelHeight"].isna().all() else 5.8
        if np.isnan(fb_rel_s_norm):
            fb_rel_s_norm = float(group["RelSide_norm"].mean()) if not group["RelSide_norm"].isna().all() else 1.6
        if np.isnan(fb_ext):
            fb_ext = float(group["Extension"].mean()) if not group["Extension"].isna().all() else 5.8
        if np.isnan(fb_x):
            fb_x = float(x170.loc[group.index].mean()) if not x170.loc[group.index].isna().all() else 0.0
        if np.isnan(fb_y):
            fb_y = float(y170.loc[group.index].mean()) if not y170.loc[group.index].isna().all() else 30.0
        if np.isnan(fb_z):
            fb_z = float(z170.loc[group.index].mean()) if not z170.loc[group.index].isna().all() else 5.5

        ref_map[(pitcher, yr)] = (
            primary_fb,
            fb_velo,
            fb_ivb,
            fb_hb_norm,
            fb_rel_h,
            fb_rel_s_norm,
            fb_ext,
            fb_x,
            fb_y,
            fb_z,
        )

    # Construcción alineada de los vectores de referencia
    keys = list(zip(out["pitcher_anon_id"], out["year"]))
    ref_records = [ref_map[k] for k in keys]

    primary_fb_series = pd.Series([r[0] for r in ref_records], index=out.index, dtype="string")
    fb_velo_s = pd.Series([r[1] for r in ref_records], index=out.index, dtype="float64")
    fb_ivb_s = pd.Series([r[2] for r in ref_records], index=out.index, dtype="float64")
    fb_hb_norm_s = pd.Series([r[3] for r in ref_records], index=out.index, dtype="float64")
    fb_rel_h_s = pd.Series([r[4] for r in ref_records], index=out.index, dtype="float64")
    fb_rel_s_s = pd.Series([r[5] for r in ref_records], index=out.index, dtype="float64")
    fb_ext_s = pd.Series([r[6] for r in ref_records], index=out.index, dtype="float64")
    fb_x_s = pd.Series([r[7] for r in ref_records], index=out.index, dtype="float64")
    fb_y_s = pd.Series([r[8] for r in ref_records], index=out.index, dtype="float64")
    fb_z_s = pd.Series([r[9] for r in ref_records], index=out.index, dtype="float64")

    # Asignación de primary_fb_id
    out["primary_fb_id"] = primary_fb_series

    # Diferenciales vs recta primaria (float64)
    out["velo_diff_vs_fb"] = (
        (pd.to_numeric(out["RelSpeed"], errors="coerce") - fb_velo_s)
        .fillna(0.0)
        .astype("float64")
    )
    out["ivb_diff_vs_fb"] = (
        (pd.to_numeric(out["InducedVertBreak"], errors="coerce") - fb_ivb_s)
        .fillna(0.0)
        .astype("float64")
    )
    out["hb_diff_vs_fb"] = (
        (pd.to_numeric(out["HorzBreak_norm"], errors="coerce") - fb_hb_norm_s)
        .fillna(0.0)
        .astype("float64")
    )

    # Túnel Versión A (Release Point - definición de las bases)
    rel_h = pd.to_numeric(out["RelHeight"], errors="coerce").fillna(fb_rel_h_s)
    rel_s = pd.to_numeric(out["RelSide_norm"], errors="coerce").fillna(fb_rel_s_s)
    ext = pd.to_numeric(out["Extension"], errors="coerce").fillna(fb_ext_s)

    rel_dist_in = 12.0 * np.sqrt(
        (rel_h - fb_rel_h_s) ** 2
        + (rel_s - fb_rel_s_s) ** 2
        + (ext - fb_ext_s) ** 2
    )
    out["tunnel_dist_release_in"] = rel_dist_in.fillna(0.0).astype("float64")
    out["tunnel_sim_release"] = (
        np.exp(-out["tunnel_dist_release_in"] / TUNNEL_DECAY_SCALE_IN)
        .fillna(1.0)
        .astype("float64")
    )

    # Túnel Versión B (Decision Point a 170 ms - extensión Diablos AeroVision)
    dist_170_in = 12.0 * np.sqrt(
        (x170 - fb_x_s) ** 2
        + (y170 - fb_y_s) ** 2
        + (z170 - fb_z_s) ** 2
    )
    out["tunnel_dist_170ms_in"] = dist_170_in.fillna(0.0).astype("float64")
    out["tunnel_sim_170ms"] = (
        np.exp(-out["tunnel_dist_170ms_in"] / TUNNEL_DECAY_SCALE_IN)
        .fillna(1.0)
        .astype("float64")
    )

    # Secuencia de pitcheo (prev_pitch_id)
    is_new_pa = _detect_new_pa(out)
    prev_pitch = out["pitch_id"].shift(1)
    prev_pitch_clean = np.where(is_new_pa, "NONE", prev_pitch)
    out["prev_pitch_id"] = (
        pd.Series(prev_pitch_clean, index=out.index)
        .fillna("NONE")
        .astype("string")
    )

    # Verificación final de integridad de tipos y ausencia de nulos e infinitos
    for col in ARSENAL_FEATURE_COLS:
        if col in ("primary_fb_id", "prev_pitch_id"):
            out[col] = out[col].fillna("NONE").astype("string")
        else:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
                .astype("float64")
            )

    return out

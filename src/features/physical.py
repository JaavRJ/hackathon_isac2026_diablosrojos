"""
src/features/physical.py
========================
Módulo de limpieza, imputación inteligente e ingeniería de características físicas base
para el modelo de Stuff+ LMB (Reto 2 Diablos Rojos - Diablos AeroVision).

Cumple estrictamente con docs/CONTRACT.md (HACK-002) y AGENTS.md:
- Inversión de signos horizontales para lanzadores zurdos (+ = arm-side, - = glove-side).
- Imputación de altitude_category por game_anon_id sin variables target (sin leakage).
- Imputación de PitcherThrows por pitcher_anon_id con fallback geométrico por RelSide.
- Imputación de BatterSide con 'Right' y creación de is_same_hand.
- Filtrado de anomalías de radar en velocidad y rotación.
- Cálculo de variables físicas avanzadas y geométricas de zona para Location+.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Mapeo oficial de recodificación de AutoPitchType
PITCH_TYPE_RECODE: dict[str, str] = {
    "OneSeamFastBall": "Sinker",
    "TwoSeamFastBall": "Sinker",
    "Knuckleball": "Other",
}


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia tipos de datos, recodifica tipos de pitcheo, imputa categorías faltantes
    sin data leakage (utilizando la moda por juego/pitcher y reglas físicas de fallback),
    y filtra anomalías de medición de radar.

    Trabaja estrictamente sobre una copia (df.copy()) sin mutar el DataFrame original.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos con columnas crudas de TrackMan LMB.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio, imputado y filtrado con índice reseteado.
    """
    df = df.copy()

    # 1. Conversión de EffectiveVelo a float64 con imputación de nulos usando RelSpeed
    eff_velo = pd.to_numeric(df["EffectiveVelo"], errors="coerce")
    rel_speed_fallback = pd.to_numeric(df["RelSpeed"], errors="coerce")
    df["EffectiveVelo"] = eff_velo.fillna(rel_speed_fallback).astype("float64")

    # 2. Recodificación de AutoPitchType (OneSeam/TwoSeam -> Sinker, Knuckleball -> Other)
    was_cat_pitch = isinstance(df["AutoPitchType"].dtype, pd.CategoricalDtype)
    pitch_series = df["AutoPitchType"].astype(object).replace(PITCH_TYPE_RECODE)
    if was_cat_pitch:
        df["AutoPitchType"] = pitch_series.astype("category")
    else:
        df["AutoPitchType"] = pitch_series

    # 3. Imputación de altitude_category por moda de game_anon_id (fallback: 'No Altitude')
    was_cat_alt = isinstance(df["altitude_category"].dtype, pd.CategoricalDtype)
    alt_series = df["altitude_category"].astype(object)
    mask_missing_alt = alt_series.isna() | alt_series.isin(["None", "Undefined", ""])

    valid_alt = df.loc[~mask_missing_alt]
    if not valid_alt.empty and "game_anon_id" in valid_alt.columns:
        game_alt_mode = (
            valid_alt.groupby("game_anon_id")["altitude_category"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
        )
        imputed_alt = df["game_anon_id"].map(game_alt_mode).fillna("No Altitude")
    else:
        imputed_alt = pd.Series("No Altitude", index=df.index)

    alt_series = np.where(mask_missing_alt, imputed_alt, alt_series)
    alt_series = (
        pd.Series(alt_series, index=df.index)
        .replace({"None": "No Altitude", "Undefined": "No Altitude", "": "No Altitude"})
        .fillna("No Altitude")
    )
    if was_cat_alt:
        df["altitude_category"] = alt_series.astype("category")
    else:
        df["altitude_category"] = alt_series

    # 4. Imputación de PitcherThrows por moda de pitcher_anon_id (fallback: 'Right' si RelSide >= 0 else 'Left')
    throws_series = df["PitcherThrows"].astype(object)
    mask_missing_throws = throws_series.isna() | throws_series.isin(["None", "Undefined", ""])

    valid_throws = df.loc[~mask_missing_throws]
    if not valid_throws.empty and "pitcher_anon_id" in valid_throws.columns:
        pitcher_mode = (
            valid_throws.groupby("pitcher_anon_id")["PitcherThrows"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
        )
        imputed_throws = df["pitcher_anon_id"].map(pitcher_mode)
    else:
        imputed_throws = pd.Series(np.nan, index=df.index)

    rel_side_val = pd.to_numeric(df.get("RelSide", 0.0), errors="coerce").fillna(0.0)
    fallback_throws = np.where(rel_side_val >= 0.0, "Right", "Left")

    final_throws = np.where(
        mask_missing_throws,
        imputed_throws.fillna(pd.Series(fallback_throws, index=df.index)),
        throws_series,
    )
    final_throws = (
        pd.Series(final_throws, index=df.index)
        .replace({"None": np.nan, "Undefined": np.nan, "": np.nan})
        .fillna(pd.Series(fallback_throws, index=df.index))
    )
    df["PitcherThrows"] = final_throws.astype(str)

    # 5. Imputación de BatterSide faltante con 'Right' y creación de is_same_hand
    batter_series = df["BatterSide"].astype(object)
    mask_missing_batter = batter_series.isna() | batter_series.isin(["None", "Undefined", ""])
    df["BatterSide"] = np.where(mask_missing_batter, "Right", batter_series)
    df["BatterSide"] = (
        pd.Series(df["BatterSide"], index=df.index)
        .replace({"None": "Right", "Undefined": "Right", "": "Right"})
        .fillna("Right")
        .astype(str)
    )
    df["is_same_hand"] = (df["PitcherThrows"] == df["BatterSide"]).astype(int)

    # 6. Filtrado de anomalías de medición de radar
    rel_speed = pd.to_numeric(df["RelSpeed"], errors="coerce")
    spin_rate = pd.to_numeric(df["SpinRate"], errors="coerce")
    radar_mask = rel_speed.between(65.0, 105.0) & spin_rate.between(500.0, 3600.0)
    df = df[radar_mask].reset_index(drop=True)

    return df


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las variables físicas y geométricas requeridas por docs/CONTRACT.md (HACK-002),
    incluyendo normalización por mano del lanzador (+ = arm-side, - = glove-side),
    quiebre Magnus total, proxy de eficiencia de giro, arrastre y ubicación en zona.

    Trabaja estrictamente sobre una copia (df.copy()).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos con columnas limpias e imputadas.

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido con las 12 características físicas y 4 geométricas de zona.
    """
    df = df.copy()

    # Signo de mano: +1.0 para Right, -1.0 para Left
    hand_sign = np.where(df["PitcherThrows"] == "Left", -1.0, 1.0)

    # 1. Normalización de métricas horizontales respecto a la mano del lanzador
    df["HorzBreak_norm"] = (df["HorzBreak"] * hand_sign).astype("float64")
    df["RelSide_norm"] = (df["RelSide"] * hand_sign).astype("float64")
    df["HorzRelAngle_norm"] = (df["HorzRelAngle"] * hand_sign).astype("float64")
    df["HorzApprAngle_norm"] = (df["HorzApprAngle"] * hand_sign).astype("float64")
    df["PlateLocSide_norm"] = (df["PlateLocSide"] * hand_sign).astype("float64")

    # 2. Magnitud total y ángulo de quiebre Magnus inducido
    df["TotalInducedBreak"] = np.sqrt(
        df["InducedVertBreak"] ** 2 + df["HorzBreak"] ** 2
    ).astype("float64")
    df["BreakAngle_deg"] = np.degrees(
        np.arctan2(df["HorzBreak_norm"], df["InducedVertBreak"])
    ).astype("float64")

    # 3. Componentes trigonométricas de eje de giro cíclico (normalizadas por mano)
    spin_rad = np.radians(df["SpinAxis"])
    df["SpinAxis_sin"] = (np.sin(spin_rad) * hand_sign).astype("float64")
    df["SpinAxis_cos"] = np.cos(spin_rad).astype("float64")

    # 4. Proxy físico de Spin Efficiency (H4)
    zone_time_clipped = df["ZoneTime"].clip(lower=0.35)
    df["SpinToBreakRatio"] = (
        df["TotalInducedBreak"] / ((df["SpinRate"] / 1000.0) * zone_time_clipped + 1e-5)
    ).astype("float64")

    # 5. Proxy aerodinámico de arrastre y ángulo de salida del brazo (arm slot)
    rel_speed_clipped = df["RelSpeed"].clip(lower=60.0)
    df["DragProxy"] = (df["SpeedDrop"] / rel_speed_clipped).astype("float64")

    arm_slot_rad = np.arctan2(df["RelHeight"] - 4.0, np.abs(df["RelSide"]) + 0.5)
    df["arm_slot_deg"] = np.clip(np.degrees(arm_slot_rad), 0.0, 90.0).astype("float64")

    # 6. Variables geométricas de zona para Location+
    df["dist_from_zone_center"] = np.sqrt(
        df["PlateLocSide"] ** 2 + (df["PlateLocHeight"] - 2.5) ** 2
    ).astype("float64")

    dx_edge = np.abs(df["PlateLocSide"]) - 0.833
    dz_edge = np.maximum(1.5 - df["PlateLocHeight"], df["PlateLocHeight"] - 3.5)

    outside_dist = np.sqrt(np.maximum(0.0, dx_edge) ** 2 + np.maximum(0.0, dz_edge) ** 2)
    inside_dist = np.minimum(-dx_edge, -dz_edge)
    is_inside_rect = (dx_edge <= 0.0) & (dz_edge <= 0.0)
    df["dist_to_zone_edge"] = np.where(is_inside_rect, inside_dist, outside_dist).astype("float64")

    df["is_shadow_zone"] = (
        ((np.abs(dx_edge) <= 0.25) & df["PlateLocHeight"].between(1.25, 3.75))
        | ((np.abs(dz_edge) <= 0.25) & df["PlateLocSide"].between(-1.08, 1.08))
    ).astype(int)

    cond_in = (df["in_strike_zone"] == 1)
    col_idx = np.clip(np.floor((df["PlateLocSide"] + 0.833) / (1.666 / 3.0)), 0, 2).astype(int)
    row_idx = np.clip(np.floor((df["PlateLocHeight"] - 1.5) / (2.0 / 3.0)), 0, 2).astype(int)
    quad_in = row_idx * 3 + col_idx + 1
    quad_out = np.where(
        df["PlateLocHeight"] >= 2.5,
        np.where(df["PlateLocSide"] >= 0.0, 11, 10),
        np.where(df["PlateLocSide"] >= 0.0, 13, 12),
    )
    df["zone_quadrant"] = np.where(cond_in, quad_in, quad_out).astype(int)

    # Garantizar presencia de is_same_hand si no fue precalculado
    if "is_same_hand" not in df.columns and "BatterSide" in df.columns and "PitcherThrows" in df.columns:
        df["is_same_hand"] = (df["PitcherThrows"] == df["BatterSide"]).astype(int)

    return df

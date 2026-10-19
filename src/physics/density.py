"""
src/physics/density.py
======================
Estimación de densidad continua del aire (rho_hat) por juego a partir de cinemática
de desaceleración y arrastre aerodinámico para la LMB (Diablos AeroVision).

Cumple estrictamente con docs/CONTRACT.md (HACK-003) y AGENTS.md:
- Uso exclusivo de variables cinemáticas de velocidad y aceleración (ay0, vx0, vy0, vz0,
  SpeedDrop, RelSpeed, ZoneTime).
- PROHIBICIÓN ABSOLUTA de usar InducedVertBreak, HorzBreak o quiebre Magnus en rho_hat
  para evitar circularidad científica con los modelos de movimiento (H1).
- Suavizado empírico bayesiano (Empirical Bayes shrinkage, m = 30) para juegos con
  pocos lanzamientos rápidos hacia la mediana de su categoría de altitud.
- Normalización por la mediana de juegos en 'No Altitude' (rho_hat == 1.000) y acotamiento
  en el rango físico [0.70, 1.15].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_game_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima el índice continuo de densidad del aire (rho_hat) para cada juego único
    presente en df utilizando cinemática de arrastre y suavizado empírico bayesiano.

    Trabaja estrictamente sobre una copia sin mutar el DataFrame original.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos con columnas cinemáticas e identificación de juego.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas exactas:
        ["game_anon_id", "rho_hat", "n_fastballs", "altitude_category"]
        Una fila por game_anon_id único, sin nulos, con rho_hat en float64.
    """
    if df.empty or "game_anon_id" not in df.columns:
        return pd.DataFrame(
            columns=["game_anon_id", "rho_hat", "n_fastballs", "altitude_category"]
        ).astype({
            "game_anon_id": "object",
            "rho_hat": "float64",
            "n_fastballs": "int64",
            "altitude_category": "object",
        })

    work_df = df.copy()

    # 1. Asegurar altitude_category limpia y sin nulos por juego
    if "altitude_category" not in work_df.columns:
        work_df["altitude_category"] = "No Altitude"
    else:
        alt_series = work_df["altitude_category"].astype(object)
        mask_missing = alt_series.isna() | alt_series.isin(["None", "Undefined", ""])
        if mask_missing.any():
            valid_alt = work_df.loc[~mask_missing]
            if not valid_alt.empty and "game_anon_id" in valid_alt.columns:
                game_alt_mode = (
                    valid_alt.groupby("game_anon_id")["altitude_category"]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
                )
                imputed_alt = work_df["game_anon_id"].map(game_alt_mode).fillna("No Altitude")
            else:
                imputed_alt = pd.Series("No Altitude", index=work_df.index)

            alt_series = np.where(mask_missing, imputed_alt, alt_series)
            work_df["altitude_category"] = (
                pd.Series(alt_series, index=work_df.index)
                .replace({"None": "No Altitude", "Undefined": "No Altitude", "": "No Altitude"})
                .fillna("No Altitude")
                .astype(str)
            )
        else:
            work_df["altitude_category"] = alt_series.astype(str)

    # 2. Cálculo del proxy instantáneo de arrastre específico k_i
    # k_i = ay0 / (vx0^2 + vy0^2 + vz0^2)
    # Fallback: SpeedDrop / (RelSpeed^2 * max(ZoneTime, 0.30))
    has_kinematics = all(c in work_df.columns for c in ["ay0", "vx0", "vy0", "vz0"])
    if has_kinematics:
        ay0 = pd.to_numeric(work_df["ay0"], errors="coerce")
        vx0 = pd.to_numeric(work_df["vx0"], errors="coerce")
        vy0 = pd.to_numeric(work_df["vy0"], errors="coerce")
        vz0 = pd.to_numeric(work_df["vz0"], errors="coerce")
        v2 = vx0**2 + vy0**2 + vz0**2
        kin_valid = ay0.notna() & v2.notna() & (v2 > 0.0)
        k_kin = np.where(kin_valid, ay0 / v2, np.nan)
    else:
        k_kin = np.full(len(work_df), np.nan)

    speed_drop = pd.to_numeric(work_df.get("SpeedDrop", np.nan), errors="coerce")
    rel_speed = pd.to_numeric(work_df.get("RelSpeed", np.nan), errors="coerce")
    zone_time = pd.to_numeric(work_df.get("ZoneTime", np.nan), errors="coerce")
    zone_time_clipped = np.maximum(zone_time.fillna(0.40), 0.30)
    rel_speed_sq = rel_speed**2

    fb_valid = speed_drop.notna() & rel_speed_sq.notna() & (rel_speed_sq > 0.0)
    k_fb = np.where(fb_valid, speed_drop / (rel_speed_sq * zone_time_clipped), np.nan)

    k_i = np.where(np.isnan(k_kin), k_fb, k_kin)
    work_df["_k_i"] = k_i

    # 3. Filtrar lanzamientos rápidos (RelSpeed >= 82.0; fallback: todos los pitcheos del juego)
    is_fast = (rel_speed >= 82.0) & work_df["_k_i"].notna()
    games_with_fast = set(work_df.loc[is_fast, "game_anon_id"].unique())

    keep_mask = (work_df["game_anon_id"].isin(games_with_fast) & is_fast) | (
        ~work_df["game_anon_id"].isin(games_with_fast) & work_df["_k_i"].notna()
    )
    filtered_df = work_df[keep_mask]

    # 4. Agrupar por game_anon_id
    all_games = work_df.groupby("game_anon_id", observed=True)["altitude_category"].first()
    k_game = filtered_df.groupby("game_anon_id", observed=True)["_k_i"].median()
    n_fastballs = filtered_df.groupby("game_anon_id", observed=True)["_k_i"].count()

    game_df = pd.DataFrame({
        "game_anon_id": all_games.index,
        "altitude_category": all_games.astype(str).values,
    })
    game_df["k_game"] = game_df["game_anon_id"].map(k_game)
    game_df["n_fastballs"] = game_df["game_anon_id"].map(n_fastballs).fillna(0).astype("int64")

    # 5. Mediana por categoría de altitud k_bar_alt sobre los juegos de cada categoría
    alt_median = game_df.groupby("altitude_category", observed=True)["k_game"].median()
    global_k_median = game_df["k_game"].median()
    if np.isnan(global_k_median) or global_k_median <= 0.0:
        global_k_median = 1.0

    alt_median = alt_median.fillna(global_k_median)
    k_alt_g = game_df["altitude_category"].map(alt_median).fillna(global_k_median)

    # 6. Empirical Bayes shrinkage con peso m = 30
    n = game_df["n_fastballs"].astype(float)
    weight_data = n / (n + 30.0)
    weight_prior = 30.0 / (n + 30.0)

    k_obs = game_df["k_game"]
    k_hat = np.where(
        k_obs.isna(),
        k_alt_g,
        weight_data * k_obs + weight_prior * k_alt_g,
    )

    # 7. Normalización por la mediana de k_hat en 'No Altitude' y recorte [0.70, 1.15]
    no_alt_mask = game_df["altitude_category"] == "No Altitude"
    if no_alt_mask.any():
        denom = np.nanmedian(k_hat[no_alt_mask])
    else:
        denom = np.nanmedian(k_hat)

    if np.isnan(denom) or denom <= 0.0:
        denom = 1.0

    rho_hat = np.clip(k_hat / denom, 0.70, 1.15)

    result_df = pd.DataFrame({
        "game_anon_id": game_df["game_anon_id"],
        "rho_hat": rho_hat.astype("float64"),
        "n_fastballs": game_df["n_fastballs"].astype("int64"),
        "altitude_category": game_df["altitude_category"].astype(str),
    }).reset_index(drop=True)

    return result_df


def attach_rho_hat(
    df: pd.DataFrame,
    game_rho_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adjunta la columna rho_hat (float64) a una copia de df a partir de game_rho_df
    (calculándolo con estimate_game_density(df) si game_rho_df es None).

    Rellena cualquier juego nuevo/faltante con la mediana de su altitude_category (o 1.0).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos.
    game_rho_df : pd.DataFrame | None
        Tabla por juego con columnas game_anon_id, rho_hat y altitude_category.

    Returns
    -------
    pd.DataFrame
        Copia enriquecida de df con la columna rho_hat añadida.
    """
    df_out = df.copy()
    if df_out.empty:
        df_out["rho_hat"] = pd.Series(dtype="float64")
        return df_out

    if game_rho_df is None:
        game_rho_df = estimate_game_density(df_out)

    rho_map = game_rho_df.set_index("game_anon_id")["rho_hat"].to_dict()
    mapped_rho = df_out["game_anon_id"].map(rho_map)

    # Mediana por categoría de altitud para juegos no contemplados
    if "altitude_category" in game_rho_df.columns:
        alt_median_map = (
            game_rho_df.groupby("altitude_category", observed=True)["rho_hat"]
            .median()
            .to_dict()
        )
    else:
        alt_median_map = {}

    if "altitude_category" in df_out.columns:
        alt_series = df_out["altitude_category"].astype(str)
        fallback_rho = alt_series.map(alt_median_map).fillna(1.0)
    else:
        fallback_rho = pd.Series(1.0, index=df_out.index)

    df_out["rho_hat"] = mapped_rho.fillna(fallback_rho).astype("float64")
    return df_out

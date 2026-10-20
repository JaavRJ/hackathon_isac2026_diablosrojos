"""
src/pipeline_step1.py
=====================
Pipeline Integrador del Dataset Procesado para el modelo Stuff+ LMB
(Reto 2 Diablos Rojos - Diablos AeroVision).

Cumple estrictamente con docs/CONTRACT.md, AGENTS.md y docs/TASKS.md (HACK-008):
- Ejecución secuencial de los módulos HACK-002 a HACK-006:
  1. Limpieza e imputación física (clean_and_impute).
  2. Ingeniería de variables físicas base y geometría de zona (add_physical_features).
  3. Estimación de densidad continua por juego y asignación de rho_hat (estimate_game_density, attach_rho_hat).
  4. Identidad física invariante a la altitud pitch_id (assign_invariant_pitch_id).
  5. Características de arsenal relacional, dinámica de secuencia y túneles (add_arsenal_and_tunnel_features).
  6. Calibración empírica LMB de valores de carrera (xrv_pitcher) y targets multi-objetivo
     ajustados EXCLUSIVAMENTE sobre 2024-2025 para garantizar CERO data leakage hacia 2026.
  7. Generación de reportes agregados (< 2 MB, sin filas pitch-by-pitch):
     - reports/altitude/game_density_summary.csv
     - reports/altitude/relabeling_audit.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR, PROCESSED_DATA_PATH
from src.features.arsenal import add_arsenal_and_tunnel_features
from src.features.identity import assign_invariant_pitch_id, audit_trackman_relabeling
from src.features.physical import add_physical_features, clean_and_impute
from src.io import load_raw_pitches
from src.physics.density import attach_rho_hat, estimate_game_density
from src.targets import attach_targets, fit_lmb_run_values


def _build_game_density_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el resumen agregado de la densidad continua rho_hat por categoría
    de altitud y año a nivel de juego único (game_anon_id).

    Columnas resultantes:
    ["altitude_category", "year", "n_games", "n_pitches",
     "rho_hat_median", "rho_hat_mean", "rho_hat_std", "rho_hat_p10", "rho_hat_p90"]
    """
    cols = [
        "altitude_category",
        "year",
        "n_games",
        "n_pitches",
        "rho_hat_median",
        "rho_hat_mean",
        "rho_hat_std",
        "rho_hat_p10",
        "rho_hat_p90",
    ]

    if df.empty or "game_anon_id" not in df.columns or "rho_hat" not in df.columns:
        return pd.DataFrame(columns=cols).astype({
            "altitude_category": "object",
            "year": "int64",
            "n_games": "int64",
            "n_pitches": "int64",
            "rho_hat_median": "float64",
            "rho_hat_mean": "float64",
            "rho_hat_std": "float64",
            "rho_hat_p10": "float64",
            "rho_hat_p90": "float64",
        })

    # Resumen primero a nivel de juego único
    game_level = (
        df.groupby(["game_anon_id", "altitude_category", "year"], observed=True)
        .agg(
            rho_hat=("rho_hat", "first"),
            n_pitches_game=("rho_hat", "count"),
        )
        .reset_index()
    )

    summary = (
        game_level.groupby(["altitude_category", "year"], observed=True)
        .agg(
            n_games=("game_anon_id", "nunique"),
            n_pitches=("n_pitches_game", "sum"),
            rho_hat_median=("rho_hat", "median"),
            rho_hat_mean=("rho_hat", "mean"),
            rho_hat_std=("rho_hat", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            rho_hat_p10=("rho_hat", lambda s: float(np.percentile(s, 10))),
            rho_hat_p90=("rho_hat", lambda s: float(np.percentile(s, 90))),
        )
        .reset_index()
    )

    summary = summary[cols]
    summary["altitude_category"] = summary["altitude_category"].astype(str)
    summary["year"] = summary["year"].astype(int)
    summary["n_games"] = summary["n_games"].astype(int)
    summary["n_pitches"] = summary["n_pitches"].astype(int)
    for c in ["rho_hat_median", "rho_hat_mean", "rho_hat_std", "rho_hat_p10", "rho_hat_p90"]:
        summary[c] = summary[c].astype(float)

    return summary


def build_processed_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Construye el dataset procesado integrando secuencialmente HACK-002 a HACK-006:
    1. Limpieza e imputación de features físicas (clean_and_impute).
    2. Enriquecimiento de variables físicas y geométricas de zona (add_physical_features).
    3. Estimación de densidad continua de aire por juego (estimate_game_density, attach_rho_hat).
    4. Identidad de pitcheo invariante a altitud (assign_invariant_pitch_id).
    5. Métricas de arsenal relacional, dinámica de secuencia y túneles (add_arsenal_and_tunnel_features).
    6. Calibración empírica LMB de valores de carrera sobre 2024-2025 SIN fuga de 2026
       (fit_lmb_run_values, attach_targets).
    7. Auditoría de reetiquetado de TrackMan y resumen de densidad de juegos.

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame crudo con columnas de TrackMan LMB. No se muta in-place.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        (df_procesado, metadata_dict)
        donde metadata_dict contiene:
        - "rv_table": dict devuelto por fit_lmb_run_values
        - "game_rho_df": pd.DataFrame devuelto por estimate_game_density
        - "game_density_summary": pd.DataFrame resumen de rho_hat por categoría y año
        - "relabeling_audit": pd.DataFrame devuelto por audit_trackman_relabeling
    """
    df = raw_df.copy()

    # Manejo defensivo de IDs faltantes en datos crudos
    if "pitcher_anon_id" in df.columns and df["pitcher_anon_id"].isna().any():
        df["pitcher_anon_id"] = df["pitcher_anon_id"].fillna("pitcher_unknown").astype(str)
    if "year" in df.columns and df["year"].isna().any():
        df["year"] = df["year"].fillna(2024).astype(int)
    if "game_anon_id" in df.columns and df["game_anon_id"].isna().any():
        df["game_anon_id"] = df["game_anon_id"].fillna("game_unknown").astype(str)

    # 1. Limpieza e imputación física (HACK-002)
    df = clean_and_impute(df)

    # 2. Features físicas y geométricas de zona (HACK-002)
    df = add_physical_features(df)

    # 3. Estimación y asignación de rho_hat por juego (HACK-003)
    game_rho_df = estimate_game_density(df)
    df = attach_rho_hat(df, game_rho_df)

    # 4. Asignación de identidad invariante pitch_id y pitch_cluster_id (HACK-004)
    df = assign_invariant_pitch_id(df)

    # 5. Features de arsenal y túneles de release y decisión (HACK-005)
    df = add_arsenal_and_tunnel_features(df)

    # 6. CRÍTICO: Ajustar valores de carrera LMB EXCLUSIVAMENTE sobre 2024-2025 (HACK-006)
    # Cero Data Leakage de 2026
    train_mask = df["year"].isin([2024, 2025])
    train_df = df[train_mask]
    rv_table = fit_lmb_run_values(train_df)

    # Aplicar attach_targets a todo el DataFrame (incluyendo 2026)
    df = attach_targets(df, rv_table)

    # 7. Resúmenes agregados (sin filas pitch-by-pitch)
    game_density_summary = _build_game_density_summary(df)
    relabeling_audit = audit_trackman_relabeling(df)

    metadata = {
        "rv_table": rv_table,
        "game_rho_df": game_rho_df,
        "game_density_summary": game_density_summary,
        "relabeling_audit": relabeling_audit,
    }

    return df, metadata


def run_pipeline_step1(
    raw_path: Path | str | None = None,
    output_parquet_path: Path | str | None = None,
    reports_dir: Path | str | None = None,
    sample_n: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Carga datos crudos, ejecuta build_processed_dataset, guarda el parquet procesado
    y exporta los reportes agregados en reports/altitude/.

    Parameters
    ----------
    raw_path : Path | str | None
        Ruta al parquet crudo. Si es None, usa load_raw_pitches(sample_n=sample_n).
    output_parquet_path : Path | str | None
        Ruta destino del parquet procesado. Por defecto data/stuff_processed_df.parquet.
    reports_dir : Path | str | None
        Directorio para reportes. Por defecto reports/.
    sample_n : int | None
        Si se especifica, toma una muestra determinista de lanzamientos.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        (df_procesado, metadata_dict)
    """
    if raw_path is not None:
        raw_p = Path(raw_path)
        if not raw_p.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos crudos en: {raw_p}")
        raw_df = pd.read_parquet(raw_p, engine="pyarrow")
        if sample_n is not None:
            sample_size = min(sample_n, len(raw_df))
            raw_df = raw_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        raw_df = load_raw_pitches(sample_n=sample_n)

    df, meta = build_processed_dataset(raw_df)

    # 3. Guardar dataset procesado
    out_path = Path(output_parquet_path) if output_parquet_path is not None else PROCESSED_DATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # 4. Exportar reportes agregados
    if reports_dir is None:
        base_rep = BASE_DIR / "reports"
    else:
        base_rep = Path(reports_dir)

    if base_rep.name == "altitude":
        alt_dir = base_rep
    else:
        alt_dir = base_rep / "altitude"

    alt_dir.mkdir(parents=True, exist_ok=True)

    density_csv = alt_dir / "game_density_summary.csv"
    relabel_csv = alt_dir / "relabeling_audit.csv"

    meta["game_density_summary"].to_csv(density_csv, index=False)
    meta["relabeling_audit"].to_csv(relabel_csv, index=False)

    # Validaciones estrictas
    max_bytes = 2 * 1024 * 1024  # 2 MB
    for csv_path, name in [(density_csv, "game_density_summary"), (relabel_csv, "relabeling_audit")]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Reporte {name} no se creó en {csv_path}")
        size = csv_path.stat().st_size
        if size >= max_bytes:
            raise ValueError(f"Reporte {name} supera el límite de 2 MB: {size} bytes")

        # Verificar que no contenga PitchUID ni datos pitch-by-pitch
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if "PitchUID" in cols:
            raise ValueError(f"Reporte {name} contiene PitchUID (violación de agregación)")

    return df, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Integrador del Dataset Procesado LMB (Diablos AeroVision HACK-008)"
    )
    parser.add_argument("--raw-path", type=str, default=None, help="Ruta al archivo parquet de datos crudos")
    parser.add_argument("--output", type=str, default=None, help="Ruta al archivo parquet procesado de salida")
    parser.add_argument("--reports-dir", type=str, default=None, help="Directorio para exportar reportes")
    parser.add_argument("--sample-n", type=int, default=None, help="Muestra determinista de N lanzamientos")

    args = parser.parse_args()

    t0 = time.perf_counter()
    print("Iniciando Pipeline Paso 1 (HACK-008)...")
    df_proc, meta_dict = run_pipeline_step1(
        raw_path=args.raw_path,
        output_parquet_path=args.output,
        reports_dir=args.reports_dir,
        sample_n=args.sample_n,
    )
    t1 = time.perf_counter()
    print(f"Pipeline completado exitosamente en {t1 - t0:.2f} s:")
    print(f" - Filas procesadas: {len(df_proc):,}")
    print(f" - Columnas: {len(df_proc.columns)}")
    print(f" - Parquet guardado en: {args.output or PROCESSED_DATA_PATH}")
    print(" - Reportes exportados en reports/altitude/")

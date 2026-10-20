"""
tests/test_pipeline_step1.py
============================
Suite de pruebas para HACK-008: Pipeline Integrador del Dataset Procesado (src/pipeline_step1.py).

Verifica:
1. Criterio de velocidad: build_processed_dataset(synthetic_lmb_df) corre en < 5.0 s.
2. Contrato completo de columnas: Presencia de todas las columnas de CONTRACT.md
   (HACK-002 a HACK-006) sin nulos ni infinitos.
3. Cero data leakage de 2026: fit_lmb_run_values se entrena exclusivamente con 2024-2025.
   Alterar filas de 2026 no afecta rv_table ni xrv_pitcher de 2024-2025.
4. Integridad de los reportes agregados y metadatos: game_density_summary y relabeling_audit
   pesan < 2 MB, no contienen filas pitch-by-pitch ni columnas de ID individual (PitchUID).
5. Invariancia del DataFrame crudo (sin mutación in-place).
6. Ejecución integral de run_pipeline_step1 con rutas temporales.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from src.pipeline_step1 import build_processed_dataset, run_pipeline_step1

# Columnas derivadas oficiales exigidas por docs/CONTRACT.md (§2 HACK-002 a HACK-006)
CONTRACT_DERIVED_COLS = [
    # HACK-002 (Físicas y Geometría)
    "HorzBreak_norm",
    "RelSide_norm",
    "HorzRelAngle_norm",
    "HorzApprAngle_norm",
    "PlateLocSide_norm",
    "TotalInducedBreak",
    "BreakAngle_deg",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "SpinToBreakRatio",
    "DragProxy",
    "arm_slot_deg",
    # HACK-003 (Densidad continua)
    "rho_hat",
    # HACK-004 (Identidad invariante)
    "pitch_cluster_id",
    "pitch_id",
    # HACK-005 (Arsenal y Túneles)
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
    # HACK-006 (Targets multi-objetivo y Run Values LMB)
    "target_whiff",
    "target_chase",
    "target_called_strike",
    "target_groundball",
    "target_weak_contact",
    "target_hard_contact",
    "target_out_on_pitch",
    "xrv_pitcher",
]


def test_build_processed_dataset_timing(synthetic_lmb_df: pd.DataFrame) -> None:
    """Verifica que el pipeline integrador complete su ejecución en menos de 5.0 segundos."""
    t0 = time.perf_counter()
    df_proc, meta = build_processed_dataset(synthetic_lmb_df)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"build_processed_dataset tardó {elapsed:.3f} s (límite: 5.0 s)"
    assert len(df_proc) > 0
    assert isinstance(meta, dict)


def test_contract_columns_presence_and_cleanliness(synthetic_lmb_df: pd.DataFrame) -> None:
    """Verifica presencia de todas las columnas de CONTRACT.md sin nulos ni infinitos."""
    df_proc, _ = build_processed_dataset(synthetic_lmb_df)

    # 1. Presencia de todas las columnas del contrato
    missing = [c for c in CONTRACT_DERIVED_COLS if c not in df_proc.columns]
    assert not missing, f"Faltan columnas de docs/CONTRACT.md: {missing}"

    # 2. Ausencia de nulos en todas las columnas derivadas
    for col in CONTRACT_DERIVED_COLS:
        null_count = df_proc[col].isna().sum()
        assert null_count == 0, f"Columna '{col}' contiene {null_count} valores nulos"

    # 3. Ausencia de infinitos en variables numéricas
    for col in CONTRACT_DERIVED_COLS:
        if pd.api.types.is_numeric_dtype(df_proc[col]):
            inf_count = np.isinf(df_proc[col].to_numpy()).sum()
            assert inf_count == 0, f"Columna '{col}' contiene {inf_count} valores infinitos"

    # 4. Verificación de tipos esperados
    assert df_proc["pitch_id"].dtype == "string"
    assert df_proc["primary_fb_id"].dtype == "string"
    assert df_proc["prev_pitch_id"].dtype == "string"
    assert pd.api.types.is_integer_dtype(df_proc["pitch_cluster_id"])
    assert df_proc["rho_hat"].dtype == np.float64
    assert df_proc["xrv_pitcher"].dtype == np.float64


def test_no_leakage_from_season_2026(synthetic_lmb_df: pd.DataFrame) -> None:
    """
    CRÍTICO: Verifica que fit_lmb_run_values se calibre exclusivamente con 2024-2025.
    Alterar RunsScored o resultados únicamente en 2026 no debe mutar rv_table
    ni el xrv_pitcher de 2024-2025.
    """
    df1, meta1 = build_processed_dataset(synthetic_lmb_df)

    # 1. 2026 no debe figurar en train_years
    train_years = meta1["rv_table"]["train_years"]
    assert 2026 not in train_years, f"Fuga detectada: 2026 presente en train_years: {train_years}"
    assert set(train_years).issubset({2024, 2025})

    # 2. Modificar drásticamente los resultados en 2026
    corrupted_df = synthetic_lmb_df.copy()
    mask_2026 = corrupted_df["year"] == 2026
    assert mask_2026.sum() > 0, "No hay lanzamientos de 2026 en synthetic_lmb_df"

    corrupted_df.loc[mask_2026, "RunsScored"] = 100.0
    corrupted_df.loc[mask_2026, "home_run"] = 1
    corrupted_df.loc[mask_2026, "is_hit"] = 1

    df2, meta2 = build_processed_dataset(corrupted_df)

    # 3. rv_table debe ser exactamente idéntica
    assert meta1["rv_table"]["event_xrv"] == meta2["rv_table"]["event_xrv"], (
        "rv_table mutó tras alterar datos de 2026 (fuga de datos)"
    )
    assert meta1["rv_table"]["re_matrix"] == meta2["rv_table"]["re_matrix"]

    # 4. xrv_pitcher en 2024-2025 no debe cambiar en absoluto
    mask_train = df1["year"].isin([2024, 2025])
    np.testing.assert_allclose(
        df1.loc[mask_train, "xrv_pitcher"].to_numpy(),
        df2.loc[mask_train, "xrv_pitcher"].to_numpy(),
        err_msg="xrv_pitcher en 2024-2025 cambió tras alterar 2026",
    )


def test_metadata_and_aggregated_reports(synthetic_lmb_df: pd.DataFrame) -> None:
    """Verifica la estructura y columnas de los reportes agregados en metadata."""
    _, meta = build_processed_dataset(synthetic_lmb_df)

    for required_key in ["rv_table", "game_rho_df", "game_density_summary", "relabeling_audit"]:
        assert required_key in meta, f"Falta llave '{required_key}' en metadata"

    # 1. game_density_summary
    density_sum = meta["game_density_summary"]
    expected_density_cols = [
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
    assert list(density_sum.columns) == expected_density_cols
    assert "PitchUID" not in density_sum.columns
    assert len(density_sum) > 0
    assert (density_sum["n_games"] > 0).all()
    assert (density_sum["n_pitches"] > 0).all()

    # 2. relabeling_audit
    audit = meta["relabeling_audit"]
    expected_audit_cols = [
        "pitch_type",
        "auto_pct_no_alt",
        "auto_pct_extreme",
        "auto_diff_pp",
        "invariant_pct_no_alt",
        "invariant_pct_extreme",
        "invariant_diff_pp",
    ]
    assert list(audit.columns) == expected_audit_cols
    assert "PitchUID" not in audit.columns


def test_raw_df_immutability(synthetic_lmb_df: pd.DataFrame) -> None:
    """Garantiza que build_processed_dataset no altere el DataFrame original in-place."""
    raw_copy = synthetic_lmb_df.copy(deep=True)
    _ = build_processed_dataset(synthetic_lmb_df)

    pd.testing.assert_frame_equal(synthetic_lmb_df, raw_copy)


def test_run_pipeline_step1_execution(tmp_path: Path, synthetic_lmb_df: pd.DataFrame) -> None:
    """Verifica run_pipeline_step1 completo: guardado en parquet y reportes CSV."""
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "stuff_processed.parquet"
    rep_dir = tmp_path / "reports"

    synthetic_lmb_df.to_parquet(raw_path, index=False)

    df_proc, meta = run_pipeline_step1(
        raw_path=raw_path,
        output_parquet_path=out_path,
        reports_dir=rep_dir,
    )

    # 1. Parquet creado y legible
    assert out_path.exists()
    df_disk = pd.read_parquet(out_path)
    assert len(df_disk) == len(df_proc)
    assert "rho_hat" in df_disk.columns
    assert "xrv_pitcher" in df_disk.columns

    # 2. Reportes CSV exportados en subdirectory altitude
    density_csv = rep_dir / "altitude" / "game_density_summary.csv"
    relabel_csv = rep_dir / "altitude" / "relabeling_audit.csv"

    assert density_csv.exists()
    assert relabel_csv.exists()

    # 3. Límite de tamaño (< 2 MB)
    assert density_csv.stat().st_size < 2 * 1024 * 1024
    assert relabel_csv.stat().st_size < 2 * 1024 * 1024

    # 4. Cero PitchUID ni datos pitch-by-pitch
    d_df = pd.read_csv(density_csv)
    r_df = pd.read_csv(relabel_csv)

    assert "PitchUID" not in d_df.columns
    assert "PitchUID" not in r_df.columns
    assert len(d_df) < 50
    assert len(r_df) < 20

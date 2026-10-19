"""
tests/test_identity.py
======================
Pruebas unitarias para la identidad de pitcheo invariante a la altitud (HACK-004)
y auditoría del reetiquetado de TrackMan.

Cumple estrictamente con docs/CONTRACT.md y notas del Agente Revisor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.identity import (
    CLUSTER_FEATURE_COLS,
    CLASSIFIER_FEATURE_COLS,
    FORBIDDEN_BREAK_COLS,
    INVARIANT_RELEASE_COLS,
    VALID_PITCH_IDS,
    _select_optimal_k_and_cluster,
    assign_invariant_pitch_id,
    audit_trackman_relabeling,
)
from src.features.physical import add_physical_features, clean_and_impute


@pytest.fixture
def clean_lmb_df(synthetic_lmb_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture de lanzamientos limpios e imputados con variables físicas base."""
    return add_physical_features(clean_and_impute(synthetic_lmb_df))


def test_assign_invariant_pitch_id_creates_required_columns(clean_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que assign_invariant_pitch_id genere pitch_cluster_id (int32)
    y pitch_id (tipo string de pandas) sin ningún valor nulo y dentro de las categorías válidas.
    """
    res = assign_invariant_pitch_id(clean_lmb_df)

    # 1. Conservar número de filas y no mutar el original
    assert len(res) == len(clean_lmb_df)
    assert "pitch_cluster_id" not in clean_lmb_df.columns
    assert "pitch_id" not in clean_lmb_df.columns

    # 2. Tipos de datos exactos según CONTRACT.md
    assert res["pitch_cluster_id"].dtype == np.int32
    assert res["pitch_cluster_id"].isna().sum() == 0
    assert (res["pitch_cluster_id"] >= 0).all()

    # pitch_id debe ser de tipo string de pandas sin nulos
    assert res["pitch_id"].dtype == "string"
    assert res["pitch_id"].isna().sum() == 0
    assert (res["pitch_id"] != "").all()
    assert (res["pitch_id"] != "nan").all()

    # Todos los pitch_id deben pertenecer a las categorías canónicas
    unique_ids = set(res["pitch_id"].unique())
    assert unique_ids.issubset(set(VALID_PITCH_IDS))

    # rel_velo_delta debe haberse calculado sin nulos
    assert "rel_velo_delta" in res.columns
    assert res["rel_velo_delta"].isna().sum() == 0


def test_invariance_against_magnus_break_alteration(clean_lmb_df: pd.DataFrame) -> None:
    """
    PRUEBA CRÍTICA DE INVARIANCIA (HACK-004):
    Simula la pérdida de quiebre en altitud alterando fuertemente InducedVertBreak,
    HorzBreak, VertBreak, pfxx, pfxz y alterando AutoPitchType en Extreme Altitude.
    Comprueba que pitch_cluster_id y pitch_id sean EXACTAMENTE IDÉNTICOS.
    """
    df1 = assign_invariant_pitch_id(clean_lmb_df)

    # Crear versión alterada simulando altitud extrema / reetiquetado de radar
    df_modified = clean_lmb_df.copy()
    df_modified["InducedVertBreak"] -= 4.0
    df_modified["HorzBreak"] -= 3.0
    df_modified["VertBreak"] -= 4.0
    if "TotalInducedBreak" in df_modified.columns:
        df_modified["TotalInducedBreak"] -= 5.0
    if "BreakAngle_deg" in df_modified.columns:
        df_modified["BreakAngle_deg"] += 35.0
    if "SpinToBreakRatio" in df_modified.columns:
        df_modified["SpinToBreakRatio"] *= 0.4
    if "pfxx" in df_modified.columns:
        df_modified["pfxx"] -= 2.0
    if "pfxz" in df_modified.columns:
        df_modified["pfxz"] -= 2.0

    # Alterar deliberadamente el etiquetado de TrackMan en altitud extrema
    extreme_mask = df_modified["altitude_category"] == "Extreme Altitude"
    df_modified.loc[extreme_mask, "AutoPitchType"] = "Cutter"

    df2 = assign_invariant_pitch_id(df_modified)

    # Invarianza 100%: los clusters e identidades deben ser exactamente idénticos
    assert (df1["pitch_cluster_id"] == df2["pitch_cluster_id"]).all(), (
        "pitch_cluster_id cambió al alterar el quiebre aerodinámico; viola la regla de invariancia."
    )
    assert (df1["pitch_id"] == df2["pitch_id"]).all(), (
        "pitch_id cambió al alterar el quiebre aerodinámico o AutoPitchType en altitud; "
        "viola la regla de invariancia."
    )


def test_forbidden_break_columns_are_strictly_disjoint() -> None:
    """
    Verifica que ninguna columna de quiebre o movimiento prohibida esté incluida
    en las listas de features de clasificación o clustering.
    """
    assert FORBIDDEN_BREAK_COLS.isdisjoint(set(INVARIANT_RELEASE_COLS))
    assert FORBIDDEN_BREAK_COLS.isdisjoint(set(CLASSIFIER_FEATURE_COLS))
    assert FORBIDDEN_BREAK_COLS.isdisjoint(set(CLUSTER_FEATURE_COLS))


def test_audit_trackman_relabeling_structure_and_values(clean_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que audit_trackman_relabeling produzca la tabla de auditoría
    con las 7 columnas esperadas, porcentajes válidos y diferencias consistentes.
    """
    df_with_id = assign_invariant_pitch_id(clean_lmb_df)
    audit_df = audit_trackman_relabeling(df_with_id)

    expected_cols = [
        "pitch_type",
        "auto_pct_no_alt",
        "auto_pct_extreme",
        "auto_diff_pp",
        "invariant_pct_no_alt",
        "invariant_pct_extreme",
        "invariant_diff_pp",
    ]
    assert list(audit_df.columns) == expected_cols
    assert len(audit_df) > 0

    assert (audit_df["pitch_type"] != "").all()

    for col in ["auto_pct_no_alt", "auto_pct_extreme", "invariant_pct_no_alt", "invariant_pct_extreme"]:
        assert (audit_df[col] >= 0.0).all()
        assert (audit_df[col] <= 100.0).all()

    np.testing.assert_allclose(
        audit_df["auto_diff_pp"].to_numpy(),
        (audit_df["auto_pct_extreme"] - audit_df["auto_pct_no_alt"]).to_numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        audit_df["invariant_diff_pp"].to_numpy(),
        (audit_df["invariant_pct_extreme"] - audit_df["invariant_pct_no_alt"]).to_numpy(),
        atol=1e-5,
    )

    audit_df2 = audit_trackman_relabeling(clean_lmb_df)
    assert list(audit_df2.columns) == expected_cols
    assert len(audit_df2) == len(audit_df)


def test_optimal_k_selection_by_bic() -> None:
    """
    Prueba el criterio BIC para la selección de K (Nota 2 del Revisor):
    Construye un lanzador con N=90 lanzamientos (max_k = min(6, 90//15) = 6)
    pero exactamente 3 clusters bien separados en el espacio cinemático de release.
    Comprueba que el algoritmo selecciona K == 3 en lugar de forzar K == 6.
    """
    rng = np.random.default_rng(42)

    # 30 rectas (RelSpeed ~95, SpinRate ~2400, RelSide ~1.8, RelHeight ~6.0)
    fb_velo = rng.normal(95.0, 0.8, 30)
    fb_spin = rng.normal(2400.0, 40.0, 30)
    fb_sin = rng.normal(0.0, 0.05, 30)
    fb_cos = rng.normal(0.9, 0.05, 30)

    # 30 sliders (RelSpeed ~84, SpinRate ~2300, RelSide ~1.9, RelHeight ~5.9)
    sl_velo = rng.normal(84.0, 0.8, 30)
    sl_spin = rng.normal(2300.0, 40.0, 30)
    sl_sin = rng.normal(-0.7, 0.05, 30)
    sl_cos = rng.normal(0.1, 0.05, 30)

    # 30 cambios (RelSpeed ~86, SpinRate ~1700, RelSide ~1.7, RelHeight ~6.0)
    ch_velo = rng.normal(86.0, 0.8, 30)
    ch_spin = rng.normal(1700.0, 40.0, 30)
    ch_sin = rng.normal(0.8, 0.05, 30)
    ch_cos = rng.normal(0.2, 0.05, 30)

    df_pitcher = pd.DataFrame({
        "pitcher_anon_id": ["pitcher_custom"] * 90,
        "year": [2025] * 90,
        "altitude_category": ["No Altitude"] * 90,
        "AutoPitchType": ["Four-Seam"] * 30 + ["Slider"] * 30 + ["Changeup"] * 30,
        "RelSpeed": np.concatenate([fb_velo, sl_velo, ch_velo]),
        "SpinRate": np.concatenate([fb_spin, sl_spin, ch_spin]),
        "SpinAxis_sin": np.concatenate([fb_sin, sl_sin, ch_sin]),
        "SpinAxis_cos": np.concatenate([fb_cos, sl_cos, ch_cos]),
        "RelHeight": [6.0] * 90,
        "RelSide_norm": [1.8] * 90,
        "Extension": [6.2] * 90,
    })

    # Verificar función auxiliar directa
    p90 = np.percentile(df_pitcher["RelSpeed"], 90)
    df_pitcher["rel_velo_delta"] = df_pitcher["RelSpeed"] - p90
    X_grp = df_pitcher[CLUSTER_FEATURE_COLS].to_numpy()
    X_scaled = (X_grp - np.mean(X_grp, axis=0)) / (np.std(X_grp, axis=0) + 1e-6)

    best_k, labels = _select_optimal_k_and_cluster(X_scaled, max_k=6)
    assert best_k == 3, f"Se esperaba K=3 por BIC en 3 clusters separados, pero se obtuvo K={best_k}"
    assert len(np.unique(labels)) == 3

    # Verificar integración completa en assign_invariant_pitch_id
    res = assign_invariant_pitch_id(df_pitcher)
    unique_clusters = len(res["pitch_cluster_id"].unique())
    assert unique_clusters == 3, (
        f"assign_invariant_pitch_id debía seleccionar K=3 clusters, pero asignó {unique_clusters}"
    )


def test_no_altitude_mode_threshold_5_to_9_vs_10() -> None:
    """
    Prueba el umbral estricto de >= 10 lanzamientos en 'No Altitude' (Nota 3 del Revisor):
    - Con 7 lanzamientos en No Altitude: NO alcanza el umbral de 10, por lo que NO usa
      la moda observada del cluster en No Altitude, sino la predicción del clasificador a nivel del mar.
    - Con 12 lanzamientos en No Altitude: SÍ alcanza el umbral de 10, por lo que usa
      la moda observada del cluster en No Altitude.
    """
    # Base de lanzamientos de entrenamiento a nivel del mar para el clasificador de referencia
    rng = np.random.default_rng(123)
    n_train = 60
    train_df = pd.DataFrame({
        "pitcher_anon_id": ["pitcher_ref"] * n_train,
        "year": [2025] * n_train,
        "altitude_category": ["No Altitude"] * n_train,
        "AutoPitchType": ["Four-Seam"] * 30 + ["Slider"] * 30,
        "RelSpeed": np.concatenate([rng.normal(95.0, 0.5, 30), rng.normal(84.0, 0.5, 30)]),
        "SpinRate": np.concatenate([rng.normal(2400.0, 30.0, 30), rng.normal(2300.0, 30.0, 30)]),
        "SpinAxis_sin": np.concatenate([rng.normal(0.0, 0.03, 30), rng.normal(-0.7, 0.03, 30)]),
        "SpinAxis_cos": np.concatenate([rng.normal(0.9, 0.03, 30), rng.normal(0.1, 0.03, 30)]),
        "RelHeight": [6.0] * n_train,
        "RelSide_norm": [1.8] * n_train,
        "Extension": [6.2] * n_train,
    })

    # Caso A: Un lanzador con un cluster de 15 lanzamientos cuyas características cinemáticas son
    # claramente de Four-Seam (95 mph, 2400 rpm), pero con solo 7 lanzamientos en 'No Altitude'
    # etiquetados artificialmente como 'Curveball' (el resto en 'Extreme Altitude').
    # Como 7 < 10, DEBE ignorar 'Curveball' y asignar la predicción del clasificador ('Four-Seam').
    pitcher_a = pd.DataFrame({
        "pitcher_anon_id": ["pitcher_a"] * 15,
        "year": [2025] * 15,
        "altitude_category": ["No Altitude"] * 7 + ["Extreme Altitude"] * 8,
        "AutoPitchType": ["Curveball"] * 7 + ["Curveball"] * 8,
        "RelSpeed": rng.normal(95.0, 0.3, 15),
        "SpinRate": rng.normal(2400.0, 20.0, 15),
        "SpinAxis_sin": rng.normal(0.0, 0.02, 15),
        "SpinAxis_cos": rng.normal(0.9, 0.02, 15),
        "RelHeight": [6.0] * 15,
        "RelSide_norm": [1.8] * 15,
        "Extension": [6.2] * 15,
    })

    df_combined_a = pd.concat([train_df, pitcher_a], ignore_index=True)
    res_a = assign_invariant_pitch_id(df_combined_a)
    sub_a = res_a[res_a["pitcher_anon_id"] == "pitcher_a"]

    # Al tener < 10 en No Altitude, el modelo predice Four-Seam por cinemática
    assert (sub_a["pitch_id"] == "Four-Seam").all(), (
        f"Con 7 lanzamientos en No Altitude (<10), debía predecir 'Four-Seam' vía modelo, "
        f"pero obtuvo {sub_a['pitch_id'].iloc[0]}"
    )

    # Caso B: Mismo lanzador pero con 12 lanzamientos en 'No Altitude' etiquetados como 'Curveball'.
    # Como 12 >= 10, DEBE usar la moda observada del cluster en No Altitude ('Curveball').
    pitcher_b = pd.DataFrame({
        "pitcher_anon_id": ["pitcher_b"] * 15,
        "year": [2025] * 15,
        "altitude_category": ["No Altitude"] * 12 + ["Extreme Altitude"] * 3,
        "AutoPitchType": ["Curveball"] * 12 + ["Curveball"] * 3,
        "RelSpeed": rng.normal(95.0, 0.3, 15),
        "SpinRate": rng.normal(2400.0, 20.0, 15),
        "SpinAxis_sin": rng.normal(0.0, 0.02, 15),
        "SpinAxis_cos": rng.normal(0.9, 0.02, 15),
        "RelHeight": [6.0] * 15,
        "RelSide_norm": [1.8] * 15,
        "Extension": [6.2] * 15,
    })

    df_combined_b = pd.concat([train_df, pitcher_b], ignore_index=True)
    res_b = assign_invariant_pitch_id(df_combined_b)
    sub_b = res_b[res_b["pitcher_anon_id"] == "pitcher_b"]

    # Al tener >= 10 en No Altitude, adopta la moda del cluster ('Curveball')
    assert (sub_b["pitch_id"] == "Curveball").all(), (
        f"Con 12 lanzamientos en No Altitude (>=10), debía adoptar la moda 'Curveball', "
        f"pero obtuvo {sub_b['pitch_id'].iloc[0]}"
    )


def test_edge_cases_and_robustness(clean_lmb_df: pd.DataFrame) -> None:
    """
    Verifica el comportamiento robusto ante subconjuntos pequeños, pitchers con pocos pitcheos
    y ausencia total de la categoría 'No Altitude' (usando fallback cinemático sin leakage).
    """
    # 1. Pitcher con muy pocos lanzamientos (ej. 3 lanzamientos)
    small_df = clean_lmb_df.iloc[:3].copy()
    res_small = assign_invariant_pitch_id(small_df)
    assert len(res_small) == 3
    assert res_small["pitch_cluster_id"].dtype == np.int32
    assert (res_small["pitch_cluster_id"] == 0).all()
    assert res_small["pitch_id"].dtype == "string"
    assert res_small["pitch_id"].notna().all()

    # 2. Subconjunto sin 'No Altitude': verifica uso de _ReleaseKinematicsFallbackPredictor
    extreme_only = clean_lmb_df[clean_lmb_df["altitude_category"] == "Extreme Altitude"].copy()
    res_ext = assign_invariant_pitch_id(extreme_only)
    assert len(res_ext) == len(extreme_only)
    assert res_ext["pitch_cluster_id"].isna().sum() == 0
    assert res_ext["pitch_id"].dtype == "string"
    assert res_ext["pitch_id"].isna().sum() == 0
    assert set(res_ext["pitch_id"].unique()).issubset(set(VALID_PITCH_IDS))

    # 3. Llamada sobre DataFrame crudo sin variables de normalización previas
    raw_sub = clean_lmb_df.drop(columns=["RelSide_norm", "SpinAxis_sin", "SpinAxis_cos"], errors="ignore")
    res_raw = assign_invariant_pitch_id(raw_sub)
    assert "RelSide_norm" in res_raw.columns
    assert "SpinAxis_sin" in res_raw.columns
    assert "SpinAxis_cos" in res_raw.columns
    assert res_raw["pitch_id"].dtype == "string"
    assert res_raw["pitch_id"].notna().all()

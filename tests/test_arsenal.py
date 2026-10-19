"""
tests/test_arsenal.py
=====================
Pruebas unitarias para el arsenal relacional, dinámica de secuencia (prev_pitch_id)
y métricas de túnel de pitcheo (release y punto de decisión a 170 ms) para HACK-005.

Cumple estrictamente con docs/CONTRACT.md, AGENTS.md y docs/TASKS.md:
- Verificación de las 10 columnas oficiales de HACK-005 y sus dtypes exactos.
- Cero valores NaN ni inf en todas las columnas.
- Invariancia estricta a AutoPitchType (uso exclusivo de pitch_id).
- Validación de aislamiento contextual de PA en prev_pitch_id (0-0 siempre 'NONE',
  sin filtraciones entre bateadores, lanzadores, entradas o juegos).
- Mediana de velo_diff_vs_fb approx 0.0 para la propia recta primaria.
- Caso analítico controlado de tunnel_dist_release_in, tunnel_sim_release,
  tunnel_dist_170ms_in y tunnel_sim_170ms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.arsenal import (
    ARSENAL_FEATURE_COLS,
    DECISION_POINT_TIME,
    FASTBALL_TYPES,
    TUNNEL_DECAY_SCALE_IN,
    add_arsenal_and_tunnel_features,
)
from src.features.identity import assign_invariant_pitch_id
from src.features.physical import add_physical_features, clean_and_impute
from src.physics.density import attach_rho_hat


@pytest.fixture
def prepared_lmb_df(synthetic_lmb_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture completo con pipeline previo (HACK-001 a HACK-004)."""
    df = clean_and_impute(synthetic_lmb_df)
    df = add_physical_features(df)
    df = attach_rho_hat(df)
    df = assign_invariant_pitch_id(df)
    return df


def test_creates_all_10_columns_with_exact_types(prepared_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que add_arsenal_and_tunnel_features genere exactamente las 10 columnas
    de docs/CONTRACT.md (HACK-005) con sus tipos exactos, sin nulos ni infinitos,
    y sin mutar el DataFrame original.
    """
    df_copy = prepared_lmb_df.copy()
    res = add_arsenal_and_tunnel_features(prepared_lmb_df)

    # 1. No mutar df original
    for col in ARSENAL_FEATURE_COLS:
        assert col not in prepared_lmb_df.columns
    assert len(res) == len(prepared_lmb_df)
    assert prepared_lmb_df.equals(df_copy)

    # 2. Las 10 columnas existen
    for col in ARSENAL_FEATURE_COLS:
        assert col in res.columns, f"Columna faltante: {col}"

    # 3. Tipos exactos según CONTRACT.md
    assert res["primary_fb_id"].dtype == "string"
    assert res["prev_pitch_id"].dtype == "string"
    assert res["primary_fb_id"].isna().sum() == 0
    assert res["prev_pitch_id"].isna().sum() == 0
    assert (res["primary_fb_id"] != "").all()
    assert (res["prev_pitch_id"] != "").all()

    numeric_cols = [c for c in ARSENAL_FEATURE_COLS if c not in ("primary_fb_id", "prev_pitch_id")]
    for col in numeric_cols:
        assert res[col].dtype == np.float64, f"{col} no es float64 (es {res[col].dtype})"
        assert res[col].isna().sum() == 0, f"{col} contiene NaN"
        assert not np.isinf(res[col]).any(), f"{col} contiene valores infinitos"


def test_prev_pitch_id_on_count_0_0_is_none(prepared_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que en todo lanzamiento con Balls == 0 y Strikes == 0,
    prev_pitch_id sea estrictamente 'NONE' (primer pitcheo de turno).
    """
    res = add_arsenal_and_tunnel_features(prepared_lmb_df)

    mask_0_0 = (res["Balls"] == 0) & (res["Strikes"] == 0)
    assert mask_0_0.sum() > 0, "Debe haber lanzamientos con conteo 0-0 en el dataset"
    assert (res.loc[mask_0_0, "prev_pitch_id"] == "NONE").all()


def test_prev_pitch_id_context_isolation(prepared_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que prev_pitch_id nunca mezcle lanzamientos de distintos bateadores,
    distintos pitchers, distintas medias entradas o distintos juegos.
    """
    res = add_arsenal_and_tunnel_features(prepared_lmb_df)

    for i in range(1, len(res)):
        curr = res.iloc[i]
        prev = res.iloc[i - 1]

        context_changed = (
            curr["game_anon_id"] != prev["game_anon_id"]
            or curr["pitcher_anon_id"] != prev["pitcher_anon_id"]
            or curr["batter_anon_id"] != prev["batter_anon_id"]
            or curr["Inning"] != prev["Inning"]
            or curr["Top/Bottom"] != prev["Top/Bottom"]
        )

        if context_changed:
            assert curr["prev_pitch_id"] == "NONE", (
                f"Fuga contextual en fila {i}: prev_pitch_id={curr['prev_pitch_id']} "
                f"a pesar de cambio en juego/pitcher/bateador/entrada"
            )


def test_prev_pitch_id_sequence_within_pa() -> None:
    """
    Verifica una secuencia controlada dentro de una misma PA donde el segundo
    y tercer lanzamiento toman correctamente el pitch_id del lanzamiento anterior.
    """
    records = [
        {
            "game_anon_id": "game_seq_01",
            "Inning": 1,
            "Top/Bottom": "Top",
            "pitcher_anon_id": "p_seq",
            "batter_anon_id": "b_seq",
            "Balls": 0,
            "Strikes": 0,
            "year": 2024,
            "altitude_category": "No Altitude",
            "pitch_id": "Four-Seam",
            "RelSpeed": 95.0,
            "InducedVertBreak": 16.0,
            "HorzBreak_norm": 8.0,
            "RelHeight": 6.0,
            "RelSide_norm": 2.0,
            "Extension": 6.0,
        },
        {
            "game_anon_id": "game_seq_01",
            "Inning": 1,
            "Top/Bottom": "Top",
            "pitcher_anon_id": "p_seq",
            "batter_anon_id": "b_seq",
            "Balls": 0,
            "Strikes": 1,
            "year": 2024,
            "altitude_category": "No Altitude",
            "pitch_id": "Slider",
            "RelSpeed": 85.0,
            "InducedVertBreak": 2.0,
            "HorzBreak_norm": -6.0,
            "RelHeight": 6.0,
            "RelSide_norm": 2.0,
            "Extension": 6.0,
        },
        {
            "game_anon_id": "game_seq_01",
            "Inning": 1,
            "Top/Bottom": "Top",
            "pitcher_anon_id": "p_seq",
            "batter_anon_id": "b_seq",
            "Balls": 1,
            "Strikes": 1,
            "year": 2024,
            "altitude_category": "No Altitude",
            "pitch_id": "Curveball",
            "RelSpeed": 79.0,
            "InducedVertBreak": -10.0,
            "HorzBreak_norm": -4.0,
            "RelHeight": 6.0,
            "RelSide_norm": 2.0,
            "Extension": 6.0,
        },
    ]
    df_seq = pd.DataFrame(records)
    res = add_arsenal_and_tunnel_features(df_seq)

    assert res.loc[0, "prev_pitch_id"] == "NONE"
    assert res.loc[1, "prev_pitch_id"] == "Four-Seam"
    assert res.loc[2, "prev_pitch_id"] == "Slider"


def test_median_velo_diff_for_primary_fastball_controlled() -> None:
    """
    Verifica que para los lanzamientos de la propia recta primaria del pitcher,
    la mediana de velo_diff_vs_fb sea pytest.approx(0.0, abs=1e-5) en un lanzador controlado.
    """
    records = []
    # Velocidades simétricas [92.0, 94.0, 96.0] -> media = 94.0 -> diffs = [-2.0, 0.0, 2.0] -> mediana = 0.0
    for v in [92.0, 94.0, 96.0]:
        records.append({
            "pitcher_anon_id": "p_sym",
            "year": 2024,
            "altitude_category": "No Altitude",
            "pitch_id": "Four-Seam",
            "RelSpeed": v,
            "InducedVertBreak": 16.0,
            "HorzBreak_norm": 8.0,
            "RelHeight": 6.0,
            "RelSide_norm": 2.0,
            "Extension": 6.0,
            "Balls": 0,
            "Strikes": 0,
        })
    df_sym = pd.DataFrame(records)
    res = add_arsenal_and_tunnel_features(df_sym)

    fb_rows = res[res["pitch_id"] == res["primary_fb_id"]]
    assert len(fb_rows) == 3
    assert fb_rows["velo_diff_vs_fb"].median() == pytest.approx(0.0, abs=1e-5)
    assert fb_rows["velo_diff_vs_fb"].mean() == pytest.approx(0.0, abs=1e-5)


def test_median_velo_diff_for_primary_fastball_synthetic(prepared_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que en el dataset sintético, para los lanzamientos de la recta primaria
    en No Altitude de cada lanzador, la media sea 0.0 (+/- 1e-5) y la mediana approx 0.0 (+/- 3.0 mph).
    """
    res = add_arsenal_and_tunnel_features(prepared_lmb_df)

    for (p, y), group in res.groupby(["pitcher_anon_id", "year"]):
        fb_no_alt = group[
            (group["pitch_id"] == group["primary_fb_id"])
            & (group["altitude_category"] == "No Altitude")
        ]
        if len(fb_no_alt) > 0:
            assert fb_no_alt["velo_diff_vs_fb"].mean() == pytest.approx(0.0, abs=1e-5)
            assert abs(float(fb_no_alt["velo_diff_vs_fb"].median())) < 3.5


def test_invariance_to_autopitchtype(prepared_lmb_df: pd.DataFrame) -> None:
    """
    REGLA CRÍTICA: Verifica que alterar AutoPitchType en el DataFrame de entrada
    (cuando pitch_id ya está presente) NO cambia ninguna de las 10 columnas
    de add_arsenal_and_tunnel_features (demostrando uso exclusivo de pitch_id).
    """
    res_clean = add_arsenal_and_tunnel_features(prepared_lmb_df)

    # Scramble total de AutoPitchType
    scrambled_df = prepared_lmb_df.copy()
    scrambled_df["AutoPitchType"] = "TotallyFabricatedPitchType"

    res_scrambled = add_arsenal_and_tunnel_features(scrambled_df)

    for col in ARSENAL_FEATURE_COLS:
        if col in ("primary_fb_id", "prev_pitch_id"):
            assert (res_clean[col] == res_scrambled[col]).all(), f"Discrepancia en {col}"
        else:
            np.testing.assert_allclose(
                res_clean[col].values,
                res_scrambled[col].values,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Discrepancia numérica en {col}",
            )


def test_analytical_tunnel_metrics_case() -> None:
    """
    Verifica un caso analítico controlado con valores exactos conocidos de:
    - tunnel_dist_release_in
    - tunnel_sim_release
    - tunnel_dist_170ms_in
    - tunnel_sim_170ms
    """
    # Pitch 1: Recta Four-Seam (referencia del pitcher)
    # PitchTrajectory:
    # X(0.17) = 50.0 - 100*0.17 + 10*(0.17^2) = 50 - 17 + 0.289 = 33.289 pies
    # Y(0.17) = 6.0 + 0 - 16*(0.17^2) = 6 - 0.4624 = 5.5376 pies
    # Z(0.17) = 2.0 + 5*0.17 + 0 = 2.85 pies
    p1 = {
        "pitcher_anon_id": "p_analytical",
        "year": 2024,
        "game_anon_id": "g_ana",
        "Inning": 1,
        "Top/Bottom": "Top",
        "batter_anon_id": "b_ana",
        "Balls": 0,
        "Strikes": 0,
        "altitude_category": "No Altitude",
        "pitch_id": "Four-Seam",
        "RelSpeed": 95.0,
        "InducedVertBreak": 16.0,
        "HorzBreak_norm": 8.0,
        "RelHeight": 6.0,
        "RelSide_norm": 2.0,
        "Extension": 6.0,
        "PitchTrajectoryXc0": 50.0,
        "PitchTrajectoryXc1": -100.0,
        "PitchTrajectoryXc2": 10.0,
        "PitchTrajectoryYc0": 6.0,
        "PitchTrajectoryYc1": 0.0,
        "PitchTrajectoryYc2": -16.0,
        "PitchTrajectoryZc0": 2.0,
        "PitchTrajectoryZc1": 5.0,
        "PitchTrajectoryZc2": 0.0,
    }

    # Pitch 2: Slider con desviaciones analíticas exactas:
    # Release: RelSide_norm desplazado 0.25 pies (3.0 pulgadas), RelHeight y Extension idénticos.
    # Distancia en release = 3.0 pulgadas -> sim_release = exp(-3.0 / 6.0) = exp(-0.5)
    # Decisión a 170 ms: PitchTrajectoryXc0 desplazado 0.5 pies (6.0 pulgadas), Y y Z idénticos.
    # Distancia en 170 ms = 6.0 pulgadas -> sim_170ms = exp(-6.0 / 6.0) = exp(-1.0)
    p2 = {
        "pitcher_anon_id": "p_analytical",
        "year": 2024,
        "game_anon_id": "g_ana",
        "Inning": 1,
        "Top/Bottom": "Top",
        "batter_anon_id": "b_ana",
        "Balls": 0,
        "Strikes": 1,
        "altitude_category": "No Altitude",
        "pitch_id": "Slider",
        "RelSpeed": 85.0,
        "InducedVertBreak": 2.0,
        "HorzBreak_norm": -6.0,
        "RelHeight": 6.0,
        "RelSide_norm": 2.25,  # +0.25 ft = +3.0 in
        "Extension": 6.0,
        "PitchTrajectoryXc0": 50.5,  # +0.5 ft = +6.0 in
        "PitchTrajectoryXc1": -100.0,
        "PitchTrajectoryXc2": 10.0,
        "PitchTrajectoryYc0": 6.0,
        "PitchTrajectoryYc1": 0.0,
        "PitchTrajectoryYc2": -16.0,
        "PitchTrajectoryZc0": 2.0,
        "PitchTrajectoryZc1": 5.0,
        "PitchTrajectoryZc2": 0.0,
    }

    df_ana = pd.DataFrame([p1, p2])
    res = add_arsenal_and_tunnel_features(df_ana)

    # Pitcheo 1 (recta primaria de referencia consigo misma)
    assert res.loc[0, "primary_fb_id"] == "Four-Seam"
    assert res.loc[0, "tunnel_dist_release_in"] == pytest.approx(0.0, abs=1e-6)
    assert res.loc[0, "tunnel_sim_release"] == pytest.approx(1.0, abs=1e-6)
    assert res.loc[0, "tunnel_dist_170ms_in"] == pytest.approx(0.0, abs=1e-6)
    assert res.loc[0, "tunnel_sim_170ms"] == pytest.approx(1.0, abs=1e-6)
    assert res.loc[0, "velo_diff_vs_fb"] == pytest.approx(0.0, abs=1e-6)
    assert res.loc[0, "prev_pitch_id"] == "NONE"

    # Pitcheo 2 (slider)
    assert res.loc[1, "primary_fb_id"] == "Four-Seam"
    assert res.loc[1, "tunnel_dist_release_in"] == pytest.approx(3.0, abs=1e-6)
    expected_sim_release = float(np.exp(-3.0 / TUNNEL_DECAY_SCALE_IN))
    assert res.loc[1, "tunnel_sim_release"] == pytest.approx(expected_sim_release, abs=1e-6)

    assert res.loc[1, "tunnel_dist_170ms_in"] == pytest.approx(6.0, abs=1e-6)
    expected_sim_170 = float(np.exp(-6.0 / TUNNEL_DECAY_SCALE_IN))
    assert res.loc[1, "tunnel_sim_170ms"] == pytest.approx(expected_sim_170, abs=1e-6)

    assert res.loc[1, "velo_diff_vs_fb"] == pytest.approx(-10.0, abs=1e-6)
    assert res.loc[1, "prev_pitch_id"] == "Four-Seam"


def test_pitch_usage_pct_bounds(prepared_lmb_df: pd.DataFrame) -> None:
    """
    Verifica que pitch_usage_pct esté estrictamente en (0, 1] y sume 1.0
    para cada (pitcher_anon_id, year).
    """
    res = add_arsenal_and_tunnel_features(prepared_lmb_df)

    assert (res["pitch_usage_pct"] > 0.0).all()
    assert (res["pitch_usage_pct"] <= 1.0).all()

    for (p, y), group in res.groupby(["pitcher_anon_id", "year"]):
        # La suma de las proporciones de uso de cada pitch_id único debe ser 1.0
        unique_pitches = group.drop_duplicates(subset=["pitch_id"])
        total_usage = unique_pitches["pitch_usage_pct"].sum()
        assert total_usage == pytest.approx(1.0, abs=1e-5)


def test_altitude_only_pitcher_rho_hat_adjustment() -> None:
    """
    Verifica que si un pitcher solo lanzó en altitud extrema, fb_ivb y fb_hb_norm
    se corrijan dividiendo entre la mediana de rho_hat del lanzador.
    """
    records = [{
        "pitcher_anon_id": "p_extreme_only",
        "year": 2025,
        "altitude_category": "Extreme Altitude",
        "rho_hat": 0.80,
        "pitch_id": "Four-Seam",
        "RelSpeed": 92.0,
        "InducedVertBreak": 8.0,
        "HorzBreak_norm": 4.0,
        "RelHeight": 6.0,
        "RelSide_norm": 2.0,
        "Extension": 6.0,
        "Balls": 0,
        "Strikes": 0,
    }]
    df_alt = pd.DataFrame(records)
    res = add_arsenal_and_tunnel_features(df_alt)

    # fb_ivb debe ser raw_ivb / rho_hat = 8.0 / 0.80 = 10.0
    # ivb_diff_vs_fb = 8.0 - 10.0 = -2.0
    assert res.loc[0, "ivb_diff_vs_fb"] == pytest.approx(-2.0, abs=1e-5)

    # fb_hb_norm debe ser raw_hb_norm / rho_hat = 4.0 / 0.80 = 5.0
    # hb_diff_vs_fb = 4.0 - 5.0 = -1.0
    assert res.loc[0, "hb_diff_vs_fb"] == pytest.approx(-1.0, abs=1e-5)


def test_empty_dataframe_handling() -> None:
    """Verifica que un DataFrame vacío devuelva las 10 columnas con tipos correctos."""
    empty_df = pd.DataFrame()
    res = add_arsenal_and_tunnel_features(empty_df)

    assert len(res) == 0
    for col in ARSENAL_FEATURE_COLS:
        assert col in res.columns
        if col in ("primary_fb_id", "prev_pitch_id"):
            assert res[col].dtype == "string"
        else:
            assert res[col].dtype == np.float64

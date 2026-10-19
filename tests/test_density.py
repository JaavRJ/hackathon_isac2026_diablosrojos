"""
tests/test_density.py
=====================
Pruebas unitarias para la estimación de densidad continua del aire (rho_hat)
por juego y su acoplamiento al DataFrame (HACK-003, Reto 2 Diablos Rojos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.physical import add_physical_features, clean_and_impute
from src.physics.density import attach_rho_hat, estimate_game_density


@pytest.fixture
def prepared_lmb_df(synthetic_lmb_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture que entrega el dataset sintético limpio y con features físicas."""
    cleaned = clean_and_impute(synthetic_lmb_df)
    return add_physical_features(cleaned)


def test_estimate_game_density_schema_and_types(prepared_lmb_df: pd.DataFrame) -> None:
    """Verifica el esquema, nombres exactos de columnas y tipos de datos devueltos."""
    game_rho = estimate_game_density(prepared_lmb_df)

    expected_cols = ["game_anon_id", "rho_hat", "n_fastballs", "altitude_category"]
    assert list(game_rho.columns) == expected_cols

    # Una fila por game_anon_id único
    assert len(game_rho) == prepared_lmb_df["game_anon_id"].nunique()
    assert game_rho["game_anon_id"].is_unique

    # Tipos de datos correctos
    assert game_rho["rho_hat"].dtype == np.float64
    assert np.issubdtype(game_rho["n_fastballs"].dtype, np.integer)
    assert game_rho["altitude_category"].dtype == object or isinstance(
        game_rho["altitude_category"].dtype, pd.StringDtype
    )

    # Cero valores nulos
    assert not game_rho.isna().any().any()


def test_rho_hat_no_altitude_median_normalized(prepared_lmb_df: pd.DataFrame) -> None:
    """Verifica que la mediana de rho_hat en juegos de 'No Altitude' sea 1.000 (+- 0.005)."""
    game_rho = estimate_game_density(prepared_lmb_df)
    no_alt_games = game_rho[game_rho["altitude_category"] == "No Altitude"]

    assert len(no_alt_games) > 0
    median_no_alt = float(no_alt_games["rho_hat"].median())
    assert median_no_alt == pytest.approx(1.000, abs=0.005)


def test_rho_hat_physical_altitude_ordering(prepared_lmb_df: pd.DataFrame) -> None:
    """Verifica estrictamente la jerarquía física: Extreme Altitude < Medium Altitude < No Altitude."""
    game_rho = estimate_game_density(prepared_lmb_df)

    median_extreme = float(
        game_rho.loc[game_rho["altitude_category"] == "Extreme Altitude", "rho_hat"].median()
    )
    median_medium = float(
        game_rho.loc[game_rho["altitude_category"] == "Medium Altitude", "rho_hat"].median()
    )
    median_no = float(
        game_rho.loc[game_rho["altitude_category"] == "No Altitude", "rho_hat"].median()
    )

    assert median_extreme < median_medium < median_no
    # En CDMX (Extreme Altitude), rho_hat esperado está en el rango ~0.80 - 0.89
    assert 0.75 <= median_extreme <= 0.90
    assert 0.88 <= median_medium <= 0.96


def test_rho_hat_bounds_and_no_nulls(prepared_lmb_df: pd.DataFrame) -> None:
    """Verifica que todos los valores de rho_hat estén acotados en [0.70, 1.15] sin NaN."""
    game_rho = estimate_game_density(prepared_lmb_df)
    assert (game_rho["rho_hat"] >= 0.70).all()
    assert (game_rho["rho_hat"] <= 1.15).all()
    assert not game_rho["rho_hat"].isna().any()

    # Verificar también tras adjuntar con attach_rho_hat
    attached_df = attach_rho_hat(prepared_lmb_df, game_rho)
    assert "rho_hat" in attached_df.columns
    assert len(attached_df) == len(prepared_lmb_df)
    assert attached_df["rho_hat"].dtype == np.float64
    assert not attached_df["rho_hat"].isna().any()
    assert (attached_df["rho_hat"] >= 0.70).all()
    assert (attached_df["rho_hat"] <= 1.15).all()


def test_non_circularity_magnus_break(prepared_lmb_df: pd.DataFrame) -> None:
    """
    CRÍTICO (Revisión HACK-003):
    Verifica que alterar arbitrariamente InducedVertBreak y HorzBreak NO cambie en absoluto
    la estimación de rho_hat, garantizando la no circularidad científica con H1.
    """
    df_original = prepared_lmb_df.copy()
    df_perturbed = prepared_lmb_df.copy()

    # Modificar drásticamente los quiebres inducidos
    df_perturbed["InducedVertBreak"] = df_perturbed["InducedVertBreak"] * 2.5 + 10.0
    df_perturbed["HorzBreak"] = df_perturbed["HorzBreak"] * -1.8 - 5.0
    if "TotalInducedBreak" in df_perturbed.columns:
        df_perturbed["TotalInducedBreak"] = 999.0

    rho_orig = estimate_game_density(df_original)
    rho_pert = estimate_game_density(df_perturbed)

    pd.testing.assert_series_equal(
        rho_orig["rho_hat"],
        rho_pert["rho_hat"],
        check_exact=True,
        obj="rho_hat invariante a modificaciones de Magnus break",
    )

    # Verificar también en attach_rho_hat
    att_orig = attach_rho_hat(df_original)
    att_pert = attach_rho_hat(df_perturbed)
    pd.testing.assert_series_equal(
        att_orig["rho_hat"],
        att_pert["rho_hat"],
        check_exact=True,
        obj="attached rho_hat invariante a modificaciones de Magnus break",
    )


def test_empirical_bayes_shrinkage_low_sample() -> None:
    """
    Verifica que un juego con pocos pitcheos (ej. n_fastballs = 2) se contrae fuertemente
    hacia la mediana de su categoría de altitud.
    """
    # Creamos un conjunto sintético pequeño con 3 juegos en 'No Altitude':
    # 2 juegos con 50 pitcheos a arrastre estándar (~0.0014),
    # y 1 juego con solo 2 pitcheos con arrastre atípicamente alto (~0.0028).
    rows = []
    # Juego 1 y 2: estándar
    for gid in ["game_norm_1", "game_norm_2"]:
        for _ in range(50):
            rows.append({
                "game_anon_id": gid,
                "altitude_category": "No Altitude",
                "RelSpeed": 92.0,
                "ZoneTime": 0.42,
                "SpeedDrop": 7.0,
                "ay0": 25.0,
                "vx0": 2.5,
                "vy0": -133.0,
                "vz0": -3.0,
            })

    # Juego 3: outlier con solo 2 pitcheos
    for _ in range(2):
        rows.append({
            "game_anon_id": "game_outlier_low_n",
            "altitude_category": "No Altitude",
            "RelSpeed": 92.0,
            "ZoneTime": 0.42,
            "SpeedDrop": 14.0,  # 2x arrastre
            "ay0": 50.0,       # 2x ay0
            "vx0": 2.5,
            "vy0": -133.0,
            "vz0": -3.0,
        })

    toy_df = pd.DataFrame(rows)
    res = estimate_game_density(toy_df)

    outlier_res = res[res["game_anon_id"] == "game_outlier_low_n"].iloc[0]
    norm_res = res[res["game_anon_id"] != "game_outlier_low_n"]

    assert outlier_res["n_fastballs"] == 2
    median_norm_rho = norm_res["rho_hat"].median()

    # Sin shrinkage, el juego tendría rho_hat = 2.0 (o clip a 1.15).
    # Con shrinkage m=30: peso del dato = 2/32 = 0.0625, peso del prior = 30/32 = 0.9375.
    # El valor esperado de rho_hat es aproximadamente 1.0 + 0.0625 * (2.0 - 1.0) = 1.0625.
    # Por tanto, debe estar mucho más cerca de 1.0 que de 2.0 (rho_hat < 1.10).
    assert outlier_res["rho_hat"] < 1.10
    assert abs(outlier_res["rho_hat"] - median_norm_rho) < 0.10


def test_fallback_drag_when_kinematics_missing(prepared_lmb_df: pd.DataFrame) -> None:
    """Verifica que el fallback basado en SpeedDrop funcione si ay0/vx0/vy0/vz0 no están presentes."""
    df_no_kin = prepared_lmb_df.drop(columns=["ay0", "vx0", "vy0", "vz0"])

    game_rho = estimate_game_density(df_no_kin)
    assert not game_rho.isna().any().any()

    # Mantiene el orden físico
    median_extreme = float(
        game_rho.loc[game_rho["altitude_category"] == "Extreme Altitude", "rho_hat"].median()
    )
    median_no = float(
        game_rho.loc[game_rho["altitude_category"] == "No Altitude", "rho_hat"].median()
    )
    assert median_extreme < median_no
    assert float(game_rho.loc[game_rho["altitude_category"] == "No Altitude", "rho_hat"].median()) == pytest.approx(1.0, abs=0.005)


def test_attach_rho_hat_missing_game_fallback(prepared_lmb_df: pd.DataFrame) -> None:
    """Verifica que attach_rho_hat asigne la mediana de la categoría a juegos no incluidos en game_rho_df."""
    game_rho = estimate_game_density(prepared_lmb_df)

    # Excluir intencionalmente un juego de game_rho
    excluded_game = game_rho.iloc[0]["game_anon_id"]
    excluded_alt = game_rho.iloc[0]["altitude_category"]
    partial_game_rho = game_rho[game_rho["game_anon_id"] != excluded_game]

    attached = attach_rho_hat(prepared_lmb_df, partial_game_rho)

    # El juego excluido debe recibir la mediana de su categoría
    expected_alt_median = partial_game_rho.loc[
        partial_game_rho["altitude_category"] == excluded_alt, "rho_hat"
    ].median()

    assigned_rho = attached.loc[attached["game_anon_id"] == excluded_game, "rho_hat"].iloc[0]
    assert assigned_rho == pytest.approx(expected_alt_median, rel=1e-5)
    assert not attached["rho_hat"].isna().any()


def test_raw_synthetic_df_with_null_altitude_category(synthetic_lmb_df: pd.DataFrame) -> None:
    """Verifica que funcione directamente sobre synthetic_lmb_df crudo que contiene altitude_category nulos."""
    # synthetic_lmb_df tiene 20 filas con altitude_category = None
    assert synthetic_lmb_df["altitude_category"].isna().any()

    game_rho = estimate_game_density(synthetic_lmb_df)
    assert len(game_rho) == synthetic_lmb_df["game_anon_id"].nunique()
    assert not game_rho.isna().any().any()
    assert set(game_rho["altitude_category"]) == {"No Altitude", "Medium Altitude", "Extreme Altitude"}

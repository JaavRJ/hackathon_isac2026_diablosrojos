"""
tests/test_splits_baseline.py
=============================
Suite de pruebas unitarias para las particiones oficiales (holdouts) y el baseline nulo:
- src/splits.py (get_pitcher_holdout_splits, get_season_holdout_split, get_park_holdout_split)
- src/baseline.py (CountBaselineModel, evaluate_against_baseline)

Cumple estrictamente con:
- docs/CONTRACT.md §1 y §2 (TARGET_COLS)
- docs/TASKS.md (HACK-007)
- AGENTS.md §4 (Validación obligatoria y baseline nulo)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLS
from src.features.physical import clean_and_impute
from src.targets import attach_targets, fit_lmb_run_values
from src.splits import (
    get_pitcher_holdout_splits,
    get_season_holdout_split,
    get_park_holdout_split,
)
from src.baseline import CountBaselineModel, evaluate_against_baseline


@pytest.fixture
def processed_lmb_df(synthetic_lmb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset sintético limpio con clean_and_impute y con los 8 targets oficiales
    adjuntados mediante fit_lmb_run_values y attach_targets.
    """
    cleaned = clean_and_impute(synthetic_lmb_df)
    # Entrenar run values exclusivamente con datos históricos (<= 2025)
    train_historical = cleaned[cleaned["year"].isin([2024, 2025])]
    rv_table = fit_lmb_run_values(train_historical)
    with_targets = attach_targets(cleaned, rv_table)
    return with_targets


# ==============================================================================
# 1. Pruebas de Particiones Oficiales (src/splits.py)
# ==============================================================================

def test_pitcher_holdout_splits_disjoint(processed_lmb_df: pd.DataFrame):
    """
    Verifica que en cada fold de get_pitcher_holdout_splits:
    - Los conjuntos de pitcher_anon_id entre train y test sean completamente disjuntos.
    - Ningún registro de la temporada 2026 aparezca en train ni en test.
    - Los índices devueltos funcionen correctamente tanto con .loc como con .iloc.
    """
    df = processed_lmb_df
    splits = get_pitcher_holdout_splits(df, n_splits=5)

    assert len(splits) == 5

    all_test_indices = []
    for train_idx, test_idx in splits:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(train_idx) > 0
        assert len(test_idx) > 0

        # Verificación explícita obligatoria por Reviewer Checklist
        train_pitchers = set(df.loc[train_idx, "pitcher_anon_id"])
        test_pitchers = set(df.loc[test_idx, "pitcher_anon_id"])
        assert train_pitchers.isdisjoint(test_pitchers)

        # Cero data leakage de la temporada 2026
        train_years = set(df.loc[train_idx, "year"])
        test_years = set(df.loc[test_idx, "year"])
        assert 2026 not in train_years
        assert 2026 not in test_years
        assert train_years.issubset({2024, 2025})
        assert test_years.issubset({2024, 2025})

        all_test_indices.extend(test_idx.tolist())

    # La unión de los test_idx debe cubrir exactamente todos los registros históricos de 2024 y 2025
    expected_historical_indices = df.index[df["year"].isin([2024, 2025])].tolist()
    assert sorted(all_test_indices) == sorted(expected_historical_indices)


def test_pitcher_holdout_splits_custom_index(processed_lmb_df: pd.DataFrame):
    """
    Verifica que get_pitcher_holdout_splits funcione correctamente en DataFrames con
    índices no RangeIndex (e.g. enteros arbitrarios o index reordenado).
    """
    df = processed_lmb_df.copy()
    # Asignar un índice no secuencial
    df.index = [i * 10 + 3 for i in range(len(df))]

    splits = get_pitcher_holdout_splits(df, n_splits=3)
    assert len(splits) == 3

    for train_idx, test_idx in splits:
        train_pitchers = set(df.loc[train_idx, "pitcher_anon_id"])
        test_pitchers = set(df.loc[test_idx, "pitcher_anon_id"])
        assert train_pitchers.isdisjoint(test_pitchers)
        assert not df.loc[train_idx, "year"].isin([2026]).any()
        assert not df.loc[test_idx, "year"].isin([2026]).any()


def test_season_holdout_split(processed_lmb_df: pd.DataFrame):
    """
    Verifica la partición temporal oficial de get_season_holdout_split:
    - train solo contiene {2024, 2025}.
    - test solo contiene {2026}.
    - Los conjuntos de años son disjuntos.
    - La suma de registros coincide con el total de filas del DataFrame.
    """
    df = processed_lmb_df
    train_idx, test_idx = get_season_holdout_split(df)

    assert isinstance(train_idx, np.ndarray)
    assert isinstance(test_idx, np.ndarray)
    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert len(train_idx) + len(test_idx) == len(df)

    train_years = set(df.loc[train_idx, "year"])
    test_years = set(df.loc[test_idx, "year"])

    assert train_years.isdisjoint(test_years)
    assert train_years == {2024, 2025}
    assert test_years == {2026}


def test_park_holdout_split(processed_lmb_df: pd.DataFrame):
    """
    Verifica la partición oficial de altitud de get_park_holdout_split:
    - train solo contiene 'No Altitude' y 'Medium Altitude' de 2024-2025.
    - test solo contiene 'Extreme Altitude' de 2024-2025.
    - Las categorías de altitud son mutuamente disjuntas.
    - Cero registros de 2026 en ninguna partición.
    """
    df = processed_lmb_df
    train_idx, test_idx = get_park_holdout_split(df)

    assert isinstance(train_idx, np.ndarray)
    assert isinstance(test_idx, np.ndarray)
    assert len(train_idx) > 0
    assert len(test_idx) > 0

    train_alts = set(df.loc[train_idx, "altitude_category"])
    test_alts = set(df.loc[test_idx, "altitude_category"])

    assert train_alts.isdisjoint(test_alts)
    assert train_alts.issubset({"No Altitude", "Medium Altitude"})
    assert test_alts == {"Extreme Altitude"}

    # Verificación de años
    assert set(df.loc[train_idx, "year"]).issubset({2024, 2025})
    assert set(df.loc[test_idx, "year"]).issubset({2024, 2025})


def test_splits_error_handling():
    """
    Verifica el manejo de errores en particiones cuando faltan columnas o registros.
    """
    # DataFrame vacío o sin años 2024-2025
    empty_df = pd.DataFrame({"year": [2026, 2026], "pitcher_anon_id": ["p1", "p2"]})
    with pytest.raises(ValueError, match="No se encontraron registros"):
        get_pitcher_holdout_splits(empty_df, n_splits=2)

    with pytest.raises(ValueError, match="2024 o 2025"):
        get_season_holdout_split(empty_df)

    # Sin columna year
    no_year_df = pd.DataFrame({"pitcher_anon_id": ["p1", "p2"]})
    with pytest.raises(ValueError, match="columna 'year'"):
        get_season_holdout_split(no_year_df)


# ==============================================================================
# 2. Pruebas de Baseline Nulo y Evaluación (src/baseline.py)
# ==============================================================================

def test_count_baseline_fit_and_predict(processed_lmb_df: pd.DataFrame):
    """
    Verifica el ajuste y predicción de CountBaselineModel:
    - Genera exactamente las 8 columnas oficiales de TARGET_COLS.
    - Las columnas son float64 y no contienen NaN.
    - Conserva el mismo índice del DataFrame de entrada.
    - La predicción es determinista y constante para el mismo conteo.
    """
    df = processed_lmb_df
    train_idx, test_idx = get_season_holdout_split(df)
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    model = CountBaselineModel()
    fitted_model = model.fit(train_df)
    assert fitted_model is model
    assert model.is_fitted_

    pred_df = model.predict(test_df)

    # 1. Columnas exactas y orden
    assert list(pred_df.columns) == TARGET_COLS

    # 2. Tipos y no NaN
    for col in TARGET_COLS:
        assert pred_df[col].dtype == np.float64
        assert not pred_df[col].isna().any(), f"NaN encontrado en predicción de {col}"

    # 3. Mismo índice
    assert pred_df.index.equals(test_df.index)

    # 4. Predicciones constantes por conteo
    # Filtrar dos filas con el mismo conteo en test_df
    count_series = test_df["count"].astype(str)
    first_count = count_series.iloc[0]
    same_count_mask = count_series == first_count

    if same_count_mask.sum() > 1:
        sub_preds = pred_df.loc[same_count_mask]
        for col in TARGET_COLS:
            # Todos los valores para el mismo conteo deben ser idénticos
            assert sub_preds[col].nunique() == 1, (
                f"Predicción no constante por conteo para {col} en conteo {first_count}"
            )


def test_count_baseline_fallback_for_unseen_count(processed_lmb_df: pd.DataFrame):
    """
    Verifica que ante un conteo nunca visto en fit (e.g. '4-2'), predict asigne
    la media global del target sin generar NaN ni lanzar excepción.
    """
    train_df = processed_lmb_df[processed_lmb_df["count"] == "0-0"].copy()
    model = CountBaselineModel().fit(train_df)

    unseen_df = pd.DataFrame({
        "count": ["4-2", "9-9"],
        "Balls": [4, 9],
        "Strikes": [2, 9],
    })

    preds = model.predict(unseen_df)
    assert len(preds) == 2
    assert not preds.isna().any().any()

    # Cada predicción debe coincidir exactamente con la media global de train_df
    for col in TARGET_COLS:
        expected_mean = float(train_df[col].mean())
        np.testing.assert_allclose(preds[col].to_numpy(), expected_mean)


def test_count_baseline_unfitted_raises():
    """
    Verifica que llamar predict antes de fit lance RuntimeError.
    """
    model = CountBaselineModel()
    dummy_df = pd.DataFrame({"count": ["0-0"]})
    with pytest.raises(RuntimeError, match="debe ser entrenado"):
        model.predict(dummy_df)


def test_evaluate_against_baseline_identity_skill_score_zero(processed_lmb_df: pd.DataFrame):
    """
    Verifica que al evaluar las predicciones del baseline contra sí mismas:
    - El skill_score y skill_score_pct sean exactamente 0.0 para todos los targets.
    - Para xrv_pitcher: model_rmse == baseline_rmse.
    - Para targets binarios: model_brier == baseline_brier, model_logloss == baseline_logloss.
    """
    df = processed_lmb_df
    train_idx, test_idx = get_season_holdout_split(df)
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    model = CountBaselineModel().fit(train_df)
    baseline_preds = model.predict(test_df)

    # Evaluar baseline contra sí mismo
    metrics_df = evaluate_against_baseline(
        y_true_df=test_df,
        y_pred_df=baseline_preds,
        baseline_pred_df=baseline_preds,
    )

    assert len(metrics_df) == 8
    assert list(metrics_df["target"]) == TARGET_COLS

    for _, row in metrics_df.iterrows():
        assert row["skill_score"] == pytest.approx(0.0, abs=1e-7)
        assert row["skill_score_pct"] == pytest.approx(0.0, abs=1e-7)

        if row["target_type"] == "continuous":
            assert row["model_rmse"] == pytest.approx(row["baseline_rmse"], abs=1e-7)
            assert np.isnan(row["model_brier"])
            assert np.isnan(row["model_logloss"])
        else:
            assert row["model_brier"] == pytest.approx(row["baseline_brier"], abs=1e-7)
            assert row["model_logloss"] == pytest.approx(row["baseline_logloss"], abs=1e-7)
            assert row["logloss_skill_score"] == pytest.approx(0.0, abs=1e-7)
            assert np.isnan(row["model_rmse"])


def test_evaluate_against_baseline_improved_predictor(processed_lmb_df: pd.DataFrame):
    """
    Verifica que un predictor mejorado que el baseline (mezclado con la verdad fundamental):
    - Tenga skill_score > 0.0 y skill_score_pct > 0.0 en todos los targets.
    - Menor RMSE para xrv_pitcher y menor Brier / Log-Loss para targets binarios.
    """
    df = processed_lmb_df
    train_idx, test_idx = get_season_holdout_split(df)
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    model = CountBaselineModel().fit(train_df)
    baseline_preds = model.predict(test_df)

    # Predictor mejorado: interpolación hacia y_true
    improved_preds = baseline_preds.copy()
    for col in TARGET_COLS:
        improved_preds[col] = 0.5 * baseline_preds[col] + 0.5 * test_df[col]

    metrics_df = evaluate_against_baseline(
        y_true_df=test_df,
        y_pred_df=improved_preds,
        baseline_pred_df=baseline_preds,
    )

    for _, row in metrics_df.iterrows():
        assert row["skill_score"] > 0.0, f"skill_score debe ser positivo para {row['target']}"
        assert row["skill_score_pct"] > 0.0

        if row["target_type"] == "continuous":
            assert row["model_rmse"] < row["baseline_rmse"]
        else:
            assert row["model_brier"] < row["baseline_brier"]
            assert row["model_logloss"] < row["baseline_logloss"]
            assert row["logloss_skill_score"] > 0.0

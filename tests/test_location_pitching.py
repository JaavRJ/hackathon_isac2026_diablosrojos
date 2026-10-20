"""
tests/test_location_pitching.py
================================
Pruebas unitarias y de integración para LocationPlusModel y PitchingPlusModel
(HACK-011).

Verifica:
1. Rechazo estricto de columnas no autorizadas en LocationPlusModel mediante whitelist
   (assert_valid_location_features rechaza rho_hat, RelHeight, Extension, RelSpeed, SpinRate, etc.).
2. Rechazo de variables de target (leakage) e IDs en ambos modelos.
3. Media = 100.0 (+- 0.05) y std = 10.0 (+- 0.05) sobre el conjunto de entrenamiento
   en sabor neutral para location_plus_neutral y pitching_plus_neutral.
4. En LocationPlusModel, location_plus_neutral y location_plus_cdmx son idénticos
   cuando la ubicación y contexto son iguales (delta_location_cdmx == 0.0).
5. En PitchingPlusModel, self.rho_extreme_median_ se calcula empíricamente de Extreme Altitude.
6. predict_all_targets() devuelve los 8 targets calibrados (en [0, 1] para los 7 binarios)
   y los scores Plus neutral y CDMX sin NaN.
7. predict() genera exactamente TARGET_COLS y es compatible con evaluate_against_baseline.
8. Integración con y sin movement_translator para ajuste de entorno CDMX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline import CountBaselineModel, evaluate_against_baseline
from src.config import FORBIDDEN_TARGET_COLS, TARGET_COLS
from src.models.location_pitching import (
    LOCATION_FEATURES,
    PITCHING_FEATURES,
    LocationPlusModel,
    PitchingPlusModel,
    assert_valid_location_features,
)
from src.pipeline_step1 import build_processed_dataset


@pytest.fixture(scope="module")
def processed_lmb_dataset(synthetic_lmb_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Procesa el dataset sintético LMB con build_processed_dataset y divide en train (<=2025) y test (2026).
    """
    proc_df, _ = build_processed_dataset(synthetic_lmb_df)
    train_df = proc_df[proc_df["year"] <= 2025].copy().reset_index(drop=True)
    test_df = proc_df[proc_df["year"] == 2026].copy().reset_index(drop=True)
    return train_df, test_df


# ==============================================================================
# 1. Pruebas de Validación de Features y Whitelist Estricta
# ==============================================================================

def test_location_features_allow_valid():
    """Verifica que las features oficiales de Location+ pasan la validación sin error."""
    assert_valid_location_features(LOCATION_FEATURES)


@pytest.mark.parametrize("unauthorized_col", [
    "rho_hat",
    "RelHeight",
    "RelSide",
    "Extension",
    "VertRelAngle",
    "RelSpeed",
    "SpinRate",
    "InducedVertBreak",
    "HorzBreak",
    "EffectiveVelo",
    "ZoneSpeed",
    "TotalInducedBreak",
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
])
def test_location_model_rejects_unauthorized_features(unauthorized_col: str):
    """
    Verifica que LocationPlusModel rechaza terminantemente mediante whitelist
    cualquier columna fuera de LOCATION_FEATURES (incluyendo rho_hat, variables
    físicas de release, velocidad, spin o quiebre).
    """
    test_cols = LOCATION_FEATURES + [unauthorized_col]
    with pytest.raises(ValueError, match="Features no autorizadas detectadas en LocationPlusModel"):
        assert_valid_location_features(test_cols)

    with pytest.raises(ValueError, match="Features no autorizadas detectadas en LocationPlusModel"):
        LocationPlusModel(feature_cols=test_cols)


@pytest.mark.parametrize("leakage_col", [
    "target_whiff",
    "target_called_strike",
    "xrv_pitcher",
    "RunsScored",
    "ExitSpeed",
    "PitchCall",
    "single",
    "home_run",
])
def test_location_and_pitching_reject_forbidden_targets(leakage_col: str):
    """Verifica que ambos modelos rechazan columnas de target/resultado directo como features."""
    loc_cols = LOCATION_FEATURES + [leakage_col]
    with pytest.raises(ValueError, match="Fuga de datos"):
        assert_valid_location_features(loc_cols)

    with pytest.raises(ValueError, match="Fuga de datos"):
        LocationPlusModel(feature_cols=loc_cols)

    pitching_cols = PITCHING_FEATURES + [leakage_col]
    with pytest.raises(ValueError, match="Fuga de datos"):
        PitchingPlusModel(feature_cols=pitching_cols)


@pytest.mark.parametrize("id_col", [
    "pitcher_anon_id",
    "batter_anon_id",
    "game_anon_id",
    "PitchUID",
])
def test_location_and_pitching_reject_id_cols(id_col: str):
    """Verifica que ambos modelos rechazan columnas de identificación/agrupación."""
    with pytest.raises(ValueError, match="Fuga de datos"):
        assert_valid_location_features(LOCATION_FEATURES + [id_col])

    with pytest.raises(ValueError, match="Fuga de datos"):
        LocationPlusModel(feature_cols=LOCATION_FEATURES + [id_col])

    with pytest.raises(ValueError, match="Fuga de datos"):
        PitchingPlusModel(feature_cols=PITCHING_FEATURES + [id_col])


# ==============================================================================
# 2. Pruebas de LocationPlusModel (Entrenamiento y Normalización)
# ==============================================================================

def test_location_plus_model_fit_and_scores(processed_lmb_dataset):
    """
    Verifica que LocationPlusModel:
    1. Ajusta exitosamente sobre train_df (<= 2025) usando exclusivamente LOCATION_FEATURES.
    2. En el conjunto de referencia evaluado en sabor neutral,
       location_plus_neutral tiene media 100.0 (+- 0.05) y std 10.0 (+- 0.05).
    3. location_plus_neutral y location_plus_cdmx son idénticos (delta_location_cdmx == 0.0)
       porque Location+ evalúa comando puro independiente de parque.
    """
    train_df, test_df = processed_lmb_dataset

    loc_model = LocationPlusModel()
    loc_model.fit(train_df)
    assert loc_model.is_fitted_

    scored_train = loc_model.predict_scores(train_df)

    # Validar tipos y ausencia de NaN
    for col in ["location_plus_neutral", "location_plus_cdmx", "delta_location_cdmx"]:
        assert col in scored_train.columns
        assert scored_train[col].dtype == np.float64
        assert not scored_train[col].isna().any(), f"{col} contiene NaN"

    # Validar regla estricta de normalización Score+ en sabor neutral
    mean_neutral = float(scored_train["location_plus_neutral"].mean())
    std_neutral = float(scored_train["location_plus_neutral"].std(ddof=0))

    assert np.isclose(mean_neutral, 100.0, atol=0.05), f"media de neutral fue {mean_neutral}, esperada 100.0 +- 0.05"
    assert np.isclose(std_neutral, 10.0, atol=0.05), f"std de neutral fue {std_neutral}, esperada 10.0 +- 0.05"

    # Validar que location_plus_neutral y location_plus_cdmx son idénticos y delta == 0.0
    np.testing.assert_allclose(scored_train["location_plus_neutral"], scored_train["location_plus_cdmx"], atol=1e-8)
    np.testing.assert_allclose(scored_train["delta_location_cdmx"], 0.0, atol=1e-8)

    # Evaluar en test_df (2026)
    scored_test = loc_model.predict_scores(test_df)
    assert len(scored_test) == len(test_df)
    assert not scored_test["location_plus_neutral"].isna().any()
    np.testing.assert_allclose(scored_test["location_plus_neutral"], scored_test["location_plus_cdmx"], atol=1e-8)


# ==============================================================================
# 3. Pruebas de PitchingPlusModel (Multi-Target y Normalización)
# ==============================================================================

def test_pitching_plus_model_fit_and_scores(processed_lmb_dataset):
    """
    Verifica que PitchingPlusModel:
    1. Ajusta exitosamente sobre train_df.
    2. Calcula empíricamente self.rho_extreme_median_ a partir de Extreme Altitude.
    3. pitching_plus_neutral tiene media 100.0 (+- 0.05) y std 10.0 (+- 0.05)
       sobre el set de entrenamiento evaluado en neutral.
    4. delta_pitching_cdmx == pitching_plus_cdmx - pitching_plus_neutral.
    """
    train_df, test_df = processed_lmb_dataset

    pitching_model = PitchingPlusModel()
    pitching_model.fit(train_df)
    assert pitching_model.is_fitted_

    # Comprobar que rho_extreme_median_ se calculó empíricamente de train_df
    extreme_expected = float(train_df.loc[train_df["altitude_category"] == "Extreme Altitude", "rho_hat"].median())
    assert np.isclose(pitching_model.rho_extreme_median_, extreme_expected, atol=1e-4)

    scored_train = pitching_model.predict_scores(train_df)

    for col in ["pitching_plus_neutral", "pitching_plus_cdmx", "delta_pitching_cdmx"]:
        assert col in scored_train.columns
        assert scored_train[col].dtype == np.float64
        assert not scored_train[col].isna().any()

    mean_neutral = float(scored_train["pitching_plus_neutral"].mean())
    std_neutral = float(scored_train["pitching_plus_neutral"].std(ddof=0))

    assert np.isclose(mean_neutral, 100.0, atol=0.05), f"media fue {mean_neutral}, esperada 100.0 +- 0.05"
    assert np.isclose(std_neutral, 10.0, atol=0.05), f"std fue {std_neutral}, esperada 10.0 +- 0.05"

    expected_delta = scored_train["pitching_plus_cdmx"] - scored_train["pitching_plus_neutral"]
    np.testing.assert_allclose(scored_train["delta_pitching_cdmx"], expected_delta, atol=1e-6)


def test_pitching_plus_model_predict_all_targets(processed_lmb_dataset):
    """
    Verifica que predict_all_targets():
    1. Retorna las 8 predicciones con nombres exactos pred_target_*.
    2. Las 7 probabilidades binarias están acotadas en [0.0, 1.0].
    3. Incluye pitching_plus_neutral, pitching_plus_cdmx y delta_pitching_cdmx.
    4. Mantiene el índice original de df y cero NaNs.
    """
    train_df, test_df = processed_lmb_dataset
    pitching_model = PitchingPlusModel()
    pitching_model.fit(train_df)

    preds_df = pitching_model.predict_all_targets(test_df)

    expected_target_preds = [
        "pred_target_whiff",
        "pred_target_chase",
        "pred_target_called_strike",
        "pred_target_groundball",
        "pred_target_weak_contact",
        "pred_target_hard_contact",
        "pred_target_out_on_pitch",
        "pred_xrv_pitcher",
    ]

    for p_col in expected_target_preds:
        assert p_col in preds_df.columns, f"Falta {p_col} en predict_all_targets"
        assert not preds_df[p_col].isna().any()

    # Validar límites de probabilidades binarias [0, 1]
    binary_pred_cols = [c for c in expected_target_preds if c != "pred_xrv_pitcher"]
    for b_col in binary_pred_cols:
        vals = preds_df[b_col]
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"{b_col} fuera de [0, 1]"

    assert "pitching_plus_neutral" in preds_df.columns
    assert "pitching_plus_cdmx" in preds_df.columns
    assert "delta_pitching_cdmx" in preds_df.columns
    assert len(preds_df) == len(test_df)
    assert (preds_df.index == test_df.index).all()


# ==============================================================================
# 4. Pruebas de Compatibilidad con CountBaselineModel y evaluate_against_baseline
# ==============================================================================

def test_pitching_plus_model_predict_and_evaluate_against_baseline(processed_lmb_dataset):
    """
    Verifica que predict(df) produce las 8 columnas exactas de TARGET_COLS y
    se evalúa directamente contra el CountBaselineModel mediante evaluate_against_baseline.
    """
    train_df, test_df = processed_lmb_dataset

    # 1. Ajustar baseline nulo por conteo
    baseline_model = CountBaselineModel()
    baseline_model.fit(train_df)
    baseline_preds = baseline_model.predict(test_df)

    # 2. Ajustar PitchingPlusModel
    pitching_model = PitchingPlusModel()
    pitching_model.fit(train_df)

    # 3. Predecir con predict(test_df)
    model_preds = pitching_model.predict(test_df)
    assert list(model_preds.columns) == TARGET_COLS
    assert not model_preds.isna().any().any()

    # 4. Evaluar con suite de baseline oficial (HACK-007)
    eval_df = evaluate_against_baseline(
        y_true_df=test_df[TARGET_COLS],
        y_pred_df=model_preds,
        baseline_pred_df=baseline_preds,
    )

    assert isinstance(eval_df, pd.DataFrame)
    assert len(eval_df) == len(TARGET_COLS)
    assert "target" in eval_df.columns
    assert "skill_score" in eval_df.columns


# ==============================================================================
# 5. Integración con MovementTranslator (Mock)
# ==============================================================================

class MockMovementTranslator:
    """Mock que implementa translate_movement según docs/CONTRACT.md."""

    def __init__(self, factor: float = 0.852) -> None:
        self.factor = factor
        self.call_count = 0

    def translate_movement(self, df: pd.DataFrame, target_rho: float) -> pd.DataFrame:
        self.call_count += 1
        out = df.copy()
        out["InducedVertBreak"] = out["InducedVertBreak"] * (target_rho / 1.0)
        out["HorzBreak_norm"] = out["HorzBreak_norm"] * (target_rho / 1.0)
        out["rho_hat"] = float(target_rho)
        return out


def test_pitching_plus_with_movement_translator(processed_lmb_dataset):
    """
    Verifica que PitchingPlusModel invoca correctamente el traductor físico
    cuando es provisto.
    """
    train_df, test_df = processed_lmb_dataset
    mock_translator = MockMovementTranslator()

    pitching_model = PitchingPlusModel(movement_translator=mock_translator)
    pitching_model.fit(train_df)
    assert mock_translator.call_count >= 1

    initial_calls = mock_translator.call_count
    scored = pitching_model.predict_scores(test_df, rho_cdmx=0.852)
    # Debe haber llamado al translator tanto para neutral (1.0) como para CDMX (0.852)
    assert mock_translator.call_count >= initial_calls + 2
    assert "pitching_plus_neutral" in scored.columns
    assert "pitching_plus_cdmx" in scored.columns

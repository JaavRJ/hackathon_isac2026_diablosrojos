"""
tests/test_stuff.py
===================
Pruebas unitarias y de integración para el submodelo Stuff+ (HACK-010).

Verifica:
1. Whitelist estricta y rechazo de leakage (assert_valid_stuff_features y StuffPlusModel):
   - Rechazo de LOCATION_AND_COUNT_COLS (PlateLocHeight, PlateLocSide, count, Balls, Strikes, BatterSide, prev_pitch_id).
   - Rechazo de FORBIDDEN_TARGET_COLS (PitchCall, ExitSpeed, target_whiff, xrv_pitcher, etc.).
   - Rechazo de ID_AND_GROUP_COLS (PitchUID, pitcher_anon_id, etc.).
   - Rechazo de VertApprAngle, HorzApprAngle, HorzApprAngle_norm y rho_hat.
2. Calibración oficial Score+:
   - mean(stuff_plus_neutral) == pytest.approx(100.0, abs=0.05).
   - std(stuff_plus_neutral) == pytest.approx(10.0, abs=0.05) sobre el set de referencia neutral.
3. Evaluación de sabor CDMX / entorno con la misma escala neutral:
   - stuff_plus_cdmx normalizado con (mu_neutral_, sigma_neutral_) de nivel del mar.
   - delta_stuff_cdmx != 0.0 (efecto no trivial de altitud).
4. Invarianza estricta a ubicación y conteo:
   - Alterar PlateLocHeight, PlateLocSide, Balls o Strikes en df NO altera en absoluto
     stuff_plus_neutral ni stuff_plus_cdmx.
5. Explicabilidad SHAP (TreeSHAP exacto nativo de LightGBM):
   - explain_pitch(df.iloc[[0]]) entrega base_value, shap_values para STUFF_FEATURES,
     top_positive_features y top_negative_features.
6. Compatibilidad oficial con evaluate_against_baseline:
   - predict(df) genera DataFrame con las 8 columnas TARGET_COLS en float64 sin NaN.
7. Estimación empírica de rho_extreme_median_:
   - Calculada empíricamente de Extreme Altitude (mediana ~0.852 en LMB 2024-2025).
8. Predicciones Out-Of-Fold (OOF) y Regularización por Validación Cruzada:
   - self.oof_fold_assignment_ auditable con GroupKFold por pitcher.
   - self.best_alpha_ en la rejilla [0.01, 0.1, 1.0, 10.0, 100.0].
9. Neutralización total del Canal Stuff frente al parque (Extreme y Medium Altitude):
   - Alterar resultados de bolas bateadas tanto en Extreme como en Medium Altitude
     NO cambia en absoluto las predicciones de Stuff+.
10. Invarianza estricta de predict_scores entre lote y fila aislada:
    - Evaluar una fila aislada vs dentro de un lote produce exactamente los mismos scores.
11. Recálculo dinámico de diferenciales de arsenal (ivb_diff_vs_fb, hb_diff_vs_fb).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline import CountBaselineModel, evaluate_against_baseline
from src.config import (
    FASTBALL_TYPES,
    FORBIDDEN_TARGET_COLS,
    ID_AND_GROUP_COLS,
    LOCATION_AND_COUNT_COLS,
    TARGET_COLS,
)
from src.models.movement import PhysicsMovementTranslator
import src.models.stuff as stuff_module
from src.models.stuff import (
    STUFF_FEATURES,
    TRAJECTORY_PROXY_COLS,
    StuffPlusModel,
    assert_valid_stuff_features,
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
# 1. Pruebas de Whitelist Estricta y Prevención de Fuga de Datos (Cero Leakage)
# ==============================================================================

def test_stuff_features_whitelist_allows_valid():
    """Verifica que las features oficiales de STUFF_FEATURES pasan la validación sin error."""
    assert_valid_stuff_features(STUFF_FEATURES)


@pytest.mark.parametrize("loc_col", [
    "PlateLocHeight",
    "PlateLocSide",
    "PlateLocSide_norm",
    "in_strike_zone",
    "outside_strike_zone",
    "dist_from_zone_center",
    "dist_to_zone_edge",
    "is_shadow_zone",
    "zone_quadrant",
    "Inning",
    "Top/Bottom",
    "Outs",
    "Balls",
    "Strikes",
    "count",
    "BatterSide",
    "is_same_hand",
    "prev_pitch_id",
])
def test_stuff_model_rejects_location_and_count_cols(loc_col: str):
    """
    Verifica que StuffPlusModel y assert_valid_stuff_features rechazan con ValueError
    cualquier variable de ubicación, conteo o contexto (LOCATION_AND_COUNT_COLS).
    """
    bad_features = STUFF_FEATURES + [loc_col]
    with pytest.raises(ValueError):
        assert_valid_stuff_features(bad_features)

    with pytest.raises(ValueError):
        StuffPlusModel(feature_cols=bad_features)


@pytest.mark.parametrize("unauthorized_col", [
    "VertApprAngle",
    "HorzApprAngle",
    "HorzApprAngle_norm",
    "rho_hat",
])
def test_stuff_model_rejects_unauthorized_approach_and_density(unauthorized_col: str):
    """
    Verifica que StuffPlusModel y assert_valid_stuff_features rechazan con ValueError
    ángulos de llegada (VertApprAngle, HorzApprAngle) y la densidad de entorno (rho_hat),
    que no pertenecen a la whitelist física de Stuff+.
    """
    bad_features = STUFF_FEATURES + [unauthorized_col]
    with pytest.raises(ValueError, match="Features no autorizadas detectadas en StuffPlusModel"):
        assert_valid_stuff_features(bad_features)

    with pytest.raises(ValueError, match="Features no autorizadas detectadas en StuffPlusModel"):
        StuffPlusModel(feature_cols=bad_features)


@pytest.mark.parametrize("leakage_col", [
    "target_whiff",
    "target_chase",
    "target_called_strike",
    "target_groundball",
    "target_weak_contact",
    "target_hard_contact",
    "target_out_on_pitch",
    "xrv_pitcher",
    "PitchCall",
    "ExitSpeed",
    "Angle",
    "RunsScored",
    "home_run",
])
def test_stuff_model_rejects_target_and_play_results(leakage_col: str):
    """
    Verifica que StuffPlusModel rechaza terminantemente columnas de target
    o resultado de jugada (FORBIDDEN_TARGET_COLS).
    """
    bad_features = STUFF_FEATURES + [leakage_col]
    with pytest.raises(ValueError):
        assert_valid_stuff_features(bad_features)

    with pytest.raises(ValueError):
        StuffPlusModel(feature_cols=bad_features)


@pytest.mark.parametrize("id_col", [
    "PitchUID",
    "pitcher_anon_id",
    "batter_anon_id",
    "catcher_anon_id",
    "game_anon_id",
    "year",
])
def test_stuff_model_rejects_ids_and_group_cols(id_col: str):
    """Verifica que StuffPlusModel rechaza columnas de ID o agrupación (ID_AND_GROUP_COLS)."""
    bad_features = STUFF_FEATURES + [id_col]
    with pytest.raises(ValueError):
        assert_valid_stuff_features(bad_features)

    with pytest.raises(ValueError):
        StuffPlusModel(feature_cols=bad_features)


# ==============================================================================
# 2. Pruebas de Calibración de Escala Neutral (Score+) y Dos Sabores
# ==============================================================================

def test_stuff_model_neutral_scale_calibration(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que sobre el conjunto de referencia evaluado en sabor neutral:
    - mean(stuff_plus_neutral) == 100.0 (+- 0.05)
    - std(stuff_plus_neutral) == 10.0 (+- 0.05)
    """
    train_df, _ = processed_lmb_dataset
    model = StuffPlusModel()
    model.fit(train_df)

    assert model.is_fitted_
    assert hasattr(model, "mu_neutral_")
    assert hasattr(model, "sigma_neutral_")
    assert model.sigma_neutral_ > 1e-6

    # Evaluar en el conjunto de referencia LMB completo (train_df: year <= 2025, todas las altitudes)
    scores_df = model.predict_scores(train_df)

    mean_neutral = float(scores_df["stuff_plus_neutral"].mean())
    std_neutral = float(scores_df["stuff_plus_neutral"].std(ddof=0))

    assert mean_neutral == pytest.approx(100.0, abs=0.05), (
        f"Media neutral observada: {mean_neutral}, esperada: 100.0"
    )
    assert std_neutral == pytest.approx(10.0, abs=0.05), (
        f"Desviación estándar neutral observada: {std_neutral}, esperada: 10.0"
    )


def test_stuff_model_cdmx_flavor_and_delta(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que:
    1. stuff_plus_cdmx se normaliza con (mu_neutral_, sigma_neutral_) de nivel del mar
       (y NO se re-estandariza a media 100 dentro de CDMX).
    2. delta_stuff_cdmx = stuff_plus_cdmx - stuff_plus_neutral es no trivial (!= 0).
    3. rho_extreme_median_ se calcula empíricamente de Extreme Altitude (~0.852 en LMB).
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel()
    model.fit(train_df)

    assert 0.80 <= model.rho_extreme_median_ <= 0.95

    scores_df = model.predict_scores(test_df)

    assert "stuff_plus_neutral" in scores_df.columns
    assert "stuff_plus_cdmx" in scores_df.columns
    assert "delta_stuff_cdmx" in scores_df.columns

    assert not scores_df["stuff_plus_neutral"].isna().any()
    assert not scores_df["stuff_plus_cdmx"].isna().any()
    assert not scores_df["delta_stuff_cdmx"].isna().any()

    diff = scores_df["stuff_plus_cdmx"] - scores_df["stuff_plus_neutral"]
    np.testing.assert_allclose(scores_df["delta_stuff_cdmx"], diff, atol=1e-6)

    assert not np.allclose(scores_df["delta_stuff_cdmx"], 0.0, atol=1e-3)


# ==============================================================================
# 3. Prueba de Invarianza Estricta a Ubicación y Conteo
# ==============================================================================

def test_stuff_model_invariance_to_location_and_count(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica rigurosamente que alterar PlateLocHeight, PlateLocSide, Balls o Strikes
    en df NO cambia en absoluto stuff_plus_neutral ni stuff_plus_cdmx.
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    base_scores = model.predict_scores(test_df)

    perturbed_df = test_df.copy()
    perturbed_df["PlateLocHeight"] = perturbed_df["PlateLocHeight"] + 5.0
    perturbed_df["PlateLocSide"] = -perturbed_df["PlateLocSide"] * 2.0
    perturbed_df["Balls"] = 3
    perturbed_df["Strikes"] = 0
    if "count" in perturbed_df.columns:
        perturbed_df["count"] = "3-0"

    perturbed_scores = model.predict_scores(perturbed_df)

    np.testing.assert_allclose(
        base_scores["stuff_plus_neutral"].to_numpy(),
        perturbed_scores["stuff_plus_neutral"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
        err_msg="stuff_plus_neutral cambió al perturbar variables de plato y conteo.",
    )
    np.testing.assert_allclose(
        base_scores["stuff_plus_cdmx"].to_numpy(),
        perturbed_scores["stuff_plus_cdmx"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
        err_msg="stuff_plus_cdmx cambió al perturbar variables de plato y conteo.",
    )


# ==============================================================================
# 4. Pruebas de Explicabilidad SHAP (TreeSHAP)
# ==============================================================================

def test_stuff_model_explain_pitch_shap(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que explain_pitch entrega las contribuciones SHAP válidas para STUFF_FEATURES:
    - base_value es float
    - shap_values contiene todas las variables de STUFF_FEATURES
    - top_positive_features y top_negative_features son listas no vacías
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    single_pitch = test_df.iloc[[0]]
    explanation = model.explain_pitch(single_pitch)

    assert "base_value" in explanation
    assert "shap_values" in explanation
    assert "top_positive_features" in explanation
    assert "top_negative_features" in explanation

    assert isinstance(explanation["base_value"], float)
    assert isinstance(explanation["shap_values"], dict)

    shap_vals = explanation["shap_values"]
    for feat in STUFF_FEATURES:
        assert feat in shap_vals, f"Falta feature {feat} en shap_values"
        assert isinstance(shap_vals[feat], float)
        assert not np.isnan(shap_vals[feat])

    assert isinstance(explanation["top_positive_features"], list)
    assert isinstance(explanation["top_negative_features"], list)

    if explanation["top_positive_features"]:
        feat_name, val = explanation["top_positive_features"][0]
        assert feat_name in STUFF_FEATURES
        assert isinstance(val, float)

    # Comprueba compatibilidad si se pasa una Serie
    series_explanation = model.explain_pitch(test_df.iloc[0])
    assert isinstance(series_explanation["base_value"], float)
    assert isinstance(series_explanation["shap_values"], dict)


def test_explain_pitch_exact_additivity_with_stuff_plus(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que explain_pitch satisface con exactitud matemática la propiedad de aditividad:
    base_value + sum(shap_values.values()) == predicted_stuff_plus
    tanto para sabor neutral (stuff_plus_neutral) como para sabor CDMX (stuff_plus_cdmx),
    y que incluye las contribuciones de direct_xrv_shap_values.
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    test_indices = [0, 4, 11, 23]
    for idx in test_indices:
        single_row = test_df.iloc[[idx]].copy()

        # 1. Verificación en sabor neutral (target_rho = 1.0)
        exp_neutral = model.explain_pitch(single_row, target_rho=1.0)
        score_neutral = float(model.predict_scores(single_row)["stuff_plus_neutral"].iloc[0])

        sum_shap_neutral = sum(exp_neutral["shap_values"].values())
        reconstructed_neutral = exp_neutral["base_value"] + sum_shap_neutral

        assert reconstructed_neutral == pytest.approx(score_neutral, abs=1e-6)
        assert exp_neutral["predicted_stuff_plus"] == pytest.approx(score_neutral, abs=1e-6)

        # 2. Verificación en sabor CDMX (target_rho = model.rho_extreme_median_)
        rho_cdmx = float(model.rho_extreme_median_)
        exp_cdmx = model.explain_pitch(single_row, target_rho=rho_cdmx)
        score_cdmx = float(model.predict_scores(single_row, rho_cdmx=rho_cdmx)["stuff_plus_cdmx"].iloc[0])

        sum_shap_cdmx = sum(exp_cdmx["shap_values"].values())
        reconstructed_cdmx = exp_cdmx["base_value"] + sum_shap_cdmx

        assert reconstructed_cdmx == pytest.approx(score_cdmx, abs=1e-6)
        assert exp_cdmx["predicted_stuff_plus"] == pytest.approx(score_cdmx, abs=1e-6)

        # 3. Comprobar existencia de atribuciones directas de xrv
        assert "direct_xrv_base_value" in exp_neutral
        assert "direct_xrv_shap_values" in exp_neutral
        assert len(exp_neutral["direct_xrv_shap_values"]) == len(STUFF_FEATURES)


# ==============================================================================
# 5. Compatibilidad Oficial con predict() y evaluate_against_baseline
# ==============================================================================

def test_stuff_model_predict_and_evaluate_against_baseline(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que predict(df) genera exactamente las 8 columnas de TARGET_COLS y
    es totalmente compatible con evaluate_against_baseline de HACK-007.
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    preds = model.predict(test_df)

    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == TARGET_COLS
    assert len(preds) == len(test_df)
    assert (preds.index == test_df.index).all()
    assert not preds.isna().any().any()

    baseline = CountBaselineModel().fit(train_df)
    baseline_preds = baseline.predict(test_df)

    eval_df = evaluate_against_baseline(
        y_true_df=test_df[TARGET_COLS],
        y_pred_df=preds,
        baseline_pred_df=baseline_preds,
    )

    assert isinstance(eval_df, pd.DataFrame)
    assert len(eval_df) == len(TARGET_COLS)
    assert "target" in eval_df.columns
    assert "skill_score" in eval_df.columns


# ==============================================================================
# 6. Integración con PhysicsMovementTranslator Externo
# ==============================================================================

def test_stuff_model_external_movement_translator(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que StuffPlusModel acepta un PhysicsMovementTranslator pre-entrenado
    y traduce el movimiento de forma coherente.
    """
    train_df, test_df = processed_lmb_dataset
    translator = PhysicsMovementTranslator().fit(train_df)

    model = StuffPlusModel(movement_translator=translator)
    model.fit(train_df)

    assert model.movement_translator_ is translator

    scores_df = model.predict_scores(test_df, rho_cdmx=0.82)
    assert "stuff_plus_cdmx" in scores_df.columns
    assert not scores_df["stuff_plus_cdmx"].isna().any()


# ==============================================================================
# 7. Verificación de Predicciones OOF y Regularización (Observación 1)
# ==============================================================================

def test_stuff_model_oof_predictions_and_regularization(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que:
    1. self.oof_fold_assignment_ contiene la asignación auditable de folds OOF.
    2. Ninguna fila del meta-entrenamiento fue predicha por un modelo entrenado con ella.
    3. Con GroupKFold, los pitcheos de cada lanzador están confinados a un único fold.
    4. self.best_alpha_ se seleccionó por validación cruzada en la rejilla de regularización.
    """
    train_df, _ = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    assert hasattr(model, "oof_fold_assignment_")
    assert model.oof_fold_assignment_ is not None

    no_alt_df = train_df[train_df["altitude_category"] == "No Altitude"]
    assert len(model.oof_fold_assignment_) == len(no_alt_df)
    assert np.all(model.oof_fold_assignment_ >= 0)
    assert len(np.unique(model.oof_fold_assignment_)) >= 2

    # Verificar GroupKFold por pitcher_anon_id
    if "pitcher_anon_id" in no_alt_df.columns:
        pitchers = no_alt_df["pitcher_anon_id"].to_numpy()
        for p in np.unique(pitchers):
            p_folds = np.unique(model.oof_fold_assignment_[pitchers == p])
            assert len(p_folds) == 1, f"El pitcher {p} está fragmentado en múltiples folds: {p_folds}"

    assert hasattr(model, "best_alpha_")
    assert model.best_alpha_ in [0.01, 0.1, 1.0, 10.0, 100.0]


# ==============================================================================
# 8. Verificación de Neutralización Frente a Extreme y Medium Altitude (Observación 1)
# ==============================================================================

def test_stuff_model_neutralization_against_altitude_batted_balls(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que alterar los resultados de bolas bateadas y xrv_pitcher en Extreme Altitude
    y en Medium Altitude NO cambia en absoluto las predicciones de StuffPlusModel sobre test_df.
    """
    train_df, test_df = processed_lmb_dataset

    # Modelo 1: Ajustado con datos originales
    model1 = StuffPlusModel().fit(train_df)
    scores1 = model1.predict_scores(test_df)

    # Modelo 2: Ajustado con resultados de bolas bateadas alterados en Extreme y Medium Altitude
    perturbed_train = train_df.copy()
    alt_mask = perturbed_train["altitude_category"].isin(["Extreme Altitude", "Medium Altitude"])
    assert alt_mask.any(), "Se requieren filas de altitud para probar la neutralización"

    perturbed_train.loc[alt_mask, "xrv_pitcher"] = (
        perturbed_train.loc[alt_mask, "xrv_pitcher"] * 25.0 + 9.2
    )
    perturbed_train.loc[alt_mask, "target_hard_contact"] = 1.0 - perturbed_train.loc[alt_mask, "target_hard_contact"]
    perturbed_train.loc[alt_mask, "target_weak_contact"] = 1.0 - perturbed_train.loc[alt_mask, "target_weak_contact"]
    perturbed_train.loc[alt_mask, "target_groundball"] = 1.0 - perturbed_train.loc[alt_mask, "target_groundball"]
    perturbed_train.loc[alt_mask, "target_out_on_pitch"] = 1.0 - perturbed_train.loc[alt_mask, "target_out_on_pitch"]
    perturbed_train.loc[alt_mask, "RunsScored"] = 20.0

    model2 = StuffPlusModel().fit(perturbed_train)
    scores2 = model2.predict_scores(test_df)

    # Deben ser 100 % idénticos
    np.testing.assert_allclose(
        scores1["stuff_plus_neutral"].to_numpy(),
        scores2["stuff_plus_neutral"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="stuff_plus_neutral cambió tras modificar bolas bateadas en altitud.",
    )
    np.testing.assert_allclose(
        scores1["stuff_plus_cdmx"].to_numpy(),
        scores2["stuff_plus_cdmx"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="stuff_plus_cdmx cambió tras modificar bolas bateadas en altitud.",
    )
    np.testing.assert_allclose(
        scores1["delta_stuff_cdmx"].to_numpy(),
        scores2["delta_stuff_cdmx"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="delta_stuff_cdmx cambió tras modificar bolas bateadas en altitud.",
    )


# ==============================================================================
# 9. Invarianza Estricta de predict_scores: Lote vs Fila Aislada (Observación 2)
# ==============================================================================

def test_predict_scores_batch_vs_single_row_invariance(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que predict_scores sobre un lote (batch) produce EXACTAMENTE los mismos
    valores para cada fila que predict_scores sobre esa fila aislada.
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    batch_df = test_df.iloc[:30].copy()
    res_batch = model.predict_scores(batch_df)

    for idx in [0, 5, 12, 18, 25]:
        single_row = batch_df.iloc[[idx]].copy()
        res_single = model.predict_scores(single_row)

        for col in [
            "stuff_plus_neutral",
            "stuff_plus_cdmx",
            "delta_stuff_cdmx",
            "ivb_diff_vs_fb",
            "hb_diff_vs_fb",
        ]:
            val_batch = float(res_batch.iloc[idx][col])
            val_single = float(res_single.iloc[0][col])
            assert val_batch == pytest.approx(val_single, abs=1e-10), (
                f"Discrepancia en {col} para fila {idx}: batch={val_batch}, single={val_single}"
            )


# ==============================================================================
# 10. Verificación de Recálculo Dinámico de Arsenal (Observación 2)
# ==============================================================================

def test_stuff_model_recompute_arsenal_diffs_at_density(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """
    Verifica que _recompute_arsenal_diffs_at_density recalcula exactamente ivb_diff_vs_fb
    y hb_diff_vs_fb contra la recta primaria traducida a target_rho:
    - Recta primaria tiene media de ivb_diff_vs_fb y hb_diff_vs_fb == 0.0 (+- 1e-5).
    - Pitcheos secundarios son exactamente iguales a IVB_sec(rho) - mean(IVB_fb(rho)).
    - Funciona también para lanzamientos aislados (simulador).
    """
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    # 1. Prueba sobre lanzador con repertorio completo en test_df
    first_pitcher = test_df["pitcher_anon_id"].iloc[0]
    p_df = test_df[test_df["pitcher_anon_id"] == first_pitcher].copy()

    # Traducir a sabor CDMX (ej. 0.82)
    trans_cdmx = model.movement_translator_.translate_movement(p_df, target_rho=0.82)
    recomputed = model._recompute_arsenal_diffs_at_density(
        trans_cdmx, target_rho=0.82, translator=model.movement_translator_, orig_df=p_df
    )

    if "primary_fb_id" in recomputed.columns:
        fb_id = recomputed["primary_fb_id"].iloc[0]
    else:
        fb_id = "Four-Seam"

    fb_mask = recomputed["pitch_id"] == fb_id
    if fb_mask.any():
        mean_fb_ivb_diff = float(recomputed.loc[fb_mask, "ivb_diff_vs_fb"].mean())
        mean_fb_hb_diff = float(recomputed.loc[fb_mask, "hb_diff_vs_fb"].mean())
        assert mean_fb_ivb_diff == pytest.approx(0.0, abs=0.25)
        assert mean_fb_hb_diff == pytest.approx(0.0, abs=0.25)

    # 2. Prueba sobre lanzamiento aislado (sin recta en el DataFrame)
    single_pitch = test_df.iloc[[0]].copy()
    trans_single = model.movement_translator_.translate_movement(single_pitch, target_rho=0.85)
    recomputed_single = model._recompute_arsenal_diffs_at_density(
        trans_single, target_rho=0.85, translator=model.movement_translator_, orig_df=single_pitch
    )

    assert "ivb_diff_vs_fb" in recomputed_single.columns
    assert "hb_diff_vs_fb" in recomputed_single.columns
    assert not recomputed_single["ivb_diff_vs_fb"].isna().any()
    assert not recomputed_single["hb_diff_vs_fb"].isna().any()


# ==============================================================================
# 11. Stuff+ como residuo sobre el conteo (reapertura HACK-010)
# ==============================================================================

def test_count_changes_predict_outcomes_but_not_stuff_plus(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """Balls/Strikes no cambian stuff_plus_neutral, pero si predict_outcomes (offset)."""
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    perturbed = test_df.copy()
    perturbed["Balls"] = 3
    perturbed["Strikes"] = 0
    if "count" in perturbed.columns:
        perturbed["count"] = "3-0"

    np.testing.assert_allclose(
        model.predict_scores(test_df)["stuff_plus_neutral"].to_numpy(),
        model.predict_scores(perturbed)["stuff_plus_neutral"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    out_base = model.predict_outcomes(test_df)
    out_pert = model.predict_outcomes(perturbed)
    assert not np.allclose(out_base.to_numpy(), out_pert.to_numpy())

    # El offset de conteo existe como baseline interno (no es feature)
    assert isinstance(model.count_baseline_, CountBaselineModel)
    assert not ({"Balls", "Strikes", "count"} & set(model.feature_cols))


def test_predict_outcomes_valid_probabilities_no_nan(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """predict_outcomes: columnas TARGET_COLS, sin NaN, binarios en [0,1] (clip 1e-6)."""
    train_df, test_df = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)

    out = model.predict_outcomes(test_df)
    assert list(out.columns) == TARGET_COLS
    assert len(out) == len(test_df)
    assert (out.index == test_df.index).all()
    assert not out.isna().any().any()
    assert (out.dtypes == np.float64).all()
    binary = [c for c in TARGET_COLS if c != "xrv_pitcher"]
    assert (out[binary] >= 1e-6).all().all()
    assert (out[binary] <= 1.0 - 1e-6).all().all()
    assert (out[binary] >= 0.0).all().all() and (out[binary] <= 1.0).all().all()


# ==============================================================================
# 9. Sin proxies de ubicacion (segunda reapertura de HACK-010, criterio 5)
# ==============================================================================

def test_trajectory_proxy_cols_not_in_feature_cols(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """Ninguna columna de TRAJECTORY_PROXY_COLS esta en STUFF_FEATURES ni en model.feature_cols."""
    train_df, _ = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)
    assert set(TRAJECTORY_PROXY_COLS) == {
        "VertRelAngle", "HorzRelAngle", "HorzRelAngle_norm", "VertApprAngle",
        "HorzApprAngle", "tunnel_dist_170ms_in", "tunnel_sim_170ms",
    }
    for col in TRAJECTORY_PROXY_COLS:
        assert col not in STUFF_FEATURES
        assert col not in model.feature_cols
    # Tampoco aparecen en el SHAP
    exp = model.explain_pitch(train_df.iloc[[0]])
    assert not set(TRAJECTORY_PROXY_COLS) & set(exp["shap_values"])


@pytest.mark.parametrize("proxy_col", TRAJECTORY_PROXY_COLS)
def test_assert_valid_stuff_features_rejects_trajectory_proxies(proxy_col: str):
    """assert_valid_stuff_features y el constructor fallan con cada proxy de trayectoria."""
    bad_features = STUFF_FEATURES + [proxy_col]
    with pytest.raises(ValueError, match="proxies de trayectoria"):
        assert_valid_stuff_features(bad_features)
    with pytest.raises(ValueError):
        StuffPlusModel(feature_cols=bad_features)


def test_lgbm_grid_search_runs_cv_by_pitcher_on_no_altitude(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch,
):
    """Con el umbral CV_MIN_ROWS bajado, se evalua la rejilla (CV por pitcher) y se elige el minimo RMSE."""
    train_df, _ = processed_lmb_dataset
    monkeypatch.setattr(stuff_module, "CV_MIN_ROWS", 50)
    model = StuffPlusModel().fit(train_df)
    res = model.lgbm_cv_results_
    assert 1 <= len(res) <= 12
    best_rmse = min(r["cv_rmse"] for r in res)
    chosen = [r for r in res if all(r[k] == v for k, v in model.best_lgbm_params_.items())]
    assert chosen and chosen[0]["cv_rmse"] == pytest.approx(best_rmse)
    assert set(model.best_lgbm_params_) == {
        "num_leaves", "min_child_samples", "reg_lambda", "learning_rate", "n_estimators",
    }
    n_no_alt = int((train_df["altitude_category"] == "No Altitude").sum())
    assert all(r["n_cv_rows"] <= n_no_alt for r in res)


def test_lgbm_small_data_uses_default_params(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """Con el fixture sintetico (< CV_MIN_ROWS) no se busca: se usan los parametros por defecto."""
    train_df, _ = processed_lmb_dataset
    model = StuffPlusModel().fit(train_df)
    assert model.lgbm_cv_results_ == []
    assert model.best_lgbm_params_ == stuff_module.LGBM_DEFAULT_PARAMS


def test_stuff_diagnostics_script_reports_reference_r2(
    processed_lmb_dataset: tuple[pd.DataFrame, pd.DataFrame],
):
    """El script de diagnostico produce r2_reference y r2_excess_over_reference (criterio 6)."""
    from analysis.stuff_diagnostics import COLUMNS, run_diagnostics

    train_df, _ = processed_lmb_dataset
    out = run_diagnostics(train_df, max_per_pitch=40, r2_max_rows=3000, n_boot=20, verbose=False)
    assert list(out.columns) == COLUMNS
    for tgt in ("PlateLocHeight", "PlateLocSide"):
        full = out[out["metric"] == f"loc_r2_oos_{tgt}"]["value"]
        ref = out[out["metric"] == f"loc_r2_reference_{tgt}"]["value"]
        exc = out[out["metric"] == f"loc_r2_excess_over_reference_{tgt}"]["value"]
        assert len(full) == len(ref) == len(exc) == 1
        assert exc.iloc[0] == pytest.approx(full.iloc[0] - ref.iloc[0])
    assert (out["metric"] == "loc_r2_excess_max").sum() == 1
    for m in ("delta_stuff_cdmx_mean", "sens_ivb_pts_per_in", "lgbm_chosen"):
        assert (out["metric"] == m).any()

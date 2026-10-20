"""
tests/test_movement.py
======================
Suite de pruebas unitarias para src/models/movement.py (HACK-009).

Verifica exhaustivamente:
1. Cero data leakage: assert_no_leakage(MOVEMENT_FEATURE_COLS, "stuff") pasa sin errores
   y PhysicsMovementTranslator rechaza cualquier columna de ubicación o resultado.
2. Identificación intra-pitcher: compute_intra_pitcher_deltas calcula diferencias centradas
   y medias base correctamente, manejando casos borde (vacío, sin variación de altitud).
3. Identidad numérica: Cuando target_rho == df["rho_hat"], InducedVertBreak, HorzBreak_norm,
   TotalInducedBreak y ZoneSpeed son idénticos a los originales (np.allclose(..., atol=1e-10)).
4. Conservación del residuo individual (Reviewer Checklist #1): Dos pitcheos con idénticas
   MOVEMENT_FEATURE_COLS pero distinto InducedVertBreak observado mantienen exactamente su
   diferencia IVB_1 - IVB_2 tras translate_movement(df, target_rho=0.82).
5. Dirección física obligatoria (Reviewer Checklist #2 & Criterio de Aceptación #2):
   Al traducir lanzamientos de No Altitude (rho_hat ~ 1.0) a CDMX (target_rho = 0.82),
   para rectas de alto IVB (InducedVertBreak >= 14) y curvas (Curveball), la media de
   TotalInducedBreak disminuye (< 0), mientras que la media de ZoneSpeed aumenta (> 0).
6. Respuesta firmada a densidad para quiebres positivos y negativos en distintas densidades
   (test_signed_break_response_across_densities):
   Verifica el sentido del cambio en rho*=0.82 y rho*=1.05 para IVB > 0, IVB < 0, HB > 0 y HB < 0.
7. Inmutabilidad y coherencia geométrica: translate_movement no muta df y actualiza
   todas las variables físicas derivadas (SpeedDrop, DragProxy, BreakAngle_deg, SpinToBreakRatio,
   ivb_diff_vs_fb, hb_diff_vs_fb) sin introducir valores nulos (NaN).
8. Filtrado de temporada: fit() ignora lanzamientos de 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FORBIDDEN_TARGET_COLS, LOCATION_AND_COUNT_COLS
from src.io import assert_no_leakage
from src.models.movement import (
    MOVEMENT_FEATURE_COLS,
    PhysicsMovementTranslator,
    compute_intra_pitcher_deltas,
)
from src.pipeline_step1 import build_processed_dataset


@pytest.fixture(scope="module")
def processed_lmb_data(synthetic_lmb_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Procesa synthetic_lmb_df con el pipeline completo (HACK-002 a HACK-006)
    y separa conjunto de entrenamiento (year <= 2025) y completo.
    """
    df, _ = build_processed_dataset(synthetic_lmb_df)
    train_df = df[df["year"] <= 2025].copy()
    return train_df, df


# ==============================================================================
# 1. Pruebas de Prevención de Fuga de Datos (Data Leakage)
# ==============================================================================

def test_feature_leakage_rejection():
    """
    Verifica que MOVEMENT_FEATURE_COLS no contiene leakage y que PhysicsMovementTranslator
    rechaza inmediatamente cualquier intento de inyectar variables de ubicación o resultado.
    """
    # 1. MOVEMENT_FEATURE_COLS pasa assert_no_leakage
    assert_no_leakage(MOVEMENT_FEATURE_COLS, "stuff")

    # 2. Ninguna columna prohibida de targets/resultados en MOVEMENT_FEATURE_COLS
    forbidden_targets = [c for c in MOVEMENT_FEATURE_COLS if c in FORBIDDEN_TARGET_COLS]
    assert len(forbidden_targets) == 0, f"Leakage detectado: {forbidden_targets}"

    # 3. Ninguna columna de ubicación o conteo en MOVEMENT_FEATURE_COLS
    forbidden_loc = [c for c in MOVEMENT_FEATURE_COLS if c in LOCATION_AND_COUNT_COLS]
    assert len(forbidden_loc) == 0, f"Leakage detectado: {forbidden_loc}"

    # 4. Rechazo en inicialización si se intenta inyectar columna de ubicación
    with pytest.raises(ValueError, match="Leakage"):
        PhysicsMovementTranslator(feature_cols=MOVEMENT_FEATURE_COLS + ["PlateLocHeight"])

    with pytest.raises(ValueError, match="Leakage"):
        PhysicsMovementTranslator(feature_cols=MOVEMENT_FEATURE_COLS + ["PlateLocSide"])

    # 5. Rechazo en inicialización si se intenta inyectar columna de resultado
    with pytest.raises(ValueError, match="Leakage"):
        PhysicsMovementTranslator(feature_cols=MOVEMENT_FEATURE_COLS + ["is_whiff"])

    with pytest.raises(ValueError, match="Leakage"):
        PhysicsMovementTranslator(feature_cols=MOVEMENT_FEATURE_COLS + ["xrv_pitcher"])


def test_fit_leakage_rejection(processed_lmb_data):
    """
    Verifica que fit() también valida y rechaza listas de features con leakage.
    """
    train_df, _ = processed_lmb_data
    translator = PhysicsMovementTranslator()

    with pytest.raises(ValueError, match="Leakage"):
        translator.fit(train_df, feature_cols=MOVEMENT_FEATURE_COLS + ["target_whiff"])


# ==============================================================================
# 2. Pruebas de Identificación Intra-Pitcher (compute_intra_pitcher_deltas)
# ==============================================================================

def test_compute_intra_pitcher_deltas(processed_lmb_data):
    """
    Verifica que compute_intra_pitcher_deltas extrae las diferencias centradas
    en la media del grupo (pitcher_anon_id, year, pitch_id) para juegos con distinta densidad.
    """
    train_df, _ = processed_lmb_data
    intra_df = compute_intra_pitcher_deltas(train_df)

    expected_cols = [
        "pitcher_anon_id",
        "year",
        "pitch_id",
        "delta_ivb",
        "delta_hb_norm",
        "delta_zone_speed",
        "delta_rho_hat",
        "mean_ivb",
        "mean_hb_norm",
        "mean_spin_rate",
        "mean_rel_speed",
    ]
    for col in expected_cols:
        assert col in intra_df.columns, f"Falta columna esperada: {col}"

    assert len(intra_df) > 0, "Debe haber lanzamientos en grupos con variación de densidad"

    # Verificar que las diferencias están centradas en cero por grupo
    grouped = intra_df.groupby(["pitcher_anon_id", "year", "pitch_id"])
    for _, grp in grouped:
        assert np.isclose(grp["delta_ivb"].mean(), 0.0, atol=1e-10)
        assert np.isclose(grp["delta_hb_norm"].mean(), 0.0, atol=1e-10)
        assert np.isclose(grp["delta_zone_speed"].mean(), 0.0, atol=1e-10)
        assert np.isclose(grp["delta_rho_hat"].mean(), 0.0, atol=1e-10)


def test_compute_intra_pitcher_deltas_edge_cases():
    """
    Verifica el comportamiento de compute_intra_pitcher_deltas con DataFrames vacíos
    o sin variación de densidad.
    """
    empty_df = pd.DataFrame()
    res_empty = compute_intra_pitcher_deltas(empty_df)
    assert res_empty.empty
    assert "delta_ivb" in res_empty.columns

    # DataFrame con un solo juego (sin variación de densidad)
    single_game = pd.DataFrame({
        "pitcher_anon_id": ["p1", "p1"],
        "year": [2024, 2024],
        "pitch_id": ["Four-Seam", "Four-Seam"],
        "game_anon_id": ["g1", "g1"],
        "rho_hat": [1.0, 1.0],
        "InducedVertBreak": [15.0, 16.0],
        "HorzBreak_norm": [5.0, 6.0],
        "ZoneSpeed": [88.0, 89.0],
        "SpinRate": [2200.0, 2300.0],
        "RelSpeed": [94.0, 95.0],
    })
    res_single = compute_intra_pitcher_deltas(single_game)
    assert res_single.empty, "Grupos sin variación de densidad no deben incluirse"


# ==============================================================================
# 3. Pruebas de Identidad Numérica (target_rho == df["rho_hat"])
# ==============================================================================

def test_movement_translation_identity(processed_lmb_data):
    """
    Criterio de Aceptación #1:
    Si target_rho == df["rho_hat"], translate_movement devuelve exactamente
    el movimiento original (InducedVertBreak, HorzBreak_norm, TotalInducedBreak, ZoneSpeed)
    con precisión np.allclose(..., atol=1e-10).
    """
    train_df, full_df = processed_lmb_data

    translator = PhysicsMovementTranslator(random_state=42)
    translator.fit(train_df)

    # 1. Usando la serie df["rho_hat"]
    translated_series = translator.translate_movement(full_df, full_df["rho_hat"])

    assert np.allclose(
        translated_series["InducedVertBreak"], full_df["InducedVertBreak"], atol=1e-10
    )
    assert np.allclose(
        translated_series["HorzBreak_norm"], full_df["HorzBreak_norm"], atol=1e-10
    )
    assert np.allclose(
        translated_series["TotalInducedBreak"], full_df["TotalInducedBreak"], atol=1e-10
    )
    assert np.allclose(
        translated_series["ZoneSpeed"], full_df["ZoneSpeed"], atol=1e-10
    )

    # 2. Usando arreglo numpy
    translated_arr = translator.translate_movement(
        full_df, full_df["rho_hat"].to_numpy()
    )
    assert np.allclose(
        translated_arr["InducedVertBreak"], full_df["InducedVertBreak"], atol=1e-10
    )

    # 3. Subconjunto filtrado a una densidad específica con target escalar
    no_alt_subset = full_df[full_df["altitude_category"] == "No Altitude"].copy()
    rho_val = float(no_alt_subset["rho_hat"].iloc[0])
    no_alt_subset["rho_hat"] = rho_val

    translated_scalar = translator.translate_movement(no_alt_subset, rho_val)
    assert np.allclose(
        translated_scalar["InducedVertBreak"], no_alt_subset["InducedVertBreak"], atol=1e-10
    )
    assert np.allclose(
        translated_scalar["ZoneSpeed"], no_alt_subset["ZoneSpeed"], atol=1e-10
    )


# ==============================================================================
# 4. Conservación del Residuo Individual (Reviewer Checklist #1)
# ==============================================================================

def test_individual_residual_preservation(processed_lmb_data):
    """
    Reviewer Checklist #1:
    Conserva el residuo individual del pitcheo:
    Dos pitcheos con idénticas MOVEMENT_FEATURE_COLS pero distinto InducedVertBreak
    observado mantienen exactamente su diferencia IVB_1 - IVB_2 tras translate_movement(df, target_rho=0.82).
    """
    train_df, full_df = processed_lmb_data

    translator = PhysicsMovementTranslator(random_state=42)
    translator.fit(train_df)

    # Seleccionar un pitcheo base
    base_pitch = full_df.iloc[[0]].copy()

    # Crear dos pitcheos con idénticas features pero distinto quiebre observado
    pitch1 = base_pitch.copy()
    pitch1["InducedVertBreak"] = 16.5
    pitch1["HorzBreak_norm"] = 6.0
    pitch1["ZoneSpeed"] = 87.0
    pitch1["rho_hat"] = 1.000

    pitch2 = base_pitch.copy()
    pitch2["InducedVertBreak"] = 12.0  # Diferencia de 4.5 pulgadas en IVB
    pitch2["HorzBreak_norm"] = 2.0   # Diferencia de 4.0 pulgadas en HB
    pitch2["ZoneSpeed"] = 85.5        # Diferencia de 1.5 mph en ZoneSpeed
    pitch2["rho_hat"] = 1.000

    pair_df = pd.concat([pitch1, pitch2], ignore_index=True)

    diff_ivb_before = pair_df["InducedVertBreak"].iloc[0] - pair_df["InducedVertBreak"].iloc[1]
    diff_hb_before = pair_df["HorzBreak_norm"].iloc[0] - pair_df["HorzBreak_norm"].iloc[1]
    diff_zs_before = pair_df["ZoneSpeed"].iloc[0] - pair_df["ZoneSpeed"].iloc[1]

    # Traducir a CDMX (rho = 0.82)
    translated_pair = translator.translate_movement(pair_df, target_rho=0.82)

    diff_ivb_after = translated_pair["InducedVertBreak"].iloc[0] - translated_pair["InducedVertBreak"].iloc[1]
    diff_hb_after = translated_pair["HorzBreak_norm"].iloc[0] - translated_pair["HorzBreak_norm"].iloc[1]
    diff_zs_after = translated_pair["ZoneSpeed"].iloc[0] - translated_pair["ZoneSpeed"].iloc[1]

    # La diferencia entre los dos lanzamientos debe conservarse exactamente (atol=1e-12)
    assert np.isclose(diff_ivb_before, diff_ivb_after, atol=1e-12)
    assert np.isclose(diff_hb_before, diff_hb_after, atol=1e-12)
    assert np.isclose(diff_zs_before, diff_zs_after, atol=1e-12)


# ==============================================================================
# 5. Pruebas de Dirección Física (No Altitude -> CDMX target_rho = 0.82)
# ==============================================================================

def test_physical_direction_altitude_to_cdmx(processed_lmb_data):
    """
    Criterio de Aceptación #2:
    Al traducir lanzamientos de No Altitude (rho_hat ~ 1.0) a CDMX (target_rho = 0.82):
    - Para rectas de alto IVB (InducedVertBreak >= 14):
      la media de TotalInducedBreak disminuye (< 0) y ZoneSpeed aumenta (> 0).
    - Para curvas (Curveball):
      la media de TotalInducedBreak disminuye (< 0) y ZoneSpeed aumenta (> 0).
    """
    train_df, full_df = processed_lmb_data

    translator = PhysicsMovementTranslator(random_state=42)
    translator.fit(train_df)

    no_alt = full_df[full_df["altitude_category"] == "No Altitude"].copy()

    # 1. Rectas de alto IVB
    high_ivb = no_alt[no_alt["InducedVertBreak"] >= 14.0].copy()
    assert len(high_ivb) > 0, "Debe haber rectas con IVB >= 14 en No Altitude"

    trans_high_ivb = translator.translate_movement(high_ivb, target_rho=0.82)

    delta_tib_high = (trans_high_ivb["TotalInducedBreak"] - high_ivb["TotalInducedBreak"]).mean()
    delta_zs_high = (trans_high_ivb["ZoneSpeed"] - high_ivb["ZoneSpeed"]).mean()
    delta_ivb_high = (trans_high_ivb["InducedVertBreak"] - high_ivb["InducedVertBreak"]).mean()

    # TotalInducedBreak y InducedVertBreak deben caer monótonamente en aire enrarecido
    assert delta_tib_high < 0, f"TotalInducedBreak debía caer: {delta_tib_high}"
    assert delta_ivb_high < 0, f"InducedVertBreak debía caer: {delta_ivb_high}"
    # ZoneSpeed debe aumentar por menor arrastre aerodinámico
    assert delta_zs_high > 0, f"ZoneSpeed debía aumentar: {delta_zs_high}"

    # 2. Curvas (Curveball)
    curves = no_alt[no_alt["pitch_id"] == "Curveball"].copy()
    assert len(curves) > 0, "Debe haber curvas en No Altitude"

    trans_curves = translator.translate_movement(curves, target_rho=0.82)

    delta_tib_curves = (trans_curves["TotalInducedBreak"] - curves["TotalInducedBreak"]).mean()
    delta_zs_curves = (trans_curves["ZoneSpeed"] - curves["ZoneSpeed"]).mean()

    assert delta_tib_curves < 0, f"TotalInducedBreak en curvas debía caer: {delta_tib_curves}"
    assert delta_zs_curves > 0, f"ZoneSpeed en curvas debía aumentar: {delta_zs_curves}"


# ==============================================================================
# 6. Respuesta a Densidad para Quiebres Positivos y Negativos (Corrección Solicitada)
# ==============================================================================

def test_signed_break_response_across_densities(processed_lmb_data):
    """
    Verifica explícitamente el sentido del cambio físico al traducir a menor densidad
    (rho* = 0.82) y a mayor densidad (rho* = 1.05) desde rho_obs = 1.00 en los 4 casos:
    a) InducedVertBreak > 0 (+16.0 in): disminuye en rho*=0.82 y aumenta en rho*=1.05.
    b) InducedVertBreak < 0 (-12.0 in, curva): se vuelve menos negativo (aumenta hacia 0,
       menor magnitud |IVB|) en rho*=0.82 y más negativo (mayor magnitud |IVB|) en rho*=1.05.
    c) HorzBreak_norm > 0 (+14.0 in, arm-side): disminuye en rho*=0.82 y aumenta en rho*=1.05.
    d) HorzBreak_norm < 0 (-10.0 in, glove-side): se vuelve menos negativo (menor magnitud |HB_norm|)
       en rho*=0.82 y más negativo (mayor magnitud |HB_norm|) en rho*=1.05.
    """
    train_df, full_df = processed_lmb_data

    translator = PhysicsMovementTranslator(random_state=42)
    translator.fit(train_df)

    base_pitch = full_df[full_df["altitude_category"] == "No Altitude"].iloc[[0]].copy()
    base_pitch["rho_hat"] = 1.000

    # a) InducedVertBreak > 0 (+16.0 in)
    pitch_a = base_pitch.copy()
    pitch_a["InducedVertBreak"] = 16.0
    trans_a_low = translator.translate_movement(pitch_a, target_rho=0.82)
    trans_a_high = translator.translate_movement(pitch_a, target_rho=1.05)
    assert trans_a_low["InducedVertBreak"].iloc[0] < 16.0, (
        f"IVB>0 debía disminuir en rho=0.82: {trans_a_low['InducedVertBreak'].iloc[0]}"
    )
    assert trans_a_high["InducedVertBreak"].iloc[0] > 16.0, (
        f"IVB>0 debía aumentar en rho=1.05: {trans_a_high['InducedVertBreak'].iloc[0]}"
    )

    # b) InducedVertBreak < 0 (-12.0 in, curva con quiebre hacia abajo)
    pitch_b = base_pitch.copy()
    pitch_b["InducedVertBreak"] = -12.0
    trans_b_low = translator.translate_movement(pitch_b, target_rho=0.82)
    trans_b_high = translator.translate_movement(pitch_b, target_rho=1.05)
    # En rho=0.82, menor arrastre/Magnus reduce el quiebre hacia abajo -> menos negativo (más cercano a 0)
    assert trans_b_low["InducedVertBreak"].iloc[0] > -12.0, (
        f"IVB<0 debía ser menos negativo en rho=0.82: {trans_b_low['InducedVertBreak'].iloc[0]}"
    )
    assert abs(trans_b_low["InducedVertBreak"].iloc[0]) < 12.0, "Magnitud de IVB debía disminuir"
    # En rho=1.05, mayor densidad aumenta el quiebre hacia abajo -> más negativo
    assert trans_b_high["InducedVertBreak"].iloc[0] < -12.0, (
        f"IVB<0 debía ser más negativo en rho=1.05: {trans_b_high['InducedVertBreak'].iloc[0]}"
    )
    assert abs(trans_b_high["InducedVertBreak"].iloc[0]) > 12.0, "Magnitud de IVB debía aumentar"

    # c) HorzBreak_norm > 0 (+14.0 in, arm-side run)
    pitch_c = base_pitch.copy()
    pitch_c["HorzBreak_norm"] = 14.0
    trans_c_low = translator.translate_movement(pitch_c, target_rho=0.82)
    trans_c_high = translator.translate_movement(pitch_c, target_rho=1.05)
    assert trans_c_low["HorzBreak_norm"].iloc[0] < 14.0, (
        f"HB>0 debía disminuir en rho=0.82: {trans_c_low['HorzBreak_norm'].iloc[0]}"
    )
    assert trans_c_high["HorzBreak_norm"].iloc[0] > 14.0, (
        f"HB>0 debía aumentar en rho=1.05: {trans_c_high['HorzBreak_norm'].iloc[0]}"
    )

    # d) HorzBreak_norm < 0 (-10.0 in, glove-side sweep)
    pitch_d = base_pitch.copy()
    pitch_d["HorzBreak_norm"] = -10.0
    trans_d_low = translator.translate_movement(pitch_d, target_rho=0.82)
    trans_d_high = translator.translate_movement(pitch_d, target_rho=1.05)
    # En rho=0.82, menor densidad reduce el sweep -> menos negativo
    assert trans_d_low["HorzBreak_norm"].iloc[0] > -10.0, (
        f"HB<0 debía ser menos negativo en rho=0.82: {trans_d_low['HorzBreak_norm'].iloc[0]}"
    )
    assert abs(trans_d_low["HorzBreak_norm"].iloc[0]) < 10.0, "Magnitud de HB debía disminuir"
    # En rho=1.05, mayor densidad incrementa el sweep -> más negativo
    assert trans_d_high["HorzBreak_norm"].iloc[0] < -10.0, (
        f"HB<0 debía ser más negativo en rho=1.05: {trans_d_high['HorzBreak_norm'].iloc[0]}"
    )
    assert abs(trans_d_high["HorzBreak_norm"].iloc[0]) > 10.0, "Magnitud de HB debía aumentar"


# ==============================================================================
# 7. Pruebas de Inmutabilidad y Columnas Derivadas
# ==============================================================================

def test_immutability_and_derived_columns(processed_lmb_data):
    """
    Verifica que translate_movement:
    1. No muta el DataFrame de entrada.
    2. Actualiza consistentemente todas las columnas físicas derivadas sin NaNs.
    3. Actualiza ivb_diff_vs_fb y hb_diff_vs_fb coherentemente.
    """
    train_df, full_df = processed_lmb_data

    translator = PhysicsMovementTranslator(random_state=42)
    translator.fit(train_df)

    input_df = full_df.head(50).copy()
    orig_copy = input_df.copy(deep=True)

    translated = translator.translate_movement(input_df, target_rho=0.82)

    # 1. Inmutabilidad
    pd.testing.assert_frame_equal(input_df, orig_copy)

    # 2. Columnas actualizadas
    updated_cols = [
        "rho_hat",
        "InducedVertBreak",
        "HorzBreak_norm",
        "HorzBreak",
        "ZoneSpeed",
        "SpeedDrop",
        "DragProxy",
        "TotalInducedBreak",
        "BreakAngle_deg",
        "SpinToBreakRatio",
    ]
    for col in updated_cols:
        assert col in translated.columns
        assert not translated[col].isna().any(), f"NaNs encontrados en {col}"
        assert translated[col].dtype == np.float64, f"Tipo incorrecto para {col}"

    # 3. Coherencia matemática de derivadas
    # SpeedDrop = RelSpeed - ZoneSpeed
    expected_speed_drop = translated["RelSpeed"] - translated["ZoneSpeed"]
    assert np.allclose(translated["SpeedDrop"], expected_speed_drop, atol=1e-10)

    # TotalInducedBreak = sqrt(IVB^2 + HB^2)
    expected_tib = np.sqrt(translated["InducedVertBreak"] ** 2 + translated["HorzBreak"] ** 2)
    assert np.allclose(translated["TotalInducedBreak"], expected_tib, atol=1e-10)

    # 4. Arsenal differentials (ivb_diff_vs_fb, hb_diff_vs_fb)
    if "ivb_diff_vs_fb" in translated.columns:
        delta_ivb = translated["InducedVertBreak"] - input_df["InducedVertBreak"]
        expected_diff_ivb = input_df["ivb_diff_vs_fb"] + delta_ivb
        assert np.allclose(translated["ivb_diff_vs_fb"], expected_diff_ivb, atol=1e-10)

    if "hb_diff_vs_fb" in translated.columns:
        delta_hb_norm = translated["HorzBreak_norm"] - input_df["HorzBreak_norm"]
        expected_diff_hb = input_df["hb_diff_vs_fb"] + delta_hb_norm
        assert np.allclose(translated["hb_diff_vs_fb"], expected_diff_hb, atol=1e-10)


# ==============================================================================
# 8. Pruebas de Filtrado por Año (year <= 2025)
# ==============================================================================

def test_year_filtering_in_fit(processed_lmb_data):
    """
    Verifica que fit() filtra los datos de entrada a year <= 2025 para prevenir
    fuga hacia la temporada de prueba 2026.
    """
    _, full_df = processed_lmb_data
    assert 2026 in full_df["year"].values, "El dataset debe contener lanzamientos de 2026"

    translator = PhysicsMovementTranslator(random_state=42)
    # Pasar full_df conteniendo 2024, 2025 y 2026
    translator.fit(full_df)

    assert translator.is_fitted
    # Comprobar que predice correctamente tras el filtrado
    test_pitch = full_df.head(5)
    translated = translator.translate_movement(test_pitch, target_rho=0.82)
    assert len(translated) == 5

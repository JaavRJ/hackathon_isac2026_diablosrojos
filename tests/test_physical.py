"""
tests/test_physical.py
======================
Suite de pruebas unitarias exhaustivas para src/features/physical.py (HACK-002).
Valida:
- Cero lectura de datos reales (utiliza synthetic_lmb_df).
- Inmutabilidad del DataFrame de entrada.
- Conversión e imputación de EffectiveVelo.
- Recodificación de AutoPitchType (OneSeam/TwoSeam -> Sinker, Knuckleball -> Other).
- Imputación inteligente de altitude_category por game_anon_id sin data leakage.
- Imputación de PitcherThrows por pitcher_anon_id y fallback físico por RelSide.
- Imputación de BatterSide con 'Right' y cálculo de is_same_hand.
- Filtrado estricto de anomalías de radar (velocidad y rotación).
- Inversión de signo horizontal para lanzadores zurdos (HorzBreak_norm, etc.).
- Presencia, tipo float64 y ausencia de valores infinitos en columnas de CONTRACT.md.
- Variables geométricas de zona (dist_from_zone_center, is_shadow_zone, zone_quadrant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.physical import clean_and_impute, add_physical_features


class TestCleanAndImpute:
    """Pruebas para clean_and_impute."""

    def test_clean_and_impute_does_not_mutate_input(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """Verifica que clean_and_impute trabaje sobre una copia sin mutar el DataFrame original."""
        original_copy = synthetic_lmb_df.copy(deep=True)
        cleaned = clean_and_impute(synthetic_lmb_df)

        assert cleaned is not synthetic_lmb_df
        # Comprobar que synthetic_lmb_df no cambió
        pd.testing.assert_frame_equal(synthetic_lmb_df, original_copy)

    def test_effective_velo_conversion_and_imputation(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """Verifica que EffectiveVelo sea float64 y que los nulos/'None' se reemplacen con RelSpeed."""
        df_sample = synthetic_lmb_df.head(20).copy()
        df_sample.loc[0, "EffectiveVelo"] = "None"
        df_sample.loc[1, "EffectiveVelo"] = np.nan
        df_sample.loc[2, "EffectiveVelo"] = "94.321"

        cleaned = clean_and_impute(df_sample)

        assert cleaned["EffectiveVelo"].dtype == np.float64
        assert not cleaned["EffectiveVelo"].isna().any()
        assert cleaned.loc[0, "EffectiveVelo"] == cleaned.loc[0, "RelSpeed"]
        assert cleaned.loc[1, "EffectiveVelo"] == cleaned.loc[1, "RelSpeed"]
        assert pytest.approx(cleaned.loc[2, "EffectiveVelo"], rel=1e-3) == 94.321

    def test_autopitchtype_recoding(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """Verifica la recodificación de OneSeamFastBall/TwoSeamFastBall -> Sinker y Knuckleball -> Other."""
        df_sample = synthetic_lmb_df.head(10).copy()
        df_sample["AutoPitchType"] = df_sample["AutoPitchType"].astype(object)
        df_sample.loc[0, "AutoPitchType"] = "OneSeamFastBall"
        df_sample.loc[1, "AutoPitchType"] = "TwoSeamFastBall"
        df_sample.loc[2, "AutoPitchType"] = "Knuckleball"
        df_sample.loc[3, "AutoPitchType"] = "Four-Seam"

        cleaned = clean_and_impute(df_sample)

        assert cleaned.loc[0, "AutoPitchType"] == "Sinker"
        assert cleaned.loc[1, "AutoPitchType"] == "Sinker"
        assert cleaned.loc[2, "AutoPitchType"] == "Other"
        assert cleaned.loc[3, "AutoPitchType"] == "Four-Seam"

    def test_altitude_category_imputation_by_game(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """Verifica la imputación por moda dentro del mismo game_anon_id y fallback a 'No Altitude'."""
        df_sample = synthetic_lmb_df.head(30).copy()
        # Modificar juego game_0001 para que tenga filas nulas/'None'/'Undefined'
        g1_mask = df_sample["game_anon_id"] == "game_0001"
        df_sample.loc[g1_mask, "altitude_category"] = "Extreme Altitude"
        g1_indices = df_sample[g1_mask].index
        df_sample.loc[g1_indices[0], "altitude_category"] = None
        df_sample.loc[g1_indices[1], "altitude_category"] = "None"
        df_sample.loc[g1_indices[2], "altitude_category"] = "Undefined"

        # Juego sin datos válidos
        df_sample.loc[df_sample["game_anon_id"] == "game_0002", "altitude_category"] = None

        cleaned = clean_and_impute(df_sample)

        # Cero nulos en altitude_category
        assert not cleaned["altitude_category"].isna().any()
        assert "None" not in cleaned["altitude_category"].values
        assert "Undefined" not in cleaned["altitude_category"].values

        # Filas de game_0001 deben recibir Extreme Altitude
        assert cleaned.loc[g1_indices[0], "altitude_category"] == "Extreme Altitude"
        assert cleaned.loc[g1_indices[1], "altitude_category"] == "Extreme Altitude"
        assert cleaned.loc[g1_indices[2], "altitude_category"] == "Extreme Altitude"

        # Filas de game_0002 deben recibir fallback 'No Altitude'
        g2_cleaned = cleaned[cleaned["game_anon_id"] == "game_0002"]
        assert (g2_cleaned["altitude_category"] == "No Altitude").all()

    def test_pitcher_throws_imputation_mode_and_relside_fallback(
        self, synthetic_lmb_df: pd.DataFrame
    ) -> None:
        """Verifica la imputación de PitcherThrows por moda de pitcher y fallback por RelSide."""
        df_sample = synthetic_lmb_df.head(20).copy()
        p1 = df_sample.loc[0, "pitcher_anon_id"]
        # p1 tiene la mayoría como 'Left'
        p1_mask = df_sample["pitcher_anon_id"] == p1
        df_sample.loc[p1_mask, "PitcherThrows"] = "Left"
        first_p1_idx = df_sample[p1_mask].index[0]
        df_sample.loc[first_p1_idx, "PitcherThrows"] = None

        # Pitcher nuevo completamente nulo: fallback según RelSide
        df_sample.loc[17:19, "pitcher_anon_id"] = "pitcher_test_new"
        p_new_mask = df_sample["pitcher_anon_id"] == "pitcher_test_new"
        df_sample.loc[p_new_mask, "PitcherThrows"] = np.nan
        df_sample.loc[p_new_mask, "RelSide"] = 1.8  # positivo -> Right

        cleaned = clean_and_impute(df_sample)

        assert not cleaned["PitcherThrows"].isna().any()
        assert cleaned.loc[first_p1_idx, "PitcherThrows"] == "Left"

        p_new_cleaned = cleaned.loc[cleaned["pitcher_anon_id"] == "pitcher_test_new"]
        assert (p_new_cleaned["PitcherThrows"] == "Right").all()

    def test_pitcher_throws_negative_relside_fallback_to_left(
        self, synthetic_lmb_df: pd.DataFrame
    ) -> None:
        """Verifica que el fallback cuando RelSide < 0 asigne 'Left'."""
        df_sample = synthetic_lmb_df.head(5).copy()
        df_sample["PitcherThrows"] = np.nan
        df_sample["RelSide"] = -1.5

        cleaned = clean_and_impute(df_sample)
        assert (cleaned["PitcherThrows"] == "Left").all()

    def test_batter_side_imputation_and_is_same_hand(
        self, synthetic_lmb_df: pd.DataFrame
    ) -> None:
        """Verifica imputación de BatterSide con 'Right' y creación de is_same_hand."""
        df_sample = synthetic_lmb_df.head(10).copy()
        df_sample.loc[0, "BatterSide"] = "None"
        df_sample.loc[1, "BatterSide"] = np.nan
        df_sample.loc[2, "BatterSide"] = "Undefined"
        df_sample.loc[3, "BatterSide"] = "Left"

        cleaned = clean_and_impute(df_sample)

        assert not cleaned["BatterSide"].isna().any()
        assert cleaned.loc[0, "BatterSide"] == "Right"
        assert cleaned.loc[1, "BatterSide"] == "Right"
        assert cleaned.loc[2, "BatterSide"] == "Right"
        assert cleaned.loc[3, "BatterSide"] == "Left"

        # Validar is_same_hand
        expected_same_hand = (cleaned["PitcherThrows"] == cleaned["BatterSide"]).astype(int)
        assert (cleaned["is_same_hand"] == expected_same_hand).all()
        assert set(cleaned["is_same_hand"].unique()).issubset({0, 1})

    def test_radar_anomaly_filtering(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """Verifica que lanzamientos fuera de [65, 105] mph o [500, 3600] rpm sean descartados."""
        df_sample = synthetic_lmb_df.head(10).copy()
        # Modificar filas con anomalías
        df_sample.loc[0, "RelSpeed"] = 64.9  # Descartar
        df_sample.loc[1, "RelSpeed"] = 65.0  # Conservar
        df_sample.loc[2, "RelSpeed"] = 105.0  # Conservar
        df_sample.loc[3, "RelSpeed"] = 105.1  # Descartar
        df_sample.loc[4, "SpinRate"] = 499.0  # Descartar
        df_sample.loc[5, "SpinRate"] = 500.0  # Conservar
        df_sample.loc[6, "SpinRate"] = 3600.0  # Conservar
        df_sample.loc[7, "SpinRate"] = 3601.0  # Descartar
        # Normales
        df_sample.loc[8, "RelSpeed"] = 92.0
        df_sample.loc[8, "SpinRate"] = 2200.0
        df_sample.loc[9, "RelSpeed"] = 94.0
        df_sample.loc[9, "SpinRate"] = 2300.0

        cleaned = clean_and_impute(df_sample)

        assert len(cleaned) == 6
        assert (cleaned["RelSpeed"] >= 65.0).all()
        assert (cleaned["RelSpeed"] <= 105.0).all()
        assert (cleaned["SpinRate"] >= 500.0).all()
        assert (cleaned["SpinRate"] <= 3600.0).all()


class TestAddPhysicalFeatures:
    """Pruebas para add_physical_features."""

    def test_left_handed_sign_inversion(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """
        Criterio de Aceptación 2:
        Para un pitcher zurdo (PitcherThrows == 'Left'), HorzBreak_norm == -HorzBreak, etc.
        Para un pitcher diestro (PitcherThrows == 'Right'), HorzBreak_norm == HorzBreak.
        """
        cleaned = clean_and_impute(synthetic_lmb_df)
        featured = add_physical_features(cleaned)

        lefties = featured[featured["PitcherThrows"] == "Left"]
        righties = featured[featured["PitcherThrows"] == "Right"]

        assert len(lefties) > 0, "Debe haber lanzadores zurdos en synthetic_lmb_df"
        assert len(righties) > 0, "Debe haber lanzadores diestros en synthetic_lmb_df"

        # Verificación para zurdos (signo invertido)
        np.testing.assert_allclose(lefties["HorzBreak_norm"], -lefties["HorzBreak"], rtol=1e-5)
        np.testing.assert_allclose(lefties["RelSide_norm"], -lefties["RelSide"], rtol=1e-5)
        np.testing.assert_allclose(lefties["HorzRelAngle_norm"], -lefties["HorzRelAngle"], rtol=1e-5)
        np.testing.assert_allclose(lefties["HorzApprAngle_norm"], -lefties["HorzApprAngle"], rtol=1e-5)
        np.testing.assert_allclose(lefties["PlateLocSide_norm"], -lefties["PlateLocSide"], rtol=1e-5)

        # Verificación para diestros (signo positivo)
        np.testing.assert_allclose(righties["HorzBreak_norm"], righties["HorzBreak"], rtol=1e-5)
        np.testing.assert_allclose(righties["RelSide_norm"], righties["RelSide"], rtol=1e-5)
        np.testing.assert_allclose(righties["HorzRelAngle_norm"], righties["HorzRelAngle"], rtol=1e-5)
        np.testing.assert_allclose(righties["HorzApprAngle_norm"], righties["HorzApprAngle"], rtol=1e-5)
        np.testing.assert_allclose(righties["PlateLocSide_norm"], righties["PlateLocSide"], rtol=1e-5)

    def test_contract_columns_exist_and_types_float64_without_inf(
        self, synthetic_lmb_df: pd.DataFrame
    ) -> None:
        """
        Criterio de Aceptación 3:
        Todas las columnas de docs/CONTRACT.md (HACK-002) existen y son float64 sin inf.
        """
        cleaned = clean_and_impute(synthetic_lmb_df)
        featured = add_physical_features(cleaned)

        contract_float64_cols = [
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
            "dist_from_zone_center",
            "dist_to_zone_edge",
        ]

        for col in contract_float64_cols:
            assert col in featured.columns, f"Columna requerida '{col}' falta en el DataFrame"
            assert featured[col].dtype == np.float64, f"Columna '{col}' debe ser float64, obtenido {featured[col].dtype}"
            assert not np.isinf(featured[col]).any(), f"Columna '{col}' contiene valores infinitos"
            assert not featured[col].isna().any(), f"Columna '{col}' contiene valores nulos"

        # Columnas enteras de zona
        int_cols = ["is_shadow_zone", "zone_quadrant", "is_same_hand"]
        for col in int_cols:
            assert col in featured.columns, f"Columna '{col}' falta en el DataFrame"
            assert np.issubdtype(featured[col].dtype, np.integer), f"Columna '{col}' debe ser de tipo entero"
            assert not featured[col].isna().any(), f"Columna '{col}' contiene valores nulos"

    def test_numerical_accuracy_physical_metrics(self) -> None:
        """Verifica cálculos físicos con casos de prueba analíticos exactos."""
        test_df = pd.DataFrame([
            {
                "PitcherThrows": "Right",
                "InducedVertBreak": 3.0,
                "HorzBreak": 4.0,
                "SpinAxis": 90.0,
                "SpinRate": 2000.0,
                "ZoneTime": 0.40,
                "RelSpeed": 90.0,
                "SpeedDrop": 9.0,
                "RelHeight": 5.5,
                "RelSide": 1.5,
                "HorzRelAngle": -1.0,
                "HorzApprAngle": -2.0,
                "PlateLocSide": 0.0,
                "PlateLocHeight": 2.5,
                "in_strike_zone": 1,
            },
            {
                "PitcherThrows": "Left",
                "InducedVertBreak": 10.0,
                "HorzBreak": -10.0,  # Leftie: HorzBreak_norm = -(-10) = +10 -> BreakAngle = atan2(10, 10) = 45 deg
                "SpinAxis": 270.0,
                "SpinRate": 2500.0,
                "ZoneTime": 0.30,  # clipped to 0.35
                "RelSpeed": 50.0,  # clipped to 60.0
                "SpeedDrop": 6.0,
                "RelHeight": 4.0,
                "RelSide": -0.5,
                "HorzRelAngle": 1.0,
                "HorzApprAngle": 2.0,
                "PlateLocSide": -1.5,
                "PlateLocHeight": 4.0,
                "in_strike_zone": 0,
            },
        ])

        featured = add_physical_features(test_df)

        # Fila 0 (Righty)
        # TotalInducedBreak: sqrt(3^2 + 4^2) = 5.0
        assert pytest.approx(featured.loc[0, "TotalInducedBreak"]) == 5.0
        # HorzBreak_norm = 4.0
        assert pytest.approx(featured.loc[0, "HorzBreak_norm"]) == 4.0
        # BreakAngle_deg: atan2(4, 3) en deg = 53.130102...
        expected_angle_0 = np.degrees(np.arctan2(4.0, 3.0))
        assert pytest.approx(featured.loc[0, "BreakAngle_deg"]) == expected_angle_0
        # SpinAxis_sin: sin(90 deg) * 1.0 = 1.0
        assert pytest.approx(featured.loc[0, "SpinAxis_sin"]) == 1.0
        # SpinAxis_cos: cos(90 deg) = 0.0
        assert pytest.approx(featured.loc[0, "SpinAxis_cos"], abs=1e-6) == 0.0
        # SpinToBreakRatio: 5.0 / ((2000/1000) * 0.40 + 1e-5) = 5.0 / 0.80001
        assert pytest.approx(featured.loc[0, "SpinToBreakRatio"], rel=1e-4) == 5.0 / 0.80001
        # DragProxy: 9.0 / 90.0 = 0.10
        assert pytest.approx(featured.loc[0, "DragProxy"]) == 0.10
        # arm_slot_deg: deg(atan2(5.5 - 4.0, 1.5 + 0.5)) = deg(atan2(1.5, 2.0)) = 36.869897...
        expected_slot_0 = np.degrees(np.arctan2(1.5, 2.0))
        assert pytest.approx(featured.loc[0, "arm_slot_deg"]) == expected_slot_0
        # dist_from_zone_center: sqrt(0^2 + (2.5 - 2.5)^2) = 0.0
        assert pytest.approx(featured.loc[0, "dist_from_zone_center"]) == 0.0
        # in_strike_zone == 1 -> zone_quadrant in 1..9
        assert 1 <= featured.loc[0, "zone_quadrant"] <= 9

        # Fila 1 (Lefty)
        # HorzBreak_norm = -(-10.0) = 10.0
        assert pytest.approx(featured.loc[1, "HorzBreak_norm"]) == 10.0
        # BreakAngle_deg = atan2(10.0, 10.0) = 45.0 deg
        assert pytest.approx(featured.loc[1, "BreakAngle_deg"]) == 45.0
        # SpinAxis_sin: sin(270 deg) * (-1.0) = (-1.0) * (-1.0) = 1.0
        assert pytest.approx(featured.loc[1, "SpinAxis_sin"]) == 1.0
        # ZoneTime clipped to 0.35: 2.5 * 0.35 + 1e-5 = 0.87501
        assert pytest.approx(featured.loc[1, "SpinToBreakRatio"], rel=1e-4) == featured.loc[1, "TotalInducedBreak"] / 0.87501
        # RelSpeed clipped to 60.0: 6.0 / 60.0 = 0.10
        assert pytest.approx(featured.loc[1, "DragProxy"]) == 0.10
        # in_strike_zone == 0 -> zone_quadrant in 10..13
        assert 10 <= featured.loc[1, "zone_quadrant"] <= 13

    def test_zone_quadrant_distribution(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """Verifica que zone_quadrant mapee 1..9 dentro de zona y 10..13 fuera de zona."""
        cleaned = clean_and_impute(synthetic_lmb_df)
        featured = add_physical_features(cleaned)

        in_zone = featured[featured["in_strike_zone"] == 1]
        out_zone = featured[featured["in_strike_zone"] == 0]

        assert set(in_zone["zone_quadrant"].unique()).issubset(set(range(1, 10)))
        assert set(out_zone["zone_quadrant"].unique()).issubset({10, 11, 12, 13})

    def test_dist_to_zone_edge_analytical_cases(self) -> None:
        """
        Verifica el cálculo analítico exacto de dist_to_zone_edge tanto para puntos interiores como exteriores:
        1) Centro exacto: PlateLocSide=0.0, PlateLocHeight=2.5 -> dist_to_zone_edge == 0.833
        2) Punto sobre el borde: PlateLocSide=0.833, PlateLocHeight=2.5 -> dist_to_zone_edge == 0.0
        3) Punto interior cercano al borde superior: PlateLocSide=0.0, PlateLocHeight=3.4 -> dist_to_zone_edge == 0.1
        4) Punto exterior en esquina: PlateLocSide=1.833, PlateLocHeight=4.5 -> dist_to_zone_edge == sqrt(1.0^2 + 1.0^2)
        """
        test_df = pd.DataFrame([
            # 1) Centro exacto
            {
                "PitcherThrows": "Right", "InducedVertBreak": 0.0, "HorzBreak": 0.0,
                "SpinAxis": 0.0, "SpinRate": 2000.0, "ZoneTime": 0.4, "RelSpeed": 90.0,
                "SpeedDrop": 8.0, "RelHeight": 5.8, "RelSide": 1.5, "HorzRelAngle": 0.0,
                "HorzApprAngle": 0.0, "PlateLocSide": 0.0, "PlateLocHeight": 2.5, "in_strike_zone": 1,
            },
            # 2) Sobre el borde lateral
            {
                "PitcherThrows": "Right", "InducedVertBreak": 0.0, "HorzBreak": 0.0,
                "SpinAxis": 0.0, "SpinRate": 2000.0, "ZoneTime": 0.4, "RelSpeed": 90.0,
                "SpeedDrop": 8.0, "RelHeight": 5.8, "RelSide": 1.5, "HorzRelAngle": 0.0,
                "HorzApprAngle": 0.0, "PlateLocSide": 0.833, "PlateLocHeight": 2.5, "in_strike_zone": 1,
            },
            # 3) Interior cercano al borde superior
            {
                "PitcherThrows": "Right", "InducedVertBreak": 0.0, "HorzBreak": 0.0,
                "SpinAxis": 0.0, "SpinRate": 2000.0, "ZoneTime": 0.4, "RelSpeed": 90.0,
                "SpeedDrop": 8.0, "RelHeight": 5.8, "RelSide": 1.5, "HorzRelAngle": 0.0,
                "HorzApprAngle": 0.0, "PlateLocSide": 0.0, "PlateLocHeight": 3.4, "in_strike_zone": 1,
            },
            # 4) Exterior en esquina
            {
                "PitcherThrows": "Right", "InducedVertBreak": 0.0, "HorzBreak": 0.0,
                "SpinAxis": 0.0, "SpinRate": 2000.0, "ZoneTime": 0.4, "RelSpeed": 90.0,
                "SpeedDrop": 8.0, "RelHeight": 5.8, "RelSide": 1.5, "HorzRelAngle": 0.0,
                "HorzApprAngle": 0.0, "PlateLocSide": 1.833, "PlateLocHeight": 4.5, "in_strike_zone": 0,
            },
        ])

        featured = add_physical_features(test_df)

        # 1) Centro exacto
        assert pytest.approx(featured.loc[0, "dist_to_zone_edge"], abs=1e-5) == 0.833
        # 2) Sobre el borde
        assert pytest.approx(featured.loc[1, "dist_to_zone_edge"], abs=1e-5) == 0.0
        # 3) Interior cercano al borde superior
        assert pytest.approx(featured.loc[2, "dist_to_zone_edge"], abs=1e-5) == 0.1
        # 4) Exterior en esquina (dx = 1.833 - 0.833 = 1.0, dz = 4.5 - 3.5 = 1.0)
        assert pytest.approx(featured.loc[3, "dist_to_zone_edge"], abs=1e-5) == np.sqrt(1.0**2 + 1.0**2)

    def test_end_to_end_on_synthetic_lmb_df(self, synthetic_lmb_df: pd.DataFrame) -> None:
        """
        Prueba de integración de punta a punta:
        Ejecuta clean_and_impute y add_physical_features sobre synthetic_lmb_df completo.
        Verifica los 3 criterios de aceptación sin leer el parquet real.
        """
        cleaned = clean_and_impute(synthetic_lmb_df)
        featured = add_physical_features(cleaned)

        # Criterio 1: Cero nulos en variables clave
        assert not cleaned["altitude_category"].isna().any()
        assert not cleaned["PitcherThrows"].isna().any()
        assert not cleaned["BatterSide"].isna().any()

        # Criterio 2: Signos invertidos para zurdos
        lefties = featured[featured["PitcherThrows"] == "Left"]
        np.testing.assert_allclose(lefties["HorzBreak_norm"], -lefties["HorzBreak"])

        # Criterio 3: Columnas de CONTRACT.md presentes, float64, sin inf
        expected_cols = [
            "HorzBreak_norm", "RelSide_norm", "HorzRelAngle_norm", "HorzApprAngle_norm",
            "PlateLocSide_norm", "TotalInducedBreak", "BreakAngle_deg", "SpinAxis_sin",
            "SpinAxis_cos", "SpinToBreakRatio", "DragProxy", "arm_slot_deg"
        ]
        for col in expected_cols:
            assert col in featured.columns
            assert featured[col].dtype == np.float64
            assert not np.isinf(featured[col]).any()

"""
Tests unitarios para src/io.py, src/config.py y fixture synthetic_lmb_df.
Cumple con docs/TASKS.md HACK-001 y docs/CONTRACT.md §1 y §4.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from src.config import (
    FORBIDDEN_TARGET_COLS,
    ID_AND_GROUP_COLS,
    LOCATION_AND_COUNT_COLS,
    DATA_DICT_PATH,
    TARGET_COLS,
    PITCH_TYPE_MAP,
    FASTBALL_TYPES,
    DECISION_POINT_TIME,
    TUNNEL_DECAY_SCALE,
)
from src.io import assert_no_leakage, load_raw_pitches
from tests.conftest import OFFICIAL_84_COLUMNS


# ==============================================================================
# a) Pruebas de synthetic_lmb_df (Shape, 84 Columnas, Valores, Estado y Física)
# ==============================================================================

def test_synthetic_df_shape_and_columns(synthetic_lmb_df: pd.DataFrame):
    """Verifica que synthetic_lmb_df tiene shape (1200, 84) y coincide 1 a 1 con el contrato."""
    assert synthetic_lmb_df.shape == (1200, 84), f"Shape inesperado: {synthetic_lmb_df.shape}"
    assert list(synthetic_lmb_df.columns) == OFFICIAL_84_COLUMNS, "Las columnas no coinciden con OFFICIAL_84_COLUMNS"

    # Si el archivo del diccionario existe en disco, verificar coincidencia exacta con él
    if DATA_DICT_PATH is not None and DATA_DICT_PATH.exists():
        dict_df = pd.read_csv(DATA_DICT_PATH)
        assert list(synthetic_lmb_df.columns) == dict_df["column_name"].tolist()


def test_synthetic_df_pitchers_games_seasons(synthetic_lmb_df: pd.DataFrame):
    """Verifica la presencia de 12 pitchers, 18 juegos y 3 temporadas."""
    # 12 pitchers nombrados pitcher_0001 .. pitcher_0012
    pitchers = set(synthetic_lmb_df["pitcher_anon_id"].unique())
    expected_pitchers = {f"pitcher_{i:04d}" for i in range(1, 13)}
    assert pitchers == expected_pitchers, f"Pitchers no coinciden: {pitchers}"

    # 18 juegos
    games = set(synthetic_lmb_df["game_anon_id"].unique())
    expected_games = {f"game_{i:04d}" for i in range(1, 19)}
    assert games == expected_games, f"Juegos no coinciden: {games}"

    # 3 temporadas: 2024, 2025, 2026
    seasons = set(synthetic_lmb_df["year"].unique())
    assert seasons == {2024, 2025, 2026}, f"Temporadas no coinciden: {seasons}"


def test_synthetic_df_null_altitudes_distribution(synthetic_lmb_df: pd.DataFrame):
    """Verifica exactamente 20 filas con altitude_category = None compartiendo juego con filas no nulas."""
    null_alt = synthetic_lmb_df[synthetic_lmb_df["altitude_category"].isna()]
    assert len(null_alt) == 20, f"Se esperaban exactamente 20 nulos en altitude_category, pero hay {len(null_alt)}"

    # Verificar que cada juego que tiene una fila nula también tiene filas no nulas (para permitir imputación por juego en HACK-002)
    for gid in null_alt["game_anon_id"].unique():
        non_null_count = (
            (synthetic_lmb_df["game_anon_id"] == gid) & synthetic_lmb_df["altitude_category"].notna()
        ).sum()
        assert non_null_count > 0, f"El juego {gid} solo tiene filas nulas en altitude_category"


def test_synthetic_df_effective_velo_format(synthetic_lmb_df: pd.DataFrame):
    """Verifica que EffectiveVelo es de tipo object con strings numéricos y algunos 'None'."""
    assert synthetic_lmb_df["EffectiveVelo"].dtype == "object"
    none_count = (synthetic_lmb_df["EffectiveVelo"] == "None").sum()
    assert none_count > 0, "Debe haber valores 'None' como string en EffectiveVelo"

    # Verificar que los no-'None' son convertibles a float
    numeric_mask = synthetic_lmb_df["EffectiveVelo"] != "None"
    converted = pd.to_numeric(synthetic_lmb_df.loc[numeric_mask, "EffectiveVelo"], errors="coerce")
    assert converted.isna().sum() == 0, "Existen valores en EffectiveVelo no convertibles a float"


def test_synthetic_df_count_matches_balls_and_strikes(synthetic_lmb_df: pd.DataFrame):
    """Verifica que el conteo pre-pitcheo coincide 100% con Balls y Strikes en todas las filas."""
    expected_count = (
        synthetic_lmb_df["Balls"].astype(int).astype(str)
        + "-"
        + synthetic_lmb_df["Strikes"].astype(int).astype(str)
    )
    assert (expected_count == synthetic_lmb_df["count"]).all(), (
        "Inconsistencia detectada: count debe ser f'{Balls}-{Strikes}' previo al lanzamiento en 100% de las filas."
    )


def test_synthetic_df_ordering(synthetic_lmb_df: pd.DataFrame):
    """Verifica que los pitcheos están estrictamente ordenados por orden de aparición al plato."""
    sort_cols = ["game_anon_id", "Inning", "Top/Bottom", "batter_anon_id", "Balls", "Strikes"]
    sorted_df = synthetic_lmb_df.sort_values(by=sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(synthetic_lmb_df, sorted_df)


def test_synthetic_df_realistic_physics(synthetic_lmb_df: pd.DataFrame):
    """Verifica que en Extreme Altitude hay menor desaceleración (ay0, SpeedDrop) y menor InducedVertBreak."""
    medians = synthetic_lmb_df.groupby("altitude_category")[["ay0", "SpeedDrop", "InducedVertBreak"]].median()

    # Extreme Altitude < Medium Altitude < No Altitude
    assert medians.loc["Extreme Altitude", "ay0"] < medians.loc["Medium Altitude", "ay0"] < medians.loc["No Altitude", "ay0"]
    assert medians.loc["Extreme Altitude", "SpeedDrop"] < medians.loc["Medium Altitude", "SpeedDrop"] < medians.loc["No Altitude", "SpeedDrop"]
    assert medians.loc["Extreme Altitude", "InducedVertBreak"] < medians.loc["Medium Altitude", "InducedVertBreak"] < medians.loc["No Altitude", "InducedVertBreak"]


# ==============================================================================
# b) y c) Pruebas de assert_no_leakage (Detección de Leakage y Features Permitidas)
# ==============================================================================

@pytest.mark.parametrize("target_col", FORBIDDEN_TARGET_COLS)
def test_assert_no_leakage_blocks_each_forbidden_target(target_col: str):
    """Verifica que cada columna de FORBIDDEN_TARGET_COLS lanza ValueError en todos los modelos."""
    for model_type in ["stuff", "location", "pitching"]:
        with pytest.raises(ValueError, match="Fuga de datos"):
            assert_no_leakage(["RelSpeed", target_col], model_type=model_type)


@pytest.mark.parametrize("id_col", ID_AND_GROUP_COLS)
def test_assert_no_leakage_blocks_each_id_col(id_col: str):
    """Verifica que cada columna de ID_AND_GROUP_COLS lanza ValueError en todos los modelos."""
    for model_type in ["stuff", "location", "pitching"]:
        with pytest.raises(ValueError, match="Fuga de datos"):
            assert_no_leakage(["RelSpeed", id_col], model_type=model_type)


@pytest.mark.parametrize("loc_col", LOCATION_AND_COUNT_COLS)
def test_assert_no_leakage_blocks_location_and_count_in_stuff(loc_col: str):
    """Verifica que cada columna de LOCATION_AND_COUNT_COLS lanza ValueError exclusivamente en modelo stuff."""
    with pytest.raises(ValueError, match="Fuga de datos"):
        assert_no_leakage(["RelSpeed", loc_col], model_type="stuff")


@pytest.mark.parametrize("loc_col", LOCATION_AND_COUNT_COLS)
def test_assert_no_leakage_permits_location_and_count_in_location_and_pitching(loc_col: str):
    """Verifica que LOCATION_AND_COUNT_COLS sí están permitidas en modelos de location y pitching."""
    # No deben lanzar excepción
    assert_no_leakage(["RelSpeed", loc_col], model_type="location")
    assert_no_leakage(["RelSpeed", loc_col], model_type="pitching")


def test_assert_no_leakage_permits_valid_physical_features():
    """Verifica que features físicas válidas pasan sin error en stuff, location y pitching."""
    valid_stuff_features = [
        "RelSpeed", "SpinRate", "SpinAxis_sin", "SpinAxis_cos",
        "InducedVertBreak", "HorzBreak_norm", "RelHeight", "RelSide_norm",
        "Extension", "VertRelAngle", "HorzRelAngle_norm", "SpeedDrop"
    ]
    assert_no_leakage(valid_stuff_features, model_type="stuff")
    assert_no_leakage(valid_stuff_features, model_type="location")
    assert_no_leakage(valid_stuff_features, model_type="pitching")


# ==============================================================================
# d) Pruebas de load_raw_pitches con Dataset Temporal (Sin Leer 206 MB)
# ==============================================================================

def test_load_raw_pitches_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifica la carga mediante load_raw_pitches usando un parquet temporal sintético
    sin leer el archivo real de 206 MB, comprobando selección de columnas y muestreo determinista.
    """
    import src.io as io_mod

    # Crear dataset de prueba pequeño en tmp_path
    tmp_parquet = tmp_path / "mock_raw_pitches.parquet"
    mock_df = pd.DataFrame({
        "PitchUID": [f"uid_{i:04d}" for i in range(100)],
        "year": [2025] * 100,
        "RelSpeed": [90.0 + i * 0.1 for i in range(100)],
        "SpinRate": [2200.0 + i for i in range(100)],
    })
    mock_df.to_parquet(tmp_parquet, engine="pyarrow")

    # Parchear RAW_DATA_PATH en src.io
    monkeypatch.setattr(io_mod, "RAW_DATA_PATH", tmp_parquet)

    # 1. Carga completa
    df_all = io_mod.load_raw_pitches()
    assert len(df_all) == 100
    assert list(df_all.columns) == ["PitchUID", "year", "RelSpeed", "SpinRate"]

    # 2. Selección de columnas
    cols = ["PitchUID", "year", "RelSpeed"]
    df_cols = io_mod.load_raw_pitches(columns=cols)
    assert list(df_cols.columns) == cols
    assert len(df_cols) == 100

    # 3. Muestreo determinista
    df_sample1 = io_mod.load_raw_pitches(columns=cols, sample_n=50)
    df_sample2 = io_mod.load_raw_pitches(columns=cols, sample_n=50)
    assert len(df_sample1) == 50
    assert list(df_sample1.columns) == cols
    pd.testing.assert_frame_equal(df_sample1, df_sample2)


def test_load_raw_pitches_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifica que load_raw_pitches lanza FileNotFoundError si RAW_DATA_PATH no existe."""
    import src.io as io_mod
    monkeypatch.setattr(io_mod, "RAW_DATA_PATH", tmp_path / "archivo_inexistente.parquet")
    with pytest.raises(FileNotFoundError):
        io_mod.load_raw_pitches()


# ==============================================================================
# e) Pruebas de Configuración Global (src/config.py)
# ==============================================================================

def test_config_constants():
    """Verifica que las constantes y parámetros requeridos en src/config.py estén bien definidos."""
    assert DECISION_POINT_TIME == 0.17
    assert TUNNEL_DECAY_SCALE == 6.0
    assert "Four-Seam" in FASTBALL_TYPES
    assert "Sinker" in FASTBALL_TYPES
    assert "Cutter" in FASTBALL_TYPES
    assert PITCH_TYPE_MAP.get("OneSeamFastBall") == "Sinker"
    assert PITCH_TYPE_MAP.get("TwoSeamFastBall") == "Sinker"
    assert PITCH_TYPE_MAP.get("Knuckleball") == "Other"
    assert "xrv_pitcher" in TARGET_COLS

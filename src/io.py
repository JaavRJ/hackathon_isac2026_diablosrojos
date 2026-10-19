"""
Módulo de entrada/salida y validación contra fuga de datos (Data Leakage).
Cumple con docs/CONTRACT.md §1 y AGENTS.md §3.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    FORBIDDEN_TARGET_COLS,
    ID_AND_GROUP_COLS,
    LOCATION_AND_COUNT_COLS,
    RAW_DATA_PATH,
)


def load_raw_pitches(
    columns: list[str] | None = None,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """
    Carga el dataset crudo de lanzamientos desde RAW_DATA_PATH utilizando pyarrow.

    Parameters
    ----------
    columns : list[str] | None, default=None
        Lista de columnas a cargar. Si es None, carga todas las columnas.
    sample_n : int | None, default=None
        Si se especifica, devuelve una muestra aleatoria determinista (random_state=42).

    Returns
    -------
    pd.DataFrame
        DataFrame con los lanzamientos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo parquet no existe en RAW_DATA_PATH.
    """
    target_path = Path(RAW_DATA_PATH) if RAW_DATA_PATH is not None else None
    if target_path is None or not target_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos crudos en: {target_path}"
        )

    df = pd.read_parquet(target_path, columns=columns, engine="pyarrow")

    if sample_n is not None:
        sample_size = min(sample_n, len(df))
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def assert_no_leakage(feature_cols: list[str], model_type: str) -> None:
    """
    Valida estrictamente que una lista de features no contenga columnas prohibidas
    por rol, de acuerdo con docs/CONTRACT.md §1.

    Parameters
    ----------
    feature_cols : list[str]
        Lista de nombres de columnas que se planea usar como matriz de entrada X.
    model_type : str
        Tipo de modelo a entrenar ('stuff', 'location', 'pitching').

    Raises
    ------
    ValueError
        Si se detecta cualquier columna prohibida para el tipo de modelo dado.
    """
    # 1. Regla universal: Prohibidas las columnas de target / resultados en cualquier modelo
    forbidden_targets_found = [c for c in feature_cols if c in FORBIDDEN_TARGET_COLS]
    if forbidden_targets_found:
        raise ValueError(
            f"Fuga de datos (Leakage) detectada: Columnas de resultado/target prohibidas "
            f"en la matriz de features: {forbidden_targets_found}"
        )

    # 2. Regla universal: Prohibidas las columnas de IDs o grupos directos en cualquier modelo
    ids_found = [c for c in feature_cols if c in ID_AND_GROUP_COLS]
    if ids_found:
        raise ValueError(
            f"Fuga de datos (Leakage) detectada: Columnas de ID/agrupación prohibidas "
            f"en la matriz de features: {ids_found}"
        )

    # 3. Regla específica para Stuff+ puro: Prohibidas ubicación de plato y conteo/contexto
    norm_model_type = model_type.lower().strip()
    if norm_model_type in ("stuff", "stuff+", "stuff_plus"):
        loc_and_count_found = [c for c in feature_cols if c in LOCATION_AND_COUNT_COLS]
        if loc_and_count_found:
            raise ValueError(
                f"Fuga de datos (Leakage) detectada en modelo Stuff+: Columnas de ubicación/conteo "
                f"prohibidas en Stuff+ puro: {loc_and_count_found}"
            )

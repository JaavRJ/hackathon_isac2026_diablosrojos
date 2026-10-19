"""
src/splits.py
=============
Generadores deterministas de particiones oficiales (holdouts) para validación de modelos:
1. Partición por lanzador (Pitcher Holdout via GroupKFold sobre 2024-2025).
2. Partición por temporada (Season Holdout: 2024-2025 train vs 2026 test).
3. Partición por altitud de parque (Park Holdout: No/Med Altitude 2024-2025 train vs Extreme Altitude 2024-2025 test).

Cumple estrictamente con:
- docs/CONTRACT.md
- docs/TASKS.md (HACK-007)
- AGENTS.md §4 (Validación obligatoria)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_pitcher_holdout_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera n_splits particiones disjuntas por lanzador (`pitcher_anon_id`)
    utilizando únicamente datos históricos de las temporadas 2024 y 2025.
    Garantiza cero fuga de datos (data leakage) respecto a la temporada 2026.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos que contiene 'year' y 'pitcher_anon_id'.
    n_splits : int, default=5
        Número de folds para la partición por grupos.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Lista de n_splits tuplas (train_idx, test_idx) con los índices correspondientes a cada fold.
        Los arrays contienen las etiquetas del índice del DataFrame de modo que tanto
        `df.loc[train_idx]` como `df.iloc[train_idx]` (en DataFrames con RangeIndex) y
        `df.loc[train_idx, 'pitcher_anon_id']` funcionen sin ambigüedad.
    """
    if "year" not in df.columns or "pitcher_anon_id" not in df.columns:
        raise ValueError("df debe contener las columnas 'year' y 'pitcher_anon_id'.")

    mask = df["year"].isin([2024, 2025])
    if not mask.any():
        raise ValueError("No se encontraron registros de las temporadas 2024 y 2025.")

    filtered_df = df.loc[mask]
    groups = filtered_df["pitcher_anon_id"].to_numpy()
    n_unique_pitchers = len(np.unique(groups))
    if n_unique_pitchers < n_splits:
        raise ValueError(
            f"El número de lanzadores únicos ({n_unique_pitchers}) es menor que n_splits ({n_splits})."
        )

    gkf = GroupKFold(n_splits=n_splits)
    filtered_indices = filtered_df.index.to_numpy()

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_pos, te_pos in gkf.split(filtered_df, groups=groups):
        train_idx = filtered_indices[tr_pos]
        test_idx = filtered_indices[te_pos]

        # Verificación estricta de disjunción de lanzadores
        train_pitchers = set(df.loc[train_idx, "pitcher_anon_id"])
        test_pitchers = set(df.loc[test_idx, "pitcher_anon_id"])
        assert train_pitchers.isdisjoint(test_pitchers), (
            f"Fuga de lanzadores detectada: {train_pitchers.intersection(test_pitchers)}"
        )

        # Verificación estricta de exclusión de la temporada 2026
        train_years = set(df.loc[train_idx, "year"])
        test_years = set(df.loc[test_idx, "year"])
        assert 2026 not in train_years, "train_idx contiene datos de la temporada 2026."
        assert 2026 not in test_years, "test_idx contiene datos de la temporada 2026."

        splits.append((train_idx, test_idx))

    return splits


def get_season_holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera la partición temporal oficial:
    - Train: temporadas 2024 y 2025.
    - Test: temporada 2026 (holdout a futuro).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos que contiene la columna 'year'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (train_idx, test_idx) con las etiquetas de índice de cada partición.
    """
    if "year" not in df.columns:
        raise ValueError("df debe contener la columna 'year'.")

    train_idx = df.index[df["year"].isin([2024, 2025])].to_numpy()
    test_idx = df.index[df["year"] == 2026].to_numpy()

    if len(train_idx) == 0:
        raise ValueError("No se encontraron registros de 2024 o 2025 para train_idx.")
    if len(test_idx) == 0:
        raise ValueError("No se encontraron registros de 2026 para test_idx.")

    train_years = set(df.loc[train_idx, "year"])
    test_years = set(df.loc[test_idx, "year"])
    assert train_years.isdisjoint(test_years), (
        f"Fuga temporal detectada: intersección de años {train_years.intersection(test_years)}"
    )
    assert train_years.issubset({2024, 2025}), f"Años inválidos en train: {train_years}"
    assert test_years.issubset({2026}), f"Años inválidos en test: {test_years}"

    return train_idx, test_idx


def get_park_holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera la partición de altitud oficial sobre datos históricos (2024-2025):
    - Train: 'No Altitude' y 'Medium Altitude' (densidad normal/moderada).
    - Test: 'Extreme Altitude' (estadios de altitud extrema, e.g. CDMX).

    Garantiza que la temporada 2026 no esté presente en ninguna partición.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos que contiene 'altitude_category' y 'year'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (train_idx, test_idx) con las etiquetas de índice de cada partición.
    """
    if "altitude_category" not in df.columns or "year" not in df.columns:
        raise ValueError("df debe contener 'altitude_category' y 'year'.")

    hist_mask = df["year"].isin([2024, 2025])
    train_mask = df["altitude_category"].isin(["No Altitude", "Medium Altitude"]) & hist_mask
    test_mask = (df["altitude_category"] == "Extreme Altitude") & hist_mask

    train_idx = df.index[train_mask].to_numpy()
    test_idx = df.index[test_mask].to_numpy()

    if len(train_idx) == 0:
        raise ValueError("No se encontraron registros de No Altitude / Medium Altitude en 2024-2025.")
    if len(test_idx) == 0:
        raise ValueError("No se encontraron registros de Extreme Altitude en 2024-2025.")

    train_alts = set(df.loc[train_idx, "altitude_category"])
    test_alts = set(df.loc[test_idx, "altitude_category"])
    assert train_alts.isdisjoint(test_alts), (
        f"Fuga de altitud detectada: {train_alts.intersection(test_alts)}"
    )

    train_years = set(df.loc[train_idx, "year"])
    test_years = set(df.loc[test_idx, "year"])
    assert train_years.issubset({2024, 2025}), f"Años inválidos en train: {train_years}"
    assert test_years.issubset({2024, 2025}), f"Años inválidos en test: {test_years}"

    return train_idx, test_idx

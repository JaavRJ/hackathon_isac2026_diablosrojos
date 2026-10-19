"""
src/baseline.py
===============
Modelo nulo de referencia (Count Baseline Model) y funciones de evaluación comparativa:
- CountBaselineModel: predice para cada uno de los 8 targets la media empírica condicionada
  únicamente por el conteo de bolas y strikes `(Balls, Strikes)` o la columna `count`.
- evaluate_against_baseline: evalúa un modelo comparándolo contra las predicciones del baseline,
  calculando RMSE para xrv_pitcher (continuo), Log-Loss y Brier Score para targets binarios,
  y el skill score de mejora relativa.

Cumple con:
- docs/CONTRACT.md §1 y §2 (TARGET_COLS)
- docs/TASKS.md (HACK-007)
- AGENTS.md §4 (Baseline nulo obligatorio)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLS

CONTINUOUS_TARGET = "xrv_pitcher"
BINARY_TARGETS = [col for col in TARGET_COLS if col != CONTINUOUS_TARGET]


class CountBaselineModel:
    """
    Modelo nulo de referencia basado exclusivamente en el conteo (Balls, Strikes).
    Calcula y almacena la media empírica de cada uno de los 8 targets oficiales
    por conteo `(Balls, Strikes)` o columna `count`, con fallback a la media global
    para conteos no observados durante el entrenamiento.
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self.count_means_: dict[str, dict[str, float]] = {}
        self.global_means_: dict[str, float] = {}

    @staticmethod
    def _extract_count_keys(df: pd.DataFrame) -> pd.Series:
        """
        Extrae o construye una Serie de strings con el formato 'B-S' (e.g. '0-0', '3-2').
        """
        if "count" in df.columns and not df["count"].isna().all():
            cnt = df["count"].astype(str)
            if ("Balls" in df.columns and "Strikes" in df.columns) and cnt.isin(["nan", "None", ""]).any():
                b = pd.to_numeric(df["Balls"], errors="coerce").fillna(0).astype(int)
                s = pd.to_numeric(df["Strikes"], errors="coerce").fillna(0).astype(int)
                fallback = b.astype(str) + "-" + s.astype(str)
                cnt = cnt.mask(cnt.isin(["nan", "None", ""]), fallback)
            return cnt
        elif "Balls" in df.columns and "Strikes" in df.columns:
            b = pd.to_numeric(df["Balls"], errors="coerce").fillna(0).astype(int)
            s = pd.to_numeric(df["Strikes"], errors="coerce").fillna(0).astype(int)
            return b.astype(str) + "-" + s.astype(str)
        else:
            raise ValueError(
                "El DataFrame debe contener la columna 'count' o las columnas ('Balls', 'Strikes')."
            )

    def fit(self, train_df: pd.DataFrame) -> "CountBaselineModel":
        """
        Ajusta el modelo nulo calculando la media de cada target por conteo y global.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame de entrenamiento con los 8 targets oficiales y columna(s) de conteo.

        Returns
        -------
        CountBaselineModel
            Instancia ajustada (self).
        """
        missing_targets = [col for col in TARGET_COLS if col not in train_df.columns]
        if missing_targets:
            raise ValueError(
                f"Faltan columnas de target obligatorias en train_df: {missing_targets}"
            )

        counts = self._extract_count_keys(train_df)

        self.global_means_ = {}
        for col in TARGET_COLS:
            vals = pd.to_numeric(train_df[col], errors="coerce")
            if vals.isna().any():
                raise ValueError(f"Columna target '{col}' contiene valores nulos no permitidos.")
            self.global_means_[col] = float(vals.mean())

        grouped = train_df[TARGET_COLS].groupby(counts).mean()
        self.count_means_ = {
            col: grouped[col].to_dict()
            for col in TARGET_COLS
        }

        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera predicciones nulas para los 8 targets basadas exclusivamente en el conteo.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada con columna 'count' o ('Balls', 'Strikes').

        Returns
        -------
        pd.DataFrame
            DataFrame con el mismo índice que df y las 8 columnas exactas de TARGET_COLS
            en tipo float64 sin valores NaN.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "CountBaselineModel debe ser entrenado con fit() antes de llamar a predict()."
            )

        counts = self._extract_count_keys(df)
        pred_dict: dict[str, np.ndarray] = {}

        for col in TARGET_COLS:
            mapping = self.count_means_[col]
            fallback = self.global_means_[col]
            pred_series = counts.map(mapping).fillna(fallback).to_numpy(dtype=np.float64)
            pred_dict[col] = pred_series

        pred_df = pd.DataFrame(pred_dict, index=df.index)
        return pred_df[TARGET_COLS]


def evaluate_against_baseline(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    baseline_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evalúa las predicciones de un modelo contra las predicciones del baseline nulo
    para los 8 targets oficiales.

    Para `xrv_pitcher` (continuo):
        calcula model_rmse, baseline_rmse, y skill_score = 1 - (model_rmse / baseline_rmse).
    Para los 7 targets binarios:
        recorta probabilidades a [1e-6, 1 - 1e-6], calcula model_logloss, baseline_logloss,
        model_brier, baseline_brier, skill_score = 1 - (model_brier / baseline_brier),
        y logloss_skill_score = 1 - (model_logloss / baseline_logloss).

    Parameters
    ----------
    y_true_df : pd.DataFrame
        Valores reales de los targets.
    y_pred_df : pd.DataFrame
        Predicciones del modelo a evaluar.
    baseline_pred_df : pd.DataFrame
        Predicciones del baseline nulo por conteo.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por target y las columnas de métricas comparativas.
    """
    for col in TARGET_COLS:
        if col not in y_true_df.columns:
            raise ValueError(f"Falta columna '{col}' en y_true_df.")
        if col not in y_pred_df.columns:
            raise ValueError(f"Falta columna '{col}' en y_pred_df.")
        if col not in baseline_pred_df.columns:
            raise ValueError(f"Falta columna '{col}' en baseline_pred_df.")

    if not (len(y_true_df) == len(y_pred_df) == len(baseline_pred_df)):
        raise ValueError("DataFrames deben tener la misma cantidad de filas.")

    eps = 1e-6
    rows: list[dict] = []

    for col in TARGET_COLS:
        y_true = pd.to_numeric(y_true_df[col], errors="coerce").to_numpy(dtype=np.float64)
        y_pred = pd.to_numeric(y_pred_df[col], errors="coerce").to_numpy(dtype=np.float64)
        y_base = pd.to_numeric(baseline_pred_df[col], errors="coerce").to_numpy(dtype=np.float64)

        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isnan(y_base).any():
            raise ValueError(f"Se detectaron valores NaN en target '{col}'.")

        if col == CONTINUOUS_TARGET:
            target_type = "continuous"
            model_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            baseline_rmse = float(np.sqrt(np.mean((y_true - y_base) ** 2)))

            skill_score = (
                1.0 - (model_rmse / baseline_rmse)
                if baseline_rmse > 1e-12
                else 0.0
            )
            skill_score_pct = 100.0 * skill_score

            model_logloss = np.nan
            baseline_logloss = np.nan
            model_brier = np.nan
            baseline_brier = np.nan
            logloss_skill_score = np.nan

        else:
            target_type = "binary"
            model_rmse = np.nan
            baseline_rmse = np.nan

            p_model = np.clip(y_pred, eps, 1.0 - eps)
            p_base = np.clip(y_base, eps, 1.0 - eps)

            model_brier = float(np.mean((p_model - y_true) ** 2))
            baseline_brier = float(np.mean((p_base - y_true) ** 2))

            skill_score = (
                1.0 - (model_brier / baseline_brier)
                if baseline_brier > 1e-12
                else 0.0
            )
            skill_score_pct = 100.0 * skill_score

            model_logloss = float(-np.mean(y_true * np.log(p_model) + (1.0 - y_true) * np.log(1.0 - p_model)))
            baseline_logloss = float(-np.mean(y_true * np.log(p_base) + (1.0 - y_true) * np.log(1.0 - p_base)))

            logloss_skill_score = (
                1.0 - (model_logloss / baseline_logloss)
                if baseline_logloss > 1e-12
                else 0.0
            )

        rows.append({
            "target": col,
            "target_type": target_type,
            "model_rmse": model_rmse,
            "baseline_rmse": baseline_rmse,
            "model_logloss": model_logloss,
            "baseline_logloss": baseline_logloss,
            "model_brier": model_brier,
            "baseline_brier": baseline_brier,
            "skill_score": skill_score,
            "skill_score_pct": skill_score_pct,
            "logloss_skill_score": logloss_skill_score,
        })

    result_df = pd.DataFrame(rows)
    return result_df

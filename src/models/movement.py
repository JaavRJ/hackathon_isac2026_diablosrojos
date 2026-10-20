"""
src/models/movement.py
======================
Etapa A: Traductor Físico Intra-Pitcher de Movimiento y Velocidad de Llegada
por Densidad del Aire (Diablos AeroVision - Reto 2 Diablos Rojos).

Cumple estrictamente con docs/CONTRACT.md, AGENTS.md y docs/TASKS.md (HACK-009):
- Modela cómo cambia el movimiento inducido (InducedVertBreak, HorzBreak_norm,
  TotalInducedBreak) y la velocidad de llegada a home (ZoneSpeed, SpeedDrop, DragProxy)
  ante variaciones en la densidad efectiva del aire (rho_hat).
- Modela la respuesta a densidad condicionada por el quiebre base con restricción
  física explícita de monotonía (monotone_constraints) en LGBMRegressor sobre la magnitud
  del quiebre:
  * Ajusta estimadores del quiebre base en release para estimar la magnitud base y construye
    la interacción rho_x_base_mag = rho_hat * abs(base_break).
  * Ajusta regresores LGBMRegressor de magnitud |InducedVertBreak| y |HorzBreak_norm| con
    restricción +1 en rho_hat y +1 en rho_x_base_mag, recuperando el signo del quiebre.
  * Ajusta ZoneSpeed con restricción monótona -1 en rho_hat (menor arrastre aerodinámico).
  * Combina con la elasticidad intra-pitcher (beta_intra) calculada por compute_intra_pitcher_deltas.
- REGLA CRÍTICA DEL RESIDUO INDIVIDUAL:
  Conserva el residuo individual observado de cada lanzamiento:
      IVB(rho*) = IVB_obs + [IVB_hat(X, rho*) - IVB_hat(X, rho_obs)]
      HB_norm(rho*) = HB_norm_obs + [HB_norm_hat(X, rho*) - HB_norm_hat(X, rho_obs)]
  Si target_rho == df["rho_hat"], translate_movement devuelve exactamente los valores
  originales (Delta = 0.0).
- Cero data leakage: Valida mediante assert_no_leakage que ninguna variable de ubicación
  en el plato ni resultados de jugadas entre como feature.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.identity import assign_invariant_pitch_id
from src.io import assert_no_leakage

# 11 columnas de features autorizadas para la física de movimiento (docs/CONTRACT.md §1)
MOVEMENT_FEATURE_COLS: list[str] = [
    "RelSpeed",
    "SpinRate",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "RelHeight",
    "RelSide_norm",
    "Extension",
    "VertRelAngle",
    "HorzRelAngle_norm",
    "arm_slot_deg",
    "rho_hat",
]


def compute_intra_pitcher_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las diferencias físicas intra-pitcher centradas en la media de cada grupo
    (pitcher_anon_id, year, pitch_id) para lanzamientos en al menos 2 juegos con
    distinta densidad del aire (rho_hat).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos procesados con TrackMan LMB.

    Returns
    -------
    pd.DataFrame
        Subconjunto de lanzamientos que pertenecen a grupos con variación de densidad,
        incluyendo las columnas centradas:
        - delta_ivb, delta_hb_norm, delta_zone_speed, delta_rho_hat
        y las medias del grupo:
        - mean_ivb, mean_hb_norm, mean_spin_rate, mean_rel_speed
    """
    req_group_cols = ["pitcher_anon_id", "year", "pitch_id"]
    output_cols = [
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

    if df.empty:
        return pd.DataFrame(columns=output_cols)

    data = df.copy()

    # Si falta pitch_id pero existe AutoPitchType, asignar pitch_id invariante
    if "pitch_id" not in data.columns and "AutoPitchType" in data.columns:
        data = assign_invariant_pitch_id(data)

    if not all(col in data.columns for col in ["pitcher_anon_id", "year", "pitch_id", "rho_hat"]):
        return pd.DataFrame(columns=output_cols)

    # Identificar grupos con lanzamientos en al menos 2 juegos con distinta densidad rho_hat
    has_game = "game_anon_id" in data.columns
    valid_indices: list[Any] = []

    for _, grp in data.groupby(req_group_cols):
        n_games = grp["game_anon_id"].nunique() if has_game else 2
        n_rho = grp["rho_hat"].nunique()
        if n_games >= 2 and n_rho >= 2:
            valid_indices.extend(grp.index)

    if not valid_indices:
        return pd.DataFrame(columns=output_cols)

    sub = data.loc[valid_indices].copy()
    grouped = sub.groupby(req_group_cols)

    # Medias base del grupo (pitcher_anon_id, year, pitch_id)
    sub["mean_ivb"] = grouped["InducedVertBreak"].transform("mean").astype("float64")
    sub["delta_ivb"] = (sub["InducedVertBreak"] - sub["mean_ivb"]).astype("float64")

    sub["mean_hb_norm"] = grouped["HorzBreak_norm"].transform("mean").astype("float64")
    sub["delta_hb_norm"] = (sub["HorzBreak_norm"] - sub["mean_hb_norm"]).astype("float64")

    sub["mean_zone_speed"] = grouped["ZoneSpeed"].transform("mean").astype("float64")
    sub["delta_zone_speed"] = (sub["ZoneSpeed"] - sub["mean_zone_speed"]).astype("float64")

    sub["mean_rho_hat"] = grouped["rho_hat"].transform("mean").astype("float64")
    sub["delta_rho_hat"] = (sub["rho_hat"] - sub["mean_rho_hat"]).astype("float64")

    sub["mean_spin_rate"] = grouped["SpinRate"].transform("mean").astype("float64")
    sub["mean_rel_speed"] = grouped["RelSpeed"].transform("mean").astype("float64")

    return sub


class PhysicsMovementTranslator:
    """
    Traductor físico de movimiento aerodinámico y velocidad por densidad del aire.

    Modela la respuesta a la densidad condicionada por el quiebre base con restricción
    física explícita de monotonía (monotone_constraints) en LGBMRegressor sobre la magnitud
    del quiebre:
    - Ajusta estimadores del quiebre base en release para estimar la magnitud base y construye
      la interacción rho_x_base_mag = rho_hat * abs(base_break).
    - Ajusta regresores LGBMRegressor de magnitud |InducedVertBreak| y |HorzBreak_norm| con
      restricciones monótonas (+1 en rho_hat y +1 en rho_x_base_mag), recuperando el signo del quiebre.
    - Menor densidad rho_hat disminuye la magnitud del quiebre inducido (Magnus + SSW).
    - Menor densidad rho_hat disminuye el arrastre (DragProxy), incrementando ligeramente ZoneSpeed.
    - Preserva de forma exacta el residuo de cada lanzamiento individual.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        random_state: int = 42,
        n_estimators: int = 80,
        learning_rate: float = 0.05,
        weight_lgb: float = 0.5,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weight_lgb = weight_lgb

        self.feature_cols: list[str] = (
            list(feature_cols) if feature_cols is not None else list(MOVEMENT_FEATURE_COLS)
        )
        # Validación obligatoria contra fuga de datos (CONTRACT.md §1)
        assert_no_leakage(self.feature_cols, "stuff")

        self.rel_cols: list[str] = [c for c in self.feature_cols if c != "rho_hat"]

        self.is_fitted: bool = False
        self._model_base_ivb: LGBMRegressor | None = None
        self._model_base_hb: LGBMRegressor | None = None
        self._model_mag_ivb: LGBMRegressor | None = None
        self._model_mag_hb: LGBMRegressor | None = None
        self._model_zs: LGBMRegressor | None = None

        # Sensibilidades físicas intra-pitcher predeterminadas
        self.beta_ivb: float = 1.0
        self.beta_hb: float = 1.0
        self.beta_zs: float = -8.0

        # Medianas de imputación defensiva
        self._feature_medians: dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
    ) -> "PhysicsMovementTranslator":
        """
        Entrena los modelos de quiebre y velocidad sobre el dataset de entrenamiento.

        Parameters
        ----------
        train_df : pd.DataFrame
            Conjunto de entrenamiento de lanzamientos.
        feature_cols : Sequence[str] | None, default=None
            Lista opcional para sobreescribir las features de entrada (sujeta a assert_no_leakage).

        Returns
        -------
        self : PhysicsMovementTranslator
        """
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
            self.rel_cols = [c for c in self.feature_cols if c != "rho_hat"]

        # Validación obligatoria de fuga de datos en cada fit
        assert_no_leakage(self.feature_cols, "stuff")

        df = train_df.copy()

        # Filtrar a year <= 2025 si la columna existe y contiene filas <= 2025
        if "year" in df.columns:
            mask_year = df["year"] <= 2025
            if mask_year.any():
                df = df.loc[mask_year].copy()

        # 1. Estimar elasticidades intra-pitcher
        intra_df = compute_intra_pitcher_deltas(df)
        if len(intra_df) > 0 and "delta_rho_hat" in intra_df.columns:
            d_rho = intra_df["delta_rho_hat"].to_numpy(dtype=np.float64)
            d_zs = intra_df["delta_zone_speed"].to_numpy(dtype=np.float64)
            denom_zs = float(np.sum(d_rho**2))
            if denom_zs > 1e-6:
                est_zs = float(np.sum(d_rho * d_zs) / denom_zs)
                # En física el arrastre se debilita en menor densidad -> d(ZS)/d(rho) < 0
                if est_zs < 0:
                    self.beta_zs = est_zs
                else:
                    self.beta_zs = -8.0

            m_ivb = intra_df["mean_ivb"].to_numpy(dtype=np.float64)
            d_ivb = intra_df["delta_ivb"].to_numpy(dtype=np.float64)
            x_ivb = d_rho * m_ivb
            denom_ivb = float(np.sum(x_ivb**2))
            if denom_ivb > 1e-6:
                est_ivb = float(np.sum(x_ivb * d_ivb) / denom_ivb)
                if est_ivb > 0:
                    self.beta_ivb = est_ivb
                else:
                    self.beta_ivb = 1.0

            m_hb = intra_df["mean_hb_norm"].to_numpy(dtype=np.float64)
            d_hb = intra_df["delta_hb_norm"].to_numpy(dtype=np.float64)
            x_hb = d_rho * m_hb
            denom_hb = float(np.sum(x_hb**2))
            if denom_hb > 1e-6:
                est_hb = float(np.sum(x_hb * d_hb) / denom_hb)
                if est_hb > 0:
                    self.beta_hb = est_hb
                else:
                    self.beta_hb = 1.0

        # 2. Calcular medianas para imputación defensiva
        for col in self.feature_cols:
            if col in df.columns:
                med = pd.to_numeric(df[col], errors="coerce").median()
                self._feature_medians[col] = float(med) if not pd.isna(med) else 0.0
            else:
                self._feature_medians[col] = 0.0

        # 3. Construir matriz de características X
        X = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce").fillna(self._feature_medians[col])
            else:
                X[col] = self._feature_medians[col]

        y_ivb = pd.to_numeric(df["InducedVertBreak"], errors="coerce").fillna(10.0)
        y_hb = pd.to_numeric(df["HorzBreak_norm"], errors="coerce").fillna(0.0)
        y_zs = pd.to_numeric(df["ZoneSpeed"], errors="coerce").fillna(85.0)

        # 4. Ajustar estimadores de magnitud de quiebre base en release (sin rho_hat)
        X_rel = X[self.rel_cols]

        self._model_base_ivb = LGBMRegressor(
            random_state=self.random_state,
            n_estimators=max(40, self.n_estimators - 20),
            learning_rate=self.learning_rate,
            verbose=-1,
        )
        self._model_base_ivb.fit(X_rel, np.abs(y_ivb))

        self._model_base_hb = LGBMRegressor(
            random_state=self.random_state,
            n_estimators=max(40, self.n_estimators - 20),
            learning_rate=self.learning_rate,
            verbose=-1,
        )
        self._model_base_hb.fit(X_rel, np.abs(y_hb))

        base_ivb_mag = self._model_base_ivb.predict(X_rel)
        base_hb_mag = self._model_base_hb.predict(X_rel)

        # 5. Construir matrices aumentadas con interacción rho_x_base_mag
        X_mag_ivb = X.copy()
        X_mag_ivb["rho_x_base_mag"] = X["rho_hat"] * base_ivb_mag

        X_mag_hb = X.copy()
        X_mag_hb["rho_x_base_mag"] = X["rho_hat"] * base_hb_mag

        # Restricciones monótonas para magnitud: +1 en rho_hat y +1 en rho_x_base_mag
        mc_mag = [0] * len(self.feature_cols) + [1]
        if "rho_hat" in self.feature_cols:
            mc_mag[self.feature_cols.index("rho_hat")] = 1

        self._model_mag_ivb = LGBMRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=-1,
            monotone_constraints=mc_mag,
        )
        self._model_mag_ivb.fit(X_mag_ivb, np.abs(y_ivb))

        self._model_mag_hb = LGBMRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=-1,
            monotone_constraints=mc_mag,
        )
        self._model_mag_hb.fit(X_mag_hb, np.abs(y_hb))

        # Restricción monótona para ZoneSpeed: -1 sobre rho_hat (menor densidad -> mayor velocidad en home)
        mc_zs = [0] * len(self.feature_cols)
        if "rho_hat" in self.feature_cols:
            mc_zs[self.feature_cols.index("rho_hat")] = -1

        self._model_zs = LGBMRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=-1,
            monotone_constraints=mc_zs,
        )
        self._model_zs.fit(X, y_zs)

        self.is_fitted = True
        return self

    def translate_movement(
        self,
        df: pd.DataFrame,
        target_rho: float | pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """
        Traduce el movimiento físico y la velocidad de llegada de los lanzamientos a
        la densidad del aire especificada en target_rho.

        Conserva estrictamente el residuo individual de cada pitcheo:
        IVB(rho*) = IVB_obs + [IVB_hat(X, rho*) - IVB_hat(X, rho_obs)]

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con los lanzamientos a traducir.
        target_rho : float | pd.Series | np.ndarray
            Densidad de destino (por ejemplo, 1.0 para nivel del mar o 0.82 para CDMX).

        Returns
        -------
        pd.DataFrame
            Copia de df con movimiento y velocidad actualizados coherentemente.
        """
        if (
            not self.is_fitted
            or self._model_base_ivb is None
            or self._model_base_hb is None
            or self._model_mag_ivb is None
            or self._model_mag_hb is None
            or self._model_zs is None
        ):
            raise RuntimeError(
                "El modelo PhysicsMovementTranslator debe ser entrenado con fit() antes de translate_movement()."
            )

        out = df.copy()
        n = len(out)
        if n == 0:
            return out

        # Parsear target_rho a arreglo 1D float64
        if isinstance(target_rho, (int, float, np.floating, np.integer)):
            target_rho_arr = np.full(n, float(target_rho), dtype=np.float64)
        elif isinstance(target_rho, pd.Series):
            target_rho_arr = target_rho.to_numpy(dtype=np.float64)
        elif isinstance(target_rho, np.ndarray):
            target_rho_arr = target_rho.astype(np.float64)
        elif hasattr(target_rho, "__iter__"):
            target_rho_arr = np.array(list(target_rho), dtype=np.float64)
        else:
            raise ValueError(f"target_rho no reconocido: {type(target_rho)}")

        if "rho_hat" in out.columns:
            obs_rho_arr = pd.to_numeric(out["rho_hat"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        else:
            obs_rho_arr = np.ones(n, dtype=np.float64)

        # Vector booleano de igualdad exacta de densidad
        is_same_rho = np.isclose(target_rho_arr, obs_rho_arr, atol=1e-12)

        # Construir matrices X_obs y X_target para LightGBM
        X_obs = pd.DataFrame(index=out.index)
        for col in self.feature_cols:
            if col == "rho_hat":
                X_obs[col] = obs_rho_arr
            elif col in out.columns:
                X_obs[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                    self._feature_medians.get(col, 0.0)
                )
            else:
                X_obs[col] = self._feature_medians.get(col, 0.0)

        X_rel = X_obs[self.rel_cols]

        # Quiebre base y signo de quiebre
        base_ivb_mag = self._model_base_ivb.predict(X_rel)
        base_hb_mag = self._model_base_hb.predict(X_rel)

        if "InducedVertBreak" in out.columns:
            ivb_s = pd.to_numeric(out["InducedVertBreak"], errors="coerce").fillna(0.0)
            sign_ivb = np.where(ivb_s < 0, -1.0, 1.0)
        else:
            sign_ivb = np.ones(n, dtype=np.float64)

        if "HorzBreak_norm" in out.columns:
            hb_s = pd.to_numeric(out["HorzBreak_norm"], errors="coerce").fillna(0.0)
            sign_hb = np.where(hb_s < 0, -1.0, 1.0)
        else:
            sign_hb = np.ones(n, dtype=np.float64)

        # Matrices de interacción para modelos de magnitud
        X_obs_ivb = X_obs.copy()
        X_obs_ivb["rho_x_base_mag"] = obs_rho_arr * base_ivb_mag

        X_target_ivb = X_obs.copy()
        X_target_ivb["rho_hat"] = target_rho_arr
        X_target_ivb["rho_x_base_mag"] = target_rho_arr * base_ivb_mag

        X_obs_hb = X_obs.copy()
        X_obs_hb["rho_x_base_mag"] = obs_rho_arr * base_hb_mag

        X_target_hb = X_obs.copy()
        X_target_hb["rho_hat"] = target_rho_arr
        X_target_hb["rho_x_base_mag"] = target_rho_arr * base_hb_mag

        # Predicciones estructurales de magnitud
        pred_mag_obs_ivb = self._model_mag_ivb.predict(X_obs_ivb)
        pred_mag_target_ivb = self._model_mag_ivb.predict(X_target_ivb)
        delta_mag_lgb_ivb = pred_mag_target_ivb - pred_mag_obs_ivb

        pred_mag_obs_hb = self._model_mag_hb.predict(X_obs_hb)
        pred_mag_target_hb = self._model_mag_hb.predict(X_target_hb)
        delta_mag_lgb_hb = pred_mag_target_hb - pred_mag_obs_hb

        # Predicciones estructurales de velocidad de llegada (ZoneSpeed)
        X_target_zs = X_obs.copy()
        X_target_zs["rho_hat"] = target_rho_arr
        pred_zs_obs = self._model_zs.predict(X_obs)
        pred_zs_target = self._model_zs.predict(X_target_zs)
        delta_zs_lgb = pred_zs_target - pred_zs_obs

        # Diferencia de densidad
        delta_rho = target_rho_arr - obs_rho_arr

        # Elasticidades físicas intra-pitcher
        delta_mag_intra_ivb = self.beta_ivb * delta_rho * pred_mag_obs_ivb
        delta_mag_intra_hb = self.beta_hb * delta_rho * pred_mag_obs_hb
        delta_zs_intra = self.beta_zs * delta_rho

        # Combinación de LightGBM con elasticidad intra-pitcher
        delta_mag_ivb = (
            self.weight_lgb * delta_mag_lgb_ivb + (1.0 - self.weight_lgb) * delta_mag_intra_ivb
        )
        delta_mag_hb = (
            self.weight_lgb * delta_mag_lgb_hb + (1.0 - self.weight_lgb) * delta_mag_intra_hb
        )
        delta_zs = self.weight_lgb * delta_zs_lgb + (1.0 - self.weight_lgb) * delta_zs_intra

        # Aplicación del signo: mayor densidad aumenta magnitud, menor densidad disminuye magnitud
        delta_ivb = sign_ivb * delta_mag_ivb
        delta_hb_norm = sign_hb * delta_mag_hb

        # Garantizar residuo exactamente cero si la densidad de destino coincide con la observada
        delta_ivb[is_same_rho] = 0.0
        delta_hb_norm[is_same_rho] = 0.0
        delta_zs[is_same_rho] = 0.0

        # Actualizar rho_hat
        out["rho_hat"] = target_rho_arr.astype(np.float64)

        # Valores originales base
        orig_ivb = (
            pd.to_numeric(out["InducedVertBreak"], errors="coerce")
            if "InducedVertBreak" in out.columns
            else pd.Series(sign_ivb * pred_mag_obs_ivb, index=out.index)
        )
        orig_hb_norm = (
            pd.to_numeric(out["HorzBreak_norm"], errors="coerce")
            if "HorzBreak_norm" in out.columns
            else pd.Series(sign_hb * pred_mag_obs_hb, index=out.index)
        )
        orig_zs = (
            pd.to_numeric(out["ZoneSpeed"], errors="coerce")
            if "ZoneSpeed" in out.columns
            else pd.Series(pred_zs_obs, index=out.index)
        )

        # Signo de mano del lanzador (+1 derecha, -1 izquierda)
        if "PitcherThrows" in out.columns:
            hand_sign = np.where(
                out["PitcherThrows"].astype(str).str.upper().str.startswith("R"), 1.0, -1.0
            )
        else:
            hand_sign = np.ones(n, dtype=np.float64)

        # Actualización de quiebres y velocidad
        out["InducedVertBreak"] = (orig_ivb + delta_ivb).astype(np.float64)
        out["HorzBreak_norm"] = (orig_hb_norm + delta_hb_norm).astype(np.float64)
        out["HorzBreak"] = (out["HorzBreak_norm"] * hand_sign).astype(np.float64)
        out["ZoneSpeed"] = (orig_zs + delta_zs).astype(np.float64)

        # SpeedDrop y DragProxy
        if "RelSpeed" in out.columns:
            rel_speed = pd.to_numeric(out["RelSpeed"], errors="coerce").fillna(90.0)
        else:
            rel_speed = out["ZoneSpeed"] + 6.0
        out["SpeedDrop"] = (rel_speed - out["ZoneSpeed"]).astype(np.float64)
        out["DragProxy"] = (out["SpeedDrop"] / rel_speed.clip(lower=60.0)).astype(np.float64)

        # TotalInducedBreak y BreakAngle_deg
        out["TotalInducedBreak"] = np.sqrt(
            out["InducedVertBreak"] ** 2 + out["HorzBreak"] ** 2
        ).astype(np.float64)
        out["BreakAngle_deg"] = np.rad2deg(
            np.arctan2(out["HorzBreak_norm"], out["InducedVertBreak"])
        ).astype(np.float64)

        # SpinToBreakRatio
        if "ZoneTime" in out.columns:
            zone_time = pd.to_numeric(out["ZoneTime"], errors="coerce").fillna(0.40)
        else:
            zone_time = pd.Series(
                0.435 + (90.0 - rel_speed.clip(lower=50.0, upper=110.0)) * 0.0035,
                index=out.index,
            )

        if "SpinRate" in out.columns:
            spin_rate = pd.to_numeric(out["SpinRate"], errors="coerce").fillna(2200.0)
        else:
            spin_rate = pd.Series(2200.0, index=out.index)

        out["SpinToBreakRatio"] = (
            out["TotalInducedBreak"]
            / ((spin_rate / 1000.0) * zone_time.clip(lower=0.35) + 1e-5)
        ).astype(np.float64)

        # Si ivb_diff_vs_fb y hb_diff_vs_fb están en out, actualizarlos coherentemente
        if "ivb_diff_vs_fb" in out.columns:
            out["ivb_diff_vs_fb"] = (
                pd.to_numeric(out["ivb_diff_vs_fb"], errors="coerce").fillna(0.0) + delta_ivb
            ).astype(np.float64)
        if "hb_diff_vs_fb" in out.columns:
            out["hb_diff_vs_fb"] = (
                pd.to_numeric(out["hb_diff_vs_fb"], errors="coerce").fillna(0.0) + delta_hb_norm
            ).astype(np.float64)

        # Para las filas donde is_same_rho es True, conservar columnas originales idénticas si ya existían
        if is_same_rho.any():
            for col in [
                "InducedVertBreak",
                "HorzBreak_norm",
                "HorzBreak",
                "ZoneSpeed",
                "TotalInducedBreak",
                "SpeedDrop",
                "DragProxy",
                "BreakAngle_deg",
                "SpinToBreakRatio",
                "ivb_diff_vs_fb",
                "hb_diff_vs_fb",
            ]:
                if col in df.columns:
                    out.loc[is_same_rho, col] = df.loc[is_same_rho, col]

        return out

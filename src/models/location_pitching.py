"""
src/models/location_pitching.py
================================
Submodelos Location+ y Pitching+ Multi-Target ajustados por entorno
(Reto 2 Diablos Rojos - Diablos AeroVision).

Cumple estrictamente con:
- docs/CONTRACT.md §1, §2 y §3
- docs/TASKS.md (HACK-011)
- AGENTS.md §1, §2, §3 y §4

Modelos:
1. LocationPlusModel:
   - Features autorizadas (LOCATION_FEATURES): variables geométricas de plato,
     zona de strike, conteo (Balls, Strikes, Outs), lateralidad (BatterSide,
     PitcherThrows, is_same_hand) y pitch_id invariante. Whitelist estricta sin
     acceso a densidad del aire ni variables físicas de velocidad/giro/movimiento.
   - Prohibición estricta validada mediante assert_valid_location_features.
   - Estimación multi-target sobre targets de comando y resultado combinados hacia xrv_pitcher.
   - Normalización oficial Score+ (media 100.0, std 10.0 en sabor neutral LMB).
   - location_plus_neutral y location_plus_cdmx idénticos (mide comando puro independiente de parque).

2. PitchingPlusModel:
   - Features autorizadas (PITCHING_FEATURES): Unión de características físicas de Stuff+,
     métricas de túnel y arsenal, geometría de plato/zona, conteo y contexto.
   - Validación rigurosa anti-fuga (assert_no_leakage).
   - Predicción explícita de los 8 targets oficiales de las bases:
     P(whiff), P(chase), P(called_strike), P(groundball), P(weak_contact),
     P(hard_contact), P(out_on_pitch) y xrv_pitcher directo/meta-regresión.
   - Sabores neutral (rho_hat = 1.000) y CDMX / Extreme Altitude (rho_cdmx específico o
     la mediana empírica observada en Extreme Altitude ~0.852) con movimiento traducido
     y cálculo de delta_pitching_cdmx.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge, RidgeCV

from src.config import FORBIDDEN_TARGET_COLS, ID_AND_GROUP_COLS, TARGET_COLS
from src.io import assert_no_leakage

# ==============================================================================
# 1. Definición de Columnas Autorizadas y Prohibidas
# ==============================================================================

# Columnas de velocidad, giro y movimiento físico TERMINANTEMENTE PROHIBIDAS en LocationPlusModel
FORBIDDEN_LOCATION_PHYSICAL_FEATURES = [
    "RelSpeed",
    "EffectiveVelo",
    "ZoneSpeed",
    "SpinRate",
    "SpinAxis",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "InducedVertBreak",
    "VertBreak",
    "HorzBreak",
    "HorzBreak_norm",
    "TotalInducedBreak",
    "BreakAngle_deg",
    "SpinToBreakRatio",
    "DragProxy",
    "pfxx",
    "pfxz",
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "tunnel_dist_release_in",
    "tunnel_sim_release",
    "tunnel_dist_170ms_in",
    "tunnel_sim_170ms",
]

# Features canónicas de ubicación y contexto autorizadas para LocationPlusModel (docs/CONTRACT.md §1.3)
LOCATION_FEATURES = [
    "PlateLocHeight",
    "PlateLocSide_norm",
    "in_strike_zone",
    "dist_from_zone_center",
    "dist_to_zone_edge",
    "is_shadow_zone",
    "zone_quadrant",
    "Balls",
    "Strikes",
    "Outs",
    "BatterSide",
    "PitcherThrows",
    "is_same_hand",
    "pitch_id",
]
VALID_LOCATION_FEATURES = frozenset(LOCATION_FEATURES)

# Features canónicas para PitchingPlusModel (unión de Stuff+, Location+ y contexto)
PITCHING_FEATURES = [
    # Físicas de release y movimiento
    "RelSpeed",
    "EffectiveVelo",
    "ZoneSpeed",
    "SpinRate",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "RelHeight",
    "RelSide_norm",
    "Extension",
    "InducedVertBreak",
    "HorzBreak_norm",
    "TotalInducedBreak",
    "BreakAngle_deg",
    "VertRelAngle",
    "HorzRelAngle_norm",
    "SpinToBreakRatio",
    "arm_slot_deg",
    # Arsenal relacional y túneles
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "tunnel_dist_release_in",
    "tunnel_sim_release",
    "tunnel_dist_170ms_in",
    "tunnel_sim_170ms",
    "pitch_usage_pct",
    # Ubicación y zona de strike
    "PlateLocHeight",
    "PlateLocSide_norm",
    "in_strike_zone",
    "dist_from_zone_center",
    "dist_to_zone_edge",
    "is_shadow_zone",
    "zone_quadrant",
    # Conteo y contexto del juego
    "Balls",
    "Strikes",
    "Outs",
    "BatterSide",
    "PitcherThrows",
    "is_same_hand",
    "pitch_id",
    # Ángulos de aproximación y secuencia
    "VertApprAngle",
    "HorzApprAngle_norm",
    "Inning",
    "prev_pitch_id",
    # Densidad de entorno
    "rho_hat",
]

# Columnas binarias de resultado
BINARY_TARGETS = [c for c in TARGET_COLS if c != "xrv_pitcher"]

# Targets de comando específicos para LocationPlusModel
LOCATION_COMMAND_TARGETS = [
    "target_called_strike",
    "target_chase",
    "target_whiff",
    "target_weak_contact",
    "target_hard_contact",
]


def assert_valid_location_features(feature_cols: list[str]) -> None:
    """
    Valida estrictamente que una lista de features para LocationPlusModel cumpla con
    docs/CONTRACT.md §1:
    1. Llama a assert_no_leakage(feature_cols, 'location') para prohibir targets e IDs.
    2. Aplica una whitelist estricta: cualquier columna que no pertenezca a LOCATION_FEATURES
       (incluyendo rho_hat, RelHeight, RelSide, Extension, VertRelAngle, y toda variable
       de velocidad, spin o quiebre físico) lanza ValueError.

    Parameters
    ----------
    feature_cols : list[str]
        Lista de nombres de features que se planea usar.

    Raises
    ------
    ValueError
        Si se detecta cualquier columna fuera de LOCATION_FEATURES.
    """
    assert_no_leakage(feature_cols, "location")
    unauthorized = [c for c in feature_cols if c not in VALID_LOCATION_FEATURES]
    if unauthorized:
        raise ValueError(
            f"Features no autorizadas detectadas en LocationPlusModel: {unauthorized}. "
            f"LocationPlusModel solo admite variables de la lista autorizada LOCATION_FEATURES: {LOCATION_FEATURES}"
        )


def _prepare_feature_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_categories: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    """
    Prepara la matriz de features para LightGBM asegurando tipos de datos compatibles
    (categorías registradas, booleanos como enteros y numéricos sin NaN).
    """
    X = pd.DataFrame(index=df.index)
    stored_categories: dict[str, list[Any]] = {}

    cat_candidate_cols = {
        "BatterSide",
        "PitcherThrows",
        "pitch_id",
        "prev_pitch_id",
        "zone_quadrant",
    }

    for col in feature_cols:
        if col not in df.columns:
            # Fallback defensivo si alguna feature calculable falta
            if col == "PlateLocSide_norm" and "PlateLocSide" in df.columns and "PitcherThrows" in df.columns:
                hand_sign = np.where(df["PitcherThrows"].astype(str).str.upper() == "RIGHT", 1.0, -1.0)
                X[col] = pd.to_numeric(df["PlateLocSide"], errors="coerce").fillna(0.0) * hand_sign
                continue
            elif col == "is_same_hand" and "BatterSide" in df.columns and "PitcherThrows" in df.columns:
                X[col] = (df["BatterSide"].astype(str) == df["PitcherThrows"].astype(str)).astype(int)
                continue
            elif col == "rho_hat":
                X[col] = 1.0
                continue
            else:
                raise ValueError(f"Falta columna requerida '{col}' en DataFrame de entrada.")

        series = df[col]

        if col in cat_candidate_cols or series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
            str_series = series.astype(str).fillna("UNKNOWN")
            if cat_categories is not None and col in cat_categories:
                cats = cat_categories[col]
                clean_series = str_series.where(str_series.isin(cats), other=np.nan)
                X[col] = pd.Categorical(clean_series, categories=cats)
            else:
                cat_obj = pd.Categorical(str_series)
                X[col] = cat_obj
                stored_categories[col] = list(cat_obj.categories)
        elif series.dtype == "bool":
            X[col] = series.astype(int)
        else:
            X[col] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64)

    return X, (cat_categories if cat_categories is not None else stored_categories)


# ==============================================================================
# 2. Submodelo Location+ (LocationPlusModel)
# ==============================================================================

class LocationPlusModel:
    """
    Submodelo Location+ que evalúa el valor esperado de la ubicación del lanzamiento
    condicionado a conteo, lateralidad y tipo invariante de pitcheo (sin acceso a
    características físicas de velocidad o movimiento).
    """

    def __init__(self, feature_cols: list[str] | None = None) -> None:
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(LOCATION_FEATURES)
        assert_valid_location_features(self.feature_cols)

        self.is_fitted_: bool = False
        self.submodels_: dict[str, Any] = {}
        self.meta_model_: RidgeCV | None = None
        self.cat_categories_: dict[str, list[Any]] = {}
        self.mu_neutral_: float = 0.0
        self.sigma_neutral_: float = 1.0

    def fit(self, train_df: pd.DataFrame) -> "LocationPlusModel":
        """
        Entrena los submodelos de comando y resultado y calibra la escala Plus neutral.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame de entrenamiento con features de ubicación y targets oficiales.

        Returns
        -------
        LocationPlusModel
            Instancia ajustada (self).
        """
        assert_valid_location_features(self.feature_cols)

        # Filtrar a 2024-2025 para garantizar cero fuga temporal de 2026
        df_train = train_df.copy()
        if "year" in df_train.columns:
            df_train = df_train[df_train["year"] <= 2025].copy()

        if len(df_train) == 0:
            raise ValueError("train_df está vacío tras filtrar a year <= 2025.")

        X_train, self.cat_categories_ = _prepare_feature_df(df_train, self.feature_cols)
        n_samples = len(X_train)
        min_child = min(20, max(2, n_samples // 5))

        # 1. Entrenar submodelos para los targets de comando
        self.submodels_ = {}
        target_preds_dict: dict[str, np.ndarray] = {}

        for tgt in LOCATION_COMMAND_TARGETS:
            if tgt not in df_train.columns:
                raise ValueError(f"Falta columna de target '{tgt}' en train_df.")
            y = pd.to_numeric(df_train[tgt], errors="coerce").fillna(0.0).to_numpy()

            if len(np.unique(y)) < 2:
                const_val = float(np.mean(y))
                self.submodels_[tgt] = ("constant", const_val)
                target_preds_dict[tgt] = np.full(n_samples, const_val, dtype=np.float64)
            else:
                clf = LGBMClassifier(
                    random_state=42,
                    n_estimators=60,
                    learning_rate=0.05,
                    min_child_samples=min_child,
                    verbose=-1,
                )
                clf.fit(X_train, y)
                self.submodels_[tgt] = clf
                target_preds_dict[tgt] = clf.predict_proba(X_train)[:, 1].astype(np.float64)

        # Submodelo regresor directo para xrv_pitcher
        if "xrv_pitcher" not in df_train.columns:
            raise ValueError("Falta columna de target 'xrv_pitcher' en train_df.")
        y_xrv = pd.to_numeric(df_train["xrv_pitcher"], errors="coerce").fillna(0.0).to_numpy()

        reg_xrv = LGBMRegressor(
            random_state=42,
            n_estimators=60,
            learning_rate=0.05,
            min_child_samples=min_child,
            verbose=-1,
        )
        reg_xrv.fit(X_train, y_xrv)
        self.submodels_["xrv_pitcher"] = reg_xrv
        target_preds_dict["xrv_pitcher"] = reg_xrv.predict(X_train).astype(np.float64)

        # 2. Meta-modelo RidgeCV para combinar las predicciones hacia xrv_pitcher
        stacked_features = [target_preds_dict[tgt] for tgt in LOCATION_COMMAND_TARGETS] + [target_preds_dict["xrv_pitcher"]]
        M_train = np.column_stack(stacked_features)

        self.meta_model_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.meta_model_.fit(M_train, y_xrv)

        # 3. Guardar media y desviación estándar de referencia en sabor neutral
        pred_neutral_train = self._predict_expected_xrv(df_train)
        self.mu_neutral_ = float(np.mean(pred_neutral_train))
        std_val = float(np.std(pred_neutral_train, ddof=0))
        self.sigma_neutral_ = float(max(std_val, 1e-6))

        self.is_fitted_ = True
        return self

    def _predict_expected_xrv(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula el índice de valor de comando esperado (expected_xrv_location)
        evaluando los submodelos y el meta-regresor Ridge.
        """
        X, _ = _prepare_feature_df(df, self.feature_cols, self.cat_categories_)
        n_samples = len(X)
        pred_cols: list[np.ndarray] = []

        for tgt in LOCATION_COMMAND_TARGETS:
            sub = self.submodels_[tgt]
            if isinstance(sub, tuple) and sub[0] == "constant":
                pred_cols.append(np.full(n_samples, sub[1], dtype=np.float64))
            else:
                p = sub.predict_proba(X)[:, 1].astype(np.float64)
                pred_cols.append(p)

        reg_xrv = self.submodels_["xrv_pitcher"]
        pred_xrv_direct = reg_xrv.predict(X).astype(np.float64)
        pred_cols.append(pred_xrv_direct)

        M = np.column_stack(pred_cols)
        return self.meta_model_.predict(M).astype(np.float64)

    def predict_scores(self, df: pd.DataFrame, rho_cdmx: float | None = None) -> pd.DataFrame:
        """
        Genera los scores Location+ en sabor neutral y sabor CDMX, y calcula delta_location_cdmx.
        Como Location+ mide comando puro condicionado al conteo, lateralidad y tipo de pitcheo
        sin depender de la densidad del aire del parque, tanto location_plus_neutral como
        location_plus_cdmx se evalúan en la misma escala neutral, siendo idénticos cuando la
        ubicación y contexto son iguales (delta_location_cdmx == 0.0).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada con las features de LOCATION_FEATURES.
        rho_cdmx : float | None, default=None
            Parámetro opcional para compatibilidad con la interfaz de evaluación por entorno.
            No altera la evaluación geométrica neutral de Location+.

        Returns
        -------
        pd.DataFrame
            Copia de df con location_plus_neutral, location_plus_cdmx y delta_location_cdmx.
        """
        if not self.is_fitted_:
            raise RuntimeError("LocationPlusModel debe entrenarse con fit() antes de predict_scores().")

        pred_neutral = self._predict_expected_xrv(df)
        loc_plus_neutral = 100.0 + 10.0 * (pred_neutral - self.mu_neutral_) / self.sigma_neutral_
        loc_plus_cdmx = loc_plus_neutral.copy()
        delta_loc = np.zeros(len(df), dtype=np.float64)

        out = df.copy()
        out["location_plus_neutral"] = loc_plus_neutral.astype(np.float64)
        out["location_plus_cdmx"] = loc_plus_cdmx.astype(np.float64)
        out["delta_location_cdmx"] = delta_loc.astype(np.float64)

        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice los targets oficiales desde la perspectiva de ubicación para compatibilidad.
        """
        if not self.is_fitted_:
            raise RuntimeError("LocationPlusModel debe entrenarse con fit() antes de predict().")

        X, _ = _prepare_feature_df(df, self.feature_cols, self.cat_categories_)
        res: dict[str, np.ndarray] = {}

        for tgt in LOCATION_COMMAND_TARGETS:
            sub = self.submodels_[tgt]
            if isinstance(sub, tuple) and sub[0] == "constant":
                res[tgt] = np.full(len(X), sub[1], dtype=np.float64)
            else:
                res[tgt] = np.clip(sub.predict_proba(X)[:, 1], 0.0, 1.0).astype(np.float64)

        res["xrv_pitcher"] = self._predict_expected_xrv(df)

        # Fallbacks neutros si se solicitan los targets restantes
        if "target_groundball" not in res:
            res["target_groundball"] = np.full(len(X), 0.45, dtype=np.float64)
        if "target_out_on_pitch" not in res:
            res["target_out_on_pitch"] = np.full(len(X), 0.20, dtype=np.float64)

        return pd.DataFrame(res, index=df.index)[TARGET_COLS].astype(np.float64)


# ==============================================================================
# 3. Submodelo Pitching+ (PitchingPlusModel)
# ==============================================================================

class PitchingPlusModel:
    """
    Submodelo Pitching+ Multi-Target que combina las características físicas del pitcheo
    (Stuff+) con el comando de plato (Location+) y el contexto del juego para predecir
    los 8 targets oficiales y generar los scores pitching_plus_neutral y pitching_plus_cdmx.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        movement_translator: Any | None = None,
    ) -> None:
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(PITCHING_FEATURES)
        assert_no_leakage(self.feature_cols, "pitching")

        self.movement_translator = movement_translator
        self.is_fitted_: bool = False
        self.submodels_: dict[str, Any] = {}
        self.meta_ridge_: Ridge | None = None
        self.cat_categories_: dict[str, list[Any]] = {}
        self.mu_neutral_: float = 0.0
        self.sigma_neutral_: float = 1.0
        self.rho_extreme_median_: float = 0.852  # Mediana empírica en Extreme Altitude (~0.852)

    @staticmethod
    def _translate_df_movement(
        df: pd.DataFrame,
        target_rho: float,
        translator: Any | None = None,
    ) -> pd.DataFrame:
        """
        Traduce el movimiento físico y la velocidad del DataFrame a la densidad target_rho.
        Si se suministra un movement_translator con método translate_movement, lo utiliza;
        en caso contrario, realiza el escalado físico proporcional directo por densidad.
        """
        if translator is not None and hasattr(translator, "translate_movement"):
            translated = translator.translate_movement(df, target_rho=float(target_rho)).copy()
            translated["rho_hat"] = float(target_rho)
            return translated

        out = df.copy()
        obs_rho = out["rho_hat"].astype(float) if "rho_hat" in out.columns else pd.Series(1.0, index=out.index)
        obs_rho = obs_rho.replace(0.0, 1.0).fillna(1.0)
        ratio = float(target_rho) / obs_rho

        out["rho_hat"] = float(target_rho)
        for col in ["InducedVertBreak", "HorzBreak_norm", "ivb_diff_vs_fb", "hb_diff_vs_fb"]:
            if col in out.columns:
                out[col] = (out[col] * ratio).astype(np.float64)

        if "TotalInducedBreak" in out.columns and "InducedVertBreak" in out.columns and "HorzBreak_norm" in out.columns:
            out["TotalInducedBreak"] = np.sqrt(out["InducedVertBreak"] ** 2 + out["HorzBreak_norm"] ** 2).astype(np.float64)

        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        movement_translator: Any | None = None,
    ) -> "PitchingPlusModel":
        """
        Entrena los 8 submodelos multi-target y el meta-regresor Ridge sobre 2024-2025.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame de entrenamiento.
        movement_translator : Any | None, optional
            Instancia de traductor de movimiento para calibración de sabores de altitud.

        Returns
        -------
        PitchingPlusModel
            Instancia ajustada (self).
        """
        assert_no_leakage(self.feature_cols, "pitching")
        if movement_translator is not None:
            self.movement_translator = movement_translator

        df_train = train_df.copy()
        if "year" in df_train.columns:
            df_train = df_train[df_train["year"] <= 2025].copy()

        if len(df_train) == 0:
            raise ValueError("train_df está vacío tras filtrar a year <= 2025.")

        X_train, self.cat_categories_ = _prepare_feature_df(df_train, self.feature_cols)
        n_samples = len(X_train)
        min_child = min(20, max(2, n_samples // 5))

        self.submodels_ = {}
        target_preds_dict: dict[str, np.ndarray] = {}

        # 1. Entrenar los 7 clasificadores binarios
        for tgt in BINARY_TARGETS:
            if tgt not in df_train.columns:
                raise ValueError(f"Falta columna de target '{tgt}' en train_df.")
            y_b = pd.to_numeric(df_train[tgt], errors="coerce").fillna(0.0).to_numpy()

            if len(np.unique(y_b)) < 2:
                const_val = float(np.mean(y_b))
                self.submodels_[tgt] = ("constant", const_val)
                target_preds_dict[tgt] = np.full(n_samples, const_val, dtype=np.float64)
            else:
                clf = LGBMClassifier(
                    random_state=42,
                    n_estimators=60,
                    learning_rate=0.05,
                    min_child_samples=min_child,
                    verbose=-1,
                )
                clf.fit(X_train, y_b)
                self.submodels_[tgt] = clf
                p = clf.predict_proba(X_train)[:, 1].astype(np.float64)
                target_preds_dict[tgt] = np.clip(p, 0.0, 1.0)

        # 2. Entrenar regresor directo para xrv_pitcher
        if "xrv_pitcher" not in df_train.columns:
            raise ValueError("Falta columna de target 'xrv_pitcher' en train_df.")
        y_xrv = pd.to_numeric(df_train["xrv_pitcher"], errors="coerce").fillna(0.0).to_numpy()

        reg_xrv = LGBMRegressor(
            random_state=42,
            n_estimators=60,
            learning_rate=0.05,
            min_child_samples=min_child,
            verbose=-1,
        )
        reg_xrv.fit(X_train, y_xrv)
        self.submodels_["xrv_pitcher"] = reg_xrv
        target_preds_dict["xrv_pitcher"] = reg_xrv.predict(X_train).astype(np.float64)

        # 3. Meta-regresor Ridge sobre las 8 predicciones hacia xrv_pitcher
        M_train = np.column_stack([target_preds_dict[tgt] for tgt in TARGET_COLS])
        self.meta_ridge_ = Ridge(alpha=1.0)
        self.meta_ridge_.fit(M_train, y_xrv)

        # 4. Normalización en sabor neutral (rho_hat = 1.000)
        df_neutral_train = self._translate_df_movement(
            df_train,
            target_rho=1.000,
            translator=self.movement_translator,
        )
        pred_neutral_train = self._predict_meta_xrv(df_neutral_train)
        self.mu_neutral_ = float(np.mean(pred_neutral_train))
        std_val = float(np.std(pred_neutral_train, ddof=0))
        self.sigma_neutral_ = float(max(std_val, 1e-6))

        # 5. Estimar empíricamente la mediana de rho_hat en Extreme Altitude
        # En datos reales LMB (reports/altitude/game_density_summary.csv), la mediana observada
        # en Extreme Altitude para 2024-2025 es ~0.852 (agrupando múltiples parques de altura, no solo CDMX).
        if "altitude_category" in df_train.columns and "rho_hat" in df_train.columns:
            extreme_mask = df_train["altitude_category"] == "Extreme Altitude"
            if extreme_mask.any():
                self.rho_extreme_median_ = float(df_train.loc[extreme_mask, "rho_hat"].median())
            else:
                self.rho_extreme_median_ = float(df_train["rho_hat"].quantile(0.10))
        elif "rho_hat" in df_train.columns:
            self.rho_extreme_median_ = float(df_train["rho_hat"].quantile(0.10))
        else:
            self.rho_extreme_median_ = 0.852

        self.is_fitted_ = True
        return self

    def _predict_submodels(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Genera las predicciones individuales de cada uno de los 8 submodelos.
        """
        X, _ = _prepare_feature_df(df, self.feature_cols, self.cat_categories_)
        n_samples = len(X)
        preds: dict[str, np.ndarray] = {}

        for tgt in BINARY_TARGETS:
            sub = self.submodels_[tgt]
            if isinstance(sub, tuple) and sub[0] == "constant":
                preds[tgt] = np.full(n_samples, sub[1], dtype=np.float64)
            else:
                p = sub.predict_proba(X)[:, 1].astype(np.float64)
                preds[tgt] = np.clip(p, 0.0, 1.0)

        reg_xrv = self.submodels_["xrv_pitcher"]
        preds["xrv_pitcher_direct"] = reg_xrv.predict(X).astype(np.float64)

        # Matriz M para el meta-regresor
        M = np.column_stack([preds[tgt] for tgt in BINARY_TARGETS] + [preds["xrv_pitcher_direct"]])
        preds["xrv_pitcher"] = self.meta_ridge_.predict(M).astype(np.float64)

        return preds

    def _predict_meta_xrv(self, df: pd.DataFrame) -> np.ndarray:
        """
        Evalúa el índice final esperado de xrv_pitcher usando el meta-regresor Ridge.
        """
        preds = self._predict_submodels(df)
        return preds["xrv_pitcher"]

    def predict_all_targets(
        self,
        df: pd.DataFrame,
        movement_translator: Any | None = None,
        rho_cdmx: float | None = None,
    ) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las 8 predicciones calibradas de los targets,
        así como pitching_plus_neutral, pitching_plus_cdmx y delta_pitching_cdmx.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de lanzamientos.
        movement_translator : Any | None, optional
            Traductor físico de movimiento para evaluación de altitud.
        rho_cdmx : float | None, default=None
            Densidad relativa objetivo para sabor CDMX / Extreme Altitude. Si es None,
            utiliza la mediana empírica estimada de los datos de entrenamiento
            (self.rho_extreme_median_ ~0.852 en Extreme Altitude de LMB).

        Returns
        -------
        pd.DataFrame
            DataFrame con las 8 predicciones y los scores Plus en float64 sin NaN.
        """
        if not self.is_fitted_:
            raise RuntimeError("PitchingPlusModel debe entrenarse con fit() antes de predict_all_targets().")

        translator = movement_translator if movement_translator is not None else self.movement_translator
        target_rho = float(rho_cdmx) if rho_cdmx is not None else float(self.rho_extreme_median_)

        # 1. Predicciones observadas de los 8 targets oficiales
        raw_preds = self._predict_submodels(df)

        # 2. Sabor neutral (rho_hat = 1.000)
        df_neutral = self._translate_df_movement(df, target_rho=1.000, translator=translator)
        pred_neutral = self._predict_meta_xrv(df_neutral)
        pitching_plus_neutral = 100.0 + 10.0 * (pred_neutral - self.mu_neutral_) / self.sigma_neutral_

        # 3. Sabor CDMX / Extreme Altitude (rho_hat = target_rho con la escala neutral)
        df_cdmx = self._translate_df_movement(df, target_rho=target_rho, translator=translator)
        pred_cdmx = self._predict_meta_xrv(df_cdmx)
        pitching_plus_cdmx = 100.0 + 10.0 * (pred_cdmx - self.mu_neutral_) / self.sigma_neutral_

        delta_pitching = pitching_plus_cdmx - pitching_plus_neutral

        result_df = pd.DataFrame(
            {
                "pred_target_whiff": raw_preds["target_whiff"],
                "pred_target_chase": raw_preds["target_chase"],
                "pred_target_called_strike": raw_preds["target_called_strike"],
                "pred_target_groundball": raw_preds["target_groundball"],
                "pred_target_weak_contact": raw_preds["target_weak_contact"],
                "pred_target_hard_contact": raw_preds["target_hard_contact"],
                "pred_target_out_on_pitch": raw_preds["target_out_on_pitch"],
                "pred_xrv_pitcher": raw_preds["xrv_pitcher"],
                "pitching_plus_neutral": pitching_plus_neutral,
                "pitching_plus_cdmx": pitching_plus_cdmx,
                "delta_pitching_cdmx": delta_pitching,
            },
            index=df.index,
        ).astype(np.float64)

        return result_df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las columnas exactas TARGET_COLS para compatibilidad
        directa con evaluate_against_baseline de HACK-007.
        """
        all_targets = self.predict_all_targets(df)
        col_mapping = {
            "pred_target_whiff": "target_whiff",
            "pred_target_chase": "target_chase",
            "pred_target_called_strike": "target_called_strike",
            "pred_target_groundball": "target_groundball",
            "pred_target_weak_contact": "target_weak_contact",
            "pred_target_hard_contact": "target_hard_contact",
            "pred_target_out_on_pitch": "target_out_on_pitch",
            "pred_xrv_pitcher": "xrv_pitcher",
        }
        pred_cols = {target_name: all_targets[pred_name] for pred_name, target_name in col_mapping.items()}
        return pd.DataFrame(pred_cols, index=df.index)[TARGET_COLS].astype(np.float64)

    def predict_scores(
        self,
        df: pd.DataFrame,
        movement_translator: Any | None = None,
        rho_cdmx: float | None = None,
    ) -> pd.DataFrame:
        """
        Adjunta los scores pitching_plus_neutral, pitching_plus_cdmx y delta_pitching_cdmx
        a una copia del DataFrame original df.
        """
        all_preds = self.predict_all_targets(df, movement_translator=movement_translator, rho_cdmx=rho_cdmx)
        out = df.copy()
        out["pitching_plus_neutral"] = all_preds["pitching_plus_neutral"]
        out["pitching_plus_cdmx"] = all_preds["pitching_plus_cdmx"]
        out["delta_pitching_cdmx"] = all_preds["delta_pitching_cdmx"]

        for col in all_preds.columns:
            if col.startswith("pred_"):
                out[col] = all_preds[col]

        return out

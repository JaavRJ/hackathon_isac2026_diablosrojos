"""
src/models/stuff.py
===================
Submodelo Stuff+ Multi-Target en Sabores Neutral y CDMX / Entorno
(Reto 2 Diablos Rojos - Diablos AeroVision).

Cumple estrictamente con:
- docs/CONTRACT.md §1, §2 y §3
- docs/TASKS.md (HACK-010)
- AGENTS.md §1, §2, §3 y §4

Arquitectura y Principios Físicos:
1. Features Autorizadas (STUFF_FEATURES):
   - Variables físicas de release y trayectoria: velocidad (RelSpeed, EffectiveVelo,
     ZoneSpeed), giro (SpinRate, SpinAxis_sin, SpinAxis_cos), punto de liberación
     (RelHeight, RelSide_norm, Extension, VertRelAngle, HorzRelAngle_norm), quiebre
     inducido (InducedVertBreak, HorzBreak_norm, TotalInducedBreak, BreakAngle_deg,
     SpinToBreakRatio) y geometría de brazo (arm_slot_deg, PitcherThrows).
   - Arsenal relacional y túneles: velo_diff_vs_fb, ivb_diff_vs_fb, hb_diff_vs_fb,
     tunnel_dist_release_in, tunnel_sim_release, tunnel_dist_170ms_in,
     tunnel_sim_170ms y pitch_usage_pct.
   - CERO DATA LEAKAGE: Whitelist estricta que excluye terminantemente ubicación
     de plato (PlateLoc*), conteo (Balls, Strikes, count, Outs), lateralidad de
     bateador (BatterSide, is_same_hand), secuencia (prev_pitch_id), ángulos de
     llegada (VertApprAngle, HorzApprAngle, HorzApprAngle_norm) y densidad de entorno (rho_hat).

2. Arquitectura Multi-Target en el Canal Stuff Puro:
   - Para evitar la contaminación por el carry aerodinámico del parque en bolas bateadas
     en altitud (tanto Extreme Altitude como Medium Altitude), el entrenamiento de los
     submodelos y del combinador del Canal Stuff utiliza exclusivamente observaciones
     de No Altitude (altitude_category == 'No Altitude').
   - Entrena modelos LightGBM independientes para los 8 targets oficiales de las bases:
     P(whiff), P(chase), P(called_strike), P(groundball), P(weak_contact),
     P(hard_contact), P(out_on_pitch) y xrv_pitcher directo.
   - Genera predicciones Out-Of-Fold (OOF) mediante GroupKFold por pitcher_anon_id
     (con fallback a KFold determinista si no hay suficientes lanzadores).
   - Registra de forma auditable la asignación de pliegues en self.oof_fold_assignment_.
   - Meta-regresor lineal con restricción de signos físicos / no negatividad (Ridge con
     coeficientes no negativos) hacia xrv_pitcher evaluado en Canal Stuff puro.
   - Sintoniza el parámetro de regularización alpha en una rejilla ([0.01, 0.1, 1.0, 10.0, 100.0])
     mediante validación cruzada interna guardando self.best_alpha_.
   - Tras calibrar el combinador sobre OOF, reajusta los 8 submodelos sobre todo el set
     de entrenamiento del canal Stuff para inferencia.

3. Recálculo Dinámico de Diferenciales de Arsenal por Densidad con Invarianza de Lote:
   - Al traducir el movimiento a target_rho (sabor neutral 1.000 o sabor CDMX), los
     diferenciales ivb_diff_vs_fb y hb_diff_vs_fb se recalculan con respecto a la
     recta primaria de referencia del lanzador congelada en fit (self.primary_fb_ref_dict_)
     o reconstruida fila a fila a partir de sus propios diferenciales, garantizando
     que la predicción de una fila sea 100 % idéntica si se evalúa sola o en un lote.

4. Dos Sabores con la Misma Regla de Escala Neutral (Score+):
   - Escala estándar Score+ = 100.0 + 10.0 * (pred - mu_neutral) / sigma_neutral,
     donde (mu_neutral, sigma_neutral) se calibran sobre el conjunto de referencia LMB
     evaluado a nivel del mar (rho_hat = 1.000).
   - stuff_plus_neutral: Evalúa el lanzamiento tras traducir su movimiento y arsenal a rho_0 = 1.000.
   - stuff_plus_cdmx: Evalúa el lanzamiento traduciendo su movimiento y arsenal a la densidad del
     estadio de altura (rho_cdmx, calculada empíricamente como la mediana de Extreme Altitude
     ~0.852 en LMB 2024-2025 o especificada manualmente), usando ESTRICTAMENTE la misma
     escala (mu_neutral, sigma_neutral) para cuantificar la pérdida/ganancia aerodinámica neta.
   - delta_stuff_cdmx: stuff_plus_cdmx - stuff_plus_neutral.

5. Explicabilidad SHAP (TreeSHAP exacto nativo de LightGBM):
   - Descompone la predicción del pitcheo mediante booster_.predict(X_row, pred_contrib=True),
     entregando base_value, shap_values por feature y rankings de factores positivos y negativos.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

from src.config import FASTBALL_TYPES, TARGET_COLS
from src.io import assert_no_leakage
from src.models.movement import PhysicsMovementTranslator

# ==============================================================================
# 1. Definición de Features Autorizadas (docs/CONTRACT.md §1 y TASKS.md HACK-010)
# ==============================================================================

STUFF_FEATURES: list[str] = [
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
    "PitcherThrows",
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "tunnel_dist_release_in",
    "tunnel_sim_release",
    "tunnel_dist_170ms_in",
    "tunnel_sim_170ms",
    "pitch_usage_pct",
]
VALID_STUFF_FEATURES = frozenset(STUFF_FEATURES)

# Targets binarios oficiales de resultado
BINARY_TARGETS = [c for c in TARGET_COLS if c != "xrv_pitcher"]


def assert_valid_stuff_features(feature_cols: Sequence[str]) -> None:
    """
    Valida estrictamente que una lista de features para StuffPlusModel cumpla con
    docs/CONTRACT.md §1 y TASKS.md (HACK-010):
    1. Llama a assert_no_leakage(feature_cols, 'stuff') para prohibir targets de juego,
       resultados directos, IDs y variables de plato/conteo.
    2. Aplica una whitelist estricta: cualquier columna que no pertenezca a STUFF_FEATURES
       (incluyendo VertApprAngle, HorzApprAngle, HorzApprAngle_norm, rho_hat, cualquier PlateLoc*,
       Balls, Strikes, count, Outs, BatterSide, is_same_hand, prev_pitch_id) lanza ValueError.

    Parameters
    ----------
    feature_cols : Sequence[str]
        Lista de nombres de features que se planea usar.

    Raises
    ------
    ValueError
        Si se detecta cualquier columna prohibida o fuera de STUFF_FEATURES.
    """
    feature_list = list(feature_cols)
    assert_no_leakage(feature_list, "stuff")

    unauthorized = [c for c in feature_list if c not in VALID_STUFF_FEATURES]
    if unauthorized:
        raise ValueError(
            f"Features no autorizadas detectadas en StuffPlusModel: {unauthorized}. "
            f"StuffPlusModel solo admite variables físicas de la lista autorizada STUFF_FEATURES: {STUFF_FEATURES}"
        )


def _prepare_stuff_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_categories: dict[str, list[Any]] | None = None,
    feature_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    """
    Prepara la matriz de features para LightGBM asegurando tipos de datos compatibles
    (categorías registradas, booleanos como enteros y numéricos sin NaN ni infinitos).
    """
    X = pd.DataFrame(index=df.index)
    stored_categories: dict[str, list[Any]] = {}
    medians = feature_medians or {}

    cat_cols = {"PitcherThrows"}

    for col in feature_cols:
        if col not in df.columns:
            # Fallback defensivo si alguna feature derivable falta en tiempo de predicción
            if col == "TotalInducedBreak" and "InducedVertBreak" in df.columns and "HorzBreak_norm" in df.columns:
                ivb = pd.to_numeric(df["InducedVertBreak"], errors="coerce").fillna(0.0)
                hb = pd.to_numeric(df["HorzBreak_norm"], errors="coerce").fillna(0.0)
                X[col] = np.sqrt(ivb**2 + hb**2).astype(np.float64)
                continue
            elif col == "BreakAngle_deg" and "InducedVertBreak" in df.columns and "HorzBreak_norm" in df.columns:
                ivb = pd.to_numeric(df["InducedVertBreak"], errors="coerce").fillna(0.0)
                hb = pd.to_numeric(df["HorzBreak_norm"], errors="coerce").fillna(0.0)
                X[col] = np.rad2deg(np.arctan2(hb, ivb)).astype(np.float64)
                continue
            elif col == "HorzBreak_norm" and "HorzBreak" in df.columns and "PitcherThrows" in df.columns:
                hand_sign = np.where(df["PitcherThrows"].astype(str).str.upper().str.startswith("R"), 1.0, -1.0)
                X[col] = pd.to_numeric(df["HorzBreak"], errors="coerce").fillna(0.0) * hand_sign
                continue
            elif col == "RelSide_norm" and "RelSide" in df.columns and "PitcherThrows" in df.columns:
                hand_sign = np.where(df["PitcherThrows"].astype(str).str.upper().str.startswith("R"), 1.0, -1.0)
                X[col] = pd.to_numeric(df["RelSide"], errors="coerce").fillna(0.0) * hand_sign
                continue
            elif col == "HorzRelAngle_norm" and "HorzRelAngle" in df.columns and "PitcherThrows" in df.columns:
                hand_sign = np.where(df["PitcherThrows"].astype(str).str.upper().str.startswith("R"), 1.0, -1.0)
                X[col] = pd.to_numeric(df["HorzRelAngle"], errors="coerce").fillna(0.0) * hand_sign
                continue
            elif col in medians:
                X[col] = float(medians[col])
                continue
            else:
                raise ValueError(f"Falta columna requerida '{col}' en DataFrame de entrada.")

        series = df[col]

        if col in cat_cols:
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
            default_val = float(medians.get(col, 0.0))
            X[col] = pd.to_numeric(series, errors="coerce").fillna(default_val).astype(np.float64)

    return X, (cat_categories if cat_categories is not None else stored_categories)


# ==============================================================================
# 2. Submodelo Stuff+ (StuffPlusModel)
# ==============================================================================

class StuffPlusModel:
    """
    Submodelo Stuff+ Multi-Target en Sabores Neutral y CDMX / Entorno.
    
    Estima la calidad intrínseca del pitcheo a partir exclusivamente de sus características
    físicas de soltura, movimiento aerodinámico y arsenal relacional, sin acceso alguno a
    ubicación en el plato ni conteo.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        movement_translator: Any | None = None,
    ) -> None:
        self.feature_cols: list[str] = list(feature_cols) if feature_cols is not None else list(STUFF_FEATURES)
        assert_valid_stuff_features(self.feature_cols)

        self.movement_translator: Any | None = movement_translator
        self.movement_translator_: Any | None = movement_translator
        self.is_fitted_: bool = False
        self.submodels_: dict[str, Any] = {}
        self.booster_: Any | None = None
        self.meta_ridge_: Ridge | None = None
        self.cat_categories_: dict[str, list[Any]] = {}
        self.feature_medians_: dict[str, float] = {}
        self.mu_neutral_: float = 0.0
        self.sigma_neutral_: float = 1.0
        # Mediana empírica en Extreme Altitude (~0.852 en LMB 2024-2025; agrupa múltiples parques de altura)
        self.rho_extreme_median_: float = 0.852
        # Registro auditable de predicciones Out-Of-Fold (OOF)
        self.oof_fold_assignment_: np.ndarray | None = None
        self.best_alpha_: float = 1.0
        # Diccionario congelado de referencia de recta primaria por lanzador
        self.primary_fb_ref_dict_: dict[Any, dict[str, Any]] = {}

    def _fit_primary_fb_reference(self, train_df: pd.DataFrame) -> None:
        """
        Construye y guarda la referencia congelada de la recta primaria por lanzador
        a nivel del mar (rho_hat = 1.000) para garantizar que predict_scores sea
        100 % invariante al tamaño del lote (batch vs fila aislada).
        """
        self.primary_fb_ref_dict_ = {}
        if "pitcher_anon_id" not in train_df.columns or "pitch_id" not in train_df.columns:
            return

        has_primary = "primary_fb_id" in train_df.columns
        if has_primary:
            fb_mask = train_df["pitch_id"] == train_df["primary_fb_id"]
        else:
            fb_mask = train_df["pitch_id"].isin(FASTBALL_TYPES)

        if not fb_mask.any():
            return

        fb_df = train_df.loc[fb_mask].copy()
        if self.movement_translator_ is not None and hasattr(self.movement_translator_, "translate_movement"):
            fb_sea_level = self.movement_translator_.translate_movement(fb_df, target_rho=1.0)
        else:
            fb_sea_level = fb_df.copy()
            fb_sea_level["rho_hat"] = 1.0

        num_cols = [
            "RelSpeed", "SpinRate", "SpinAxis_sin", "SpinAxis_cos",
            "RelHeight", "RelSide_norm", "Extension", "VertRelAngle",
            "HorzRelAngle_norm", "arm_slot_deg", "InducedVertBreak",
            "HorzBreak_norm", "ZoneSpeed",
        ]
        available_num = [c for c in num_cols if c in fb_sea_level.columns]
        has_year = "year" in fb_sea_level.columns

        # Referencia general por pitcher_anon_id
        for p_id, p_grp in fb_sea_level.groupby("pitcher_anon_id"):
            row_dict: dict[str, Any] = {"rho_hat": 1.0}
            for c in available_num:
                row_dict[c] = float(pd.to_numeric(p_grp[c], errors="coerce").mean())
            if "PitcherThrows" in p_grp.columns:
                row_dict["PitcherThrows"] = p_grp["PitcherThrows"].mode().iloc[0]
            self.primary_fb_ref_dict_[p_id] = row_dict

        # Referencia específica por (pitcher_anon_id, year)
        if has_year:
            for (p_id, y_val), py_grp in fb_sea_level.groupby(["pitcher_anon_id", "year"]):
                row_dict = {"rho_hat": 1.0}
                for c in available_num:
                    row_dict[c] = float(pd.to_numeric(py_grp[c], errors="coerce").mean())
                if "PitcherThrows" in py_grp.columns:
                    row_dict["PitcherThrows"] = py_grp["PitcherThrows"].mode().iloc[0]
                self.primary_fb_ref_dict_[(p_id, int(y_val))] = row_dict

    def _recompute_arsenal_diffs_at_density(
        self,
        translated_df: pd.DataFrame,
        target_rho: float = 1.0,
        translator: Any | None = None,
        orig_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Recalcula ivb_diff_vs_fb y hb_diff_vs_fb contra la recta primaria del lanzador
        traducida a la misma densidad target_rho.

        Garantiza invariancia estricta entre lotes y filas individuales: la predicción
        para una fila no depende de qué otras filas acompañan al lanzamiento en el DataFrame.
        """
        out = translated_df.copy()
        n = len(out)
        if n == 0:
            return out

        trans = translator if translator is not None else self.movement_translator_
        if trans is None or not hasattr(trans, "translate_movement"):
            return out

        src = orig_df if orig_df is not None else out
        ref_dict = getattr(self, "primary_fb_ref_dict_", {})

        fb_ref_rows: list[dict[str, Any]] = []
        has_pitcher = "pitcher_anon_id" in src.columns
        has_year = "year" in src.columns

        for idx in range(n):
            row = src.iloc[idx]
            p_id = row.get("pitcher_anon_id", None) if has_pitcher else None
            y_val = int(row.get("year", 0)) if has_year and not pd.isna(row.get("year", None)) else None

            # 1. Buscar en diccionario de referencia congelado en fit
            proto = None
            if ref_dict:
                if p_id is not None and y_val is not None and (p_id, y_val) in ref_dict:
                    proto = ref_dict[(p_id, y_val)].copy()
                elif p_id is not None and p_id in ref_dict:
                    proto = ref_dict[p_id].copy()

            if proto is not None:
                fb_row = proto.copy()
            else:
                # 2. Fallback fila por fila: reconstruir prototipo a partir de sus propios diferenciales
                fb_row = {}
                velo_diff = row.get("velo_diff_vs_fb", 0.0)
                ivb_diff = row.get("ivb_diff_vs_fb", 0.0)
                hb_diff = row.get("hb_diff_vs_fb", 0.0)

                fb_row["RelSpeed"] = float(row.get("RelSpeed", 90.0)) - float(velo_diff if not pd.isna(velo_diff) else 0.0)
                fb_row["InducedVertBreak"] = float(row.get("InducedVertBreak", 10.0)) - float(ivb_diff if not pd.isna(ivb_diff) else 0.0)
                fb_row["HorzBreak_norm"] = float(row.get("HorzBreak_norm", 0.0)) - float(hb_diff if not pd.isna(hb_diff) else 0.0)
                fb_row["ZoneSpeed"] = float(row.get("ZoneSpeed", 84.0))
                fb_row["SpinRate"] = float(row.get("SpinRate", 2200.0))
                fb_row["SpinAxis_sin"] = float(row.get("SpinAxis_sin", 0.0))
                fb_row["SpinAxis_cos"] = float(row.get("SpinAxis_cos", 1.0))
                fb_row["RelHeight"] = float(row.get("RelHeight", 5.8))
                fb_row["RelSide_norm"] = float(row.get("RelSide_norm", 1.8))
                fb_row["Extension"] = float(row.get("Extension", 6.0))
                fb_row["VertRelAngle"] = float(row.get("VertRelAngle", -2.0))
                fb_row["HorzRelAngle_norm"] = float(row.get("HorzRelAngle_norm", 0.0))
                fb_row["arm_slot_deg"] = float(row.get("arm_slot_deg", 50.0))
                fb_row["PitcherThrows"] = row.get("PitcherThrows", "Right")

            if "HorzBreak_norm" in fb_row:
                p_throws = str(fb_row.get("PitcherThrows", "Right")).upper()
                h_sign = 1.0 if p_throws.startswith("R") else -1.0
                fb_row["HorzBreak"] = fb_row["HorzBreak_norm"] * h_sign

            fb_row["rho_hat"] = 1.0
            fb_ref_rows.append(fb_row)

        fb_ref_df = pd.DataFrame(fb_ref_rows, index=out.index)
        fb_translated = trans.translate_movement(fb_ref_df, target_rho=float(target_rho))

        out["ivb_diff_vs_fb"] = (
            out["InducedVertBreak"].to_numpy(dtype=np.float64)
            - fb_translated["InducedVertBreak"].to_numpy(dtype=np.float64)
        ).astype(np.float64)
        out["hb_diff_vs_fb"] = (
            out["HorzBreak_norm"].to_numpy(dtype=np.float64)
            - fb_translated["HorzBreak_norm"].to_numpy(dtype=np.float64)
        ).astype(np.float64)

        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        movement_translator: PhysicsMovementTranslator | None = None,
    ) -> "StuffPlusModel":
        """
        Entrena los 8 submodelos multi-target y el meta-regresor lineal del Canal Stuff,
        utilizando exclusivamente observaciones de No Altitude para neutralizar totalmente
        el carry del parque, predicciones Out-Of-Fold (OOF) con validación cruzada y
        referencia congelada de recta primaria.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame de entrenamiento.
        movement_translator : PhysicsMovementTranslator | None, optional
            Traductor físico de movimiento para evaluación de altitud. Si es None, se entrena
            uno internamente sobre train_df.

        Returns
        -------
        StuffPlusModel
            Instancia ajustada (self).
        """
        assert_valid_stuff_features(self.feature_cols)

        df_train = train_df.copy()
        if "year" in df_train.columns:
            mask_year = df_train["year"] <= 2025
            if mask_year.any():
                df_train = df_train.loc[mask_year].copy()

        if len(df_train) == 0:
            raise ValueError("train_df está vacío tras filtrar a year <= 2025.")

        # 1. Traductor físico de movimiento (Etapa A)
        if movement_translator is not None:
            self.movement_translator_ = movement_translator
        elif self.movement_translator is not None:
            self.movement_translator_ = self.movement_translator
        else:
            self.movement_translator_ = PhysicsMovementTranslator().fit(df_train)

        # 2. Mediana empírica de rho_hat en Extreme Altitude
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

        # 3. Referencia congelada de recta primaria a nivel del mar para invariancia de lote
        self._fit_primary_fb_reference(df_train)

        # 4. Neutralización del Canal Stuff frente al parque en altitud:
        # Ajustamos los 8 submodelos y el combinador utilizando exclusivamente observaciones
        # de 'No Altitude' para que ninguna etiqueta inflada por el carry del parque
        # (ni de Extreme Altitude ni de Medium Altitude) contamine la física pura del pitcheo.
        if "altitude_category" in df_train.columns:
            no_alt_mask = df_train["altitude_category"] == "No Altitude"
            if no_alt_mask.sum() >= 20:
                df_stuff_train = df_train.loc[no_alt_mask].copy()
            else:
                non_extreme = df_train["altitude_category"] != "Extreme Altitude"
                if non_extreme.sum() >= 20:
                    df_stuff_train = df_train.loc[non_extreme].copy()
                else:
                    df_stuff_train = df_train.copy()
        else:
            df_stuff_train = df_train.copy()

        # 5. Medianas defensivas para imputación en predicción calculadas sobre el canal Stuff
        self.feature_medians_ = {}
        for col in self.feature_cols:
            if col in df_stuff_train.columns and col != "PitcherThrows":
                med = pd.to_numeric(df_stuff_train[col], errors="coerce").median()
                self.feature_medians_[col] = float(med) if not pd.isna(med) else 0.0
            else:
                self.feature_medians_[col] = 0.0

        # 6. Matriz de características X_train
        X_train, self.cat_categories_ = _prepare_stuff_features(
            df_stuff_train, self.feature_cols, feature_medians=self.feature_medians_
        )
        n_samples = len(X_train)

        # 7. Partición para predicciones Out-Of-Fold (OOF)
        if "pitcher_anon_id" in df_stuff_train.columns:
            n_pitchers = df_stuff_train["pitcher_anon_id"].nunique()
            if n_pitchers >= 3:
                n_splits = min(5, n_pitchers)
                gkf = GroupKFold(n_splits=n_splits)
                splits = list(gkf.split(X_train, groups=df_stuff_train["pitcher_anon_id"]))
            else:
                kf = KFold(n_splits=min(5, max(2, n_samples // 10)), shuffle=True, random_state=42)
                splits = list(kf.split(X_train))
        else:
            kf = KFold(n_splits=min(5, max(2, n_samples // 10)), shuffle=True, random_state=42)
            splits = list(kf.split(X_train))

        self.oof_fold_assignment_ = np.full(n_samples, -1, dtype=int)
        oof_preds: dict[str, np.ndarray] = {tgt: np.zeros(n_samples, dtype=np.float64) for tgt in TARGET_COLS}

        # 8. Estimación Out-Of-Fold de los 8 submodelos
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            self.oof_fold_assignment_[val_idx] = fold_idx
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            min_child_fold = min(20, max(2, len(X_tr) // 5))

            for tgt in BINARY_TARGETS:
                if tgt not in df_stuff_train.columns:
                    raise ValueError(f"Falta columna de target '{tgt}' en train_df.")
                y_b_tr = pd.to_numeric(df_stuff_train[tgt].iloc[train_idx], errors="coerce").fillna(0.0).to_numpy()

                if len(np.unique(y_b_tr)) < 2:
                    oof_preds[tgt][val_idx] = float(np.mean(y_b_tr))
                else:
                    clf_fold = LGBMClassifier(
                        random_state=42,
                        n_estimators=60,
                        learning_rate=0.05,
                        min_child_samples=min_child_fold,
                        verbose=-1,
                    )
                    clf_fold.fit(X_tr, y_b_tr)
                    p_val = clf_fold.predict_proba(X_val)[:, 1].astype(np.float64)
                    oof_preds[tgt][val_idx] = np.clip(p_val, 0.0, 1.0)

            if "xrv_pitcher" not in df_stuff_train.columns:
                raise ValueError("Falta columna de target 'xrv_pitcher' en train_df.")
            y_xrv_tr = pd.to_numeric(df_stuff_train["xrv_pitcher"].iloc[train_idx], errors="coerce").fillna(0.0).to_numpy()
            reg_xrv_fold = LGBMRegressor(
                random_state=42,
                n_estimators=60,
                learning_rate=0.05,
                min_child_samples=min_child_fold,
                verbose=-1,
            )
            reg_xrv_fold.fit(X_tr, y_xrv_tr)
            oof_preds["xrv_pitcher"][val_idx] = reg_xrv_fold.predict(X_val).astype(np.float64)

        # 9. Construcción de matriz OOF con restricción de signos físicos:
        # +P(whiff), +P(chase), +P(called_strike), +P(groundball),
        # +P(weak_contact), -P(hard_contact), +P(out_on_pitch) y +xrv_pitcher
        M_oof = np.column_stack([
            oof_preds["target_whiff"],
            oof_preds["target_chase"],
            oof_preds["target_called_strike"],
            oof_preds["target_groundball"],
            oof_preds["target_weak_contact"],
            -oof_preds["target_hard_contact"],  # Inversión física de castigo
            oof_preds["target_out_on_pitch"],
            oof_preds["xrv_pitcher"],
        ])
        y_xrv_all = pd.to_numeric(df_stuff_train["xrv_pitcher"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        # 10. Selección de regularización mediante Validación Cruzada sobre la rejilla de alphas
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = 1.0
        best_mse = float("inf")

        for a in alphas:
            fold_mses: list[float] = []
            for train_idx, val_idx in splits:
                ridge_cv = Ridge(alpha=a, positive=True)
                ridge_cv.fit(M_oof[train_idx], y_xrv_all[train_idx])
                pred_val = ridge_cv.predict(M_oof[val_idx])
                fold_mses.append(float(np.mean((y_xrv_all[val_idx] - pred_val) ** 2)))
            avg_mse = float(np.mean(fold_mses))
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = a

        self.best_alpha_ = float(best_alpha)

        # Ajuste final del combinador sobre todas las predicciones OOF
        self.meta_ridge_ = Ridge(alpha=self.best_alpha_, positive=True)
        self.meta_ridge_.fit(M_oof, y_xrv_all)
        if np.all(self.meta_ridge_.coef_ <= 0):
            self.meta_ridge_.coef_[-1] = 1.0

        # 11. Reajuste de los 8 submodelos sobre todo df_stuff_train para producción e inferencia
        min_child_full = min(20, max(2, n_samples // 5))
        self.submodels_ = {}

        for tgt in BINARY_TARGETS:
            y_b = pd.to_numeric(df_stuff_train[tgt], errors="coerce").fillna(0.0).to_numpy()
            if len(np.unique(y_b)) < 2:
                const_val = float(np.mean(y_b))
                self.submodels_[tgt] = ("constant", const_val)
            else:
                clf = LGBMClassifier(
                    random_state=42,
                    n_estimators=60,
                    learning_rate=0.05,
                    min_child_samples=min_child_full,
                    verbose=-1,
                )
                clf.fit(X_train, y_b)
                self.submodels_[tgt] = clf

        reg_xrv = LGBMRegressor(
            random_state=42,
            n_estimators=60,
            learning_rate=0.05,
            min_child_samples=min_child_full,
            verbose=-1,
        )
        reg_xrv.fit(X_train, y_xrv_all)
        self.submodels_["xrv_pitcher"] = reg_xrv
        self.booster_ = reg_xrv.booster_

        # 12. Calibración de Escala Neutral (rho_0 = 1.000) con recálculo dinámico de arsenal
        df_neutral_train = self.movement_translator_.translate_movement(df_stuff_train, target_rho=1.000)
        df_neutral_train = self._recompute_arsenal_diffs_at_density(
            df_neutral_train, target_rho=1.000, translator=self.movement_translator_, orig_df=df_stuff_train
        )
        pred_neutral_train = self._predict_stuff_value(df_neutral_train)
        self.mu_neutral_ = float(np.mean(pred_neutral_train))
        std_val = float(np.std(pred_neutral_train, ddof=0))
        self.sigma_neutral_ = float(max(std_val, 1e-6))

        self.is_fitted_ = True
        return self

    def _predict_submodels(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Genera las predicciones individuales de cada uno de los 8 submodelos.
        """
        X, _ = _prepare_stuff_features(
            df,
            self.feature_cols,
            cat_categories=self.cat_categories_,
            feature_medians=self.feature_medians_,
        )
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
        preds["xrv_pitcher"] = reg_xrv.predict(X).astype(np.float64)
        return preds

    def _predict_stuff_value(self, df: pd.DataFrame) -> np.ndarray:
        """
        Evalúa el índice final esperado de calidad física (Canal Stuff puro)
        utilizando el meta-regresor con signos físicos restringidos.
        """
        raw_preds = self._predict_submodels(df)
        M = np.column_stack([
            raw_preds["target_whiff"],
            raw_preds["target_chase"],
            raw_preds["target_called_strike"],
            raw_preds["target_groundball"],
            raw_preds["target_weak_contact"],
            -raw_preds["target_hard_contact"],
            raw_preds["target_out_on_pitch"],
            raw_preds["xrv_pitcher"],
        ])
        return self.meta_ridge_.predict(M).astype(np.float64)

    def predict_scores(
        self,
        df: pd.DataFrame,
        movement_translator: PhysicsMovementTranslator | None = None,
        rho_cdmx: float | None = None,
    ) -> pd.DataFrame:
        """
        Calcula los scores Stuff+ en sabores neutral y CDMX / entorno, y su diferencia,
        con recálculo dinámico de diferenciales de arsenal por densidad.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de lanzamientos.
        movement_translator : PhysicsMovementTranslator | None, optional
            Traductor físico de movimiento para evaluación de altitud. Si es None, utiliza
            el traductor ajustado en fit().
        rho_cdmx : float | None, default=None
            Densidad relativa objetivo para sabor CDMX / Extreme Altitude. Si es None,
            utiliza la mediana empírica estimada de los datos de entrenamiento
            (self.rho_extreme_median_ ~0.852 en Extreme Altitude de LMB).

        Returns
        -------
        pd.DataFrame
            Copia de df con stuff_plus_neutral, stuff_plus_cdmx y delta_stuff_cdmx en float64.
        """
        if not self.is_fitted_:
            raise RuntimeError("StuffPlusModel debe entrenarse con fit() antes de predict_scores().")

        translator = movement_translator if movement_translator is not None else self.movement_translator_
        target_rho = float(rho_cdmx) if rho_cdmx is not None else float(self.rho_extreme_median_)

        # 1. Sabor Neutral (rho_hat = 1.000) con arsenal recalculado
        df_neutral = translator.translate_movement(df, target_rho=1.000)
        df_neutral = self._recompute_arsenal_diffs_at_density(
            df_neutral, target_rho=1.000, translator=translator, orig_df=df
        )
        pred_neutral = self._predict_stuff_value(df_neutral)
        stuff_plus_neutral = 100.0 + 10.0 * (pred_neutral - self.mu_neutral_) / self.sigma_neutral_

        # 2. Sabor CDMX / Entorno (rho_hat = target_rho con ESTRICTAMENTE LA MISMA ESCALA NEUTRAL)
        df_cdmx = translator.translate_movement(df, target_rho=target_rho)
        df_cdmx = self._recompute_arsenal_diffs_at_density(
            df_cdmx, target_rho=target_rho, translator=translator, orig_df=df
        )
        pred_cdmx = self._predict_stuff_value(df_cdmx)
        stuff_plus_cdmx = 100.0 + 10.0 * (pred_cdmx - self.mu_neutral_) / self.sigma_neutral_

        # 3. Delta Stuff+ por entorno
        delta_stuff = stuff_plus_cdmx - stuff_plus_neutral

        out = df.copy()
        out["stuff_plus_neutral"] = stuff_plus_neutral.astype(np.float64)
        out["stuff_plus_cdmx"] = stuff_plus_cdmx.astype(np.float64)
        out["delta_stuff_cdmx"] = delta_stuff.astype(np.float64)

        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las columnas exactas TARGET_COLS para compatibilidad
        directa con evaluate_against_baseline de HACK-007 y suites de validación.
        """
        if not self.is_fitted_:
            raise RuntimeError("StuffPlusModel debe entrenarse con fit() antes de predict().")

        raw_preds = self._predict_submodels(df)
        raw_preds["xrv_pitcher"] = self._predict_stuff_value(df)

        return pd.DataFrame(raw_preds, index=df.index)[TARGET_COLS].astype(np.float64)

    def explain_pitch(self, df_row: pd.DataFrame | pd.Series) -> dict[str, Any]:
        """
        Calcula la explicabilidad local exacta TreeSHAP para un lanzamiento individual
        mediante booster_.predict(X_row, pred_contrib=True).

        Parameters
        ----------
        df_row : pd.DataFrame | pd.Series
            Fila con los atributos del lanzamiento. Si contiene múltiples filas, se utiliza la primera.

        Returns
        -------
        dict[str, Any]
            Diccionario con:
            - 'base_value': Valor base en escala Stuff+.
            - 'shap_values': dict con la contribución SHAP de cada feature en puntos de Stuff+.
            - 'top_positive_features': lista de tuplas (feature, contribución) con impacto positivo.
            - 'top_negative_features': lista de tuplas (feature, contribución) con impacto negativo.
            - 'raw_base_value': Valor base en escala directa xrv_pitcher.
            - 'raw_shap_values': dict con las contribuciones TreeSHAP en escala xrv_pitcher.
        """
        if not self.is_fitted_:
            raise RuntimeError("StuffPlusModel debe entrenarse con fit() antes de explain_pitch().")

        if isinstance(df_row, pd.Series):
            row_df = df_row.to_frame().T
        elif isinstance(df_row, pd.DataFrame):
            row_df = df_row.iloc[[0]].copy()
        else:
            raise ValueError("df_row debe ser un pd.DataFrame o pd.Series.")

        X_row, _ = _prepare_stuff_features(
            row_df,
            self.feature_cols,
            cat_categories=self.cat_categories_,
            feature_medians=self.feature_medians_,
        )

        contrib = self.booster_.predict(X_row, pred_contrib=True)[0]
        base_val_raw = float(contrib[-1])
        feat_contribs_raw = contrib[:-1]

        scale = 10.0 / self.sigma_neutral_
        base_value_plus = float(100.0 + scale * (base_val_raw - self.mu_neutral_))
        shap_values_plus = {
            feat: float(scale * c) for feat, c in zip(self.feature_cols, feat_contribs_raw)
        }

        # Ordenar features por contribución en Stuff+
        sorted_desc = sorted(shap_values_plus.items(), key=lambda x: x[1], reverse=True)
        top_pos = [(f, float(v)) for f, v in sorted_desc if v > 0]
        if not top_pos:
            top_pos = [(f, float(v)) for f, v in sorted_desc[:3]]

        sorted_asc = sorted(shap_values_plus.items(), key=lambda x: x[1])
        top_neg = [(f, float(v)) for f, v in sorted_asc if v < 0]
        if not top_neg:
            top_neg = [(f, float(v)) for f, v in sorted_asc[:3]]

        return {
            "base_value": base_value_plus,
            "shap_values": shap_values_plus,
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
            "raw_base_value": base_val_raw,
            "raw_shap_values": {
                feat: float(c) for feat, c in zip(self.feature_cols, feat_contribs_raw)
            },
        }

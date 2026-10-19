"""
src/features/identity.py
========================
Identidad de pitcheo invariante a la altitud (HACK-004) para Diablos AeroVision.

TrackMan reetiqueta los tipos de lanzamiento en altitud porque su clasificador
depende del quiebre inducido (InducedVertBreak y HorzBreak), los cuales se
debilitan de forma natural con una menor densidad del aire (rho).

Este módulo define clusters de pitcheo por (pitcher_anon_id, year) utilizando
EXCLUSIVAMENTE características físicas en el momento del release que la densidad
del aire NO altera:
- RelSpeed (mph) y diferencial relativo a la velocidad máxima (percentil 90) del pitcher (rel_velo_delta)
- SpinRate (rpm)
- SpinAxis_sin y SpinAxis_cos (componentes cíclicas del eje de giro normalizadas por mano)
- RelHeight (pies)
- RelSide_norm (pies, normalizado por mano)
- Extension (pies)

REGLA CRÍTICA DE INVARIANCIA:
Está 100% prohibido usar variables de quiebre o movimiento (InducedVertBreak,
VertBreak, HorzBreak, HorzBreak_norm, TotalInducedBreak, BreakAngle_deg,
SpinToBreakRatio, pfxx, pfxz) para determinar la identidad del pitcheo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture

# Mapeo oficial de recodificación de AutoPitchType
PITCH_TYPE_RECODE: dict[str, str] = {
    "OneSeamFastBall": "Sinker",
    "TwoSeamFastBall": "Sinker",
    "Knuckleball": "Other",
}

# Tipos de pitcheo válidos canónicos (docs/CONTRACT.md HACK-004)
VALID_PITCH_IDS: list[str] = [
    "Four-Seam",
    "Sinker",
    "Cutter",
    "Slider",
    "Sweeper",
    "Curveball",
    "Changeup",
    "Splitter",
    "Other",
]

# Columnas invariantes de release para clustering y clasificación
INVARIANT_RELEASE_COLS: list[str] = [
    "RelSpeed",
    "SpinRate",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "RelHeight",
    "RelSide_norm",
    "Extension",
]

CLASSIFIER_FEATURE_COLS: list[str] = [
    "RelSpeed",
    "rel_velo_delta",
    "SpinRate",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "RelHeight",
    "RelSide_norm",
    "Extension",
]

CLUSTER_FEATURE_COLS: list[str] = [
    "rel_velo_delta",
    "SpinRate",
    "SpinAxis_sin",
    "SpinAxis_cos",
    "RelHeight",
    "RelSide_norm",
    "Extension",
]

# Columnas estrictamente prohibidas para la asignación de identidad
FORBIDDEN_BREAK_COLS: frozenset[str] = frozenset([
    "InducedVertBreak",
    "VertBreak",
    "HorzBreak",
    "HorzBreak_norm",
    "TotalInducedBreak",
    "BreakAngle_deg",
    "SpinToBreakRatio",
    "pfxx",
    "pfxz",
])


class _ConstantPitchPredictor:
    """Fallback predictor cuando hay solo 1 clase en No Altitude."""

    def __init__(self, constant_label: str = "Four-Seam") -> None:
        self.constant_label = constant_label

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n = len(X)
        return np.array([self.constant_label] * n, dtype=object)


class _ReleaseKinematicsFallbackPredictor:
    """
    Predictor heurístico de respaldo basado EXCLUSIVAMENTE en cinemática de release.
    Se utiliza cuando no existen lanzamientos en 'No Altitude' para entrenar un modelo,
    garantizando 0% de data leakage desde juegos con altitud y 0% de uso de quiebre.
    """

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            velo_delta = X["rel_velo_delta"].to_numpy()
            spin_rate = X["SpinRate"].to_numpy()
            spin_sin = X["SpinAxis_sin"].to_numpy()
            spin_cos = X["SpinAxis_cos"].to_numpy()
        else:
            # Índice 1: rel_velo_delta, 2: SpinRate, 3: SpinAxis_sin, 4: SpinAxis_cos
            velo_delta = X[:, 1]
            spin_rate = X[:, 2]
            spin_sin = X[:, 3]
            spin_cos = X[:, 4]

        n = len(velo_delta)
        preds = np.full(n, "Four-Seam", dtype=object)

        for i in range(n):
            vd = velo_delta[i]
            sr = spin_rate[i]
            s_sin = spin_sin[i]
            s_cos = spin_cos[i]

            if vd < -8.0 and s_cos < -0.2:
                preds[i] = "Curveball"
            elif vd < -5.0 and sr < 1650.0:
                preds[i] = "Splitter"
            elif vd < -5.0 and s_sin > 0.3 and sr < 2100.0:
                preds[i] = "Changeup"
            elif vd < -4.0 and s_sin < -0.4:
                preds[i] = "Sweeper"
            elif vd < -3.5 and s_sin <= 0.1 and s_cos < 0.4:
                preds[i] = "Slider"
            elif vd >= -5.0 and s_sin < -0.1:
                preds[i] = "Cutter"
            elif vd >= -4.5 and s_sin > 0.4:
                preds[i] = "Sinker"
            else:
                preds[i] = "Four-Seam"

        return preds


def _clean_pitch_type_label(s: pd.Series) -> pd.Series:
    """Normaliza y recodifica etiquetas crudas de AutoPitchType a categorías válidas."""
    cleaned = s.astype(str).replace(PITCH_TYPE_RECODE)
    cleaned = cleaned.apply(lambda x: x if x in VALID_PITCH_IDS else "Other")
    return cleaned


def _ensure_invariant_release_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que RelSide_norm, SpinAxis_sin y SpinAxis_cos estén presentes en df
    calculándolos únicamente a partir de la geometría de release y mano del lanzador.
    """
    if "RelSide_norm" not in df.columns or "SpinAxis_sin" not in df.columns or "SpinAxis_cos" not in df.columns:
        if "PitcherThrows" in df.columns:
            hand_sign = np.where(df["PitcherThrows"].astype(str) == "Left", -1.0, 1.0)
        else:
            rel_side_val = pd.to_numeric(df.get("RelSide", 0.0), errors="coerce").fillna(0.0)
            hand_sign = np.where(rel_side_val >= 0.0, 1.0, -1.0)

        if "RelSide_norm" not in df.columns:
            rel_side = pd.to_numeric(df.get("RelSide", 0.0), errors="coerce").fillna(0.0)
            df["RelSide_norm"] = (rel_side * hand_sign).astype("float64")

        if "SpinAxis_sin" not in df.columns or "SpinAxis_cos" not in df.columns:
            spin_axis = pd.to_numeric(df.get("SpinAxis", 0.0), errors="coerce").fillna(0.0)
            spin_rad = np.radians(spin_axis)
            df["SpinAxis_sin"] = (np.sin(spin_rad) * hand_sign).astype("float64")
            df["SpinAxis_cos"] = np.cos(spin_rad).astype("float64")

    return df


def _select_optimal_k_and_cluster(X_scaled: np.ndarray, max_k: int) -> tuple[int, np.ndarray]:
    """
    Selecciona el número óptimo de clusters K in {1, ..., max_k} evaluando
    el criterio de información bayesiano (BIC) sobre GaussianMixture(covariance_type='diag').

    Parameters
    ----------
    X_scaled : np.ndarray
        Matriz de características de release estandarizadas del pitcher.
    max_k : int
        Límite superior para el número de clusters (min(6, max(1, N // 15))).

    Returns
    -------
    tuple[int, np.ndarray]
        (best_k, labels) donde labels son los identificadores de cluster (int32).
    """
    n_samples = len(X_scaled)
    if n_samples <= 1 or max_k <= 1:
        return 1, np.zeros(n_samples, dtype=np.int32)

    n_distinct = len(np.unique(X_scaled, axis=0))
    effective_max_k = min(max_k, n_distinct)
    if effective_max_k <= 1:
        return 1, np.zeros(n_samples, dtype=np.int32)

    best_k = 1
    best_bic = float("inf")
    best_labels = np.zeros(n_samples, dtype=np.int32)

    for k in range(1, effective_max_k + 1):
        if k > n_samples:
            break
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=42,
                reg_covar=1e-3,
                max_iter=100,
            )
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_labels = gmm.predict(X_scaled).astype(np.int32)
        except Exception:
            continue

    return best_k, best_labels


def assign_invariant_pitch_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna una identidad de pitcheo (pitch_id) y un identificador de cluster (pitch_cluster_id)
    invariantes a la altitud y a las variaciones en la densidad del aire.

    Metodología:
    1. Trabaja sobre una copia (out = df.copy()).
    2. Utiliza estrictamente variables de release cinemáticas que no dependen de la densidad del aire.
    3. Calcula por (pitcher_anon_id, year) el diferencial de velocidad respecto al percentil 90
       del propio lanzador (rel_velo_delta = RelSpeed - pitcher_p90_velo).
    4. Entrena un clasificador de referencia HistGradientBoostingClassifier a nivel del mar
       usando ÚNICAMENTE filas con altitude_category == 'No Altitude' (si no existen, usa un
       predictor heurístico basado puramente en cinemática de release sin data leakage).
    5. Agrupa los lanzamientos de cada pitcher-temporada evaluando candidatos K in {1..min(6, floor(N/15))}
       mediante el criterio BIC con GaussianMixture.
    6. Asigna como pitch_id la moda de AutoPitchType en 'No Altitude' (si el cluster tiene >= 10 lanzamientos
       a nivel del mar) o la predicción modal del clasificador de referencia a nivel del mar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos con columnas de TrackMan LMB.

    Returns
    -------
    pd.DataFrame
        Copia enriquecida con pitch_cluster_id (int32) y pitch_id (tipo string de pandas sin nulos).
    """
    out = df.copy()

    # Garantizar presencia de variables cinemáticas de release
    out = _ensure_invariant_release_features(out)

    # Imputar de forma segura nulos en features de release si existen
    defaults = {
        "RelSpeed": 90.0,
        "SpinRate": 2200.0,
        "SpinAxis_sin": 0.0,
        "SpinAxis_cos": 0.0,
        "RelHeight": 5.8,
        "RelSide_norm": 1.8,
        "Extension": 6.0,
    }
    for col, default_val in defaults.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default_val).astype("float64")
        else:
            out[col] = float(default_val)

    # Claves de agrupación para pitcher y temporada
    group_cols: list[str] = []
    if "pitcher_anon_id" in out.columns:
        group_cols.append("pitcher_anon_id")
    if "year" in out.columns:
        group_cols.append("year")

    # Cálculo de pitcher_p90_velo y rel_velo_delta
    if group_cols:
        p90_velo = out.groupby(group_cols)["RelSpeed"].transform(
            lambda s: np.nanpercentile(s, 90) if s.notna().any() else 90.0
        )
    else:
        p90_velo = pd.Series(
            np.nanpercentile(out["RelSpeed"], 90) if out["RelSpeed"].notna().any() else 90.0,
            index=out.index,
        )

    p90_velo = p90_velo.fillna(90.0)
    out["rel_velo_delta"] = (out["RelSpeed"] - p90_velo).astype("float64")

    # Limpieza de AutoPitchType para entrenamiento y asignación de moda
    if "AutoPitchType" in out.columns:
        clean_auto = _clean_pitch_type_label(out["AutoPitchType"])
    else:
        clean_auto = pd.Series("Four-Seam", index=out.index)

    # 4. Entrenar clasificador de referencia a nivel del mar (ESTRICTAMENTE en 'No Altitude')
    has_alt_col = "altitude_category" in out.columns
    if has_alt_col:
        alt_series = out["altitude_category"].astype(str)
        no_alt_mask = (alt_series == "No Altitude") & clean_auto.notna() & (clean_auto != "Other")
    else:
        no_alt_mask = pd.Series(False, index=out.index)

    y_train = clean_auto.loc[no_alt_mask]
    unique_classes = np.unique(y_train) if len(y_train) > 0 else np.array([])

    if len(unique_classes) >= 2 and no_alt_mask.sum() >= 2:
        X_train = out.loc[no_alt_mask, CLASSIFIER_FEATURE_COLS].to_numpy()
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=60)
        clf.fit(X_train, y_train)
    elif len(unique_classes) == 1:
        clf = _ConstantPitchPredictor(str(unique_classes[0]))
    else:
        # No hay filas en No Altitude: usar predictor heurístico basado puramente en release
        clf = _ReleaseKinematicsFallbackPredictor()

    # Inicializar columnas objetivo
    out["pitch_cluster_id"] = np.zeros(len(out), dtype=np.int32)
    out["pitch_id"] = "Four-Seam"

    # 5. Agrupamiento y asignación de identidad por (pitcher_anon_id, year)
    if not group_cols:
        group_tuples = [(None, out)]
    else:
        group_tuples = out.groupby(group_cols, sort=False)

    for _, group in group_tuples:
        group_idx = group.index
        n_pitches = len(group)
        if n_pitches == 0:
            continue

        # Regla de selección de K mediante BIC: candidatos K in {1..min(6, floor(N/15))}
        max_k = min(6, max(1, n_pitches // 15))

        X_grp = group[CLUSTER_FEATURE_COLS].to_numpy(dtype=np.float64)

        # Estandarización robusta intra-pitcher
        grp_mean = np.mean(X_grp, axis=0)
        grp_std = np.std(X_grp, axis=0)
        grp_std_adj = np.where(grp_std < 1e-6, 1.0, grp_std)
        X_scaled = (X_grp - grp_mean) / grp_std_adj

        _, cluster_labels = _select_optimal_k_and_cluster(X_scaled, max_k)
        out.loc[group_idx, "pitch_cluster_id"] = cluster_labels

        # 6. Asignar pitch_id a cada cluster presente
        present_clusters = np.unique(cluster_labels)
        for c in present_clusters:
            c_mask = (cluster_labels == c)
            sub_c_idx = group_idx[c_mask]

            assigned_id: str | None = None
            if has_alt_col:
                sub_c_alt = out.loc[sub_c_idx, "altitude_category"].astype(str)
                no_alt_in_c = (sub_c_alt == "No Altitude")
                # Exige al menos 10 lanzamientos en No Altitude para usar la moda observada
                if no_alt_in_c.sum() >= 10:
                    modes = clean_auto.loc[sub_c_idx[no_alt_in_c]].mode()
                    if not modes.empty:
                        assigned_id = str(modes.iloc[0])

            if assigned_id is None:
                X_c_features = out.loc[sub_c_idx, CLASSIFIER_FEATURE_COLS].to_numpy()
                preds = clf.predict(X_c_features)
                modes = pd.Series(preds).mode()
                assigned_id = str(modes.iloc[0]) if not modes.empty else "Four-Seam"

            out.loc[sub_c_idx, "pitch_id"] = assigned_id

    # Garantizar tipos exactos según CONTRACT.md: int32 y tipo string de pandas sin nulos
    out["pitch_cluster_id"] = out["pitch_cluster_id"].fillna(0).astype(np.int32)
    out["pitch_id"] = out["pitch_id"].fillna("Other").astype("string")

    return out


def audit_trackman_relabeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara para los mismos lanzadores que lanzaron tanto en 'No Altitude' como
    en 'Extreme Altitude' el cambio en la distribución de tipos de pitcheo
    clasificados por TrackMan (AutoPitchType) contra la identidad física invariante (pitch_id).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos. Si no contiene 'pitch_id', se invoca
        assign_invariant_pitch_id(df).

    Returns
    -------
    pd.DataFrame
        DataFrame resumen con columnas:
        ["pitch_type", "auto_pct_no_alt", "auto_pct_extreme", "auto_diff_pp",
         "invariant_pct_no_alt", "invariant_pct_extreme", "invariant_diff_pp"]
    """
    result_cols = [
        "pitch_type",
        "auto_pct_no_alt",
        "auto_pct_extreme",
        "auto_diff_pp",
        "invariant_pct_no_alt",
        "invariant_pct_extreme",
        "invariant_diff_pp",
    ]

    if "pitch_id" not in df.columns:
        df = assign_invariant_pitch_id(df)

    if "pitcher_anon_id" not in df.columns or "altitude_category" not in df.columns:
        return pd.DataFrame(columns=result_cols)

    valid_alt = df[df["altitude_category"].notna()].copy()
    valid_alt["alt_str"] = valid_alt["altitude_category"].astype(str)

    # Identificar lanzadores con actividad en ambas altitudes extremas
    p_alts = valid_alt.groupby("pitcher_anon_id")["alt_str"].unique()
    dual_pitchers = [
        p for p, alts in p_alts.items()
        if "No Altitude" in alts and "Extreme Altitude" in alts
    ]

    if not dual_pitchers:
        return pd.DataFrame(columns=result_cols)

    sub = valid_alt[
        valid_alt["pitcher_anon_id"].isin(dual_pitchers)
        & valid_alt["alt_str"].isin(["No Altitude", "Extreme Altitude"])
    ].copy()

    if sub.empty:
        return pd.DataFrame(columns=result_cols)

    sub["auto_clean"] = _clean_pitch_type_label(sub["AutoPitchType"]) if "AutoPitchType" in sub.columns else "Four-Seam"
    sub["inv_clean"] = sub["pitch_id"].astype(str)

    # Identificar el universo de tipos de pitcheo presentes
    all_types = sorted(list(set(sub["auto_clean"].unique()).union(set(sub["inv_clean"].unique()))))
    all_types = [t for t in all_types if t not in ["nan", "None", "", "<NA>"]]

    ordered_types = [t for t in VALID_PITCH_IDS if t in all_types] + [t for t in all_types if t not in VALID_PITCH_IDS]

    mask_no = sub["alt_str"] == "No Altitude"
    mask_ext = sub["alt_str"] == "Extreme Altitude"

    n_no = mask_no.sum()
    n_ext = mask_ext.sum()

    auto_no = sub.loc[mask_no, "auto_clean"]
    auto_ext = sub.loc[mask_ext, "auto_clean"]
    inv_no = sub.loc[mask_no, "inv_clean"]
    inv_ext = sub.loc[mask_ext, "inv_clean"]

    rows = []
    for pt in ordered_types:
        auto_pct_no = (100.0 * (auto_no == pt).sum() / n_no) if n_no > 0 else 0.0
        auto_pct_ext = (100.0 * (auto_ext == pt).sum() / n_ext) if n_ext > 0 else 0.0
        auto_diff = auto_pct_ext - auto_pct_no

        inv_pct_no = (100.0 * (inv_no == pt).sum() / n_no) if n_no > 0 else 0.0
        inv_pct_ext = (100.0 * (inv_ext == pt).sum() / n_ext) if n_ext > 0 else 0.0
        inv_diff = inv_pct_ext - inv_pct_no

        rows.append({
            "pitch_type": pt,
            "auto_pct_no_alt": float(auto_pct_no),
            "auto_pct_extreme": float(auto_pct_ext),
            "auto_diff_pp": float(auto_diff),
            "invariant_pct_no_alt": float(inv_pct_no),
            "invariant_pct_extreme": float(inv_pct_ext),
            "invariant_diff_pp": float(inv_diff),
        })

    return pd.DataFrame(rows, columns=result_cols)

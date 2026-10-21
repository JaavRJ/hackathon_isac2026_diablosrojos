"""Diagnosticos de Stuff+ (HACK-010, segunda reapertura): ubicacion, sensibilidad, CDMX, rejilla.

Entrena ``StuffPlusModel`` (year <= 2025) y ``PhysicsMovementTranslator`` igual que
``analysis/export_scores.py`` y escribe ``reports/validation/stuff_diagnostics.csv`` en
formato largo: ``metric, pitch_id, value, ci_low, ci_high, n``.

Metricas (``pitch_id`` = pitch shape invariante o ``ALL``; en metricas que no son por pitch_id
la columna lleva una etiqueta, documentada abajo)
------------------------------------------------------------------------------------------
(a) ``loc_r2_oos_<target>`` (``ALL``): R2 fuera de muestra (pooled OOF) de reconstruir
    ``PlateLocHeight``, ``PlateLocSide`` y ``PlateLocSide_norm`` desde ``model.feature_cols``
    con LightGBM y GroupKFold por pitcher sobre filas No Altitude, year <= 2025 (submuestra
    de hasta ``--r2-max-rows``). ``ci_low``/``ci_high`` = minimo/maximo del R2 por fold.
    ``loc_r2_reference_<target>``: R2 con SOLO RelSpeed, EffectiveVelo, ZoneSpeed,
    velo_diff_vs_fb y pitch_id (categoria), mismo LightGBM, misma muestra y mismos folds.
    ``loc_r2_excess_over_reference_<target>`` = R2_completo - R2_referencia (criterio 6:
    <= 0.20 en PlateLocHeight y PlateLocSide). ``loc_r2_excess_max`` = maximo sobre los ejes.
    (Los nombres ``r2_reference``/``r2_excess_over_reference`` son estas metricas.)
    ``loc_importance_gain_pct:<target>:<feature>``: importancia (ganancia, % del total) del
    modelo ajustado en todos los folds de entrenamiento promediada. Si el R2 > 0.30 se agregan
    ``loc_r2_ablation:<target>:<sin_|solo_><grupo>`` (un fold) para localizar la fuente.
(b) ``sens_ivb_pts_per_in``, ``sens_hb_pts_per_in``, ``sens_speed_pts_per_mph``: pendiente
    centrada de Stuff+ (puntos por unidad) sobre el sabor neutral, con derivados recalculados
    (TotalInducedBreak, BreakAngle_deg, SpinToBreakRatio y el diferencial contra la recta de
    referencia congelada; para velocidad tambien EffectiveVelo, ZoneSpeed y velo_diff_vs_fb).
    Paso centrado +-4 in (IVB, HB) o +-2 mph (RelSpeed), dividido por 2h. ``value`` = mediana,
    ``ci_low``/``ci_high`` = cuartiles Q25/Q75 (rango intercuartil, no un IC de la mediana).
(c) ``delta_stuff_cdmx_mean``: media de ``delta_stuff_cdmx`` (``predict_scores`` con la mediana
    de rho de Extreme Altitude), IC 95 % por bootstrap agrupado por pitcher (``--n-boot``).
(d) ``lgbm_cv_rmse`` (``pitch_id`` = ``nl=<..>|mc=<..>|rl=<..>``): RMSE OOF (CV por pitcher,
    No Altitude <= 2025, residuo de xrv_pitcher) de cada combinacion de la rejilla;
    ``lgbm_chosen``: la combinacion elegida (``value`` = su RMSE).

(b) y (c) usan una submuestra estratificada por pitch_id (``--max-per-pitch``) porque
``predict_scores`` recalcula la referencia de recta fila por fila.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from src.config import BASE_DIR
from src.models.movement import PhysicsMovementTranslator
from src.models.stuff import StuffPlusModel, _prepare_stuff_features

SEED = 7
R2_CRITERION = 0.20  # criterio 6 (redefinido): R2_completo - R2_referencia <= 0.20 por eje
REFERENCE_NUMERIC = ["RelSpeed", "EffectiveVelo", "ZoneSpeed", "velo_diff_vs_fb"]
LOC_TARGETS = ("PlateLocHeight", "PlateLocSide", "PlateLocSide_norm")
COLUMNS = ["metric", "pitch_id", "value", "ci_low", "ci_high", "n"]

# Grupos para la ablacion de la reconstruccion de ubicacion (solo si R2 > 0.30)
FEATURE_GROUPS: dict[str, list[str]] = {
    "release_pos": ["RelHeight", "RelSide_norm", "Extension", "arm_slot_deg"],
    "velocidad": ["RelSpeed", "EffectiveVelo", "ZoneSpeed", "velo_diff_vs_fb"],
    "movimiento": ["InducedVertBreak", "HorzBreak_norm", "TotalInducedBreak", "BreakAngle_deg",
                   "SpinToBreakRatio", "ivb_diff_vs_fb", "hb_diff_vs_fb"],
    "giro": ["SpinRate", "SpinAxis_sin", "SpinAxis_cos"],
    "tunel_release": ["tunnel_dist_release_in", "tunnel_sim_release"],
    "otros": ["PitcherThrows", "pitch_usage_pct"],
}


def _row(metric: str, pitch_id: str, value: float, lo: float = np.nan, hi: float = np.nan, n: int = 0) -> dict[str, Any]:
    return {"metric": metric, "pitch_id": pitch_id, "value": float(value),
            "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _lgbm_loc() -> LGBMRegressor:
    return LGBMRegressor(random_state=42, verbose=-1, n_estimators=200, learning_rate=0.1, min_child_samples=50)


# ---------------------------------------------------------------- (a) R2 de ubicacion
def location_r2(model: StuffPlusModel, df: pd.DataFrame, max_rows: int, n_splits: int = 5) -> list[dict[str, Any]]:
    d = df[(df["year"] <= 2025) & (df["altitude_category"] == "No Altitude")]
    d = d.dropna(subset=[t for t in LOC_TARGETS if t in d.columns])
    if len(d) > max_rows:
        d = d.sample(n=max_rows, random_state=SEED)
    X, _ = _prepare_stuff_features(d, model.feature_cols, cat_categories=model.cat_categories_,
                                   feature_medians=model.feature_medians_)
    groups = d["pitcher_anon_id"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, groups=groups))
    Xref = d[REFERENCE_NUMERIC].apply(pd.to_numeric, errors="coerce").astype(float)
    Xref = Xref.fillna({c: float(model.feature_medians_.get(c, 0.0)) for c in REFERENCE_NUMERIC})
    Xref["pitch_id"] = pd.Categorical(d["pitch_id"].astype(str))
    rows: list[dict[str, Any]] = []
    r2_all: list[float] = []
    excess: list[float] = []
    for tgt in [t for t in LOC_TARGETS if t in d.columns]:
        y = d[tgt].to_numpy(dtype=float)
        oof = np.zeros(len(y))
        fold_r2: list[float] = []
        gain = np.zeros(len(model.feature_cols))
        for tr, te in folds:
            reg = _lgbm_loc().fit(X.iloc[tr], y[tr])
            oof[te] = reg.predict(X.iloc[te])
            fold_r2.append(_r2(y[te], oof[te]))
            g = reg.booster_.feature_importance(importance_type="gain").astype(float)
            gain += g / max(g.sum(), 1e-12) / len(folds)
        r2 = _r2(y, oof)
        r2_all.append(r2)
        oof_ref = np.zeros(len(y))
        for tr, te in folds:
            oof_ref[te] = _lgbm_loc().fit(Xref.iloc[tr], y[tr]).predict(Xref.iloc[te])
        r2_ref = _r2(y, oof_ref)
        excess.append(r2 - r2_ref)
        rows.append(_row(f"loc_r2_reference_{tgt}", "ALL", r2_ref, n=len(y)))
        rows.append(_row(f"loc_r2_excess_over_reference_{tgt}", "ALL", r2 - r2_ref, n=len(y)))
        rows.append(_row(f"loc_r2_oos_{tgt}", "ALL", r2, min(fold_r2), max(fold_r2), len(y)))
        for f, gv in zip(model.feature_cols, gain):
            rows.append(_row(f"loc_importance_gain_pct:{tgt}:{f}", "ALL", 100.0 * gv, n=len(y)))
        if r2 - r2_ref > R2_CRITERION:
            tr, te = folds[0]
            for gname, cols in FEATURE_GROUPS.items():
                cols = [c for c in cols if c in model.feature_cols]
                keep_without = [c for c in model.feature_cols if c not in cols]
                for tag, use in (("sin_", keep_without), ("solo_", cols)):
                    if not use:
                        continue
                    reg = _lgbm_loc().fit(X.iloc[tr][use], y[tr])
                    rows.append(_row(f"loc_r2_ablation:{tgt}:{tag}{gname}", "ALL",
                                     _r2(y[te], reg.predict(X.iloc[te][use])), n=len(te)))
    rows.append(_row("loc_r2_max", "ALL", max(r2_all), n=len(d)))
    rows.append(_row("loc_r2_excess_max", "ALL", max(excess), n=len(d)))
    return rows


# ---------------------------------------------------------------- muestreo
def stratified_sample(df: pd.DataFrame, max_per_pitch: int) -> pd.DataFrame:
    if max_per_pitch <= 0:
        return df
    parts = [g.sample(n=min(len(g), max_per_pitch), random_state=SEED) for _, g in df.groupby("pitch_id", observed=True)]
    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------- (b) sensibilidad
def _neutral_matrix(model: StuffPlusModel, translator: PhysicsMovementTranslator, df: pd.DataFrame):
    d = translator.translate_movement(df, target_rho=1.0)
    d = model._recompute_arsenal_diffs_at_density(d, target_rho=1.0, translator=translator, orig_df=df)
    X, _ = _prepare_stuff_features(d, model.feature_cols, cat_categories=model.cat_categories_,
                                   feature_medians=model.feature_medians_)
    return d, X


def _plus(model: StuffPlusModel, X: pd.DataFrame) -> np.ndarray:
    return 100.0 + 10.0 * (model._predict_stuff_value(X) - model.mu_neutral_) / model.sigma_neutral_


def _zone_time(d: pd.DataFrame) -> np.ndarray:
    if "ZoneTime" in d.columns:
        return pd.to_numeric(d["ZoneTime"], errors="coerce").fillna(0.40).clip(lower=0.35).to_numpy()
    rs = pd.to_numeric(d["RelSpeed"], errors="coerce").fillna(90.0).clip(50.0, 110.0)
    return (0.435 + (90.0 - rs) * 0.0035).clip(lower=0.35).to_numpy()


def _perturb(Xn: pd.DataFrame, zt: np.ndarray, feat: str, delta: float) -> pd.DataFrame:
    """Mueve un rasgo y recalcula derivados y diferenciales (referencia de recta congelada)."""
    X = Xn.copy()
    if feat in ("InducedVertBreak", "HorzBreak_norm"):
        X[feat] = X[feat] + delta
        ivb, hb = X["InducedVertBreak"].to_numpy(), X["HorzBreak_norm"].to_numpy()
        X["TotalInducedBreak"] = np.sqrt(ivb ** 2 + hb ** 2)
        X["BreakAngle_deg"] = np.rad2deg(np.arctan2(hb, ivb))
        X["SpinToBreakRatio"] = X["TotalInducedBreak"] / ((X["SpinRate"] / 1000.0) * zt + 1e-5)
        X["ivb_diff_vs_fb" if feat == "InducedVertBreak" else "hb_diff_vs_fb"] += delta
    else:  # RelSpeed
        for c in ("RelSpeed", "EffectiveVelo", "ZoneSpeed", "velo_diff_vs_fb"):
            X[c] = X[c] + delta
    return X


def sensitivity(model: StuffPlusModel, Xn: pd.DataFrame, d: pd.DataFrame, pitch_id: pd.Series) -> list[dict[str, Any]]:
    zt = _zone_time(d)
    specs = [("sens_ivb_pts_per_in", "InducedVertBreak", 4.0),
             ("sens_hb_pts_per_in", "HorzBreak_norm", 4.0),
             ("sens_speed_pts_per_mph", "RelSpeed", 2.0)]
    rows: list[dict[str, Any]] = []
    pid = pitch_id.to_numpy()
    for metric, feat, h in specs:
        slope = (_plus(model, _perturb(Xn, zt, feat, +h)) - _plus(model, _perturb(Xn, zt, feat, -h))) / (2.0 * h)
        for key in [*sorted(pd.unique(pid)), "ALL"]:
            s = slope if key == "ALL" else slope[pid == key]
            q25, q50, q75 = np.nanpercentile(s, [25, 50, 75])
            rows.append(_row(metric, str(key), q50, q25, q75, len(s)))
    return rows


# ---------------------------------------------------------------- (c) Delta CDMX
def delta_cdmx(delta: np.ndarray, pitcher: np.ndarray, pitch_id: np.ndarray, n_boot: int) -> list[dict[str, Any]]:
    codes, uniq = pd.factorize(pitcher)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for key in [*sorted(pd.unique(pitch_id)), "ALL"]:
        m = np.ones(len(delta), bool) if key == "ALL" else pitch_id == key
        sums = np.bincount(codes[m], weights=delta[m], minlength=len(uniq))
        cnts = np.bincount(codes[m], minlength=len(uniq)).astype(float)
        present = np.flatnonzero(cnts > 0)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, len(present), len(present)), minlength=len(present))
            boots[b] = (w * sums[present]).sum() / max((w * cnts[present]).sum(), 1.0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(_row("delta_stuff_cdmx_mean", str(key), delta[m].mean(), lo, hi, int(m.sum())))
    return rows


# ---------------------------------------------------------------- (d) rejilla
def grid_rows(model: StuffPlusModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in model.lgbm_cv_results_:
        rows.append(_row("lgbm_cv_rmse", f"nl={r['num_leaves']}|mc={r['min_child_samples']}|rl={r['reg_lambda']:g}",
                         r["cv_rmse"], n=r["n_cv_rows"]))
    b = model.best_lgbm_params_
    label = f"nl={b['num_leaves']}|mc={b['min_child_samples']}|rl={b['reg_lambda']:g}"
    match = [r for r in model.lgbm_cv_results_ if r["num_leaves"] == b["num_leaves"]
             and r["min_child_samples"] == b["min_child_samples"] and r["reg_lambda"] == b["reg_lambda"]]
    rows.append(_row("lgbm_chosen", label, match[0]["cv_rmse"] if match else np.nan,
                     n=match[0]["n_cv_rows"] if match else 0))
    return rows


def run_diagnostics(df: pd.DataFrame, *, max_per_pitch: int = 5000, r2_max_rows: int = 250_000,
                    n_boot: int = 1000, verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    train = df[df["year"] <= 2025]
    translator = PhysicsMovementTranslator().fit(train)
    model = StuffPlusModel().fit(train, movement_translator=translator)
    log(f"modelo entrenado (n_train={len(train):,}); params={model.best_lgbm_params_}")

    rows: list[dict[str, Any]] = grid_rows(model)
    rows += location_r2(model, df, r2_max_rows)
    log("R2 de ubicacion listo")

    sample = stratified_sample(df, max_per_pitch)
    scored = model.predict_scores(sample, movement_translator=translator)
    log(f"predict_scores sobre {len(sample):,} filas")
    rows += delta_cdmx(scored["delta_stuff_cdmx"].to_numpy(), sample["pitcher_anon_id"].astype(str).to_numpy(),
                       sample["pitch_id"].astype(str).to_numpy(), n_boot)

    d, Xn = _neutral_matrix(model, translator, sample)
    base_check = float(np.abs(_plus(model, Xn) - scored["stuff_plus_neutral"].to_numpy()).max())
    if base_check > 1e-6:
        raise RuntimeError(f"La replica del sabor neutral no coincide con predict_scores (max |dif|={base_check})")
    rows += sensitivity(model, Xn, d, sample["pitch_id"].astype(str))
    log("sensibilidad lista")
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Diagnosticos de Stuff+ (ubicacion, sensibilidad, CDMX, rejilla)")
    ap.add_argument("--input", type=Path, default=BASE_DIR / "data" / "stuff_processed_df.parquet")
    ap.add_argument("--output", type=Path, default=BASE_DIR / "reports" / "validation" / "stuff_diagnostics.csv")
    ap.add_argument("--max-per-pitch", type=int, default=5000, help="filas por pitch_id para (b) y (c); 0 = todas")
    ap.add_argument("--r2-max-rows", type=int, default=250_000)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"No existe dataset procesado: {args.input}")
    out = run_diagnostics(pd.read_parquet(args.input), max_per_pitch=args.max_per_pitch,
                          r2_max_rows=args.r2_max_rows, n_boot=args.n_boot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    ex = out[out["metric"].str.startswith("loc_r2_excess_over_reference_")]
    for _, r in ex.iterrows():
        print(f"{r['metric']}: {r['value']:.3f}")
    exmax = float(out.loc[out["metric"] == "loc_r2_excess_max", "value"].iloc[0])
    print(f"Exceso maximo sobre la referencia: {exmax:.3f} "
          f"({'CUMPLE' if exmax <= R2_CRITERION else 'NO CUMPLE'} el criterio <= {R2_CRITERION})")
    print(f"Escrito {args.output} ({len(out)} filas)")


if __name__ == "__main__":
    main()

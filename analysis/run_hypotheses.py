"""Reproducible, cluster-aware descriptive analyses for H1--H4 and altitude channels.

The park component is a residual decomposition, not a causal mediation estimate.
Run: python -m analysis.run_hypotheses --input data/stuff_processed_df.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EXTREME, NEUTRAL = "Extreme Altitude", "No Altitude"
PITCH_METRICS = ["InducedVertBreak", "HorzBreak_norm", "TotalInducedBreak", "ZoneSpeed",
                 "target_whiff", "home_run", "delta_stuff_cdmx"]


def _ci(values, rng, n_boot=1000):
    """Percentile CI for independent pitcher-level estimates."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p = min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (len(draws) + 1))
    return float(lo), float(hi), float(p)


def _row(metric, estimand, cluster_values, rng, n_boot):
    values = np.asarray(cluster_values, dtype=float)
    values = values[np.isfinite(values)]
    lo, hi, p = _ci(values, rng, n_boot)
    return {"hypothesis": estimand, "metric": metric,
            "estimate": float(values.mean()) if len(values) >= 2 else np.nan,
            "ci_low": lo, "ci_high": hi, "p_value": p, "n_pitchers": int(len(values))}


def _bootstrap_interval(draws, estimate):
    """Summarize already-generated cluster-bootstrap coefficient draws."""
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    lo, hi = np.quantile(x, [0.025, 0.975])
    p = min(1.0, 2 * (min((x <= 0).sum(), (x >= 0).sum()) + 1) / (len(x) + 1))
    return float(estimate), float(lo), float(hi), float(p)


def _paired_pitch_deltas(df, metrics):
    required = {"pitcher_anon_id", "pitch_id", "altitude_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    relevant = df[df.altitude_category.isin([NEUTRAL, EXTREME])]
    metrics = [m for m in metrics if m in relevant.columns]
    pair_keys = ["pitcher_anon_id"] + (["year"] if "year" in relevant.columns else []) + ["pitch_id", "altitude_category"]
    grouped = relevant.groupby(pair_keys, observed=True)[metrics].mean()
    wide = grouped.unstack("altitude_category")
    records = []
    for metric in metrics:
        if metric not in wide:
            continue
        cols = wide[metric]
        if NEUTRAL not in cols or EXTREME not in cols:
            continue
        paired = (cols[EXTREME] - cols[NEUTRAL]).dropna().rename("delta").reset_index()
        for pitch_id, grp in paired.groupby("pitch_id", observed=True):
            # First average repeated seasons within pitcher; then bootstrap these
            # pitcher-level values, so repeated pitches/seasons are not treated
            # as independent units.
            records.append((metric, pitch_id, grp.groupby("pitcher_anon_id").delta.mean().to_numpy()))
    return records


def _ols_r2(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    good = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[good], y[good]
    if len(y) < max(4, x.shape[1] + 2):
        return np.nan
    design = np.column_stack([np.ones(len(y)), x])
    pred = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denom = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / denom) if denom else np.nan


def _paired_batted_ball_park(df, pitchers):
    """Pitcher-paired xRV difference on BIP, adjusted for EV and launch angle.

    Returns a pitcher-indexed per-pitch contribution (conditional BIP contrast
    times that pitcher's pooled BIP share) and the adjustment coefficients.
    Missing BIP flags, covariates, overlap, or a rank-deficient design means the
    park channel is not estimable; it is never replaced by a general residual.
    """
    required = {"pitcher_anon_id", "altitude_category", "xrv_pitcher", "is_ball_in_play", "ExitSpeed", "Angle"}
    if required - set(df.columns):
        return pd.Series(dtype=float), {}
    sample = df[df.pitcher_anon_id.isin(pitchers) & df.altitude_category.isin([NEUTRAL, EXTREME])].copy()
    bip = sample[sample.is_ball_in_play.eq(1)].dropna(subset=["xrv_pitcher", "ExitSpeed", "Angle"])
    counts = bip.groupby(["pitcher_anon_id", "altitude_category"], observed=True).size().unstack("altitude_category")
    bip = bip[bip.pitcher_anon_id.isin(counts.dropna(subset=[NEUTRAL, EXTREME]).index)]
    pitch_counts = sample.groupby(["pitcher_anon_id", "altitude_category"], observed=True).size().unstack("altitude_category")
    matched = counts.dropna(subset=[NEUTRAL, EXTREME]).index.intersection(pitch_counts.dropna(subset=[NEUTRAL, EXTREME]).index)
    if len(matched) < 2:
        return pd.Series(dtype=float), {}
    bip = bip[bip.pitcher_anon_id.isin(matched)].copy()
    # Within-pitcher demeaning gives pitcher fixed effects without constructing
    # a very wide dummy matrix. Altitude is identified only within pitchers.
    bip["extreme"] = bip.altitude_category.eq(EXTREME).astype(float)
    variables = ["extreme", "ExitSpeed", "Angle"]
    centered = bip[variables + ["xrv_pitcher"]] - bip.groupby("pitcher_anon_id", observed=True)[variables + ["xrv_pitcher"]].transform("mean")
    x = centered[variables].to_numpy(float)
    y = centered.xrv_pitcher.to_numpy(float)
    if len(y) < 4 or np.linalg.matrix_rank(x) < x.shape[1]:
        return pd.Series(dtype=float), {}
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    bip["adjusted_xrv"] = bip.xrv_pitcher - beta[1] * bip.ExitSpeed - beta[2] * bip.Angle
    adjusted = bip.groupby(["pitcher_anon_id", "altitude_category"], observed=True).adjusted_xrv.mean().unstack("altitude_category")
    paired = adjusted.dropna(subset=[NEUTRAL, EXTREME])
    total_counts = sample.groupby(["pitcher_anon_id", "altitude_category"], observed=True).size().unstack("altitude_category")
    bip_counts = bip.groupby(["pitcher_anon_id", "altitude_category"], observed=True).size().unstack("altitude_category")
    shares = bip_counts.reindex(paired.index) / total_counts.reindex(paired.index)
    mean_share = shares[[NEUTRAL, EXTREME]].mean(axis=1)
    per_pitch_park = (paired[EXTREME] - paired[NEUTRAL]) * mean_share
    return per_pitch_park.dropna(), {"exit_speed_coef": float(beta[1]), "angle_coef": float(beta[2])}


def _h3_paired_gb_slopes(df, rng, n_boot):
    """Slope of paired xRV/HR changes on each pitcher's stable No-Altitude GB rate."""
    group = df[df.altitude_category.isin([NEUTRAL, EXTREME])].groupby(
        ["pitcher_anon_id", "altitude_category"], observed=True
    ).agg(gb_rate=("target_groundball", "mean"), xrv=("xrv_pitcher", "mean"),
          hr_rate=("home_run", "mean"))
    wide = group.unstack("altitude_category")
    if not {NEUTRAL, EXTREME}.issubset(wide["gb_rate"].columns):
        return []
    paired = pd.DataFrame(index=wide.index)
    paired["baseline_gb_rate"] = wide["gb_rate"][NEUTRAL]
    paired["delta_xrv"] = wide["xrv"][EXTREME] - wide["xrv"][NEUTRAL]
    paired["delta_hr_rate"] = wide["hr_rate"][EXTREME] - wide["hr_rate"][NEUTRAL]
    paired = paired.dropna()
    out = []
    for outcome in ("delta_xrv", "delta_hr_rate"):
        def slope(frame):
            x = np.column_stack([np.ones(len(frame)), frame.baseline_gb_rate.to_numpy(float)])
            if len(frame) < 3 or np.linalg.matrix_rank(x) < 2:
                return np.nan
            return float(np.linalg.lstsq(x, frame[outcome].to_numpy(float), rcond=None)[0][1])
        point = slope(paired)
        ids = paired.index.to_numpy()
        draws = []
        if np.isfinite(point):
            for _ in range(n_boot):
                picked = rng.choice(ids, len(ids), replace=True)
                draw = slope(paired.loc[picked].reset_index(drop=True))
                if np.isfinite(draw):
                    draws.append(draw)
        estimate, lo, hi, p = _bootstrap_interval(draws, point)
        out.append({"hypothesis": f"H3:paired_baseline_GB_x_delta:{outcome}", "metric": outcome,
                    "estimate": estimate, "ci_low": lo, "ci_high": hi, "p_value": p,
                    "n_pitchers": int(len(paired)),
                    "interpretation": "pendiente descriptiva; solo pitchers pareados; GB basal = tasa No Altitude"})
    return out


def run_hypotheses(df: pd.DataFrame, output_dir: str | Path, *, seed=2026, n_boot=1000, stuff_model=None):
    """Create the three task tables and return them as a name-to-frame mapping."""
    if "pitch_id" not in df:
        raise ValueError("Se requiere pitch_id invariante; AutoPitchType no es sustituto válido")
    rng = np.random.default_rng(seed)
    delta_rows = []
    for metric, pitch_id, vals in _paired_pitch_deltas(df, PITCH_METRICS):
        row = _row(metric, f"intra_pitcher_delta:{pitch_id}", vals, rng, n_boot)
        row["pitch_id"] = pitch_id
        delta_rows.append(row)
    if "delta_stuff_cdmx" not in df.columns:
        delta_rows.append({"pitch_id": "ALL", "hypothesis": "not_estimable:delta_stuff_cdmx_missing",
                           "metric": "delta_stuff_cdmx", "estimate": np.nan, "ci_low": np.nan,
                           "ci_high": np.nan, "p_value": np.nan, "n_pitchers": 0})
    delta = pd.DataFrame(delta_rows, columns=["pitch_id", "hypothesis", "metric", "estimate", "ci_low", "ci_high", "p_value", "n_pitchers"])

    # Total observed contrast is paired within pitcher. The Stuff component must
    # come from the same fitted HACK-010 model evaluated on the same rows at two
    # densities. Never substitute an OLS-on-the-outcome regression for that model.
    needed = ["InducedVertBreak", "HorzBreak_norm", "xrv_pitcher", "altitude_category", "pitcher_anon_id"]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para descomposición: {sorted(missing)}")
    valid = df[df.altitude_category.isin([NEUTRAL, EXTREME])].dropna(subset=["InducedVertBreak", "HorzBreak_norm", "xrv_pitcher"]).copy()
    if valid.empty:
        parts = {"total": np.nan, "stuff": np.nan, "park": np.nan}
        cluster_parts = {}
    else:
        means = valid.groupby(["pitcher_anon_id", "altitude_category"], observed=True).xrv_pitcher.mean().unstack("altitude_category")
        cluster_parts = {}
        if NEUTRAL in means and EXTREME in means:
            cluster_parts["total"] = (means[EXTREME] - means[NEUTRAL]).dropna()
        if stuff_model is not None and getattr(stuff_model, "is_fitted_", False):
            extreme_rho = float(df.loc[df.altitude_category.eq(EXTREME), "rho_hat"].median()) if "rho_hat" in df and df.altitude_category.eq(EXTREME).any() else float(getattr(stuff_model, "rho_extreme_median_", 0.852))
            scores = stuff_model.predict_scores(valid, rho_cdmx=extreme_rho)
            # Convert Score+ delta back to the model's xRV scale. predict_scores
            # also rebuilds density-specific primary-fastball arsenal features.
            valid["stuff_cf_delta_xrv"] = ((scores["delta_stuff_cdmx"] / 10.0)
                                           * float(stuff_model.sigma_neutral_))
            stuff_means = valid.groupby("pitcher_anon_id", observed=True).stuff_cf_delta_xrv.mean()
            cluster_parts["stuff"] = stuff_means
            common = cluster_parts.get("total", pd.Series(dtype=float)).index.intersection(stuff_means.index)
            cluster_parts["total"] = cluster_parts["total"].loc[common]
            cluster_parts["stuff"] = stuff_means.loc[common]
            park, _park_controls = _paired_batted_ball_park(df, common)
            common_decomp = common.intersection(park.index)
            if len(common_decomp) >= 2:
                cluster_parts["total"] = cluster_parts["total"].loc[common_decomp]
                cluster_parts["stuff"] = cluster_parts["stuff"].loc[common_decomp]
                cluster_parts["park"] = park.loc[common_decomp]
                cluster_parts["unallocated"] = (cluster_parts["total"] - cluster_parts["stuff"] - cluster_parts["park"])
    channel_rows = []
    for name in ("total", "stuff", "park", "unallocated"):
        vals = np.asarray(cluster_parts.get(name, np.array([])), dtype=float)
        row = _row("xrv_pitcher", f"descriptive_channel:{name}", vals, rng, n_boot)
        row["channel"] = name
        descriptions = {
            "total": "diferencia observada intra-pitcher",
            "stuff": "contrafactual del modelo Stuff HACK-010 en las mismas filas, dos densidades",
            "park": "contraste xRV en BIP, pareado por pitcher y condicionado a ExitSpeed/Angle; no causal",
            "unallocated": "residuo no atribuido tras Stuff y parque BIP; no se asigna a ningún canal",
        }
        row["interpretation"] = descriptions[name]
        if name == "park":
            row["conditional_batted_ball_controls"] = "ExitSpeed, Angle; is_ball_in_play=1; matched pitchers only"
            row["estimable"] = int(row["n_pitchers"] >= 2)
        if name == "unallocated":
            row["estimable"] = int(row["n_pitchers"] >= 2)
        channel_rows.append(row)
    channel = pd.DataFrame(channel_rows)

    # Cluster estimates are formed within pitcher before bootstrap/inference.
    summary_rows = []
    # H1: paired movement changes for invariant pitch identities.
    for metric, pid, vals in _paired_pitch_deltas(df, ["InducedVertBreak", "TotalInducedBreak"]):
        summary_rows.append(_row(metric, f"H1:{pid}", vals, rng, n_boot))
    # H2: relative change in Stuff+ for resilient vs aerodynamically dependent families.
    if "delta_stuff_cdmx" in df:
        fam = df[df.pitch_id.isin(["Changeup", "Splitter", "Four-Seam", "Curveball", "Slider"])].copy()
        fam["family"] = np.where(fam.pitch_id.isin(["Changeup", "Splitter"]), "resilient", "aero_dependent")
        pivot = fam.groupby(["pitcher_anon_id", "family"], observed=True).delta_stuff_cdmx.mean().unstack("family")
        if {"resilient", "aero_dependent"}.issubset(pivot.columns):
            vals = (pivot["resilient"] - pivot["aero_dependent"]).dropna().to_numpy()
        else:
            vals = np.array([])
        summary_rows.append(_row("delta_stuff_cdmx", "H2:resilient_minus_aero_dependent", vals, rng, n_boot))
    else:
        summary_rows.append({"hypothesis": "H2:not_estimable_delta_stuff_cdmx_missing", "metric": "delta_stuff_cdmx",
                             "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n_pitchers": 0})
    # H3: paired within-pitcher changes, using each pitcher's stable No-Altitude
    # ground-ball rate as exposure. Unpaired pitchers never enter either contrast.
    if {"target_groundball", "home_run", "xrv_pitcher"}.issubset(df.columns):
        summary_rows.extend(_h3_paired_gb_slopes(df, rng, n_boot))
    # H4: joint regressions within Extreme Altitude, with pitcher-cluster bootstrap
    # coefficient intervals. This tests spin and observed shape on a common sample.
    if {"SpinRate", "SpinToBreakRatio", "InducedVertBreak", "HorzBreak_norm", "xrv_pitcher", "target_whiff"}.issubset(df.columns):
        h4 = df[df.altitude_category.eq(EXTREME)].dropna(subset=["SpinRate", "SpinToBreakRatio", "InducedVertBreak", "HorzBreak_norm"]).groupby("pitcher_anon_id", observed=True).agg(
            SpinRate=("SpinRate", "mean"), SpinToBreakRatio=("SpinToBreakRatio", "mean"),
            InducedVertBreak=("InducedVertBreak", "mean"), HorzBreak_norm=("HorzBreak_norm", "mean"),
            xrv_pitcher=("xrv_pitcher", "mean"), target_whiff=("target_whiff", "mean"))
        for outcome in ("xrv_pitcher", "target_whiff"):
            estimates = {c: [] for c in ["SpinRate", "SpinToBreakRatio", "InducedVertBreak", "HorzBreak_norm"]}
            ids = h4.index.to_numpy()
            if len(ids) >= 2:
                for _ in range(n_boot):
                    picked = rng.choice(ids, len(ids), replace=True)
                    sample = h4.loc[picked].reset_index(drop=True)
                    xx = np.column_stack([np.ones(len(sample)), *[sample[c].to_numpy(float) for c in estimates]])
                    if len(sample) < xx.shape[1] or np.linalg.matrix_rank(xx) < xx.shape[1]:
                        continue
                    yy = sample[outcome].to_numpy(float)
                    beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
                    for c, b in zip(estimates, beta[1:]): estimates[c].append(float(b))
            for c, draws in estimates.items():
                if len(h4) >= 2:
                    xx_full = np.column_stack([np.ones(len(h4)), *[h4[c0].to_numpy(float) for c0 in estimates]])
                    if len(h4) < xx_full.shape[1] or np.linalg.matrix_rank(xx_full) < xx_full.shape[1]:
                        point = np.nan
                    else:
                        full_beta = np.linalg.lstsq(xx_full, h4[outcome].to_numpy(float), rcond=None)[0]
                        point = float(full_beta[1 + list(estimates).index(c)])
                else:
                    point = np.nan
                estimate, lo, hi, p = _bootstrap_interval(draws, point)
                summary_rows.append({"hypothesis": f"H4:joint_extreme:{outcome}:{c}", "metric": outcome,
                                     "predictor": c, "estimate": estimate,
                                     "ci_low": lo, "ci_high": hi, "p_value": p, "n_pitchers": int(len(ids)),
                                     "interpretation": "regresión conjunta en Extreme Altitude; bootstrap por pitcher"})
    summary = pd.DataFrame(summary_rows)
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    outputs = {"delta_by_pitch_shape.csv": delta, "channel_decomposition.csv": channel,
               "hypotheses_h1_h4_summary.csv": summary}
    for name, table in outputs.items():
        table.to_csv(outdir / name, index=False)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parquet procesado, nunca el parquet crudo")
    parser.add_argument("--output-dir", default="reports/altitude")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--model", help="Joblib StuffPlusModel HACK-010 ya entrenado; si se omite, se ajusta con el parquet procesado")
    parser.add_argument("--save-model", help="Guarda una caché joblib reutilizable en ruta temporal fuera del repositorio")
    args = parser.parse_args(argv)
    df = pd.read_parquet(args.input)
    # The HACK-010 implementation itself trains only on No Altitude, after limiting
    # training seasons to 2024--25. Use that approved model for same-row counterfactuals.
    if args.model:
        import joblib
        model = joblib.load(args.model)
        if not getattr(model, "is_fitted_", False):
            raise ValueError("El modelo HACK-010 indicado no está ajustado")
    else:
        from src.models.stuff import StuffPlusModel
        model = StuffPlusModel().fit(df)
    if args.save_model:
        import joblib
        cache_path = Path(args.save_model)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, cache_path)
    scored = model.predict_scores(df)
    tables = run_hypotheses(scored, args.output_dir, seed=args.seed, n_boot=args.bootstrap, stuff_model=model)
    for name, table in tables.items():
        print(f"{name}: {len(table)} filas")
    print("Canal parque = residuo descriptivo; no se interpreta como efecto causal identificado.")


if __name__ == "__main__":
    main()

"""Generador de cifras y tablas LaTeX para los entregables escritos (HACK-018).

Lee UNICAMENTE ``reports/**/*.csv`` y los parametros de escala de ``models/*.joblib``;
escribe ``docs/latex/generated/cifras.tex`` (macros ``\\cf{clave}``) y una tabla por
archivo en ``docs/latex/generated/tab_*.tex``. Ningun numero de los documentos se
escribe a mano: si un CSV cambia, se vuelve a correr este script y el PDF se actualiza.

Uso::

    python -m analysis.build_paper_numbers
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.sculptor.optimizer import PitchSculptor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
OUT = ROOT / "docs" / "latex" / "generated"

PITCH_ORDER = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
PITCH_ES = {"Four-Seam": "Recta 4 costuras", "Sinker": "Sinker", "Cutter": "Cutter", "Slider": "Slider",
            "Curveball": "Curva", "Changeup": "Cambio", "Splitter": "Splitter"}
MODEL_ORDER = ["CountBaselineModel", "Stuff+", "Location+", "Pitching+"]
MODEL_ES = {"CountBaselineModel": "Baseline nulo (conteo)", "Stuff+": "Stuff+", "Location+": "Location+",
            "Pitching+": "Pitching+"}
HOLDOUT_ES = {"pitcher_fold": "Pitcher (5 folds)", "season": "Temporada 2026", "park": "Parque (Extreme)"}
TARGET_ES = {"target_whiff": "whiff", "target_chase": "chase", "target_called_strike": "strike cantado",
             "target_groundball": "rodado", "target_weak_contact": "contacto débil",
             "target_hard_contact": "contacto duro", "target_out_on_pitch": "out en el pitcheo"}

# Casos de entrevista: (pitcher, temporada) del leaderboard. Los rankings usan MIN_PITCHES_RANK.
CASE_A = ("pitcher_00778", (2024, 2025))   # fichaje resiliente
CASE_B = ("pitcher_00204", 2025)           # trampa de nivel del mar
CASE_C = ("pitcher_00466", 2025)           # rediseño + casa vs gira
MIN_PITCHES_RANK = 500
FS_HIGH, FS_LOW = 0.6, 0.3   # umbrales de uso de la recta 4C para los grupos del leaderboard


MINUS = r"\ensuremath{-}"


def num(x: float, nd: int = 2, sign: bool = False) -> str:
    """Numero listo para LaTeX (signo menos tipografico, valido en texto y en math)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    s = f"{abs(x):,.{nd}f}"
    if x < 0 and float(s.replace(",", "")) != 0:
        return MINUS + s
    return ("+" if sign else "") + s


def pct(x: float, nd: int = 1, sign: bool = False) -> str:
    return num(100 * x, nd, sign)


def pval(p: float) -> str:
    return r"\ensuremath{<}0.01" if p < 0.01 else num(p, 2)


def ci(lo: float, hi: float, nd: int = 2, scale: float = 1.0) -> str:
    return f"[{num(scale * lo, nd)}, {num(scale * hi, nd)}]"


class Cifras:
    def __init__(self) -> None:
        self.items: dict[str, str] = {}

    def __setitem__(self, key: str, value: str) -> None:
        if key in self.items:
            raise KeyError(f"clave duplicada: {key}")
        self.items[key] = value

    def tex(self) -> str:
        lines = ["% ARCHIVO GENERADO por analysis/build_paper_numbers.py. No editar a mano.",
                 r"\makeatletter"]
        for k, v in sorted(self.items.items()):
            lines.append(rf"\@namedef{{cf@{k}}}{{{v}}}")
            # Variante sin signo (``clave.abs``) para frases como "pierde X puntos".
            if v.startswith(MINUS) or (v.startswith("+") and len(v) > 1):
                lines.append(rf"\@namedef{{cf@{k}.abs}}{{{v.removeprefix(MINUS).removeprefix('+')}}}")
        lines.append(r"\makeatother")
        return "\n".join(lines) + "\n"


def tabular(spec: str, header: list[str], rows: list[list[str]], midrules: tuple[int, ...] = (),
            groups: list[tuple[str, int]] | None = None) -> str:
    """``groups``: encabezado superior opcional [(texto, n_columnas), ...] con cmidrule."""
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule"]
    if groups:
        cells, rules, col = [], [], 1
        for text, span in groups:
            cells.append(text if span == 1 else rf"\multicolumn{{{span}}}{{c}}{{{text}}}")
            if text and span > 1:
                rules.append(rf"\cmidrule(lr){{{col}-{col + span - 1}}}")
            col += span
        out += [" & ".join(cells) + r" \\", " ".join(rules)]
    out += [" & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append(r"\midrule")
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "% ARCHIVO GENERADO por analysis/build_paper_numbers.py. No editar a mano.\n" + "\n".join(out) + "\n"


def load() -> dict[str, pd.DataFrame]:
    files = {
        "holdout": "validation/holdout_metrics.csv",
        "calib": "validation/calibration_deciles.csv",
        "stab": "validation/year_to_year_stability.csv",
        "diag": "validation/stuff_diagnostics.csv",
        "density": "altitude/game_density_summary.csv",
        "relabel": "altitude/relabeling_audit.csv",
        "shape": "altitude/delta_by_pitch_shape.csv",
        "hyp": "altitude/hypotheses_h1_h4_summary.csv",
        "chan": "altitude/channel_decomposition.csv",
        "lb": "leaderboards/pitcher_season_leaderboard.csv",
        "ars": "leaderboards/pitcher_arsenal_breakdown.csv",
    }
    return {k: pd.read_csv(REPORTS / v) for k, v in files.items()}


def build(out_dir: Path = OUT) -> dict[str, str]:
    """Calcula todas las cifras y escribe los .tex; devuelve {nombre_archivo: contenido}."""
    d = load()
    c = Cifras()
    files: dict[str, str] = {}

    # ---------------- Datos y densidad ----------------
    den = d["density"]
    c["datos.pitcheos"] = f"{int(den.n_pitches.sum()):,}"
    c["datos.juegos"] = f"{int(den.n_games.sum()):,}"
    c["datos.pitchers"] = f"{d['lb'].pitcher_anon_id.nunique():,}"
    for cat, key in [("No Altitude", "no"), ("Medium Altitude", "med"), ("Extreme Altitude", "ext")]:
        sub = den[den.altitude_category == cat]
        c[f"rho.{key}.pitcheos"] = f"{int(sub.n_pitches.sum()):,}"
        c[f"rho.{key}.min"] = num(sub.rho_hat_median.min(), 3)
        c[f"rho.{key}.max"] = num(sub.rho_hat_median.max(), 3)
    rows = []
    for _, r in den.sort_values(["altitude_category", "year"]).iterrows():
        rows.append([r.altitude_category, str(r.year), f"{int(r.n_games):,}", f"{int(r.n_pitches):,}",
                     num(r.rho_hat_median, 3), num(r.rho_hat_p10, 3), num(r.rho_hat_p90, 3)])
    files["tab_densidad.tex"] = tabular("llrrrrr", ["Categoría", "Año", "Juegos", "Pitcheos", r"$\hat\rho$ mediana",
                                                     "p10", "p90"], rows, midrules=(3, 6))

    c["rho.ext.p90max"] = num(den[den.altitude_category == "Extreme Altitude"].rho_hat_p90.max(), 3)
    c["rho.med.p10min"] = num(den[den.altitude_category == "Medium Altitude"].rho_hat_p10.min(), 3)
    stuff = joblib.load(MODELS / "stuff_model.joblib")
    loc = joblib.load(MODELS / "location_model.joblib")
    pit = joblib.load(MODELS / "pitching_model.joblib")
    c["rho.cdmx"] = num(stuff.rho_extreme_median_, 3)
    c["escala.stuff.sigma"] = num(stuff.sigma_neutral_, 4)
    c["escala.stuff.punto"] = num(stuff.sigma_neutral_ / 10, 5)
    c["escala.loc.sigma"] = num(loc.sigma_neutral_, 4)
    c["escala.pit.sigma"] = num(pit.sigma_neutral_, 4)
    c["stuff.alpha"] = num(stuff.best_alpha_, 2)
    bp = stuff.best_lgbm_params_
    c["stuff.lgbm"] = (f"num\\_leaves={bp.get('num_leaves')}, min\\_child\\_samples={bp.get('min_child_samples')}, "
                       f"reg\\_lambda={bp.get('reg_lambda')}")

    diag = d["diag"].set_index("metric")
    c["diag.loc.r2ref"] = num(diag.loc["loc_r2_reference_PlateLocHeight", "value"], 2)
    c["diag.loc.r2oos"] = num(diag.loc["loc_r2_oos_PlateLocHeight", "value"], 2)
    c["diag.loc.excess"] = num(diag.loc["loc_r2_excess_over_reference_PlateLocHeight", "value"], 2)

    rel = d["relabel"].set_index("pitch_type")
    for p, k in [("Changeup", "ch"), ("Slider", "sl"), ("Cutter", "fc"), ("Sinker", "si")]:
        c[f"relabel.{k}.auto"] = num(rel.loc[p, "auto_diff_pp"], 2, sign=True)
        c[f"relabel.{k}.inv"] = num(rel.loc[p, "invariant_diff_pp"], 2, sign=True)
    rows = [[PITCH_ES.get(p, p), num(rel.loc[p, "auto_diff_pp"], 2, True), num(rel.loc[p, "invariant_diff_pp"], 2, True)]
            for p in PITCH_ORDER if p in rel.index]
    files["tab_reetiquetado.tex"] = tabular("lrr", ["Familia", r"$\Delta$ uso AutoPitchType (pp)",
                                                     r"$\Delta$ uso \texttt{pitch\_id} (pp)"], rows)

    # ---------------- Validacion fuera de muestra ----------------
    h = d["holdout"].copy()
    h["grp"] = h.holdout.str.replace(r"_\d$", "", regex=True)
    g = h.groupby(["grp", "model", "target"])[["rmse", "log_loss", "corr_with_holdout_xrv"]].mean()
    rows, mids = [], []
    for gi, grp in enumerate(["pitcher_fold", "season", "park"]):
        if gi:
            mids.append(len(rows))
        for m in MODEL_ORDER:
            x = g.loc[(grp, m, "xrv_pitcher")]
            ll = lambda t: g.loc[(grp, m, t), "log_loss"]
            rows.append([HOLDOUT_ES[grp] if m == MODEL_ORDER[0] else "", MODEL_ES[m], num(x.rmse, 4),
                         num(x.corr_with_holdout_xrv, 3), num(ll("target_whiff"), 4), num(ll("target_groundball"), 4),
                         num(ll("target_hard_contact"), 4)])
    files["tab_holdouts.tex"] = tabular("llrrrrr", ["Holdout", "Modelo", "RMSE xRV", r"$r$ xRV", "LL whiff",
                                                     "LL rodado", "LL c. duro"], rows, midrules=tuple(mids))
    for grp, k in [("pitcher_fold", "pit"), ("season", "sea"), ("park", "park")]:
        for m, mk in [("CountBaselineModel", "base"), ("Stuff+", "stuff"), ("Location+", "loc"), ("Pitching+", "pit")]:
            c[f"val.{k}.{mk}.r"] = num(g.loc[(grp, m, "xrv_pitcher"), "corr_with_holdout_xrv"], 3)
            c[f"val.{k}.{mk}.rmse"] = num(g.loc[(grp, m, "xrv_pitcher"), "rmse"], 4)
    # mejora relativa de log-loss vs baseline por target
    rows = []
    stuff_wins = 0
    targets = [t for t in TARGET_ES]
    for t in targets:
        r = [TARGET_ES[t]]
        for m in ["Stuff+", "Pitching+"]:
            for grp in ["pitcher_fold", "season", "park"]:
                imp = 1 - g.loc[(grp, m, t), "log_loss"] / g.loc[(grp, "CountBaselineModel", t), "log_loss"]
                r.append(pct(imp, 1, sign=True))
                if m == "Stuff+" and imp > 0:
                    stuff_wins += 1
        rows.append(r)
    files["tab_mejora_logloss.tex"] = tabular("lrrrrrr", ["Target", "Pitcher", "Temporada", "Parque", "Pitcher",
                                                          "Temporada", "Parque"], rows,
                                              groups=[("", 1), (r"Stuff+ (\%)", 3), (r"Pitching+ (\%)", 3)])
    c["val.stuff.ganados"] = str(stuff_wins)
    c["val.stuff.total"] = str(3 * len(targets))

    def imp(m, grp, t):
        return 1 - g.loc[(grp, m, t), "log_loss"] / g.loc[(grp, "CountBaselineModel", t), "log_loss"]

    for t, k in [("target_whiff", "whiff"), ("target_groundball", "gb"), ("target_called_strike", "cs")]:
        vals = [imp("Stuff+", grp, t) for grp in ["pitcher_fold", "season", "park"]]
        c[f"val.stuff.{k}.min"] = pct(min(vals), 1)
        c[f"val.stuff.{k}.max"] = pct(max(vals), 1)
    vals = [imp("Pitching+", grp, t) for grp in ["pitcher_fold", "season", "park"] for t in targets]
    c["val.pit.ll.min"] = pct(min(vals), 1)
    c["val.pit.ll.max"] = pct(max(vals), 1)
    oop = [-imp("Location+", grp, "target_out_on_pitch") for grp in ["pitcher_fold", "season", "park"]]
    c["val.loc.oop.min"] = pct(min(oop), 1)
    c["val.loc.oop.max"] = pct(max(oop), 1)
    c["val.loc.gb.mejor"] = pct(-max(imp("Location+", grp, "target_groundball") for grp in ["pitcher_fold", "season", "park"]), 0)
    c["val.loc.gb.peor"] = pct(-min(imp("Location+", grp, "target_groundball") for grp in ["pitcher_fold", "season", "park"]), 0)

    cal = d["calib"].copy()
    cal["gap"] = (cal.mean_predicted - cal.mean_observed).abs()
    cal["grp"] = cal.holdout.str.replace(r"_\d$", "", regex=True)
    cg = cal.groupby(["grp", "model"]).gap.agg(["mean", "max"])
    rows = []
    for m in MODEL_ORDER:
        rows.append([MODEL_ES[m]] + [num(cg.loc[(grp, m), stat], 4) for grp in ["pitcher_fold", "season", "park"]
                                      for stat in ["mean", "max"]])
    files["tab_calibracion.tex"] = tabular("lrrrrrr", ["Modelo", "Media", "Máx", "Media", "Máx", "Media", "Máx"], rows,
                                            groups=[("", 1), ("Pitcher", 2), ("Temporada 2026", 2), ("Parque", 2)])
    c["cal.max.global"] = num(cal.gap.max(), 3)
    c["cal.stuff.park.mean"] = num(cg.loc[("park", "Stuff+"), "mean"], 4)

    st = d["stab"].set_index("model")
    for m, k in [("Stuff+", "stuff"), ("Location+", "loc"), ("Pitching+", "pit")]:
        c[f"stab.{k}.r"] = num(st.loc[m, "pearson_r"], 2)
        c[f"stab.{k}.rho"] = num(st.loc[m, "spearman_r"], 2)
    c["stab.n"] = str(int(st.n_pitchers.iloc[0]))

    # ---------------- H1-H4 y canales ----------------
    sh = d["shape"]

    def shape(p, metric):
        return sh[(sh.pitch_id == p) & (sh.metric == metric)].iloc[0]

    rows = []
    for p in PITCH_ORDER:
        tib, ivb, hb, zs = (shape(p, m) for m in ["TotalInducedBreak", "InducedVertBreak", "HorzBreak_norm", "ZoneSpeed"])
        rows.append([PITCH_ES[p], str(int(tib.n_pitchers)), num(tib.estimate, 2, True) + " " + ci(tib.ci_low, tib.ci_high),
                     num(ivb.estimate, 2, True), num(hb.estimate, 2, True), num(zs.estimate, 2, True)])
    files["tab_h1.tex"] = tabular("lrlrrr", ["Familia", "Pitchers", r"$\Delta$ quiebre total (in) [IC 95\%]",
                                             r"$\Delta$ IVB", r"$\Delta$ HB", r"$\Delta$ ZoneSpeed (mph)"], rows)
    for p in PITCH_ORDER:
        k = p.replace("-", "").lower()
        t = shape(p, "TotalInducedBreak")
        c[f"h1.{k}.tib"] = num(t.estimate, 1)
        c[f"h1.{k}.tibabs"] = num(abs(t.estimate), 1)
        c[f"h1.{k}.n"] = str(int(t.n_pitchers))
        c[f"h1.{k}.zs"] = num(shape(p, "ZoneSpeed").estimate, 2, True)
        w = shape(p, "target_whiff")
        c[f"h1.{k}.whiff"] = pct(w.estimate, 2, True)
        c[f"h1.{k}.whiffp"] = pval(w.p_value)
        hr = shape(p, "home_run")
        c[f"h1.{k}.hr"] = pct(hr.estimate, 2, True)
        c[f"h1.{k}.hrp"] = pval(hr.p_value)
    c["h1.splitter.p"] = num(shape("Splitter", "TotalInducedBreak").p_value, 3)
    c["h1.splitter.ivbp"] = num(shape("Splitter", "InducedVertBreak").p_value, 2)
    rows = []
    for p in PITCH_ORDER:
        w, hr = shape(p, "target_whiff"), shape(p, "home_run")
        rows.append([PITCH_ES[p], pct(w.estimate, 2, True) + " " + ci(w.ci_low, w.ci_high, 2, 100), pval(w.p_value),
                     pct(hr.estimate, 2, True) + " " + ci(hr.ci_low, hr.ci_high, 2, 100), pval(hr.p_value)])
    files["tab_riesgo.tex"] = tabular("llrlr", ["Familia", r"$\Delta$ whiff (pp) [IC]", "$p$",
                                                r"$\Delta$ HR por pitcheo (pp) [IC]", "$p$"], rows)

    hy = d["hyp"].set_index("hypothesis")
    h2 = hy.loc["H2:resilient_minus_aero_dependent"]
    c["h2.est"] = num(h2.estimate, 2)
    c["h2.ci"] = ci(h2.ci_low, h2.ci_high)
    c["h2.n"] = str(int(h2.n_pitchers))
    h3 = hy.loc["H3:paired_baseline_GB_x_delta:delta_xrv"]
    c["h3.est"] = num(h3.estimate, 3)
    c["h3.ci"] = ci(h3.ci_low, h3.ci_high, 3)
    c["h3.p"] = num(h3.p_value, 2)
    h3b = hy.loc["H3:paired_baseline_GB_x_delta:delta_hr_rate"]
    c["h3.hr.est"] = num(h3b.estimate, 4)
    c["h3.hr.p"] = num(h3b.p_value, 2)
    c["h3.n"] = str(int(h3.n_pitchers))
    rows = []
    for tgt, tk in [("xrv_pitcher", "xrv"), ("target_whiff", "whiff")]:
        for pred, pk in [("SpinRate", "spin"), ("SpinToBreakRatio", "s2b"), ("InducedVertBreak", "ivb"),
                         ("HorzBreak_norm", "hb")]:
            r = hy.loc[f"H4:joint_extreme:{tgt}:{pred}"]
            rows.append([tgt.replace("_", r"\_"), pred.replace("_", r"\_"), f"{r.estimate:.2e}",
                         f"[{r.ci_low:.2e}, {r.ci_high:.2e}]", pval(r.p_value)])
            c[f"h4.{tk}.{pk}.p"] = pval(r.p_value)
            c[f"h4.{tk}.{pk}.prel"] = (r"\ensuremath{<}0.01" if r.p_value < 0.01 else "=" + num(r.p_value, 2))
            c[f"h4.{tk}.{pk}.per100"] = num(100 * r.estimate, 4)
    c["h4.n"] = str(int(hy.loc["H4:joint_extreme:xrv_pitcher:SpinRate", "n_pitchers"]))
    files["tab_h4.tex"] = tabular("llrlr", ["Resultado", "Predictor", "Coef.", "IC 95\\%", "$p$"], rows, midrules=(4,))

    ch = d["chan"].set_index("channel")
    for k in ["total", "stuff", "park", "unallocated"]:
        r = ch.loc[k]
        c[f"canal.{k}"] = num(r.estimate, 4)
        c[f"canal.{k}.per100"] = num(100 * r.estimate, 2)
        c[f"canal.{k}.ci"] = ci(r.ci_low, r.ci_high, 4)
        c[f"canal.{k}.share"] = pct(r.share_of_total, 0)
    c["canal.n"] = str(int(ch.loc["total", "n_pitchers"]))
    c["riesgo.hr.suben"] = str(int(sum(shape(p, "home_run").estimate > 0 for p in PITCH_ORDER)))
    c["riesgo.hr.familias"] = str(len(PITCH_ORDER))
    c["riesgo.hr.signif"] = str(int(sum(shape(p, "home_run").p_value < 0.05 for p in PITCH_ORDER)))
    c["canal.ratio"] = num(ch.loc["park", "estimate"] / ch.loc["stuff", "estimate"], 1)

    # ---------------- Contrafactual de Stuff+ por familia (arsenal) ----------------
    a = d["ars"]
    rows = []
    for p in PITCH_ORDER:
        x = a[a.pitch_id == p]
        wts = x.n_pitches

        def wavg(col):
            ok = x[col].notna()
            return np.average(x.loc[ok, col], weights=wts[ok]) if ok.any() else np.nan
        dlt = wavg("delta_stuff_cdmx")
        c[f"cf.{p.replace('-', '').lower()}.delta"] = num(dlt, 1)
        c[f"cf.{p.replace('-', '').lower()}.neutral"] = num(wavg("stuff_plus_neutral"), 1)
        c[f"cf.{p.replace('-', '').lower()}.cdmx"] = num(wavg("stuff_plus_cdmx"), 1)
        rows.append([PITCH_ES[p], f"{int(wts.sum()):,}", num(wavg("stuff_plus_neutral"), 1), num(wavg("stuff_plus_cdmx"), 1),
                     num(dlt, 2, True), num(wavg("ivb_neutral"), 1) + r" $\to$ " + num(wavg("ivb_cdmx"), 1),
                     num(wavg("zone_speed_neutral"), 1) + r" $\to$ " + num(wavg("zone_speed_cdmx"), 1)])
    files["tab_contrafactual.tex"] = tabular("lrrrrrr", ["Familia", "Pitcheos", "Stuff+ neutral", "Stuff+ CDMX",
                                                         r"$\Delta$", "IVB (in)", "ZoneSpeed (mph)"], rows)

    lb = d["lb"]
    q = lb[lb.n_pitches >= MIN_PITCHES_RANK]
    c["lb.min"] = str(MIN_PITCHES_RANK)
    c["lb.n"] = str(len(q))
    c["lb.delta.mean"] = num(q.delta_stuff_cdmx.mean(), 2)
    c["lb.delta.median"] = num(q.delta_stuff_cdmx.median(), 2)
    c["lb.delta.min"] = num(q.delta_stuff_cdmx.min(), 2)
    c["lb.delta.max"] = num(q.delta_stuff_cdmx.max(), 2, True)
    c["lb.delta.pos"] = str(int((q.delta_stuff_cdmx > 0).sum()))
    # uso de la recta 4 costuras vs delta: correlacion por pitcher-temporada
    fs = a[a.pitch_id == "Four-Seam"][["pitcher_anon_id", "year", "usage_pct"]]
    m = q.merge(fs, on=["pitcher_anon_id", "year"], how="left").fillna({"usage_pct": 0.0})
    c["lb.corr.fsuso"] = num(np.corrcoef(m.usage_pct, m.delta_stuff_cdmx)[0, 1], 2)
    hi, lo = m[m.usage_pct >= FS_HIGH], m[m.usage_pct < FS_LOW]
    c["lb.fsalto.umbral"] = pct(FS_HIGH, 0)
    c["lb.fsbajo.umbral"] = pct(FS_LOW, 0)
    c["lb.fsalto.n"] = str(len(hi))
    c["lb.fsalto.delta"] = num(hi.delta_stuff_cdmx.mean(), 2)
    c["lb.fsbajo.n"] = str(len(lo))
    c["lb.fsbajo.delta"] = num(lo.delta_stuff_cdmx.mean(), 2)

    cols = ["n_pitches", "pitcher_throws", "arm_slot_deg_mean", "rel_speed_mean", "stuff_plus_neutral", "stuff_plus_cdmx",
            "delta_stuff_cdmx", "location_plus_neutral", "pitching_plus_neutral", "pitching_plus_cdmx", "gb_rate_bip"]

    def season(pid, yr, key):
        r = lb[(lb.pitcher_anon_id == pid) & (lb.year == yr)].iloc[0]
        c[f"{key}.id"] = pid.replace("_", r"\_")
        c[f"{key}.year"] = str(yr)
        c[f"{key}.n"] = f"{int(r.n_pitches):,}"
        c[f"{key}.mano"] = "zurdo" if r.pitcher_throws == "Left" else "derecho"
        c[f"{key}.slot"] = num(r.arm_slot_deg_mean, 0)
        c[f"{key}.velo"] = num(r.rel_speed_mean, 1)
        c[f"{key}.sn"] = num(r.stuff_plus_neutral, 1)
        c[f"{key}.sc"] = num(r.stuff_plus_cdmx, 1)
        c[f"{key}.sd"] = num(r.delta_stuff_cdmx, 2, True)
        c[f"{key}.ln"] = num(r.location_plus_neutral, 1)
        c[f"{key}.pn"] = num(r.pitching_plus_neutral, 1)
        c[f"{key}.pc"] = num(r.pitching_plus_cdmx, 1)
        c[f"{key}.gb"] = pct(r.gb_rate_bip, 0)
        c[f"{key}.rp"] = num(r.runs_prevented_per_100, 2, True)
        c[f"{key}.ct"] = num(100 * r.canal_total, 2, True)
        rank = (q.stuff_plus_neutral > r.stuff_plus_neutral).sum() + 1
        c[f"{key}.rank.sn"] = str(int(rank))
        c[f"{key}.rank.delta"] = str(int((q.delta_stuff_cdmx > r.delta_stuff_cdmx).sum() + 1))
        return r

    def arsenal(pid, yr, key):
        x = a[(a.pitcher_anon_id == pid) & (a.year == yr)].sort_values("usage_pct", ascending=False)
        rows = []
        for _, r in x.iterrows():
            pk = r.pitch_id.replace("-", "").lower()
            c[f"{key}.{pk}.uso"] = pct(r.usage_pct, 0)
            c[f"{key}.{pk}.sn"] = num(r.stuff_plus_neutral, 1)
            c[f"{key}.{pk}.sc"] = num(r.stuff_plus_cdmx, 1)
            c[f"{key}.{pk}.sd"] = num(r.delta_stuff_cdmx, 2, True)
            rows.append([PITCH_ES.get(r.pitch_id, r.pitch_id), f"{int(r.n_pitches)}", pct(r.usage_pct, 0) + r"\%",
                         num(r.rel_speed_mean, 1), num(r.ivb_neutral, 1) + r" $\to$ " + num(r.ivb_cdmx, 1),
                         num(r.stuff_plus_neutral, 1), num(r.stuff_plus_cdmx, 1), num(r.delta_stuff_cdmx, 2, True)])
        files[f"tab_{key}_arsenal.tex"] = tabular("lrrrrrrr", ["Pitcheo", "n", "Uso", "Velo", "IVB (in)", "Stuff+ N",
                                                                "Stuff+ CDMX", r"$\Delta$"], rows)
        return x

    pa, (ya1, ya2) = CASE_A
    season(pa, ya1, "ca1")
    season(pa, ya2, "ca2")
    arsenal(pa, ya2, "ca")
    pb, yb = CASE_B
    season(pb, yb, "cb")
    arsenal(pb, yb, "cb")
    c["cb.pctl"] = pct((q.stuff_plus_neutral < lb[(lb.pitcher_anon_id == pb) & (lb.year == yb)].stuff_plus_neutral.iloc[0]).mean(), 0)
    pc, yc = CASE_C
    rc = season(pc, yc, "cc")
    xc = arsenal(pc, yc, "cc")
    for k in ["canal_total", "canal_parque_bip"]:
        c[f"cc.{k.replace('_', '')}"] = num(100 * rc[k], 2)

    # Sculptor sobre la recta del caso C (mismo motor que la pestaña del dashboard)
    fsr = xc[xc.pitch_id == "Four-Seam"].iloc[0]
    # Se excluye al propio pitcher del catalogo para que sus pares no lo incluyan a el mismo.
    sculptor = PitchSculptor(a[a.pitcher_anon_id != pc].reset_index(drop=True))
    params = {"PitcherThrows": fsr.PitcherThrows, "arm_slot_deg": fsr.arm_slot_deg, "RelHeight": fsr.RelHeight,
              "fb_velo": fsr.fb_velo, "pitch_id": "Four-Seam", "user_stuff_plus_cdmx": fsr.stuff_plus_cdmx}
    res = sculptor.find_feasible_references(params, include_coaching_guide=False)
    if res["status"] != "OK":
        raise RuntimeError(f"Sculptor sin pares para el caso C: {res['reason']}")
    c["sc.peers"] = str(res["n_peers"])
    fam = res["intra_family"]
    c["sc.fs.nrows"] = str(fam["n_rows"])
    c["sc.fs.p25"] = num(fam["p25_stuff_plus_cdmx"], 1)
    c["sc.fs.p75"] = num(fam["p75_stuff_plus_cdmx"], 1)
    c["sc.fs.best"] = num(fam["best_reference"]["stuff_plus_cdmx"], 1)
    c["sc.fs.gain"] = num(res["delta_gain_cdmx"], 1, True)
    rep = {r["pitch_id"]: r for r in res["repertoire"]}
    rows = []
    for r in res["repertoire"]:
        rows.append([PITCH_ES.get(r["pitch_id"], r["pitch_id"]), str(r["n_pitchers"]), f"{r['n_pitches']:,}",
                     num(r["median_stuff_plus_cdmx"], 1)])
        pk = r["pitch_id"].replace("-", "").lower()
        c[f"sc.rep.{pk}"] = num(r["median_stuff_plus_cdmx"], 1)
        c[f"sc.rep.{pk}.n"] = str(r["n_pitchers"])
    files["tab_sculptor.tex"] = tabular("lrrr", ["Pitcheo (en pares factibles)", "Pitchers", "Pitcheos",
                                                 "Stuff+ CDMX mediana"], rows)
    c["sc.slot.tol"] = num(res["filters"]["arm_slot_tol_deg"], 0)
    c["sc.relh.tol"] = num(res["filters"]["rel_height_tol_ft"], 2)
    c["sc.velo.tol"] = num(res["filters"]["fb_velo_tol_mph"], 0)

    _velovision_macros(c)

    files["cifras.tex"] = c.tex()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    return files


def _velovision_macros(c: "Cifras") -> None:
    """Cifras de la validacion piloto de VeloVision (17-D). Sin CSV: 'no disponible', sin tronar."""
    keys = ["vv.n", "vv.icc", "vv.mae.ann", "vv.mae.ann.n", "vv.mae.det", "vv.mae.det.n", "vv.rel.ms",
            "vv.window"]
    path = REPORTS / "validation" / "vision_validation_summary.csv"
    vals: dict[str, str] = {}
    if path.is_file():
        try:
            df = pd.read_csv(path)

            def get(metric: str, measure: str) -> tuple[float, int] | None:
                r = df[(df.metric == metric) & (df.measure == measure)]
                if r.empty or pd.isna(r.iloc[0]["value"]):
                    return None
                return float(r.iloc[0]["value"]), int(r.iloc[0]["n"])

            slot = "arm_slot_dataset_deg"
            spec = {"vv.n": ("n_both_annotators", "all", 0), "vv.icc": ("icc_2_1", slot, 3),
                    "vv.mae.ann": ("mae_vs_human_mean", slot, 1),
                    "vv.mae.det": ("mae_vs_human_mean_detected_frame", slot, 1),
                    "vv.rel.ms": ("release_frame_mae_ms", "release_frame", 0),
                    "vv.window": ("recommended_combined_slot_error_deg", slot, 0)}
            for key, (metric, measure, nd) in spec.items():
                got = get(metric, measure)
                if got:
                    vals[key] = ("> 0.999" if key == "vv.icc" and got[0] >= 0.9995 else num(got[0], nd))
                    if key in ("vv.mae.ann", "vv.mae.det"):
                        vals[key + ".n"] = str(got[1])
        except Exception:  # CSV malformado: se degrada igual que si no existiera
            vals = {}
    for k in keys:
        c[k] = vals.get(k, "no disponible")


def build_figures(out_dir: Path = OUT) -> list[Path]:
    """Figuras del documento técnico (PDF vectorial). Fuera del test de igualdad byte a byte."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load()
    sh, a = d["shape"], d["ars"]
    fams = PITCH_ORDER[::-1]
    labels = [PITCH_ES[p] for p in fams]
    tib = sh[sh.metric == "TotalInducedBreak"].set_index("pitch_id").loc[fams]
    cf = [np.average(a[a.pitch_id == p].delta_stuff_cdmx, weights=a[a.pitch_id == p].n_pitches) for p in fams]

    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    y = np.arange(len(fams))
    ax1.barh(y, tib.estimate, color="#B3121B", alpha=0.85, height=0.6)
    ax1.errorbar(tib.estimate, y, xerr=[tib.estimate - tib.ci_low, tib.ci_high - tib.estimate], fmt="none",
                 ecolor="#222222", capsize=2.5, lw=0.9)
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("Δ quiebre total, Extreme − No Altitude (in)")
    ax1.set_title("Observado dentro del pitcher (IC 95 %)", fontsize=9)
    ax2.barh(y, cf, color="#52525B", alpha=0.85, height=0.6)
    ax2.set_xlabel("Δ Stuff+ al traducir a ρ̂ CDMX (puntos)")
    ax2.set_title("Contrafactual del modelo", fontsize=9)
    for ax in (ax1, ax2):
        ax.axvline(0, color="#222222", lw=0.8)
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_h1.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return [path]


if __name__ == "__main__":
    written = build()
    figs = build_figures()
    print(f"{len(written)} archivos y {len(figs)} figuras en {OUT.relative_to(ROOT)}")

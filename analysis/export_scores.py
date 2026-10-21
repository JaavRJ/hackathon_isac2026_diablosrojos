"""Exportador oficial del Entregable 1 (HACK-014).

Ajusta (o recibe) PhysicsMovementTranslator, StuffPlusModel, LocationPlusModel y
PitchingPlusModel, puntua todos los lanzamientos y escribe:

* ``<deliverables>/stuff_plus_scores_by_pitch.parquet`` (+ ``.csv.gz``): una fila por
  ``PitchUID`` (ignorado en git). Unico archivo con granularidad de lanzamiento.
* ``<reports>/pitcher_season_leaderboard.csv``: por ``(pitcher_anon_id, year)``.
* ``<reports>/pitcher_arsenal_breakdown.csv``: por ``(pitcher_anon_id, year, pitch_id)``.
* ``<reports>/movement_map_grid.csv``: grilla ``(IVB_bin, HB_norm_bin, pitch_id)``.
* ``<models>/{movement_translator,stuff_model,location_model,pitching_model}.joblib``.

Los CSV de ``reports/`` jamas contienen ``PitchUID`` ni filas individuales y pesan < 2 MB
(se valida al escribir). Se agrupa siempre por ``pitch_id`` invariante, nunca por
``AutoPitchType``.

Definiciones y limitaciones documentadas
----------------------------------------
* ``stuff_plus_*``/``location_plus_*``/``pitching_plus_*`` salen de ``predict_scores`` de los
  modelos (Score+ = 100 + 10 z, referencia neutral); ``exp_*`` son las probabilidades/xRV
  esperados de ``StuffPlusModel.predict_outcomes`` (conteo + residuo Stuff, tal como se
  observo el lanzamiento, sin traduccion de densidad).
* "Carreras prevenidas" no existe como columna: se deriva como ``runs_prevented_xrv =
  sum(xrv_pitcher)`` (``+`` = previene carreras, valor empirico LMB vs. el promedio de la
  liga por estado de conteo; NO es un contrafactual contra un reemplazo) y
  ``runs_prevented_per_100 = 100 * mean(xrv_pitcher)``.
* ``gb_rate_bip`` = sum(target_groundball) / sum(is_ball_in_play) (NaN sin pelotas en
  juego); ``gb_pct_pitches`` = media de ``target_groundball`` por lanzamiento.
* ``*_neutral`` / ``*_cdmx`` fisicos (IVB, HB_norm, ZoneSpeed) son medias del movimiento
  traducido por la Etapa A a rho = 1.0 y rho = ``rho_cdmx`` (mediana de Extreme Altitude,
  que mezcla varios parques; no es solo CDMX). ``RelSpeed`` es de release y no cambia.
* Location+ no depende de la densidad: ``location_plus_cdmx == location_plus_neutral``.
* Entrenamiento: traductor, Stuff+, Location+ y Pitching+ se ajustan SOLO con year <= 2025
  (``_fit_models``); 2026 se puntua out-of-sample. Los scores de 2024-2025 son in-sample.
* Descomposicion de tres terminos por (pitcher, year) en el leaderboard (``canal_total``
  = xRV medio Extreme - No Altitude del pitcher; ``canal_stuff`` = contrafactual de Stuff
  ``delta_stuff_cdmx / 10 * sigma_neutral_`` promediado en las mismas filas que
  ``run_hypotheses``; ``canal_parque_bip`` = contraste BIP ajustado por ExitSpeed/Angle, via
  ``analysis.run_hypotheses._paired_batted_ball_park`` aplicado a cada temporada;
  ``residuo_no_atribuido`` = total - stuff - parque). Unidad: xRV (carreras) por lanzamiento.
  Umbral de n: >= ``min_pitches_alt`` (30) lanzamientos validos en CADA altitud (No y
  Extreme) y >= ``min_bip_alt`` (5) pelotas en juego en cada una; ademas la regresion de parque
  requiere >= 2 pitchers pareados en la temporada. Si no se cumple, las cuatro columnas quedan
  en NaN explicito (nunca se rellenan). Cuando no son NaN, stuff + parque + residuo = total.
* ``pitcher_arsenal_breakdown``: ``PitcherThrows``, ``arm_slot_deg`` y ``RelHeight`` son
  valores por pitcher-temporada (media) repetidos en cada pitch_id; ``fb_velo`` = RelSpeed medio
  de la recta primaria (Four-Seam/Sinker/Cutter mas usada del pitcher-temporada; NaN si no
  tiene recta); ``rho_cdmx_used`` = rho objetivo usado para los scores CDMX.
* ``movement_map_grid`` usa el movimiento *observado* (bins de ``bin_in`` pulgadas) y
  suprime celdas con menos de ``min_cell_n`` lanzamientos para no exponer puntos.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from analysis.run_hypotheses import EXTREME, NEUTRAL, _paired_batted_ball_park
from src.config import BASE_DIR, FASTBALL_TYPES, TARGET_COLS
from src.models.location_pitching import LocationPlusModel, PitchingPlusModel
from src.models.movement import PhysicsMovementTranslator
from src.models.stuff import StuffPlusModel

SCORES_FILE = "stuff_plus_scores_by_pitch"
MAX_REPORT_BYTES = 2 * 1024 * 1024
ID_COLS = ["PitchUID", "pitcher_anon_id", "year", "pitch_id"]
SCORE_COLS = [
    "stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx",
    "location_plus_neutral", "location_plus_cdmx",
    "pitching_plus_neutral", "pitching_plus_cdmx",
]
EXP_COLS = [f"exp_{t}" for t in TARGET_COLS]
# Columnas auxiliares (solo agregados; no se escriben en el archivo por PitchUID).
AUX_COLS = [
    "PitcherThrows", "arm_slot_deg", "RelHeight", "altitude_category", "ExitSpeed", "Angle", "RelSpeed", "EffectiveVelo", "SpinRate",
    "InducedVertBreak", "HorzBreak_norm", "tunnel_sim_release", "tunnel_sim_170ms",
    "target_groundball", "is_ball_in_play", "xrv_pitcher",
]
TRANSLATED_COLS = ["ivb_neutral", "ivb_cdmx", "hb_neutral", "hb_cdmx", "zone_speed_neutral", "zone_speed_cdmx"]


def _fit_models(df: pd.DataFrame) -> dict[str, Any]:
    """Patron de run_validation: traductor ajustado primero y compartido por los modelos."""
    train = df[df["year"] <= 2025] if "year" in df and (df["year"] <= 2025).any() else df
    translator = PhysicsMovementTranslator().fit(train)
    stuff = StuffPlusModel().fit(train, movement_translator=translator)
    location = LocationPlusModel().fit(train)
    pitching = PitchingPlusModel(movement_translator=translator).fit(train)
    return {"movement_translator": translator, "stuff_model": stuff,
            "location_model": location, "pitching_model": pitching}


def _score_chunk(chunk: pd.DataFrame, models: dict[str, Any], rho: float) -> pd.DataFrame:
    translator = models["movement_translator"]
    stuff = models["stuff_model"].predict_scores(chunk, rho_cdmx=rho)
    loc = models["location_model"].predict_scores(chunk, rho_cdmx=rho)
    pit = models["pitching_model"].predict_all_targets(chunk, rho_cdmx=rho)
    exp = models["stuff_model"].predict_outcomes(chunk)
    out = chunk[[c for c in ID_COLS + AUX_COLS if c in chunk.columns]].copy()
    for col in ("stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx"):
        out[col] = stuff[col].to_numpy()
    out["location_plus_neutral"] = loc["location_plus_neutral"].to_numpy()
    out["location_plus_cdmx"] = loc["location_plus_cdmx"].to_numpy()
    out["pitching_plus_neutral"] = pit["pitching_plus_neutral"].to_numpy()
    out["pitching_plus_cdmx"] = pit["pitching_plus_cdmx"].to_numpy()
    for tgt in TARGET_COLS:
        out[f"exp_{tgt}"] = exp[tgt].to_numpy()
    for tag, r in (("neutral", 1.0), ("cdmx", rho)):
        moved = translator.translate_movement(chunk, target_rho=r)
        out[f"ivb_{tag}"] = moved["InducedVertBreak"].to_numpy()
        out[f"hb_{tag}"] = moved["HorzBreak_norm"].to_numpy()
        out[f"zone_speed_{tag}"] = moved["ZoneSpeed"].to_numpy()
    return out


def _round(table: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Redondea a ``decimals`` salvo las columnas del canal (xRV ~1e-4, sin redondear)."""
    num = [c for c in table.select_dtypes("number").columns if c not in CHANNEL_COLS]
    table[num] = table[num].round(decimals)
    return table


CHANNEL_COLS = ["canal_total", "canal_stuff", "canal_parque_bip", "residuo_no_atribuido"]


def _channel_decomposition(
    s: pd.DataFrame, sigma_neutral: float, min_pitches_alt: int, min_bip_alt: int,
) -> pd.DataFrame:
    """Tres terminos por (pitcher, year); NaN explicito si no hay n suficiente."""
    rows = []
    sub = s[s["altitude_category"].isin([NEUTRAL, EXTREME])]
    for year, sy in sub.groupby("year", observed=True):
        valid = sy.dropna(subset=["InducedVertBreak", "HorzBreak_norm", "xrv_pitcher"])
        if valid.empty:
            continue
        keys = ["pitcher_anon_id", "altitude_category"]
        n_alt = valid.groupby(keys, observed=True).size().unstack("altitude_category")
        bip = valid[valid["is_ball_in_play"].eq(1)].dropna(subset=["ExitSpeed", "Angle"])
        n_bip = bip.groupby(keys, observed=True).size().unstack("altitude_category")
        if not {NEUTRAL, EXTREME} <= set(n_alt.columns) or not {NEUTRAL, EXTREME} <= set(n_bip.columns):
            continue
        ok = n_alt[(n_alt[NEUTRAL] >= min_pitches_alt) & (n_alt[EXTREME] >= min_pitches_alt)].index
        ok = ok.intersection(n_bip[(n_bip[NEUTRAL] >= min_bip_alt) & (n_bip[EXTREME] >= min_bip_alt)].index)
        if len(ok) < 2:
            continue
        v = valid[valid["pitcher_anon_id"].isin(ok)].copy()
        means = v.groupby(keys, observed=True).xrv_pitcher.mean().unstack("altitude_category")
        total = means[EXTREME] - means[NEUTRAL]
        v["stuff_cf"] = v["delta_stuff_cdmx"] / 10.0 * sigma_neutral
        stuff = v.groupby("pitcher_anon_id", observed=True).stuff_cf.mean()
        park, _ = _paired_batted_ball_park(v, ok)
        idx = total.index.intersection(stuff.index).intersection(park.index)
        for pid in idx:
            rows.append({"pitcher_anon_id": pid, "year": year, "canal_total": total[pid],
                         "canal_stuff": stuff[pid], "canal_parque_bip": park[pid],
                         "residuo_no_atribuido": total[pid] - stuff[pid] - park[pid]})
    cols = ["pitcher_anon_id", "year", *CHANNEL_COLS]
    return pd.DataFrame(rows, columns=cols)


def _leaderboard(
    s: pd.DataFrame, sigma_neutral: float = 1.0, min_pitches_alt: int = 30, min_bip_alt: int = 5,
) -> pd.DataFrame:
    spec: dict[str, tuple[str, str]] = {
        "n_pitches": ("pitch_id", "size"),
        "arm_slot_deg_mean": ("arm_slot_deg", "mean"),
        "rel_speed_mean": ("RelSpeed", "mean"),
        "effective_velo_mean": ("EffectiveVelo", "mean"),
        "zone_speed_neutral_mean": ("zone_speed_neutral", "mean"),
        "zone_speed_cdmx_mean": ("zone_speed_cdmx", "mean"),
        "stuff_plus_neutral": ("stuff_plus_neutral", "mean"),
        "stuff_plus_cdmx": ("stuff_plus_cdmx", "mean"),
        "delta_stuff_cdmx": ("delta_stuff_cdmx", "mean"),
        "location_plus_neutral": ("location_plus_neutral", "mean"),
        "location_plus_cdmx": ("location_plus_cdmx", "mean"),
        "pitching_plus_neutral": ("pitching_plus_neutral", "mean"),
        "pitching_plus_cdmx": ("pitching_plus_cdmx", "mean"),
        "gb_pct_pitches": ("target_groundball", "mean"),
        "gb_sum": ("target_groundball", "sum"),
        "bip_sum": ("is_ball_in_play", "sum"),
        "runs_prevented_xrv": ("xrv_pitcher", "sum"),
        "runs_prevented_per_100": ("xrv_pitcher", "mean"),
        "exp_xrv_per_100": ("exp_xrv_pitcher", "mean"),
    }
    spec = {k: v for k, v in spec.items() if v[0] in s.columns}
    if "PitcherThrows" in s.columns:
        spec["pitcher_throws"] = ("PitcherThrows", "first")
    lb = s.groupby(["pitcher_anon_id", "year"], observed=True).agg(**spec).reset_index()
    if {"gb_sum", "bip_sum"} <= set(lb.columns):
        lb["gb_rate_bip"] = np.where(lb["bip_sum"] > 0, lb["gb_sum"] / lb["bip_sum"].replace(0, np.nan), np.nan)
        lb = lb.drop(columns=["gb_sum", "bip_sum"])
    chan = _channel_decomposition(s, sigma_neutral, min_pitches_alt, min_bip_alt)
    lb = lb.merge(chan, on=["pitcher_anon_id", "year"], how="left")  # sin match -> NaN explicito
    if "runs_prevented_per_100" in lb:
        lb["runs_prevented_per_100"] = lb["runs_prevented_per_100"] * 100.0
    if "exp_xrv_per_100" in lb:
        lb["exp_xrv_per_100"] = lb["exp_xrv_per_100"] * 100.0
    return _round(lb)


def _arsenal(s: pd.DataFrame, rho: float = float('nan')) -> pd.DataFrame:
    spec: dict[str, tuple[str, str]] = {
        "n_pitches": ("pitch_id", "size"),
        "rel_speed_mean": ("RelSpeed", "mean"),
        "spin_rate_mean": ("SpinRate", "mean"),
        "ivb_neutral": ("ivb_neutral", "mean"), "ivb_cdmx": ("ivb_cdmx", "mean"),
        "hb_norm_neutral": ("hb_neutral", "mean"), "hb_norm_cdmx": ("hb_cdmx", "mean"),
        "zone_speed_neutral": ("zone_speed_neutral", "mean"), "zone_speed_cdmx": ("zone_speed_cdmx", "mean"),
        "tunnel_sim_release": ("tunnel_sim_release", "mean"),
        "tunnel_sim_170ms": ("tunnel_sim_170ms", "mean"),
        "stuff_plus_neutral": ("stuff_plus_neutral", "mean"),
        "stuff_plus_cdmx": ("stuff_plus_cdmx", "mean"),
        "delta_stuff_cdmx": ("delta_stuff_cdmx", "mean"),
        "location_plus_neutral": ("location_plus_neutral", "mean"),
        "location_plus_cdmx": ("location_plus_cdmx", "mean"),
        "pitching_plus_neutral": ("pitching_plus_neutral", "mean"),
        "pitching_plus_cdmx": ("pitching_plus_cdmx", "mean"),
    }
    spec = {k: v for k, v in spec.items() if v[0] in s.columns}
    ars = s.groupby(["pitcher_anon_id", "year", "pitch_id"], observed=True).agg(**spec).reset_index()
    total = ars.groupby(["pitcher_anon_id", "year"], observed=True)["n_pitches"].transform("sum")
    ars["usage_pct"] = ars["n_pitches"] / total
    key = ["pitcher_anon_id", "year"]
    py = s.groupby(key, observed=True).agg(arm_slot_deg=("arm_slot_deg", "mean"), RelHeight=("RelHeight", "mean"),
                                           PitcherThrows=("PitcherThrows", "first")).reset_index()
    fb = s[s["pitch_id"].isin(FASTBALL_TYPES)]
    counts = fb.groupby(key + ["pitch_id"], observed=True).size().rename("n").reset_index()
    counts = counts.sort_values(key + ["n", "pitch_id"], ascending=[True, True, False, True])
    primary = counts.drop_duplicates(key)[key + ["pitch_id"]].rename(columns={"pitch_id": "primary_fb"})
    fbv = fb.merge(primary, on=key)
    fbv = fbv[fbv["pitch_id"] == fbv["primary_fb"]].groupby(key, observed=True).RelSpeed.mean().rename("fb_velo").reset_index()
    ars = ars.merge(py, on=key, how="left").merge(fbv, on=key, how="left")
    ars["rho_cdmx_used"] = float(rho)
    return _round(ars)


def _movement_grid(s: pd.DataFrame, bin_in: float, min_cell_n: int) -> pd.DataFrame:
    g = s.dropna(subset=["InducedVertBreak", "HorzBreak_norm"]).copy()
    g["IVB_bin"] = (np.floor(g["InducedVertBreak"] / bin_in) * bin_in + bin_in / 2).round(2)
    g["HB_norm_bin"] = (np.floor(g["HorzBreak_norm"] / bin_in) * bin_in + bin_in / 2).round(2)
    grid = g.groupby(["IVB_bin", "HB_norm_bin", "pitch_id"], observed=True).agg(
        n_pitches=("pitch_id", "size"),
        stuff_plus_neutral=("stuff_plus_neutral", "mean"),
        stuff_plus_cdmx=("stuff_plus_cdmx", "mean"),
        delta_stuff_cdmx=("delta_stuff_cdmx", "mean"),
    ).reset_index()
    return _round(grid[grid["n_pitches"] >= min_cell_n].reset_index(drop=True))


def _write_report(table: pd.DataFrame, path: Path) -> None:
    if "PitchUID" in table.columns:
        raise ValueError(f"{path.name} no puede contener PitchUID")
    table.to_csv(path, index=False)
    if path.stat().st_size >= MAX_REPORT_BYTES:
        raise ValueError(f"{path.name} pesa {path.stat().st_size} bytes (limite < 2 MB)")


def export_scores(
    df: pd.DataFrame,
    output_root: Path | str,
    *,
    deliverables_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    rho_cdmx: float | None = None,
    models: dict[str, Any] | None = None,
    chunk_size: int = 100_000,
    bin_in: float = 2.0,
    min_cell_n: int = 5,
    min_pitches_alt: int = 30,
    min_bip_alt: int = 5,
    write_csv_gz: bool = True,
) -> dict[str, Any]:
    """Puntua ``df`` (dataset procesado) y escribe los entregables bajo ``output_root``.

    Por defecto las salidas van a ``output_root/{data/deliverables, reports/leaderboards,
    models}``. ``models`` permite inyectar los cuatro modelos ya ajustados (llaves
    ``movement_translator``, ``stuff_model``, ``location_model``, ``pitching_model``);
    si es None se ajustan con year <= 2025. Devuelve rutas y tablas agregadas.
    """
    root = Path(output_root)
    deliver = Path(deliverables_dir) if deliverables_dir else root / "data" / "deliverables"
    reports = Path(reports_dir) if reports_dir else root / "reports" / "leaderboards"
    mdir = Path(models_dir) if models_dir else root / "models"
    for d in (deliver, reports, mdir):
        d.mkdir(parents=True, exist_ok=True)
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas (usa pitch_id invariante): {missing}")

    valid = df[df["PitchUID"].notna()]
    valid = valid.drop_duplicates(subset="PitchUID", keep="first")
    if valid.empty:
        raise ValueError("No hay PitchUID validos para exportar")

    if models is None:
        models = _fit_models(valid)
    rho = float(rho_cdmx) if rho_cdmx is not None else float(models["stuff_model"].rho_extreme_median_)

    parts = []
    for start in range(0, len(valid), chunk_size):
        parts.append(_score_chunk(valid.iloc[start:start + chunk_size], models, rho))
    scored = pd.concat(parts, ignore_index=True)
    scored["pitch_id"] = scored["pitch_id"].astype(str)

    deliver_cols = ID_COLS + SCORE_COLS + EXP_COLS
    deliverable = scored[deliver_cols].copy()
    for col in SCORE_COLS + EXP_COLS:
        deliverable[col] = deliverable[col].astype(np.float32)
    paths: dict[str, Path] = {"scores_parquet": deliver / f"{SCORES_FILE}.parquet"}
    deliverable.to_parquet(paths["scores_parquet"], index=False)
    if write_csv_gz:
        paths["scores_csv_gz"] = deliver / f"{SCORES_FILE}.csv.gz"
        deliverable.to_csv(paths["scores_csv_gz"], index=False, compression="gzip")

    tables = {
        "pitcher_season_leaderboard": _leaderboard(
            scored, float(models["stuff_model"].sigma_neutral_), min_pitches_alt, min_bip_alt),
        "pitcher_arsenal_breakdown": _arsenal(scored, rho),
        "movement_map_grid": _movement_grid(scored, bin_in, min_cell_n),
    }
    for name, table in tables.items():
        paths[name] = reports / f"{name}.csv"
        _write_report(table, paths[name])

    for name, model in models.items():
        paths[name] = mdir / f"{name}.joblib"
        joblib.dump(model, paths[name])

    return {"paths": paths, "tables": tables, "n_rows": len(deliverable), "rho_cdmx": rho,
            "n_dropped_invalid_or_duplicate": int(len(df) - len(valid))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exportador del Entregable 1 (scores por PitchUID + agregados)")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "stuff_processed_df.parquet")
    parser.add_argument("--deliverables-dir", type=Path, default=BASE_DIR / "data" / "deliverables")
    parser.add_argument("--reports-dir", type=Path, default=BASE_DIR / "reports" / "leaderboards")
    parser.add_argument("--models-dir", type=Path, default=BASE_DIR / "models")
    parser.add_argument("--rho-cdmx", type=float, default=None,
                        help="rho objetivo CDMX; por defecto la mediana de Extreme Altitude del modelo Stuff")
    parser.add_argument("--no-csv-gz", action="store_true", help="omite el .csv.gz del archivo por PitchUID")
    args = parser.parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"No existe dataset procesado: {args.input}. Ejecuta primero HACK-008.")
    data = pd.read_parquet(args.input)
    result = export_scores(
        data, BASE_DIR, deliverables_dir=args.deliverables_dir, reports_dir=args.reports_dir,
        models_dir=args.models_dir, rho_cdmx=args.rho_cdmx, write_csv_gz=not args.no_csv_gz,
    )
    print(f"Exportadas {result['n_rows']:,} filas (rho_cdmx={result['rho_cdmx']:.4f}); "
          f"descartadas por PitchUID nulo/duplicado: {result['n_dropped_invalid_or_duplicate']}")
    for name, path in result["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

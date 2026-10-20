"""Validación de Stuff+, Location+ y Pitching+ frente al baseline por conteo.

Las transformaciones aprendidas (incluido PhysicsMovementTranslator) se ajustan
por separado dentro de cada partición, usando solo filas de entrenamiento.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

from src.baseline import CountBaselineModel
from src.config import BASE_DIR, TARGET_COLS
from src.models.location_pitching import LocationPlusModel, PitchingPlusModel
from src.models.movement import PhysicsMovementTranslator
from src.models.stuff import StuffPlusModel
from src.targets import attach_targets, fit_lmb_run_values
from src.splits import (
    get_pitcher_holdout_splits,
    get_park_holdout_split,
    get_season_holdout_split,
)

MODEL_NAMES = ("Stuff+", "Location+", "Pitching+")
BINARY_TARGETS = [c for c in TARGET_COLS if c != "xrv_pitcher"]


def _make_models(train: pd.DataFrame) -> dict[str, object]:
    """Ajusta cada transformador/modelo únicamente en el subset recibido."""
    translator = PhysicsMovementTranslator().fit(train)
    return {
        "Stuff+": StuffPlusModel().fit(train, movement_translator=translator),
        "Location+": LocationPlusModel().fit(train),
        "Pitching+": PitchingPlusModel(movement_translator=translator).fit(train),
    }


def _predictions(model: object, frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model, StuffPlusModel):
        return model.predict(frame)
    if isinstance(model, LocationPlusModel):
        return model.predict(frame)
    if isinstance(model, PitchingPlusModel):
        return model.predict(frame)
    result = model.predict(frame)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result, index=frame.index, columns=TARGET_COLS)


def _metrics(name: str, split_name: str, truth: pd.DataFrame, prediction: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for target in TARGET_COLS:
        y = pd.to_numeric(truth[target], errors="coerce").to_numpy(float)
        p = pd.to_numeric(prediction[target], errors="coerce").to_numpy(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError(f"NaN/inf en {target} ({name}, {split_name})")
        row = {"holdout": split_name, "model": name, "target": target, "n": len(y),
               "rmse": float(mean_squared_error(y, p) ** .5),
               "mae": float(mean_absolute_error(y, p)), "log_loss": np.nan}
        row["corr_with_holdout_xrv"] = (
            float(pearsonr(p, y).statistic) if target == "xrv_pitcher" and len(y) >= 2
            and np.std(p) > 0 and np.std(y) > 0 else np.nan
        )
        if target in BINARY_TARGETS:
            row["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        rows.append(row)
    return rows


def _calibration(name: str, split_name: str, truth: pd.DataFrame, prediction: pd.DataFrame) -> list[dict]:
    y = pd.to_numeric(truth["xrv_pitcher"], errors="coerce")
    p = pd.to_numeric(prediction["xrv_pitcher"], errors="coerce")
    bins = pd.qcut(p.rank(method="first"), q=min(10, len(p)), labels=False, duplicates="drop")
    temp = pd.DataFrame({"actual": y, "predicted": p, "decile": bins})
    return [dict(holdout=split_name, model=name, decile=int(i + 1), n=int(len(g)),
                 mean_predicted=float(g.predicted.mean()), mean_observed=float(g.actual.mean()))
            for i, (_, g) in enumerate(temp.groupby("decile", observed=True))]


def _score_split(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, label: str,
    model_factory: Callable[[pd.DataFrame], dict[str, object]] = _make_models,
) -> tuple[list[dict], list[dict]]:
    train, test = df.loc[train_idx].copy(), df.loc[test_idx].copy()
    if train.empty or test.empty:
        raise ValueError(f"Partición vacía: {label}")
    # Recalibra xRV con outcomes de train en cada partición cuando están disponibles.
    required_rv = {"RunsScored", "OutsOnPlay", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes"}
    if required_rv.issubset(train.columns):
        rv_table = fit_lmb_run_values(train)
        train = attach_targets(train, rv_table)
        test = attach_targets(test, rv_table)
    # Tanto el traductor como los tres modelos nacen y se ajustan aquí con train.
    models = model_factory(train)
    baseline = CountBaselineModel().fit(train)
    base_pred = baseline.predict(test)
    metric_rows: list[dict] = _metrics("CountBaselineModel", label, test, base_pred)
    calibration_rows: list[dict] = _calibration("CountBaselineModel", label, test, base_pred)
    for name, model in models.items():
        pred = _predictions(model, test)
        metric_rows.extend(_metrics(name, label, test, pred))
        calibration_rows.extend(_calibration(name, label, test, pred))
    return metric_rows, calibration_rows


def _stability(df: pd.DataFrame, model_factory: Callable[[pd.DataFrame], dict[str, object]] = _make_models) -> pd.DataFrame:
    train = df[df.year.isin([2024, 2025])].copy()
    # Estabilidad descriptiva in-sample sobre 2024-2025; los holdouts de métricas son OOS.
    models = model_factory(train)
    rows = []
    for name, model in models.items():
        score_col = {"Stuff+": "stuff_plus_neutral", "Location+": "location_plus_neutral",
                     "Pitching+": "pitching_plus_neutral"}[name]
        if isinstance(model, StuffPlusModel):
            scored = model.predict_scores(train)
            score_col = "stuff_plus_neutral"
        elif isinstance(model, LocationPlusModel):
            scored = model.predict_scores(train)
        else:
            scored = model.predict_scores(train)
        if score_col not in scored:
            # predict_scores methods return their score columns; retain clear failure if contract drifts.
            raise ValueError(f"{name} no produjo {score_col}")
        temp = train[["pitcher_anon_id", "year"]].copy()
        temp["score"] = scored[score_col].to_numpy()
        counts = temp.groupby(["pitcher_anon_id", "year"]).size()
        means = temp.groupby(["pitcher_anon_id", "year"]).score.mean().unstack()
        valid = []
        for pitcher in means.index:
            if (pitcher, 2024) in counts.index and (pitcher, 2025) in counts.index and counts.loc[(pitcher, 2024)] >= 150 and counts.loc[(pitcher, 2025)] >= 150:
                valid.append(pitcher)
        pair = means.reindex(valid).dropna(subset=[2024, 2025]) if valid else pd.DataFrame(columns=[2024, 2025])
        n = len(pair)
        rows.append({"model": name, "score_basis": "in_sample_train_2024_2025", "n_pitchers": n,
                     "pearson_r": float(pearsonr(pair[2024], pair[2025]).statistic) if n >= 2 else np.nan,
                     "spearman_r": float(spearmanr(pair[2024], pair[2025]).statistic) if n >= 2 else np.nan})
    return pd.DataFrame(rows)


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    table.to_csv(temp, index=False)
    temp.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run_validation(
    df: pd.DataFrame,
    reports_dir: Path | str | None = None,
    model_factory: Callable[[pd.DataFrame], dict[str, object]] = _make_models,
    *,
    resume: bool = False,
    run_signature: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Corre los holdouts oficiales con checkpoints agregados reanudables."""
    partitions = [(f"pitcher_fold_{i + 1}", tr, te) for i, (tr, te) in enumerate(get_pitcher_holdout_splits(df, 5))]
    partitions += [("season", *get_season_holdout_split(df)), ("park", *get_park_holdout_split(df))]
    labels = [p[0] for p in partitions]
    out = Path(reports_dir) if reports_dir is not None else None
    status_path = out / "run_status.json" if out is not None else None
    signature = run_signature or f"rows={len(df)};columns={','.join(df.columns)}"
    completed: list[str] = []
    metrics = pd.DataFrame()
    calibration = pd.DataFrame()
    stability = pd.DataFrame()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    if resume:
        if out is None or status_path is None or not status_path.exists():
            raise ValueError("--resume requiere run_status.json existente en reports-dir")
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        if saved.get("run_signature") != signature:
            raise ValueError("El input/configuración no coincide con el checkpoint; no se reanudó")
        completed = [name for name in saved.get("completed_holdouts", []) if name in labels]
        if completed:
            metrics = pd.read_csv(out / "holdout_metrics_partial.csv")
            calibration = pd.read_csv(out / "calibration_deciles_partial.csv")
            metrics = metrics[metrics.holdout.isin(completed)].copy()
            calibration = calibration[calibration.holdout.isin(completed)].copy()
        if saved.get("stability_complete") and (out / "year_to_year_stability_partial.csv").exists():
            stability = pd.read_csv(out / "year_to_year_stability_partial.csv")
    elif out is not None:
        _atomic_json({"complete": False, "run_signature": signature, "completed_holdouts": [], "stability_complete": False}, status_path)

    for label, train_idx, test_idx in partitions:
        if label in completed:
            print(f"Reanudando: se omite {label} (checkpoint agregado presente)", flush=True)
            continue
        print(f"Validando {label}: train={len(train_idx):,}, test={len(test_idx):,}", flush=True)
        m, c = _score_split(df, train_idx, test_idx, label, model_factory)
        metrics = pd.concat([metrics, pd.DataFrame(m)], ignore_index=True)
        calibration = pd.concat([calibration, pd.DataFrame(c)], ignore_index=True)
        completed.append(label)
        if out is not None:
            _atomic_csv(metrics, out / "holdout_metrics_partial.csv")
            _atomic_csv(calibration, out / "calibration_deciles_partial.csv")
            _atomic_json({"complete": False, "run_signature": signature, "completed_holdouts": completed, "stability_complete": False}, status_path)

    if stability.empty:
        stability = _stability(df, model_factory)
        if out is not None:
            _atomic_csv(stability, out / "year_to_year_stability_partial.csv")
            _atomic_json({"complete": False, "run_signature": signature, "completed_holdouts": completed, "stability_complete": True}, status_path)
    outputs = {
        "holdout_metrics": metrics,
        "calibration_deciles": calibration,
        "year_to_year_stability": stability,
    }
    if out is not None:
        for stem, table in outputs.items():
            _atomic_csv(table, out / f"{stem}.csv")
        _atomic_json({"complete": True, "run_signature": signature, "completed_holdouts": labels, "stability_complete": True}, status_path)
    return outputs



def _build_run_signature(input_path: Path) -> str:
    """Firma portable del contenido de entrada y fuentes de modelado/configuración."""
    digest = hashlib.sha256()
    with input_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    sources = [
        Path(__file__), BASE_DIR / "src" / "baseline.py", BASE_DIR / "src" / "config.py",
        BASE_DIR / "src" / "models" / "movement.py", BASE_DIR / "src" / "models" / "stuff.py",
        BASE_DIR / "src" / "models" / "location_pitching.py", BASE_DIR / "src" / "splits.py",
        BASE_DIR / "src" / "targets.py",
    ]
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description="Validación de modelos AeroVision contra baseline por conteo")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "stuff_processed_df.parquet")
    parser.add_argument("--reports-dir", type=Path, default=BASE_DIR / "reports" / "validation")
    parser.add_argument("--resume", action="store_true", help="reanuda desde checkpoints agregados compatibles")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"No existe dataset procesado: {args.input}. Ejecuta primero HACK-008.")
    data = pd.read_parquet(args.input)
    signature = _build_run_signature(args.input)
    results = run_validation(data, args.reports_dir, resume=args.resume, run_signature=signature)
    print(f"Validación completada: {len(data):,} filas; reportes en {args.reports_dir}")
    # No se fuerza superioridad: skills negativos representan métricas peores al baseline.
    print(results["holdout_metrics"].groupby(["holdout", "model", "target"])[["rmse", "mae", "log_loss"]].mean().to_string())


if __name__ == "__main__":
    main()





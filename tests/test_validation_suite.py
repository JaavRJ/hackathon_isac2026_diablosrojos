from __future__ import annotations

import numpy as np
import pandas as pd
from analysis import run_validation as validation
from analysis.run_validation import _build_run_signature, _metrics, _score_split
from src.config import TARGET_COLS


def _frame(n=24):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"count": ["0-0", "1-1"] * (n // 2), "Balls": [0, 1] * (n // 2),
        "Strikes": [0, 1] * (n // 2), "year": [2024] * n,
        "pitcher_anon_id": [f"p{i}" for i in range(n)], "altitude_category": ["No Altitude"] * n})
    for target in TARGET_COLS:
        df[target] = rng.integers(0, 2, n).astype(float) if target != "xrv_pitcher" else rng.normal(size=n)
    return df


def test_metrics_report_rmse_mae_and_binary_logloss():
    truth = pd.DataFrame({c: [0., 1.] for c in TARGET_COLS})
    pred = pd.DataFrame({c: [.2, .8] for c in TARGET_COLS})
    pred["xrv_pitcher"] = [.25, .75]
    rows = _metrics("model", "unit", truth, pred)
    binary = next(row for row in rows if row["target"] == "target_whiff")
    continuous = next(row for row in rows if row["target"] == "xrv_pitcher")
    assert binary["log_loss"] > 0
    assert np.isnan(continuous["log_loss"])
    assert continuous["rmse"] == .25
    assert continuous["mae"] == .25
    assert np.isfinite(continuous["corr_with_holdout_xrv"])


def test_score_split_factory_receives_training_rows_only():
    df = _frame()
    train_idx, test_idx = df.index[:16].to_numpy(), df.index[16:].to_numpy()
    seen = {}
    class DummyModel:
        def predict(self, frame):
            return pd.DataFrame({col: np.full(len(frame), .5) for col in TARGET_COLS}, index=frame.index)
    def factory(train):
        seen["indices"] = set(train.index)
        return {"Stuff+": DummyModel(), "Location+": DummyModel(), "Pitching+": DummyModel()}
    metrics, calibration = _score_split(df, train_idx, test_idx, "tiny", factory)
    assert seen["indices"] == set(train_idx)
    assert all(row["n"] == len(test_idx) for row in metrics)
    assert calibration
    assert all(row["holdout"] == "tiny" for row in calibration)


def test_metrics_dont_mask_bad_model_performance():
    truth = pd.DataFrame({c: [0., 1.] for c in TARGET_COLS})
    pred = pd.DataFrame({c: [.5, .5] for c in TARGET_COLS})
    pred["xrv_pitcher"] = [10., -10.]
    row = next(r for r in _metrics("model", "season", truth, pred) if r["target"] == "xrv_pitcher")
    assert row["rmse"] > 1
    assert row["mae"] > 1



def test_resume_skips_completed_holdouts(tmp_path, monkeypatch):
    df = _frame(24)
    monkeypatch.setattr(validation, "get_pitcher_holdout_splits", lambda frame, n_splits: [(np.arange(8), np.arange(8, 12)), (np.arange(12), np.arange(12, 16))])
    monkeypatch.setattr(validation, "get_season_holdout_split", lambda frame: (np.arange(8), np.arange(8, 12)))
    monkeypatch.setattr(validation, "get_park_holdout_split", lambda frame: (np.arange(12), np.arange(12, 16)))
    monkeypatch.setattr(validation, "_stability", lambda frame, factory: pd.DataFrame([{"model": "fixture"}]))
    reports = tmp_path / "validation"
    reports.mkdir()
    (reports / "run_status.json").write_text('{"complete": false, "run_signature": "same", "completed_holdouts": ["pitcher_fold_1", "pitcher_fold_2"], "stability_complete": false}', encoding="utf-8")
    pd.DataFrame([{"holdout": name, "model": "saved", "target": "xrv_pitcher", "n": 1,
                   "rmse": 0., "mae": 0., "log_loss": np.nan, "corr_with_holdout_xrv": np.nan}
                  for name in ["pitcher_fold_1", "pitcher_fold_2"]]).to_csv(reports / "holdout_metrics_partial.csv", index=False)
    pd.DataFrame([{"holdout": name, "model": "saved", "decile": 1, "n": 1,
                   "mean_predicted": 0., "mean_observed": 0.}
                  for name in ["pitcher_fold_1", "pitcher_fold_2"]]).to_csv(reports / "calibration_deciles_partial.csv", index=False)
    calls = []
    class DummyModel:
        def predict(self, frame):
            return pd.DataFrame({col: np.full(len(frame), .5) for col in TARGET_COLS}, index=frame.index)
    def factory(train):
        calls.append(set(train.index))
        return {"Stuff+": DummyModel(), "Location+": DummyModel(), "Pitching+": DummyModel()}
    outputs = validation.run_validation(df, reports, factory, resume=True, run_signature="same")
    assert len(calls) == 2  # solo season y park; pitcher folds se recuperaron del checkpoint
    assert {"pitcher_fold_1", "pitcher_fold_2", "season", "park"}.issubset(set(outputs["holdout_metrics"].holdout))
    assert (reports / "holdout_metrics.csv").exists()
    assert '"complete": true' in (reports / "run_status.json").read_text(encoding="utf-8")


def test_run_signature_uses_content_not_location_or_mtime(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "nested" / "second.parquet"
    second.parent.mkdir()
    first.write_bytes(b"same input content")
    second.write_bytes(b"same input content")
    assert _build_run_signature(first) == _build_run_signature(second)
    second.write_bytes(b"different input content")
    assert _build_run_signature(first) != _build_run_signature(second)

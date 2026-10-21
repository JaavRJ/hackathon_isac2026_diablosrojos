"""Pruebas de HACK-014: exportador del Entregable 1 sobre el fixture sintetico."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from analysis.export_scores import SCORE_COLS, export_scores
from src.pipeline_step1 import build_processed_dataset


@pytest.fixture(scope="module")
def exported(synthetic_lmb_df, tmp_path_factory):
    proc, _ = build_processed_dataset(synthetic_lmb_df)
    root = tmp_path_factory.mktemp("export")
    result = export_scores(proc, root, min_cell_n=1, chunk_size=500,
                           min_pitches_alt=1, min_bip_alt=1)
    return proc, root, result


def test_no_pitchuid_and_small_csvs_in_reports(exported):
    _, root, result = exported
    reports = root / "reports" / "leaderboards"
    files = sorted(reports.glob("*.csv"))
    assert {f.name for f in files} == {
        "pitcher_season_leaderboard.csv", "pitcher_arsenal_breakdown.csv", "movement_map_grid.csv"}
    for f in files:
        assert f.stat().st_size < 2 * 1024 * 1024
        assert "PitchUID" not in pd.read_csv(f).columns


def test_reports_are_aggregated_not_per_pitch(exported):
    proc, _, result = exported
    lb = result["tables"]["pitcher_season_leaderboard"]
    assert len(lb) < len(proc)
    assert not lb.duplicated(["pitcher_anon_id", "year"]).any()
    assert lb["n_pitches"].sum() == len(proc)
    assert {"stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx", "runs_prevented_xrv"} <= set(lb.columns)
    ars = result["tables"]["pitcher_arsenal_breakdown"]
    assert not ars.duplicated(["pitcher_anon_id", "year", "pitch_id"]).any()
    assert "AutoPitchType" not in ars.columns
    grid = result["tables"]["movement_map_grid"]
    assert {"IVB_bin", "HB_norm_bin", "pitch_id", "delta_stuff_cdmx"} <= set(grid.columns)


def test_pitch_file_has_all_valid_pitchuids_without_duplicates(exported):
    proc, root, result = exported
    out = pd.read_parquet(result["paths"]["scores_parquet"])
    assert out["PitchUID"].is_unique
    assert set(out["PitchUID"]) == set(proc["PitchUID"].dropna())
    assert len(out) == proc["PitchUID"].nunique()
    for col in ["pitcher_anon_id", "year", "pitch_id", *SCORE_COLS, "exp_target_whiff", "exp_xrv_pitcher"]:
        assert col in out.columns
    assert out[SCORE_COLS].notna().all().all()
    assert (root / "data" / "deliverables" / "stuff_plus_scores_by_pitch.csv.gz").exists()
    delta = out["stuff_plus_cdmx"] - out["stuff_plus_neutral"]
    assert (delta - out["delta_stuff_cdmx"]).abs().max() < 1e-3


def test_models_serialized_in_tmp_path(exported):
    _, root, result = exported
    for name in ("movement_translator", "stuff_model", "location_model", "pitching_model"):
        path = root / "models" / f"{name}.joblib"
        assert path.exists()
        assert result["paths"][name] == path
    assert getattr(joblib.load(root / "models" / "stuff_model.joblib"), "is_fitted_", False)


def test_arsenal_has_peer_filter_columns(exported):
    _, _, result = exported
    ars = result["tables"]["pitcher_arsenal_breakdown"]
    for col in ["PitcherThrows", "arm_slot_deg", "RelHeight", "fb_velo", "rho_cdmx_used", "n_pitches"]:
        assert col in ars.columns
    assert np.allclose(ars["rho_cdmx_used"], result["rho_cdmx"], atol=1e-4)
    assert ars["PitcherThrows"].notna().all()
    assert ars["fb_velo"].dropna().between(50, 110).all()
    # fb_velo es constante dentro de pitcher-temporada
    assert (ars.groupby(["pitcher_anon_id", "year"]).fb_velo.nunique(dropna=True) <= 1).all()


def test_leaderboard_three_term_decomposition_sums_to_total(exported):
    _, _, result = exported
    lb = result["tables"]["pitcher_season_leaderboard"]
    terms = ["canal_stuff", "canal_parque_bip", "residuo_no_atribuido"]
    for col in ["canal_total", *terms]:
        assert col in lb.columns
    ok = lb[terms].notna().all(axis=1)
    assert ok.any(), "el sintetico debe producir al menos un pitcher-temporada estimable"
    sums = lb.loc[ok, terms].sum(axis=1)
    np.testing.assert_allclose(sums, lb.loc[ok, "canal_total"], atol=1e-10)
    # NaN en bloque: nunca un termino relleno cuando falta otro
    assert (lb[["canal_total", *terms]].isna().all(axis=1) | ok).all()


def test_decomposition_is_explicit_nan_without_enough_n(exported, synthetic_lmb_df):
    proc, root, result = exported
    out = export_scores(proc, root / "strict", models=joblib_models(root), min_cell_n=1,
                        min_pitches_alt=10_000, min_bip_alt=10_000)
    lb = out["tables"]["pitcher_season_leaderboard"]
    assert {"canal_stuff", "canal_parque_bip", "residuo_no_atribuido"} <= set(lb.columns)
    assert lb[["canal_stuff", "canal_parque_bip", "residuo_no_atribuido"]].isna().all().all()
    assert (lb["canal_total"].isna()).all()


def joblib_models(root):
    return {n: joblib.load(root / "models" / f"{n}.joblib")
            for n in ("movement_translator", "stuff_model", "location_model", "pitching_model")}


def test_channel_columns_not_rounded_and_csv_roundtrip(exported):
    _, root, result = exported
    path = result["paths"]["pitcher_season_leaderboard"]
    lb = pd.read_csv(path)
    ok = lb[["canal_stuff", "canal_parque_bip", "residuo_no_atribuido"]].notna().all(axis=1)
    sums = lb.loc[ok, ["canal_stuff", "canal_parque_bip", "residuo_no_atribuido"]].sum(axis=1)
    np.testing.assert_allclose(sums, lb.loc[ok, "canal_total"], atol=1e-10)
    assert (lb.loc[ok, "canal_stuff"].abs() > 0).any()
    assert not np.allclose(lb.loc[ok, "canal_stuff"], lb.loc[ok, "canal_stuff"].round(4), atol=0, rtol=0)
    for f in path.parent.glob("*.csv"):
        assert f.stat().st_size < 2 * 1024 * 1024

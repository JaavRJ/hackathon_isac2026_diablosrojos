import numpy as np
import pandas as pd

from analysis.run_hypotheses import run_hypotheses


class _Translator:
    def translate_movement(self, frame, target_rho):
        return frame.copy()


class _StuffModel:
    is_fitted_ = True
    movement_translator_ = _Translator()
    rho_extreme_median_ = 0.8
    sigma_neutral_ = 0.5

    @staticmethod
    def predict_scores(frame, rho_cdmx):
        out = frame.copy()
        # A known same-row counterfactual delta in Score+ units.
        out["delta_stuff_cdmx"] = 2.0
        return out


def _fixture():
    rows = []
    # Deliberately provide conflicting AutoPitchType labels: all comparisons must
    # still be based on the invariant pitch_id.
    for pitcher in range(8):
        for pid, baseline in [("Four-Seam", 18.0), ("Changeup", 8.0)]:
            for altitude, shift in [("No Altitude", 0.0), ("Extreme Altitude", -2.0)]:
                for rep in range(3):
                    ivb = baseline + shift + pitcher * 0.03
                    rows.append({
                        "pitcher_anon_id": f"p{pitcher}", "pitch_id": pid,
                        "AutoPitchType": "Slider" if altitude == "No Altitude" else "Cutter",
                        "altitude_category": altitude, "InducedVertBreak": ivb,
                        "HorzBreak_norm": 4.0 + (shift * 0.2),
                        "TotalInducedBreak": float(np.hypot(ivb, 4.0 + shift * 0.2)),
                        "ZoneSpeed": 85.0 + shift * 0.1, "target_whiff": float(pid == "Changeup"),
                        "home_run": float(altitude == "Extreme Altitude" and rep == 0),
                        "delta_stuff_cdmx": shift + (1.0 if pid == "Changeup" else 0.0),
                        "xrv_pitcher": 0.5 - 0.02 * ivb + (0.2 if altitude == "Extreme Altitude" else 0),
                        "target_groundball": float(pid == "Changeup"), "SpinRate": 2200 + pitcher,
                        "SpinToBreakRatio": 0.02 * ivb,
                    })
    return pd.DataFrame(rows)


def test_tables_have_cluster_intervals_and_channel_sum(tmp_path):
    tables = run_hypotheses(_fixture(), tmp_path, seed=7, n_boot=80, stuff_model=_StuffModel())
    assert set(tables) == {"delta_by_pitch_shape.csv", "channel_decomposition.csv", "hypotheses_h1_h4_summary.csv"}
    for name in tables:
        saved = pd.read_csv(tmp_path / name)
        assert {"ci_low", "ci_high", "p_value", "n_pitchers"}.issubset(saved.columns)
    channels = tables["channel_decomposition.csv"].set_index("channel").estimate
    assert np.isclose(channels["stuff"] + channels["park"], channels["total"], atol=1e-12)
    assert "descriptivo" in tables["channel_decomposition.csv"].set_index("channel").loc["park", "interpretation"]


def test_pitch_shape_results_use_invariant_pitch_id_and_pitcher_clusters(tmp_path):
    table = run_hypotheses(_fixture(), tmp_path, seed=7, n_boot=80, stuff_model=_StuffModel())["delta_by_pitch_shape.csv"]
    assert set(table.pitch_id) == {"Four-Seam", "Changeup"}
    assert set(table.n_pitchers) == {8}
    # Extreme minus neutral should reproduce the constructed movement loss.
    ivb = table[table.metric.eq("InducedVertBreak")]
    assert np.allclose(ivb.estimate, -2.0)
    assert (ivb.ci_low <= ivb.estimate).all() and (ivb.ci_high >= ivb.estimate).all()


def test_bootstrap_draw_interval_is_not_bootstrapped_again():
    from analysis.run_hypotheses import _bootstrap_interval
    draws = np.arange(1000, dtype=float)
    estimate, low, high, _ = _bootstrap_interval(draws, 500.0)
    assert estimate == 500.0
    assert np.isclose(low, 24.975)
    assert np.isclose(high, 974.025)


def test_rank_deficient_h3_h4_are_explicitly_not_estimable(tmp_path):
    df = _fixture()
    # Constant pitcher GB rate and identical spin predictors make the interaction
    # and joint H4 design rank deficient; the script must not emit arbitrary slopes.
    df["target_groundball"] = 0.0
    df["SpinRate"] = 2200.0
    df["SpinToBreakRatio"] = 0.02
    table = run_hypotheses(df, tmp_path, seed=9, n_boot=20, stuff_model=_StuffModel())["hypotheses_h1_h4_summary.csv"]
    bad = table[table.hypothesis.str.startswith(("H3:", "H4:"))]
    assert bad.estimate.isna().all()


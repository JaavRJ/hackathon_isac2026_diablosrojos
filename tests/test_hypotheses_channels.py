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
                        "is_ball_in_play": float(rep != 2),
                        "ExitSpeed": 90.0 + rep * 2.0 + (2.0 if pid == "Changeup" else 0.0),
                        "Angle": 15.0 + rep * 7.0 + (11.0 if pid == "Changeup" else 0.0),
                        "delta_stuff_cdmx": shift + (1.0 if pid == "Changeup" else 0.0),
                        "xrv_pitcher": 0.5 - 0.02 * ivb + (0.2 if altitude == "Extreme Altitude" else 0),
                        "target_groundball": float(pid == "Changeup" and pitcher % 3 != 0), "SpinRate": 2200 + pitcher,
                        "SpinToBreakRatio": 0.02 * ivb,
                    })
    return pd.DataFrame(rows)


def test_three_term_decomposition_reconstructs_total(tmp_path):
    tables = run_hypotheses(_fixture(), tmp_path, seed=7, n_boot=80, stuff_model=_StuffModel())
    assert set(tables) == {"delta_by_pitch_shape.csv", "channel_decomposition.csv", "hypotheses_h1_h4_summary.csv"}
    for name in tables:
        saved = pd.read_csv(tmp_path / name)
        assert {"ci_low", "ci_high", "p_value", "n_pitchers"}.issubset(saved.columns)
    table = tables["channel_decomposition.csv"].set_index("channel")
    assert not any("two_channel" in c or "NOT_ESTIMABLE" in table.astype(str).values for c in table.columns)
    assert (table.decomposition_type == "three_term").all()
    est = table.estimate
    assert np.isclose(est["stuff"] + est["park"] + est["unallocated"], est["total"], rtol=0.0, atol=1e-10)
    # The residual is a genuine, non-trivial term with its own cluster inference.
    res = table.loc["unallocated"]
    assert not np.isclose(res.estimate, 0.0, atol=1e-8)
    for col in ("ci_low", "ci_high", "p_value", "n_pitchers"):
        assert np.isfinite(res[col])
    assert res.ci_low <= res.estimate <= res.ci_high
    assert res.n_pitchers == table.loc["park", "n_pitchers"] == 8
    assert "no se atribuye a ningún canal" in res.term_label
    assert "no atribuido" in res.interpretation
    # Residual is not added to park.
    assert np.isclose(est["park"], table.loc["park", "estimate"]) and est["park"] != est["park"] + est["unallocated"]
    shares = table.share_of_total
    assert est["total"] != 0
    assert np.isclose(shares["stuff"] + shares["park"] + shares["unallocated"], 1.0, atol=1e-10)
    assert np.isclose(shares["total"], 1.0)
    park_row = table.loc["park"]
    assert "BIP" in park_row.interpretation
    assert park_row.conditional_batted_ball_controls == "ExitSpeed, Angle; is_ball_in_play=1; matched pitchers only"


def test_share_of_total_is_nan_when_total_is_zero(tmp_path):
    df = _fixture()
    df["xrv_pitcher"] = 0.0
    table = run_hypotheses(df, tmp_path, seed=7, n_boot=20, stuff_model=_StuffModel())["channel_decomposition.csv"]
    assert table.share_of_total.isna().all()


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


def test_park_channel_uses_only_paired_batted_balls_and_ev_angle(tmp_path):
    df = _fixture()
    extreme_hr = df.altitude_category.eq("Extreme Altitude") & df.home_run.eq(1)
    df.loc[extreme_hr, "xrv_pitcher"] += 0.3
    base = run_hypotheses(df, tmp_path / "base", seed=13, n_boot=30, stuff_model=_StuffModel())["channel_decomposition.csv"].set_index("channel")
    expected_park = base.loc["park", "estimate"]
    assert expected_park > 0
    # A large change on non-BIP Extreme pitches changes total/unallocated but
    # cannot leak into the park channel estimate.
    non_bip_extreme = df.altitude_category.eq("Extreme Altitude") & df.is_ball_in_play.eq(0)
    df.loc[non_bip_extreme, "xrv_pitcher"] += 25.0
    changed = run_hypotheses(df, tmp_path / "changed", seed=13, n_boot=30, stuff_model=_StuffModel())["channel_decomposition.csv"].set_index("channel")
    assert np.isclose(changed.loc["park", "estimate"], expected_park)
    assert not np.isclose(changed.loc["total", "estimate"], base.loc["total", "estimate"])


def test_h3_excludes_pitchers_without_neutral_extreme_pair(tmp_path):
    df = _fixture()
    expected = run_hypotheses(df, tmp_path / "paired", seed=23, n_boot=40, stuff_model=_StuffModel())["hypotheses_h1_h4_summary.csv"]
    extra = df[(df.pitcher_anon_id == "p0") & (df.altitude_category == "Extreme Altitude")].copy()
    extra["pitcher_anon_id"] = "unpaired"
    extra["target_groundball"] = 1.0
    extra["home_run"] = 1.0
    extra["xrv_pitcher"] = 100.0
    actual = run_hypotheses(pd.concat([df, extra], ignore_index=True), tmp_path / "with_unpaired",
                            seed=23, n_boot=40, stuff_model=_StuffModel())["hypotheses_h1_h4_summary.csv"]
    for hypothesis in ["H3:paired_baseline_GB_x_delta:delta_xrv", "H3:paired_baseline_GB_x_delta:delta_hr_rate"]:
        lhs = expected.set_index("hypothesis").loc[hypothesis]
        rhs = actual.set_index("hypothesis").loc[hypothesis]
        assert lhs.n_pitchers == rhs.n_pitchers == 8
        assert np.isclose(lhs.estimate, rhs.estimate)


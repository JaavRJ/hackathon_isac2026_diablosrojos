import numpy as np
import pandas as pd
import pytest

from src.vision.prospect_card import (
    DISCLAIMER,
    DEFAULT_CATALOG,
    build_prospect_card,
    velo_narrowing_report,
    weighted_percentile,
)


def _synthetic(n_fb=10, n_sl=3):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_fb):
        rows.append(dict(pitcher_anon_id=f"p{i}", year=2025, pitch_id="Four-Seam",
                         n_pitches=int(rng.integers(20, 200)), PitcherThrows="Right",
                         arm_slot_deg=34 + i * 0.4, RelHeight=5.6 + 0.05 * i,
                         fb_velo=90 + 0.5 * i, stuff_plus_cdmx=95 + 3 * i,
                         delta_stuff_cdmx=-1.0 - 0.2 * i))
    for i in range(n_sl):
        rows.append(dict(pitcher_anon_id=f"s{i}", year=2025, pitch_id="Slider",
                         n_pitches=50, PitcherThrows="Right", arm_slot_deg=35,
                         RelHeight=5.7, fb_velo=91, stuff_plus_cdmx=100 + i,
                         delta_stuff_cdmx=0.5))
    # zurdo y fuera de ventana
    rows.append(dict(pitcher_anon_id="L1", year=2025, pitch_id="Four-Seam", n_pitches=80,
                     PitcherThrows="Left", arm_slot_deg=35, RelHeight=5.7, fb_velo=91,
                     stuff_plus_cdmx=300, delta_stuff_cdmx=9))
    rows.append(dict(pitcher_anon_id="far", year=2025, pitch_id="Four-Seam", n_pitches=80,
                     PitcherThrows="Right", arm_slot_deg=50, RelHeight=5.7, fb_velo=91,
                     stuff_plus_cdmx=400, delta_stuff_cdmx=9))
    return pd.DataFrame(rows)


def _fam(card, pid):
    return next(f for f in card["families"] if f["pitch_id"] == pid)


def test_weighted_percentiles_match_independent_recompute():
    df = _synthetic()
    card = build_prospect_card(35, "Right", catalog=df)
    fb = df[(df.pitch_id == "Four-Seam") & (df.PitcherThrows == "Right")
            & ((df.arm_slot_deg - 35).abs() <= 6)]
    v = fb.stuff_plus_cdmx.to_numpy()
    w = fb.n_pitches.to_numpy().astype(float)
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    pos = (np.cumsum(w) - w / 2) / w.sum()
    for p in (10, 25, 50, 75, 90):
        expected = np.interp(p / 100, pos, v)
        assert _fam(card, "Four-Seam")["stuff_plus_cdmx"][f"p{p}"] == pytest.approx(expected)
    # propiedades basicas
    assert weighted_percentile([1, 2, 3], [1, 1, 1], [50])[0] == pytest.approx(2.0)
    assert weighted_percentile([1, 100], [1000, 1], [50])[0] < 10


def test_filters_hand_window_height_velo():
    df = _synthetic()
    card = build_prospect_card(35, "Derecho", catalog=df)
    assert card["filters_applied"]["PitcherThrows"] == "Right"
    assert card["n_pitchers_total"] == 13  # sin zurdo ni 'far'
    assert _fam(card, "Four-Seam")["n_pitchers"] == 10
    left = build_prospect_card(35, "Zurdo", min_pitchers=1, catalog=df)
    assert left["n_pitchers_total"] == 1
    # altura
    h = build_prospect_card(35, "Right", rel_height_ft=5.25, min_pitchers=1, catalog=df)
    assert _fam(h, "Four-Seam")["n_pitchers"] == 4  # RelHeight <= 5.75
    # velocidad
    v = build_prospect_card(35, "Right", observed_velo_mph=90, min_pitchers=1, catalog=df)
    assert _fam(v, "Four-Seam")["n_pitchers"] == 4  # fb_velo 90..91.5
    assert v["sculptor_params"]["fb_velo"] == 90.0
    assert v["filters_applied"]["fb_velo_mph"]["tol"] == 1.5


def test_window_widens_with_error():
    df = _synthetic()
    a = build_prospect_card(35, "Right", catalog=df)
    b = build_prospect_card(35, "Right", slot_error_deg=2.0, catalog=df)
    c = build_prospect_card(35, "Right", slot_error_deg=5.0, catalog=df)
    assert a["window_deg"] == 6.0 and b["window_deg"] == 6.0 and c["window_deg"] == 6.0
    e = build_prospect_card(35, "Right", slot_error_deg=14.0, catalog=df)
    assert e["window_deg"] == 14.0
    assert e["n_pitchers_total"] == 13  # 'far' (15 deg de distancia) aun fuera
    d = build_prospect_card(35, "Right", slot_error_deg=16.0, catalog=df)
    assert d["window_deg"] == 16.0 and d["n_pitchers_total"] == 14


def test_invalid_error_and_nan_slot():
    df = _synthetic()
    for bad in (-3.0, float("nan")):
        c = build_prospect_card(35, "Right", slot_error_deg=bad, catalog=df)
        assert c["error_source"] == "aún no medido" and c["window_deg"] == 6.0
    with pytest.raises(ValueError):
        build_prospect_card(float("nan"), "Right", catalog=df)


def test_insufficient_peers():
    df = _synthetic(n_fb=10, n_sl=3)
    card = build_prospect_card(35, "Right", min_pitchers=8, catalog=df)
    sl = _fam(card, "Slider")
    assert sl["status"] == "INSUFFICIENT_PEERS"
    assert "stuff_plus_cdmx" not in sl and "delta_stuff_cdmx" not in sl
    assert _fam(card, "Four-Seam")["status"] == "OK"
    assert card["families"][-1]["pitch_id"] == "Slider"  # insuficientes al final


def test_sorted_by_median_desc():
    df = _synthetic(n_sl=9)
    card = build_prospect_card(35, "Right", min_pitchers=8, catalog=df)
    meds = [f["stuff_plus_cdmx"]["p50"] for f in card["families"] if f["status"] == "OK"]
    assert meds == sorted(meds, reverse=True)


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _all_keys(x)


def test_no_point_value_or_pitcher_id():
    df = _synthetic()
    card = build_prospect_card(35, "Right", observed_velo_mph=91, catalog=df)
    keys = set(_all_keys(card))
    forbidden = {"stuff_plus", "stuff_plus_prospect", "prospect_stuff_plus", "point_estimate",
                 "pitcher_anon_id", "value", "mean"}
    assert not (keys & forbidden)
    assert "pitcher_anon_id" not in str(card)
    assert card["disclaimer"] == DISCLAIMER == (
        "Rango condicionado a lo medible en 2D; el video no mide spin, quiebre ni "
        "velocidad y no se empareja con pitchers del dataset.")


def test_error_source():
    df = _synthetic()
    assert build_prospect_card(35, "Right", catalog=df)["error_source"] == "aún no medido"
    assert build_prospect_card(35, "Right", slot_error_deg=3.0, catalog=df)["error_source"] == "medido"


def test_slot_error_kind_overrides_label_only():
    df = _synthetic()
    base = build_prospect_card(35, "Right", slot_error_deg=9.0, catalog=df)
    for kind in ("sistemático supuesto", "medido (MAE de validación)", "combinado (MAE + sistemático)"):
        c = build_prospect_card(35, "Right", slot_error_deg=9.0, catalog=df, slot_error_kind=kind)
        assert c["error_source"] == kind
        assert c["window_deg"] == base["window_deg"] and c["families"] == base["families"]
    assert build_prospect_card(35, "Right", catalog=df, slot_error_kind=None)["error_source"] == "aún no medido"


def test_catalog_path_and_missing_columns(tmp_path):
    p = tmp_path / "cat.csv"
    _synthetic().to_csv(p, index=False)
    assert build_prospect_card(35, "Right", catalog=str(p))["n_pitchers_total"] == 13
    with pytest.raises(ValueError):
        build_prospect_card(35, "Right", catalog=_synthetic().drop(columns=["n_pitches"]))
    with pytest.raises(ValueError):
        build_prospect_card(35, "Ambidiestro", catalog=_synthetic())


def test_velo_narrowing_report_structure():
    rows = velo_narrowing_report(35, "Right", 91, min_pitchers=1, catalog=_synthetic())
    fb = next(r for r in rows if r["pitch_id"] == "Four-Seam")
    assert fb["width_p25_p75_without_velo"] is not None
    assert fb["width_p25_p75_with_velo"] is not None
    assert fb["n_pitchers_with_velo"] <= fb["n_pitchers_without_velo"]


def test_real_catalog_right_35deg():
    assert DEFAULT_CATALOG.exists()
    card = build_prospect_card(35.0, "Right")
    ok = [f for f in card["families"] if f["status"] == "OK"]
    assert ok, "al menos una familia con cifras"
    for f in ok:
        s = f["stuff_plus_cdmx"]
        assert s["p10"] <= s["p25"] <= s["p50"] <= s["p75"] <= s["p90"]
    meds = [f["stuff_plus_cdmx"]["p50"] for f in ok]
    assert meds == sorted(meds, reverse=True)

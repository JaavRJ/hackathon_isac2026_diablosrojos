"""Tests de la validacion de VeloVision con 2 anotadores (17-D). Sin MediaPipe ni videos reales:
la extraccion de pose se monkeypatchea y los "clips" son archivos de bytes sinteticos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import validate_vision as VV  # noqa: E402
from src.vision import pose_slot as ps  # noqa: E402
from tests.test_vision_core import H, W, dataset_slot, skeleton, wrist_for  # noqa: E402

SECRET = "istock_secreto"
HEIGHT = 6.0
THROWS = "R"
FRAME = 12
DETECTED = 15            # el "sistema" detecta 3 frames despues del anotado
VISION_HS = (6.5, 1.0)   # (rel_height, rel_side) que mide la vision


def _kp(rel_h, rel_s):
    """Frame de un derecho de frente; ambos tobillos en x=0.55 para que la escala humana (nariz->tobillo de
    apoyo) coincida con la de la vision (nariz->punto medio de tobillos)."""
    k = skeleton(THROWS, wrist_for(THROWS, HEIGHT, rel_h, rel_s))
    k[ps.R_ANKLE, :2] = k[ps.L_ANKLE, :2]
    k[ps.NOSE, 0] = k[ps.L_ANKLE, 0]      # distancia nariz-tobillo vertical, como supone wrist_for
    return k


def _seq(rel_h, rel_s):
    kp = np.stack([_kp(rel_h, rel_s) for _ in range(30)])
    return ps.PoseSequence(kp, 30.0, W, H)


def _rows(clip_id, annotator, kp, frame=FRAME):
    pts = {"nariz": ps.NOSE, "hombro": ps.R_SHOULDER, "codo": ps.R_ELBOW, "muneca": ps.R_WRIST,
           "cadera": ps.R_HIP, "tobillo_apoyo": ps.L_ANKLE}
    return [{"clip_id": clip_id, "frame_idx": frame, "annotator": annotator, "landmark": n,
             "x": float(kp[i, 0]), "y": float(kp[i, 1]), "throws": THROWS, "height_ft": HEIGHT,
             "timestamp": "2026-01-01T00:00:00+00:00"} for n, i in pts.items()]


@pytest.fixture
def env(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    seqs = {}

    def add_clip(code, name, rel_h=VISION_HS[0], rel_s=VISION_HS[1]):
        f = clips / name
        f.write_bytes(f"video-{code}-{name}".encode())
        seqs[VV.file_hash(f)] = _seq(rel_h, rel_s)
        return f

    monkeypatch.setattr(VV, "extract_clip", lambda p: seqs[VV.file_hash(p)])
    monkeypatch.setattr(VV.PS, "detect_release_frame",
                        lambda seq, throws: {"frame_idx": DETECTED, "confidence": 1.0, "candidates": [],
                                             "peak_score": 5.0, "wrist_ahead": True, "valid": True})

    def run(rows, registry_codes, n_files=None):
        reg = [{"clip_code": c, "file_hash": VV.file_hash(clips / n)} for c, n in registry_codes if (clips / n).exists()]
        reg += [{"clip_code": c, "file_hash": "deadbeef00000000"} for c, n in registry_codes if not (clips / n).exists()]
        pd.DataFrame(reg).to_csv(tmp_path / "registry.csv", index=False)
        pd.DataFrame(rows).to_csv(tmp_path / "ann.csv", index=False)
        out = tmp_path / "out" / "summary.csv"
        res = VV.run(tmp_path / "ann.csv", clips, tmp_path / "registry.csv", out, verbose=False)
        return res, out

    return add_clip, run, tmp_path


def _val(summary, metric, measure):
    r = summary[(summary["metric"] == metric) & (summary["measure"] == measure)].iloc[0]
    return r["value"], r["n"], r["note"]


SLOT = "arm_slot_dataset_deg"
BIO = "arm_angle_biomech_deg"


def test_icc_known_values():
    a = [1, 2, 3, 4, 5]
    icc, note = VV.icc_2_1(a, a)
    assert icc == pytest.approx(1.0) and note == ""
    icc, _ = VV.icc_2_1(a, [x + 1 for x in a])           # sesgo constante: ICC(2,1) = 5/6 (calculo a mano)
    assert icc == pytest.approx(5 / 6)
    icc, note = VV.icc_2_1([1.0], [1.0])
    assert icc is None and "2 clips" in note


def test_perfect_agreement_and_zero_error(env):
    add_clip, run, _ = env
    specs = {"C001": (6.5, 1.0), "C002": (5.8, 2.2), "C003": (6.9, 0.2)}
    rows, reg = [], []
    for i, (code, (h, s)) in enumerate(specs.items()):
        add_clip(code, f"{SECRET}_{i}.mp4", h, s)
        reg.append((code, f"{SECRET}_{i}.mp4"))
        for ann in "AB":
            rows += _rows(code, ann, _kp(h, s))
    res, _ = run(rows, reg)
    sm = res["summary"]
    for meas in (SLOT, BIO):
        assert _val(sm, "icc_2_1", meas)[0] == pytest.approx(1.0)
        assert _val(sm, "mean_diff_A_minus_B", meas)[0] == pytest.approx(0.0, abs=1e-6)
        assert _val(sm, "mae_vs_human_mean", meas)[0] == pytest.approx(0.0, abs=1e-6)
        assert _val(sm, "rmse_vs_human_mean", meas)[0] == pytest.approx(0.0, abs=1e-6)
    assert _val(sm, "n_both_annotators", "all")[0] == 3
    assert _val(sm, "n_one_annotator", "all")[0] == 0


def test_mae_against_known_shift_and_noisy_icc(env):
    """La vision mide (6.5, 1.0); los humanos marcan (6.0, 1.0) +/- una variacion simetrica entre A y B."""
    add_clip, run, _ = env
    clips_hs = {"C001": (6.5, 1.0), "C002": (5.8, 2.2), "C003": (6.9, 0.2), "C004": (6.2, 1.5)}
    rows, reg, exp_err, a_vals, b_vals = [], [], [], [], []
    for i, (code, (h, s)) in enumerate(clips_hs.items()):
        add_clip(code, f"{SECRET}_{i}.mp4", h, s)
        reg.append((code, f"{SECRET}_{i}.mp4"))
        ha, hb = h - 0.5 + 0.1 * i, h - 0.5 - 0.1 * i          # el promedio humano queda 0.5 ft por debajo
        rows += _rows(code, "A", _kp(ha, s)) + _rows(code, "B", _kp(hb, s))
        sa, sb = dataset_slot(ha, s), dataset_slot(hb, s)
        a_vals.append(sa)
        b_vals.append(sb)
        exp_err.append(dataset_slot(h, s) - (sa + sb) / 2)
    res, _ = run(rows, reg)
    sm = res["summary"]
    e = np.array(exp_err)
    assert _val(sm, "mae_vs_human_mean", SLOT)[0] == pytest.approx(np.abs(e).mean(), abs=1e-6)
    assert _val(sm, "rmse_vs_human_mean", SLOT)[0] == pytest.approx(np.sqrt((e ** 2).mean()), abs=1e-6)
    assert _val(sm, "bias_vs_human_mean", SLOT)[0] == pytest.approx(e.mean(), abs=1e-6)
    assert _val(sm, "mean_diff_A_minus_B", SLOT)[0] == pytest.approx(np.mean(np.array(a_vals) - np.array(b_vals)), abs=1e-6)
    icc = _val(sm, "icc_2_1", SLOT)[0]
    expected, _ = VV.icc_2_1(a_vals, b_vals)
    assert icc == pytest.approx(expected, abs=1e-6) and icc < 1.0
    assert _val(sm, "mae_vs_human_mean", SLOT)[1] == 4


def test_release_frame_error(env):
    add_clip, run, _ = env
    rows, reg = [], []
    for i, code in enumerate(("C001", "C002")):
        add_clip(code, f"{SECRET}_{i}.mp4")
        reg.append((code, f"{SECRET}_{i}.mp4"))
        for ann in "AB":
            rows += _rows(code, ann, _kp(*VISION_HS))
    res, _ = run(rows, reg)
    sm = res["summary"]
    assert _val(sm, "release_frame_mean_error_frames", "release_frame")[0] == pytest.approx(DETECTED - FRAME)
    assert _val(sm, "release_frame_mae_ms", "release_frame")[0] == pytest.approx((DETECTED - FRAME) / 30.0 * 1000.0)
    bc = res["by_clip"]
    assert (bc["release_frame_error_frames"] == DETECTED - FRAME).all()


def test_clip_without_file_is_omitted(env):
    add_clip, run, _ = env
    add_clip("C001", f"{SECRET}_0.mp4")
    rows = []
    for code in ("C001", "C002"):           # C002 esta en el registro pero su archivo no existe localmente
        for ann in "AB":
            rows += _rows(code, ann, _kp(*VISION_HS))
    res, _ = run(rows, [("C001", f"{SECRET}_0.mp4"), ("C002", f"{SECRET}_missing.mp4")])
    sm = res["summary"]
    assert _val(sm, "n_omitted_clips", "all")[0] == 1
    assert _val(sm, "n_both_annotators", "all")[0] == 1
    bc = res["by_clip"]
    assert bc.loc[bc["clip_code"] == "C002", "status"].iloc[0] == "omitido_sin_archivo"


def test_single_annotator_counts_and_is_not_in_metrics(env):
    add_clip, run, _ = env
    rows, reg = [], []
    for i, code in enumerate(("C001", "C002", "C003")):
        add_clip(code, f"{SECRET}_{i}.mp4", 6.0 + 0.3 * i, 1.0)
        reg.append((code, f"{SECRET}_{i}.mp4"))
        rows += _rows(code, "A", _kp(6.0 + 0.3 * i, 1.0))
        if i < 2:
            rows += _rows(code, "B", _kp(6.0 + 0.3 * i, 1.0))
    res, _ = run(rows, reg)
    sm = res["summary"]
    assert _val(sm, "n_both_annotators", "all")[0] == 2
    assert _val(sm, "n_one_annotator", "all")[0] == 1
    assert _val(sm, "icc_2_1", SLOT)[1] == 2


def test_no_filenames_in_csvs(env):
    add_clip, run, tmp = env
    rows, reg = [], []
    for i, code in enumerate(("C001", "C002")):
        add_clip(code, f"{SECRET}_{i}.mp4")
        reg.append((code, f"{SECRET}_{i}.mp4"))
        for ann in "AB":
            rows += _rows(code, ann, _kp(*VISION_HS))
    _, out = run(rows, reg)
    by_clip = out.with_name("vision_validation_by_clip.csv")
    for f in (out, by_clip):
        text = f.read_text(encoding="utf-8")
        assert SECRET not in text and ".mp4" not in text and str(tmp) not in text
    assert set(pd.read_csv(by_clip)["clip_code"]) == {"C001", "C002"}
    assert list(pd.read_csv(out).columns) == VV.SUMMARY_COLUMNS


def test_fewer_than_two_clips_leaves_icc_empty(env):
    add_clip, run, _ = env
    add_clip("C001", f"{SECRET}_0.mp4")
    rows = _rows("C001", "A", _kp(*VISION_HS)) + _rows("C001", "B", _kp(*VISION_HS))
    res, out = run(rows, [("C001", f"{SECRET}_0.mp4")])
    sm = pd.read_csv(out)
    r = sm[(sm["metric"] == "icc_2_1") & (sm["measure"] == SLOT)].iloc[0]
    assert pd.isna(r["value"]) and "2 clips" in r["note"] and r["n"] == 1


def test_recommendation_uses_combined_error(env):
    add_clip, run, _ = env
    rows, reg = [], []
    for i, code in enumerate(("C001", "C002")):
        add_clip(code, f"{SECRET}_{i}.mp4")
        reg.append((code, f"{SECRET}_{i}.mp4"))
        for ann in "AB":
            rows += _rows(code, ann, _kp(VISION_HS[0] - 0.5, VISION_HS[1]))
    res, _ = run(rows, reg)
    rec = res["recommendation"]
    assert rec["mae"] > 0 and rec["sys"] > 0
    assert rec["combined"] == pytest.approx(ps.combined_slot_error_deg(rec["mae"], rec["sys"]))

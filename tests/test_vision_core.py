"""Tests rapidos del nucleo de vision (17-A). Sin videos reales."""
import time

import cv2
import numpy as np
import pytest

from src.vision import pose_slot as ps
from src.vision.pose_slot import (PoseModelUnavailable, PoseSequence, annotate_frame,
                                  detect_release_frame, extract_pose_sequence, measure_release)

W, H = 640, 480           # tamano por defecto que asume el modulo para arreglos sinteticos
ANK_Y, NOSE_Y = 0.9, 0.2


def dataset_slot(rel_h, rel_s):
    """Copia literal de la formula de src/features/physical.py:190-191."""
    rad = np.arctan2(rel_h - 4.0, np.abs(rel_s) + 0.5)
    return float(np.clip(np.degrees(rad), 0.0, 90.0))


def skeleton(throws="R", wrist=None, vis=1.0):
    """Un frame (33,3) de pie y de frente. El lado derecho del sujeto queda a la izquierda de la imagen."""
    k = np.full((33, 3), np.nan)
    k[:, 2] = vis
    k[ps.NOSE, :2] = (0.5, NOSE_Y)
    k[ps.R_SHOULDER, :2], k[ps.L_SHOULDER, :2] = (0.42, 0.30), (0.58, 0.30)
    k[ps.R_ELBOW, :2], k[ps.L_ELBOW, :2] = (0.38, 0.38), (0.62, 0.38)
    k[ps.R_WRIST, :2], k[ps.L_WRIST, :2] = (0.36, 0.46), (0.64, 0.46)
    k[ps.R_HIP, :2], k[ps.L_HIP, :2] = (0.45, 0.55), (0.55, 0.55)
    k[ps.R_ANKLE, :2], k[ps.L_ANKLE, :2] = (0.45, ANK_Y), (0.55, ANK_Y)
    for i in range(33):
        if np.isnan(k[i, 0]):
            k[i, :2] = (0.5, 0.5)
    if wrist is not None:
        k[ps.R_WRIST if throws == "R" else ps.L_WRIST, :2] = wrist
    return k


def wrist_for(throws, height_ft, rel_h, rel_s):
    """Posicion normalizada de la muneca que produce (rel_h, rel_s) conocidos."""
    ft_px = ps.ANTHROPOMETRIC_NOSE_FRACTION * height_ft / ((ANK_Y - NOSE_Y) * H)
    sign = -1.0 if throws == "R" else 1.0       # el lado lanzador en la imagen
    return (0.5 + sign * rel_s / ft_px / W, ANK_Y - rel_h / ft_px / H)


@pytest.mark.parametrize("throws", ["R", "L"])
@pytest.mark.parametrize("rel_h,rel_s", [(6.5, 1.0), (5.8, 2.2), (4.6, 2.8), (7.0, -0.3), (3.5, 1.0), (6.9, 0.0)])
def test_arm_slot_matches_dataset_formula(throws, rel_h, rel_s):
    height = 6.0
    kp = skeleton(throws, wrist_for(throws, height, rel_h, rel_s))[None]
    m = measure_release(kp, 0, throws, height)
    assert m["rel_height_ft"] == pytest.approx(rel_h, abs=1e-9)
    assert m["rel_side_ft"] == pytest.approx(rel_s, abs=1e-9)
    assert m["arm_slot_dataset_deg"] == pytest.approx(dataset_slot(rel_h, rel_s), abs=1e-9)
    assert m["quality_ok"] and m["quality_score"] > 0.9 and m["quality_issues"] == []


def test_slot_independent_of_mirroring():
    kp = skeleton("R", wrist_for("R", 6.0, 6.5, 1.0))[None]
    mir = kp.copy()
    mir[..., 0] = 1.0 - mir[..., 0]
    a = measure_release(kp, 0, "R", 6.0)["arm_slot_dataset_deg"]
    b = measure_release(mir, 0, "R", 6.0)["arm_slot_dataset_deg"]
    assert a == pytest.approx(b, abs=1e-9)


def test_biomech_angle_sign_and_range():
    def bio(w):
        return measure_release(skeleton("R", w)[None], 0, "R", 6.0)["arm_angle_biomech_deg"]
    up, flat, down = bio((0.40, 0.10)), bio((0.30, 0.30)), bio((0.40, 0.50))
    assert 0 < up <= 90 and flat == pytest.approx(0.0, abs=1e-9) and -90 <= down < 0


def test_detect_release_finds_speed_peak():
    T, peak = 40, 25
    seq = np.stack([skeleton("R") for _ in range(T)])
    seq[:, ps.R_ELBOW, :2] = (0.40, 0.17)
    wr = np.tile(np.array([0.40, 0.45]), (T, 1))              # brazo recogido, casi quieto
    for t, xy in {peak - 2: (0.30, 0.20), peak - 1: (0.34, 0.08), peak: (0.40, 0.05),
                  peak + 1: (0.46, 0.08), peak + 2: (0.50, 0.14)}.items():
        wr[t] = xy                                             # barrido rapido con brazo extendido
    seq[:, ps.R_WRIST, :2] = wr
    res = detect_release_frame(seq, "R")
    assert abs(res["frame_idx"] - peak) <= 1
    assert 0.0 < res["confidence"] <= 1.0
    assert 1 <= len(res["candidates"]) <= 3
    assert res["candidates"][0]["frame_idx"] == res["frame_idx"]


def test_detect_release_tolerates_missing_landmarks():
    seq = np.stack([skeleton("R") for _ in range(30)])
    seq[5:12, ps.R_WRIST] = np.nan
    seq[:, ps.R_ELBOW, 2] = 0.0
    res = detect_release_frame(seq, "R")
    assert set(res) == {"frame_idx", "confidence", "candidates"}
    empty = detect_release_frame(np.full((10, 33, 3), np.nan), "L")
    assert empty["confidence"] == 0.0 and empty["candidates"] == []
    assert detect_release_frame(seq[:2], "R")["confidence"] == 0.0


def test_quality_drops_with_visibility_size_and_cropping():
    good = measure_release(skeleton("R", wrist_for("R", 6.0, 6.5, 1.0))[None], 0, "R", 6.0)
    lowvis = skeleton("R", wrist_for("R", 6.0, 6.5, 1.0), vis=0.2)[None]
    m_vis = measure_release(lowvis, 0, "R", 6.0)
    assert m_vis["quality_score"] < good["quality_score"] - 0.3 and m_vis["quality_issues"]
    assert not m_vis["quality_ok"]

    small = skeleton("R", wrist_for("R", 6.0, 6.5, 1.0))
    small[:, :2] = 0.5 + (small[:, :2] - 0.5) * 0.35
    m_small = measure_release(small[None], 0, "R", 6.0)
    assert m_small["quality_score"] < good["quality_score"]
    assert any("pequeño" in s for s in m_small["quality_issues"])

    cut = skeleton("R", wrist_for("R", 6.0, 6.5, 1.0))
    cut[[ps.L_ANKLE, ps.R_ANKLE], 1] = 1.05
    m_cut = measure_release(cut[None], 0, "R", 6.0)
    assert any("cortado" in s.lower() for s in m_cut["quality_issues"])
    assert m_cut["quality_score"] < good["quality_score"]


def test_no_person_and_bad_inputs():
    m = measure_release(np.full((3, 33, 3), np.nan), 1, "R", 6.0)
    assert m["quality_score"] < 0.3 and m["arm_slot_dataset_deg"] is None and m["quality_issues"]
    m2 = measure_release(skeleton("R")[None], 0, "R", 12.0)
    assert m2["arm_slot_dataset_deg"] is None and not m2["quality_ok"]
    with pytest.raises(ValueError):
        measure_release(skeleton("R")[None], 0, "X", 6.0)


def test_annotate_frame_returns_rgb_and_avoids_brand_red():
    kp = skeleton("R", wrist_for("R", 6.0, 6.5, 1.0))[None]
    m = measure_release(kp, 0, "R", 6.0)
    img = annotate_frame(kp, 0, m)
    assert img.dtype == np.uint8 and img.shape == (H, W, 3)
    assert not np.any(np.all(img == (0xCE, 0x11, 0x26), axis=2))
    assert (img != 255).any()
    seq = PoseSequence(kp, 30.0, 640, 360, [np.zeros((270, 480, 3), np.uint8)])
    assert annotate_frame(seq, 0, m).shape == (270, 480, 3)


# ------------------------------------------------------ extract_pose_sequence
def write_video(path, n=45, fps=30, size=(320, 240)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    for i in range(n):
        f = np.full((size[1], size[0], 3), 200, np.uint8)
        cv2.rectangle(f, (20 + i, 40), (60 + i, 120), (30, 60, 90), -1)
        vw.write(f)
    vw.release()


def test_extract_without_person_is_low_quality_and_respects_limits(tmp_path, monkeypatch):
    """Backend simulado 'sin persona': plomeria de cv2, max_seconds y target_width."""
    monkeypatch.setattr(ps, "_make_detector", lambda: (lambda rgb: None))
    v = tmp_path / "clip.mp4"
    write_video(v, n=45, fps=30, size=(1280, 720))
    seq = extract_pose_sequence(v, max_seconds=1.0, target_width=640)
    assert seq.width == 640 and seq.height == 360
    assert seq.n_frames == 30 and seq.fps == pytest.approx(30, abs=1)
    assert len(seq.frames) == 30 and seq.frames[0].shape[1] <= 480
    m = measure_release(seq, detect_release_frame(seq, "R")["frame_idx"], "R", 6.0)
    assert m["quality_score"] < 0.3 and not m["quality_ok"] and m["quality_issues"]
    assert annotate_frame(seq, 0, m).dtype == np.uint8
    write_video(tmp_path / "small.mp4", n=10, size=(320, 240))        # nunca amplia
    assert extract_pose_sequence(tmp_path / "small.mp4", target_width=640).width == 320


def test_extract_with_real_backend_or_clear_error(tmp_path):
    pytest.importorskip("mediapipe")
    v = tmp_path / "clip.mp4"
    write_video(v, n=30, fps=30, size=(320, 240))
    try:
        seq = extract_pose_sequence(v, max_seconds=0.5, target_width=256)
    except PoseModelUnavailable as e:
        assert "pose_landmarker" in str(e)
        pytest.skip("modelo .task de MediaPipe no disponible en este entorno")
    assert seq.width <= 256 and seq.n_frames <= 16
    m = measure_release(seq, detect_release_frame(seq, "R")["frame_idx"], "R", 6.0)
    assert not m["quality_ok"]                 # un rectangulo no es una persona


def test_missing_file_raises_valueerror(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_make_detector", lambda: (lambda rgb: None))
    with pytest.raises(ValueError):
        extract_pose_sequence(tmp_path / "nope.mp4")


def test_measurement_speed_is_milliseconds():
    """Rendimiento: deteccion+medicion de un clip de 6 s a 30 fps (180 frames)."""
    seq = np.stack([skeleton("R") for _ in range(180)])
    t0 = time.perf_counter()
    r = detect_release_frame(seq, "R")
    measure_release(seq, r["frame_idx"], "R", 6.0)
    per_frame_ms = (time.perf_counter() - t0) * 1000 / 180
    print(f"deteccion+medicion: {per_frame_ms:.3f} ms/frame")
    assert per_frame_ms < 5.0

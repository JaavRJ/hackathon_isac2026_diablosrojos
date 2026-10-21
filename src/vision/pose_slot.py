"""VeloVision 17-A: nucleo de vision (video monocular -> soltura -> arm slot).

Honestidad (AGENTS.md sec. 2, puntos 5 y 6): un video monocular NO mide spin,
eje, quiebre ni velocidad, y no se empareja con ningun ``pitcher_anon_id``.
Este modulo solo estima, en 2D, el punto de soltura relativo y el arm slot.

API de MediaPipe
----------------
Se usa la API clasica ``mp.solutions.pose`` si existe (mediapipe < 0.10.3x).
En mediapipe 1.x (probado: 1.1.1) esa API ya no existe, y se usa la API de
Tasks (``mediapipe.tasks.python.vision.PoseLandmarker``, modo IMAGE por frame),
que necesita un archivo de modelo ``pose_landmarker_*.task`` que NO viene en el
paquete. Se busca en: variable de entorno ``VELOVISION_POSE_MODEL``,
``models/pose_landmarker_{lite,full,heavy}.task`` o ``models/pose_landmarker.task``.
Si no esta, ``extract_pose_sequence`` lanza ``PoseModelUnavailable`` con
instrucciones. La medicion y la deteccion funcionan sin MediaPipe: aceptan un
``PoseSequence`` o directamente un arreglo (T, 33, 3|2) de keypoints
normalizados (x, y, visibilidad) en el orden de 33 landmarks de MediaPipe.

Convenciones
------------
* Coordenadas normalizadas: x en [0,1] hacia la derecha, y en [0,1] hacia abajo.
* ``throws`` es ``"R"`` o ``"L"``. Se usa el landmark ANATOMICO del lado
  lanzador (MediaPipe etiqueta derecha/izquierda del sujeto, no de la imagen).
* Un arreglo sin ``PoseSequence`` asume cuadro de 640x480 y 30 fps
  (``DEFAULT_WIDTH/HEIGHT/FPS``) para pasar de coordenadas normalizadas a pixeles.

Heuristicas documentadas
------------------------
* Escala pixeles->pies: ``ANTHROPOMETRIC_NOSE_FRACTION = 0.93``. La altura de la
  nariz sobre el suelo es ~0.93 de la estatura (el ojo esta a ~0.936 H y la nariz
  un poco abajo). Distancia nariz->tobillos (punto medio) en pixeles
  = 0.93 * estatura; se toma el percentil 80 de la distancia sobre los frames
  validos, para quedarse con el sujeto erguido y no con la zancada agachada.
* ``rel_height_ft``: altura de la muñeca sobre el tobillo de apoyo (el del lado
  opuesto al brazo lanzador), en pies. ``rel_side_ft``: distancia horizontal de
  la muñeca al centro del cuerpo (punto medio de caderas), positiva hacia el
  lado del brazo lanzador (el signo sale de la posicion de los hombros, asi que
  no depende de hacia donde mira la camara).
* ``arm_slot_dataset_deg``: formula EXACTA de ``src/features/physical.py:190``:
  ``clip(degrees(atan2(rel_height - 4.0, |rel_side| + 0.5)), 0, 90)``.
* ``arm_angle_biomech_deg``: angulo hombro->muñeca sobre la horizontal,
  ``atan2(hombro_y - muñeca_y, |dx|)`` en grados: +90 = brazo vertical arriba,
  0 = horizontal, valores negativos (hasta -90) si la muñeca queda por debajo
  del hombro. Solo informativo: NO filtra el catalogo.
* Release: maxima velocidad de la muñeca lanzadora (en torsos/segundo) ponderada
  por la extension del brazo (|hombro-muñeca| / longitud del brazo, umbral
  0.6-0.9) y por estar "por delante" del hombro (en 2D: mas lejos de la cadera
  que el hombro). Tolera landmarks faltantes (visibilidad < 0.3 => NaN).
* Calidad (0-1) = 0.5*visibilidad de landmarks clave + 0.25*tamaño del sujeto
  (alto nariz->tobillo / alto del cuadro, pleno >= 0.5) + 0.25*cuerpo completo
  en cuadro; x0.85 si la vista parece de perfil. Umbral sugerido ``QUALITY_MIN``.

Rendimiento (CPU, medido en 17-A): ver ``tests/test_vision_core.py`` (la parte
de medicion/deteccion es del orden de milisegundos por clip; el costo dominante
es la inferencia de MediaPipe por frame).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import cv2
import numpy as np

# ---------------------------------------------------------------- constantes
ANTHROPOMETRIC_NOSE_FRACTION = 0.93
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 640, 480, 30.0
VIS_MIN = 0.5          # visibilidad para considerar un landmark "visible"
VIS_NAN = 0.3          # por debajo => landmark tratado como faltante (NaN)
QUALITY_MIN = 0.6      # umbral sugerido para la puerta de calidad (17-C)
NUM_LANDMARKS = 33

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

# Esqueleto simple para dibujar (pares de indices).
_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
          (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]
_NAVY = (0x1F, 0x3A, 0x5F)   # RGB
_TEAL = (0x0B, 0x7A, 0x66)

_MODEL_ENV = "VELOVISION_POSE_MODEL"


class PoseModelUnavailable(RuntimeError):
    """No hay backend de pose utilizable (falta el modelo .task de MediaPipe)."""


@dataclass
class PoseSequence:
    """Landmarks por frame de un clip.

    landmarks: (T, 33, 3) float, columnas (x, y, visibilidad); NaN/0 = sin persona.
    fps/width/height: del clip procesado (width/height ya reducidos).
    frames: lista de T arreglos RGB uint8 reducidos (para la barra de frames),
        o None cuando la secuencia es sintetica.
    """

    landmarks: np.ndarray
    fps: float = DEFAULT_FPS
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    frames: Optional[list] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.landmarks = _as_lm_array(self.landmarks)

    @property
    def n_frames(self) -> int:
        return int(self.landmarks.shape[0])

    @classmethod
    def from_keypoints(cls, kp: Any, fps: float = DEFAULT_FPS,
                       width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> "PoseSequence":
        return cls(_as_lm_array(kp), fps, width, height, None)


SeqLike = Union[PoseSequence, np.ndarray, Sequence]


def _as_lm_array(kp: Any) -> np.ndarray:
    a = np.asarray(kp, dtype="float64")
    if a.ndim == 2:                      # un solo frame
        a = a[None]
    if a.ndim != 3 or a.shape[1] != NUM_LANDMARKS or a.shape[2] not in (2, 3):
        raise ValueError(f"keypoints deben tener forma (T, 33, 2|3); recibido {a.shape}")
    if a.shape[2] == 2:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,))], axis=2)
    return a


def _resolve(seq: SeqLike) -> PoseSequence:
    return seq if isinstance(seq, PoseSequence) else PoseSequence.from_keypoints(seq)


def _idx(throws: str) -> dict:
    t = str(throws).strip().upper()[:1]
    if t not in ("R", "L"):
        raise ValueError("throws debe ser 'R' o 'L'")
    if t == "R":
        return dict(sh=R_SHOULDER, el=R_ELBOW, wr=R_WRIST, hip=R_HIP, ank=R_ANKLE,
                    osh=L_SHOULDER, ohip=L_HIP, oank=L_ANKLE)
    return dict(sh=L_SHOULDER, el=L_ELBOW, wr=L_WRIST, hip=L_HIP, ank=L_ANKLE,
                osh=R_SHOULDER, ohip=R_HIP, oank=R_ANKLE)


def _px(seq: PoseSequence, landmark: int, nan_below: float = VIS_NAN) -> np.ndarray:
    """(T, 2) posicion en pixeles; NaN donde la visibilidad es baja o falta."""
    a = seq.landmarks[:, landmark, :]
    xy = np.stack([a[:, 0] * seq.width, a[:, 1] * seq.height], axis=1)
    bad = ~np.isfinite(xy).all(axis=1) | ~(np.nan_to_num(a[:, 2]) >= nan_below)
    xy[bad] = np.nan
    return xy


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Punto medio tolerante a NaN (usa el que exista; NaN si faltan ambos)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([a, b]), axis=0)


def _vis(seq: PoseSequence, landmark: int) -> np.ndarray:
    return np.nan_to_num(seq.landmarks[:, landmark, 2], nan=0.0)


# --------------------------------------------------------------- extraccion
def _find_model() -> Optional[Path]:
    env = os.environ.get(_MODEL_ENV)
    cands = [Path(env)] if env else []
    root = Path(__file__).resolve().parents[2] / "models"
    cands += [root / f"pose_landmarker_{v}.task" for v in ("lite", "full", "heavy")]
    cands.append(root / "pose_landmarker.task")
    for c in cands:
        if c.is_file():
            return c
    return None


def _make_detector() -> Callable[[np.ndarray], Optional[np.ndarray]]:
    """Devuelve f(rgb_uint8) -> (33,3) [x,y,vis] o None si no hay persona."""
    import mediapipe as mp  # import tardio: el resto del modulo no lo necesita

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):   # API clasica
        pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def run_classic(rgb: np.ndarray) -> Optional[np.ndarray]:
            res = pose.process(rgb)
            if not res.pose_landmarks:
                return None
            return np.array([[l.x, l.y, l.visibility] for l in res.pose_landmarks.landmark])
        return run_classic

    model = _find_model()
    if model is None:
        raise PoseModelUnavailable(
            "mediapipe no trae mp.solutions.pose y falta el modelo de Tasks. Descarga "
            "pose_landmarker_lite.task (Google MediaPipe Models) a models/ o define "
            f"{_MODEL_ENV} con su ruta. La medicion con keypoints funciona sin el.")
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as _rm
    from mediapipe.tasks.python.core.base_options import BaseOptions
    opts = mpv.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=_rm.VisionTaskRunningMode.IMAGE,
        num_poses=1)
    landmarker = mpv.PoseLandmarker.create_from_options(opts)

    def run_tasks(rgb: np.ndarray) -> Optional[np.ndarray]:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = landmarker.detect(img)
        if not res.pose_landmarks:
            return None
        return np.array([[l.x, l.y, getattr(l, "visibility", 1.0) or 0.0]
                         for l in res.pose_landmarks[0]])
    return run_tasks


def extract_pose_sequence(video_path: Union[str, Path], max_seconds: float = 6.0,
                          target_width: int = 640, store_width: int = 480) -> PoseSequence:
    """Lee hasta ``max_seconds`` del video, reduce a ``target_width`` (solo reduce,
    nunca amplia) y corre MediaPipe Pose por frame. Los frames sin persona quedan
    en NaN (la calidad resultante es baja, sin excepciones). ``frames`` guarda copias
    RGB de ancho ``store_width`` (<= target_width) para la barra de frames.
    """
    detector = _make_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps < 1.0:
            fps = DEFAULT_FPS
        max_frames = max(1, int(round(max_seconds * fps)))
        lms, frames = [], []
        width = height = None
        while len(lms) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            h0, w0 = bgr.shape[:2]
            if width is None:
                width = min(int(target_width), w0)
                height = max(1, int(round(h0 * width / w0)))
            if (bgr.shape[1], bgr.shape[0]) != (width, height):
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = detector(rgb)
            lms.append(np.full((NUM_LANDMARKS, 3), np.nan) if out is None else out)
            sw = min(int(store_width), width)
            small = rgb if sw == width else cv2.resize(
                rgb, (sw, max(1, int(round(height * sw / width)))), interpolation=cv2.INTER_AREA)
            frames.append(small)
    finally:
        cap.release()
    if not lms:
        return PoseSequence(np.full((1, NUM_LANDMARKS, 3), np.nan), fps, int(target_width),
                            int(target_width * 3 // 4), None)
    return PoseSequence(np.stack(lms), fps, width, height, frames)


# ----------------------------------------------------------- release frame
def _nms_peaks(score: np.ndarray, k: int = 3, min_sep: int = 3) -> list:
    s = np.nan_to_num(score, nan=0.0).copy()
    peaks = []
    for _ in range(k):
        i = int(np.argmax(s))
        if s[i] <= 0:
            break
        peaks.append((i, float(s[i])))
        s[max(0, i - min_sep): i + min_sep + 1] = 0.0
    return peaks


def detect_release_frame(seq: SeqLike, throws: str) -> dict:
    """Sugiere el frame de soltura: velocidad maxima de la muñeca lanzadora (en
    torsos/s), ponderada por la extension del brazo y por estar por delante del
    hombro. Devuelve ``{frame_idx, confidence, candidates}`` con ``candidates``
    = top 3 ``{frame_idx, score}``. Sin datos validos: frame central, confianza 0.
    """
    s = _resolve(seq)
    ix = _idx(throws)
    T = s.n_frames
    fallback = {"frame_idx": T // 2, "confidence": 0.0, "candidates": []}
    if T < 3:
        return fallback
    wr, sh, el = _px(s, ix["wr"]), _px(s, ix["sh"]), _px(s, ix["el"])
    hip_mid = _mid(_px(s, L_HIP), _px(s, R_HIP))
    sh_mid = _mid(_px(s, L_SHOULDER), _px(s, R_SHOULDER))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        torso = np.nanmedian(np.linalg.norm(sh_mid - hip_mid, axis=1))
        arm_len = np.nanmedian(np.linalg.norm(sh - el, axis=1) + np.linalg.norm(el - wr, axis=1))
    if not np.isfinite(torso) or torso <= 0:
        torso = 0.25 * s.height
    if not np.isfinite(arm_len) or arm_len <= 0:
        arm_len = 1.1 * torso
    speed = np.full(T, np.nan)
    speed[1:-1] = np.linalg.norm(wr[2:] - wr[:-2], axis=1) / 2.0 * s.fps / torso
    with np.errstate(all="ignore"):
        ext = np.linalg.norm(wr - sh, axis=1) / arm_len
        ext_w = np.clip((ext - 0.6) / 0.3, 0.0, 1.0)
        ahead = np.linalg.norm(wr - hip_mid, axis=1) >= np.linalg.norm(sh - hip_mid, axis=1)
        gate = np.where(np.isfinite(ahead), np.where(ahead, 1.0, 0.3), 1.0)
        score = speed * ext_w * gate
    score = np.nan_to_num(score, nan=0.0)
    peaks = _nms_peaks(score)
    if not peaks:
        return fallback
    best_i, best_v = peaks[0]
    second = peaks[1][1] if len(peaks) > 1 else 0.0
    prominence = 1.0 - second / best_v
    vis = float(np.mean([_vis(s, ix["wr"])[best_i], _vis(s, ix["sh"])[best_i], _vis(s, ix["el"])[best_i]]))
    conf = float(np.clip(prominence * vis, 0.0, 1.0))
    return {"frame_idx": int(best_i), "confidence": conf,
            "candidates": [{"frame_idx": int(i), "score": float(v)} for i, v in peaks]}


# ------------------------------------------------------------------ medicion
def _arm_slot_dataset(rel_height_ft: float, rel_side_ft: float) -> float:
    """Misma formula que src/features/physical.py:190-191."""
    rad = math.atan2(rel_height_ft - 4.0, abs(rel_side_ft) + 0.5)
    return float(np.clip(math.degrees(rad), 0.0, 90.0))


def _pt(seq: PoseSequence, lm: int, i: int) -> np.ndarray:
    a = seq.landmarks[i, lm]
    return np.array([a[0] * seq.width, a[1] * seq.height])


def measure_release(seq: SeqLike, frame_idx: int, throws: str, height_ft: float) -> dict:
    """Mide el punto de soltura en ``frame_idx``. Ver docstring del modulo.

    Devuelve ``rel_height_ft, rel_side_ft, arm_slot_dataset_deg,
    arm_angle_biomech_deg`` (``None`` si no se pudo medir), ``quality_score``
    (0-1), ``quality_issues`` (textos en español con accion), ``quality_ok``
    (score >= QUALITY_MIN y medicion posible) y ``ft_per_px``.
    """
    s = _resolve(seq)
    ix = _idx(throws)
    T = s.n_frames
    i = int(np.clip(frame_idx, 0, T - 1))
    lm = s.landmarks[i]
    issues: list = []
    out = dict(frame_idx=i, throws=str(throws).upper()[:1], height_ft=float(height_ft),
               rel_height_ft=None, rel_side_ft=None, arm_slot_dataset_deg=None,
               arm_angle_biomech_deg=None, ft_per_px=None)

    key = [NOSE, L_SHOULDER, R_SHOULDER, ix["el"], ix["wr"], L_HIP, R_HIP, L_ANKLE, R_ANKLE]
    vis = np.nan_to_num(lm[key, 2], nan=0.0)
    vis_score = float(np.clip(vis.mean(), 0, 1))
    if np.isnan(lm[:, :2]).all():
        issues.append("No se detecta a ninguna persona en este frame: asegura que el lanzador "
                      "salga completo, bien iluminado y de frente a la cámara.")
    elif (vis[3:5] < VIS_MIN).any():
        issues.append("Brazo lanzador poco visible: evita que el brazo quede tapado y "
                      "mejora la iluminación o el contraste con el fondo.")
    if vis_score < VIS_MIN and not issues:
        issues.append("Landmarks poco visibles: regraba con más luz y sin objetos que tapen al lanzador.")

    # tamaño del sujeto y cuerpo completo (coordenadas normalizadas de la imagen)
    xy = lm[key, :2]
    finite = np.isfinite(xy).all(axis=1)
    ank_mid_y = float(_mid(lm[L_ANKLE, 1:2], lm[R_ANKLE, 1:2])[0])
    frac = float(ank_mid_y - lm[NOSE, 1]) if np.isfinite(ank_mid_y) and np.isfinite(lm[NOSE, 1]) else 0.0
    size_score = float(np.clip((frac - 0.2) / 0.3, 0.0, 1.0))
    if size_score < 0.67 and finite.any():
        issues.append("Sujeto muy pequeño en el cuadro: acércate o haz zoom para que el lanzador "
                      "ocupe al menos la mitad del alto del video.")
    margin = 0.02
    inside = finite & (xy[:, 0] > margin) & (xy[:, 0] < 1 - margin) & (xy[:, 1] > margin) & (xy[:, 1] < 1 - margin)
    body_score = float(inside.mean())
    if body_score < 1.0 and finite.any():
        issues.append("Cuerpo cortado: aleja la cámara para que se vean cabeza, brazos y pies completos.")

    # vista no frontal (hombros muy juntos respecto al torso)
    view_factor = 1.0
    sh_l, sh_r, hp_l, hp_r = (_pt(s, k, i) for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    if np.isfinite([sh_l, sh_r, hp_l, hp_r]).all():
        torso = float(np.linalg.norm((sh_l + sh_r) / 2 - (hp_l + hp_r) / 2))
        if torso > 0 and np.linalg.norm(sh_l - sh_r) / torso < 0.5:
            view_factor = 0.85
            issues.append("Vista de perfil: coloca la cámara de frente o ligeramente de 3/4 "
                          "(a la altura del pecho) para medir el slot.")

    quality = (0.5 * vis_score + 0.25 * size_score + 0.25 * body_score) * view_factor

    # escala px -> pies a partir de la estatura (nariz->tobillos)
    ft_per_px = None
    try:
        h = float(height_ft)
    except (TypeError, ValueError):
        h = float("nan")
    if not (math.isfinite(h) and 4.0 <= h <= 7.5):
        issues.append("Estatura fuera de rango (4.0-7.5 ft): captura la estatura real del lanzador.")
    else:
        nose, la, ra = _px(s, NOSE), _px(s, L_ANKLE), _px(s, R_ANKLE)
        with np.errstate(all="ignore"):
            ank = _mid(la, ra)
            d = np.linalg.norm(nose - ank, axis=1)
        d = d[np.isfinite(d) & (d > 1)]
        if d.size:
            ft_per_px = ANTHROPOMETRIC_NOSE_FRACTION * h / float(np.percentile(d, 80))
    if ft_per_px is None:
        issues.append("No se pudo escalar a pies (faltan nariz o tobillos): "
                      "asegura que el cuerpo completo salga en el video.")
        quality = min(quality, 0.3)
    out["ft_per_px"] = ft_per_px

    # medicion
    sh_t, wr_t = _pt(s, ix["sh"], i), _pt(s, ix["wr"], i)
    sh_o = _pt(s, ix["osh"], i)
    hips = np.array([_pt(s, ix["hip"], i), _pt(s, ix["ohip"], i)])
    ank_sup = [(_vis(s, ix["oank"])[i], _pt(s, ix["oank"], i)), (_vis(s, ix["ank"])[i], _pt(s, ix["ank"], i))]
    ank_sup = [p for v, p in sorted(ank_sup, key=lambda t: -t[0]) if np.isfinite(p).all()]
    ok_geom = bool(np.isfinite([sh_t, wr_t, sh_o]).all()) and ank_sup and np.isfinite(hips).all()
    if ft_per_px is not None and ok_geom and vis[4] >= VIS_NAN:
        support = ank_sup[0]
        center_x = float(hips[:, 0].mean())
        sign = 1.0 if sh_t[0] >= sh_o[0] else -1.0
        rel_h = (support[1] - wr_t[1]) * ft_per_px
        rel_s = (wr_t[0] - center_x) * sign * ft_per_px
        out["rel_height_ft"] = float(rel_h)
        out["rel_side_ft"] = float(rel_s)
        out["arm_slot_dataset_deg"] = _arm_slot_dataset(rel_h, rel_s)
        out["arm_angle_biomech_deg"] = float(math.degrees(
            math.atan2(sh_t[1] - wr_t[1], abs(wr_t[0] - sh_t[0]))))
    else:
        quality = min(quality, 0.3)
        if ft_per_px is not None:
            issues.append("Faltan landmarks de hombro, muñeca, cadera o tobillo en este frame: "
                          "elige otro frame de soltura o regraba con el cuerpo completo visible.")

    out["quality_score"] = float(np.clip(quality, 0.0, 1.0))
    out["quality_issues"] = issues
    out["quality_ok"] = bool(out["quality_score"] >= QUALITY_MIN
                             and out["arm_slot_dataset_deg"] is not None)
    return out


# ----------------------------------------------------------------- anotacion
def annotate_frame(seq: SeqLike, frame_idx: int, measures: Optional[dict] = None) -> np.ndarray:
    """Frame RGB con esqueleto (navy), punto de soltura (teal) y arco del angulo
    hombro->muñeca. Sin frames guardados usa un lienzo blanco. No usa el rojo de marca."""
    s = _resolve(seq)
    i = int(np.clip(frame_idx, 0, s.n_frames - 1))
    if s.frames is not None and len(s.frames) > i:
        img = np.ascontiguousarray(s.frames[i]).copy()
    else:
        img = np.full((s.height, s.width, 3), 255, dtype=np.uint8)
    H, W = img.shape[:2]
    lm = s.landmarks[i]

    def p(k: int) -> Optional[tuple]:
        x, y, v = lm[k]
        if not (np.isfinite(x) and np.isfinite(y)) or not np.nan_to_num(v) >= VIS_NAN:
            return None
        return int(round(x * W)), int(round(y * H))

    th = max(2, W // 200)
    for a, b in _EDGES:
        pa, pb = p(a), p(b)
        if pa and pb:
            cv2.line(img, pa, pb, _NAVY, th, cv2.LINE_AA)
    for k in (0, 11, 12, 13, 14, 15, 16, 23, 24, 27, 28):
        q = p(k)
        if q:
            cv2.circle(img, q, th + 1, _NAVY, -1, cv2.LINE_AA)

    throws = (measures or {}).get("throws")
    if throws:
        ix = _idx(throws)
        sh, wr = p(ix["sh"]), p(ix["wr"])
        if sh and wr:
            cv2.line(img, sh, wr, _TEAL, th + 1, cv2.LINE_AA)
            cv2.circle(img, wr, th * 3, _TEAL, -1, cv2.LINE_AA)
            cv2.circle(img, wr, th * 3 + 2, (255, 255, 255), 1, cv2.LINE_AA)
            dx, dy = wr[0] - sh[0], wr[1] - sh[1]
            sx = 1 if dx >= 0 else -1
            r = int(max(20, min(W, H) * 0.12))
            ang = math.degrees(math.atan2(dy, abs(dx)))     # y hacia abajo => negativo = arriba
            # arco desde la horizontal (hacia el lado de la muñeca) hasta la linea hombro->muñeca
            a0, a1 = (0, ang) if sx > 0 else (180, 180 - ang)
            lo, hi = min(a0, a1), max(a0, a1)
            cv2.ellipse(img, sh, (r, r), 0, lo, hi, _TEAL, th, cv2.LINE_AA)
            cv2.line(img, sh, (sh[0] + sx * r, sh[1]), _NAVY, 1, cv2.LINE_AA)
            slot = measures.get("arm_slot_dataset_deg")
            bio = measures.get("arm_angle_biomech_deg")
            txt = (f"slot {slot:.0f} deg" if slot is not None else "slot n/d") + \
                  (f" | hombro-muneca {bio:.0f} deg" if bio is not None else "")
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _NAVY, 2, cv2.LINE_AA)
    return img

"""Validacion de VeloVision contra 2 anotadores humanos (HACK-017, 17-D).

Compara la medicion automatica (``src/vision/pose_slot.py``) con los puntos que A y B marcan a mano en la
pestana "Modo anotacion" (``reports/validation/vision_manual_annotations.csv``).

Emparejamiento clip <-> archivo: se hashea cada ``.mp4``/``.mov`` de ``--clips`` con SHA-1 del contenido (misma
funcion que ``dashboard/velovision_tab.py``), se toman los primeros 16 caracteres y se busca en el registro
``vision_clip_registry.csv`` (``clip_code`` <-> ``file_hash``). Un ``clip_id`` sin archivo local se OMITE y se
reporta. El registro solo se LEE: nunca se crean codigos aqui.

Medida humana: cada anotador da 6 puntos en un frame (nariz, hombro lanzador, codo, muneca, cadera, tobillo de
apoyo). Se arma una secuencia de 1 frame y se llama a ``measure_release`` (la MISMA geometria y constantes que la
vision, sin duplicar la formula). Aproximaciones, porque el humano solo anota el frame de soltura:

* Origen lateral: ``measure_release`` toma la mediana de la x del tobillo de apoyo en los primeros frames de la
  entrega; con 1 frame es la x del tobillo anotado en el frame de soltura (la zancada ya lo movio respecto de la
  goma). La vision usa los primeros frames reales, asi que esta diferencia entra al error medido.
* Escala px->pies: nariz a tobillo de apoyo (la vision usa el punto medio de ambos tobillos, percentil 80).
* Hombro contrario: no se anota; se coloca del lado del tobillo de apoyo respecto del hombro lanzador (solo fija
  el signo del lado, vista de frente). Cadera: el unico punto se usa para ambas caderas (solo valida finitud).
* Ancho/alto en pixeles: los del clip procesado por la vision (la anotacion esta normalizada).

Medida automatica: ``extract_pose_sequence`` del mismo clip (primeros ``min(6 s, 360 frames)``, como la app) y
``measure_release`` en el ``frame_idx`` anotado (comparacion de igual a igual) y en el frame que detecta el sistema.

Metricas (formato largo ``metric, measure, value, n, note``): ICC(2,1) y diferencia media A-B; MAE, RMSE y sesgo
(vision - promedio humano); error del frame de release (detectado - anotado) en frames y ms; N con ambos
anotadores, con uno solo y omitidos. El error sistematico respecto de TrackMan NO se mide (AGENTS.md 2.6).

Los CSV NUNCA llevan nombres de archivo ni rutas: solo codigos ``C###``.

Uso::

    python -m analysis.validate_vision --annotations reports/validation/vision_manual_annotations.csv \\
        --clips data/vision_clips --registry reports/validation/vision_clip_registry.csv \\
        --output reports/validation/vision_validation_summary.csv
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import pose_slot as PS  # noqa: E402

VIDEO_EXT = {".mp4", ".mov"}
CLIP_ID_RE = re.compile(r"^C\d{3,5}$")
HASH_LEN = 16
MAX_SECONDS = 6.0           # igual que dashboard/velovision_tab.py
MAX_FRAMES = 360
TARGET_WIDTH = 640
LANDMARK_NAMES = ["nariz", "hombro", "codo", "muneca", "cadera", "tobillo_apoyo"]
MEASURES = ["arm_slot_dataset_deg", "arm_angle_biomech_deg"]
SUMMARY_COLUMNS = ["metric", "measure", "value", "n", "note"]
BY_CLIP_COLUMNS = [
    "clip_code", "frame_idx_annotated", "frame_idx_detected", "status", "n_annotators", "release_valid",
    "release_frame_error_frames", "release_frame_error_ms",
    "A_slot_deg", "B_slot_deg", "human_mean_slot_deg", "vision_slot_deg_annotated_frame", "vision_slot_deg_detected_frame",
    "A_biomech_deg", "B_biomech_deg", "human_mean_biomech_deg", "vision_biomech_deg_annotated_frame",
    "vision_biomech_deg_detected_frame", "slot_sys_err_deg",
]
_SHORT = {"arm_slot_dataset_deg": "slot", "arm_angle_biomech_deg": "biomech"}


# ---------------------------------------------------------------- emparejar
def file_hash(path: Path) -> str:
    """Hash corto del archivo: SHA-1 del contenido, igual que ``dashboard/velovision_tab.py`` (16 caracteres,
    como guarda ``dashboard.annotation_tab.register_clip``; no se importa ese modulo porque carga Streamlit)."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:HASH_LEN]


def load_registry(path: Path) -> dict:
    """``{file_hash(16): clip_code}``; vacio si no existe."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    reg = pd.read_csv(p, dtype=str)
    return {str(h)[:HASH_LEN]: str(c) for c, h in zip(reg["clip_code"], reg["file_hash"])}


def match_clips(clips_dir: Path, registry: dict) -> dict:
    """``{clip_code: Path}`` de los videos locales cuyo hash esta en el registro."""
    out: dict = {}
    d = Path(clips_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            code = registry.get(file_hash(p)[:HASH_LEN])
            if code is not None and code not in out:
                out[code] = p
    return out


# ------------------------------------------------------------------ metricas
def icc_2_1(a, b) -> tuple:
    """ICC(2,1) (efectos aleatorios, acuerdo absoluto, un evaluador) para 2 evaluadores. ``(icc, nota)``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, k = len(a), 2
    if n < 2:
        return None, "ICC no calculable: se necesitan al menos 2 clips con ambos anotadores"
    x = np.stack([a, b], axis=1)
    grand = x.mean()
    msr = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if not math.isfinite(den) or den <= 1e-12:
        return None, "ICC no calculable: sin varianza entre clips"
    return float(np.clip((msr - mse) / den, -1.0, 1.0)), ""


def _err_stats(vision, human) -> Optional[dict]:
    v, h = np.asarray(vision, float), np.asarray(human, float)
    if len(v) == 0:
        return None
    e = v - h
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()), "n": len(e)}


# ------------------------------------------------------------ medida humana
def human_sequence(points: dict, throws: str, width: int, height: int) -> PS.PoseSequence:
    """Secuencia de 1 frame con los 6 puntos anotados (coordenadas normalizadas). Ver aproximaciones arriba."""
    ix = PS._idx(throws)
    lm = np.full((1, PS.NUM_LANDMARKS, 3), np.nan)
    lm[0, :, 2] = 0.0

    def put(i, name):
        lm[0, i, :2] = points[name]
        lm[0, i, 2] = 1.0

    put(PS.NOSE, "nariz")
    put(ix["sh"], "hombro")
    put(ix["el"], "codo")
    put(ix["wr"], "muneca")
    put(ix["hip"], "cadera")
    put(ix["ohip"], "cadera")
    put(ix["oank"], "tobillo_apoyo")
    sx, sy = points["hombro"]
    side = 1.0 if points["tobillo_apoyo"][0] >= sx else -1.0     # hombro contrario del lado del pie de apoyo
    lm[0, ix["osh"], :2] = (sx + side * 0.05, sy)
    lm[0, ix["osh"], 2] = 1.0
    return PS.PoseSequence(lm, PS.DEFAULT_FPS, int(width), int(height))


def human_measures(points: dict, throws: str, height_ft: float, width: int, height: int) -> dict:
    m = PS.measure_release(human_sequence(points, throws, width, height), 0, throws, height_ft)
    return {k: m.get(k) for k in MEASURES}


# ------------------------------------------------------------- vision en clip
def extract_clip(path: Path) -> PS.PoseSequence:
    """Misma ventana que la app: ``min(6 s, 360 frames)`` desde el inicio, ancho 640."""
    fps = 0.0
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps < 1.0:
        fps = PS.DEFAULT_FPS
    return PS.extract_pose_sequence(str(path), min(MAX_SECONDS, MAX_FRAMES / fps), TARGET_WIDTH)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ------------------------------------------------------------------- proceso
def _load_annotations(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["clip_id", "frame_idx", "annotator", "landmark", "x", "y", "throws", "height_ft"])
    return pd.read_csv(p, dtype={"clip_id": str, "annotator": str})


def _annotation_units(df: pd.DataFrame) -> tuple[dict, int]:
    """``{(clip_id, frame_idx): {annotator: {points, throws, height_ft}}}`` y N de anotaciones incompletas."""
    units: dict = {}
    incomplete = 0
    for (cid, fr, ann), g in df.groupby(["clip_id", "frame_idx", "annotator"], sort=True):
        last = g.drop_duplicates("landmark", keep="last").set_index("landmark")
        if not all(n in last.index for n in LANDMARK_NAMES):
            incomplete += 1
            continue
        pts = {n: (float(last.loc[n, "x"]), float(last.loc[n, "y"])) for n in LANDMARK_NAMES}
        units.setdefault((str(cid), int(fr)), {})[str(ann)] = {
            "points": pts, "throws": str(last.iloc[0]["throws"]).upper()[:1], "height_ft": float(last.iloc[0]["height_ft"])}
    return units, incomplete


def run(annotations: Path, clips: Path, registry: Path, output: Path, by_clip: Optional[Path] = None,
        verbose: bool = True) -> dict:
    output = Path(output)
    by_clip = Path(by_clip) if by_clip else output.with_name("vision_validation_by_clip.csv")
    units, incomplete = _annotation_units(_load_annotations(annotations))
    files = match_clips(clips, load_registry(registry))

    rows: list = []
    omitted_clips: set = set()
    cache: dict = {}
    n_both = n_one = 0
    for (cid, fr), anns in sorted(units.items()):
        if not CLIP_ID_RE.match(cid) or cid not in files:
            omitted_clips.add(cid if CLIP_ID_RE.match(cid) else "(codigo invalido)")
            if CLIP_ID_RE.match(cid):
                rows.append({"clip_code": cid, "frame_idx_annotated": fr, "status": "omitido_sin_archivo",
                             "n_annotators": len(anns)})
            continue
        if cid not in cache:
            try:
                cache[cid] = extract_clip(files[cid])
            except Exception as exc:                      # video ilegible o sin modelo: se reporta el tipo, no la ruta
                cache[cid] = type(exc).__name__
        seq = cache[cid]
        both = "A" in anns and "B" in anns
        n_both += both
        n_one += not both
        row = {"clip_code": cid, "frame_idx_annotated": fr, "n_annotators": len(anns),
               "status": "ok" if both else "un_anotador"}
        if isinstance(seq, str):
            row["status"] = f"error_vision:{seq}"
            rows.append(row)
            continue
        first = anns.get("A") or anns.get("B")
        throws, height_ft = first["throws"], first["height_ft"]
        hum = {a: human_measures(d["points"], d["throws"], d["height_ft"], seq.width, seq.height)
               for a, d in anns.items()}
        vis_a = PS.measure_release(seq, fr, throws, height_ft)
        det = PS.detect_release_frame(seq, throws)
        vis_d = PS.measure_release(seq, det["frame_idx"], throws, height_ft)
        row.update(frame_idx_detected=int(det["frame_idx"]), release_valid=bool(det["valid"]),
                   release_frame_error_frames=int(det["frame_idx"]) - fr,
                   release_frame_error_ms=(int(det["frame_idx"]) - fr) / float(seq.fps) * 1000.0,
                   slot_sys_err_deg=vis_a.get("slot_sys_err_deg"))
        for meas, sh in _SHORT.items():
            hv = {a: _num(hum[a][meas]) for a in hum}
            row[f"A_{sh}_deg"], row[f"B_{sh}_deg"] = hv.get("A"), hv.get("B")
            if both and None not in (hv["A"], hv["B"]):
                row[f"human_mean_{sh}_deg"] = (hv["A"] + hv["B"]) / 2.0
            row[f"vision_{sh}_deg_annotated_frame"] = _num(vis_a.get(meas))
            row[f"vision_{sh}_deg_detected_frame"] = _num(vis_d.get(meas))
        rows.append(row)

    df = pd.DataFrame(rows).reindex(columns=BY_CLIP_COLUMNS)
    summary = _summary(df, n_both, n_one, len(omitted_clips), incomplete)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(output, index=False)
    by_clip.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(by_clip, index=False)
    rec = _recommendation(summary, df)
    if verbose:
        print(f"Lanzamientos con A y B: {n_both} | con uno solo: {n_one} | clips omitidos (sin archivo): {len(omitted_clips)}")
        print(rec["text"])
        print(f"Resumen: {output.name} | por clip: {by_clip.name}")
    return {"summary": pd.DataFrame(summary, columns=SUMMARY_COLUMNS), "by_clip": df, "recommendation": rec}


def _summary(df: pd.DataFrame, n_both: int, n_one: int, n_omitted: int, incomplete: int) -> list:
    out: list = []

    def add(metric, measure, value, n, note=""):
        out.append({"metric": metric, "measure": measure, "value": value, "n": n, "note": note})

    add("n_both_annotators", "all", n_both, n_both, "lanzamientos (clip, frame) anotados por A y B")
    add("n_one_annotator", "all", n_one, n_one, "lanzamientos anotados por uno solo; no entran a las metricas")
    add("n_omitted_clips", "all", n_omitted, n_omitted, "clip_id sin archivo local o sin codigo valido"
        + (f"; {incomplete} anotacion(es) incompleta(s) descartada(s)" if incomplete else ""))
    ok = df[df["status"] == "ok"] if len(df) else df
    for meas, sh in _SHORT.items():
        a, b = (pd.to_numeric(ok.get(c), errors="coerce") for c in (f"A_{sh}_deg", f"B_{sh}_deg")) \
            if len(ok) else (pd.Series(dtype=float),) * 2
        m = a.notna() & b.notna()
        n = int(m.sum())
        icc, note = icc_2_1(a[m], b[m])
        add("icc_2_1", meas, icc, n, note)
        add("mean_diff_A_minus_B", meas, float((a[m] - b[m]).mean()) if n else None, n,
            "" if n else "sin clips con ambos anotadores")
        for tag, col, label in (("", f"vision_{sh}_deg_annotated_frame", "frame anotado"),
                                ("_detected_frame", f"vision_{sh}_deg_detected_frame", "frame detectado por el sistema")):
            hm = pd.to_numeric(ok.get(f"human_mean_{sh}_deg"), errors="coerce") if len(ok) else pd.Series(dtype=float)
            v = pd.to_numeric(ok.get(col), errors="coerce") if len(ok) else pd.Series(dtype=float)
            mm = hm.notna() & v.notna()
            st = _err_stats(v[mm], hm[mm])
            nn = int(mm.sum())
            note = f"vision en {label} contra promedio de A y B" + ("" if nn else "; sin datos")
            for name in ("mae", "rmse", "bias"):
                add(f"{name}_vs_human_mean{tag}", meas, st[name] if st else None, nn, note)
    err = pd.to_numeric(ok.get("release_frame_error_frames"), errors="coerce").dropna() if len(ok) else pd.Series(dtype=float)
    ms = pd.to_numeric(ok.get("release_frame_error_ms"), errors="coerce").dropna() if len(ok) else pd.Series(dtype=float)
    nn = len(err)
    note = "detectado - anotado (el anotado es el de A, o de B si A no esta; comun en ambos por emparejar clip y frame)"
    add("release_frame_mean_error_frames", "release_frame", float(err.mean()) if nn else None, nn, note)
    add("release_frame_mae_frames", "release_frame", float(err.abs().mean()) if nn else None, nn, "")
    add("release_frame_mean_error_ms", "release_frame", float(ms.mean()) if nn else None, nn, note)
    add("release_frame_mae_ms", "release_frame", float(ms.abs().mean()) if nn else None, nn, "")
    rec = _recommendation(out, df)
    add("recommended_combined_slot_error_deg", "arm_slot_dataset_deg", rec["combined"], rec["n"],
        rec["note"])
    add("systematic_error_vs_trackman", "all", None, 0, "no medido: AGENTS.md 2.6; ambos anotadores comparten el sesgo "
        "de origen del video, asi que este MAE no lo detecta")
    return out


def _recommendation(summary: list, df: pd.DataFrame) -> dict:
    mae = next((r["value"] for r in summary if r["metric"] == "mae_vs_human_mean" and r["measure"] == "arm_slot_dataset_deg"), None)
    sys_err = pd.to_numeric(df.get("slot_sys_err_deg"), errors="coerce").dropna() if len(df) else pd.Series(dtype=float)
    if mae is None or not len(sys_err):
        return {"mae": mae, "sys": None, "combined": None, "n": 0,
                "note": "sin datos para recomendar", "text": "Recomendacion de ventana: sin datos (falta MAE_slot o error sistematico)."}
    sysm = float(sys_err.median())
    comb = PS.combined_slot_error_deg(float(mae), sysm)
    n = int(next((r["n"] for r in summary if r["metric"] == "mae_vs_human_mean" and r["measure"] == "arm_slot_dataset_deg"), 0))
    note = f"sqrt(MAE_slot^2 + sys^2) con MAE_slot={mae:.2f} deg y sys tipico (mediana)={sysm:.2f} deg"
    return {"mae": float(mae), "sys": sysm, "combined": comb, "n": n, "note": note,
            "text": (f"Recomendacion de ventana: MAE_slot = {mae:.2f} deg (n={n}); sys tipico = {sysm:.2f} deg; "
                     f"ventana combinada = +/- {comb:.2f} deg. Decide si se conecta a la ficha.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", default="reports/validation/vision_manual_annotations.csv")
    ap.add_argument("--clips", default="data/vision_clips")
    ap.add_argument("--registry", default="reports/validation/vision_clip_registry.csv")
    ap.add_argument("--output", default="reports/validation/vision_validation_summary.csv")
    ap.add_argument("--by-clip", default=None, help="por defecto, vision_validation_by_clip.csv junto a --output")
    a = ap.parse_args(argv)
    r = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p      # noqa: E731
    run(r(a.annotations), r(a.clips), r(a.registry), r(a.output), r(a.by_clip) if a.by_clip else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

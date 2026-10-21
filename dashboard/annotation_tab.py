"""Pestaña "Modo anotación" (HACK-017, 17-C): anotación manual de puntos del cuerpo para validar VeloVision (17-D).

Dos anotadores ("A" y "B") hacen clic, en orden, sobre nariz, hombro lanzador, codo, muñeca, cadera y tobillo de
apoyo de un frame. Se guarda en ``reports/validation/vision_manual_annotations.csv`` con coordenadas
normalizadas (0-1). Sin datos personales: ``clip_id`` es un código (p. ej. ``C001``), nunca un nombre, y el
nombre del archivo de video no se guarda.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import components as C
from dashboard import velovision_tab as VT
from src.vision import pose_slot as PS

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "reports" / "validation" / "vision_manual_annotations.csv"
COLUMNS = ["clip_id", "frame_idx", "annotator", "landmark", "x", "y", "throws", "height_ft", "timestamp"]
LANDMARKS = [("nariz", "Nariz"), ("hombro", "Hombro lanzador"), ("codo", "Codo lanzador"),
             ("muneca", "Muñeca lanzadora"), ("cadera", "Cadera"), ("tobillo_apoyo", "Tobillo de apoyo")]
ANNOTATORS = ["A", "B"]
CLIP_ID_RE = re.compile(r"^C\d{3,5}$")          # solo códigos; un nombre de persona nunca cumple el formato
CLIP_ID_HELP = "El código del clip tiene formato C001 (letra C y 3 a 5 dígitos); nunca un nombre de persona."
REGISTRY_PATH = ROOT / "reports" / "validation" / "vision_clip_registry.csv"
REGISTRY_COLUMNS = ["clip_code", "file_hash"]
DISPLAY_WIDTH = 720
NAVY_BGR = (0x5F, 0x3A, 0x1F)
TEAL_BGR = (0x66, 0x7A, 0x0B)


# ----------------------------------------------------------------------------
# Lógica pura (probada sin servidor)
# ----------------------------------------------------------------------------
def load_annotations(path: Path | str = ANNOTATIONS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, dtype={"clip_id": str, "annotator": str})


def build_rows(clip_id: str, frame_idx: int, annotator: str, points: list, throws: str, height_ft: float,
               timestamp: str | None = None) -> list[dict]:
    """Seis filas (una por punto) con coordenadas normalizadas. Valida el código y el número de puntos."""
    if not CLIP_ID_RE.match(clip_id or ""):
        raise ValueError(CLIP_ID_HELP)
    if annotator not in ANNOTATORS:
        raise ValueError("El anotador debe ser 'A' o 'B'.")
    if len(points) != len(LANDMARKS):
        raise ValueError(f"Faltan puntos: se necesitan {len(LANDMARKS)} y hay {len(points)}.")
    ts = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [{"clip_id": clip_id, "frame_idx": int(frame_idx), "annotator": annotator, "landmark": name,
             "x": round(float(x), 5), "y": round(float(y), 5), "throws": str(throws).upper()[:1],
             "height_ft": round(float(height_ft), 3), "timestamp": ts}
            for (name, _), (x, y) in zip(LANDMARKS, points)]


def save_annotation(rows: list[dict], path: Path | str = ANNOTATIONS_PATH) -> pd.DataFrame:
    """Agrega las filas al CSV; reemplaza la anotación previa del mismo (clip, frame, anotador)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = load_annotations(p)
    new = pd.DataFrame(rows, columns=COLUMNS)
    if len(old) and len(new):
        k = new.iloc[0]
        drop = (old["clip_id"] == k["clip_id"]) & (old["frame_idx"] == k["frame_idx"]) & (old["annotator"] == k["annotator"])
        old = old[~drop]
    out = pd.concat([old, new], ignore_index=True)[COLUMNS]
    out.to_csv(p, index=False)
    return out


def progress_by_annotator(df: pd.DataFrame) -> dict:
    """Lanzamientos (clip, frame) anotados por cada anotador y los anotados por ambos."""
    sets = {a: set() for a in ANNOTATORS}
    for (clip, frame, ann), _ in df.groupby(["clip_id", "frame_idx", "annotator"]) if len(df) else []:
        if ann in sets:
            sets[ann].add((clip, int(frame)))
    return {"por_anotador": {a: len(s) for a, s in sets.items()}, "ambos": len(sets["A"] & sets["B"]),
            "clips": int(df["clip_id"].nunique()) if len(df) else 0}


def register_clip(file_hash: str, registry: Path | str = REGISTRY_PATH, annotations: pd.DataFrame | None = None) -> str:
    """Código ``C###`` determinista por contenido: el mismo video da el mismo código en cualquier sesión.

    ``vision_clip_registry.csv`` (``clip_code``, ``file_hash``) guarda el hash corto del archivo; quien primero
    registra un video recibe el siguiente código libre. Así A y B anotan el mismo clip con el mismo ``clip_id``.
    """
    p = Path(registry)
    short = str(file_hash)[:16]
    reg = pd.read_csv(p, dtype=str) if p.exists() and p.stat().st_size else pd.DataFrame(columns=REGISTRY_COLUMNS)
    hit = reg.loc[reg["file_hash"] == short, "clip_code"]
    if len(hit):
        return str(hit.iloc[0])
    used = pd.concat([reg["clip_code"], annotations["clip_id"] if annotations is not None and len(annotations)
                      else pd.Series(dtype=str)], ignore_index=True)
    code = next_clip_code(pd.DataFrame({"clip_id": used}))
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([reg, pd.DataFrame([{"clip_code": code, "file_hash": short}])], ignore_index=True)[REGISTRY_COLUMNS].to_csv(p, index=False)
    return code


def next_clip_code(df: pd.DataFrame) -> str:
    nums = [int(m.group(1)) for c in (df["clip_id"].astype(str) if len(df) else []) if (m := re.fullmatch(r"C(\d{3,5})", c))]
    return f"C{(max(nums) + 1) if nums else 1:03d}"


def draw_points(img: np.ndarray, pts: list, width: int = DISPLAY_WIDTH) -> np.ndarray:
    """Frame RGB escalado a ``width`` con los puntos ya anotados (numerados). No usa el rojo de marca."""
    import cv2
    h, w = img.shape[:2]
    out = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_CUBIC)
    H, W = out.shape[:2]
    rgb_navy, rgb_teal = NAVY_BGR[::-1], TEAL_BGR[::-1]
    for i, (x, y) in enumerate(pts):
        c = (int(round(x * W)), int(round(y * H)))
        cv2.circle(out, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, c, 7, rgb_navy if i % 2 == 0 else rgb_teal, -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (c[0] + 11, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (c[0] + 11, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb_navy, 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
def _steps_html(n_done: int) -> str:
    chips = []
    for i, (_, label) in enumerate(LANDMARKS):
        cls = "done" if i < n_done else ("now" if i == n_done else "")
        chips.append(f'<span class="{cls}">{i + 1} · {C.esc(label)}</span>')
    return f'<div class="dr-ann-step">{"".join(chips)}</div>'


def _undo(k: str) -> None:
    pts = st.session_state.get(k) or []
    if pts:
        pts.pop()


def _reset(k: str) -> None:
    st.session_state[k] = []


def render_annotation_tab(path: Path | str | None = None) -> None:
    path = Path(path) if path else ANNOTATIONS_PATH
    registry = path.parent / "vision_clip_registry.csv" if path != ANNOTATIONS_PATH else REGISTRY_PATH
    st.markdown("### Modo anotación · validación de VeloVision")
    st.caption("Herramienta interna: dos anotadores marcan los mismos puntos del cuerpo a mano para medir cuánto se "
               "equivoca la visión. El clip se identifica con un código C001 derivado del video (nunca un nombre); el video no se guarda.")
    df = load_annotations(path)
    prog = progress_by_annotator(df)
    st.markdown(
        '<div class="dr-stats">'
        + "".join(f'<span class="dr-stat">Anotador {a}: <b>{prog["por_anotador"][a]}</b> lanzamientos</span>' for a in ANNOTATORS)
        + f'<span class="dr-stat">Anotados por A y B: <b>{prog["ambos"]}</b></span>'
        f'<span class="dr-stat">Clips con anotaciones: <b>{prog["clips"]}</b></span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        clip = VT.pick_clip("an")
    with c2:
        annotator = st.radio("Anotador", ANNOTATORS, horizontal=True, key="an_annotator")
        clip_slot = st.empty()
        hand = st.radio("Mano lanzadora", list(VT.HAND_LABELS), horizontal=True, key="an_hand")
        height_ft = st.number_input("Estatura del lanzador (ft)", VT.FT_RANGE[0], VT.FT_RANGE[1], 6.0, 0.05, key="an_height")
    if clip is None:
        st.markdown(C.empty_state("target", "Elige un clip para anotar",
                                  "Sube un video o elige un clip de demostración local. El frame de soltura "
                                  "confirmado en VeloVision se propone por defecto."), unsafe_allow_html=True)
        return
    data, digest = clip
    clip_id = register_clip(digest, registry, df)
    clip_slot.text_input("Código del clip (fijo para este video)", value=clip_id, disabled=True, key=f"an_clip_id_{clip_id}",
                         help=CLIP_ID_HELP + " Se deriva del contenido del video: A y B ven el mismo código.")
    throws = VT.HAND_LABELS[hand]
    seq = VT.get_sequence(data, digest)
    if seq is None:
        return
    if not seq.frames:
        st.markdown(C.callout("Este clip no trae frames para mostrar; sube el video original."), unsafe_allow_html=True)
        return

    default = VT.confirmed_frame(digest, throws)
    if default is None:
        default = int(PS.detect_release_frame(seq, throws)["frame_idx"])
    n = len(seq.frames)
    fkey = f"an_frame_{digest[:10]}"
    st.session_state.setdefault(fkey, min(max(default, 0), n - 1))
    frame = int(st.number_input("Frame a anotar", 0, n - 1, key=fkey, help="Por defecto, la soltura confirmada en VeloVision."))

    pkey = f"an_pts_{digest[:10]}_{frame}_{annotator}"
    pts: list = st.session_state.setdefault(pkey, [])
    done = len(pts)
    st.markdown(_steps_html(done), unsafe_allow_html=True)
    if done < len(LANDMARKS):
        st.markdown(f'<div class="note" style="margin:.4rem 0">Haz clic en: <b>{LANDMARKS[done][1]}</b> '
                    f'(punto {done + 1} de {len(LANDMARKS)})</div>', unsafe_allow_html=True)

    from streamlit_image_coordinates import streamlit_image_coordinates
    img = draw_points(np.asarray(seq.frames[frame]), pts)
    click = streamlit_image_coordinates(img, key=f"an_img_{pkey}", width=img.shape[1])
    if click and done < len(LANDMARKS):
        sig = (pkey, click.get("x"), click.get("y"), click.get("unix_time"))
        if st.session_state.get("an_last_click") != sig:
            st.session_state["an_last_click"] = sig
            pts.append((click["x"] / click["width"], click["y"] / click["height"]))
            st.rerun()

    b1, b2, b3 = st.columns([1, 1, 1.4])
    b1.button("Deshacer último punto", key="an_undo", on_click=_undo, args=(pkey,), disabled=not pts, width="stretch")
    b2.button("Borrar todos", key="an_reset", on_click=_reset, args=(pkey,), disabled=not pts, width="stretch")
    valid_id = bool(CLIP_ID_RE.match(clip_id or ""))
    if not valid_id:
        st.markdown(C.callout(CLIP_ID_HELP),
                    unsafe_allow_html=True)
    if b3.button("Guardar anotación", key="an_save", type="primary", width="stretch",
                 disabled=not (valid_id and len(pts) == len(LANDMARKS))):
        save_annotation(build_rows(clip_id, frame, annotator, pts, throws, height_ft), path)
        st.session_state[pkey] = []
        st.session_state["an_saved"] = f"Anotación guardada: clip {clip_id}, frame {frame}, anotador {annotator}."
        st.rerun()
    if st.session_state.get("an_saved"):
        st.markdown(f'<span class="dr-ok">{C.icon("check")}{C.esc(st.session_state.pop("an_saved"))}</span>',
                    unsafe_allow_html=True)
    if len(df):
        with st.expander("Ver el avance por clip", expanded=False):
            tab = df.groupby(["clip_id", "annotator"])["frame_idx"].nunique().unstack(fill_value=0)
            st.dataframe(tab.rename(columns=lambda a: f"Anotador {a}"), width="stretch")

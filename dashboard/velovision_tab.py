"""Pestaña VeloVision (HACK-017, subtarea 17-C): video de celular -> punto de soltura -> ficha de prospecto.

Flujo: guía de captura -> video + estatura + mano (+ radar opcional) -> pose por frame (MediaPipe,
``src/vision/pose_slot.py``) -> release sugerido + confirmación humana -> medidas con su error -> puerta de
calidad -> rangos de Stuff+ CDMX por familia (``src/vision/prospect_card.py``).

Honestidad (AGENTS.md sec. 2): el video es monocular y 2D. La pestaña nunca afirma medir spin, quiebre ni
velocidad, nunca da un valor puntual de Stuff+ del prospecto y nunca empareja el video con un
``pitcher_anon_id``. El video se procesa en un archivo temporal que se borra de inmediato; en memoria solo
quedan los landmarks y unos frames reducidos de la sesión actual (``st.session_state``), no un caché global.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from dashboard import components as C
from src.vision import pose_slot as PS
from src.vision import prospect_card as PC

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "data" / "vision_clips"      # local, no versionado
VIDEO_TYPES = ["mp4", "mov"]
MAX_UPLOAD_MB = 60
MAX_SECONDS = 6.0
MAX_FRAMES = 360                                 # tope de memoria para cámaras lentas (120 fps -> 3 s)
TARGET_WIDTH = 640
SEQ_CACHE_SIZE = 2
HAND_LABELS = {"Derecho": "Right", "Zurdo": "Left"}
HAND_NAMES = {v: k for k, v in HAND_LABELS.items()}
FT_RANGE = (4.0, 7.5)                            # mismo rango que valida measure_release
CM_PER_FT = 30.48
ERR_PENDING = "error de medición aún no medido (pendiente de validación)"

GUIDE = [
    ("camera", "Cámara de frente", "Detrás del home o desde el jardín central, a la altura del pecho y fija en un tripié."),
    ("person", "Cuerpo completo", "De la cabeza a los pies, sin cortes ni objetos que tapen el brazo lanzador."),
    ("film", "Cámara lenta", "Si el celular lo permite, 120 fps o más: ayuda a elegir el frame de soltura."),
    ("frames", "Desde antes de la entrega", "Graba desde ANTES de que el pitcher inicie la entrega (pie de apoyo quieto sobre la goma): el origen lateral se toma de los primeros frames."),
    ("ruler", "Estatura conocida", "La escala sale de la estatura real del lanzador; sin ella no hay medida."),
]
PRIVACY = ("Tu video se procesa en un archivo temporal que se borra al terminar y no se guarda en el servidor. "
           "En memoria solo quedan puntos del cuerpo y unos frames reducidos de esta sesión.")


# ----------------------------------------------------------------------------
# Entrada de video (compartida con el Modo anotación)
# ----------------------------------------------------------------------------
def list_clips() -> list[Path]:
    """Clips de demostración locales (``data/vision_clips/``); vacío si la carpeta no existe."""
    if not CLIPS_DIR.is_dir():
        return []
    return sorted(p for p in CLIPS_DIR.iterdir() if p.suffix.lower().lstrip(".") in VIDEO_TYPES)


def pick_clip(prefix: str) -> tuple[bytes, str] | None:
    """Subida de ``.mp4``/``.mov`` o clip de demostración local. Devuelve ``(bytes, sha1)`` o ``None``."""
    up = st.file_uploader(f"Video del lanzamiento (.mp4 o .mov, máx. {MAX_UPLOAD_MB} MB)", type=VIDEO_TYPES,
                          key=f"{prefix}_upload")
    clips = list_clips()
    chosen = None
    if clips:
        names = ["— ninguno —"] + [p.name for p in clips]
        sel = st.selectbox("…o un clip de demostración local", names, key=f"{prefix}_clip_pick",
                           help="Carpeta data/vision_clips/ (local, no versionada).")
        chosen = clips[names.index(sel) - 1] if sel != names[0] else None
    if up is not None:
        data = up.getvalue()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            st.markdown(C.callout(f"El video pesa {len(data) / 1e6:.0f} MB y el máximo es {MAX_UPLOAD_MB} MB. "
                                  "Recórtalo a unos segundos alrededor del lanzamiento y vuelve a subirlo."),
                        unsafe_allow_html=True)
            return None
    elif chosen is not None:
        data = chosen.read_bytes()
    else:
        return None
    return data, hashlib.sha1(data).hexdigest()


def _probe(path: str) -> tuple[float, int]:
    """(fps, frames totales) del archivo; valores por defecto si no se puede leer (el extractor avisará)."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return (fps if fps >= 1.0 else 30.0), max(n, 0)
    except Exception:  # pragma: no cover - cv2 siempre esta instalado
        return 30.0, 0


def get_sequence(data: bytes, digest: str) -> PS.PoseSequence | None:
    """Pose por frame del video, con caché por hash en la SESIÓN (no global).

    Muestra una barra de progreso (estimada por tiempo: el extractor no reporta avance) y los errores como
    mensajes claros. Devuelve ``None`` si no se pudo procesar. El temporal se borra siempre.
    """
    cache: OrderedDict = st.session_state.setdefault("vv_seq_cache", OrderedDict())
    if digest in cache:
        cache.move_to_end(digest)
        return cache[digest]

    bar = st.progress(0.0, text="Preparando el video…")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        fps, total = _probe(tmp)
        seconds = min(MAX_SECONDS, MAX_FRAMES / fps)
        expected = max(1, min(total or int(seconds * fps), int(seconds * fps)))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(PS.extract_pose_sequence, tmp, seconds, TARGET_WIDTH)
            while not fut.done():
                est = min(0.95, (time.perf_counter() - t0) / (expected * 0.05))   # ~50 ms/frame, solo estimación
                bar.progress(est, text=f"Detectando la pose en ~{expected} frames (avance estimado)…")
                time.sleep(0.1)
            seq = fut.result()
        bar.progress(1.0, text="Listo")
    except PS.PoseModelUnavailable:
        bar.empty()
        st.markdown(C.callout("VeloVision no encontró el modelo de pose (models/pose_landmarker_full.task). "
                              "Sin ese archivo no se puede analizar video; el resto de la app sigue funcionando."),
                    unsafe_allow_html=True)
        return None
    except Exception as exc:  # video ilegible, formato raro, memoria, etc.: mensaje claro, nunca traceback
        bar.empty()
        st.markdown(C.callout("No se pudo leer el video. Prueba con otro archivo .mp4 o .mov grabado con la cámara "
                              f"del celular. Detalle técnico: {type(exc).__name__}."), unsafe_allow_html=True)
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    time.sleep(0.15)
    bar.empty()
    cache[digest] = seq
    while len(cache) > SEQ_CACHE_SIZE:
        cache.popitem(last=False)
    return seq


# ----------------------------------------------------------------------------
# Callbacks y helpers de estado
# ----------------------------------------------------------------------------
def _unconfirm() -> None:
    st.session_state["vv_confirmed"] = None


def _pick_frame(key: str, idx: int) -> None:
    st.session_state[key] = int(idx)
    st.session_state["vv_confirmed"] = None


def _confirm(digest: str, throws: str, frame: int) -> None:
    st.session_state["vv_confirmed"] = {"digest": digest, "throws": throws, "frame": int(frame)}


def confirmed_frame(digest: str, throws: str) -> int | None:
    c = st.session_state.get("vv_confirmed")
    return int(c["frame"]) if c and c["digest"] == digest and c["throws"] == throws else None


def _send_to_sculptor(params: dict) -> None:
    """Carga la mecánica medida como pitcheo activo del Sculptor (``active_pitch_params``).

    Guarda también la base (parámetros del Simulador al hacer clic) para que ``apply_velovision_override``
    sepa cuándo el Simulador cambió y hay que soltar el override.
    """
    base = dict(st.session_state.get("active_pitch_params") or {})
    new = {**base, **params}
    st.session_state["vv_override"] = {"base": base, "params": new}
    st.session_state["active_pitch_params"] = new
    st.session_state["vv_sent"] = True


def apply_velovision_override() -> None:
    """Mantiene los parámetros de VeloVision en ``active_pitch_params`` frente al Simulador.

    El Simulador reescribe ``active_pitch_params`` en cada ejecución; ``app.py`` llama esto justo antes de
    pintar el Sculptor. El override vale mientras el Simulador no cambie respecto del momento del clic.
    Quita el Stuff+ del Simulador de ``last_sim_results`` para que el Sculptor no lo tome como del prospecto.
    """
    ov = st.session_state.get("vv_override")
    if not ov:
        return
    if (st.session_state.get("active_pitch_params") or {}) != ov["base"]:
        st.session_state["vv_override"] = None
        st.session_state["vv_sent"] = False
        return
    st.session_state["active_pitch_params"] = dict(ov["params"])
    last = dict(st.session_state.get("last_sim_results") or {})
    last.pop("stuff_cdmx", None)
    st.session_state["last_sim_results"] = last


@st.cache_data(show_spinner=False, max_entries=64)
def _card(slot: float, throws: str, err: float | None, velo: float | None, kind: str | None = None) -> dict:
    return PC.build_prospect_card(slot, throws, slot_error_deg=err, observed_velo_mph=velo, slot_error_kind=kind)


@st.cache_data(show_spinner=False)
def _pitchers_of_hand(throws: str) -> int:
    """Total de pitchers de esa mano en el catálogo (para el % de cobertura de la ventana)."""
    import pandas as pd
    df = pd.read_csv(PC.DEFAULT_CATALOG, usecols=["pitcher_anon_id", "PitcherThrows"])
    return int(df.loc[df["PitcherThrows"].map(PC.normalize_throws) == PC.normalize_throws(throws), "pitcher_anon_id"].nunique())


@st.cache_data(show_spinner=False, max_entries=32)
def _narrowing(slot: float, throws: str, err: float | None, velo: float, kind: str | None = None) -> list:
    return PC.velo_narrowing_report(slot, throws, velo, slot_error_deg=err, slot_error_kind=kind)


KIND_SYS = "sistemático supuesto"
KIND_MAE = "medido (MAE de validación)"
KIND_BOTH = "combinado (MAE + sistemático)"


def slot_error_for(m: dict, mae_slot_deg: float | None = None) -> tuple[float | None, str | None]:
    """``(error, tipo)`` del slot para la ventana. Hoy no hay MAE de validación (17-D), así que el tipo es
    siempre "sistemático supuesto"; ``(None, None)`` si no hay error sistemático."""
    sys_err = m.get("slot_sys_err_deg")
    combine = getattr(PS, "combined_slot_error_deg", None)
    if sys_err is None or combine is None:
        return (None if mae_slot_deg is None else float(mae_slot_deg)), (None if mae_slot_deg is None else KIND_MAE)
    return float(combine(mae_slot_deg, sys_err)), (KIND_SYS if mae_slot_deg is None else KIND_BOTH)


def _err_text(sys_val: float | None, unit: str, decimals: int) -> str:
    if sys_val is None:
        return ERR_PENDING
    return f"{ERR_PENDING} · error sistemático supuesto ±{sys_val:.{decimals}f} {unit}"


# ----------------------------------------------------------------------------
# Secciones
# ----------------------------------------------------------------------------
def _inputs() -> dict:
    """Video, mano, estatura y radar. Devuelve los valores (el video como ``(bytes, hash)`` o ``None``)."""
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        clip = pick_clip("vv")
    with right:
        hand = st.radio("Mano lanzadora", list(HAND_LABELS), horizontal=True, key="vv_hand")
        unit = st.radio("Estatura en", ["ft", "cm"], horizontal=True, key="vv_unit")
        if unit == "ft":
            h_ft = st.number_input("Estatura del lanzador (ft)", FT_RANGE[0], FT_RANGE[1], 6.0, 0.05, key="vv_h_ft")
        else:
            cm = st.number_input("Estatura del lanzador (cm)", round(FT_RANGE[0] * CM_PER_FT), round(FT_RANGE[1] * CM_PER_FT),
                                 183, 1, key="vv_h_cm")
            h_ft = cm / CM_PER_FT
        use_radar = st.checkbox("Tengo una lectura de radar de la recta", key="vv_use_radar",
                                help="Opcional. La ingresas tú: el video no la mide.")
        velo = None
        if use_radar:
            velo = float(st.number_input("Velocidad de la recta del radar (mph)", 60.0, 105.0, 90.0, 0.5, key="vv_velo"))
    return {"clip": clip, "throws": HAND_LABELS[hand], "height_ft": float(h_ft), "velo": velo}


def _frame_picker(seq: PS.PoseSequence, digest: str, throws: str) -> tuple[int, dict]:
    """Barra de frames con el release sugerido y sus candidatos; devuelve (frame elegido, detección)."""
    det = PS.detect_release_frame(seq, throws)
    cands = [int(c["frame_idx"]) for c in det["candidates"]]
    n = seq.n_frames
    key = f"vv_frame_{digest[:10]}_{throws}"
    st.session_state.setdefault(key, int(det["frame_idx"]))
    sel = int(st.session_state[key])
    conf = confirmed_frame(digest, throws)
    st.markdown(C.frame_strip(n, int(det["frame_idx"]), cands, sel, conf), unsafe_allow_html=True)
    if n > 1:
        st.slider("Frame", 0, n - 1, key=key, on_change=_unconfirm, label_visibility="collapsed")
        sel = int(st.session_state[key])
    return sel, det


def _measure_cards(m: dict) -> str:
    ok = bool(m["quality_ok"])
    tag = "" if ok else "calidad insuficiente"
    c1 = C.measure_card("up", "Altura de soltura", m["rel_height_ft"], "ft",
                        _err_text(m.get("rel_height_sys_err_ft"), "ft", 2), tag, 2)
    c2 = C.measure_card("right", "Lado de soltura", m["rel_side_ft"], "ft",
                        _err_text(m.get("rel_side_sys_err_ft"), "ft", 2), tag, 2)
    c3 = C.measure_card("arm", "Arm slot (definición del dataset)", m["arm_slot_dataset_deg"], "°",
                        _err_text(m.get("slot_sys_err_deg"), "°", 1), "filtra el catálogo" if ok else tag)
    c4 = C.measure_card("target", "Ángulo del brazo (biomecánico)", m["arm_angle_biomech_deg"], "°",
                        ERR_PENDING, "informativo, no filtra")
    return f'<div class="dr-mini2" style="grid-template-columns:repeat(2,1fr);gap:.7rem">{c1}{c2}{c3}{c4}</div>'


def _render_ficha(m: dict, throws: str, velo: float | None, digest: str, height_ft: float) -> None:
    slot = float(m["arm_slot_dataset_deg"])
    err, kind = slot_error_for(m)
    card = _card(slot, throws, err, velo, kind)

    st.markdown(f'<div class="dr-h4" style="font-size:1rem;margin-top:.6rem">{C.icon("users")}Ficha de prospecto · '
                f'rango de Stuff+ CDMX por familia</div>', unsafe_allow_html=True)
    if err is None:
        error_label = "aún no medido"
    elif kind == KIND_SYS:
        error_label = f"{kind} ±{err:.1f}° · error de medición aún no medido"
    else:
        error_label = f"{card['error_source']} ±{err:.1f}°"
    total_hand = _pitchers_of_hand(throws)
    cover = 100.0 * card["n_pitchers_total"] / total_hand if total_hand else 0.0
    st.markdown(
        '<div class="dr-stats">'
        f'<span class="dr-stat">Arm slot medido <b>{slot:.1f}°</b></span>'
        f'<span class="dr-stat">Ventana de búsqueda <b>±{card["window_deg"]:.1f}°</b></span>'
        f'<span class="dr-stat">Cobertura: <b>{cover:.0f} %</b> de los pitchers de esa mano'
        + (" (incluye el filtro de radar)" if velo else "") + '</span>'
        f'<span class="dr-stat">Error del slot: <b>{C.esc(error_label)}</b></span>'
        f'<span class="dr-stat">Pitchers LMB comparables <b>{card["n_pitchers_total"]}</b></span>'
        + (f'<span class="dr-stat">Recta del radar <b>{velo:.1f} mph</b> ± {PC.FB_VELO_TOL_MPH:g}</span>' if velo else "")
        + '</div>', unsafe_allow_html=True)
    if card["n_pitchers_total"] == 0:
        st.markdown(C.callout("No hay pitchers de la LMB con esta mano y este arm slot dentro de la ventana. "
                              "No se muestran rangos."), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="dr-card">{C.family_ranges(card["families"], card["filters_applied"]["min_pitchers"])}</div>',
                    unsafe_allow_html=True)
    notes = []
    if cover > 60:
        notes.append("Con este margen de error el slot distingue poco; la diferencia la marca sobre todo la familia de pitcheo.")
    notes += ["Orden: mediana de Stuff+ CDMX de mayor a menor. Cada barra resume pitchers reales de la LMB con una "
             "mecánica parecida; no es un valor del prospecto."]
    if err is not None:
        notes.append("El error del slot incluye solo el error sistemático supuesto; el error de medición contra "
                     "anotadores humanos queda pendiente de la validación, y la ventana mínima es ±6°.")
    else:
        notes.append("El error de medición aún no se midió: se usa la ventana mínima de ±6°.")
    st.markdown("".join(f'<div class="note">{C.esc(n)}</div>' for n in notes), unsafe_allow_html=True)
    st.markdown(C.disclaimer_badge(card["disclaimer"]), unsafe_allow_html=True)

    if velo is not None:
        st.markdown(f'<div class="dr-h4" style="margin-top:.8rem">{C.icon("gauge")}Efecto del radar sobre el ancho del rango</div>',
                    unsafe_allow_html=True)
        st.markdown(C.narrowing_table(_narrowing(slot, throws, err, velo, kind)), unsafe_allow_html=True)
        st.markdown('<div class="note">Se reporta el ancho p25–p75 sin y con la lectura de radar; el radar no '
                    'siempre lo reduce.</div>', unsafe_allow_html=True)

    _render_sculptor_button(m, throws, velo, digest, slot)


def _render_sculptor_button(m: dict, throws: str, velo: float | None, digest: str, slot: float) -> None:
    st.markdown(f'<div class="dr-h4" style="margin-top:.8rem">{C.icon("target")}Llevar esta mecánica al Sculptor</div>',
                unsafe_allow_html=True)
    h_default = m.get("rel_height_ft")
    h_default = None if h_default is None else float(min(8.0, max(3.0, h_default)))
    c1, c2, c3 = st.columns([1, 1, 1.3], vertical_alignment="bottom")
    h = c1.number_input("Altura de soltura (ft)", 3.0, 8.0, h_default, 0.05, key=f"vv_sc_h_{digest[:8]}_{h_default}",
                        help="El Sculptor la necesita; se llena con la altura medida y puedes corregirla.")
    v = c2.number_input("Velocidad de la recta (mph)", 60.0, 105.0, velo, 0.5, key=f"vv_sc_v_{digest[:8]}_{velo}",
                        help="El Sculptor la necesita; ingresa la del radar.")
    ready = h is not None and v is not None
    params = {"PitcherThrows": throws, "arm_slot_deg": float(min(90.0, max(0.0, slot))),
              "RelHeight": float(h) if h is not None else None, "fb_velo": float(v) if v is not None else None}
    c3.button("Buscar referencias factibles", key="vv_to_sculptor", type="primary", width="stretch",
              disabled=not ready, on_click=_send_to_sculptor, args=(params,))
    if not ready:
        st.markdown('<div class="note">Falta la altura o la velocidad de la recta: ingrésalas para continuar '
                    '(el Sculptor no acepta valores vacíos).</div>', unsafe_allow_html=True)
    elif st.session_state.get("vv_sent"):
        st.markdown(f'<span class="dr-ok">{C.icon("check")}Parámetros cargados: abre la pestaña Sculptor.</span>',
                    unsafe_allow_html=True)


def _render_quality_gate(m: dict) -> None:
    st.markdown(C.callout(
        f"Calidad {m['quality_score']:.2f} (mínimo {PS.QUALITY_MIN:.2f}): no se generan rangos con este video."),
        unsafe_allow_html=True)
    for issue in m["quality_issues"]:
        st.markdown(C.callout(issue), unsafe_allow_html=True)
    if not m["quality_issues"]:
        st.markdown(C.callout("Regraba siguiendo la guía de captura o elige otro frame de soltura."),
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------------
def render_velovision_tab() -> None:
    st.markdown("### VeloVision · ficha de prospecto desde video")
    st.caption("Un video de celular da el punto de soltura y el arm slot con su error; con eso se ve qué rango de "
               "Stuff+ CDMX tienen pitchers reales de la LMB con una mecánica parecida.")
    st.markdown(C.disclaimer_badge(PC.DISCLAIMER), unsafe_allow_html=True)
    st.markdown(C.capture_guide(GUIDE), unsafe_allow_html=True)
    st.markdown(f'<div class="note" style="margin:.4rem 0 .8rem">{C.icon("lock")} {C.esc(PRIVACY)}</div>',
                unsafe_allow_html=True)

    inp = _inputs()
    if inp["clip"] is None:
        st.markdown(C.empty_state(
            "film", "Sube un video para empezar",
            "Graba de frente y con el cuerpo completo (mira la guía de arriba) y súbelo, o elige un clip de "
            "demostración si tienes alguno en la carpeta local."), unsafe_allow_html=True)
        return

    data, digest = inp["clip"]
    throws, height_ft, velo = inp["throws"], inp["height_ft"], inp["velo"]
    seq = get_sequence(data, digest)
    if seq is None:
        return

    st.markdown(f'<div class="dr-h4" style="margin-top:.6rem">{C.icon("frames")}Elige el frame de soltura</div>',
                unsafe_allow_html=True)
    col_img, col_meas = st.columns([1.25, 1], gap="large")
    with col_img:
        sel, det = _frame_picker(seq, digest, throws)
        m = PS.measure_release(seq, sel, throws, height_ft)
        st.image(PS.annotate_frame(seq, sel, m), width="stretch")
        cands = [int(c["frame_idx"]) for c in det["candidates"]]
        key = f"vv_frame_{digest[:10]}_{throws}"
        cols = st.columns(len(cands) + 1) if cands else st.columns(1)
        for i, (col, f) in enumerate(zip(cols, cands)):
            col.button(f"Candidato {i + 1}: frame {f}", key=f"vv_cand_{i}", width="stretch",
                       on_click=_pick_frame, args=(key, f))
        cols[-1].button(f"Confirmar soltura (frame {sel})", key="vv_confirm", type="primary", width="stretch",
                        on_click=_confirm, args=(digest, throws, sel))
        st.markdown(f'<div class="note">Sugerido por el sistema: frame {det["frame_idx"]} '
                    f'(confianza {det["confidence"]:.2f}). Mueve la barra para elegir otro.</div>', unsafe_allow_html=True)
    with col_meas:
        st.markdown(_measure_cards(m), unsafe_allow_html=True)
        origin = m.get("origin_notes")
        origin = ("; ".join(origin) if isinstance(origin, (list, tuple)) else origin) or None
        if origin:
            st.markdown(C.note_box(origin), unsafe_allow_html=True)
        st.markdown('<div class="note">Altura y lado son aproximaciones en 2D con origen distinto al de los datos '
                    'de la liga (TrackMan); por eso se reportan con su error.</div>', unsafe_allow_html=True)

    gate_ok = bool(m["quality_ok"]) and m["quality_score"] >= PS.QUALITY_MIN and m["arm_slot_dataset_deg"] is not None
    if not gate_ok:
        _render_quality_gate(m)
        return
    if confirmed_frame(digest, throws) != sel:
        st.markdown(C.note_box("Confirma la soltura (botón de arriba) para generar la ficha de prospecto."),
                    unsafe_allow_html=True)
        return
    st.markdown(f'<span class="dr-ok">{C.icon("check")}Soltura confirmada en el frame {sel}</span>', unsafe_allow_html=True)
    _render_ficha(m, throws, velo, digest, height_ft)

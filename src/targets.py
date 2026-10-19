"""
src/targets.py
==============
Calibración empírica LMB de la expectativa de carreras (RE12), valores de carrera
desde la perspectiva del lanzador (`xrv_pitcher`) y targets multi-objetivo.

Cumple con:
- docs/CONTRACT.md §1 y §2 (HACK-006).
- docs/TASKS.md (HACK-006).
- AGENTS.md §2 (Física correcta, sin constantes fijas inventadas de MLB).

PROHIBICIÓN ESTRICTA:
Cero constantes hardcodeadas de MLB (e.g. -1.40 o +0.28) y cero ajustes post-hoc
forzados sobre la jerarquía. Todos los valores de carrera se estiman empíricamente
a partir de `RunsScored` y `OutsOnPlay` del DataFrame de entrenamiento (`year <= 2025`).
El suavizado de estados escasos se deriva exclusivamente de estadísticas marginales
de la propia muestra histórica.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.config import TARGET_COLS

# Eventos canónicos para el cálculo de valores de carrera
EVENT_KEYS = [
    "home_run",
    "triple",
    "double",
    "single",
    "walk_or_hbp",
    "strikeout",
    "out_in_play",
    "called_strike",
    "swinging_strike",
    "foul_lt2",
    "foul_2s",
    "ball",
]

# Conjunto de eventos terminales de aparición al plato
TERMINAL_EVENTS = {
    "home_run",
    "triple",
    "double",
    "single",
    "walk_or_hbp",
    "strikeout",
    "out_in_play",
}


def classify_events(df: pd.DataFrame) -> pd.Series:
    """
    Clasifica de manera exhaustiva y vectorizada cada lanzamiento en uno de los
    12 eventos canónicos de béisbol para la estimación de valores lineales de carrera.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos que contiene columnas de resultado de TrackMan.

    Returns
    -------
    pd.Series
        Serie con la categoría de evento para cada fila (strings).
    """
    is_hr = df.get("home_run", 0) == 1
    is_3b = df.get("triple", 0) == 1
    is_2b = df.get("double", 0) == 1
    is_1b = df.get("single", 0) == 1

    # Ponche (strikeout): indicador directo o KorBB/play_result
    korbb_str = df.get("KorBB", pd.Series("", index=df.index)).astype(str)
    presult_str = df.get("play_result", pd.Series("", index=df.index)).astype(str)
    pcall_str = df.get("PitchCall", pd.Series("", index=df.index)).astype(str)

    is_so = (
        (df.get("strikeout", 0) == 1)
        | (korbb_str == "Strikeout")
        | (presult_str == "Strikeout")
    )

    # Base por bolas o golpe por lanzamiento
    is_bb = (
        (df.get("base_on_balls", 0) == 1)
        | (df.get("is_hit_by_pitch", 0) == 1)
        | (korbb_str == "Walk")
        | presult_str.isin(["Walk", "HitByPitch"])
        | (pcall_str == "HitByPitch")
    )

    # Out en jugada (batazo que produce out sin ser ponche)
    outs_on_play = df.get("OutsOnPlay", pd.Series(0, index=df.index)).fillna(0)
    is_bip = df.get("is_ball_in_play", 0) == 1
    is_out_ip = (
        (is_bip | presult_str.isin(["Out", "FieldersChoice", "Sacrifice", "DoublePlay"]))
        & (outs_on_play > 0)
        & (~is_so)
    )

    # Fouls diferenciados por conteo de strikes (<2 vs 2 strikes)
    strikes = df.get("Strikes", pd.Series(0, index=df.index)).fillna(0).astype(int)
    is_foul_call = (
        pcall_str.str.startswith("Foul")
        | pcall_str.isin(["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"])
        | (presult_str == "FoulBall")
    )
    is_foul_lt2 = is_foul_call & (strikes < 2)
    is_foul_2s = is_foul_call & (strikes >= 2)

    # Strikes cantados y abanicados (whiff / swinging)
    is_sw_str = (
        (df.get("is_swinging_strike", 0) == 1)
        | (pcall_str == "StrikeSwinging")
        | (df.get("is_whiff", 0) == 1)
    ) & (~is_so) & (~is_foul_call)

    is_call_str = (
        (df.get("is_called_strike", 0) == 1)
        | (pcall_str == "StrikeCalled")
    ) & (~is_so)

    conds = [
        is_hr,
        is_3b,
        is_2b,
        is_1b,
        is_so,
        is_bb,
        is_out_ip,
        is_foul_lt2,
        is_foul_2s,
        is_sw_str,
        is_call_str,
    ]
    choices = [
        "home_run",
        "triple",
        "double",
        "single",
        "strikeout",
        "walk_or_hbp",
        "out_in_play",
        "foul_lt2",
        "foul_2s",
        "swinging_strike",
        "called_strike",
    ]

    event_array = np.select(conds, choices, default="ball")
    return pd.Series(event_array, index=df.index, name="event_type")


def fit_lmb_run_values(train_df: pd.DataFrame) -> dict[str, Any]:
    """
    Ajusta empíricamente los valores lineales de carrera (LMB Run Values) y la
    matriz de expectativa de carreras RE12 por conteo utilizando exclusivamente
    los datos de entrenamiento de la LMB (`year <= 2025`).

    Metodología 100% empírica:
    - Cero constantes inventadas de MLB.
    - Preserva el orden cronológico exacto de los lanzamientos dentro de cada media entrada.
    - Calcula la suma sufija de `RunsScored` (`runs_to_end`) en cada media entrada.
    - Estima la matriz de expectativa de carreras RE12 `(Outs, Balls, Strikes)` a partir
      del promedio empírico observado de `runs_to_end` en cada celda, suavizado mediante
      Empirical Bayes con el prior marginal aditivo de la propia muestra:
      `mu_marg(o, b, s) = max(0.0, R_o[o] + (R_b[b] - R_bar) + (R_s[s] - R_bar))`.
    - Estima el valor de cada evento observed `xrv` a partir del cambio empírico en la
      expectativa de carrera observada hasta el final de la media entrada (`x_obs`) y
      la transición de estado a un paso (`x_trans`).
    - Reporta como resultado del modelo si la jerarquía física se cumple empíricamente
      (`empirical_hierarchy_holds`).

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame de lanzamientos con datos de entrenamiento. Debe incluir columnas
        'year', 'RunsScored', 'OutsOnPlay', 'Inning', 'Top/Bottom', 'Outs', 'Balls', 'Strikes'.

    Returns
    -------
    dict[str, Any]
        Diccionario con llaves:
        - 're_matrix': dict con tuplas (Outs, Balls, Strikes) -> float con expectativa de carreras.
        - 'event_xrv': dict con los 12 eventos canónicos -> float (perspectiva pitcher: + = bueno).
        - 'count_transition_xrv': dict con transiciones de conteo -> float.
        - 'train_years': list[int] con los años usados para el ajuste (excluye estrictamente 2026).
        - 'empirical_hierarchy_holds': bool indicando si la jerarquía física se cumple empíricamente.

    Raises
    ------
    ValueError
        Si `train_df` está vacío, no contiene datos para `year <= 2025`, o carece
        de las columnas necesarias.
    """
    if train_df is None or len(train_df) == 0:
        raise ValueError("train_df está vacío.")

    if "year" not in train_df.columns:
        raise ValueError("train_df no contiene la columna 'year'.")

    # Filtrar exclusivamente datos históricos (<= 2025) para evitar data leakage
    df = train_df[train_df["year"] <= 2025].copy()
    if len(df) == 0:
        raise ValueError(
            "train_df no contiene registros de entrenamiento para year <= 2025."
        )

    required_cols = [
        "game_anon_id", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes",
        "RunsScored", "OutsOnPlay"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en train_df: {missing}")

    train_years = sorted(int(y) for y in df["year"].unique())

    # 1. Suma sufija de carreras: runs_to_end_of_inning
    # Preserva el orden relativo original de los lanzamientos dentro de cada media entrada
    df["runs_to_end"] = df.groupby(
        ["game_anon_id", "Inning", "Top/Bottom"], sort=False, observed=False
    )["RunsScored"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1]).astype(np.float64)

    # 2. Medias marginales observadas en la muestra (2024-25)
    r_bar = float(df["runs_to_end"].mean()) if len(df) > 0 else 0.0

    r_o: dict[int, float] = {}
    for o in (0, 1, 2):
        sub_o = df[df["Outs"] == o]["runs_to_end"]
        r_o[o] = float(sub_o.mean()) if len(sub_o) > 0 else r_bar

    r_b: dict[int, float] = {}
    for b in (0, 1, 2, 3):
        sub_b = df[df["Balls"] == b]["runs_to_end"]
        r_b[b] = float(sub_b.mean()) if len(sub_b) > 0 else r_bar

    r_s: dict[int, float] = {}
    for s in (0, 1, 2):
        sub_s = df[df["Strikes"] == s]["runs_to_end"]
        r_s[s] = float(sub_s.mean()) if len(sub_s) > 0 else r_bar

    # Estadísticas por estado empírico (Outs, Balls, Strikes)
    state_stats = df.groupby(
        ["Outs", "Balls", "Strikes"], observed=False
    )["runs_to_end"].agg(["count", "mean"])

    valid_counts = [
        state_stats.loc[(o, b, s), "count"]
        for o in range(3) for b in range(4) for s in range(3)
        if (o, b, s) in state_stats.index and state_stats.loc[(o, b, s), "count"] > 0
    ]
    m_prior = float(np.median(valid_counts)) if len(valid_counts) > 0 else 1.0

    # 3. Matriz de Expectativa de Carreras RE12 mediante Empirical Bayes
    # mu_marg(o, b, s) = max(0.0, R_o[o] + (R_b[b] - R_bar) + (R_s[s] - R_bar))
    re_matrix: dict[tuple[int, int, int], float] = {}
    for o in range(3):
        for b in range(4):
            for s in range(3):
                mu_marg = max(0.0, r_o[o] + (r_b[b] - r_bar) + (r_s[s] - r_bar))
                if (o, b, s) in state_stats.index and state_stats.loc[(o, b, s), "count"] > 0:
                    n_cell = float(state_stats.loc[(o, b, s), "count"])
                    y_cell = float(state_stats.loc[(o, b, s), "mean"])
                    re_val = (n_cell * y_cell + m_prior * mu_marg) / (n_cell + m_prior)
                else:
                    re_val = mu_marg
                re_matrix[(o, b, s)] = float(re_val)

    # 4. Clasificación de eventos observados
    df["event_type"] = classify_events(df)

    outs = df["Outs"].astype(int).values
    balls = df["Balls"].astype(int).values
    strikes = df["Strikes"].astype(int).values
    runs_scored = df["RunsScored"].fillna(0).values
    outs_on_play = df["OutsOnPlay"].fillna(0).astype(int).values
    runs_to_end = df["runs_to_end"].values
    events = df["event_type"].values
    n_rows = len(df)

    # RE en el estado previo al lanzamiento
    re_curr = np.array([re_matrix.get((o, b, s), r_bar) for o, b, s in zip(outs, balls, strikes)])

    # Cambio observado hasta el final de la media entrada (perspectiva del pitcher: + = bueno)
    # x_obs = -(runs_to_end - re_curr) = re_curr - runs_to_end
    x_obs = re_curr - runs_to_end

    # Transición de estado a un paso
    # x_trans = -(RunsScored + RE(next_state) - RE(curr_state))
    re_next = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        ev = events[i]
        o = outs[i]
        b = balls[i]
        s = strikes[i]
        if ev == "ball":
            re_next[i] = re_matrix.get((o, min(b + 1, 3), s), r_bar)
        elif ev in ("called_strike", "swinging_strike", "foul_lt2"):
            re_next[i] = re_matrix.get((o, b, min(s + 1, 2)), r_bar)
        elif ev == "foul_2s":
            re_next[i] = re_matrix.get((o, b, 2), r_bar)
        elif ev in ("strikeout", "out_in_play"):
            o_next = o + max(1, outs_on_play[i])
            re_next[i] = 0.0 if o_next >= 3 else re_matrix.get((o_next, 0, 0), r_bar)
        else:
            re_next[i] = re_matrix.get((o, 0, 0), r_bar)

    x_trans = -(runs_scored + re_next - re_curr)

    df["x_obs"] = x_obs
    df["x_trans"] = x_trans

    # 5. Estimación de event_xrv mediante estadísticas observadas en la muestra
    event_counts = df["event_type"].value_counts().to_dict()
    valid_ev_counts = [c for c in event_counts.values() if c > 0]
    m_e = float(np.median(valid_ev_counts)) if len(valid_ev_counts) > 0 else 1.0

    terminal_hits = {"home_run", "triple", "double", "single", "walk_or_hbp"}
    sub_hits = df[df["event_type"].isin(terminal_hits)]
    mean_hits_obs = float(sub_hits["x_obs"].mean()) if len(sub_hits) > 0 else 0.0

    sub_outs = df[df["event_type"].isin({"strikeout", "out_in_play"})]
    mean_outs_obs = float(sub_outs["x_obs"].mean()) if len(sub_outs) > 0 else 0.0

    event_xrv: dict[str, float] = {}
    for ev in EVENT_KEYS:
        sub = df[df["event_type"] == ev]
        n_ev = len(sub)
        if ev in terminal_hits:
            raw_val = float(sub["x_obs"].mean()) if n_ev > 0 else mean_hits_obs
            prior_val = mean_hits_obs
        elif ev in {"strikeout", "out_in_play"}:
            raw_val = float(sub["x_obs"].mean()) if n_ev > 0 else mean_outs_obs
            prior_val = mean_outs_obs
        else:
            raw_val = float(sub["x_trans"].mean()) if n_ev > 0 else 0.0
            prior_val = raw_val

        shrunk_val = (n_ev * raw_val + m_e * prior_val) / (n_ev + m_e) if (n_ev + m_e) > 0 else raw_val
        event_xrv[ev] = float(shrunk_val)

    # Evaluar empíricamente si la jerarquía física se cumple sin forzar ningún ajuste
    empirical_hierarchy_holds = bool(
        event_xrv["strikeout"] > event_xrv["called_strike"] > event_xrv["ball"] > event_xrv["single"] > event_xrv["home_run"]
    )

    # 6. Transiciones de conteo observadas
    count_transition_xrv: dict[str, float] = {
        "ball": event_xrv["ball"],
        "called_strike": event_xrv["called_strike"],
        "swinging_strike": event_xrv["swinging_strike"],
        "foul_lt2": event_xrv["foul_lt2"],
        "foul_2s": event_xrv["foul_2s"],
    }
    for b in range(4):
        for s in range(3):
            if b < 3:
                diff_b = np.mean([
                    re_matrix[(o, b, s)] - re_matrix[(o, b + 1, s)]
                    for o in (0, 1, 2)
                ])
                count_transition_xrv[f"({b},{s})->({b+1},{s})"] = float(diff_b)
            if s < 2:
                diff_s = np.mean([
                    re_matrix[(o, b, s)] - re_matrix[(o, b, s + 1)]
                    for o in (0, 1, 2)
                ])
                count_transition_xrv[f"({b},{s})->({b},{s+1})"] = float(diff_s)

    return {
        "re_matrix": re_matrix,
        "event_xrv": event_xrv,
        "count_transition_xrv": count_transition_xrv,
        "train_years": train_years,
        "empirical_hierarchy_holds": empirical_hierarchy_holds,
    }


def attach_targets(df: pd.DataFrame, rv_table: dict[str, Any]) -> pd.DataFrame:
    """
    Agrega las 8 columnas target oficiales multi-objetivo y el valor de carrera
    empírico (`xrv_pitcher`) a un DataFrame de lanzamientos.

    Columnas creadas (docs/CONTRACT.md §2 HACK-006):
    1. target_whiff: float64
    2. target_chase: float64
    3. target_called_strike: float64
    4. target_groundball: float64
    5. target_weak_contact: float64
    6. target_hard_contact: float64
    7. target_out_on_pitch: float64
    8. xrv_pitcher: float64

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de lanzamientos. No se modifica in-place (trabaja sobre copia).
    rv_table : dict[str, Any]
        Tabla de valores de carrera ajustada con `fit_lmb_run_values`.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con las 8 columnas target agregadas en float64 sin NaN.
    """
    if "event_xrv" not in rv_table:
        raise ValueError("rv_table no contiene la llave obligatoria 'event_xrv'.")

    out = df.copy()

    # 1. target_whiff
    out["target_whiff"] = (out.get("is_whiff", 0).fillna(0) == 1).astype(np.float64)

    # 2. target_chase (swing a pitcheo fuera de zona)
    out["target_chase"] = (
        out.get("swung_outside_strike_zone", 0).fillna(0) == 1
    ).astype(np.float64)

    # 3. target_called_strike
    out["target_called_strike"] = (
        out.get("is_called_strike", 0).fillna(0) == 1
    ).astype(np.float64)

    # 4. target_groundball
    hit_type_col = out.get("hit_type", pd.Series(None, index=out.index))
    out["target_groundball"] = (hit_type_col == "GroundBall").astype(np.float64)

    # 5. target_weak_contact: bola en juego y (ExitSpeed < 85 o hit_type == 'Popup')
    is_bip = out.get("is_ball_in_play", 0).fillna(0) == 1
    exit_speed = pd.to_numeric(out.get("ExitSpeed", np.nan), errors="coerce")
    is_weak = is_bip & ((exit_speed < 85.0) | (hit_type_col == "Popup"))
    out["target_weak_contact"] = is_weak.astype(np.float64)

    # 6. target_hard_contact: bola en juego y (ExitSpeed >= 95 o home_run == 1)
    is_hr = out.get("home_run", 0).fillna(0) == 1
    is_hard = is_bip & ((exit_speed >= 95.0) | is_hr)
    out["target_hard_contact"] = is_hard.astype(np.float64)

    # 7. target_out_on_pitch: OutsOnPlay > 0 o strikeout == 1
    outs_on_play = pd.to_numeric(out.get("OutsOnPlay", 0), errors="coerce").fillna(0)
    is_so = out.get("strikeout", 0).fillna(0) == 1
    out["target_out_on_pitch"] = ((outs_on_play > 0) | is_so).astype(np.float64)

    # 8. xrv_pitcher: asigna a cada lanzamiento su valor de carrera según el evento observado
    event_series = classify_events(out)
    event_map = rv_table["event_xrv"]
    out["xrv_pitcher"] = event_series.map(event_map).fillna(0.0).astype(np.float64)

    # Validación final estricta de presencia, tipos y ausencia de NaN
    for target in TARGET_COLS:
        if target not in out.columns:
            raise RuntimeError(f"Columna target '{target}' no fue creada correctamente.")
        if out[target].dtype != np.float64:
            out[target] = out[target].astype(np.float64)
        if out[target].isna().any():
            raise RuntimeError(f"Columna target '{target}' contiene valores NaN.")

    return out

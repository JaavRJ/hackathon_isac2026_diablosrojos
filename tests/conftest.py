"""
Configuración y fixtures globales para la suite de pruebas (pytest).
Proporciona el dataset sintético determinista LMB de 1,200 filas con las 84 columnas oficiales.
Cumple con docs/CONTRACT.md §4 y docs/TASKS.md HACK-001.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
import pytest

# Lista canónica de las 84 columnas del diccionario (stuff_plus_data_dictionary.csv)
OFFICIAL_84_COLUMNS = [
    "year", "PitchUID", "game_anon_id", "pitcher_anon_id", "batter_anon_id", "catcher_anon_id",
    "PitcherThrows", "BatterSide", "altitude_category", "AutoPitchType", "RelSpeed", "EffectiveVelo",
    "ZoneSpeed", "SpinRate", "SpinAxis", "Tilt", "RelHeight", "RelSide", "Extension",
    "VertBreak", "InducedVertBreak", "HorzBreak", "VertRelAngle", "HorzRelAngle", "VertApprAngle",
    "HorzApprAngle", "SpeedDrop", "ZoneTime", "x0", "y0", "z0", "vx0", "vy0", "vz0",
    "ax0", "ay0", "az0", "pfxx", "pfxz", "PitchTrajectoryXc0", "PitchTrajectoryXc1", "PitchTrajectoryXc2",
    "PitchTrajectoryYc0", "PitchTrajectoryYc1", "PitchTrajectoryYc2", "PitchTrajectoryZc0",
    "PitchTrajectoryZc1", "PitchTrajectoryZc2", "PitchCall", "KorBB", "play_result", "hit_type",
    "ExitSpeed", "Angle", "Direction", "Distance", "is_swing", "is_whiff", "is_contact",
    "is_called_strike", "is_swinging_strike", "is_ball_in_play", "is_batted", "is_hit",
    "single", "double", "triple", "home_run", "base_on_balls", "strikeout", "is_hit_by_pitch",
    "RunsScored", "OutsOnPlay", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "count",
    "PlateLocHeight", "PlateLocSide", "in_strike_zone", "outside_strike_zone", "swung_outside_strike_zone"
]


def make_synthetic_lmb_df(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame determinista de lanzamientos LMB con las 84 columnas exactas,
    física realista por altitud y orden de aparición al plato.

    Parameters
    ----------
    n_rows : int, default=1200
        Número total de lanzamientos a generar.
    seed : int, default=42
        Semilla para el generador determinista.

    Returns
    -------
    pd.DataFrame
        DataFrame con 84 columnas exactas y n_rows filas.
    """
    rng = np.random.default_rng(seed)

    # 18 juegos repartidos equitativamente en 3 categorías de altitud
    game_altitudes: dict[str, str] = {}
    for g in range(1, 19):
        gid = f"game_{g:04d}"
        if g <= 6:
            game_altitudes[gid] = "No Altitude"
        elif g <= 12:
            game_altitudes[gid] = "Medium Altitude"
        else:
            game_altitudes[gid] = "Extreme Altitude"

    # 3 temporadas repartidas uniformemente (2024, 2025, 2026)
    game_seasons: dict[str, int] = {}
    for g in range(1, 19):
        gid = f"game_{g:04d}"
        mod = (g - 1) % 6
        if mod in (0, 1):
            game_seasons[gid] = 2024
        elif mod in (2, 3):
            game_seasons[gid] = 2025
        else:
            game_seasons[gid] = 2026

    # 12 pitchers: pitcher_0001 .. pitcher_0012
    # 9 diestros (Right), 3 zurdos (Left)
    pitchers = [f"pitcher_{i:04d}" for i in range(1, 13)]
    pitcher_hand = {p: ("Right" if int(p.split("_")[1]) <= 9 else "Left") for p in pitchers}

    # Receptores y bateadores
    catchers = [f"catcher_{i:04d}" for i in range(1, 7)]
    batters = [f"batter_{i:04d}" for i in range(1, 37)]
    batter_hands = {b: str(rng.choice(["Right", "Left", "Switch"], p=[0.68, 0.28, 0.04])) for b in batters}

    # Tipos de pitcheo TrackMan
    pitch_types = [
        "Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter",
        "OneSeamFastBall", "TwoSeamFastBall", "Knuckleball", "Other"
    ]
    pitch_type_probs = [0.35, 0.18, 0.12, 0.15, 0.08, 0.06, 0.02, 0.015, 0.015, 0.005, 0.005]

    # Distribución de 1200 lanzamientos en 18 juegos
    # 12 juegos con 67 lanzamientos, 6 juegos con 66 lanzamientos = 1200
    base_n = n_rows // 18
    rem = n_rows % 18
    pitches_per_game = [base_n + (1 if g < rem else 0) for g in range(18)]

    rows: list[dict] = []
    pitch_counter = 0

    for g_idx, n_pitches in enumerate(pitches_per_game):
        gid = f"game_{g_idx + 1:04d}"
        season = game_seasons[gid]
        altitude = game_altitudes[gid]

        home_pitcher = pitchers[(g_idx * 2) % 12]
        away_pitcher = pitchers[(g_idx * 2 + 1) % 12]
        catcher = catchers[g_idx % 6]

        inning = 1
        half = "Top"
        outs = 0
        balls = 0
        strikes = 0
        batter_idx = 0

        for _ in range(n_pitches):
            pitch_counter += 1
            pitch_uid = str(uuid.UUID(int=1000000000 + pitch_counter))

            current_pitcher = home_pitcher if half == "Top" else away_pitcher
            current_batter = batters[batter_idx % len(batters)]
            p_throws = pitcher_hand[current_pitcher]
            b_side = batter_hands[current_batter]

            # Física realista condicionada por altitud:
            # Extreme Altitude tiene menor arrastre (menor ay0 y SpeedDrop) y menor quiebre Magnus (InducedVertBreak)
            if altitude == "Extreme Altitude":
                ay0 = float(rng.normal(21.5, 1.0))
                speed_drop = float(rng.normal(5.8, 0.3))
                ivb = float(rng.normal(7.5, 2.0))
                hb = float(rng.normal(3.2, 3.0))
            elif altitude == "Medium Altitude":
                ay0 = float(rng.normal(22.9, 1.0))
                speed_drop = float(rng.normal(6.3, 0.3))
                ivb = float(rng.normal(8.8, 2.0))
                hb = float(rng.normal(3.8, 3.0))
            else:  # No Altitude
                ay0 = float(rng.normal(25.1, 1.0))
                speed_drop = float(rng.normal(7.0, 0.3))
                ivb = float(rng.normal(10.5, 2.0))
                hb = float(rng.normal(5.5, 3.0))

            ptype = str(rng.choice(pitch_types, p=pitch_type_probs))
            is_fb = ptype in ("Four-Seam", "Sinker", "Cutter", "OneSeamFastBall", "TwoSeamFastBall")
            rel_speed = float(rng.normal(93.5, 2.0) if is_fb else rng.normal(84.0, 2.5))
            zone_speed = rel_speed - speed_drop
            zone_time = float(0.435 + (90.0 - rel_speed) * 0.0035)

            spin_rate = float(rng.normal(2280, 180))
            spin_axis = float(rng.uniform(30.0, 330.0))
            tilt = f"{int(rng.integers(1, 13)):02d}:{int(rng.choice([0, 15, 30, 45])):02d}"

            rel_height = float(rng.normal(5.77, 0.25))
            rel_side = float((1.6 if p_throws == "Right" else -1.6) + rng.normal(0, 0.2))
            extension = float(rng.normal(5.86, 0.3))

            vert_break = ivb - 36.5
            vert_rel_angle = float(rng.normal(-1.0, 0.6))
            horz_rel_angle = float(rng.normal(-1.25 if p_throws == "Right" else 1.25, 0.8))
            vert_appr_angle = float(rng.normal(-6.3, 0.7))
            horz_appr_angle = float(rng.normal(-0.74, 0.8))

            x0 = rel_side
            y0 = 50.0
            z0 = rel_height
            vy0 = float(-rel_speed * 1.467 * 0.985)
            vx0 = float(rng.normal(2.7 if p_throws == "Right" else -2.7, 1.0))
            vz0 = float(rng.normal(-3.3, 1.0))
            ax0 = float(rng.normal(-3.3 if p_throws == "Right" else 3.3, 2.0))
            az0 = float(-23.8 + (ivb - 8.0) * 0.6)
            pfxx = float(hb * 0.45)
            pfxz = float(ivb * 0.45)

            # Polinomios de trayectoria
            xc0 = 54.64 + float(rng.normal(0, 0.1))
            xc1 = vy0
            xc2 = ay0 / 2.0
            yc0 = rel_height
            yc1 = vz0
            yc2 = az0 / 2.0
            zc0 = rel_side
            zc1 = vx0
            zc2 = ax0 / 2.0

            # Ubicación en home plate
            plate_loc_h = float(rng.normal(2.43, 0.6))
            plate_loc_s = float(rng.normal(-0.02, 0.6))
            in_zone = 1 if (1.5 <= plate_loc_h <= 3.5 and abs(plate_loc_s) <= 0.83) else 0
            out_zone = float(1 - in_zone)

            # Dinámica de swing y resultados
            is_swing_prob = 0.65 if in_zone else 0.30
            is_swing = float(rng.random() < is_swing_prob)
            swung_out_zone = float(is_swing == 1.0 and out_zone == 1.0)

            if is_swing:
                is_whiff = float(rng.random() < 0.22)
                is_contact = float(1.0 - is_whiff)
                is_called_strike = 0.0
                is_swinging_strike = is_whiff
                if is_whiff:
                    pitch_call = "StrikeSwinging"
                    is_ball_in_play = 0.0
                else:
                    if rng.random() < 0.45:
                        pitch_call = "InPlay"
                        is_ball_in_play = 1.0
                    else:
                        pitch_call = "FoulBall"
                        is_ball_in_play = 0.0
            else:
                is_whiff = 0.0
                is_contact = 0.0
                is_swinging_strike = 0.0
                is_ball_in_play = 0.0
                if in_zone:
                    pitch_call = "StrikeCalled"
                    is_called_strike = 1.0
                else:
                    pitch_call = "BallCalled"
                    is_called_strike = 0.0

            is_batted = is_ball_in_play
            if is_ball_in_play:
                is_hit = float(rng.random() < 0.32)
                hit_type = str(rng.choice(["GroundBall", "LineDrive", "FlyBall", "Popup"], p=[0.45, 0.25, 0.20, 0.10]))
                exit_speed = float(rng.normal(88.0, 10.0))
                angle = float(rng.normal(12.0, 15.0))
                direction = float(rng.normal(0.0, 25.0))
                distance = float(max(10.0, exit_speed * 2.5 + angle * 3.0))
                if is_hit:
                    hit_roll = rng.random()
                    single = float(hit_roll < 0.65)
                    double = float(0.65 <= hit_roll < 0.85)
                    triple = float(0.85 <= hit_roll < 0.90)
                    home_run = float(hit_roll >= 0.90)
                    play_result = "Single" if single else ("Double" if double else ("Triple" if triple else "HomeRun"))
                    runs_scored = float(home_run)
                    outs_on_play = 0.0
                else:
                    single = double = triple = home_run = 0.0
                    play_result = "Out"
                    runs_scored = 0.0
                    outs_on_play = 1.0
            else:
                is_hit = single = double = triple = home_run = 0.0
                hit_type = None
                exit_speed = None
                angle = None
                direction = None
                distance = None
                runs_scored = 0.0
                outs_on_play = 0.0
                play_result = pitch_call

            # Estado previo al lanzamiento (Outs, Balls, Strikes y count)
            current_outs = float(min(outs, 2))
            current_balls = float(balls)
            current_strikes = float(strikes)
            count_str = f"{int(current_balls)}-{int(current_strikes)}"

            terminal_pa = False
            base_on_balls = 0.0
            strikeout = 0.0
            is_hit_by_pitch = 0.0

            if is_ball_in_play:
                terminal_pa = True
                if outs_on_play > 0:
                    outs += 1
            elif pitch_call in ("StrikeSwinging", "StrikeCalled"):
                if strikes == 2:
                    strikeout = 1.0
                    terminal_pa = True
                    outs += 1
                else:
                    strikes += 1
            elif pitch_call == "FoulBall":
                if strikes < 2:
                    strikes += 1
            elif pitch_call == "BallCalled":
                if balls == 3:
                    base_on_balls = 1.0
                    terminal_pa = True
                else:
                    balls += 1

            if strikeout:
                korbb = "Strikeout"
            elif base_on_balls:
                korbb = "Walk"
            else:
                korbb = "Undefined"

            row = {
                "year": np.int32(season),
                "PitchUID": pitch_uid,
                "game_anon_id": gid,
                "pitcher_anon_id": current_pitcher,
                "batter_anon_id": current_batter,
                "catcher_anon_id": catcher,
                "PitcherThrows": p_throws,
                "BatterSide": b_side,
                "altitude_category": altitude,
                "AutoPitchType": ptype,
                "RelSpeed": rel_speed,
                "EffectiveVelo": f"{rel_speed + float(rng.normal(0, 0.3)):.5f}",
                "ZoneSpeed": zone_speed,
                "SpinRate": spin_rate,
                "SpinAxis": spin_axis,
                "Tilt": tilt,
                "RelHeight": rel_height,
                "RelSide": rel_side,
                "Extension": extension,
                "VertBreak": vert_break,
                "InducedVertBreak": ivb,
                "HorzBreak": hb,
                "VertRelAngle": vert_rel_angle,
                "HorzRelAngle": horz_rel_angle,
                "VertApprAngle": vert_appr_angle,
                "HorzApprAngle": horz_appr_angle,
                "SpeedDrop": speed_drop,
                "ZoneTime": zone_time,
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "vx0": vx0,
                "vy0": vy0,
                "vz0": vz0,
                "ax0": ax0,
                "ay0": ay0,
                "az0": az0,
                "pfxx": pfxx,
                "pfxz": pfxz,
                "PitchTrajectoryXc0": xc0,
                "PitchTrajectoryXc1": xc1,
                "PitchTrajectoryXc2": xc2,
                "PitchTrajectoryYc0": yc0,
                "PitchTrajectoryYc1": yc1,
                "PitchTrajectoryYc2": yc2,
                "PitchTrajectoryZc0": zc0,
                "PitchTrajectoryZc1": zc1,
                "PitchTrajectoryZc2": zc2,
                "PitchCall": pitch_call,
                "KorBB": korbb,
                "play_result": play_result,
                "hit_type": hit_type,
                "ExitSpeed": exit_speed,
                "Angle": angle,
                "Direction": direction,
                "Distance": distance,
                "is_swing": is_swing,
                "is_whiff": is_whiff,
                "is_contact": is_contact,
                "is_called_strike": is_called_strike,
                "is_swinging_strike": is_swinging_strike,
                "is_ball_in_play": is_ball_in_play,
                "is_batted": is_batted,
                "is_hit": is_hit,
                "single": single,
                "double": double,
                "triple": triple,
                "home_run": home_run,
                "base_on_balls": base_on_balls,
                "strikeout": strikeout,
                "is_hit_by_pitch": is_hit_by_pitch,
                "RunsScored": runs_scored,
                "OutsOnPlay": outs_on_play,
                "Inning": float(inning),
                "Top/Bottom": half,
                "Outs": current_outs,
                "Balls": current_balls,
                "Strikes": current_strikes,
                "count": count_str,
                "PlateLocHeight": plate_loc_h,
                "PlateLocSide": plate_loc_s,
                "in_strike_zone": np.int32(in_zone),
                "outside_strike_zone": out_zone,
                "swung_outside_strike_zone": swung_out_zone,
            }
            rows.append(row)

            if terminal_pa:
                balls = 0
                strikes = 0
                batter_idx += 1
                if outs >= 3:
                    outs = 0
                    if half == "Top":
                        half = "Bottom"
                    else:
                        half = "Top"
                        inning += 1

    df = pd.DataFrame(rows)

    # Orden cronológico estricto por aparición al plato:
    # game_anon_id, Inning, Top/Bottom (Top antes de Bottom), batter_anon_id, Balls, Strikes
    df["Top/Bottom"] = pd.Categorical(df["Top/Bottom"], categories=["Top", "Bottom"], ordered=True)
    df = df.sort_values(
        by=["game_anon_id", "Inning", "Top/Bottom", "batter_anon_id", "Balls", "Strikes"]
    ).reset_index(drop=True)

    # Exactamente 20 filas con altitude_category = None distribuidas en juegos que sí tienen filas no nulas
    null_alt_indices = [
        35, 85, 145, 205, 265, 325, 385, 445, 505, 565,
        625, 685, 745, 805, 865, 925, 985, 1045, 1105, 1165
    ]
    df.loc[null_alt_indices, "altitude_category"] = None

    # Valores de EffectiveVelo como object (strings numéricos + algunos 'None' para probar HACK-002)
    none_velo_indices = [15, 80, 160, 240, 320, 410, 520, 630, 710, 830, 940, 1020, 1110, 1180]
    df.loc[none_velo_indices, "EffectiveVelo"] = "None"
    df["EffectiveVelo"] = df["EffectiveVelo"].astype("object")

    # Tipos categóricos idénticos al parquet real
    df["AutoPitchType"] = df["AutoPitchType"].astype("category")
    df["PitchCall"] = df["PitchCall"].astype("category")
    df["play_result"] = df["play_result"].astype("category")
    df["Top/Bottom"] = df["Top/Bottom"].astype("category")

    # Asegurar orden exacto de las 84 columnas oficiales
    return df[OFFICIAL_84_COLUMNS]


@pytest.fixture(scope="session")
def synthetic_lmb_df() -> pd.DataFrame:
    """
    Fixture de sesión que entrega el dataset sintético determinista de 1,200 filas
    con las 84 columnas oficiales sin requerir el archivo de datos real.
    """
    return make_synthetic_lmb_df(n_rows=1200, seed=42)

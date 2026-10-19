"""
step1_feature_engineering.py
Pipeline de limpieza, imputación inteligente, ingeniería de variables físicas y de arsenal (3D),
y construcción de los 8 targets multi-objetivo y Run Value empírico para LMB.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Asegurar importación de config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, DECISION_POINT_TIME,
    TUNNEL_DECAY_SCALE, PITCH_TYPE_MAP, FASTBALL_TYPES, TARGET_COLS
)

def run_feature_engineering():
    if not RAW_DATA_PATH or not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {RAW_DATA_PATH}")

    print(f"1. Cargando dataset desde: {RAW_DATA_PATH} ...")
    df = pd.read_parquet(RAW_DATA_PATH)
    print(f"   Shape original: {df.shape}")

    # ==========================================
    # A. LIMPIEZA E IMPUTACIÓN INTELIGENTE
    # ==========================================
    print("2. Limpiando tipos de datos e imputando categorías...")
    df['EffectiveVelo'] = pd.to_numeric(df['EffectiveVelo'], errors='coerce')

    # Unificar categorías raras de AutoPitchType
    df['AutoPitchType'] = df['AutoPitchType'].astype(str).replace(PITCH_TYPE_MAP)

    # Imputar altitude_category usando el mismo game_anon_id si existe
    game_alt_map = (
        df.dropna(subset=['altitude_category'])
        .loc[lambda x: ~x['altitude_category'].isin(['None', 'Undefined'])]
        .groupby('game_anon_id')['altitude_category']
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    )
    mask_null_alt = df['altitude_category'].isna() | df['altitude_category'].isin(['None', 'Undefined'])
    df.loc[mask_null_alt, 'altitude_category'] = df.loc[mask_null_alt, 'game_anon_id'].map(game_alt_map)
    df['altitude_category'] = df['altitude_category'].fillna('No Altitude')

    # Imputar PitcherThrows con la moda por pitcher o por el signo de RelSide
    df['PitcherThrows'] = df['PitcherThrows'].replace({'None': np.nan, 'Undefined': np.nan})
    pitcher_hand_map = df.dropna(subset=['PitcherThrows']).groupby('pitcher_anon_id')['PitcherThrows'].first()
    df['PitcherThrows'] = df['PitcherThrows'].fillna(df['pitcher_anon_id'].map(pitcher_hand_map))
    df['PitcherThrows'] = df['PitcherThrows'].fillna(np.where(df['RelSide'] >= 0, 'Right', 'Left'))

    df['BatterSide'] = df['BatterSide'].replace({'None': 'Right', 'Undefined': 'Right'})
    df['is_same_hand'] = (df['PitcherThrows'] == df['BatterSide']).astype(int)

    # ==========================================
    # B. INGENIERÍA FÍSICA (HIPÓTESIS H1 y H4)
    # ==========================================
    print("3. Creando variables físicas avanzadas (Magnus, Spin Efficiency, Arrastre)...")
    # Normalizar HorzBreak respecto a la mano (+ = Arm-Side Run, - = Glove-Side Cut)
    hand_sign = np.where(df['PitcherThrows'] == 'Left', -1.0, 1.0)
    df['HorzBreak_norm'] = df['HorzBreak'] * hand_sign
    df['RelSide_norm'] = df['RelSide'] * hand_sign
    df['HorzRelAngle_norm'] = df['HorzRelAngle'] * hand_sign
    df['HorzApprAngle_norm'] = df['HorzApprAngle'] * hand_sign
    df['PlateLocSide_norm'] = df['PlateLocSide'] * hand_sign

    # Movimiento Magnus total inducido
    df['TotalInducedBreak'] = np.sqrt(df['InducedVertBreak']**2 + df['HorzBreak']**2)
    df['BreakAngle'] = np.degrees(np.arctan2(df['HorzBreak_norm'], df['InducedVertBreak']))

    # Proxy físico de Spin Efficiency (H4)
    df['SpinToBreakRatio'] = df['TotalInducedBreak'] / (
        (df['SpinRate'] / 1000.0) * df['ZoneTime'].clip(lower=0.35) + 1e-5
    )

    # Proxy aerodinámico de arrastre
    df['DragProxy'] = df['SpeedDrop'] / df['RelSpeed'].clip(lower=60.0)

    # SpinAxis cíclico normalizado por mano
    spin_rad = np.radians(df['SpinAxis'])
    df['SpinAxis_sin'] = np.sin(spin_rad) * hand_sign
    df['SpinAxis_cos'] = np.cos(spin_rad)

    # ==========================================
    # C. POSICIÓN EN EL PUNTO DE DECISIÓN (TÚNEL 3D)
    # ==========================================
    print(f"4. Evaluando polinomios de trayectoria en el Decision Point (t = {DECISION_POINT_TIME}s)...")
    t_dec = DECISION_POINT_TIME
    df['Tunnel_X'] = df['PitchTrajectoryXc0'] + df['PitchTrajectoryXc1']*t_dec + df['PitchTrajectoryXc2']*(t_dec**2)
    df['Tunnel_Y'] = df['PitchTrajectoryYc0'] + df['PitchTrajectoryYc1']*t_dec + df['PitchTrajectoryYc2']*(t_dec**2)
    df['Tunnel_Z'] = df['PitchTrajectoryZc0'] + df['PitchTrajectoryZc1']*t_dec + df['PitchTrajectoryZc2']*(t_dec**2)

    # ==========================================
    # D. FEATURES RELACIONALES DE ARSENAL (25% EVALUACIÓN)
    # ==========================================
    print("5. Calculando repertorio y diferenciales contra la Recta Primaria...")
    df_fb = df[df['AutoPitchType'].isin(FASTBALL_TYPES) & df['pitcher_anon_id'].notna()]

    fb_counts = (
        df_fb.groupby(['pitcher_anon_id', 'year', 'AutoPitchType'])
        .size()
        .reset_index(name='n_pitches')
        .sort_values(['pitcher_anon_id', 'year', 'n_pitches'], ascending=[True, True, False])
    )
    primary_fb_type = (
        fb_counts.drop_duplicates(['pitcher_anon_id', 'year'])[['pitcher_anon_id', 'year', 'AutoPitchType']]
        .rename(columns={'AutoPitchType': 'primary_fb_type'})
    )

    df_fb_primary = df_fb.merge(primary_fb_type, on=['pitcher_anon_id', 'year'])
    df_fb_primary = df_fb_primary[df_fb_primary['AutoPitchType'] == df_fb_primary['primary_fb_type']]

    fb_stats = df_fb_primary.groupby(['pitcher_anon_id', 'year']).agg(
        fb_velo=('RelSpeed', 'mean'),
        fb_ivb=('InducedVertBreak', 'mean'),
        fb_hb_norm=('HorzBreak_norm', 'mean'),
        fb_rel_height=('RelHeight', 'mean'),
        fb_rel_side_norm=('RelSide_norm', 'mean'),
        fb_ext=('Extension', 'mean'),
        fb_tunnel_x=('Tunnel_X', 'mean'),
        fb_tunnel_y=('Tunnel_Y', 'mean'),
        fb_tunnel_z=('Tunnel_Z', 'mean'),
    ).reset_index()

    df = df.merge(fb_stats, on=['pitcher_anon_id', 'year'], how='left')

    # Rellenar con medianas para pitchers sin rectas registradas
    for col, base_col in [
        ('fb_velo', 'RelSpeed'), ('fb_ivb', 'InducedVertBreak'), ('fb_hb_norm', 'HorzBreak_norm'),
        ('fb_rel_height', 'RelHeight'), ('fb_rel_side_norm', 'RelSide_norm'), ('fb_ext', 'Extension'),
        ('fb_tunnel_x', 'Tunnel_X'), ('fb_tunnel_y', 'Tunnel_Y'), ('fb_tunnel_z', 'Tunnel_Z')
    ]:
        df[col] = df[col].fillna(df[base_col].median())

    df['velo_diff_vs_fb'] = df['RelSpeed'] - df['fb_velo']
    df['ivb_diff_vs_fb'] = df['InducedVertBreak'] - df['fb_ivb']
    df['hb_diff_vs_fb'] = df['HorzBreak_norm'] - df['fb_hb_norm']

    df['release_dist_vs_fb'] = np.sqrt(
        (df['RelHeight'] - df['fb_rel_height'])**2 +
        (df['RelSide_norm'] - df['fb_rel_side_norm'])**2 +
        (df['Extension'] - df['fb_ext'])**2
    ) * 12.0

    df['tunnel_dist_170ms_in'] = np.sqrt(
        (df['Tunnel_X'] - df['fb_tunnel_x'])**2 +
        (df['Tunnel_Y'] - df['fb_tunnel_y'])**2 +
        (df['Tunnel_Z'] - df['fb_tunnel_z'])**2
    ) * 12.0

    df['tunnel_similarity'] = np.exp(-df['tunnel_dist_170ms_in'] / TUNNEL_DECAY_SCALE)

    # Frecuencia de uso del pitcheo en el repertorio
    pitcher_total = df.groupby(['pitcher_anon_id', 'year']).size().rename('total_pitches_pitcher')
    pitcher_type_cnt = df.groupby(['pitcher_anon_id', 'year', 'AutoPitchType']).size().rename('type_pitches_pitcher')
    df = df.merge(pitcher_total, on=['pitcher_anon_id', 'year'], how='left')
    df = df.merge(pitcher_type_cnt, on=['pitcher_anon_id', 'year', 'AutoPitchType'], how='left')
    df['pitch_usage_pct'] = (df['type_pitches_pitcher'] / df['total_pitches_pitcher'].clip(lower=1)).fillna(0.2)

    # ==========================================
    # E. UBICACIÓN Y CONTEO (LOCATION+ Y PITCHING+)
    # ==========================================
    print("6. Creando variables de zona de strike y comando...")
    df['dist_from_zone_center'] = np.sqrt(df['PlateLocSide']**2 + (df['PlateLocHeight'] - 2.5)**2)
    dx_edge = np.abs(df['PlateLocSide']) - 0.833
    dz_edge = np.maximum(1.5 - df['PlateLocHeight'], df['PlateLocHeight'] - 3.5)
    df['dist_to_zone_edge'] = np.sqrt(np.maximum(0, dx_edge)**2 + np.maximum(0, dz_edge)**2)
    df['is_shadow_zone'] = (
        ((np.abs(dx_edge) <= 0.25) & (df['PlateLocHeight'].between(1.25, 3.75))) |
        ((np.abs(dz_edge) <= 0.25) & (df['PlateLocSide'].between(-1.08, 1.08)))
    ).astype(int)

    cond_in = df['in_strike_zone'] == 1
    col_idx = np.clip(np.floor((df['PlateLocSide'] + 0.833) / (1.666 / 3)), 0, 2).astype(int)
    row_idx = np.clip(np.floor((df['PlateLocHeight'] - 1.5) / (2.0 / 3)), 0, 2).astype(int)
    quad_in = row_idx * 3 + col_idx + 1
    quad_out = np.where(
        df['PlateLocHeight'] >= 2.5,
        np.where(df['PlateLocSide'] >= 0, 11, 10),
        np.where(df['PlateLocSide'] >= 0, 13, 12)
    )
    df['zone_quadrant'] = np.where(cond_in, quad_in, quad_out)

    # ==========================================
    # F. TARGETS MULTI-OBJETIVO Y LINEAR WEIGHTS
    # ==========================================
    print("7. Construyendo los 8 targets multi-objetivo y Run Value empírico...")
    df['target_whiff'] = df['is_whiff'].astype(float)
    df['target_chase'] = df['swung_outside_strike_zone'].astype(float)
    df['target_called_strike'] = df['is_called_strike'].astype(float)
    df['target_groundball'] = (df['hit_type'] == 'GroundBall').astype(float)
    df['target_weak_contact'] = (
        (df['is_ball_in_play'] == 1) &
        ((df['ExitSpeed'] < 85.0) | (df['hit_type'] == 'Popup'))
    ).astype(float)
    df['target_hard_contact'] = (
        (df['is_ball_in_play'] == 1) &
        ((df['ExitSpeed'] >= 95.0) | (df['home_run'] == 1))
    ).astype(float)
    df['target_out_on_pitch'] = ((df['OutsOnPlay'] > 0) | (df['strikeout'] == 1)).astype(float)

    # Run Value desde la perspectiva del lanzador (+ = previene carreras)
    pitcher_rv = np.zeros(len(df), dtype=np.float64)
    pitcher_rv = np.where(df['home_run'] == 1, -1.40, pitcher_rv)
    pitcher_rv = np.where(df['triple'] == 1, -1.05, pitcher_rv)
    pitcher_rv = np.where(df['double'] == 1, -0.78, pitcher_rv)
    pitcher_rv = np.where(df['single'] == 1, -0.47, pitcher_rv)
    pitcher_rv = np.where((df['base_on_balls'] == 1) | (df['is_hit_by_pitch'] == 1), -0.33, pitcher_rv)
    pitcher_rv = np.where(df['strikeout'] == 1, +0.28, pitcher_rv)
    pitcher_rv = np.where((df['OutsOnPlay'] > 0) & (df['strikeout'] == 0), +0.26, pitcher_rv)

    is_non_term = (
        (df['home_run'] == 0) & (df['triple'] == 0) & (df['double'] == 0) &
        (df['single'] == 0) & (df['base_on_balls'] == 0) & (df['is_hit_by_pitch'] == 0) &
        (df['strikeout'] == 0) & (df['OutsOnPlay'] == 0)
    )
    is_strike_ev = (df['is_called_strike'] == 1) | (df['is_swinging_strike'] == 1)
    is_foul_ev = df['PitchCall'].isin(['FoulBall', 'FoulBallNotFieldable', 'FoulBallFieldable'])
    is_ball_ev = df['PitchCall'].isin(['BallCalled', 'BallinDirt'])

    pitcher_rv = np.where(is_non_term & is_strike_ev, +0.065, pitcher_rv)
    pitcher_rv = np.where(is_non_term & is_foul_ev & (df['Strikes'] < 2), +0.055, pitcher_rv)
    pitcher_rv = np.where(is_non_term & is_foul_ev & (df['Strikes'] == 2), +0.005, pitcher_rv)

    ball_cost = np.where(
        df['Balls'] == 0, -0.035,
        np.where(df['Balls'] == 1, -0.055, -0.095)
    )
    pitcher_rv = np.where(is_non_term & is_ball_ev, ball_cost, pitcher_rv)
    df['pitcher_run_value'] = pitcher_rv

    # Filtrar anomalías extremas de radar
    df_clean = df[
        df['RelSpeed'].between(65.0, 105.0) &
        df['SpinRate'].between(500.0, 3600.0)
    ].copy()

    print(f"8. Guardando {len(df_clean):,} registros procesados en: {PROCESSED_DATA_PATH} ...")
    df_clean.to_parquet(PROCESSED_DATA_PATH, index=False)
    print("¡step1_feature_engineering completado con éxito!")

if __name__ == "__main__":
    run_feature_engineering()

"""
validate_hypotheses.py
Script para validar empíricamente con datos reales de LMB las 4 hipótesis científicas (H1-H4)
establecidas en la literatura del reto de Diablos Rojos.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DATA_PATH

def validate_hypotheses():
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: {PROCESSED_DATA_PATH} no existe. Corre primero step1_feature_engineering.py")
        return

    print("Cargando dataset procesado...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Total pitcheos analizados: {len(df):,}")

    print("\n" + "="*70)
    print("=== H1 y H2: MOVIMIENTO Y RUN VALUE POR TIPO DE PITCH Y ALTITUD ===")
    print("="*70)
    h1_table = df.groupby(['AutoPitchType', 'altitude_category']).agg(
        n_pitches=('PitchUID', 'count'),
        velo_mean=('RelSpeed', 'mean'),
        ivb_mean=('InducedVertBreak', 'mean'),
        hb_norm_mean=('HorzBreak_norm', 'mean'),
        total_break=('TotalInducedBreak', 'mean'),
        spin_rate=('SpinRate', 'mean'),
        whiff_rate=('target_whiff', 'mean'),
        hr_rate=('home_run', 'mean'),
        gb_rate=('target_groundball', 'mean'),
        run_val_100=('pitcher_run_value', lambda x: x.mean() * 100)  # Carreras prevenidas por cada 100 pitches
    ).reset_index()

    main_pitches = ['Four-Seam', 'Slider', 'Sinker', 'Changeup', 'Curveball', 'Cutter', 'Splitter']
    print(h1_table[h1_table['AutoPitchType'].isin(main_pitches)].to_string(index=False))

    print("\n" + "="*70)
    print("=== H3: PREVENCIÓN DE CARRERAS DE GROUNDBALLS EN ALTITUD ===")
    print("="*70)
    h3_table = df.groupby(['altitude_category', 'target_groundball']).agg(
        n_pitches=('PitchUID', 'count'),
        rv_mean=('pitcher_run_value', 'mean'),
        hr_rate=('home_run', 'mean'),
        hard_contact_rate=('target_hard_contact', 'mean')
    ).reset_index()
    print(h3_table.to_string(index=False))

    print("\n" + "="*70)
    print("=== H4: SPIN EFFICIENCY (SpinToBreakRatio) POR ALTITUD ===")
    print("="*70)
    h4_table = df.groupby(['AutoPitchType', 'altitude_category']).agg(
        spin_rate=('SpinRate', 'mean'),
        spin_to_break=('SpinToBreakRatio', 'mean'),
        drag=('DragProxy', 'mean')
    ).reset_index()
    print(h4_table[h4_table['AutoPitchType'].isin(main_pitches)].to_string(index=False))

if __name__ == "__main__":
    validate_hypotheses()

"""
Configuración global del proyecto Hackathon ISAC 2026 - Diablos Rojos
Centraliza rutas de archivos, columnas por rol, constantes físicas y ponderaciones de Linear Weights.
"""
from pathlib import Path
import os

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Búsqueda inteligente de rutas del dataset
POSSIBLE_RAW_PATHS = [
    BASE_DIR / "data" / "stuff_model_df.parquet",
    Path(r"C:\Users\L13Yoga\Downloads\Diablos - Hackathon 2026\stuff_model_df.parquet"),
    Path(os.path.expanduser(r"~\Downloads\Diablos - Hackathon 2026\stuff_model_df.parquet")),
]

POSSIBLE_DICT_PATHS = [
    BASE_DIR / "data" / "stuff_plus_data_dictionary.csv",
    Path(r"C:\Users\L13Yoga\Downloads\Diablos - Hackathon 2026\stuff_plus_data_dictionary.csv"),
    Path(os.path.expanduser(r"~\Downloads\Diablos - Hackathon 2026\stuff_plus_data_dictionary.csv")),
]

RAW_DATA_PATH = None
for p in POSSIBLE_RAW_PATHS:
    if p.exists():
        RAW_DATA_PATH = p
        break

DATA_DICT_PATH = None
for p in POSSIBLE_DICT_PATHS:
    if p.exists():
        DATA_DICT_PATH = p
        break

# Directorios de salida
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PROCESSED_DATA_PATH = DATA_DIR / "stuff_processed_df.parquet"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Parámetros físicos
DECISION_POINT_TIME = 0.17  # segundos tras release (~23.8 ft de home plate)
TUNNEL_DECAY_SCALE = 6.0    # escala de decaimiento en pulgadas

# Mapeo de tipos de pitcheo
PITCH_TYPE_MAP = {
    'OneSeamFastBall': 'Sinker',
    'TwoSeamFastBall': 'Sinker',
    'Knuckleball': 'Other',
}

FASTBALL_TYPES = ['Four-Seam', 'Sinker', 'Cutter']

# Targets multi-objetivo
TARGET_COLS = [
    'target_whiff',
    'target_chase',
    'target_called_strike',
    'target_groundball',
    'target_weak_contact',
    'target_hard_contact',
    'target_out_on_pitch',
    'pitcher_run_value',
]

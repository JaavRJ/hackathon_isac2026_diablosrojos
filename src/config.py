"""
Configuración global del proyecto Hackathon ISAC 2026 - Diablos Rojos.
Centraliza rutas de archivos, contratos de columnas por rol, constantes físicas y targets.
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
if RAW_DATA_PATH is None:
    RAW_DATA_PATH = Path(r"C:\Users\L13Yoga\Downloads\Diablos - Hackathon 2026\stuff_model_df.parquet")

DATA_DICT_PATH = None
for p in POSSIBLE_DICT_PATHS:
    if p.exists():
        DATA_DICT_PATH = p
        break
if DATA_DICT_PATH is None:
    DATA_DICT_PATH = Path(r"C:\Users\L13Yoga\Downloads\Diablos - Hackathon 2026\stuff_plus_data_dictionary.csv")

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

# ==============================================================================
# Contrato de Columnas por Rol (docs/CONTRACT.md §1)
# ==============================================================================

# 1.1 Columnas Prohibidas como Features en CUALQUIER Modelo (docs/CONTRACT.md §1.1)
FORBIDDEN_TARGET_COLS = [
    "PitchCall", "KorBB", "play_result", "hit_type",
    "ExitSpeed", "Angle", "Direction", "Distance",
    "is_swing", "is_whiff", "is_contact", "is_called_strike",
    "is_swinging_strike", "is_ball_in_play", "is_batted", "is_hit",
    "single", "double", "triple", "home_run",
    "base_on_balls", "strikeout", "is_hit_by_pitch",
    "RunsScored", "OutsOnPlay", "swung_outside_strike_zone",
    # Y todos los targets derivados en HACK-006:
    "target_whiff", "target_chase", "target_called_strike",
    "target_groundball", "target_weak_contact", "target_hard_contact",
    "target_out_on_pitch", "xrv_pitcher",
]

# 1.2 Columnas Prohibidas como Features Directas (docs/CONTRACT.md §1.2)
ID_AND_GROUP_COLS = [
    "PitchUID", "game_anon_id", "pitcher_anon_id",
    "batter_anon_id", "catcher_anon_id", "year",
]

# 1.3 Columnas Prohibidas en Stuff+ Puro (docs/CONTRACT.md §1.3)
LOCATION_AND_COUNT_COLS = [
    "PlateLocHeight", "PlateLocSide", "PlateLocSide_norm",
    "in_strike_zone", "outside_strike_zone",
    "dist_from_zone_center", "dist_to_zone_edge", "is_shadow_zone", "zone_quadrant",
    "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "count",
    "BatterSide", "is_same_hand", "prev_pitch_id",
]

# Targets multi-objetivo oficiales (docs/CONTRACT.md §2)
TARGET_COLS = [
    "target_whiff",
    "target_chase",
    "target_called_strike",
    "target_groundball",
    "target_weak_contact",
    "target_hard_contact",
    "target_out_on_pitch",
    "xrv_pitcher",
]

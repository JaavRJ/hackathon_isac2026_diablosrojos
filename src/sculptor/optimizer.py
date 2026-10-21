"""Buscador de Referencias Factibles (Pitch Sculptor), HACK-016.

Alcance honesto: es una BUSQUEDA de referencias reales en el catalogo
``reports/leaderboards/pitcher_arsenal_breakdown.csv`` (calculado en HACK-014).
No inventa pitcheos, no reentrena y no reevalua perfiles sinteticos: toda cifra
devuelta es una fila (o un percentil/mediana de filas) del catalogo que pasa el
filtro de factibilidad. No importa ``StuffPlusModel`` ni lee ``models/*.joblib``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COACHING_LABEL = "[Guía cualitativa de coaching — literatura biomecánica, no estimada del dataset]"
DELTA_DISCLAIMER = "diferencia observada frente a pares reales; no es una ganancia causal garantizada"

REQUIRED_COLUMNS = [
    "pitcher_anon_id", "year", "pitch_id", "n_pitches",
    "PitcherThrows", "arm_slot_deg", "RelHeight", "fb_velo",
    "stuff_plus_neutral", "stuff_plus_cdmx", "delta_stuff_cdmx",
    "ivb_neutral", "ivb_cdmx", "hb_norm_neutral", "hb_norm_cdmx",
]

ARM_SLOT_TOL_DEG = 6.0
REL_HEIGHT_TOL_FT = 0.35
FB_VELO_TOL_MPH = 2.0
MIN_PEERS = 5
MAX_COMPARABLES = 3

_COACHING_TEXT = {
    "Four-Seam": "Priorizar la intención de backspin limpio y una trayectoria de brazo consistente; el agarre no se infiere del dataset.",
    "Sinker": "Trabajar la presión de dedos para conservar el movimiento de brazo-lado; la costura no se mide en estos datos.",
    "Cutter": "Reducir el desplazamiento lateral de la muñeca para mantener un eje estable; es una pauta general de entrenamiento.",
    "Slider": "Mantener la muñeca firme al soltar y la misma ventana de release que la recta para conservar el túnel.",
    "Sweeper": "Buscar mayor tracción lateral con la misma mecánica de brazo que el resto del repertorio.",
    "Curveball": "Enfocarse en finalizar la pronación con dedos al frente de la pelota, sin cambiar el arm slot.",
    "Changeup": "Ajustar la presión del agarre para quitar velocidad conservando la velocidad de brazo.",
    "Splitter": "Abrir los dedos sobre la costura para reducir el giro manteniendo la mecánica de la recta.",
}
_DEFAULT_COACHING = "Orientación general: replicar la mecánica de los pares y validar con el cuerpo técnico."


class PitchSculptor:
    """Busca pares biomecánicos reales de la LMB y reporta qué formas de pitcheo
    mantienen mejor ``Stuff+`` en CDMX. Acepta una ruta CSV o un DataFrame."""

    def __init__(self, catalog: str | Path | pd.DataFrame):
        if isinstance(catalog, pd.DataFrame):
            df = catalog.copy()
        else:
            df = pd.read_csv(catalog)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "El catálogo de arsenal no tiene las columnas requeridas: "
                + ", ".join(missing)
                + ". Regenera pitcher_arsenal_breakdown.csv con HACK-014 corregido."
            )
        self.catalog = df.reset_index(drop=True)

    # ------------------------------------------------------------------ filtros
    def _feasible(self, params: dict, min_pitches: int) -> pd.DataFrame:
        df = self.catalog
        mask = (
            (df["PitcherThrows"] == params["PitcherThrows"])
            & ((df["arm_slot_deg"] - float(params["arm_slot_deg"])).abs() <= ARM_SLOT_TOL_DEG)
            & ((df["RelHeight"] - float(params["RelHeight"])).abs() <= REL_HEIGHT_TOL_FT)
            & ((df["fb_velo"] - float(params["fb_velo"])).abs() <= FB_VELO_TOL_MPH)
            & (df["n_pitches"] >= min_pitches)
        )
        out = df[mask].copy()
        out["_dist"] = (
            (out["arm_slot_deg"] - float(params["arm_slot_deg"])).abs() / ARM_SLOT_TOL_DEG
            + (out["RelHeight"] - float(params["RelHeight"])).abs() / REL_HEIGHT_TOL_FT
            + (out["fb_velo"] - float(params["fb_velo"])).abs() / FB_VELO_TOL_MPH
        )
        return out

    # ----------------------------------------------------------------- búsqueda
    def find_feasible_references(
        self,
        pitch_params: dict,
        k_neighbors: int = 25,
        min_pitches: int = 50,
        include_coaching_guide: bool = True,
    ) -> dict[str, Any]:
        """Devuelve referencias reales factibles.

        ``pitch_params``: ``PitcherThrows``, ``arm_slot_deg``, ``RelHeight``,
        ``fb_velo``, ``pitch_id`` y opcional ``user_stuff_plus_cdmx``.
        ``k_neighbors`` limita las filas de la misma familia (las más cercanas
        en geometría) usadas para el rango y las referencias.
        """
        needed = ["PitcherThrows", "arm_slot_deg", "RelHeight", "fb_velo", "pitch_id"]
        absent = [k for k in needed if pitch_params.get(k) is None]
        if absent:
            raise ValueError("Faltan parámetros del pitcheo: " + ", ".join(absent))

        pitch_id = pitch_params["pitch_id"]
        feasible = self._feasible(pitch_params, min_pitches)
        n_peers = int(feasible["pitcher_anon_id"].nunique())

        result: dict[str, Any] = {
            "status": "OK",
            "reason": None,
            "n_peers": n_peers,
            "pitch_id": pitch_id,
            "filters": {
                "arm_slot_tol_deg": ARM_SLOT_TOL_DEG,
                "rel_height_tol_ft": REL_HEIGHT_TOL_FT,
                "fb_velo_tol_mph": FB_VELO_TOL_MPH,
                "min_pitches": min_pitches,
            },
            "intra_family": None,
            "delta_gain_cdmx": None,
            "delta_gain_reason": None,
            "delta_gain_note": DELTA_DISCLAIMER,
            "repertoire": [],
            "comparables": [],
            "coaching_guide_label": COACHING_LABEL,
            "coaching_guide": None,
        }

        if n_peers < MIN_PEERS:
            result["status"] = "INSUFFICIENT_PEERS"
            result["reason"] = (
                f"Solo {n_peers} pares reales cumplen el filtro de factibilidad "
                f"(mínimo {MIN_PEERS}); los filtros no se relajan."
            )
            result["delta_gain_reason"] = "Sin pares suficientes: no se emite diferencia."
            return result

        fam = feasible[feasible["pitch_id"] == pitch_id].sort_values(["_dist", "pitcher_anon_id", "year"])
        fam = fam.head(int(k_neighbors))
        if fam.empty:
            result["reason"] = f"Ningún par factible ejecuta '{pitch_id}' con el mínimo de pitcheos."
            result["delta_gain_reason"] = result["reason"]
        else:
            s = fam["stuff_plus_cdmx"]
            best = fam.loc[s.idxmax()]
            result["intra_family"] = {
                "n_rows": int(len(fam)),
                "p25_stuff_plus_cdmx": float(np.percentile(s, 25)),
                "p75_stuff_plus_cdmx": float(np.percentile(s, 75)),
                "best_reference": {
                    "pitcher_anon_id": str(best["pitcher_anon_id"]),
                    "year": int(best["year"]),
                    "pitch_id": str(best["pitch_id"]),
                    "n_pitches": int(best["n_pitches"]),
                    "stuff_plus_neutral": float(best["stuff_plus_neutral"]),
                    "stuff_plus_cdmx": float(best["stuff_plus_cdmx"]),
                    "delta_stuff_cdmx": float(best["delta_stuff_cdmx"]),
                    "ivb_neutral": float(best["ivb_neutral"]),
                    "ivb_cdmx": float(best["ivb_cdmx"]),
                    "hb_norm_neutral": float(best["hb_norm_neutral"]),
                    "hb_norm_cdmx": float(best["hb_norm_cdmx"]),
                },
            }
            user = pitch_params.get("user_stuff_plus_cdmx")
            if user is None:
                result["delta_gain_reason"] = "El pitcheo del usuario no tiene Stuff+ CDMX; no se calcula la diferencia."
            else:
                result["delta_gain_cdmx"] = float(best["stuff_plus_cdmx"]) - float(user)

            top = fam.sort_values(["stuff_plus_cdmx", "pitcher_anon_id"], ascending=[False, True])
            top = top.drop_duplicates("pitcher_anon_id").head(MAX_COMPARABLES)
            result["comparables"] = [
                {
                    "pitcher_anon_id": str(r.pitcher_anon_id),
                    "year": int(r.year),
                    "pitch_id": str(r.pitch_id),
                    "n_pitches": int(r.n_pitches),
                    "stuff_plus_cdmx": float(r.stuff_plus_cdmx),
                }
                for r in top.itertuples()
            ]

        others = feasible[feasible["pitch_id"] != pitch_id]
        rep = []
        for pid, g in others.groupby("pitch_id"):
            rep.append({
                "pitch_id": str(pid),
                "median_stuff_plus_cdmx": float(g["stuff_plus_cdmx"].median()),
                "n_pitchers": int(g["pitcher_anon_id"].nunique()),
                "n_pitches": int(g["n_pitches"].sum()),
            })
        rep.sort(key=lambda d: (-d["median_stuff_plus_cdmx"], d["pitch_id"]))
        result["repertoire"] = rep

        if include_coaching_guide:
            result["coaching_guide"] = _COACHING_TEXT.get(pitch_id, _DEFAULT_COACHING)
        return result

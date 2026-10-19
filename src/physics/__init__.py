"""
src/physics
===========
Módulo de física de vuelo, aerodinámica y densidad del aire
para el modelo Stuff+ LMB (Reto 2 Diablos Rojos - Diablos AeroVision).
"""
from __future__ import annotations

from src.physics.density import attach_rho_hat, estimate_game_density

__all__ = ["estimate_game_density", "attach_rho_hat"]

"""
tests/test_targets.py
=====================
Suite de pruebas unitarias para src/targets.py (HACK-006).

Verifica:
1. Inmutabilidad: attach_targets no muta el DataFrame de entrada.
2. Cero data leakage: fit_lmb_run_values ignora 2026 y lanza ValueError si solo hay 2026 o vacío.
3. Estimación 100% empírica y ausencia de constantes fijas: al escalar RunsScored por k,
   los valores de carrera escalan exactamente por k.
4. Jerarquía física empírica y bandera del modelo:
   empirical_hierarchy_holds es True y
   xrv(strikeout) > xrv(called_strike) > xrv(ball) > xrv(single) > xrv(home_run).
5. Monotonía empírica de la matriz RE12 en outs, bolas y strikes.
6. Presencia, tipo float64 y ausencia de NaN en las 8 columnas target (docs/CONTRACT.md §2).
7. Coherencia lógica de las definiciones de los targets multi-objetivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLS
from src.features.physical import clean_and_impute
from src.targets import attach_targets, classify_events, fit_lmb_run_values


@pytest.fixture
def cleaned_synthetic_df(synthetic_lmb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixture auxiliar: synthetic_lmb_df preprocesado con clean_and_impute y enriquecido
    de forma determinista con dinámica realista de carreras (RunsScored) condicionada a
    bases alcanzadas, conteo y outs, permitiendo evaluar el estimador 100% empírico
    sin imponer constantes inventadas ni ajustes post-hoc en el código de producción.
    """
    cleaned = clean_and_impute(synthetic_lmb_df)
    out = cleaned.copy()
    runs = np.zeros(len(out), dtype=np.float64)

    for i in range(len(out)):
        row = out.iloc[i]
        o = float(row.get("Outs", 0))
        b = float(row.get("Balls", 0))
        s = float(row.get("Strikes", 0))

        r = 0.0
        if row.get("home_run") == 1:
            r = 2.0 + 0.4 * (2.0 - o) + 0.40 * b - 0.40 * s
        elif row.get("triple") == 1:
            r = 1.5 + 0.3 * (2.0 - o) + 0.35 * b - 0.35 * s
        elif row.get("double") == 1:
            r = 1.1 + 0.3 * (2.0 - o) + 0.30 * b - 0.30 * s
        elif row.get("single") == 1:
            r = 0.7 + 0.2 * (2.0 - o) + 0.25 * b - 0.25 * s
        elif row.get("base_on_balls") == 1 or row.get("is_hit_by_pitch") == 1 or row.get("KorBB") == "Walk":
            r = 0.5 + 0.15 * (2.0 - o) + 0.20 * b - 0.10 * s
        elif b >= 2 and row.get("is_ball_in_play") == 1 and row.get("OutsOnPlay", 0) == 0:
            r = 0.6 + 0.2 * b - 0.2 * s
        runs[i] = max(0.0, r)

    out["RunsScored"] = runs
    return out


class TestFitLmbRunValues:
    """Pruebas para fit_lmb_run_values."""

    def test_excludes_2026_and_uses_historical_data(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """Verifica que fit_lmb_run_values ignore el año 2026 y solo use <= 2025."""
        rv_table = fit_lmb_run_values(cleaned_synthetic_df)

        assert "train_years" in rv_table
        assert 2026 not in rv_table["train_years"]
        assert all(y <= 2025 for y in rv_table["train_years"])
        assert len(rv_table["train_years"]) > 0

    def test_raises_value_error_on_empty_or_only_2026(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """Verifica que lance ValueError si el DataFrame está vacío o solo contiene 2026."""
        # DataFrame vacío
        empty_df = pd.DataFrame(columns=cleaned_synthetic_df.columns)
        with pytest.raises(ValueError, match="vacío"):
            fit_lmb_run_values(empty_df)

        # DataFrame que solo contiene año 2026 (holdout oficial)
        only_2026_df = cleaned_synthetic_df[cleaned_synthetic_df["year"] == 2026].copy()
        with pytest.raises(ValueError, match="year <= 2025"):
            fit_lmb_run_values(only_2026_df)

    def test_proportional_scaling_to_runs_scored(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """
        Demuestra empíricamente que los valores de carrera NO están hardcodeados:
        al multiplicar RunsScored por 2.0, los valores en event_xrv escalan proporcionalmente por 2.0.
        """
        rv_base = fit_lmb_run_values(cleaned_synthetic_df)

        scaled_df = cleaned_synthetic_df.copy()
        scaled_df["RunsScored"] = scaled_df["RunsScored"] * 2.0
        rv_scaled = fit_lmb_run_values(scaled_df)

        ev_base = rv_base["event_xrv"]
        ev_scaled = rv_scaled["event_xrv"]

        for ev_key in ev_base:
            base_val = ev_base[ev_key]
            scaled_val = ev_scaled[ev_key]
            if abs(base_val) > 1e-6:
                ratio = scaled_val / base_val
                assert pytest.approx(ratio, rel=1e-3) == 2.0, (
                    f"El evento '{ev_key}' no escaló proporcionalmente a RunsScored: "
                    f"base={base_val}, scaled={scaled_val}, ratio={ratio}"
                )

    def test_physical_hierarchy_preserved_empirically(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """
        Verifica el cumplimiento empírico de la jerarquía física desde la perspectiva del pitcher
        sin forzar cifras fijas:
        xrv(strikeout) > xrv(called_strike) > xrv(ball) > xrv(single) > xrv(home_run).
        """
        rv_table = fit_lmb_run_values(cleaned_synthetic_df)
        ev = rv_table["event_xrv"]

        assert rv_table["empirical_hierarchy_holds"] is True, (
            f"La bandera 'empirical_hierarchy_holds' reportó False con valores: {ev}"
        )

        assert ev["strikeout"] > ev["called_strike"], (
            f"Falla: strikeout ({ev['strikeout']}) <= called_strike ({ev['called_strike']})"
        )
        assert ev["called_strike"] > ev["ball"], (
            f"Falla: called_strike ({ev['called_strike']}) <= ball ({ev['ball']})"
        )
        assert ev["ball"] > ev["single"], (
            f"Falla: ball ({ev['ball']}) <= single ({ev['single']})"
        )
        assert ev["single"] > ev["home_run"], (
            f"Falla: single ({ev['single']}) <= home_run ({ev['home_run']})"
        )

        # Validar además signos esperados
        assert ev["strikeout"] > 0.0, "Un ponche debe prevenir carreras (signo positivo)"
        assert ev["called_strike"] > 0.0, "Un strike cantado debe prevenir carreras (signo positivo)"
        assert ev["ball"] < 0.0, "Una bola debe costar carreras (signo negativo)"
        assert ev["single"] < 0.0, "Un sencillo debe costar carreras (signo negativo)"
        assert ev["home_run"] < 0.0, "Un jonrón debe costar carreras (signo negativo)"

    def test_re_matrix_empirical_monotonicity(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """
        Verifica que la matriz RE12 y sus marginales empíricos reflejen la dinámica de carreras:
        - Más outs -> menor expectativa para el bateador.
        - Más bolas -> mayor expectativa para el bateador.
        - Más strikes -> menor expectativa para el bateador.
        """
        rv_table = fit_lmb_run_values(cleaned_synthetic_df)
        re = rv_table["re_matrix"]

        # 1. Monotonía marginal en Outs
        re_outs = [float(np.mean([re[(o, b, s)] for b in range(4) for s in range(3)])) for o in range(3)]
        assert re_outs[0] > re_outs[1] > re_outs[2], (
            f"Violación marginal en Outs: {re_outs}"
        )

        # 2. Monotonía marginal en Bolas
        re_balls = [float(np.mean([re[(o, b, s)] for o in range(3) for s in range(3)])) for b in range(4)]
        assert re_balls[0] < re_balls[1] < re_balls[2] < re_balls[3], (
            f"Violación marginal en Bolas: {re_balls}"
        )

        # 3. Monotonía marginal en Strikes
        re_strikes = [float(np.mean([re[(o, b, s)] for o in range(3) for b in range(4)])) for s in range(3)]
        assert re_strikes[0] > re_strikes[1] > re_strikes[2], (
            f"Violación marginal en Strikes: {re_strikes}"
        )


class TestAttachTargets:
    """Pruebas para attach_targets."""

    def test_does_not_mutate_input_dataframe(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """Verifica que attach_targets no altere el DataFrame original."""
        df_copy = cleaned_synthetic_df.copy(deep=True)
        rv_table = fit_lmb_run_values(cleaned_synthetic_df)

        result = attach_targets(cleaned_synthetic_df, rv_table)

        assert result is not cleaned_synthetic_df
        pd.testing.assert_frame_equal(cleaned_synthetic_df, df_copy)

    def test_creates_exact_8_target_columns_float64_no_nan(self, cleaned_synthetic_df: pd.DataFrame) -> None:
        """Verifica que se generen las 8 columnas exactas de CONTRACT.md en float64 sin NaN."""
        rv_table = fit_lmb_run_values(cleaned_synthetic_df)
        result = attach_targets(cleaned_synthetic_df, rv_table)

        for col in TARGET_COLS:
            assert col in result.columns, f"Columna '{col}' falta en el DataFrame resultante."
            assert result[col].dtype == np.float64, f"Columna '{col}' no es float64 (es {result[col].dtype})."
            assert not result[col].isna().any(), f"Columna '{col}' contiene valores NaN."
            assert not np.isinf(result[col]).any(), f"Columna '{col}' contiene valores infinitos."

    def test_target_definitions_correctness(self) -> None:
        """Verifica la lógica individual de cada uno de los 8 targets en casos controlados."""
        sample_data = pd.DataFrame([
            {
                "year": 2024,
                "game_anon_id": "game_0001",
                "Inning": 1.0,
                "Top/Bottom": "Top",
                "Outs": 0.0,
                "Balls": 0.0,
                "Strikes": 0.0,
                "RunsScored": 0.0,
                "OutsOnPlay": 0.0,
                "is_whiff": 1.0,
                "swung_outside_strike_zone": 1.0,
                "is_called_strike": 0.0,
                "is_swinging_strike": 1.0,
                "hit_type": None,
                "ExitSpeed": np.nan,
                "is_ball_in_play": 0.0,
                "home_run": 0.0,
                "strikeout": 0.0,
                "PitchCall": "StrikeSwinging",
            },
            {
                "year": 2024,
                "game_anon_id": "game_0001",
                "Inning": 1.0,
                "Top/Bottom": "Top",
                "Outs": 0.0,
                "Balls": 0.0,
                "Strikes": 1.0,
                "RunsScored": 0.0,
                "OutsOnPlay": 0.0,
                "is_whiff": 0.0,
                "swung_outside_strike_zone": 0.0,
                "is_called_strike": 1.0,
                "is_swinging_strike": 0.0,
                "hit_type": None,
                "ExitSpeed": np.nan,
                "is_ball_in_play": 0.0,
                "home_run": 0.0,
                "strikeout": 0.0,
                "PitchCall": "StrikeCalled",
            },
            {
                "year": 2024,
                "game_anon_id": "game_0001",
                "Inning": 1.0,
                "Top/Bottom": "Top",
                "Outs": 0.0,
                "Balls": 0.0,
                "Strikes": 2.0,
                "RunsScored": 0.0,
                "OutsOnPlay": 1.0,
                "is_whiff": 0.0,
                "swung_outside_strike_zone": 0.0,
                "is_called_strike": 0.0,
                "is_swinging_strike": 0.0,
                "hit_type": "GroundBall",
                "ExitSpeed": 82.0,
                "is_ball_in_play": 1.0,
                "home_run": 0.0,
                "strikeout": 0.0,
                "PitchCall": "InPlay",
            },
            {
                "year": 2024,
                "game_anon_id": "game_0001",
                "Inning": 1.0,
                "Top/Bottom": "Top",
                "Outs": 1.0,
                "Balls": 0.0,
                "Strikes": 0.0,
                "RunsScored": 1.0,
                "OutsOnPlay": 0.0,
                "is_whiff": 0.0,
                "swung_outside_strike_zone": 0.0,
                "is_called_strike": 0.0,
                "is_swinging_strike": 0.0,
                "hit_type": "FlyBall",
                "ExitSpeed": 102.5,
                "is_ball_in_play": 1.0,
                "home_run": 1.0,
                "strikeout": 0.0,
                "PitchCall": "InPlay",
            },
            {
                "year": 2024,
                "game_anon_id": "game_0001",
                "Inning": 1.0,
                "Top/Bottom": "Top",
                "Outs": 1.0,
                "Balls": 0.0,
                "Strikes": 2.0,
                "RunsScored": 0.0,
                "OutsOnPlay": 1.0,
                "is_whiff": 1.0,
                "swung_outside_strike_zone": 0.0,
                "is_called_strike": 0.0,
                "is_swinging_strike": 1.0,
                "hit_type": None,
                "ExitSpeed": np.nan,
                "is_ball_in_play": 0.0,
                "home_run": 0.0,
                "strikeout": 1.0,
                "PitchCall": "StrikeSwinging",
                "KorBB": "Strikeout",
            },
        ])

        rv_table = fit_lmb_run_values(sample_data)
        out = attach_targets(sample_data, rv_table)

        # Fila 0: Whiff + Chase
        assert out.loc[0, "target_whiff"] == 1.0
        assert out.loc[0, "target_chase"] == 1.0
        assert out.loc[0, "target_called_strike"] == 0.0
        assert out.loc[0, "target_groundball"] == 0.0
        assert out.loc[0, "target_weak_contact"] == 0.0
        assert out.loc[0, "target_hard_contact"] == 0.0
        assert out.loc[0, "target_out_on_pitch"] == 0.0

        # Fila 1: Called strike
        assert out.loc[1, "target_called_strike"] == 1.0
        assert out.loc[1, "target_whiff"] == 0.0

        # Fila 2: GroundBall + Weak contact (<85 mph) + Out
        assert out.loc[2, "target_groundball"] == 1.0
        assert out.loc[2, "target_weak_contact"] == 1.0
        assert out.loc[2, "target_hard_contact"] == 0.0
        assert out.loc[2, "target_out_on_pitch"] == 1.0

        # Fila 3: HomeRun + Hard contact (>=95 mph)
        assert out.loc[3, "target_hard_contact"] == 1.0
        assert out.loc[3, "target_weak_contact"] == 0.0
        assert out.loc[3, "target_out_on_pitch"] == 0.0

        # Fila 4: Strikeout -> Out on pitch
        assert out.loc[4, "target_out_on_pitch"] == 1.0


class TestClassifyEvents:
    """Pruebas para classify_events."""

    def test_classifies_all_standard_event_types(self) -> None:
        """Verifica que classify_events asigne correctamente los tipos canónicos."""
        df = pd.DataFrame([
            {"home_run": 1, "triple": 0, "double": 0, "single": 0, "strikeout": 0, "OutsOnPlay": 0, "is_ball_in_play": 1},
            {"home_run": 0, "triple": 1, "double": 0, "single": 0, "strikeout": 0, "OutsOnPlay": 0, "is_ball_in_play": 1},
            {"home_run": 0, "triple": 0, "double": 1, "single": 0, "strikeout": 0, "OutsOnPlay": 0, "is_ball_in_play": 1},
            {"home_run": 0, "triple": 0, "double": 0, "single": 1, "strikeout": 0, "OutsOnPlay": 0, "is_ball_in_play": 1},
            {"strikeout": 1, "OutsOnPlay": 1, "PitchCall": "StrikeSwinging"},
            {"base_on_balls": 1, "PitchCall": "BallCalled"},
            {"is_ball_in_play": 1, "OutsOnPlay": 1, "play_result": "Out"},
            {"PitchCall": "FoulBall", "Strikes": 1},
            {"PitchCall": "FoulBall", "Strikes": 2},
            {"PitchCall": "StrikeSwinging", "is_swinging_strike": 1, "Strikes": 0},
            {"PitchCall": "StrikeCalled", "is_called_strike": 1, "Strikes": 0},
            {"PitchCall": "BallCalled", "Strikes": 0},
        ])

        events = classify_events(df)
        expected = [
            "home_run", "triple", "double", "single", "strikeout", "walk_or_hbp",
            "out_in_play", "foul_lt2", "foul_2s", "swinging_strike", "called_strike", "ball"
        ]
        assert list(events) == expected

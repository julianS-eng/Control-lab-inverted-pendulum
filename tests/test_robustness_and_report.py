from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pendulum_lab import cli
from pendulum_lab.design import balance_strategies
from pendulum_lab.experiments import (
    RunSettings,
    run_all,
    study_sampling,
    study_structure,
)
from pendulum_lab.params import CartPoleParams, LabConfig
from pendulum_lab.report import inject_readme, render_tables
from pendulum_lab.robustness import (
    MODERATE,
    BalanceScenario,
    monte_carlo,
    monte_carlo_swingup,
    roa_grid,
    sample_plants,
)


def test_sample_plants_respects_bounds() -> None:
    rng = np.random.default_rng(0)
    nom = CartPoleParams()
    p = sample_plants(nom, MODERATE, 2000, rng)
    assert p.batch_size == 2000
    assert np.all(np.abs(np.asarray(p.l) / nom.l - 1) <= 0.2 + 1e-12)
    ratio = np.asarray(p.b_c) / nom.b_c
    assert ratio.min() >= 0.5 - 1e-12
    assert ratio.max() <= 2.0 + 1e-12
    np.testing.assert_allclose(p.J, np.asarray(p.m) * (2 * np.asarray(p.l)) ** 2 / 12)


def test_monte_carlo_is_deterministic_and_all_succeed_nominally(config: LabConfig) -> None:
    strategies = balance_strategies(config)[:2]
    a = monte_carlo(strategies, config, MODERATE, 12, seed=5)
    b = monte_carlo(strategies, config, MODERATE, 12, seed=5)
    for ra, rb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ra.success, rb.success)
        assert ra.rate == 1.0
        lo, hi = ra.ci95
        assert lo < 1.0 <= hi + 1e-12


@pytest.mark.slow
def test_swingup_monte_carlo_small(config: LabConfig) -> None:
    r = monte_carlo_swingup(config, MODERATE, 8, seed=3)
    assert r.n == 8
    assert r.rate >= 0.75


def test_roa_contains_origin_and_excludes_far_states(config: LabConfig) -> None:
    s = next(s for s in balance_strategies(config) if s.key == "lqg")
    g = roa_grid(s, config, np.array([-1.4, 0.0, 1.4]), np.array([0.0]), 4.0, seed=0)
    assert g.tolist() == [[False, True, False]]


def test_balance_scenario_shapes() -> None:
    sc = BalanceScenario()
    x0 = sc.initial_states(10, np.random.default_rng(0))
    assert x0.shape == (10, 4)
    assert np.all(np.abs(x0[:, 2]) <= sc.tilt)
    assert sc.reference(sc.t_step) == sc.step
    assert sc.disturbance(sc.t_push) == sc.push


def test_structure_study(config: LabConfig) -> None:
    s = study_structure(config)
    assert s["controllability"]["rank"] == 4
    ranks = {o["output"]: o["rank"] for o in s["observability"]}
    assert ranks == {"x and theta": 4, "x only": 4, "theta only": 3}
    assert s["unstable_pole"] > s["rhp_zero_x"] > 0


def test_direct_discrete_design_tolerates_slower_sampling(config: LabConfig) -> None:
    s = study_sampling(config)
    ms = s["max_stable_Ts_s"]
    assert ms["rho_discrete"] >= ms["rho_emulated"]
    assert ms["rho_discrete_comp"] >= ms["rho_discrete"]


def test_inject_readme(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "a\n<!-- BEGIN GENERATED: t1 -->\nold\n<!-- END GENERATED: t1 -->\n"
        "<!-- BEGIN GENERATED: t2 -->\nkeep\n<!-- END GENERATED: t2 -->\n"
    )
    assert inject_readme(readme, {"t1": "| new |"}) == ["t1"]
    text = readme.read_text()
    assert "| new |" in text
    assert "old" not in text
    assert "keep" in text


@pytest.mark.slow
def test_quick_pipeline_end_to_end(tmp_path: Path) -> None:
    results = run_all(
        tmp_path / "img",
        RunSettings(mc_samples=6, roa_points=5, delay_mc=3),
        gif=False,
        log=lambda _: None,
    )
    tables = render_tables(json.loads(json.dumps(results, default=float)))
    for key in ("balance", "margins", "monte_carlo", "roa", "delay", "sampling", "kalman"):
        assert key in tables
    for name in (
        "step_comparison",
        "recovery_comparison",
        "roa",
        "monte_carlo",
        "delay_sampling",
        "kalman_estimation",
        "swingup",
        "lqr_tradeoff",
        "open_loop",
        "pole_map",
    ):
        assert (tmp_path / "img" / f"{name}.png").exists()


def test_cli_analyze(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["controllability"]["rank"] == 4

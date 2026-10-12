import numpy as np
import pytest

from optexec import ac, simulate


@pytest.fixture
def p():
    return ac.ACParams(X=1e5, T=60.0, N=30, sigma=0.02, gamma=2.5e-7, eta=2.5e-6, eps=1e-3, S0=50.0)


@pytest.mark.parametrize("kT", [0.0, 2.0, 6.0])
def test_monte_carlo_matches_theory(p, kT):
    n = ac.optimal_trades(p, ac.lambda_for_kappa(p, kT))
    s = simulate.summarize(simulate.simulate_shortfall(p, n, n_paths=300_000, seed=7))
    assert abs(s.mean - ac.expected_cost(p, n)) < 4.5 * s.stderr_of_mean
    assert s.std**2 == pytest.approx(ac.variance_cost(p, n), rel=0.02)


def test_immediate_has_no_price_risk(p):
    sf = simulate.simulate_shortfall(p, ac.immediate_trades(p), n_paths=1000, seed=1)
    assert sf.std() == pytest.approx(0.0, abs=1e-6)


def test_student_t_shocks_have_unit_variance():
    xi = simulate.standardized_shocks(np.random.default_rng(0), (400_000, 1), df=5)
    assert xi.var() == pytest.approx(1.0, rel=0.05)
    with pytest.raises(ValueError):
        simulate.standardized_shocks(np.random.default_rng(0), (2, 2), df=2)


def test_cvar_exceeds_mean_and_var():
    x = np.random.default_rng(0).standard_normal(100_000)
    s = simulate.summarize(x)
    assert s.cvar95 > s.var95 > s.mean


def test_wrong_trade_length_rejected(p):
    with pytest.raises(ValueError):
        simulate.simulate_shortfall(p, np.ones(3), n_paths=10)

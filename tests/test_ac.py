import numpy as np
import pytest
from scipy.optimize import minimize

from optexec import ac


@pytest.fixture
def p():
    return ac.ACParams(X=1e5, T=60.0, N=30, sigma=0.02, gamma=2.5e-7, eta=2.5e-6, eps=0.0, S0=50.0)


def test_params_validation():
    with pytest.raises(ValueError):
        ac.ACParams(X=-1, T=1, N=1, sigma=1, gamma=0, eta=1)
    # eta_tilde = eta - gamma tau / 2 must be positive
    with pytest.raises(ValueError):
        ac.ACParams(X=1, T=10, N=1, sigma=1, gamma=1.0, eta=1.0)


@pytest.mark.parametrize("kT", [0.0, 0.5, 2.0, 10.0])
def test_holdings_boundaries_and_monotone(p, kT):
    x = ac.optimal_holdings(p, ac.lambda_for_kappa(p, kT))
    assert x[0] == pytest.approx(p.X)
    assert x[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(x) <= 1e-9)
    assert ac.optimal_trades(p, ac.lambda_for_kappa(p, kT)).sum() == pytest.approx(p.X)


def test_zero_risk_aversion_is_twap(p):
    assert np.allclose(ac.optimal_trades(p, 0.0), ac.twap_trades(p))


def test_more_risk_aversion_front_loads(p):
    first = [ac.optimal_trades(p, ac.lambda_for_kappa(p, k))[0] for k in (0.0, 1.0, 3.0, 8.0)]
    assert all(a < b for a, b in zip(first, first[1:]))


def test_kappa_satisfies_defining_equation(p):
    lam = 3e-6
    k = ac.kappa(p, lam)
    kt2 = lam * p.sigma**2 / p.eta_tilde
    assert np.cosh(k * p.tau) == pytest.approx(1 + 0.5 * kt2 * p.tau**2, rel=1e-12)


def test_lambda_for_kappa_roundtrip(p):
    for kT in (0.3, 1.0, 5.0):
        assert ac.kappa(p, ac.lambda_for_kappa(p, kT)) * p.T == pytest.approx(kT, rel=1e-10)


@pytest.mark.parametrize("kT", [0.0, 1.0, 4.0])
def test_closed_forms_match_direct_sums(p, kT):
    lam = ac.lambda_for_kappa(p, kT)
    n = ac.optimal_trades(p, lam)
    assert ac.expected_cost_closed_form(p, lam) == pytest.approx(ac.expected_cost(p, n), rel=1e-9)
    assert ac.variance_closed_form(p, lam) == pytest.approx(ac.variance_cost(p, n), rel=1e-9)


def test_closed_form_is_the_numerical_minimiser(p):
    """The analytic trajectory must beat or equal a generic numerical optimiser of E + lam V."""
    lam = ac.lambda_for_kappa(p, 3.0)
    n_star = ac.optimal_trades(p, lam)
    scale = p.X

    def f(z):
        return ac.utility(p, z * scale, lam) / ac.utility(p, n_star, lam)

    res = minimize(
        f, np.full(p.N, 1.0 / p.N), method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1.0}], options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert f(n_star / scale) <= res.fun + 1e-9
    assert np.allclose(res.x * scale, n_star, atol=1e-3 * p.X)


def test_frontier_is_decreasing_in_risk(p):
    f = ac.efficient_frontier(p, np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3]))
    assert np.all(np.diff(f["variance"]) < 0)
    assert np.all(np.diff(f["expected_cost"]) > 0)


def test_time_varying_reduces_to_constant(p):
    lam = ac.lambda_for_kappa(p, 2.0)
    x1 = ac.optimal_holdings(p, lam)
    x2 = ac.optimal_holdings_time_varying(p, lam, np.full(p.N, p.sigma))
    assert np.allclose(x1, x2, rtol=1e-8, atol=1e-6)


def test_time_varying_slows_down_when_volatility_is_low_then_high(p):
    lam = ac.lambda_for_kappa(p, 2.0)
    sig = np.where(np.arange(p.N) < p.N // 2, p.sigma * 0.3, p.sigma * 3.0)
    n = ac.optimal_trades_time_varying(p, lam, sig)
    assert n.sum() == pytest.approx(p.X)
    # holding inventory is cheap while volatility is low: sells less early than the constant-vol solution
    assert n[0] < ac.optimal_trades(p, lam)[0]


def test_time_varying_is_optimal_for_its_own_variance(p):
    lam = ac.lambda_for_kappa(p, 2.0)
    sig = p.sigma * (1.0 + np.sin(np.linspace(0, 3, p.N)))
    n_tv = ac.optimal_trades_time_varying(p, lam, sig)
    n_c = ac.optimal_trades(p, lam)
    assert ac.utility(p, n_tv, lam, sig) <= ac.utility(p, n_c, lam, sig) + 1e-9

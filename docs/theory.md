# Theory: the Almgren-Chriss model as implemented here

This note derives every formula used in `src/optexec/ac.py`, so the code can be checked line by line.
Reference: R. Almgren and N. Chriss, *Optimal execution of portfolio transactions*, Journal of Risk 3 (2000).

## 1. Setting

A trader must **sell** $X$ shares by time $T$. Time is cut into $N$ intervals of length $\tau = T/N$,
with $t_k = k\tau$. Let $n_k \ge 0$ be the shares sold during interval $k$ and
$x_k = X - \sum_{j\le k} n_j$ the shares still held after interval $k$ ($x_0 = X$, $x_N = 0$).

**Price.** The unaffected price is an arithmetic random walk. Our own selling moves it permanently, linearly in the quantity sold:

$$S_k = S_{k-1} + \sigma\sqrt{\tau}\,\xi_k - \gamma\, n_k,\qquad \xi_k \sim \mathcal N(0,1)\ \text{i.i.d.}$$

**Temporary impact.** The price actually obtained in interval $k$ is worse than $S_{k-1}$ by a concession that depends on the *rate* of trading:

$$\tilde S_k = S_{k-1} - \varepsilon - h\!\left(\tfrac{n_k}{\tau}\right),\qquad h(v) = \eta\, v\ \ (\text{linear case}).$$

Here $\varepsilon$ is a fixed cost per share (half the spread), $\gamma$ the permanent impact per share and $\eta$ the temporary impact per (share/time).

**Implementation shortfall.** The cost relative to selling everything at the arrival price $S_0$:

$$\mathrm{IS} = X S_0 - \sum_{k=1}^N n_k \tilde S_k .$$

## 2. Mean and variance of the shortfall

Write $S_{k-1} = S_0 + \sum_{j<k}\sigma\sqrt\tau\,\xi_j - \gamma\sum_{j<k} n_j$. Then

$$\mathrm{IS} = \underbrace{\gamma\sum_k n_k\sum_{j<k} n_j}_{\text{permanent}} + \underbrace{\sum_k n_k\big(\varepsilon + h(n_k/\tau)\big)}_{\text{temporary}} - \underbrace{\sigma\sqrt\tau\sum_{j}\xi_j\sum_{k>j} n_k}_{\text{noise}} .$$

Since $\sum_{k>j} n_k = x_j$ and $\sum_k n_k\sum_{j<k} n_j = \tfrac12\big(X^2 - \sum_k n_k^2\big)$:

$$\boxed{E = \tfrac12\gamma X^2 + \varepsilon X + \frac{\tilde\eta}{\tau}\sum_k n_k^2,\qquad \tilde\eta = \eta - \tfrac12\gamma\tau}$$

$$\boxed{V = \tau\sum_{k=1}^N \sigma_k^2\, x_k^2}\quad(\sigma_k = \sigma\ \text{when volatility is constant}).$$

Reading the formulas:

* $\tfrac12\gamma X^2$ and $\varepsilon X$ do not depend on the schedule: they are paid whatever we do.
* $\frac{\tilde\eta}{\tau}\sum n_k^2$ is the cost of trading fast: it is smallest for equal slices (TWAP).
* $V$ is the price risk of the shares still held: it is smallest when we sell early.

**Why $\tilde\eta > 0$ is required.** $\tilde\eta = \eta - \gamma\tau/2$ is the net convexity of the cost. If it is $\le 0$, the model rewards trading faster, the cost is not convex, and no optimum exists. The code refuses such parameters (`ACParams.__post_init__`). In practice this means the time step must satisfy $\tau < 2\eta/\gamma$.

## 3. The optimal trajectory

The trader minimises the mean-variance objective $U = E + \lambda V$, where $\lambda \ge 0$ is the risk aversion. Only $x_1,\dots,x_{N-1}$ are free. Using $n_k = x_{k-1}-x_k$:

$$\frac{\partial U}{\partial x_j} = \frac{2\tilde\eta}{\tau}\,(2x_j - x_{j-1} - x_{j+1}) + 2\lambda\tau\sigma_j^2 x_j = 0 .$$

With constant volatility this is the linear recurrence

$$x_{j-1} - 2x_j + x_{j+1} = \tilde\kappa^2\tau^2\, x_j,\qquad \tilde\kappa^2 = \frac{\lambda\sigma^2}{\tilde\eta}.$$

Trying $x_j \propto e^{\pm\kappa t_j}$ gives the characteristic equation $\cosh(\kappa\tau) = 1 + \tfrac12\tilde\kappa^2\tau^2$ (implemented in `ac.kappa`), and the boundary conditions $x_0 = X$, $x_N = 0$ give

$$\boxed{x_j = X\,\frac{\sinh\big(\kappa (T - t_j)\big)}{\sinh(\kappa T)},\qquad n_j = x_{j-1}-x_j .}$$

* $\kappa T \to 0$ ($\lambda\to 0$): $x_j \to X(1 - t_j/T)$, the **TWAP** (risk-neutral trader).
* Larger $\kappa T$: the sale is front-loaded; $1/\kappa$ is the "half-life" of the position.
* `ac.lambda_for_kappa` inverts the relation, so experiments can be parameterised by the dimensionless urgency $\kappa T$ instead of the raw $\lambda$.

The expected cost and variance of the optimal trajectory have closed forms (`expected_cost_closed_form`, `variance_closed_form`). The tests check them against direct summation of the definitions and against Monte Carlo.

## 4. Time-varying volatility

If $\sigma_k$ changes from interval to interval (for example an intraday pattern), the first-order conditions form a tridiagonal linear system:

$$-x_{j-1} + \Big(2 + \frac{\lambda\tau^2\sigma_j^2}{\tilde\eta}\Big)x_j - x_{j+1} = 0,\quad j=1,\dots,N-1,\ \ x_0 = X,\ x_N = 0,$$

solved with `scipy.linalg.solve_banded` in `ac.optimal_holdings_time_varying`. With constant $\sigma$ it reproduces the closed form (tested). Intuition: holding shares is cheap while volatility is low and expensive while it is high, so the schedule trades more just before volatile periods.

## 5. Non-linear temporary impact (not implemented here)

Empirical studies often find that temporary impact is **concave** in the trading rate, roughly $h(v)=\eta\,\mathrm{sgn}(v)|v|^{\beta}$ with $\beta$ near $1/2$. The expected cost then becomes

$$E = \tfrac12\gamma\Big(X^2 - \sum n_k^2\Big) + \varepsilon X + \eta\,\tau^{-\beta}\sum_k n_k^{1+\beta},$$

which has no closed-form minimiser and has to be optimised numerically. With $\beta<1$ the optimum can be extreme (a few very large slices), because trading fast is relatively cheap. This project does not implement the extension; it measures, in `calibration.fit_power_impact`, whether the data favour a concave shape, which is the first step in deciding whether the extension is needed.

## 6. Assumptions and where they break

| Assumption | Why it is questionable |
|---|---|
| Linear permanent and temporary impact | Measured impact is usually concave in size; this is checked on the data in `calibration.fit_power_impact` |
| Constant volatility | Volatility has an intraday pattern and clusters (see `volatility.py`) |
| Gaussian price increments | Returns have fat tails; the simulator accepts Student-t shocks |
| Deterministic schedule fixed in advance | A trader can adapt to the price and to liquidity (dynamic strategies) |
| One asset | Portfolio liquidation involves correlations |
| Impact does not depend on the market state | Liquidity changes during the day |
| Impact parameters known | They must be estimated, and public data cannot identify the effect of one's own orders (see `calibration.py`) |

## References

* R. Almgren, N. Chriss (2000). *Optimal execution of portfolio transactions.* Journal of Risk 3(2).
* R. Almgren (2003). *Optimal execution with nonlinear impact functions and trading-enhanced risk.* Applied Mathematical Finance 10.
* J.-P. Bouchaud, J. D. Farmer, F. Lillo (2009). *How markets slowly digest changes in supply and demand.* In Handbook of Financial Markets: Dynamics and Evolution.
* B. Tóth et al. (2011). *Anomalous price impact and the critical nature of liquidity in financial markets.* Physical Review X 1.
* T. Bollerslev (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics 31.
* A. Patton (2011). *Volatility forecast comparison using imperfect volatility proxies.* Journal of Econometrics 160.

(Check the exact bibliographic details before citing them elsewhere.)

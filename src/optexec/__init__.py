"""optexec: optimal trade execution with the Almgren-Chriss framework.

Modules
-------
ac           Discrete-time Almgren-Chriss model: closed form, frontier, time-varying volatility.
simulate     Monte Carlo simulation of execution shortfall.
data         Loaders for Binance public data, and a synthetic market with known parameters.
calibration  Volatility profile and market-impact estimation.
volatility   EWMA and GARCH(1,1) forecasting, with out-of-sample evaluation.
plots        Figures.
"""

__version__ = "0.1.0"

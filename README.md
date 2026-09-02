# BTC Options: Pure-Jump Modelling, Pricing and Hedging

Research code for testing whether Bitcoin returns contain a Brownian component, and
for pricing and hedging Deribit BTC options under a CGMY model if they do not.

**Status: work in progress.** The data pipeline, activity-index test and implied
volatility solver are complete. Calibration and the PIDE solver are still being
built — see [Module status](#module-status).

---

## Motivation

The Black–Scholes framework assumes the underlying has a continuous martingale
component. Whether that assumption holds for Bitcoin is an empirical question, and
it has a direct consequence: if the price process is pure jump, a diffusion-based
model is misspecified and the hedge derived from it is wrong.

Todorov and Tauchen's activity index gives a way to test this from high-frequency
data. On 5-minute Deribit BTC index data for May–June 2025 the estimated activity
index is **β̂ = 1.74**, and the null of β = 2 (a Brownian component present) is
rejected at p < 1e-7. That result motivates the rest of the repository: a CGMY
specification, calibrated to the option surface, then priced and hedged through a
PIDE.

## Pipeline

```
Deribit API
    │
    ├── BTC index, 5-minute OHLCV ──────► activity index β̂, pure-jump test
    │                                     (Todorov_activity_indexes.py)
    │
    └── full option chain, all strikes ─► implied vol surface + vega
        (calls and puts, inverse                (BS_Implied_Vol.py)
         option conventions)                          │
                                                      ▼
                                       CGMY calibration: vega-weighted least
                                       squares + relative-entropy regularisation
                                          (Jump_models_calibration.py)
                                                      │
                                                      ▼
                                       finite-difference PIDE pricing and hedging
                                       (CGMY_pricing_hedging_PIDE.py)
```

## Module status

| Module | Status | What it does |
| --- | --- | --- |
| `src/Deribit_option_retrieval.py` | Complete | Retrieves the BTC index and the full option chain from Deribit and builds a long panel: one row per (time bin, instrument). |
| `src/Todorov_activity_indexes.py` | Complete | Activity-index estimator β̂ and the pure-jump vs jump-diffusion test. |
| `src/BS_Implied_Vol.py` | Complete | Vectorised Black–Scholes implied volatility and vega. |
| `src/Jump_models_calibration.py` | **In progress — to be completed** | CGMY characteristic function, Carr–Madan FFT pricer, entropy regularisation, calibration driver. |
| `src/CGMY_pricing_hedging_PIDE.py` | **In progress — to be completed** | Finite-difference PIDE solver for CGMY pricing and hedging. |
| `notebooks/` | **To be added** | The notebook that runs the full analysis end to end. |

Each in-progress module carries a header docstring listing exactly which functions
work and which are still being written.

---

## 1. Data — `Deribit_option_retrieval.py`

Three streams from the public Deribit API, joined into one panel.

- **Index / spot**: OHLCV for `BTC-PERPETUAL` at a chosen resolution, via
  `get_tradingview_chart_data`.
- **Options**: tick-level trades for every listed BTC option — all strikes, all
  expiries, calls and puts, no filter — via
  `get_last_trades_by_currency_and_time`, then binned to the same resolution.
- **Dated futures**: OHLCV for the termed future sharing each option's expiry,
  fetched one instrument per distinct expiry in the panel.
- **Optional live snapshot**: `get_live_chain()` pulls mark price and mark IV for
  every listed instrument from `get_book_summary_by_currency`, so a cross-section
  is available without waiting for trades.

Two conventions are handled explicitly, and both matter for calibration.

**Inverse options.** Deribit quotes BTC options in BTC per 1 BTC of notional, not
in USD. The panel carries both `price_btc` and `price_usd`, and the pricer in the
calibration module prices the inverse payoff directly rather than converting.

**Futures as the underlying.** Options settle against the future,
so the panel carries the traded future itself. `get_dated_futures_for_expiries()`
pulls the OHLCV of the dated future matching each option expiry, and
`attach_dated_future()` merges it onto every row on `(timestamp, expiry_dt)` —
so a `BTC-27JUN25-*` option only ever picks up `BTC-27JUN25`, never another
maturity. That adds `future_instrument_name`, `future_open/high/low/price/volume`
and `future_basis` to the panel. The perpetual is carried alongside as
`perp_close`. Expiries with no listed dated future — Deribit lists futures for
weekly, monthly and quarterly maturities, not for every daily option expiry — are
reported and left as `NaN` rather than silently filled.

`index_basis_check()` reports the perpetual-versus-index basis, so substituting the
perp for the index is a visible choice rather than a silent one. That fallback is
off by default.

Main entry point:

```python
from src.Deribit_option_retrieval import get_deribit_pricing_data

panel = get_deribit_pricing_data(
    start_date="2025-05-01",
    end_date="2025-06-30",
    resolution="5",
    min_tte_days=1.0,
    fetch_dated_futures=True,
)
```

`resume_from_saved()` rebuilds the panel from cached CSVs (the dated futures are
re-fetched, since they are not in those CSVs).
`get_chain_snapshot(panel, timestamp)` extracts a single cross-section for one
calibration run.

## 2. Activity index — `Todorov_activity_indexes.py`

`Beta_hat(p, X)` estimates the activity index by comparing the power variation of
log-price increments at sampling interval Δ and at 2Δ. The two time frequencies are needed to evaluate the approximate index,
β^ and also be used in evaluate the error of approximation.

`PJvsJDTest` implements the formal test of H₀: β = 2, that is, that a Brownian
component is present. The class assembles the asymptotic variance from three
pieces: the constant K₁; a data-dependent factor K₂ built from rolling products of
the increments; and K₃, which depends on the Gamma function of the power parameter p in μ_p, μ_2p and μ_p,k,
obtained by two-dimensional quadrature. `test()` uses the normal distribution in order to test the null hypothesis


```python
from src.Todorov_activity_indexes import Beta_hat, PJvsJDTest

beta = Beta_hat(p=0.5, X=index_df)              # index_df has a 'log_price' column
test = PJvsJDTest(X=increments, p=0.5, delta_n=delta)
test.test(beta)     # {'p_value': ..., 'z': ..., 'reject H_0': ...}
```

Sampling frequency is the main thing that moves this estimate: too fine and
microstructure noise pulls β̂ down, too coarse and the sample shrinks. Five-minute
sampling is the usual compromise in this literature.

## 3. Implied volatility — `BS_Implied_Vol.py`

Two implementations of the same inversion. `implied_vol_scipy` uses SciPy's root
finder and serves as the reference. `implied_vol` is the production path: Brent's
method written to compile under Numba and exposed through `@vectorize`, so a whole
cross-section inverts in one call.

The Numba routine checks the no-arbitrage bounds before bracketing, returns `NaN`
for quotes that violate them rather than a spurious root, and widens the upper
bracket to σ = 10 before giving up. `verify_round_trip()` checks
σ → price → σ → price accuracy.

Output feeds two things: the volatility surface behind the volatility-arbitrage
signal, and the vega weights that `black_scholes_vega_vectorized` supplies to the
calibration objective.

```python
from src.BS_Implied_Vol import implied_vol, black_scholes_vega_vectorized

iv   = implied_vol(S, K, T, r, C)
vega = black_scholes_vega_vectorized(S, K, T, r, iv)
```

## 4. Calibration — `Jump_models_calibration.py` *(in progress)*

CGMY is calibrated to the option cross-section by minimising vega-weighted squared
pricing error, regularised by the relative entropy between the candidate
risk-neutral measure and a prior, following Cont and Tankov. The regularisation is
what makes an otherwise ill-posed inverse problem stable.

- `Jump_Models.φ_CGMY` — the CGMY characteristic function.
- `Jump_Models.FT_pricer` — Carr–Madan Fourier pricer for inverse options,
  vectorised across strikes, with Simpson weights and the convexity correction ω
  applied so the discounted forward is a martingale. The damping parameter α is
  negative here, since an inverse call is a put on the inverse price.
- `relative_entropy` — computes ε(Q|P) from the two CGMY Lévy measure.
- `BFGS` — BFGS method over (C, β, λ_n, λ_p), to find the optimal set of parameters
- `DE_global`— Due to the non-convex behavior of CGMY model, there are too many local-minimum.
  So the the global optimization is needed here in order to find the most plausible starting point before
  searching with BFGS above.
- `regularised_factor_bisection` — chooses the regularisation weight α by the
  discrepancy principle, bisecting on log α until the fitted error reaches δ·e₀,
  where e₀ is the error of the unregularised fit. Results are cached and warm
  started.

Fitted to 36 near-the-money Deribit BTC call strikes (0.8 < moneyness < 1.2).

**Still to be completed:** the multi-strike FFT pricer, analytic gradients of the
objective, and the hand-written BFGS optimiser that will use them. Calibration
currently runs through SLSQP.

## 5. PIDE pricing and hedging — `CGMY_pricing_hedging_PIDE.py` *(in progress)*

A finite-difference scheme for the CGMY partial integro-differential equation, on a
log-price grid spanning ±5 empirical standard deviations.

The integral term is the difficult part. The CGMY Lévy measure is not integrable at
the origin, so small jumps are treated separately from large ones. `integral_part`, for large jump,
splits the quadrature into four pieces — the first and second pieces, I and II, 
are the evalution of the negative jumps whose size are within and beyond the scope of the grid respectively.
While the third and the fourth part, III and IV, are as same as I and II chronologically 
but dealing with the positiive jumps instead.
The small-jump contribution is absorbed into effective drift and diffusion coefficients ω(ε) and
σ(ε) in `differntial_part`, which then assembles the tridiagonal operator for the
differential part.

**Still to be completed:** the tridiagonal assembly does not yet return, the
time-stepping loop and boundary conditions are unwritten, and delta extraction for
the hedge is not implemented. This module also needs final calibrated parameters,
so it is blocked on the calibration module above.

---

## Installation

```bash
git clone https://github.com/S-Homchum/Crypto-Currencies-Research.git
cd Crypto-Currencies-Research
pip install -r requirements.txt
```

Python 3.9 or later.

## References

Todorov, V. and Tauchen, G. (2010). Activity signature functions for
high-frequency data analysis. *Journal of Econometrics*, 154(2), 125–138.

Aït-Sahalia, Y. and Jacod, J. (2009). Estimating the degree of activity of jumps in high frequency data. 
*The Annals of Statistics*, 37(5A), 2202–2244.

Carr, P. and Madan, D. (1999). Option valuation using the fast Fourier transform.
*Journal of Computational Finance*, 2(4), 61–73.

Cont, R. and Tankov, P. (2004). Non-parametric calibration of jump-diffusion option
pricing models. *Journal of Computational Finance*, 7(3), 1–49.

Cont, R. and Voltchkova, E. (2005). A finite difference scheme for option pricing
in jump diffusion and exponential Lévy models. *SIAM Journal on Numerical
Analysis*, 43(4), 1596–1626.

## Author

**Surapas Homchum** — MSc Mathematical Finance, University of Warwick (from
September 2026)

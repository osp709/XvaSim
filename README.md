<p align="center">
  <img src="assets/logo.png" alt="XvaSim Logo" width="150" />
</p>

# XvaSim: Valuation Adjustment (XVA) Simulation & Calculation Engine

`XvaSim` is a high-performance Python library designed for simulating and calculating credit and valuation adjustments (XVAs) and pricing multi-asset derivatives across Interest Rates, Foreign Exchange (FX), Credit, and Inflation risk factors.

---

## 🎯 Core Pricing Philosophy: Monte Carlo as Primary Pricer

> [!IMPORTANT]
> **Monte Carlo Simulation is the primary pricing engine across `XvaSim`.**
> All public pricing functions (`price_*`) execute stochastic path simulations by default, computing path-wise discounted cash flows, simulated present values, standard errors, and terminal distributions.
> 
> **Analytical closed-form solutions (`benchmark_price_*`) are provided strictly as mathematical benchmarks** to validate, test, and verify the accuracy and convergence of the Monte Carlo simulation engine.

Every primary pricer returns both the simulated Monte Carlo price (`"price"` and `"std_error"`) and the exact analytical benchmark (`"analytical_benchmark_price"`).

---

## 📊 Supported Securities & Simulated Risk Factors

The following table summarizes all financial instruments and valuation adjustments supported by `XvaSim`, including the primary Monte Carlo pricer, dedicated analytical benchmark function, and the underlying simulated risk factors:

| Security / Instrument | Asset Class / Category | Primary MC Pricer (`price_*`) | Analytical Benchmark (`benchmark_price_*`) | Simulated Risk Factors | Description & Payoff Structure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla Interest Rate Swap (IRS)** | Rates | `price_interest_rate_swap` | `benchmark_price_interest_rate_swap` | Domestic Interest Rate $r_d(t)$ | Single-currency fixed-for-floating interest rate swap. Pays fixed/floating coupons on preset schedule. Computes par swap rate & forward annuity (PV01). |
| **Cross-Currency Swap (XCCY)** | Rates / FX | `price_cross_currency_swap` | `benchmark_price_cross_currency_swap` | Domestic IR $r_d(t)$, Foreign IR $r_f(t)$, Spot FX $S(t)$ | Multi-currency swap supporting fixed-for-floating, fixed-for-fixed, and floating-for-floating basis swaps with optional principal notional exchanges. |
| **FX Forward** | FX | `price_foreign_exchange_forward` | `benchmark_price_foreign_exchange_forward` | Domestic IR $r_d(t)$, Foreign IR $r_f(t)$, Spot FX $S(t)$ | Forward exchange contract: $N \times (S(T) - K)$ priced under domestic risk-neutral measure with Covered Interest Parity (CIP) benchmark. |
| **European FX Option (Call / Put)** | FX | `price_foreign_exchange_option` | `benchmark_price_foreign_exchange_option` | Spot FX $S(t)$, Domestic IR $r_d(t)$, Foreign IR $r_f(t)$, FX Variance $v(t)$ (Heston) | Vanilla European option on FX spot: $N \times \max(\omega(S(T) - K), 0)$ benchmarked against Garman-Kohlhagen / Black-76 / Heston semi-analytical formulas. |
| **Zero-Coupon Inflation Swap (ZCIS)** | Inflation | `price_zero_coupon_inflation_swap` | `benchmark_price_zero_coupon_inflation_swap` | Nominal IR $r_n(t)$, Real IR $r_r(t)$, CPI Index $I(t)$ | Single-exchange swap at maturity: pays fixed compounded rate $(1+K)^T - 1$ vs floating realized inflation index return $I(T)/I(0) - 1$. |
| **Year-on-Year Inflation Swap (YoY)** | Inflation | `price_year_on_year_inflation_swap` | Interpolated CPI forward projection | Nominal IR $r_n(t)$, Real IR $r_r(t)$, CPI Index $I(t)$ | Multi-period swap exchanging annual fixed rate $K$ for annual CPI growth $\frac{I(T_i)}{I(T_{i-1})} - 1$ on each reset date. |
| **CPI Index Option (Caplet / Floorlet)** | Inflation | `price_consumer_price_index_option` | `benchmark_price_consumer_price_index_option` | Nominal IR $r_n(t)$, Real IR $r_r(t)$, CPI Index $I(t)$ | European option on inflation index: Caplet $N \times \max\left(\frac{I(T)}{I(0)} - (1+K)^T, 0\right)$ and Floorlet $N \times \max\left((1+K)^T - \frac{I(T)}{I(0)}, 0\right)$. |
| **Portfolio Credit Valuation Adjustment (CVA)** | Credit / Multi-Asset | `compute_cva` | Marginal PD via CIR zero-curve | Credit Hazard Rate $\lambda(t)$, Underlying Portfolio Exposure | Path-wise Monte Carlo integration of counterparty default risk across simulated market exposure paths, discount factors, and marginal default probabilities. |
| **XVA Suite (DVA/FVA/KVA/MVA)** | Credit / Multi-Asset | `compute_dva`, `compute_fva`, `compute_kva`, `compute_mva`, `compute_total_xva` | Marginal PD via CIR zero-curve | Credit Hazard Rate, Exposure, Discount Factors | Debt, funding, capital, and initial-margin valuation adjustments with a single-pass `compute_total_xva` aggregator covering both sides of the trade and the CCAR capital charge. |
| **Portfolio (FX netting set)** | Credit / Multi-Asset | `compute_portfolio_exposure`, `simulate_portfolio_exposure`, `compute_portfolio_xva` | Marginal PD via CIR zero-curve | Spot FX $S(t)$, Domestic/foreign IR, Credit Hazard Rate | `Trade` protocol (`FXForwardTrade`, `FXEuropeanOptionTrade`), `MarketSimulation`, netting-set `Portfolio`, conditional-valuation exposure, and the end-to-end portfolio XVA ledger with EE/EPE/PFE profile. |

---

## 🎲 Supported Risk Factors & Stochastic Models

`XvaSim` features a modular dynamic registry (`ModelRegistry`) and factories (`create_ir_model`, `create_credit_model`, `create_fx_model`, `create_inflation_model`) allowing plug-and-play selection of stochastic models for each simulated risk factor. Query the live registry at runtime with `list_available_models(...)`.

| Risk Factor | Stochastic Model | Concrete Class | Registry Keys | Key Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Interest Rate** | Linear Gauss-Markov (LGM) | `LGMModel` | `lgm`, `linear_gauss_markov` | $\kappa_{\text{ann}}$, $\sigma(t)_{\text{ann}}$, Discount Curve |
| | Hull-White 1-Factor (HW1F) | `HullWhite1FModel` | `hull_white`, `hull_white_1f`, `hw1f` | $a_{\text{ann}}$, $\sigma_{\text{ann}}$, Discount Curve |
| | Vasicek Short Rate | `VasicekModel` | `vasicek` | $\kappa_{\text{ann}}$, $\theta_{\text{ann}}$, $\sigma_{\text{ann}}$, $r_0$ |
| | Cox-Ingersoll-Ross (CIR) | `CIRInterestRateModel` | `cir`, `cir_ir`, `cox_ingersoll_ross` | $\kappa_{\text{ann}}$, $\theta_{\text{ann}}$, $\sigma_{\text{ann}}$, $r_0$ |
| **Foreign Exchange (FX Spot & Volatility)** | Two-Currency Multi-Factor FX | `TwoCurrencyFXModel` | `two_currency`, `cross_currency` | Domestic/foreign IR models, $S_0$, $\sigma_{\text{fx}}$, $3\times3$ correlation |
| | Garman-Kohlhagen / Black-Scholes | `GarmanKohlhagenFXModel` | `garman_kohlhagen`, `black_scholes`, `gbm` | $S_0$, $\sigma_{\text{fx}}$, $r_d$, $r_f$ (or discount curves) |
| | Heston Stochastic Volatility FX | `HestonFXModel` | `heston`, `heston_fx` | $S_0$, $v_0$, $\kappa_v$, $\theta_v$, $\sigma_v$, $\rho_{S,v}$, $r_d$, $r_f$ |
| **Counterparty Credit / Hazard Rate** | Cox-Ingersoll-Ross (CIR) Hazard Rate | `CIRHazardRateModel` | `cir`, `cir_hazard_rate`, `cox_ingersoll_ross` | $\kappa_{\text{ann}}$, $\theta_{\text{ann}}$, $\sigma_{\text{ann}}$, $\lambda_0$ |
| **Inflation (CPI Index & Real Rates)** | Jarrow-Yildirim (JY) Two-Economy | `JarrowYildirimModel` | `jarrow_yildirim`, `jy`, `two_factor_hw` | Nominal & real IR models, $I_0$, $\sigma_I$, $3\times3$ correlation |
| | Black CPI Forward Log-Normal | `BlackInflationModel` | `black`, `black_inflation`, `lognormal` | Nominal curve, Real curve, $I_0$, $\sigma_{I,\text{ann}}$ |

> `RiskFactorType` also enumerates `EQUITY` and `COMMODITY` risk-factor categories (reserved; no concrete models implemented yet).

### Model Reference

Every model can be constructed from its parameter dataclass (`params=...`) or from individual keyword arguments; defaults are shown below. All model classes are exported from the package root (`from xvasim import ...`).

**Interest rate models (`xvasim.models.ir`)** — subclass `InterestRateModel`; all expose the discount-curve properties `discount_curve_yrs` / `discount_factors`, the curve helpers `interpolate_discount_factor(t)` and `instantaneous_forward(t)`, plus `short_rate(t, state)`, `zero_coupon_bond(t, T, state)`, `discount_path(times, state_paths)`, and `simulate_paths(times, n_paths, ...)`.

- `LGMModel` — `(params=None, *, kappa_ann=0.03, sigma_grid_yrs, sigma_values_ann, discount_curve_yrs, discount_factors)`. Piecewise-constant volatility $\sigma(t)$; helpers `h_function(t)`, `zeta(t)`, `sigma_at(t)`; analytical swaption pricing `swaption_price_normal(...)`; classmethod `calibrate_to_swaptions(...)`. Dataclass `LGMParams`.
- `HullWhite1FModel` — `(params=None, *, a_ann=0.03, sigma_ann=0.01, discount_curve_yrs, discount_factors)`. Helpers `b_function(t, T)`, `alpha(t)`. Dataclass `HullWhite1FParams`.
- `VasicekModel` — `(params=None, *, kappa_ann=0.15, theta_ann=0.03, sigma_ann=0.015, r0_ann=0.025, discount_curve_yrs=None, discount_factors=None)`. When no curve is supplied, an analytical model-implied term structure is generated. Dataclass `VasicekParams`.
- `CIRInterestRateModel` — `(params=None, *, kappa_ann=0.20, theta_ann=0.03, sigma_ann=0.08, r0_ann=0.025, discount_curve_yrs=None, discount_factors=None)`. Simulation uses full truncation (states clamped at 0); non-negative when the Feller condition $2\kappa\theta \ge \sigma^2$ holds. Dataclass `CIRInterestRateParams`.

**Credit model (`xvasim.models.credit`)**

- `CIRHazardRateModel` — `(params=None, *, kappa_ann=0.5, theta_ann=0.03, sigma_ann=0.10, lambda_0_ann=0.02)`. Closed-form `survival_probability(tenors_yrs)`, `marginal_pd(tenors_yrs)`, classmethod `calibrate_from_spreads(credit_spreads_ann, tenors_yrs)`. Dataclass `CIRHazardRateParams`.

**FX models (`xvasim.models.fx`)** — subclass `FXModel`. `simulate_paths(maturity_yrs, n_paths, n_steps, ...)` returns the tuple `(times, x_dom, x_for, fx_spot)` (for Heston, `(times, v_paths, x_dummy, fx_spot)`).

- `TwoCurrencyFXModel` — `(domestic_ir_model, foreign_ir_model, spot_fx, fx_vol_ann, correlation_matrix)`; IR components may be `InterestRateModel` instances or `LGMParams`. Constructors `from_params(...)` and `from_ir_models(...)`. Quanto drift $\rho_{f,S}\sigma_f\sigma_{fx}$ applied to the foreign rate under the domestic measure. Dataclass `TwoCurrencyFXParams`.
- `GarmanKohlhagenFXModel` — `(params=None, *, spot_fx=1.0, fx_vol_ann=0.10, domestic_rate_ann=0.0, foreign_rate_ann=0.0, discount_curve_domestic_yrs=None, discount_factors_domestic=None, discount_curve_foreign_yrs=None, discount_factors_foreign=None)`. Curve-override-capable `domestic_discount_factor(t)` / `foreign_discount_factor(t)`, `forward_rate(T)`, closed-form `closed_form_option_price(...)`; `from_params(...)`. Dataclass `GarmanKohlhagenFXParams`.
- `HestonFXModel` — `(params=None, *, spot_fx=1.0, v_0=0.04, kappa_ann=2.0, theta_ann=0.04, sigma_v_ann=0.20, rho=-0.5, domestic_rate_ann=0.0, foreign_rate_ann=0.0, discount_curve_*_yrs=None, discount_factors_*=None)`. Semi-analytical `closed_form_option_price(...)` (put via put-call parity), `is_feller_satisfied`, `num_factors == 2`; `from_params(...)`. Dataclass `HestonFXParams`.

**Inflation models (`xvasim.models.inflation`)** — subclass `InflationModel`. `simulate_paths(...)` returns an `InflationSimulationResult` with fields `times`, `nominal_states`, `real_states`, `cpi_index`, `nominal_short_rates`, `real_short_rates`, `nominal_discount_factors`; it also supports 4-tuple unpacking: `times, x_nom, x_real, cpi = result`.

- `JarrowYildirimModel` — `(nominal_ir_model, real_ir_model, base_cpi=100.0, cpi_vol_ann=0.02, correlation_matrix=None)` (default: 3×3 identity). `forward_cpi(T)`, `zero_coupon_inflation_swap_rate(T)`, `total_variance_at(T)`; constructor `from_ir_models(...)`. Dataclass `JarrowYildirimParams`.
- `BlackInflationModel` — `(params=None, *, nominal_discount_curve_yrs, nominal_discount_factors, real_discount_curve_yrs, real_discount_factors, base_cpi=100.0, cpi_vol_ann=0.02)`. `interpolate_nominal_df(t)` / `interpolate_real_df(t)`, `forward_cpi(T)`, `zero_coupon_inflation_swap_rate(T)`, closed-form `price_consumer_price_index_option_analytical(...)`; `from_params(...)`. Dataclass `BlackInflationParams`.

---

## ⚡ Quasi-Monte Carlo (QMC) & Variance Reduction

`XvaSim` provides built-in low-discrepancy sequences to accelerate Monte Carlo convergence and minimize simulation noise:

| Sequence Generator | `RandomSequenceType` Member / Alias | Description |
| :--- | :--- | :--- |
| **Sobol** | `RandomSequenceType.SOBOL` / `"sobol"` | Scrambled Sobol sequence with Owen scrambling for unbiased error estimation. |
| **Halton** | `RandomSequenceType.HALTON` / `"halton"` | Generalized scrambled Halton sequence. |
| **Latin Hypercube** | `RandomSequenceType.LATIN_HYPERCUBE` / `"lhs"` | Stratified Latin Hypercube Sampling (LHS). |
| **Pseudo-Random** | `RandomSequenceType.PSEUDO` / `"pseudo"` | Standard NumPy PRNG (`default_rng`). |

All pricing functions and model path simulators accept `random_type`, `seed`, and `scramble`. Use `compare_t0_npv_fitting` to benchmark convergence across generators.

---

## 🧮 Mathematical Foundations & Stochastic Differential Equations (SDEs)

### 1. Interest Rate Models

#### Linear Gauss-Markov (LGM) Model
The LGM state variable $x(t)$ evolves as a zero-mean Gaussian diffusion:

$$dx(t) = -\kappa\,x(t)\,dt + \sigma(t)\,dW(t), \quad x(0) = 0$$

The zero-coupon bond price $P(t, T)$ under the domestic numeraire is given by:

$$P(t,T) = \frac{P(0,T)}{P(0,t)}\exp\!\left(-H(T)\,x(t) - \tfrac{1}{2}\bigl(H(T)^2 - H(t)^2\bigr)\zeta(t)\right)$$

where:
$$H(t) = \frac{1 - e^{-\kappa t}}{\kappa}, \quad \zeta(t) = \int_0^t \sigma(s)^2 e^{-2\kappa(t-s)} ds$$

#### Hull-White 1-Factor (HW1F) Model
Exact term-structure fitting Gaussian short-rate model under the risk-neutral measure $\mathbb{Q}$:

$$dr(t) = (\theta(t) - a r(t))\,dt + \sigma\,dW(t)$$

where $\theta(t) = \frac{\partial f(0,t)}{\partial t} + a f(0,t) + \frac{\sigma^2}{2a}(1 - e^{-2at})$ fits the initial zero-coupon discount curve $P(0, t)$, with analytical bond price:

$$P(t, T) = A(t, T)\exp(-B(t, T) r(t)), \quad B(t, T) = \frac{1 - e^{-a(T-t)}}{a}$$

#### Vasicek Short-Rate Model
Mean-reverting Ornstein-Uhlenbeck short-rate process:

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\,dW(t)$$

#### Cox-Ingersoll-Ross (CIR) Interest Rate Model
Mean-reverting square-root diffusion guaranteeing non-negative interest rates when the Feller condition $2\kappa\theta \ge \sigma^2$ holds:

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t)$$

---

### 2. Foreign Exchange (FX) Models

#### Garman-Kohlhagen (Black-Scholes FX) Model
Under the domestic risk-neutral measure $\mathbb{Q}_d$, the spot exchange rate $S(t)$ (domestic currency per unit foreign currency) evolves as:

$$\frac{dS(t)}{S(t)} = (r_d(t) - r_f(t))\,dt + \sigma_{\text{fx}}\,dW(t)$$

The closed-form analytical benchmark for European call ($C$) and put ($P$) options is:

$$C = P_d(0, T) \left[ F\,N(d_1) - K\,N(d_2) \right], \quad P = P_d(0, T) \left[ K\,N(-d_2) - F\,N(-d_1) \right]$$

where $F = S_0 \frac{P_f(0, T)}{P_d(0, T)}$ and $d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma_{\text{fx}}^2 T}{\sigma_{\text{fx}}\sqrt{T}}, \; d_2 = d_1 - \sigma_{\text{fx}}\sqrt{T}$.

#### Heston Stochastic Volatility FX Model
Captures volatility smile and skew by modeling instantaneous FX variance $v(t)$ as a CIR process:

$$\frac{dS(t)}{S(t)} = (r_d(t) - r_f(t))\,dt + \sqrt{v(t)}\,dW_S(t)$$

$$dv(t) = \kappa_v (\theta_v - v(t))\,dt + \sigma_v \sqrt{v(t)}\,dW_v(t)$$

with instantaneous Brownian motion correlation:
$$d\langle W_S, W_v \rangle_t = \rho_{S, v}\,dt$$

#### Two-Currency Multi-Factor FX Model
Couples stochastic domestic interest rates $r_d(t)$, foreign interest rates $r_f(t)$ under the domestic pricing measure with a quanto drift adjustment, and the spot FX rate $S(t)$ with a $3 \times 3$ correlation structure:

$$\begin{pmatrix} dW_d(t) \\ dW_f(t) \\ dW_S(t) \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} 1 & \rho_{d, f} & \rho_{d, S} \\ \rho_{d, f} & 1 & \rho_{f, S} \\ \rho_{d, S} & \rho_{f, S} & 1 \end{pmatrix} dt \right)$$

Under the domestic risk-neutral measure $\mathbb{Q}_d$, the foreign short rate acquires a quanto adjustment:
$$dr_f(t) = \left( \theta_f(t) - a_f r_f(t) - \rho_{f, S} \sigma_f(t) \sigma_{\text{fx}} \right) dt + \sigma_f(t)\,dW_f^{\mathbb{Q}_d}(t)$$

---

### 3. Counterparty Credit & Hazard Rate Models

#### Cox-Ingersoll-Ross (CIR) Hazard Rate Model
The stochastic default intensity (hazard rate) $\lambda(t)$ is modeled as:

$$d\lambda(t) = \kappa(\theta - \lambda(t))\,dt + \sigma\sqrt{\lambda(t)}\,dW(t)$$

The closed-form survival probability curve is:

$$P_{\text{surv}}(0, t) = \mathbb{E}\left[\exp\left(-\int_0^t \lambda(s)\,ds\right)\right] = A(t)\exp(-B(t)\lambda_0)$$

where:
$$\gamma = \sqrt{\kappa^2 + 2\sigma^2}, \quad B(t) = \frac{2(e^{\gamma t} - 1)}{(\gamma + \kappa)(e^{\gamma t} - 1) + 2\gamma}, \quad A(t) = \left[ \frac{2\gamma e^{(\kappa + \gamma)t/2}}{(\gamma + \kappa)(e^{\gamma t} - 1) + 2\gamma} \right]^{\frac{2\kappa\theta}{\sigma^2}}$$

#### Path-Wise Credit Valuation Adjustment (CVA)
$$\text{CVA} = \text{LGD} \times \frac{1}{N_{\text{paths}}} \sum_{i=1}^{N_{\text{paths}}} \sum_{j=1}^{N_{\text{dates}}} \text{Exposure}_{i,j} \times \Delta \text{PD}_{i,j} \times D_{i,j}$$

#### Full XVA Suite (DVA / FVA / KVA / MVA)

XvaSim's `xvasim.cva_engine` extends CVA to a complete adjustment ledger. All functions
take simulated exposure, discount factors, and time steps directly and support the same
chunked / `numexpr` accumulation as `compute_cva`:

- **DVA** — own-bank debit valuation adjustment on the negative exposure: `compute_dva(exposure, own_marginal_pd, discount_factor, loss_given_default)`.
- **FVA** — funding valuation adjustment, decomposed into its two symmetric
  legs: `compute_fva(exposure, time_steps_yrs, discount_factor, funding_spread_borrow_ann, funding_spread_deposit_ann)`
  returns `{"fca", "fba", "fva"}` where **FCA** prices the funding cost of cash
  outflows (positive exposure / EE) at the borrowing spread and **FBA** prices
  the funding *benefit* of cash inflows (negative exposure / ENE) at the
  deposit spread.
- **KVA** — capital valuation adjustment: `compute_kva(exposure, time_steps_yrs, discount_factor, capital_charge_ann, regulatory_lgd)` places the regulatory capital charge on EPE.
- **MVA** — initial-margin valuation adjustment: `compute_mva(exposure, time_steps_yrs, discount_factor, funding_spread_borrow_ann, im_scaling, im_percentile)` prices the funding of posted initial margin $\text{IM} = \alpha \times \text{PFE}(p)$.
- **`compute_total_xva`** — single-pass aggregator returning `{"cva", "dva", "fca", "fba", "fva", "kva", "mva", "total_xva"}` with $\text{Total} = \text{CVA} - \text{DVA} + \text{FVA} + \text{KVA} + \text{MVA}$.

All five computations share a common time-discretised form:

$$\text{XVA} = \frac{1}{N_{\text{paths}}} \sum_{i} \sum_{j} \text{RiskExposure}(E_{i,j}) \times \text{Spread}_j \times D_{i,j} \times \Delta t_j$$

---

### 4. Inflation Models

#### Jarrow-Yildirim (JY) Two-Economy Inflation Model
Under the nominal risk-neutral measure $\mathbb{Q}_n$, the nominal rate $r_n(t)$, real rate $r_r(t)$, and CPI index $I(t)$ evolve as a correlated 3-factor system:

$$dr_n(t) = (\theta_n(t) - a_n r_n(t))\,dt + \sigma_n\,dW_n(t)$$

$$dr_r(t) = \left( \theta_r(t) - a_r r_r(t) - \rho_{r, I} \sigma_r \sigma_I \right) dt + \sigma_r\,dW_r(t)$$

$$\frac{dI(t)}{I(t)} = (r_n(t) - r_r(t))\,dt + \sigma_I\,dW_I(t)$$

with cross-correlations $\rho_{n, r}$, $\rho_{n, I}$, and $\rho_{r, I}$.

#### Black CPI Forward Log-Normal Model
Under the $T$-forward nominal measure $\mathbb{Q}_n^T$, the forward CPI index $F_I(t, T) = I(t)\frac{P_r(t, T)}{P_n(t, T)}$ is a martingale:

$$\frac{dF_I(t, T)}{F_I(t, T)} = \sigma_I\,dW_I^T(t)$$

Analytical Zero-Coupon Inflation Swap fair rate:
$$S_0(T) = \left( \frac{P_r(0, T)}{P_n(0, T)} \right)^{1/T} - 1$$

---

## 🚀 High-Performance Acceleration & Hardware Backends

`XvaSim` is engineered for ultra-low latency and scalable enterprise risk workloads through 5 core performance optimizations:

1. **JIT Compilation (`xvasim.jit`)**:
   - Native Numba `@njit(fastmath=True, nogil=True)` compiled kernels for simulation stepping loops: `cir_simulate_paths_kernel`, `lgm_simulate_paths_kernel`, `vasicek_simulate_paths_kernel`, `hull_white_simulate_paths_kernel`, `heston_simulate_paths_kernel`, and `discount_path_kernel`.
   - Transparent fallback when Numba is not installed or when `NUMBA_DISABLE_JIT=1` is set.
2. **Hardware Acceleration (GPU) & Tensor Backend (`xvasim.backend`)**:
   - Unified `TensorBackend` abstract interface supporting **NumPy** (CPU default), **PyTorch** (CPU/CUDA/MPS), **CuPy** (CUDA GPU), and **JAX** (XLA CPU/GPU/TPU).
   - Dynamic backend selection and context scoping via `get_backend()`, `set_backend()`, and `use_backend()`.
3. **Memory Efficiency & Chunked XVA Aggregation (`xvasim.cva_engine`)**:
   - `compute_cva(..., chunk_size=..., use_numexpr=True)` evaluates path-wise CVA in memory-friendly chunks using `numexpr` C-level multi-threaded vector evaluation.
   - `compute_cva_chunked` processes streams/generators of exposure blocks for massive portfolio risk runs exceeding system RAM.
   - `compute_dva`, `compute_fva`, `compute_kva`, `compute_mva`, and `compute_total_xva` reuse the same chunked/numexpr accumulation for the full XVA ledger.
4. **Fast CIR Credit Calibration**:
   - Purely numeric, Numba-compiled objective functions (`cir_calibration_objective_kernel` and `cir_survival_probability_kernel`) evaluated directly over 1D contiguous arrays without repetitive dataclass object allocation in L-BFGS-B iterations.
5. **QMC Sequence Caching & Stateful Generation (`xvasim.qmc`)**:
   - Thread-safe `QMCSequenceCache` with LRU eviction and hit/miss tracking.
   - `QMCSequenceGenerator` maintaining sequential generator state across simulation blocks for fast bump-and-reval Greeks and sensitivities.
   - `cached_normal_draws` and `use_cache=True` parameter on `generate_normal_draws` and `generate_brownian_increments`.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph HardwareBackend ["Hardware Acceleration & Backends (xvasim.backend)"]
        TensorABC["TensorBackend ABC"]
        NPBackend["NumPyBackend (CPU Default)"]
        TorchBackend["PyTorchBackend (CPU / CUDA / MPS)"]
        CuPyBackend["CuPyBackend (CUDA GPU)"]
        JAXBackend["JAXBackend (XLA CPU / GPU / TPU)"]
        
        TensorABC --> NPBackend
        TensorABC --> TorchBackend
        TensorABC --> CuPyBackend
        TensorABC --> JAXBackend
    end

    subgraph JITEngine ["Numba JIT Numerical Acceleration (xvasim.jit)"]
        JITKernels["Compiled Simulation Kernels <br> cir, lgm, vasicek, hull_white, heston, discount_path"]
        JITCalib["Compiled Calibration Objective & Survival Kernels"]
    end

    subgraph QMCEngine ["Quasi-Monte Carlo & Variance Reduction (xvasim.qmc)"]
        QMCSeq["RandomSequenceType (Sobol, Halton, LHS, PRNG)"]
        QMCCache["QMCSequenceCache & QMCSequenceGenerator"]
        QMCGen["Variate Generation (generate_normal_draws, generate_brownian_increments)"]
        QMCBench["Benchmarking (compare_t0_npv_fitting)"]
    end

    subgraph Modular Models ["Modular Stochastic Models (xvasim.models)"]
        Registry[ModelRegistry & Dynamic Factories <br> create_ir_model, create_fx_model, create_credit_model, create_inflation_model]
        IR[InterestRateModel ABC]
        FX[FXModel ABC]
        Credit[CreditModel ABC]
        Inf[InflationModel ABC]
        
        IR --> LGM[LGMModel]
        IR --> HW[HullWhite1FModel]
        IR --> Vas[VasicekModel]
        IR --> CIR_IR[CIRInterestRateModel]
        
        FX --> TwoCurr[TwoCurrencyFXModel]
        FX --> GK[GarmanKohlhagenFXModel]
        FX --> Hest[HestonFXModel]
        
        Credit --> CIR_Credit[CIRHazardRateModel]
        
        Inf --> JY[JarrowYildirimModel]
        Inf --> BlackInf[BlackInflationModel]
    end

    subgraph Pricing Engines ["Pricing & Valuation Engines (xvasim)"]
        MCPricers["Primary Monte Carlo Pricers <br> price_interest_rate_swap, price_cross_currency_swap <br> price_foreign_exchange_forward, price_foreign_exchange_option <br> price_zero_coupon_inflation_swap, price_year_on_year_inflation_swap, price_consumer_price_index_option"]
        Benchmarks["Analytical Benchmarks <br> benchmark_price_interest_rate_swap, benchmark_price_cross_currency_swap <br> benchmark_price_foreign_exchange_forward, benchmark_price_foreign_exchange_option <br> benchmark_price_zero_coupon_inflation_swap, benchmark_price_consumer_price_index_option"]
        CVAEngine["CVA Engine (Chunked & Numexpr) <br> compute_cva, compute_cva_chunked, compute_marginal_pd <br> compute_total_xva (CVA/DVA/FVA/KVA/MVA)"]
    end

    subgraph Portfolio Layer ["Portfolio & Exposure Layer (xvasim.portfolio)"]
        Trades["Trade protocol <br> FXForwardTrade, FXEuropeanOptionTrade"]
        Sim["MarketSimulation <br> simulate_market"]
        PortfolioSet["Portfolio (netting set) <br> compute_portfolio_exposure"]
        XVALedger["compute_portfolio_xva <br> full XVA ledger + exposure profile"]
        Trades --> PortfolioSet
        Sim --> PortfolioSet
        PortfolioSet --> XVALedger
    end

    HardwareBackend --> Modular Models
    HardwareBackend --> MCPricers
    JITEngine --> Modular Models
    JITEngine --> CVAEngine
    QMCEngine --> Modular Models
    QMCEngine --> MCPricers
    Modular Models --> MCPricers
    Modular Models --> Benchmarks
    Modular Models --> CVAEngine
    Modular Models --> Sim
    PortfolioSet --> CVAEngine
    XVALedger --> CVAEngine
```

---

## 📁 Repository & Test Suite Layout

```
XvaSim/
├── pyproject.toml              # Build & dependency config (Hatchling, uv, ruff, pyrefly, pytest, coverage)
├── README.md                   # Human & LLM documentation
├── AGENTS.md                   # Authoritative agentic rules, architecture & standards
├── docs/                       # Living project documentation
│   ├── PROJECT_TRACKER.md      # Project status, milestones, workstreams, backlog
│   └── FIXES_AND_COVERAGE.md   # Fixes log & test coverage detail
├── src/
│   └── xvasim/
│       ├── __init__.py         # Package root exports
│       ├── backend.py          # TensorBackend hardware abstraction (NumPy, PyTorch, CuPy, JAX)
│       ├── cva_engine.py       # CVA calculation, chunked evaluation & credit calibration
│       ├── jit.py              # Numba JIT simulation kernels & numerical routines
│       ├── portfolio.py        # Trades, MarketSimulation, Portfolio netting sets & portfolio XVA
│       ├── pricing_engine.py   # MC & analytical pricing for IR, FX & inflation derivatives
│       ├── qmc.py              # QMC sequences, sequence caching & variance reduction
│       ├── utils.py            # Date conversion (dates_to_years)
│       └── models/             # Modular stochastic models framework
│           ├── __init__.py     # Models package exports
│           ├── base.py         # Base ABCs (RiskFactorType, StochasticModel, InterestRateModel, etc.)
│           ├── registry.py     # ModelRegistry & dynamic factory functions
│           ├── ir/             # Interest rate models (LGM, Hull-White 1F, Vasicek, CIR)
│           ├── credit/         # Credit models (CIR hazard rate)
│           ├── fx/             # FX models (TwoCurrencyFXModel, GarmanKohlhagenFXModel, HestonFXModel)
│           └── inflation/      # Inflation models (JarrowYildirimModel, BlackInflationModel)
└── tests/                      # Full test suite (Dual compatibility: Pytest & Unittest)
    ├── conftest.py             # Shared pytest fixtures & configuration
    ├── helpers/                # Test curve builders & assertion helpers
    │   ├── assertions.py       # Statistical and pricing comparison assertions
    │   └── test_curves.py      # Flat, upward, and downward mock discount curves
    ├── unit/                   # Isolated unit tests
    │   ├── cva/                # CVA calculation & CIR calibration tests
    │   ├── models/             # Model-specific tests (IR, FX, Credit, Inflation, base, registry)
    │   ├── portfolio/          # Portfolio / netting-set exposure & XVA ledger tests
    │   ├── pricing/            # Pricing engine tests (IRS, XCCY, FX, Inflation, internals)
    │   ├── qmc/                # Quasi-Monte Carlo variate & convergence tests
    │   ├── test_backend.py     # Hardware acceleration & tensor backend tests
    │   ├── test_jit.py         # Compiled numerical kernels tests
    │   └── utils/              # Helper utilities tests
    ├── integration/            # Multi-model simulations & portfolio CVA pipeline tests
    └── benchmarks/             # Analytical benchmark vs Monte Carlo convergence tests
```

---

## 📐 Units & Naming Conventions

Strictly enforced across all public APIs and parameters:
*   **Time & Tenors**: Must use suffix `_yrs` (e.g. `maturity_yrs`, `tenors_yrs`, `pay_freq_yrs`). 6 months = `0.5`.
*   **Rates, Volatilities & Spreads**: Must use suffix `_ann` (e.g. `kappa_ann`, `sigma_ann`, `fx_vol_ann`, `cpi_vol_ann`). 2.5% = `0.025`.

---

## 🚀 Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic package management.

### Prerequisites
*   Python >= 3.14
*   `uv` installed on your system.

```bash
# Clone & install dependencies
git clone https://github.com/osp709/XvaSim.git
cd XvaSim
uv sync
```

---

## 💡 Quick Start Examples

### 1. Pricing an Interest Rate Swap (Monte Carlo with Analytical Benchmark)

```python
import numpy as np
from xvasim import (
    HullWhite1FModel,
    benchmark_price_interest_rate_swap,
    price_interest_rate_swap,
)

# 1. Calibrate initial discount curve
tenors_yrs = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
dfs = np.exp(-0.03 * tenors_yrs)

# 2. Instantiate Hull-White 1-Factor model
hw_model = HullWhite1FModel(
    a_ann=0.03,
    sigma_ann=0.01,
    discount_curve_yrs=tenors_yrs,
    discount_factors=dfs,
)

# 3. Price 5-year IRS via Monte Carlo (default 10,000 paths)
mc_result = price_interest_rate_swap(
    model=hw_model,
    fixed_rate_ann=0.03,
    tenor_yrs=5.0,
    pay_freq_yrs=0.5,
    notional=10_000_000.0,
    is_payer=True,
    n_paths=50_000,
    seed=42,
)

# 4. Compute closed-form analytical benchmark
bench_result = benchmark_price_interest_rate_swap(
    model=hw_model,
    fixed_rate_ann=0.03,
    tenor_yrs=5.0,
    pay_freq_yrs=0.5,
    notional=10_000_000.0,
    is_payer=True,
)

print(f"Monte Carlo Swap PV:    ${mc_result['price']:,.2f} ± ${mc_result['std_error']:,.2f}")
print(f"Analytical Benchmark:   ${bench_result['price']:,.2f}")
print(f"Fair Par Swap Rate:     {mc_result['fair_swap_rate']:.4%}")
print(f"Forward Annuity (PV01): {mc_result['annuity']:,.4f}")
```

### 2. Pricing a Cross-Currency Swap (XCCY)

```python
import numpy as np
from xvasim import (
    HullWhite1FModel,
    SwapLegType,
    TwoCurrencyFXModel,
    price_cross_currency_swap,
)

tenors_yrs = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
dom_dfs = np.exp(-0.03 * tenors_yrs)
for_dfs = np.exp(-0.015 * tenors_yrs)

dom_ir = HullWhite1FModel(a_ann=0.03, sigma_ann=0.01, discount_curve_yrs=tenors_yrs, discount_factors=dom_dfs)
for_ir = HullWhite1FModel(a_ann=0.02, sigma_ann=0.008, discount_curve_yrs=tenors_yrs, discount_factors=for_dfs)

fx_model = TwoCurrencyFXModel(
    domestic_ir_model=dom_ir,
    foreign_ir_model=for_ir,
    spot_fx=1.20,
    fx_vol_ann=0.10,
    correlation_matrix=np.array([
        [ 1.0,  0.3, -0.1],
        [ 0.3,  1.0,  0.2],
        [-0.1,  0.2,  1.0],
    ]),
)

xccy_res = price_cross_currency_swap(
    model=fx_model,
    domestic_rate_ann=0.03,
    foreign_spread_ann=0.0,
    domestic_leg_type=SwapLegType.FIXED,
    foreign_leg_type=SwapLegType.FLOATING,
    tenor_yrs=5.0,
    foreign_notional=1_000_000.0,
    exchange_notionals=True,
    n_paths=30_000,
    seed=42,
)

print(f"XCCY Simulated PV (USD): ${xccy_res['price']:,.2f} ± ${xccy_res['std_error']:,.2f}")
print(f"Analytical Benchmark:    ${xccy_res['analytical_benchmark_price']:,.2f}")
print(f"Fair Foreign Spread:     {xccy_res['fair_foreign_spread'] * 10_000:.2f} bps")
```

### 3. Pricing Inflation Derivatives (ZCIS & CPI Options)

```python
import numpy as np
from xvasim import (
    BlackInflationModel,
    OptionType,
    price_consumer_price_index_option,
    price_zero_coupon_inflation_swap,
)

tenors_yrs = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0])
nom_dfs = np.exp(-0.035 * tenors_yrs)
real_dfs = np.exp(-0.015 * tenors_yrs)

inf_model = BlackInflationModel(
    nominal_discount_curve_yrs=tenors_yrs,
    nominal_discount_factors=nom_dfs,
    real_discount_curve_yrs=tenors_yrs,
    real_discount_factors=real_dfs,
    base_cpi=100.0,
    cpi_vol_ann=0.015,
)

# 1. Zero-Coupon Inflation Swap
zcis_res = price_zero_coupon_inflation_swap(
    model=inf_model,
    strike_rate_ann=0.02,
    maturity_yrs=5.0,
    notional=1_000_000.0,
    is_payer=True,
)
print(f"ZCIS Monte Carlo PV:    ${zcis_res['price']:,.2f} (Benchmark: ${zcis_res['analytical_benchmark_price']:,.2f})")
print(f"Par Inflation Rate:     {zcis_res['fair_swap_rate']:.4%}")

# 2. CPI Inflation Caplet Option
cpi_opt = price_consumer_price_index_option(
    model=inf_model,
    strike_rate_ann=0.02,
    maturity_yrs=3.0,
    notional=100_000.0,
    option_type=OptionType.CALL,
)
print(f"CPI Caplet Simulated PV: ${cpi_opt['price']:,.2f} (Benchmark: ${cpi_opt['analytical_benchmark_price']:,.2f})")
```

### 4. Portfolio CVA Simulation with CIR Credit Model

```python
import numpy as np
from xvasim import (
    compute_cva,
    compute_exposure_profile,
    compute_marginal_pd,
    dates_to_years,
)

valuation_date = "2026-07-11"
dates = ["2027-07-11", "2028-07-11", "2029-07-11", "2031-07-11", "2033-07-11", "2036-07-11"]
tenors_yrs = dates_to_years(dates, valuation_date)
credit_spreads_ann = np.array([0.0150, 0.0180, 0.0210, 0.0250, 0.0270, 0.0300])

# 1. Calibrate CIR hazard rate model & marginal default probabilities
marginal_pds = compute_marginal_pd(credit_spreads_ann, tenors_yrs)

# 2. Generate simulated exposure paths
n_paths = 5_000
n_dates = len(tenors_yrs)
np.random.seed(42)
exposure = np.maximum(np.random.normal(loc=50_000.0, scale=15_000.0, size=(n_paths, n_dates)), 0.0)
discount_factor = np.tile(np.exp(-0.03 * tenors_yrs), (n_paths, 1))
marginal_pd_matrix = np.tile(marginal_pds, (n_paths, 1))

# 3. Compute Exposure Profiles (EE, EPE, PFE percentiles)
exp_profile = compute_exposure_profile(exposure, percentiles=(95.0, 99.0))
print(f"Expected Positive Exposure (EPE): ${exp_profile['epe']:,.2f}")
print(f"Peak Potential Future Exposure (Max PFE): ${exp_profile['max_pfe']:,.2f}")

# 4. Compute Portfolio CVA
cva = compute_cva(
    exposure=exposure,
    marginal_pd=marginal_pd_matrix,
    discount_factor=discount_factor,
    loss_given_default=0.60,
)
print(f"Calculated Portfolio CVA: ${cva:,.2f}")

# 5. Full XVA ledger (DVA / FVA / KVA / MVA + total) in a single pass
from xvasim import compute_total_xva

dt = np.diff(np.concatenate(([0.0], tenors_yrs)))
own_marginal_pd = compute_marginal_pd(credit_spreads_ann * 0.6, tenors_yrs)
own_pd_matrix = np.tile(own_marginal_pd, (n_paths, 1))

total_xva = compute_total_xva(
    exposure=exposure,
    time_steps_yrs=dt,
    discount_factor=discount_factor,
    counterparty_marginal_pd=marginal_pd_matrix,
    own_marginal_pd=own_pd_matrix,
    counterparty_lgd=0.60,
    own_lgd=0.55,
    funding_spread_borrow_ann=0.004,
    capital_charge_ann=0.08,
)
print(f"Total XVA: ${total_xva['total_xva']:,.2f} "
      f"(CVA ${total_xva['cva']:,.2f}, DVA ${total_xva['dva']:,.2f}, "
      f"FVA ${total_xva['fva']:,.2f}, KVA ${total_xva['kva']:,.2f}, MVA ${total_xva['mva']:,.2f})")
```

### 5. Quasi-Monte Carlo (QMC) Variance Reduction Benchmarking

```python
from xvasim import (
    GarmanKohlhagenFXModel,
    OptionType,
    RandomSequenceType,
    compare_t0_npv_fitting,
    price_foreign_exchange_option,
)

gk_model = GarmanKohlhagenFXModel(
    spot_fx=1.20, domestic_rate_ann=0.03, foreign_rate_ann=0.015, fx_vol_ann=0.12
)

# Benchmark pricing variance across PRNG, Sobol, and Halton
comparison = compare_t0_npv_fitting(
    pricer_fn=price_foreign_exchange_option,
    pricer_kwargs={
        "params": gk_model,
        "strike": 1.20,
        "maturity_yrs": 1.0,
        "notional": 100_000.0,
        "option_type": OptionType.CALL,
        "n_steps": 1,
    },
    methods=(
        RandomSequenceType.PSEUDO,
        RandomSequenceType.SOBOL,
        RandomSequenceType.HALTON,
    ),
    n_paths=4096,
    seeds=(10, 20, 30, 40, 50),
)

for method, stats in comparison["methods"].items():
    vrf = stats.get("variance_reduction_factor", 1.0)
    print(f"{method:>8}: Variance={stats['variance']:.4f}, VRF={vrf:.1f}x, MAE={stats['mean_absolute_error']:.4f}")
```

### 6. Hardware Acceleration & Multi-Device Tensor Backends

```python
import numpy as np
from xvasim import (
    available_backends,
    get_backend,
    is_backend_available,
    use_backend,
)

print(f"Available Backends: {[b.value for b in available_backends()]}")

# 1. Inspect current active backend
backend = get_backend()
print(f"Default Active Backend: {backend.name.value} on {backend.device}")

# 2. Execute risk calculations in scoped PyTorch or GPU context (if installed)
if is_backend_available("torch"):
    with use_backend("torch", device="cuda:0") as gpu_backend:
        data = gpu_backend.linspace(0.0, 5.0, 10)
        exp_data = gpu_backend.exp(data)
        print(f"Executed on GPU backend: {gpu_backend.device}")
```

### 7. Stateful QMC Generator & Cached Sequence Sensitivity Analysis

```python
import numpy as np
from xvasim import (
    QMCSequenceGenerator,
    RandomSequenceType,
    cached_normal_draws,
    clear_qmc_cache,
)

# 1. Draw cached normal variates for fast bump-and-reval Greeks
draws_base = cached_normal_draws(n_paths=10_000, dimension=5, random_type="sobol", seed=42)
draws_bumped = cached_normal_draws(n_paths=10_000, dimension=5, random_type="sobol", seed=42)
assert draws_base is not draws_bumped  # Isolated copies returned from cache

# 2. Stateful sequence progression across simulation batches
generator = QMCSequenceGenerator(dimension=4, random_type=RandomSequenceType.SOBOL, seed=42)
batch_1 = generator.draw(n_paths=2048)
batch_2 = generator.draw(n_paths=2048)
print(f"Total points drawn consecutively: {generator.total_drawn}")
clear_qmc_cache()
```

### 8. Memory-Efficient Streaming CVA on Massive Portfolios

```python
import numpy as np
from xvasim import compute_cva_chunked

n_dates = 10
marginal_pd = np.full(n_dates, 0.005)
discount_factor = np.exp(-0.03 * np.linspace(0.5, 5.0, n_dates))

# Generator streaming 100,000 paths in memory-friendly 10,000-path chunks
def stream_portfolio_exposure():
    rng = np.random.default_rng(42)
    for _ in range(10):
        yield np.maximum(rng.standard_normal((10_000, n_dates)) * 100_000.0, 0.0)

# Compute aggregated CVA across all 100k paths without RAM overflow
streamed_cva = compute_cva_chunked(
    exposure_chunks=stream_portfolio_exposure(),
    marginal_pd=marginal_pd,
    discount_factor=discount_factor,
    loss_given_default=0.60,
    use_numexpr=True,
)
print(f"Streaming Portfolio CVA: ${streamed_cva:,.2f}")
```

### 9. Model-Agnostic Calibration & Decoupled Multi-Factor Models

```python
import numpy as np
from xvasim import (
    HullWhite1FModel,
    LGMModel,
    TwoCurrencyFXModel,
    calibrate_ir_model_to_swaptions,
)

tenors_yrs = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
dfs = np.exp(-0.03 * tenors_yrs)

# 1. Model-agnostic interest rate calibration to ATM swaption normal volatilities
calibrated_params = calibrate_ir_model_to_swaptions(
    swaption_expiries_yrs=np.array([1.0, 2.0, 5.0]),
    swap_tenors_yrs=np.array([5.0, 5.0, 5.0]),
    market_normal_vols_ann=np.array([0.0080, 0.0085, 0.0090]),
    curve_yrs=tenors_yrs,
    curve_dfs=dfs,
    fixed_rates_ann=np.array([0.03, 0.03, 0.03]),
    kappa_ann=0.03,
    model_type="lgm",
)
lgm_model = LGMModel(calibrated_params)

# 2. Decoupled multi-factor FX model using arbitrary interest rate components
foreign_hw = HullWhite1FModel(a_ann=0.02, sigma_ann=0.008, discount_curve_yrs=tenors_yrs, discount_factors=dfs)
fx_model = TwoCurrencyFXModel.from_ir_models(
    domestic=lgm_model,
    foreign=foreign_hw,
    spot_fx=1.25,
    fx_vol_ann=0.11,
    correlation_matrix=np.eye(3),
)
```

### 10. Portfolio & Netting-Set XVA (FX)

```python
import numpy as np
from xvasim import (
    FXEuropeanOptionTrade,
    FXForwardTrade,
    GarmanKohlhagenFXModel,
    Portfolio,
    compute_portfolio_exposure,
    compute_portfolio_xva,
    simulate_market,
)

# 1. FX market model (Garman-Kohlhagen, deterministic curves)
fx_model = GarmanKohlhagenFXModel(
    spot_fx=1.20, fx_vol_ann=0.15,
    domestic_rate_ann=0.03, foreign_rate_ann=0.02,
)

# 2. Netting set of FX trades (forex notionals make all values in domestic ccy)
portfolio = Portfolio(
    portfolio_id="fx_book",
    trades=[
        FXForwardTrade("fwd_long", 1_000_000.0, strike_fx=1.22, maturity_yrs=3.0),
        FXForwardTrade("fwd_short", -300_000.0, strike_fx=1.19, maturity_yrs=2.0),
        FXEuropeanOptionTrade("call", 500_000.0, strike_fx=1.18, maturity_yrs=2.0,
                              option_type="call"),
        FXEuropeanOptionTrade("put", 500_000.0, strike_fx=1.24, maturity_yrs=4.0,
                              option_type="put"),
    ],
)

# 3. Shared market simulation -> netting-set exposure (positive exposure profile)
sim = simulate_market(fx_model, maturity_yrs=5.0, n_steps=10, n_paths=10_000,
                      random_type="sobol", seed=42)
exposure_res = compute_portfolio_exposure(portfolio, sim)
print(exposure_res.exposure.shape)      # (n_paths, n_dates)

# 4. Full XVA ledger in one pass: CVA/DVA/FVA (FCA/FBA)/KVA/MVA + exposure profile
credit_tenors_yrs = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
cp_spreads = np.array([0.010, 0.012, 0.016, 0.020, 0.025])
own_spreads = cp_spreads * 0.6
ledger = compute_portfolio_xva(
    portfolio=portfolio,
    fx_model=fx_model,
    maturity_yrs=5.0,
    n_steps=10,
    counterparty_credit_spreads_ann=cp_spreads,
    own_credit_spreads_ann=own_spreads,
    credit_tenors_yrs=credit_tenors_yrs,
    n_paths=10_000,
    random_type="sobol",
    seed=42,
)
print({k: round(v, 2) for k, v in ledger.items()
       if k in ("cva", "dva", "fva", "kva", "mva", "total_xva")})
```

---

## 🧪 Testing & Verification Commands

```bash
# Run full unit & benchmark test suite via Pytest
uv run pytest tests/ -v

# Run full test suite via standard Unittest
uv run python -m unittest discover tests

# Measure test coverage with strict 95% threshold enforcement
uv run coverage run -m pytest tests/
uv run coverage report -m

# Linting and formatting checks (Ruff)
uv run ruff check .

# Strict static type checking (Pyrefly on Python 3.14)
uv run pyrefly check
```

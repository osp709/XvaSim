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
| **Vanilla Interest Rate Swap (IRS)** | Rates | `price_interest_rate_swap` / `price_irs` | `benchmark_price_interest_rate_swap` / `benchmark_price_irs` | Domestic Interest Rate $r_d(t)$ | Single-currency fixed-for-floating interest rate swap. Pays fixed/floating coupons on preset schedule. Computes par swap rate & forward annuity (PV01). |
| **Cross-Currency Swap (XCCY)** | Rates / FX | `price_cross_currency_swap` / `price_xccy_swap` | `benchmark_price_cross_currency_swap` / `benchmark_price_xccy_swap` | Domestic IR $r_d(t)$, Foreign IR $r_f(t)$, Spot FX $S(t)$ | Multi-currency swap supporting fixed-for-floating, fixed-for-fixed, and floating-for-floating basis swaps with optional principal notional exchanges. |
| **FX Forward** | FX | `price_fx_forward` | `benchmark_price_fx_forward` | Domestic IR $r_d(t)$, Foreign IR $r_f(t)$, Spot FX $S(t)$ | Forward exchange contract: $N \times (S(T) - K)$ priced under domestic risk-neutral measure with Covered Interest Parity (CIP) benchmark. |
| **European FX Option (Call / Put)** | FX | `price_fx_option` | `benchmark_price_fx_option` | Spot FX $S(t)$, Domestic IR $r_d(t)$, Foreign IR $r_f(t)$, FX Variance $v(t)$ (Heston) | Vanilla European option on FX spot: $N \times \max(\omega(S(T) - K), 0)$ benchmarked against Garman-Kohlhagen / Black-76 / Heston semi-analytical formulas. |
| **Zero-Coupon Inflation Swap (ZCIS)** | Inflation | `price_zero_coupon_inflation_swap` | `benchmark_price_zero_coupon_inflation_swap` | Nominal IR $r_n(t)$, Real IR $r_r(t)$, CPI Index $I(t)$ | Single-exchange swap at maturity: pays fixed compounded rate $(1+K)^T - 1$ vs floating realized inflation index return $I(T)/I(0) - 1$. |
| **Year-on-Year Inflation Swap (YoY)** | Inflation | `price_yoy_inflation_swap` | Interpolated CPI forward projection | Nominal IR $r_n(t)$, Real IR $r_r(t)$, CPI Index $I(t)$ | Multi-period swap exchanging annual fixed rate $K$ for annual CPI growth $\frac{I(T_i)}{I(T_{i-1})} - 1$ on each reset date. |
| **CPI Index Option (Caplet / Floorlet)** | Inflation | `price_cpi_option` | `benchmark_price_cpi_option` | Nominal IR $r_n(t)$, Real IR $r_r(t)$, CPI Index $I(t)$ | European option on inflation index: Caplet $N \times \max\left(\frac{I(T)}{I(0)} - (1+K)^T, 0\right)$ and Floorlet $N \times \max\left((1+K)^T - \frac{I(T)}{I(0)}, 0\right)$. |
| **Portfolio Credit Valuation Adjustment (CVA)** | Credit / Multi-Asset | `compute_cva` | Marginal PD via CIR zero-curve | Credit Hazard Rate $\lambda(t)$, Underlying Portfolio Exposure | Path-wise Monte Carlo integration of counterparty default risk across simulated market exposure paths, discount factors, and marginal default probabilities. |

---

## 🎲 Supported Risk Factors & Stochastic Models

`XvaSim` features a modular dynamic registry (`ModelRegistry`) and factories (`create_ir_model`, `create_credit_model`, `create_fx_model`, `create_inflation_model`) allowing plug-and-play selection of stochastic models for each simulated risk factor:

| Risk Factor | Supported Stochastic Models | Model Registry Key | Concrete Class | Key Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Domestic / Foreign Interest Rate** | Linear Gauss-Markov (LGM) | `"lgm"` | `LGMModel` | $\kappa_{\text{ann}}$, $\sigma(t)_{\text{ann}}$, Discount Curve |
| | Hull-White 1-Factor (HW1F) | `"hull_white"` | `HullWhite1FModel` | $a_{\text{ann}}$, $\sigma_{\text{ann}}$, Discount Curve |
| | Vasicek Short Rate | `"vasicek"` | `VasicekModel` | $\kappa_{\text{ann}}$, $\theta_{\text{ann}}$, $\sigma_{\text{ann}}$, $r_0$ |
| | Cox-Ingersoll-Ross (CIR) | `"cir"` | `CIRInterestRateModel` | $\kappa_{\text{ann}}$, $\theta_{\text{ann}}$, $\sigma_{\text{ann}}$, $r_0$ |
| **Foreign Exchange (FX Spot & Volatility)** | Two-Currency Multi-Factor FX | `"two_currency"` | `TwoCurrencyFXModel` | Domestic IR Model, Foreign IR Model, $S_0$, $\sigma_{\text{fx}}$, Correlation Matrix |
| | Garman-Kohlhagen / Black FX | `"garman_kohlhagen"` | `GarmanKohlhagenFXModel` | $r_d$, $r_f$, $S_0$, $\sigma_{\text{fx}}$ |
| | Heston Stochastic Volatility FX | `"heston"` | `HestonFXModel` | $r_d$, $r_f$, $S_0$, $v_0$, $\kappa_v$, $\theta_v$, $\sigma_v$, $\rho_{S,v}$ |
| **Counterparty Credit / Hazard Rate** | Cox-Ingersoll-Ross (CIR) Hazard Rate | `"cir"` | `CIRHazardRateModel` | $\kappa_{\text{ann}}$, $\theta_{\text{ann}}$, $\sigma_{\text{ann}}$, $\lambda_0$ |
| **Inflation (CPI Index & Real Rates)** | Jarrow-Yildirim (JY) Two-Economy | `"jarrow_yildirim"` | `JarrowYildirimModel` | Nominal HW1F, Real HW1F, $I_0$, $\sigma_I$, $3 \times 3$ Correlation Matrix |
| | Black CPI Forward Log-Normal | `"black_inflation"` | `BlackInflationModel` | Nominal Curve, Real Curve, $I_0$, $\sigma_{I,\text{ann}}$ |

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
3. **Memory Efficiency & Streaming CVA Aggregation (`xvasim.cva_engine`)**:
   - `compute_cva(..., chunk_size=..., use_numexpr=True)` evaluates path-wise CVA in memory-friendly chunks using `numexpr` C-level multi-threaded vector evaluation.
   - `compute_cva_chunked` processes streams/generators of exposure blocks for massive portfolio risk runs exceeding system RAM.
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
        CVAEngine["CVA Engine (Chunked & Numexpr) <br> compute_cva, compute_cva_chunked, compute_marginal_pd"]
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
```

---

## 📁 Repository & Test Suite Layout

```
XvaSim/
├── pyproject.toml              # Build & dependency config (Hatchling, uv, ruff, pyrefly, pytest, coverage)
├── README.md                   # Human & LLM documentation
├── GEMINI.md                   # AI assistant context & developer guide
├── src/
│   └── xvasim/
│       ├── __init__.py         # Package root exports
│       ├── backend.py          # TensorBackend hardware abstraction (NumPy, PyTorch, CuPy, JAX)
│       ├── cva_engine.py       # CVA calculation, chunked evaluation & credit calibration
│       ├── jit.py              # Numba JIT simulation kernels & numerical routines
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
fx_model = TwoCurrencyFXModel.from_components(
    domestic=lgm_model,
    foreign=foreign_hw,
    spot_fx=1.25,
    fx_vol_ann=0.11,
    correlation_matrix=np.eye(3),
)
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

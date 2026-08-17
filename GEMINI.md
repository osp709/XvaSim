# GEMINI.md: LLM & AI Assistant Guide for XvaSim

This document provides system instructions, architectural context, mathematical models, and development rules for Gemini and other AI coding assistants working on `XvaSim`.

---

## 🏛️ Project Identity & Core Capabilities

`XvaSim` is a lightweight, high-performance Python library designed for simulating and calculating credit and valuation adjustments (XVAs) and pricing multi-asset derivatives across Interest Rates, Foreign Exchange (FX), Credit, and Inflation risk factors.

### Core Features
1. **Modular Stochastic Models Framework** (`xvasim.models`):
   - Pluggable stochastic modeling architecture for Interest Rates, FX, Credit, and Inflation risk factors.
   - Central dynamic registry and factory: `ModelRegistry`, `create_ir_model`, `create_credit_model`, `create_fx_model`, `create_inflation_model`.
   - Concrete IR models: Linear Gauss-Markov (`LGMModel`), Hull-White 1-Factor (`HullWhite1FModel`), Vasicek (`VasicekModel`), Cox-Ingersoll-Ross (`CIRInterestRateModel`).
   - Concrete Credit models: CIR Hazard Rate (`CIRHazardRateModel`).
   - FX models: Two-Currency FX (`TwoCurrencyFXModel`), Garman-Kohlhagen / Black-Scholes (`GarmanKohlhagenFXModel`), Heston Stochastic Volatility (`HestonFXModel`).
   - Inflation models: Jarrow-Yildirim Two-Economy (`JarrowYildirimModel`), Black CPI Forward (`BlackInflationModel`).
2. **Credit Valuation Adjustment (CVA)** (`xvasim.cva_engine`):
   - CIR and modular credit model spread calibration (L-BFGS-B).
   - Path-wise Monte Carlo CVA aggregation across exposure paths, discount factors, and marginal default probabilities.
3. **Derivative Pricing Engine** (`xvasim.pricing_engine`):
   - **Primary Monte Carlo Pricers**: All `price_*` functions simulate paths by default, returning simulated price, standard error, and analytical benchmark price. Fully expanded canonical names are standard across all asset classes with backwards-compatible aliases.
   - **Analytical Benchmarks**: Dedicated `benchmark_price_*` functions provide exact closed-form benchmark solutions to validate Monte Carlo simulations.
   - Single-currency Interest Rate Swaps (`price_interest_rate_swap` / `price_irs`, `benchmark_price_interest_rate_swap` / `benchmark_price_irs`).
   - Cross-Currency Swaps (`price_cross_currency_swap` / `price_xccy_swap`, `benchmark_price_cross_currency_swap` / `benchmark_price_xccy_swap`).
   - Foreign Exchange forwards & options (`price_foreign_exchange_forward` / `price_fx_forward`, `price_foreign_exchange_option` / `price_fx_option`, `benchmark_price_foreign_exchange_forward` / `benchmark_price_fx_forward`, `benchmark_price_foreign_exchange_option` / `benchmark_price_fx_option`).
   - Inflation derivatives (`price_zero_coupon_inflation_swap`, `price_year_on_year_inflation_swap` / `price_yoy_inflation_swap`, `price_consumer_price_index_option` / `price_cpi_option`, `benchmark_price_zero_coupon_inflation_swap`, `benchmark_price_consumer_price_index_option` / `benchmark_price_cpi_option`).
4. **Quasi-Monte Carlo (QMC) & Variance Reduction** (`xvasim.qmc`):
   - Low-discrepancy generators: `RandomSequenceType` (`SOBOL`, `HALTON`, `LATIN_HYPERCUBE`, `PSEUDO`).
   - Variate generation: `generate_normal_draws` (scrambled inverse-CDF) and `generate_brownian_increments`.
   - Convergence diagnostics: `compare_t0_npv_fitting` for T0 NPV accuracy and variance reduction factor (VRF) benchmarking.
   - Thread-safe `QMCSequenceCache` and stateful `QMCSequenceGenerator` for fast Greek bump-and-reval sensitivities.
   - Uniform `random_type`, `seed`, `scramble`, and `use_cache` parameters across all models and pricers.
5. **High-Performance JIT & Hardware Acceleration** (`xvasim.jit`, `xvasim.backend`):
   - Numba `@njit(fastmath=True, nogil=True)` compiled kernels for simulation stepping and discount integration.
   - Unified `TensorBackend` abstract interface supporting NumPy (CPU), PyTorch (CPU/CUDA/MPS), CuPy (CUDA), and JAX (CPU/GPU/TPU).
   - Scoped backend execution with `use_backend`, `set_backend`, and `get_backend`.

---

## 📐 Units & Naming Conventions

Strictly adhere to the following naming conventions across all public APIs, helper functions, and data structures:
*   **Time & Tenors**: Must use suffix `_yrs` (e.g. `maturity_yrs`, `tenors_yrs`, `expiry_yrs`, `pay_freq_yrs`). 6 months = `0.5`.
*   **Rates, Volatilities & Spreads**: Must use suffix `_ann` (e.g. `kappa_ann`, `sigma_ann`, `credit_spreads_ann`, `fx_vol_ann`, `cpi_vol_ann`). 2.5% = `0.025`.

---

## 📁 Repository Layout

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

## 🛠️ Coding & Quality Standards

### 1. Strict Typing (Pyrefly)
- Pyrefly runs type checking on Python 3.14 (`uv run pyrefly check`).
- All functions, methods, parameters, and return types MUST have explicit type annotations.
- Use `@typing.overload` / `@overload` for polymorphic or dual-signature interfaces.
- Avoid loose `Any` return types; cast numpy array conversions with `np.asarray(..., dtype=np.float64)` where necessary.
- In `__init__.py` files, keep `__all__` sorted alphabetically.

### 2. Linting (Ruff)
- Line length limit: **88** characters.
- Python target: `py314`.
- PEP 8 naming (`N806`): Function-local variables must be lowercase.
- Clean imports: Never leave unused imports (`F401`).
- Active rule categories: `E`, `W`, `F`, `I`, `N`, `UP`, `B`, `C4`, `RUF`.

### 3. Test Suite Architecture & Coverage
- Dual runner support: All tests must pass under both `pytest` and standard Python `unittest`.
- Minimum coverage requirement: **95%** strict enforcement via `tool.coverage.report.fail_under = 95`.
- Pytest markers defined in `pyproject.toml`:
  - `@pytest.mark.unit`: Fast, isolated unit tests.
  - `@pytest.mark.integration`: Multi-asset or end-to-end pipeline tests.
  - `@pytest.mark.benchmark`: Analytical vs Monte Carlo convergence benchmarks.
  - `@pytest.mark.slow`: Long-running simulations with large path counts.

### 4. Backwards Compatibility
- Maintain backwards compatibility with legacy parameter dataclasses (`FXLGMParams`, `LGMParams`, `CIRParams`).
- Overloaded methods and polymorphism should seamlessly accept both legacy dataclasses and new modular classes.
- Provide convenience aliases alongside canonical expanded names (e.g. `price_irs` for `price_interest_rate_swap`).

---

## 🧮 Mathematical Reference & Supported Models

### 1. Interest Rate Models
*   **LGM (Linear Gauss-Markov)**: Zero-mean Gaussian state $dx(t) = -\kappa x(t) dt + \sigma(t) dW(t)$, analytical discount bond $P(t, T) = \frac{P(0,T)}{P(0,t)} \exp\left(-H(T)x(t) - \frac{1}{2}(H(T)^2 - H(t)^2)\zeta(t)\right)$.
*   **Hull-White 1-Factor (HW1F)**: $dr(t) = (\theta(t) - a r(t))dt + \sigma dW(t)$ with exact term-structure fitting to initial discount curve.
*   **Vasicek**: Mean-reverting Gaussian diffusion $dr(t) = \kappa(\theta - r(t))dt + \sigma dW(t)$.
*   **CIR**: Mean-reverting square-root diffusion $dr(t) = \kappa(\theta - r(t))dt + \sigma\sqrt{r(t)} dW(t)$ with Feller condition $2\kappa\theta \ge \sigma^2$.

### 2. FX Models
*   **Garman-Kohlhagen (Black FX)**: $\frac{dS(t)}{S(t)} = (r_d - r_f)dt + \sigma_{\text{fx}} dW(t)$, with closed-form European call/put pricing.
*   **Heston Stochastic Volatility**: Couples spot FX SDE with variance SDE $dv(t) = \kappa_v(\theta_v - v(t))dt + \sigma_v\sqrt{v(t)} dW_v(t)$ and correlation $\rho_{S, v}$.
*   **Two-Currency Multi-Factor**: Couples domestic and foreign stochastic short rates ($r_d, r_f$) with spot FX $S(t)$ and quanto drift correction.

### 3. Credit & CVA Models
*   **CIR Hazard Rate**: Stochastic default intensity $d\lambda(t) = \kappa(\theta - \lambda(t))dt + \sigma\sqrt{\lambda(t)} dW(t)$ with analytical survival curve $P_{\text{surv}}(0, t) = A(t)\exp(-B(t)\lambda_0)$.
*   **Portfolio CVA**: Path-wise Monte Carlo integration $\text{CVA} = \text{LGD} \times \frac{1}{N_{\text{paths}}} \sum_{i} \sum_{j} \text{Exposure}_{i,j} \times \Delta\text{PD}_{i,j} \times D_{i,j}$.

### 4. Inflation Models
*   **Jarrow-Yildirim (JY)**: 3-factor coupled system under nominal measure $\mathbb{Q}_n$ for nominal rate $r_n(t)$, real rate $r_r(t)$ (with foreign quanto drift), and CPI index $I(t)$.
*   **Black CPI Forward**: Martingale forward CPI index $\frac{dF_I(t, T)}{F_I(t, T)} = \sigma_I dW_I^T(t)$ with closed-form fair ZCIS rate $S_0(T) = (P_r(0, T) / P_n(0, T))^{1/T} - 1$.

---

## 🧩 How to Add a New Model / Risk Factor

1. **Subclass the Base ABC**:
   - For interest rates: Subclass `InterestRateModel` in `src/xvasim/models/ir/<model_name>.py`.
   - Implement: `discount_curve_yrs`, `discount_factors`, `short_rate`, `zero_coupon_bond`, `discount_path`, `simulate_paths`.
   - For credit: Subclass `CreditModel` in `src/xvasim/models/credit/<model_name>.py`.
   - For FX / Inflation: Subclass `FXModel` / `InflationModel` in their respective subpackages.
   - For new asset classes: Add to `RiskFactorType` in `base.py` and define the base class.
2. **Register with ModelRegistry**:
   - Decorate the model class with `@ModelRegistry.register("<risk_factor_type>", "<model_name>")`.
3. **Export Cleanly**:
   - Export in subpackage `__init__.py`, `models/__init__.py`, and root `xvasim/__init__.py`. Keep `__all__` sorted alphabetically.
4. **Add Comprehensive Tests**:
   - Unit tests in `tests/unit/models/<category>/test_<model_name>.py` (initialization, zero-coupon bond pricing vs discount curve, path simulation, terminal distributions).
   - Multi-asset or cross-model tests in `tests/integration/`.
   - Analytical vs MC convergence tests in `tests/benchmarks/`.

---

## 🧪 Verification & Development Commands

```bash
# Install dependencies
uv sync

# Run full test suite via Pytest
uv run pytest tests/ -v

# Run full test suite via Unittest
uv run python -m unittest discover tests

# Measure test coverage with strict 95% threshold enforcement
uv run coverage run -m pytest tests/
uv run coverage report -m

# Linting checks
uv run ruff check .

# Static type checking (Pyrefly on Python 3.14)
uv run pyrefly check
```

---

## 🚀 High-Performance Optimizations & Acceleration

All 5 core performance optimizations are implemented in the codebase:

1. **JIT Compilation (`xvasim.jit`)**:
   - Native Numba `@njit(fastmath=True, nogil=True)` compiled kernels for simulation stepping loops: `cir_simulate_paths_kernel`, `lgm_simulate_paths_kernel`, `vasicek_simulate_paths_kernel`, `hull_white_simulate_paths_kernel`, `heston_simulate_paths_kernel`, and `discount_path_kernel`.
   - Transparent non-JIT fallback when Numba is not installed or when `NUMBA_DISABLE_JIT=1` is set.
2. **Memory Efficiency in CVA Aggregation (`xvasim.cva_engine`)**:
   - `compute_cva(..., chunk_size=..., use_numexpr=True)` evaluates path-wise CVA in memory-friendly chunks using `numexpr` C-level multi-threaded vector evaluation.
   - `compute_cva_chunked` processes streams/generators of exposure blocks for massive portfolio risk runs exceeding system RAM.
3. **Calibration Speedups (`_calibrate_cir` & `CIRHazardRateModel`)**:
   - Purely numeric, Numba-compiled objective functions (`cir_calibration_objective_kernel` and `cir_survival_probability_kernel`) evaluated directly over 1D arrays without repetitive dataclass object allocation in L-BFGS-B iterations.
4. **QMC Sequence Caching & Stateful Generation (`xvasim.qmc`)**:
   - Thread-safe `QMCSequenceCache` with LRU eviction and hit/miss tracking.
   - `QMCSequenceGenerator` maintaining sequential generator state across simulation blocks for fast bump-and-reval Greeks and sensitivities.
   - `cached_normal_draws` and `use_cache=True` parameter on `generate_normal_draws` and `generate_brownian_increments`.
5. **Hardware Acceleration (GPU) & Tensor Backend Abstraction (`xvasim.backend`)**:
   - Unified `TensorBackend` abstract interface with concrete backends: `NumPyBackend` (CPU default), `PyTorchBackend` (CPU/CUDA/MPS), `CuPyBackend` (CUDA), and `JAXBackend` (XLA CPU/GPU/TPU).
   - Dynamic backend selection and context scoping via `get_backend()`, `set_backend()`, and `use_backend()`.

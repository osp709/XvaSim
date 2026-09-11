# AGENTS.md

Single source of truth for `XvaSim` agentic operating rules, architecture context, and development standards. These apply to every AI-assisted coding session. If any other document disagrees with this file, this file wins.

## Project Identity & Core Capabilities

`XvaSim` is a lightweight, high-performance Python library for simulating and calculating credit and valuation adjustments (XVAs) and pricing multi-asset derivatives across Interest Rates, Foreign Exchange (FX), Credit, and Inflation risk factors.

Core capabilities:

1. **Modular Stochastic Models Framework** (`xvasim.models`): pluggable stochastic models with a central dynamic registry and factory (`ModelRegistry`, `create_ir_model`, `create_credit_model`, `create_fx_model`, `create_inflation_model`).
2. **Credit Valuation Adjustment (CVA) & Exposure Analytics** (`xvasim.cva_engine`): CIR/modular credit spread calibration (L-BFGS-B), path-wise Monte Carlo CVA aggregation, and counterparty exposure profiling (EE, EPE, Max PFE, quantile curves).
3. **Derivative Pricing Engine** (`xvasim.pricing_engine`): primary Monte Carlo pricers (`price_*`) that simulate paths by default and return simulated price, standard error, and analytical benchmark; dedicated closed-form benchmarks (`benchmark_price_*`).
4. **Portfolio & Exposure Layer** (`xvasim.portfolio`): `Trade` protocol + concrete FX trades (`FXForwardTrade`, `FXEuropeanOptionTrade`), `MarketSimulation` (shared simulated market context), `Portfolio` netting sets, `compute_portfolio_exposure`/`simulate_portfolio_exposure`, and the end-to-end `compute_portfolio_xva` ledger (CVA/DVA/FVA/KVA/MVA + exposure profile).
5. **Quasi-Monte Carlo (QMC) & Variance Reduction** (`xvasim.qmc`): Sobol/Halton/LHS/PRNG (`RandomSequenceType`), variate generation, `compare_t0_npv_fitting` convergence diagnostics, thread-safe `QMCSequenceCache`, stateful `QMCSequenceGenerator`.
6. **High-Performance JIT & Hardware Acceleration** (`xvasim.jit`, `xvasim.backend`): Numba `@njit(fastmath=True, nogil=True)` kernels and a unified `TensorBackend` abstraction (NumPy, PyTorch, CuPy, JAX) with `use_backend`/`set_backend`/`get_backend`.
7. **Automatic-Differentiation Greeks** (`xvasim.greeks`): dependency-free NumPy forward-mode AD engine (`Dual`/`Dual2` with chain rules), `compute_greeks` returning analytical, pathwise MC, and closed-form benchmarks; `GreeksResult` with per-model seed availability (delta/gamma/vega/rho).

## Core Repo Rules

- Package: `xvasim`, Python `>= 3.14`, Hatchling build backend, `src/` layout.
- Use `uv` exclusively for dependencies, environments, and tooling — never `pip`, `poetry`, or `conda`.
- Source lives in `src/xvasim`; tests live in `tests/`.
- GPU packages (PyTorch, CuPy, JAX) are optional hardware backends; never import them unconditionally or require them at package load time.
- `hatchling`, `numba`, `numexpr`, `numpy`, `scipy` remain the core runtime dependencies (see `pyproject.toml`).

## Repository Layout

```
XvaSim/
├── pyproject.toml              # Build & dependency config (Hatchling, uv, ruff, pyrefly, pytest, coverage)
├── README.md                   # Human & LLM documentation
├── AGENTS.md                   # Authoritative agentic rules, architecture & standards (this file)
├── docs/                       # Living project documentation
│   ├── PROJECT_TRACKER.md      # Project status, milestones, workstreams, backlog
│   └── FIXES_AND_COVERAGE.md   # Fixes log & test coverage detail
├── src/
│   └── xvasim/
│       ├── __init__.py         # Package root exports
│       ├── backend.py          # TensorBackend hardware abstraction (NumPy, PyTorch, CuPy, JAX)
│       ├── cva_engine.py       # CVA calculation, chunked evaluation & credit calibration
│       ├── greeks.py           # Forward-mode autodiff Greeks engine (Dual/Dual2, compute_greeks)
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
    │   ├── greeks/             # Autodiff Greeks engine tests (AD rules, per-model dispatch, error paths)
    │   ├── models/             # Model-specific tests (IR, FX, Credit, Inflation, base, registry)
    │   ├── portfolio/          # Portfolio / netting-set exposure & XVA ledger tests
    │   ├── pricing/            # Pricing engine tests (IRS, XCCY, FX, Inflation, internals, PricingResult)
    │   ├── qmc/                # Quasi-Monte Carlo variate & convergence tests
    │   ├── test_backend.py     # Hardware acceleration & tensor backend tests
    │   ├── test_jit.py         # Compiled numerical kernels tests
    │   └── utils/              # Helper utilities tests
    ├── integration/            # Multi-model simulations & portfolio CVA pipeline tests
    └── benchmarks/             # Analytical benchmark vs Monte Carlo & AD Greeks convergence tests
```

## Units & Naming Conventions

Strictly adhere to these conventions across all public APIs, helper functions, and data structures:

- Times & tenors use the `_yrs` suffix (e.g. `maturity_yrs`, `tenors_yrs`, `expiry_yrs`, `pay_freq_yrs`). 6 months = `0.5`.
- Rates, volatilities & spreads use the `_ann` suffix (e.g. `kappa_ann`, `sigma_ann`, `credit_spreads_ann`, `fx_vol_ann`, `cpi_vol_ann`). 2.5% = `0.025`.
- `__all__` must be sorted alphabetically in every `__init__.py`.
- Function-local variables must be lowercase (Ruff `N806`).

## Code Quality Gates

Every task is considered done only when all of the following pass on the final state of the code:

```bash
uv run ruff check .
uv run pyrefly check
uv run pytest tests/
uv run coverage run -m pytest tests/
uv run coverage report -m        # must be >= 95.0%
```

Fix all issues introduced by a change. Never silence a check to make it pass.

## Linting Rules (Ruff)

- Line length limit: **88** characters.
- Python target: `py314`.
- PEP 8 naming (`N806`): function-local variables must be lowercase.
- Clean imports: never leave unused imports (`F401`).
- Active rule categories: `E`, `W`, `F`, `I`, `N`, `UP`, `B`, `C4`, `RUF`.

## Typing Rules (Pyrefly)

- Pyrefly runs type checking on Python 3.14: `uv run pyrefly check` (optionally `--min-severity info`).
- Every function, method, parameter, and return value requires an explicit type annotation.
- Use `@typing.overload` / `@overload` for polymorphic or dual-signature interfaces.
- Do not wrap values already statically known `float` in redundant `float(...)` calls (`unnecessary-type-conversion`).
- Annotate `@contextlib.contextmanager` generator functions as `typing.Generator[YieldType]`, NOT `Iterator[YieldType]`; do not add redundant `, None, None` args (Ruff UP043).
- Convert numpy values explicitly with `np.asarray(..., dtype=np.float64)`; avoid loose `Any` return types.
- Keep `__all__` sorted alphabetically in `__init__.py` files.

## Test Discipline

- Every new feature or fix must add or update tests in the same change.
- Tests must pass under both `pytest` AND `python -m unittest discover tests`.
- Use the pytest markers from `pyproject.toml`: `unit`, `integration`, `benchmark`, `slow`.
- Keep the 95% coverage floor (enforced by `[tool.coverage.report] fail_under = 95`).
- Analytical-vs-MC benchmarks in `tests/benchmarks` validate Monte Carlo convergence; extend them when adding pricers or models.

## Framework & Registry Rules

### Adding a New Model or Risk Factor

1. **Subclass the Base ABC** in `src/xvasim/models/<category>/<model_name>.py`:
   - Interest rates: subclass `InterestRateModel`; implement `discount_curve_yrs`, `discount_factors`, `short_rate`, `zero_coupon_bond`, `discount_path`, `simulate_paths`.
   - Credit: subclass `CreditModel`; implement `simulate_paths`, `discount_factors`/survival interface.
   - FX / Inflation: subclass `FXModel` / `InflationModel` in their respective subpackages.
   - New asset classes: add a `RiskFactorType` member in `src/xvasim/models/base.py` and define the base class.
2. **Register with ModelRegistry**: decorate the class with `@ModelRegistry.register("<risk_factor_type>", "<model_name>")`.
3. **Export Cleanly**: export from the subpackage `__init__.py`, `models/__init__.py`, and the root `xvasim/__init__.py`; keep `__all__` sorted alphabetically.
4. **Add Comprehensive Tests**: unit tests in `tests/unit/models/<category>/test_<model_name>.py` (initialization, bond/disount pricing, path simulation, terminal distributions); multi-asset or cross-model tests in `tests/integration/`; analytical-vs-MC convergence tests in `tests/benchmarks/`.

### Model-Agnostic Architectural Patterns

The codebase implements model-agnostic abstractions and dispatchers (do not regress these):

1. **IR**: `calibrate_ir_model_to_swaptions(..., model_type="lgm")` dispatches generically; `_swaption_price_normal` / `InterestRateModel.swaption_price_normal` provide analytical swaption pricing.
2. **Credit**: `_calibrate_credit_model(spreads, tenors, model_type="cir")` and `_credit_model_survival_probability(tenors, model_or_params)` dispatch generically; `CreditModel.calibrate_from_spreads` enables self-calibration.
3. **Multi-factor composition**: `TwoCurrencyFXModel.from_ir_models` and `JarrowYildirimModel.from_ir_models` build multi-currency/multi-economy models from arbitrary component models.
4. **JIT dispatcher**: `simulate_model_paths_kernel(model_type, ...)` routes simulation stepping to the specialized kernel; generic `credit_survival_probability_kernel` / `credit_calibration_objective_kernel` for credit.
5. **Pricing result**: `PricingResult` subclasses `dict[str, Any]` with dual dict/attribute access (`res["price"]` and `res.price`); all pricers accept `model: ... | None = None` (or `params` fallback).

## Supported Model Inventory (Registry Keys)

All models are registered via `ModelRegistry` and constructible through the `create_*_model` factories or their classes directly. Both canonical keys and aliases below resolve to the same class.

| Risk factor | Concrete class | Registry keys |
|---|---|---|
| `interest_rate` | `LGMModel` | `lgm`, `linear_gauss_markov` |
| `interest_rate` | `HullWhite1FModel` | `hull_white`, `hull_white_1f`, `hw1f` |
| `interest_rate` | `VasicekModel` | `vasicek` |
| `interest_rate` | `CIRInterestRateModel` | `cir`, `cir_ir`, `cox_ingersoll_ross` |
| `credit` | `CIRHazardRateModel` | `cir`, `cir_hazard_rate`, `cox_ingersoll_ross` |
| `fx` | `TwoCurrencyFXModel` | `two_currency`, `cross_currency` |
| `fx` | `GarmanKohlhagenFXModel` | `garman_kohlhagen`, `black_scholes`, `gbm` |
| `fx` | `HestonFXModel` | `heston`, `heston_fx` |
| `inflation` | `JarrowYildirimModel` | `jarrow_yildirim`, `jy`, `two_factor_hw` |
| `inflation` | `BlackInflationModel` | `black`, `black_inflation`, `lognormal` |

Simulation return contracts: IR `simulate_paths(times, n_paths, ...)` returns an `(n_paths, n_steps+1)` state array; FX `simulate_paths(maturity_yrs, n_paths, n_steps, ...)` returns `(times, x_dom, x_for, fx_spot)` (Heston: `(times, v_paths, x_dummy, fx_spot)`); inflation `simulate_paths(...)` returns an `InflationSimulationResult` supporting 4-tuple unpacking (`times, x_nom, x_real, cpi`).

## No Legacy Preservation

- Do NOT preserve backwards compatibility with deprecated parameter dataclasses, deprecated registry keys, or deprecated aliases. The legacy `FXLGMParams` shim has been removed from `pricing_engine.py`; do not reintroduce it. The current parameter dataclasses (`LGMParams`, `HullWhite1FParams`, `VasicekParams`, `CIRInterestRateParams`, `CIRHazardRateParams`, `GarmanKohlhagenFXParams`, `HestonFXParams`, `TwoCurrencyFXParams`, `JarrowYildirimParams`, `BlackInflationParams`) are canonical, first-class API — never wrap them in legacy shims.
- When an interface changes, update every caller, export, test, and docs file to the latest API in the same change.
- Remove obsolete shims instead of wrapping them.

## Commit Etiquette

- Only commit when explicitly asked.
- Stage only intended files; never commit `*.pyc`, `__pycache__/`, `.coverage*`, `.pytest_cache/`, `.ruff_cache/`, or virtualenvs.
- Write focused, imperative commit messages consistent with the repo's existing style.

## Performance Rules

- Prefer Numba `@njit(fastmath=True, nogil=True)` kernels for hot simulation/calibration loops (`cir_simulate_paths_kernel`, `lgm_simulate_paths_kernel`, `vasicek_simulate_paths_kernel`, `hull_white_simulate_paths_kernel`, `heston_simulate_paths_kernel`, `discount_path_kernel`), with transparent pure-Python fallback when Numba is unavailable or `NUMBA_DISABLE_JIT=1`.
- Use memory-efficient chunked evaluation for large path counts: `compute_cva(..., chunk_size=..., use_numexpr=True)` and streaming `compute_cva_chunked` over exposure generators.
- Calibrate against pure-numeric Numba objective kernels (`cir_calibration_objective_kernel`, `cir_survival_probability_kernel`) evaluated on 1D arrays without per-iteration dataclass allocation.
- Expose `random_type`, `seed`, `scramble`, and `use_cache` QMC parameters where applicable; lean on `QMCSequenceCache`/`QMCSequenceGenerator` for Greek bump-and-reval sensitivity runs.
- GPU acceleration goes through `TensorBackend` only; never hard-code a hardware framework.

## Mathematical Reference & Supported Models

### Interest Rate Models

- **LGM (Linear Gauss-Markov)**: zero-mean Gaussian state $dx(t) = -\kappa x(t)\,dt + \sigma(t)\,dW(t)$, analytical discount bond $P(t,T) = \frac{P(0,T)}{P(0,t)}\exp\left(-H(T)x(t)-\frac{1}{2}(H(T)^2-H(t)^2)\zeta(t)\right)$ with $H(t)=\frac{1-e^{-\kappa t}}{\kappa}$.
- **Hull-White 1-Factor (HW1F)**: $dr(t) = (\theta(t)-a\,r(t))\,dt+\sigma\,dW(t)$, exact term-structure fitting to the initial discount curve, analytical bond via $B(t,T)=\frac{1-e^{-a(T-t)}}{a}$.
- **Vasicek**: mean-reverting Gaussian diffusion $dr(t)=\kappa(\theta-r(t))\,dt+\sigma\,dW(t)$.
- **CIR**: mean-reverting square-root diffusion $dr(t)=\kappa(\theta-r(t))\,dt+\sigma\sqrt{r(t)}\,dW(t)$, non-negative when the Feller condition $2\kappa\theta \ge \sigma^2$ holds.

### FX Models

- **Garman-Kohlhagen (Black FX)**: $\frac{dS(t)}{S(t)}=(r_d-r_f)\,dt+\sigma_{fx}\,dW(t)$ with closed-form European call/put pricing using $F = S_0 P_f(0,T)/P_d(0,T)$.
- **Heston Stochastic Volatility**: couples spot FX SDE with variance SDE $dv(t)=\kappa_v(\theta_v-v(t))\,dt+\sigma_v\sqrt{v(t)}\,dW_v(t)$ and correlation $\rho_{S,v}$.
- **Two-Currency Multi-Factor**: couples domestic and foreign stochastic short rates with spot FX and a $3\times3$ correlation structure; foreign rate carries a quanto drift $\rho_{f,S}\sigma_f\sigma_{fx}$ under the domestic measure.

### Credit & CVA Models

- **CIR Hazard Rate**: default intensity $d\lambda(t)=\kappa(\theta-\lambda(t))\,dt+\sigma\sqrt{\lambda(t)}\,dW(t)$ with analytical survival curve $P_{surv}(0,t)=A(t)\exp(-B(t)\lambda_0)$.
- **Portfolio CVA**: path-wise Monte Carlo integration $\text{CVA}=\text{LGD}\times\frac{1}{N}\sum_i\sum_j \text{Exposure}_{i,j}\times\Delta\text{PD}_{i,j}\times D_{i,j}$.

### Inflation Models

- **Jarrow-Yildirim (JY)**: 3-factor coupled system under the nominal measure $\mathbb{Q}_n$ for nominal rate $r_n(t)$, real rate $r_r(t)$ (with foreign/quanto drift), and CPI index $I(t)$ with $\frac{dI(t)}{I(t)}=(r_n(t)-r_r(t))\,dt+\sigma_I\,dW_I(t)$.
- **Black CPI Forward**: martingale forward CPI $\frac{dF_I(t,T)}{F_I(t,T)}=\sigma_I\,dW_I^T(t)$ with closed-form fair ZCIS rate $S_0(T)=\left(P_r(0,T)/P_n(0,T)\right)^{1/T}-1$.

## Verification & Development Commands

```bash
uv sync                                  # Install dependencies
uv run pytest tests/ -v                  # Full test suite via Pytest
uv run python -m unittest discover tests # Full test suite via Unittest
uv run coverage run -m pytest tests/     # Coverage with 95% floor
uv run coverage report -m
uv run ruff check .                      # Linting
uv run pyrefly check                     # Static type checking
```

## Docs Hygiene

- Keep `README.md`, `AGENTS.md`, `docs/PROJECT_TRACKER.md`, and `docs/FIXES_AND_COVERAGE.md` in sync with the latest code state.
- When work lands, update the project tracker; when fixes or coverage change, record them in the fixes & coverage doc.
- Do not re-introduce split AI-guide documents; this file is the single canonical agent reference.
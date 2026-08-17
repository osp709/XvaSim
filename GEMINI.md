# GEMINI.md: LLM & AI Assistant Guide for XvaSim

This document provides system instructions, architectural context, mathematical models, and development rules for Gemini and other AI coding assistants working on `XvaSim`.

---

## 🏛️ Project Identity & Core Capabilities

`XvaSim` is a lightweight, high-performance Python library designed for simulating and calculating credit and valuation adjustments (XVAs).

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
   - Path-wise Monte Carlo CVA aggregation.
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
   - Uniform `random_type`, `seed`, and `scramble` parameters across all models and pricers.

---

## 📐 Units & Naming Conventions

Strictly adhere to the following naming conventions across all public APIs, helper functions, and data structures:
*   **Time & Tenors**: Must use suffix `_yrs` (e.g. `maturity_yrs`, `tenors_yrs`, `expiry_yrs`). 6 months = `0.5`.
*   **Rates, Volatilities & Spreads**: Must use suffix `_ann` (e.g. `kappa_ann`, `sigma_ann`, `credit_spreads_ann`, `fx_vol_ann`, `cpi_vol_ann`). 2.5% = `0.025`.

---

## 📁 Repository Layout

```
XvaSim/
├── pyproject.toml              # Build & dependency config (Hatchling, uv, ruff, mypy)
├── README.md                   # Human & LLM documentation
├── GEMINI.md                   # AI assistant context & developer guide
├── src/
│   └── xvasim/
│       ├── __init__.py         # Package root exports
│       ├── cva_engine.py       # CVA calculation & credit model integration
│       ├── pricing_engine.py   # MC & analytical pricing for FX & inflation derivatives
│       ├── qmc.py              # Quasi-Monte Carlo sequences & variance reduction
│       ├── utils.py            # Date conversion (dates_to_years)
│       └── models/             # Modular stochastic models framework
│           ├── __init__.py     # Models package exports
│           ├── base.py         # Base ABCs (RiskFactorType, StochasticModel, InterestRateModel, CreditModel, FXModel, InflationModel)
│           ├── registry.py     # ModelRegistry & factory functions
│           ├── ir/             # Interest rate models (LGM, Hull-White 1F, Vasicek, CIR)
│           ├── credit/         # Credit models (CIR hazard rate)
│           ├── fx/             # FX models (TwoCurrencyFXModel, GarmanKohlhagenFXModel, HestonFXModel)
│           └── inflation/      # Inflation models (JarrowYildirimModel, BlackInflationModel)
└── tests/                      # Full unit test suite (discoverable via unittest)
```

---

## 🛠️ Coding & Quality Standards

### 1. Strict Typing (Mypy)
- Mypy runs with `strict = true` on Python 3.14.
- All functions, methods, and parameters MUST have explicit type annotations.
- Avoid loose `Any` return types; cast numpy operations with `np.asarray(..., dtype=np.float64)` where necessary.
- In `__init__.py` files, keep `__all__` sorted alphabetically.

### 2. Linting (Ruff)
- Line length limit: **88** characters.
- PEP 8 naming (`N806`): Function-local variables must be lowercase.
- Clean imports: Never leave unused imports (`F401`).

### 3. Backwards Compatibility
- Always maintain backwards compatibility with legacy parameter dataclasses (`FXLGMParams`, `LGMParams`, `CIRParams`).
- Overloaded methods and polymorphism should support both legacy dataclasses and new modular classes.

---

## 🧩 How to Add a New Model / Risk Factor

1. **Subclass the Base ABC**:
   - For interest rates: Subclass `InterestRateModel` in `src/xvasim/models/ir/<model_name>.py`.
   - Implement: `discount_curve_yrs`, `discount_factors`, `short_rate`, `zero_coupon_bond`, `discount_path`, `simulate_paths`.
   - For credit: Subclass `CreditModel` in `src/xvasim/models/credit/<model_name>.py`.
   - For new asset classes: Add to `RiskFactorType` in `base.py` and create the base class.
2. **Register with ModelRegistry**:
   - Add `@ModelRegistry.register("interest_rate", "<model_name>")` to your model class.
3. **Export Cleanly**:
   - Export in category `__init__.py`, `models/__init__.py`, and root `xvasim/__init__.py`.
4. **Add Unit Tests**:
   - Create tests in `tests/` covering initialization, analytical bond pricing vs discount curve, and simulation paths.

---

## 🧪 Verification & Development Commands

```bash
# Install dependencies
uv sync

# Run full test suite
uv run python -m unittest discover tests

# Linting checks
uv run ruff check .

# Static type checking
uv run mypy .
```

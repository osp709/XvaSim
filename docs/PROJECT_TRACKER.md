# XvaSim Project Tracker

Living status used by human and AI contributors to keep the project's progress visible and current. Update this file whenever work lands (see "How to Update" below).

## Quick Snapshot

| Item | Value |
| :--- | :--- |
| Branch | `main` (in sync with `origin/main`) |
| Package version | 0.1.0 |
| Python | 3.14 (repo requires `>=3.14`) |
| Tests | 305 passing (pytest), dual-runner compatible (unittest) |
| Coverage | 97.1% (floor: 95.0%) |
| Ruff lint | clean |
| Pyrefly type check | clean (0 errors) |
| Last audit | 2026-09-11 |

## Status Legend

- **[x] Done** — completed and verified by the quality gates.
- **[~] In Progress** — actively being worked on.
- **[>] Planned** — queued next.
- **[ ] Backlog** — acknowledged, not yet scheduled.

---

## Milestones

- **Initial commit** — project scaffold, CIR CVA engine, calibration tooling (`cb98270`).
- **Baseline framework** — models, price engine, QMC feature set (`35e51a5`, `5cc1aab`, `1cdb928`).
- **JIT / backend / streaming CVA** — Numba kernels, hardware backends, memory-efficient aggregation (`8922f49`).
- **Modular registry + baseline model suite** — `ModelRegistry`, factory pattern, IR/FX/inflation models (`c63d505`).
- **Credit + FX + pricing expansion** — CIR credit model, Garman-Kohlhagen FX, expanded pricers (`a2080d3`).
- **Pyrefly conformance** — removed redundant `float(...)` casts, modernized `@contextmanager` annotations across models/pricers (`6d1e59a`).
- **Repo hygiene** — expanded `.gitignore`, untracked 107 `*.pyc` + `.coverage` (`62eeac6`, `837fc1e`).
- **Agentic workflow & docs (this change)** — added `AGENTS.md` as the single source of truth, `docs/PROJECT_TRACKER.md`, `docs/FIXES_AND_COVERAGE.md`; removed `GEMINI.md` (reconciled into `AGENTS.md`); new no-legacy-compat policy.
- **Test suite repair** — replaced legacy alias/param imports in 8 test modules to match canonical APIs; 194 tests green before XVA expansion.
- **XVA suite expansion** — `compute_dva`, `compute_fva`, `compute_kva`, `compute_mva`, `compute_total_xva` plus ENE/FE exposure profile keys; FVA decomposed into symmetric FCA/FBA legs.
- **Portfolio & Exposure layer** — `Trade` protocol, `FXForwardTrade`, `FXEuropeanOptionTrade`, `MarketSimulation`, `Portfolio` (netting set), `compute_portfolio_exposure`, `compute_portfolio_xva`; `TwoCurrencyFXModel` gains `domestic_discount_factor`/`foreign_discount_factor`; `FXModel` abstract contract updated.
- **Autodiff Greeks layer** — dependency-free NumPy forward-mode AD engine (`Dual`/`Dual2`), `compute_greeks` with `autodiff`/`closed_form` and pathwise `monte_carlo` methods, `GreeksResult` container (delta/gamma/vega/rho per model seed availability).

---

## Workstreams

### Foundation & Packaging
- [x] `src/` layout, Hatchling build backend, `uv` lockfile.
- [x] Core deps (numba, numexpr, numpy, scipy) + dev tooling (ruff, pyrefly, pytest, coverage).
- [x] Strict quality gates: 95% coverage floor, Ruff rule set, Pyrefly type strictness.

### Core Engine (backend, QMC, JIT)
- [x] `TensorBackend` abstraction: NumPy, PyTorch, CuPy, JAX, with `use_backend`/`set_backend`.
- [x] Numba `@njit(fastmath=True, nogil=True)` simulation, discount, credit-calibration kernels.
- [x] QMC sequences (Sobol, Halton, LHS, PRNG), thread-safe `QMCSequenceCache`, stateful generator.
- [x] Variance-reduction benchmarking (`compare_t0_npv_fitting`).

### Stochastic Models
- [x] `RiskFactorType` + ABCs: `InterestRateModel`, `CreditModel`, `FXModel`, `InflationModel`.
- [x] `ModelRegistry` + factories (`create_ir_model`, `create_credit_model`, `create_fx_model`, `create_inflation_model`).
- [x] IR: LGM, Hull-White 1F, Vasicek, CIR.
- [x] FX: Two-Currency Multi-Factor, Garman-Kohlhagen, Heston.
- [x] Credit: CIR hazard rate.
- [x] Inflation: Jarrow-Yildirim, Black CPI Forward.

### Pricing Engine
- [x] Primary Monte Carlo pricers + analytical benchmarks for IRS, XCCY, FX forward/option, ZCIS, YoY, CPI option.
- [x] Model-agnostic calibration (`calibrate_ir_model_to_swaptions`).
- [x] `PricingResult` dual dict/attribute access.
- [x] Decoupled multi-factor construction (`from_ir_models` / `from_params`).

### CVA Engine
- [x] Path-wise chunked/numexpr CVA aggregation (`compute_cva`, `compute_cva_chunked`).
- [x] Marginal PD from calibrated CIR credit curve (`compute_marginal_pd`).
- [x] Exposure profiling (EE, EPE, ENE, ENE scalar, funding exposure, Max PFE, quantile curves).
- [x] DVA, FVA, KVA, MVA adjustments (`compute_dva`, `compute_fva`, `compute_kva`, `compute_mva`).
- [x] FVA decomposed into symmetric legs: `compute_fva` returns `{fca, fba, fva}` (funding cost on EE + funding benefit on ENE); `compute_total_xva` reports both legs.
- [x] Single-pass full XVA aggregation (`compute_total_xva`) returning component + total values.

### Portfolio & Exposure
- [x] `Trade` protocol (structural), `FXForwardTrade`, `FXEuropeanOptionTrade` (frozen dataclasses).
- [x] `MarketSimulation` (shape-validating frozen dataclass) + `simulate_market`.
- [x] `Portfolio` (netting set with dedup validation) + `compute_portfolio_exposure`.
- [x] `PortfolioExposureResult` dataclass with per-trade MTM, net/gross exposure, negative exposure.
- [x] End-to-end `compute_portfolio_xva`: simulate → aggregate → CIR credit calibration → full XVA ledger + exposure profile.
- [x] `TwoCurrencyFXModel` implements `domestic_discount_factor`/`foreign_discount_factor` abstract methods.

### Greeks & Sensitivities
- [x] NumPy forward-mode AD engine: `Dual` (first order) and `Dual2` (second order) with arithmetic, `exp`/`log`/`sqrt`/`erf`/`norm_cdf`/`relu`/`mean` operators.
- [x] `compute_greeks(trade, fx_model, ...)` returning `GreeksResult` (delta, gamma, vega, rho_domestic, rho_foreign) for FX forwards/options.
- [x] Methods: `autodiff`/`closed_form` (analytical AD) and `monte_carlo` (pathwise AD over frozen QMC draws with `std_error`).
- [x] Per-model seed handling: `spot_fx`/`fx_vol_ann`/rate seeds where expressible; `rho=None` for Two-Currency and curve-discounting GK/Heston; `vega=None` for Heston; `gamma=None` for pathwise option Monte Carlo.
- [x] Error contracts: Heston option closed-form `NotImplementedError`, Monte Carlo requires `GarmanKohlhagenFXModel`, unknown `method` raises `ValueError`.
- [x] AD-vs-analytical Black-76 and AD-vs-finite-difference convergence benchmarks.

### Quality, Tests & Hygiene
- [x] 305 tests across unit / integration / benchmarks; dual pytest + unittest.
- [x] Analytical-vs-MC, path-convergence, and AD-vs-analytical/FD Greeks benchmarks.
- [x] 97.1% coverage (above 95.0% floor).
- [x] Git hygiene: `*.pyc`, `.coverage`, caches untracked and ignored.

### Legacy Compatibility Cleanup (new policy)
- [x] Removed the legacy shim `FXLGMParams`; source exports and tests already use only canonical names (`LGMParams`, `CIRHazardRateParams`, `GarmanKohlhagenFXParams`, `TwoCurrencyFXParams`, ...).
- [x] Normalized aliases across code and tests (`price_irs`, `price_fx_forward`, `benchmark_*`, `calibrate_lgm_to_swaptions`, `from_lgm_params`, etc.) — no shorthand or legacy alias remains in `src/` or `tests/`.

### Backlog / Ideas
- [ ] Additional XVA metrics: COLVA, collateral/FVA refinements (netting & collateral CSA modelling).
- [ ] Model: two-factor Gaussian IR / multi-currency LGM extension.
- [ ] Bermudan exceptions & exercise-boundary pricing.
- [ ] Greeks for additional trade types (interest-rate/CPI derivatives) and higher-order / cross-gamma seeds beyond the `spot_fx` diagonal.
- [ ] `benchmark_price_year_on_year_inflation_swap` analytical benchmark (currently interpolated-CPI forward project).

---

## Current Focus (In Progress)

- Autodiff Greeks layer landed (`greeks.py`); full AD engine + `compute_greeks` with analytical, MC, and Black-76 benchmark convergence tests; 305 tests, coverage 97.1%, all gates green.
- Portfolio & Exposure layer and end-to-end XVA ledger stable.
- Next up: backlog ideas — collateral/COLVA XVA, IR/inflation Greeks for additional trade types, YoY inflation benchmark, two-factor Gaussian IR model.

## How to Update

When a task lands:

1. Mark the item `[x]` and add/refresh the milestone line with the commit short-hash.
2. Refresh the Quick Snapshot numbers (test count, coverage, lint/typecheck status).
3. Follow the commit etiquette rules in `AGENTS.md`; never commit doc updates ahead of their code.

## Related Documents

- `AGENTS.md` — authoritative agentic rules, architecture context & standards (single source of truth).
- `README.md` — human-facing feature & usage guide.
- `docs/FIXES_AND_COVERAGE.md` — fixes log and coverage detail.
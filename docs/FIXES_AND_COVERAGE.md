# XvaSim — Fixes & Test Coverage

Log of applied fixes and the current test-coverage picture. Regenerate the coverage numbers with the commands in the "How to Measure Coverage" section; record every fix below.

---

## Fixes Log

| Date | Commit | Description | Scope | Verification |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-08 | `6d1e59a` | Removed redundant `float(...)` casts on values already statically typed `float` (pyrefly `unnecessary-type-conversion` warnings) across `cva_engine.py`, `qmc.py`, models, and test helpers; changed `use_backend` annotation from `Iterator` to `typing.Generator` and removed redundant `, None, None` args. | `backend.py`, `cva_engine.py`, `qmc.py`, `garman_kohlhagen.py`, `heston.py`, `two_currency.py`, `black_inflation.py`, `jarrow_yildirim.py`, `hull_white.py`, `pricing_engine.py`, `tests/helpers/assertions.py`, `GEMINI.md` | 192/192 tests, coverage 97.6%, Ruff clean, Pyrefly 0 errors |
| 2026-09-08 | `837fc1e` | Repo hygiene: untracked 107 committed `*.pyc` artifacts and `.coverage`; expanded `.gitignore` with Python standard ignores (`__pycache__/`, `*.py[cod]`, caches, virtualenvs). Working-tree tracked file count dropped to 79. (Parent `62eeac6` introduced the `.gitignore`; `837fc1e` performs the actual untracking.) | repository-wide | `git ls-files` contains no `*.pyc`/`.coverage` |
| 2026-09-08 | — | Model naming convention: renamed `CIRParams` → `CIRHazardRateParams` and `GarmanKohlhagenParams` → `GarmanKohlhagenFXParams` to restore 1:1 params/model naming; added `TwoCurrencyFXParams` frozen dataclass with `TwoCurrencyFXModel.from_params` / `params` for API uniformity. | `models/credit/cir.py`, `models/fx/{garman_kohlhagen,two_currency}.py`, `cva_engine.py`, `pricing_engine.py`, package exports, tests, docs | 195/195 tests, Ruff clean, Pyrefly 0 errors |
| 2026-09-09 | — | Test suite repair: replaced legacy aliases (`_cir_survival_probability`, `FXLGMParams`, `from_lgm_params`, `from_components`, `analytical_swaption_price`) with canonical API in 8 test modules; aligned `test_two_currency`/`test_jarrow_yildirim` with `from_ir_models`. | 8 files under `tests/` | 194/194 tests, coverage 98.0%, Ruff clean, Pyrefly 0 errors |
| 2026-09-09 | — | XVA suite expansion: added ENE/FE keys to `compute_exposure_profile`; new `compute_dva`, `compute_fva`, `compute_kva`, `compute_mva`, `compute_total_xva` with chunked/numexpr support; exported from package root; new unit + integration tests. | `cva_engine.py`, `__init__.py`, `tests/unit/cva/test_cva_engine.py`, `tests/integration/test_portfolio_cva_pipeline.py` | 222/222 tests, coverage 98.0%, Ruff clean, Pyrefly 0 errors |
| 2026-09-09 | — | FVA symmetric decomposition: `compute_fva` now returns `{fca, fba, fva}` reporting the funding cost (FCA, positive-exposure leg) and the symmetric funding benefit (FBA, negative-exposure leg) separately; `compute_total_xva` extended with `fca`/`fba` keys (net FVA = FCA + FBA). | `cva_engine.py`, `tests/unit/cva/test_cva_engine.py`, `README.md` | 223/223 tests, coverage 98.0%, Ruff clean, Pyrefly 0 errors |
| 2026-09-09 | — | Legacy cleanup completed: removed residual "legacy"/"backwards-compatible" wording in `pricing_engine.py`; normalized stale alias references across `README.md`, `AGENTS.md`, and `docs/` (`price_irs`, `price_fx_forward`, `from_lgm_params`, `from_components`, `analytical_swaption_price`, `FXLGMParams`, etc.). | `pricing_engine.py`, `README.md`, `AGENTS.md`, `docs/` | 223/223 tests, coverage 98.0%, Ruff clean, Pyrefly 0 errors |
| 2026-09-11 | — | Portfolio & Exposure layer: new `src/xvasim/portfolio.py` with `Trade` structural protocol, `FXForwardTrade`, `FXEuropeanOptionTrade`, `MarketSimulation`, `Portfolio` (netting set), `compute_portfolio_exposure`, `compute_portfolio_xva`; `FXModel` gained abstract `domestic_discount_factor`/`foreign_discount_factor` (implemented on `TwoCurrencyFXModel` via `interpolate_discount_factor`); root exports + 26 unit tests + 5 integration tests. | `portfolio.py`, `__init__.py`, `models/base.py`, `models/fx/two_currency.py`, `tests/unit/portfolio/test_portfolio.py`, `tests/integration/test_portfolio_xva_pipeline.py` | 254/254 tests, coverage 97.4%, Ruff clean, Pyrefly 0 errors |

### Known & Intended Behavior Notes

- (none)

---

## Test Coverage

### Current Snapshot (measured 2026-09-11)

- Total statements: 3,284; covered: 3,222; missing: 62.
- **Overall coverage: 97.4%** (requirement: `fail_under = 95.0`).
- Branch coverage enabled; both pytest and unittest runners pass.

### Per-Module Coverage

| Module | Coverage | Missing Lines / Branches |
| :--- | :--- | :--- |
| `xvasim/__init__.py` | 100.0% | — |
| `xvasim/backend.py` | 98.4% | 392, 534; branch 502->505, 822->exit |
| `xvasim/cva_engine.py` | 98.8% | 384, 873 |
| `xvasim/jit.py` | 95.4% | 108-109, 116-117, 148, 158 |
| `xvasim/models/__init__.py` | 100.0% | — |
| `xvasim/models/base.py` | 100.0% | — |
| `xvasim/models/credit/__init__.py` | 100.0% | — |
| `xvasim/models/credit/cir.py` | 95.3% | 165-166 |
| `xvasim/models/fx/__init__.py` | 100.0% | — |
| `xvasim/models/fx/garman_kohlhagen.py` | 100.0% | — |
| `xvasim/models/fx/heston.py` | 98.4% | 393, 400 |
| `xvasim/models/fx/two_currency.py` | 98.5% | 123, 129 |
| `xvasim/models/inflation/__init__.py` | 100.0% | — |
| `xvasim/models/inflation/black_inflation.py` | 98.0% | 88-93 |
| `xvasim/models/inflation/jarrow_yildirim.py` | 96.6% | 135, 140; branches 272->276, 281->285, 286 |
| `xvasim/models/ir/__init__.py` | 100.0% | — |
| `xvasim/models/ir/cir.py` | 100.0% | — |
| `xvasim/models/ir/hull_white.py` | 100.0% | — |
| `xvasim/models/ir/lgm.py` | 98.5% | 414-415 |
| `xvasim/models/ir/vasicek.py` | 100.0% | — |
| `xvasim/models/registry.py` | 100.0% | — |
| `xvasim/portfolio.py` | 93.2% | 95, 100, 105, 118, 132-133, 370-372, 432, 602-606, 614, 632-633, 791-792 |
| `xvasim/pricing_engine.py` | 95.3% | 205, 211, 294-295, 594-596, 685, 924, 992-995, 1244-1249, 1270-1278, 1799-1810 |
| `xvasim/qmc.py` | 99.0% | branch 585->588; 622 |
| `xvasim/utils.py` | 100.0% | — |
| **TOTAL** | **97.4%** | **62 statements** |

### Modules Needing Attention

The lowest covered modules are the natural targets for the next coverage increase:

- `xvasim/portfolio.py` (93.2%) — unreachable Protocol/property stubs, defensive branches (non-`str` `trade_id`, `netting=False` empty portfolios, horizon/misaligned-credit argument guards).
- `xvasim/jit.py` (95.4%) — pure-Python fallback paths (`108-109, 116-117`) and dispatch branches (`148, 158`).
- `xvasim/models/credit/cir.py` (95.3%) — validation error branches at `165-166`.

### How to Measure Coverage

```bash
uv run coverage run -m pytest tests/
uv run coverage report -m
```

The report fails the run if coverage drops below 95.0% (`[tool.coverage.report] fail_under = 95`). After measuring, update the per-module table above if numbers moved.
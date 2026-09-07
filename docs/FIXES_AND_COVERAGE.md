# XvaSim — Fixes & Test Coverage

Log of applied fixes and the current test-coverage picture. Regenerate the coverage numbers with the commands in the "How to Measure Coverage" section; record every fix below.

---

## Fixes Log

| Date | Commit | Description | Scope | Verification |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-08 | `6d1e59a` | Removed redundant `float(...)` casts on values already statically typed `float` (pyrefly `unnecessary-type-conversion` warnings) across `cva_engine.py`, `qmc.py`, models, and test helpers; changed `use_backend` annotation from `Iterator` to `typing.Generator` and removed redundant `, None, None` args. | `backend.py`, `cva_engine.py`, `qmc.py`, `garman_kohlhagen.py`, `heston.py`, `two_currency.py`, `black_inflation.py`, `jarrow_yildirim.py`, `hull_white.py`, `pricing_engine.py`, `tests/helpers/assertions.py`, `GEMINI.md` | 192/192 tests, coverage 97.6%, Ruff clean, Pyrefly 0 errors |
| 2026-09-08 | `837fc1e` | Repo hygiene: untracked 107 committed `*.pyc` artifacts and `.coverage`; expanded `.gitignore` with Python standard ignores (`__pycache__/`, `*.py[cod]`, caches, virtualenvs). Working-tree tracked file count dropped to 79. (Parent `62eeac6` introduced the `.gitignore`; `837fc1e` performs the actual untracking.) | repository-wide | `git ls-files` contains no `*.pyc`/`.coverage` |
| 2026-09-08 | — | Model naming convention: renamed `CIRParams` → `CIRHazardRateParams` and `GarmanKohlhagenParams` → `GarmanKohlhagenFXParams` to restore 1:1 params/model naming; added `TwoCurrencyFXParams` frozen dataclass with `TwoCurrencyFXModel.from_params` / `params` for API uniformity. | `models/credit/cir.py`, `models/fx/{garman_kohlhagen,two_currency}.py`, `cva_engine.py`, `pricing_engine.py`, package exports, tests, docs | 195/195 tests, Ruff clean, Pyrefly 0 errors |

### Known & Intended Behavior Notes

- The legacy `FXLGMParams` shim (in `pricing_engine.py`) is slated for removal under the no-legacy-preservation policy (see `AGENTS.md` and `docs/PROJECT_TRACKER.md`). All model parameter dataclasses are canonical and 1:1 with their model names. Once removed, this note is obsolete and should be deleted.

---

## Test Coverage

### Current Snapshot (measured 2026-09-08)

- Total statements: 2,898; covered: 2,855; missing: 43.
- **Overall coverage: 97.8%** (requirement: `fail_under = 95.0`).
- Branch coverage enabled; both pytest and unittest runners pass.

### Per-Module Coverage

| Module | Coverage | Missing Lines / Branches |
| :--- | :--- | :--- |
| `xvasim/__init__.py` | 100.0% | — |
| `xvasim/backend.py` | 98.4% | 392, 534; branch 502->505, 822->exit |
| `xvasim/cva_engine.py` | 97.7% | 339; branch 371->373 |
| `xvasim/jit.py` | 95.4% | 108-109, 116-117, 148, 158 |
| `xvasim/models/__init__.py` | 100.0% | — |
| `xvasim/models/base.py` | 100.0% | — |
| `xvasim/models/credit/__init__.py` | 100.0% | — |
| `xvasim/models/credit/cir.py` | 95.3% | 165-166 |
| `xvasim/models/fx/__init__.py` | 100.0% | — |
| `xvasim/models/fx/garman_kohlhagen.py` | 100.0% | — |
| `xvasim/models/fx/heston.py` | 98.4% | 393, 400 |
| `xvasim/models/fx/two_currency.py` | 100.0% | — |
| `xvasim/models/inflation/__init__.py` | 100.0% | — |
| `xvasim/models/inflation/black_inflation.py` | 98.0% | 88-93 |
| `xvasim/models/inflation/jarrow_yildirim.py` | 96.6% | 135, 140; branches 292->296, 301->305, 306 |
| `xvasim/models/ir/__init__.py` | 100.0% | — |
| `xvasim/models/ir/cir.py` | 100.0% | — |
| `xvasim/models/ir/hull_white.py` | 100.0% | — |
| `xvasim/models/ir/lgm.py` | 98.5% | 431-432 |
| `xvasim/models/ir/vasicek.py` | 100.0% | — |
| `xvasim/models/registry.py` | 100.0% | — |
| `xvasim/pricing_engine.py` | 95.5% | 252, 266, 349-350, 598-605, 898, 902-913, 1142-1145, 1399-1404, 1425-1433, 1987-1998 |
| `xvasim/qmc.py` | 99.0% | branch 585->588; 622 |
| `xvasim/utils.py` | 100.0% | — |
| **TOTAL** | **97.8%** | **43 statements** |

### Modules Needing Attention

The lowest covered modules are the natural targets for the next coverage increase:

- `xvasim/jit.py` (95.4%) — pure-Python fallback paths (`108-109, 116-117`) and dispatch branches (`148, 158`).
- `xvasim/models/credit/cir.py` (95.3%) — validation error branches at `165-166`.
- `xvasim/pricing_engine.py` (95.5%) — the largest file; error paths, `n_paths=None` branches, and legacy XCCY branches.

### How to Measure Coverage

```bash
uv run coverage run -m pytest tests/
uv run coverage report -m
```

The report fails the run if coverage drops below 95.0% (`[tool.coverage.report] fail_under = 95`). After measuring, update the per-module table above if numbers moved.
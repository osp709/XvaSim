"""Quasi-Monte Carlo (QMC) and random number generation engine for XvaSim.

This module provides low-discrepancy sequence generators (Sobol, Halton,
Latin Hypercube Sampling) and pseudo-random generators for variance reduction
and noise reduction in Monte Carlo derivative pricing and stochastic simulations.

Public API
----------
- :class:`QMCSequenceCache` — thread-safe cache for pre-computed QMC draws.
- :class:`QMCSequenceGenerator` — stateful low-discrepancy sequence generator.
- :class:`RandomSequenceType` — enumeration of supported random / quasi-random types.
- :func:`cached_normal_draws` — generate or retrieve cached standard normal draws.
- :func:`clear_qmc_cache` — flush the global QMC variate cache.
- :func:`compare_t0_npv_fitting` — benchmark T0 NPV fitting error across generators.
- :func:`generate_brownian_increments` — generate Brownian increments for simulations.
- :func:`generate_normal_draws` — generate multi-dimensional standard normal variates.
- :func:`get_qmc_cache` — get the global QMCSequenceCache instance.
"""

from __future__ import annotations

import collections
import enum
import threading
import typing
import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

__all__ = [
    "QMCSequenceCache",
    "QMCSequenceGenerator",
    "RandomSequenceType",
    "cached_normal_draws",
    "clear_qmc_cache",
    "compare_t0_npv_fitting",
    "generate_brownian_increments",
    "generate_normal_draws",
    "get_qmc_cache",
]


class RandomSequenceType(enum.Enum):
    """Enumeration of supported random and quasi-random number sequence types.

    Members:
        PSEUDO: Standard pseudo-random numbers (NumPy PRNG).
        SOBOL: Sobol low-discrepancy sequence with optional scrambling.
        HALTON: Halton low-discrepancy sequence with optional scrambling.
        LATIN_HYPERCUBE: Latin Hypercube Sampling (LHS).
    """

    PSEUDO = "pseudo"
    SOBOL = "sobol"
    HALTON = "halton"
    LATIN_HYPERCUBE = "latin_hypercube"


def _parse_random_sequence_type(
    method: RandomSequenceType | str,
) -> RandomSequenceType:
    """Parse and validate a RandomSequenceType enum or string representation.

    Args:
        method: A :class:`RandomSequenceType` instance or case-insensitive string
            (e.g. ``"pseudo"``, ``"sobol"``, ``"halton"``, ``"lhs"``,
            ``"latin_hypercube"``, ``"normal"``, ``"prng"``).

    Returns:
        The matching :class:`RandomSequenceType` member.

    Raises:
        ValueError: If the string is not a recognised random sequence type.
        TypeError: If *method* is neither a string nor a :class:`RandomSequenceType`.
    """
    if isinstance(method, RandomSequenceType):
        return method
    if isinstance(method, str):
        cleaned = method.strip().lower()
        alias_map: dict[str, RandomSequenceType] = {
            "pseudo": RandomSequenceType.PSEUDO,
            "normal": RandomSequenceType.PSEUDO,
            "prng": RandomSequenceType.PSEUDO,
            "standard": RandomSequenceType.PSEUDO,
            "sobol": RandomSequenceType.SOBOL,
            "halton": RandomSequenceType.HALTON,
            "latin_hypercube": RandomSequenceType.LATIN_HYPERCUBE,
            "latinhypercube": RandomSequenceType.LATIN_HYPERCUBE,
            "lhs": RandomSequenceType.LATIN_HYPERCUBE,
        }
        if cleaned in alias_map:
            return alias_map[cleaned]
        msg = (
            f"Unsupported random sequence type '{method}'. Expected one of: "
            f"['pseudo', 'sobol', 'halton', 'latin_hypercube']."
        )
        raise ValueError(msg)
    msg = f"method must be a RandomSequenceType or str, got {type(method).__name__}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Thread-safe QMC sequence cache & stateful generator
# ---------------------------------------------------------------------------


class QMCSequenceCache:
    """Thread-safe LRU cache for generated quasi-random normal variates.

    Optimises bump-and-reval risk sensitivity calculations by storing and reusing
    variate matrices across identical simulation configurations.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._cache: collections.OrderedDict[tuple[typing.Any, ...], np.ndarray] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._enabled = True

    @property
    def hits(self) -> int:
        """Number of successful cache lookups."""
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Number of cache lookup misses."""
        with self._lock:
            return self._misses

    @property
    def size(self) -> int:
        """Current number of items stored in cache."""
        with self._lock:
            return len(self._cache)

    def is_enabled(self) -> bool:
        """Return True if caching is enabled."""
        with self._lock:
            return self._enabled

    def enable(self) -> None:
        """Enable sequence caching."""
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable sequence caching."""
        with self._lock:
            self._enabled = False

    def get(
        self,
        n_paths: int,
        dimension: int,
        random_type: RandomSequenceType | str,
        seed: int | None = None,
        scramble: bool = True,
    ) -> np.ndarray | None:
        """Retrieve cached normal draws array if available."""
        with self._lock:
            if not self._enabled:
                return None
            seq_type = _parse_random_sequence_type(random_type)
            key = (n_paths, dimension, seq_type.value, seed, scramble)
            if key in self._cache:
                self._hits += 1
                val = self._cache[key]
                self._cache.move_to_end(key)
                return val.copy()
            self._misses += 1
            return None

    def put(
        self,
        n_paths: int,
        dimension: int,
        random_type: RandomSequenceType | str,
        seed: int | None,
        scramble: bool,
        data: np.ndarray,
    ) -> None:
        """Store a generated normal draws array into cache."""
        with self._lock:
            if not self._enabled:
                return
            seq_type = _parse_random_sequence_type(random_type)
            key = (n_paths, dimension, seq_type.value, seed, scramble)
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = data.copy()
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Flush and reset all cached variates."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


_GLOBAL_QMC_CACHE = QMCSequenceCache()


def get_qmc_cache() -> QMCSequenceCache:
    """Return the global :class:`QMCSequenceCache` singleton."""
    return _GLOBAL_QMC_CACHE


def clear_qmc_cache() -> None:
    """Flush the global QMC variate cache."""
    _GLOBAL_QMC_CACHE.clear()


class QMCSequenceGenerator:
    """Stateful low-discrepancy sequence generator.

    Maintains internal engine state across sequential simulation blocks,
    enabling progressive sampling and stratified sensitivity analysis.
    """

    def __init__(
        self,
        dimension: int,
        random_type: RandomSequenceType | str = RandomSequenceType.SOBOL,
        seed: int | None = None,
        scramble: bool = True,
    ) -> None:
        """Initialize the stateful generator."""
        if dimension <= 0:
            msg = f"dimension must be positive, got {dimension}"
            raise ValueError(msg)
        self._dimension = dimension
        self._random_type = _parse_random_sequence_type(random_type)
        self._seed = seed
        self._scramble = scramble
        self._sampler: typing.Any = None
        self._total_drawn = 0
        self.reset()

    @property
    def dimension(self) -> int:
        """Number of random dimensions."""
        return self._dimension

    @property
    def random_type(self) -> RandomSequenceType:
        """Sequence generator type."""
        return self._random_type

    @property
    def total_drawn(self) -> int:
        """Total number of points drawn so far."""
        return self._total_drawn

    def reset(self) -> None:
        """Reset the generator to its initial state."""
        self._total_drawn = 0
        if self._random_type is RandomSequenceType.SOBOL:
            self._sampler = qmc.Sobol(
                d=self._dimension, scramble=self._scramble, seed=self._seed
            )
        elif self._random_type is RandomSequenceType.HALTON:
            self._sampler = qmc.Halton(
                d=self._dimension, scramble=self._scramble, seed=self._seed
            )
        elif self._random_type is RandomSequenceType.LATIN_HYPERCUBE:
            self._sampler = qmc.LatinHypercube(
                d=self._dimension, scramble=self._scramble, seed=self._seed
            )
        else:
            self._sampler = np.random.default_rng(self._seed)

    def draw(self, n_paths: int) -> np.ndarray:
        """Draw next batch of multi-dimensional standard normal variates."""
        if n_paths <= 0:
            msg = f"n_paths must be positive, got {n_paths}"
            raise ValueError(msg)

        if self._random_type is RandomSequenceType.PSEUDO:
            self._total_drawn += n_paths
            return np.asarray(
                self._sampler.standard_normal((n_paths, self._dimension)),
                dtype=np.float64,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*balance properties of Sobol.*",
                category=UserWarning,
            )
            uniform_samples = self._sampler.random(n=n_paths)

        self._total_drawn += n_paths
        u_clipped = np.clip(uniform_samples, 1e-15, 1.0 - 1e-15)
        return np.asarray(ndtri(u_clipped), dtype=np.float64)

    def draw_increments(
        self,
        n_paths: int,
        dt_vec: np.ndarray,
        num_factors: int = 1,
    ) -> np.ndarray:
        """Draw Brownian increments ΔW = Z * sqrt(Δt) from the generator."""
        dt_arr = np.asarray(dt_vec, dtype=np.float64)
        n_steps = len(dt_arr)
        if n_steps == 0:
            if num_factors == 1:
                return np.zeros((n_paths, 0), dtype=np.float64)
            return np.zeros((n_paths, 0, num_factors), dtype=np.float64)

        req_dim = n_steps * num_factors
        if self._dimension != req_dim:
            msg = (
                f"Generator dimension {self._dimension} does not match "
                f"required {req_dim} (n_steps={n_steps}, num_factors={num_factors})"
            )
            raise ValueError(msg)

        draws = self.draw(n_paths)
        sqrt_dt = np.sqrt(dt_arr)

        if num_factors == 1:
            return draws * sqrt_dt

        draws_3d = draws.reshape(n_paths, n_steps, num_factors)
        return draws_3d * sqrt_dt[:, np.newaxis]


# ---------------------------------------------------------------------------
# Core Generation Functions
# ---------------------------------------------------------------------------


def cached_normal_draws(
    n_paths: int,
    dimension: int,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = None,
    scramble: bool = True,
) -> np.ndarray:
    """Generate or retrieve cached standard normal variates.

    Args:
        n_paths: Number of paths (rows).
        dimension: Number of random dimensions (columns).
        random_type: Sequence type (:class:`RandomSequenceType` or str).
        seed: Random seed.
        scramble: Whether QMC scrambling is applied.

    Returns:
        Array of standard normal draws of shape ``(n_paths, dimension)``.
    """
    return generate_normal_draws(
        n_paths=n_paths,
        dimension=dimension,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
        use_cache=True,
    )


def generate_normal_draws(
    n_paths: int,
    dimension: int,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = None,
    scramble: bool = True,
    rng: np.random.Generator | None = None,
    use_cache: bool = False,
) -> np.ndarray:
    r"""Generate multi-dimensional standard normal variates :math:`\mathcal{N}(0, 1)`.

    For quasi-random sequences (Sobol, Halton, Latin Hypercube), samples are
    drawn uniformly from the unit hypercube :math:`[0, 1)^d` and transformed
    to standard normal variates using the inverse normal cumulative distribution
    function :math:`\Phi^{-1}` (`scipy.special.ndtri`).

    Args:
        n_paths: Number of simulation paths (rows). Must be strictly positive.
        dimension: Number of random dimensions / factors (columns).
            Must be strictly positive.
        random_type: Sequence type (:class:`RandomSequenceType` or string).
            Default is :attr:`RandomSequenceType.PSEUDO`.
        seed: Optional integer seed for reproducibility.
        scramble: If True (default), applies Owen scrambling to Sobol/Halton
            sequences for randomized QMC, providing unbiased estimators and
            valid standard error computation.
        rng: Optional pre-configured NumPy :class:`~numpy.random.Generator`.
            If provided and *random_type* is PSEUDO, this generator is used.
        use_cache: If True, checks and populates the global :class:`QMCSequenceCache`.

    Returns:
        2-D array of standard normal draws of shape ``(n_paths, dimension)``
        with ``dtype=np.float64``.

    Raises:
        ValueError: If *n_paths* <= 0 or *dimension* <= 0.
    """
    if n_paths <= 0:
        msg = f"n_paths must be strictly positive, got {n_paths}."
        raise ValueError(msg)
    if dimension <= 0:
        msg = f"dimension must be strictly positive, got {dimension}."
        raise ValueError(msg)

    seq_type = _parse_random_sequence_type(random_type)

    if use_cache and rng is None:
        cached = _GLOBAL_QMC_CACHE.get(
            n_paths=n_paths,
            dimension=dimension,
            random_type=seq_type,
            seed=seed,
            scramble=scramble,
        )
        if cached is not None:
            return cached

    if seq_type is RandomSequenceType.PSEUDO:
        generator = rng if rng is not None else np.random.default_rng(seed)
        draws = np.asarray(
            generator.standard_normal((n_paths, dimension)), dtype=np.float64
        )
    elif seq_type is RandomSequenceType.SOBOL:
        sampler = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*balance properties of Sobol.*",
                category=UserWarning,
            )
            uniform_samples = sampler.random(n=n_paths)
        u_clipped = np.clip(uniform_samples, 1e-15, 1.0 - 1e-15)
        draws = np.asarray(ndtri(u_clipped), dtype=np.float64)
    elif seq_type is RandomSequenceType.HALTON:
        sampler = qmc.Halton(d=dimension, scramble=scramble, seed=seed)
        uniform_samples = sampler.random(n=n_paths)
        u_clipped = np.clip(uniform_samples, 1e-15, 1.0 - 1e-15)
        draws = np.asarray(ndtri(u_clipped), dtype=np.float64)
    elif seq_type is RandomSequenceType.LATIN_HYPERCUBE:
        sampler = qmc.LatinHypercube(d=dimension, scramble=scramble, seed=seed)
        uniform_samples = sampler.random(n=n_paths)
        u_clipped = np.clip(uniform_samples, 1e-15, 1.0 - 1e-15)
        draws = np.asarray(ndtri(u_clipped), dtype=np.float64)
    else:  # pragma: no cover
        msg = f"Unhandled sequence type {seq_type}"
        raise ValueError(msg)

    if use_cache and rng is None:
        _GLOBAL_QMC_CACHE.put(
            n_paths=n_paths,
            dimension=dimension,
            random_type=seq_type,
            seed=seed,
            scramble=scramble,
            data=draws,
        )

    return draws


def generate_brownian_increments(
    n_paths: int,
    dt_vec: np.ndarray,
    num_factors: int = 1,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = None,
    scramble: bool = True,
    rng: np.random.Generator | None = None,
    use_cache: bool = False,
) -> np.ndarray:
    r"""Generate scaled Brownian increments :math:`\Delta W = Z \sqrt{\Delta t}`.

    Args:
        n_paths: Number of simulation paths.
        dt_vec: 1-D array of time step increments :math:`\Delta t_k`,
            shape ``(n_steps,)``.
        num_factors: Number of independent Brownian factors (default: 1).
        random_type: Sequence type (:class:`RandomSequenceType` or string).
        seed: Optional integer seed for reproducibility.
        scramble: If True (default), applies scrambling to QMC sequences.
        rng: Optional pre-configured NumPy :class:`~numpy.random.Generator`.
        use_cache: If True, utilizes variate caching.

    Returns:
        Array of Brownian increments. If ``num_factors == 1``, returns shape
        ``(n_paths, n_steps)``. If ``num_factors > 1``, returns shape
        ``(n_paths, n_steps, num_factors)``.
    """
    dt_arr = np.asarray(dt_vec, dtype=np.float64)
    n_steps = len(dt_arr)
    if n_steps == 0:
        if num_factors == 1:
            return np.zeros((n_paths, 0), dtype=np.float64)
        return np.zeros((n_paths, 0, num_factors), dtype=np.float64)

    total_dim = n_steps * num_factors
    draws = generate_normal_draws(
        n_paths=n_paths,
        dimension=total_dim,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
        rng=rng,
        use_cache=use_cache,
    )

    sqrt_dt = np.sqrt(dt_arr)

    if num_factors == 1:
        return draws * sqrt_dt

    draws_3d = draws.reshape(n_paths, n_steps, num_factors)
    return draws_3d * sqrt_dt[:, np.newaxis]


def compare_t0_npv_fitting(
    pricer_fn: typing.Callable[..., dict[str, typing.Any]],
    pricer_kwargs: dict[str, typing.Any],
    methods: tuple[RandomSequenceType | str, ...] = (
        RandomSequenceType.PSEUDO,
        RandomSequenceType.SOBOL,
        RandomSequenceType.HALTON,
        RandomSequenceType.LATIN_HYPERCUBE,
    ),
    n_paths: int = 10_000,
    seeds: tuple[int, ...] = (42, 101, 2024, 7, 999),
) -> dict[str, typing.Any]:
    r"""Benchmark T0 NPV fitting accuracy and variance reduction across QMC methods.

    Evaluates a Monte Carlo pricing function across multiple random seeds and
    sequence types (Pseudo-random, Sobol, Halton, Latin Hypercube) and computes:
    - Mean simulated price and analytical benchmark price.
    - Mean absolute pricing error :math:`\mathbb{E}[|\text{Error}|]`.
    - Maximum absolute pricing error.
    - Mean standard error.
    - Empirical variance across runs and variance reduction factor relative to PRNG.

    Args:
        pricer_fn: A pricing function (e.g.
            :func:`~xvasim.pricing_engine.price_interest_rate_swap`,
            :func:`~xvasim.pricing_engine.price_cross_currency_swap`,
            :func:`~xvasim.pricing_engine.price_foreign_exchange_option`).
        pricer_kwargs: Dictionary of pricing arguments passed to *pricer_fn*.
        methods: Tuple of sequence types to compare.
        n_paths: Number of Monte Carlo paths per simulation run.
        seeds: Sequence of seeds used to evaluate empirical dispersion.

    Returns:
        Dictionary containing comparative statistics per method and relative
        variance reduction factors.
    """
    results: dict[str, dict[str, float]] = {}
    benchmark_price: float | None = None

    for m in methods:
        seq_type = _parse_random_sequence_type(m)
        m_name = seq_type.value

        prices: list[float] = []
        std_errors: list[float] = []
        bench_prices: list[float] = []

        for s in seeds:
            kwargs = dict(pricer_kwargs)
            kwargs["n_paths"] = n_paths
            kwargs["seed"] = s
            kwargs["random_type"] = seq_type

            res = pricer_fn(**kwargs)
            price_val = float(res["price"])
            prices.append(price_val)

            if "std_error" in res:
                std_errors.append(float(res["std_error"]))

            if (
                "analytical_benchmark_price" in res
                and res["analytical_benchmark_price"] is not None
            ):
                bench_prices.append(float(res["analytical_benchmark_price"]))

        price_arr = np.array(prices, dtype=np.float64)
        mean_p = float(np.mean(price_arr))
        var_p = float(np.var(price_arr, ddof=1)) if len(prices) > 1 else 0.0

        if bench_prices:
            bench_val = bench_prices[0]
            benchmark_price = bench_val
            abs_errors = np.abs(price_arr - bench_val)
            mae = float(np.mean(abs_errors))
            max_err = float(np.max(abs_errors))
        else:
            mae = 0.0
            max_err = 0.0

        mean_se = float(np.mean(std_errors)) if std_errors else 0.0

        results[m_name] = {
            "mean_price": mean_p,
            "variance": var_p,
            "mean_std_error": mean_se,
            "mean_absolute_error": mae,
            "max_absolute_error": max_err,
        }

    # Compute variance reduction relative to pseudo
    pseudo_var = results.get("pseudo", {}).get("variance", 0.0)
    for _m_name, stats in results.items():
        if pseudo_var > 1e-18 and stats["variance"] > 1e-18:
            stats["variance_reduction_factor"] = float(pseudo_var / stats["variance"])
        else:
            stats["variance_reduction_factor"] = 1.0

    return {
        "benchmark_price": benchmark_price,
        "n_paths": n_paths,
        "num_runs": len(seeds),
        "methods": results,
    }

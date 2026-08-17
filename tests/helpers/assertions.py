"""Custom assertions and statistical test helpers for quantitative financial models."""

import numpy as np


def assert_mc_within_bounds(
    mc_price: float,
    mc_std_err: float,
    benchmark_price: float,
    num_std: float = 3.5,
    relative_tol: float = 0.05,
    min_std_floor: float = 1e-4,
) -> None:
    """Assert that a Monte Carlo estimated price is within statistical error bounds.

    Parameters
    ----------
    mc_price : float
        Simulated price from Monte Carlo engine.
    mc_std_err : float
        Standard error of the Monte Carlo estimator.
    benchmark_price : float
        Exact closed-form analytical benchmark price.
    num_std : float
        Number of standard errors allowed (default 3.5 = ~99.95% confidence).
    relative_tol : float
        Fallback relative tolerance if standard error is negligible.
    min_std_floor : float
        Minimum standard error floor to prevent division by zero.
    """
    diff = abs(mc_price - benchmark_price)
    effective_error = max(mc_std_err * num_std, abs(benchmark_price) * relative_tol)
    effective_error = max(effective_error, min_std_floor)

    if diff > effective_error:
        msg = (
            f"Monte Carlo price {mc_price:.6f} differs from benchmark "
            f"{benchmark_price:.6f} by {diff:.6f}, exceeding threshold "
            f"{effective_error:.6f} (std_err={mc_std_err:.6f}, num_std={num_std})"
        )
        raise AssertionError(msg)


def assert_martingale_property(
    discounted_paths: np.ndarray,
    initial_value: float,
    num_std: float = 3.5,
) -> None:
    """Assert that discounted price paths satisfy the Martingale property E[D(t)S(t)] = S(0).

    Parameters
    ----------
    discounted_paths : np.ndarray
        2D array of shape (n_paths, n_times) of discounted asset values.
    initial_value : float
        Expected value at time zero S(0).
    num_std : float
        Statistical tolerance in standard errors.
    """
    n_paths, n_times = discounted_paths.shape
    for t_idx in range(n_times):
        vals = discounted_paths[:, t_idx]
        mean_val = float(np.mean(vals))
        std_err = float(np.std(vals, ddof=1) / np.sqrt(n_paths))
        tol = max(num_std * std_err, 1e-4 * abs(initial_value))
        diff = abs(mean_val - initial_value)
        if diff > tol:
            msg = (
                f"Martingale test failed at step {t_idx}: E[D(t)S(t)] = {mean_val:.6f}, "
                f"expected {initial_value:.6f}, diff = {diff:.6f} > tol = {tol:.6f}"
            )
            raise AssertionError(msg)


def assert_put_call_parity(
    call_price: float,
    put_price: float,
    forward_price: float,
    discount_factor: float,
    strike: float,
    tolerance: float = 1e-5,
) -> None:
    """Assert European Put-Call parity: C - P = D * (F - K)."""
    lhs = call_price - put_price
    rhs = discount_factor * (forward_price - strike)
    diff = abs(lhs - rhs)
    if diff > tolerance:
        msg = (
            f"Put-Call Parity violated: C - P = {lhs:.6f}, "
            f"D * (F - K) = {rhs:.6f}, diff = {diff:.6e} > {tolerance:.6e}"
        )
        raise AssertionError(msg)


def assert_no_arbitrage_bounds(
    price: float,
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-6,
) -> None:
    """Assert that a derivative price satisfies no-arbitrage bounds."""
    if price < lower_bound - tolerance:
        msg = f"Price {price:.6f} violates lower bound {lower_bound:.6f}"
        raise AssertionError(msg)
    if price > upper_bound + tolerance:
        msg = f"Price {price:.6f} violates upper bound {upper_bound:.6f}"
        raise AssertionError(msg)

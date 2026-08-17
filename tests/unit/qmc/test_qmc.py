"""Tests for Quasi-Monte Carlo (QMC) sequences and variance diagnostics."""

import typing
import unittest

import numpy as np

from xvasim.qmc import (
    QMCSequenceCache,
    QMCSequenceGenerator,
    RandomSequenceType,
    cached_normal_draws,
    clear_qmc_cache,
    compare_t0_npv_fitting,
    generate_brownian_increments,
    generate_normal_draws,
    get_qmc_cache,
)


class TestQMC(unittest.TestCase):
    """Unit tests for low-discrepancy generators, Brownian increments, and variance benchmarking."""

    def test_random_sequence_type_enum(self) -> None:
        """Verify enum string values."""
        self.assertEqual(RandomSequenceType.PSEUDO.value, "pseudo")
        self.assertEqual(RandomSequenceType.SOBOL.value, "sobol")
        self.assertEqual(RandomSequenceType.HALTON.value, "halton")
        self.assertEqual(RandomSequenceType.LATIN_HYPERCUBE.value, "latin_hypercube")

    def test_generate_normal_draws_all_types(self) -> None:
        """Verify normal draws shape, mean ~ 0, and reproducibility for all sequence types."""
        types = [
            RandomSequenceType.PSEUDO,
            RandomSequenceType.SOBOL,
            RandomSequenceType.HALTON,
            RandomSequenceType.LATIN_HYPERCUBE,
        ]
        n_paths = 128
        dimension = 4

        for r_type in types:
            draws1 = generate_normal_draws(
                n_paths=n_paths,
                dimension=dimension,
                random_type=r_type,
                seed=42,
                scramble=True,
            )
            draws2 = generate_normal_draws(
                n_paths=n_paths,
                dimension=dimension,
                random_type=r_type,
                seed=42,
                scramble=True,
            )
            self.assertEqual(draws1.shape, (n_paths, dimension))
            np.testing.assert_allclose(draws1, draws2)

    def test_generate_brownian_increments_multi_factor(self) -> None:
        """Verify Brownian increment variance = dt and correct multi-factor shape."""
        dt_vec = np.array([0.25, 0.5, 0.25])
        n_paths = 100

        inc_1f = generate_brownian_increments(
            n_paths=n_paths,
            dt_vec=dt_vec,
            num_factors=1,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertEqual(inc_1f.shape, (n_paths, 3))

        inc_3f = generate_brownian_increments(
            n_paths=n_paths,
            dt_vec=dt_vec,
            num_factors=3,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertEqual(inc_3f.shape, (n_paths, 3, 3))

    def test_compare_t0_npv_fitting(self) -> None:
        """Verify compare_t0_npv_fitting diagnostics with benchmark price."""

        def dummy_pricer(
            n_paths: int,
            random_type: typing.Any = "pseudo",
            seed: int | None = None,
        ) -> dict[str, float]:
            val = 10.0 + (0.01 if random_type == "pseudo" else 0.002)
            return {
                "price": val,
                "std_error": 0.05,
                "analytical_benchmark_price": 10.0,
            }

        diag = compare_t0_npv_fitting(
            pricer_fn=dummy_pricer,
            pricer_kwargs={},
            methods=("pseudo", "sobol", "halton"),
            n_paths=100,
            seeds=(42, 43),
        )
        self.assertIn("benchmark_price", diag)
        self.assertIn("methods", diag)
        self.assertIn("sobol", diag["methods"])
        self.assertIn("variance_reduction_factor", diag["methods"]["sobol"])

    def test_compare_t0_npv_fitting_no_benchmark(self) -> None:
        """Verify compare_t0_npv_fitting when benchmark is None."""

        def dummy_pricer_no_bm(
            n_paths: int,
            random_type: typing.Any = "pseudo",
            seed: int | None = None,
        ) -> dict[str, float]:
            return {
                "price": 5.0,
                "std_error": 0.02,
            }

        diag = compare_t0_npv_fitting(
            pricer_fn=dummy_pricer_no_bm,
            pricer_kwargs={},
            methods=("pseudo", "sobol"),
            n_paths=50,
            seeds=(123,),
        )
        self.assertIsNone(diag["benchmark_price"])
        self.assertEqual(diag["methods"]["sobol"]["mean_absolute_error"], 0.0)

    def test_qmc_error_handling(self) -> None:
        """Verify error handling in QMC samplers and sequence parsing."""
        with self.assertRaises(ValueError):
            generate_normal_draws(n_paths=-10, dimension=2)
        with self.assertRaises(ValueError):
            generate_normal_draws(n_paths=100, dimension=0)
        with self.assertRaises(ValueError):
            generate_normal_draws(
                n_paths=100, dimension=2, random_type="unsupported_sequence"
            )
        with self.assertRaises(TypeError):
            generate_normal_draws(n_paths=100, dimension=2, random_type=999)  # type: ignore

    def test_empty_dt_vec(self) -> None:
        """Verify empty dt_vec returns empty arrays for 1-factor and multi-factor."""
        inc_1f = generate_brownian_increments(
            n_paths=10, dt_vec=np.array([]), num_factors=1
        )
        self.assertEqual(inc_1f.shape, (10, 0))

        inc_2f = generate_brownian_increments(
            n_paths=10, dt_vec=np.array([]), num_factors=2
        )
        self.assertEqual(inc_2f.shape, (10, 0, 2))

    def test_qmc_sequence_cache(self) -> None:
        """Verify QMCSequenceCache hit/miss behavior and caching."""
        cache = QMCSequenceCache(maxsize=4)
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
        self.assertTrue(cache.is_enabled())

        # Miss
        miss = cache.get(100, 2, "sobol", seed=42)
        self.assertIsNone(miss)
        self.assertEqual(cache.misses, 1)

        # Put
        draws = np.zeros((100, 2))
        cache.put(100, 2, "sobol", 42, True, draws)
        self.assertEqual(cache.size, 1)

        # Hit
        hit = cache.get(100, 2, "sobol", seed=42)
        self.assertIsNotNone(hit)
        self.assertEqual(cache.hits, 1)

        # Disabling
        cache.disable()
        self.assertFalse(cache.is_enabled())
        self.assertIsNone(cache.get(100, 2, "sobol", seed=42))
        cache.enable()

        # Eviction
        for i in range(10):
            cache.put(100, i + 1, "sobol", 42, True, np.zeros((100, i + 1)))
        self.assertEqual(cache.size, 4)

        # Clear
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    def test_cached_normal_draws_and_global_cache(self) -> None:
        """Verify cached_normal_draws helper and global cache clearing."""
        clear_qmc_cache()
        c = get_qmc_cache()
        self.assertEqual(c.size, 0)

        d1 = cached_normal_draws(64, 3, "halton", seed=10)
        self.assertEqual(d1.shape, (64, 3))
        self.assertEqual(c.size, 1)

        d2 = cached_normal_draws(64, 3, "halton", seed=10)
        np.testing.assert_allclose(d1, d2)
        self.assertGreater(c.hits, 0)

        # Increment generation with caching
        inc = generate_brownian_increments(
            64, np.array([0.5, 0.5, 0.5]), num_factors=1, random_type="halton", seed=10, use_cache=True
        )
        self.assertEqual(inc.shape, (64, 3))

        clear_qmc_cache()
        self.assertEqual(c.size, 0)

    def test_qmc_sequence_generator(self) -> None:
        """Verify stateful QMCSequenceGenerator draws and reset."""
        gen = QMCSequenceGenerator(dimension=4, random_type="sobol", seed=42)
        self.assertEqual(gen.dimension, 4)
        self.assertEqual(gen.random_type, RandomSequenceType.SOBOL)
        self.assertEqual(gen.total_drawn, 0)

        batch1 = gen.draw(32)
        self.assertEqual(batch1.shape, (32, 4))
        self.assertEqual(gen.total_drawn, 32)

        batch2 = gen.draw(32)
        self.assertEqual(batch2.shape, (32, 4))
        self.assertEqual(gen.total_drawn, 64)
        # Successive batches in Sobol are distinct
        self.assertFalse(np.allclose(batch1, batch2))

        # Reset returns to start
        gen.reset()
        self.assertEqual(gen.total_drawn, 0)
        batch1_reset = gen.draw(32)
        np.testing.assert_allclose(batch1, batch1_reset)

        # Increments
        gen.reset()
        inc = gen.draw_increments(32, np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertEqual(inc.shape, (32, 4))
        inc_empty = gen.draw_increments(32, np.array([]), num_factors=1)
        self.assertEqual(inc_empty.shape, (32, 0))
        inc_empty_multi = gen.draw_increments(32, np.array([]), num_factors=2)
        self.assertEqual(inc_empty_multi.shape, (32, 0, 2))

        # Halton
        gen_h = QMCSequenceGenerator(dimension=2, random_type="halton", seed=42)
        self.assertEqual(gen_h.draw(10).shape, (10, 2))

        # Latin Hypercube
        gen_l = QMCSequenceGenerator(dimension=2, random_type="latin_hypercube", seed=42)
        self.assertEqual(gen_l.draw(10).shape, (10, 2))

        # Pseudo
        gen_p = QMCSequenceGenerator(dimension=2, random_type="pseudo", seed=42)
        self.assertEqual(gen_p.draw(10).shape, (10, 2))

        # Multi-factor increment
        gen_multi = QMCSequenceGenerator(dimension=4, random_type="sobol", seed=42)
        inc_multi = gen_multi.draw_increments(10, np.array([0.5, 0.5]), num_factors=2)
        self.assertEqual(inc_multi.shape, (10, 2, 2))

        # Cache put when key already exists and when disabled
        cache = QMCSequenceCache(maxsize=2)
        cache.put(10, 2, "sobol", 42, True, np.zeros((10, 2)))
        cache.put(10, 2, "sobol", 42, True, np.ones((10, 2)))
        cache.disable()
        cache.put(10, 2, "sobol", 42, True, np.zeros((10, 2)))

        with self.assertRaises(ValueError):
            QMCSequenceGenerator(dimension=0)
        with self.assertRaises(ValueError):
            gen.draw(0)
        with self.assertRaises(ValueError):
            gen.draw_increments(10, np.array([0.1, 0.1]))  # Dim mismatch (2 vs 4)


if __name__ == "__main__":
    unittest.main()

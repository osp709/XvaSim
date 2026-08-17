"""End-to-end integration test: Swaption Calibration -> IR Simulation -> Exposure -> CIR Calibration -> CVA."""

import unittest

import numpy as np
import pytest

from xvasim.cva_engine import compute_cva, compute_marginal_pd
from xvasim.models.ir.lgm import LGMModel
from xvasim.qmc import RandomSequenceType


@pytest.mark.integration
class TestPortfolioCvaPipeline(unittest.TestCase):
    """End-to-end integration test of the complete CVA calculation pipeline."""

    def test_complete_cva_workflow(self) -> None:
        """Execute full end-to-end CVA simulation on an interest rate swap portfolio."""
        # 1. Market curves
        curve_tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
        curve_dfs = np.exp(-0.03 * curve_tenors)

        # 2. Calibrate LGM to swaptions
        swaption_expiries = np.array([1.0, 2.0, 5.0])
        swap_tenors = np.array([5.0, 5.0, 5.0])
        market_vols = np.array([0.0080, 0.0085, 0.0090])
        fixed_rates = np.array([0.03, 0.03, 0.03])

        lgm_model = LGMModel.calibrate_to_swaptions(
            swaption_expiries_yrs=swaption_expiries,
            swap_tenors_yrs=swap_tenors,
            market_normal_vols_ann=market_vols,
            curve_yrs=curve_tenors,
            curve_dfs=curve_dfs,
            fixed_rates_ann=fixed_rates,
            kappa_ann=0.03,
        )

        # 3. Simulate future rate paths
        sim_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        n_paths = 256
        x_paths = lgm_model.simulate_paths(
            times=sim_times,
            n_paths=n_paths,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        disc_paths = lgm_model.discount_path(sim_times, x_paths)

        # 4. Generate mock positive portfolio exposures
        # In a payer swap, exposure is max(V(t), 0)
        exposures = np.zeros((n_paths, len(sim_times)))
        for i, t in enumerate(sim_times):
            # Simulated swap NPV proxy proportional to x(t)
            pv_t = 100000.0 * (x_paths[:, i] * (5.0 - t))
            exposures[:, i] = np.maximum(pv_t, 0.0)

        # 5. Calibrate Credit Model & Compute Marginal PDs
        credit_tenors = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        market_spreads = np.array([0.010, 0.012, 0.014, 0.016, 0.020, 0.022, 0.025])
        marginal_pds = compute_marginal_pd(market_spreads, credit_tenors)

        # 6. Aggregate CVA across simulation steps > 0
        exp_steps = exposures[:, 1:]
        mpd_steps = np.tile(marginal_pds, (n_paths, 1))
        df_steps = disc_paths[:, 1:]

        cva_amount = compute_cva(
            exposure=exp_steps,
            marginal_pd=mpd_steps,
            discount_factor=df_steps,
            loss_given_default=0.60,
        )

        self.assertIsInstance(cva_amount, float)
        self.assertGreater(cva_amount, 0.0)


if __name__ == "__main__":
    unittest.main()

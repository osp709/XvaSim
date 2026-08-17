"""Tests for hardware acceleration and tensor backend abstraction."""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from xvasim.backend import (
    BackendType,
    CuPyBackend,
    JAXBackend,
    NumPyBackend,
    PyTorchBackend,
    TensorBackend,
    _parse_backend_type,
    available_backends,
    get_backend,
    is_backend_available,
    set_backend,
    use_backend,
)


class TestBackend(unittest.TestCase):
    """Unit tests for tensor backend abstraction and NumPy implementation."""

    def test_backend_type_enum(self) -> None:
        """Verify BackendType enum values and aliases."""
        self.assertEqual(BackendType.NUMPY.value, "numpy")
        self.assertEqual(BackendType.TORCH.value, "torch")
        self.assertEqual(BackendType.CUPY.value, "cupy")
        self.assertEqual(BackendType.JAX.value, "jax")

        self.assertEqual(_parse_backend_type("numpy"), BackendType.NUMPY)
        self.assertEqual(_parse_backend_type("np"), BackendType.NUMPY)
        self.assertEqual(_parse_backend_type("torch"), BackendType.TORCH)
        self.assertEqual(_parse_backend_type("pytorch"), BackendType.TORCH)
        self.assertEqual(_parse_backend_type("cupy"), BackendType.CUPY)
        self.assertEqual(_parse_backend_type("jax"), BackendType.JAX)
        self.assertEqual(
            _parse_backend_type(BackendType.NUMPY), BackendType.NUMPY
        )

        with self.assertRaises(ValueError):
            _parse_backend_type("unknown_backend")
        with self.assertRaises(TypeError):
            _parse_backend_type(123)  # type: ignore

    def test_is_backend_available(self) -> None:
        """NumPy should always be available."""
        self.assertTrue(is_backend_available(BackendType.NUMPY))
        self.assertTrue(is_backend_available("numpy"))
        avail = available_backends()
        self.assertIn(BackendType.NUMPY, avail)

    def test_numpy_backend_operations(self) -> None:
        """Verify core tensor operations on NumPyBackend."""
        bk: TensorBackend = get_backend("numpy", device="cpu")
        self.assertEqual(bk.name, BackendType.NUMPY)
        self.assertEqual(bk.device, "cpu")
        self.assertFalse(bk.is_gpu)

        arr = bk.array([1.0, 2.0, 3.0])
        self.assertEqual(arr.shape, (3,))
        np.testing.assert_allclose(bk.to_numpy(arr), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(bk.from_numpy(np.array([1.0, 2.0])), [1.0, 2.0])

        asarray = bk.asarray([4.0, 5.0])
        np.testing.assert_allclose(asarray, [4.0, 5.0])

        zeros = bk.zeros((2, 3))
        self.assertEqual(zeros.shape, (2, 3))
        self.assertEqual(np.sum(zeros), 0.0)

        ones = bk.ones((2, 3))
        self.assertEqual(ones.shape, (2, 3))
        self.assertEqual(np.sum(ones), 6.0)

        lin = bk.linspace(0.0, 1.0, 5)
        self.assertEqual(len(lin), 5)
        self.assertAlmostEqual(lin[-1], 1.0)

        diff = bk.diff(np.array([1.0, 3.0, 6.0]))
        np.testing.assert_allclose(diff, [2.0, 3.0])

        cumsum = bk.cumsum(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(cumsum, [1.0, 3.0, 6.0])

        sum_val = bk.sum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
        np.testing.assert_allclose(sum_val, [4.0, 6.0])

        mean_val = bk.mean(np.array([2.0, 4.0]))
        self.assertAlmostEqual(mean_val, 3.0)

        var_val = bk.var(np.array([1.0, 2.0, 3.0]), ddof=1)
        self.assertAlmostEqual(var_val, 1.0)

        exp_val = bk.exp(np.array([0.0, 1.0]))
        np.testing.assert_allclose(exp_val, [1.0, np.e])

        log_val = bk.log(np.array([1.0, np.e]))
        np.testing.assert_allclose(log_val, [0.0, 1.0])

        sqrt_val = bk.sqrt(np.array([4.0, 9.0]))
        np.testing.assert_allclose(sqrt_val, [2.0, 3.0])

        max_val = bk.maximum(np.array([1.0, 5.0]), np.array([3.0, 2.0]))
        np.testing.assert_allclose(max_val, [3.0, 5.0])

        clip_val = bk.clip(np.array([-1.0, 0.5, 2.0]), 0.0, 1.0)
        np.testing.assert_allclose(clip_val, [0.0, 0.5, 1.0])

        abs_val = bk.abs(np.array([-2.5, 3.0]))
        np.testing.assert_allclose(abs_val, [2.5, 3.0])

        m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        m2 = np.array([[2.0, 0.0], [1.0, 2.0]])
        mm = bk.matmul(m1, m2)
        np.testing.assert_allclose(mm, [[4.0, 4.0], [10.0, 8.0]])

        cov = np.array([[4.0, 2.0], [2.0, 5.0]])
        chol = bk.cholesky(cov)
        np.testing.assert_allclose(chol @ chol.T, cov)

        normals = bk.standard_normal((100, 2), seed=42)
        self.assertEqual(normals.shape, (100, 2))

    def test_context_manager_and_global_backend(self) -> None:
        """Verify use_backend context manager and set_backend."""
        initial = get_backend()
        self.assertEqual(initial.name, BackendType.NUMPY)

        with use_backend("numpy", device="cpu") as bk:
            self.assertEqual(bk.name, BackendType.NUMPY)
            self.assertEqual(get_backend().name, BackendType.NUMPY)
            with use_backend("numpy", device="cpu") as bk_nested:
                self.assertEqual(bk_nested.name, BackendType.NUMPY)

        set_backend("numpy", device="cpu")
        self.assertEqual(get_backend().name, BackendType.NUMPY)

    def test_get_backend_dispatches(self) -> None:
        """Verify get_backend dispatches to the requested backend."""
        mock_mod = MagicMock()
        mock_mod.device = lambda d: d
        with (
            patch.dict(sys.modules, {"torch": mock_mod, "cupy": mock_mod, "jax": mock_mod, "jax.numpy": mock_mod}),
            patch("xvasim.backend.is_backend_available", return_value=True),
        ):
            b_torch = get_backend("torch")
            self.assertEqual(b_torch.name, BackendType.TORCH)
            b_cupy = get_backend("cupy")
            self.assertEqual(b_cupy.name, BackendType.CUPY)
            b_jax = get_backend("jax")
            self.assertEqual(b_jax.name, BackendType.JAX)

    def test_gpu_property(self) -> None:
        """Verify is_gpu property for cuda devices."""
        bk_cuda = NumPyBackend(device="cuda:0")
        self.assertTrue(bk_cuda.is_gpu)
        bk_cpu = NumPyBackend(device="cpu")
        self.assertFalse(bk_cpu.is_gpu)

    def test_missing_backend_raises_runtime_error(self) -> None:
        """Instantiating uninstalled backends raises RuntimeError."""
        with patch("xvasim.backend.is_backend_available", return_value=False):
            with self.assertRaises(RuntimeError):
                PyTorchBackend(device="cpu")
            with self.assertRaises(RuntimeError):
                CuPyBackend(device="cuda:0")
            with self.assertRaises(RuntimeError):
                JAXBackend(device="cpu")

    def test_mock_pytorch_backend(self) -> None:
        """Verify PyTorchBackend methods using a mock torch module."""
        mock_torch = MagicMock()
        mock_torch.device = lambda d: d
        mock_torch.float64 = "float64"
        mock_torch.Tensor = type("Tensor", (), {})
        mock_tensor = MagicMock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = (
            np.array([1.0, 2.0])
        )
        mock_torch.from_numpy.return_value.to.return_value = mock_tensor
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.as_tensor.return_value = mock_tensor
        mock_torch.zeros.return_value = mock_tensor
        mock_torch.ones.return_value = mock_tensor
        mock_torch.linspace.return_value = mock_tensor
        mock_torch.diff.return_value = mock_tensor
        mock_torch.cumsum.return_value = mock_tensor
        mock_torch.sum.return_value = mock_tensor
        mock_torch.mean.return_value = mock_tensor
        mock_torch.var.return_value = mock_tensor
        mock_torch.exp.return_value = mock_tensor
        mock_torch.log.return_value = mock_tensor
        mock_torch.sqrt.return_value = mock_tensor
        mock_torch.maximum.return_value = mock_tensor
        mock_torch.clamp.return_value = mock_tensor
        mock_torch.abs.return_value = mock_tensor
        mock_torch.matmul.return_value = mock_tensor
        mock_torch.linalg.cholesky.return_value = mock_tensor
        mock_torch.randn.return_value = mock_tensor
        mock_torch.Generator.return_value = MagicMock()

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch("xvasim.backend.is_backend_available", return_value=True),
        ):
            bk = PyTorchBackend(device="cpu")
            self.assertEqual(bk.name, BackendType.TORCH)
            self.assertEqual(bk.device, "cpu")
            bk.to_numpy(mock_tensor)
            bk.to_numpy(np.array([1.0, 2.0]))
            bk.from_numpy(np.array([1.0, 2.0]))
            bk.array([1.0, 2.0])
            bk.asarray([1.0, 2.0])
            bk.zeros((2, 2))
            bk.zeros(2)
            bk.ones((2, 2))
            bk.ones(2)
            bk.linspace(0.0, 1.0, 5)
            bk.diff(mock_tensor)
            bk.cumsum(mock_tensor)
            bk.sum(mock_tensor)
            bk.sum(mock_tensor, axis=0)
            bk.mean(mock_tensor)
            bk.mean(mock_tensor, axis=0)
            bk.var(mock_tensor)
            bk.var(mock_tensor, axis=0)
            bk.exp(mock_tensor)
            bk.log(mock_tensor)
            bk.sqrt(mock_tensor)
            bk.maximum(mock_tensor, mock_tensor)
            bk.clip(mock_tensor, 0.0, 1.0)
            bk.abs(mock_tensor)
            bk.matmul(mock_tensor, mock_tensor)
            bk.cholesky(mock_tensor)
            bk.standard_normal((10, 2), seed=42)

    def test_mock_cupy_backend(self) -> None:
        """Verify CuPyBackend methods using a mock cupy module."""
        mock_cp = MagicMock()
        mock_cp.float64 = "float64"
        mock_cp.ndarray = type("ndarray", (), {})
        mock_arr = MagicMock()
        mock_arr.asnumpy.return_value = np.array([1.0, 2.0])
        mock_cp.asarray.return_value = mock_arr
        mock_cp.array.return_value = mock_arr
        mock_cp.zeros.return_value = mock_arr
        mock_cp.ones.return_value = mock_arr
        mock_cp.linspace.return_value = mock_arr
        mock_cp.diff.return_value = mock_arr
        mock_cp.cumsum.return_value = mock_arr
        mock_cp.sum.return_value = mock_arr
        mock_cp.mean.return_value = mock_arr
        mock_cp.var.return_value = mock_arr
        mock_cp.exp.return_value = mock_arr
        mock_cp.log.return_value = mock_arr
        mock_cp.sqrt.return_value = mock_arr
        mock_cp.maximum.return_value = mock_arr
        mock_cp.clip.return_value = mock_arr
        mock_cp.abs.return_value = mock_arr
        mock_cp.linalg.cholesky.return_value = mock_arr
        mock_rng = MagicMock()
        mock_rng.standard_normal.return_value = mock_arr
        mock_cp.random.default_rng.return_value = mock_rng

        with (
            patch.dict(sys.modules, {"cupy": mock_cp}),
            patch("xvasim.backend.is_backend_available", return_value=True),
        ):
            bk = CuPyBackend(device="cuda:0")
            self.assertEqual(bk.name, BackendType.CUPY)
            self.assertEqual(bk.device, "cuda:0")
            bk.to_numpy(mock_arr)
            bk.to_numpy(np.array([1.0, 2.0]))
            bk.from_numpy(np.array([1.0, 2.0]))
            bk.array([1.0, 2.0])
            bk.asarray([1.0, 2.0])
            bk.zeros((2, 2))
            bk.ones((2, 2))
            bk.linspace(0.0, 1.0, 5)
            bk.diff(mock_arr)
            bk.cumsum(mock_arr)
            bk.sum(mock_arr)
            bk.mean(mock_arr)
            bk.var(mock_arr)
            bk.exp(mock_arr)
            bk.log(mock_arr)
            bk.sqrt(mock_arr)
            bk.maximum(mock_arr, mock_arr)
            bk.clip(mock_arr, 0.0, 1.0)
            bk.abs(mock_arr)
            bk.matmul(mock_arr, mock_arr)
            bk.cholesky(mock_arr)
            bk.standard_normal((10, 2), seed=42)

    def test_mock_jax_backend(self) -> None:
        """Verify JAXBackend methods using a mock jax module."""
        mock_jax = MagicMock()
        mock_jnp = MagicMock()
        mock_jnp.float64 = "float64"
        mock_arr = MagicMock()
        mock_jnp.asarray.return_value = mock_arr
        mock_jnp.array.return_value = mock_arr
        mock_jnp.zeros.return_value = mock_arr
        mock_jnp.ones.return_value = mock_arr
        mock_jnp.linspace.return_value = mock_arr
        mock_jnp.diff.return_value = mock_arr
        mock_jnp.cumsum.return_value = mock_arr
        mock_jnp.sum.return_value = mock_arr
        mock_jnp.mean.return_value = mock_arr
        mock_jnp.var.return_value = mock_arr
        mock_jnp.exp.return_value = mock_arr
        mock_jnp.log.return_value = mock_arr
        mock_jnp.sqrt.return_value = mock_arr
        mock_jnp.maximum.return_value = mock_arr
        mock_jnp.clip.return_value = mock_arr
        mock_jnp.abs.return_value = mock_arr
        mock_jnp.linalg.cholesky.return_value = mock_arr
        mock_jax.random.PRNGKey.return_value = 0
        mock_jax.random.normal.return_value = mock_arr

        with (
            patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}),
            patch("xvasim.backend.is_backend_available", return_value=True),
        ):
            bk = JAXBackend(device="cpu")
            self.assertEqual(bk.name, BackendType.JAX)
            self.assertEqual(bk.device, "cpu")
            bk.to_numpy(mock_arr)
            bk.from_numpy(np.array([1.0, 2.0]))
            bk.array([1.0, 2.0])
            bk.asarray([1.0, 2.0])
            bk.zeros((2, 2))
            bk.ones((2, 2))
            bk.linspace(0.0, 1.0, 5)
            bk.diff(mock_arr)
            bk.cumsum(mock_arr)
            bk.sum(mock_arr)
            bk.mean(mock_arr)
            bk.var(mock_arr)
            bk.exp(mock_arr)
            bk.log(mock_arr)
            bk.sqrt(mock_arr)
            bk.maximum(mock_arr, mock_arr)
            bk.clip(mock_arr, 0.0, 1.0)
            bk.abs(mock_arr)
            bk.matmul(mock_arr, mock_arr)
            bk.cholesky(mock_arr)
            bk.standard_normal((10, 2), seed=42)
            bk.standard_normal((10, 2), seed=None)


if __name__ == "__main__":
    unittest.main()

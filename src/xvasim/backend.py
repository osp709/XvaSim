"""Hardware acceleration and tensor backend abstraction for XvaSim.

This module provides a unified abstraction layer for tensor operations and
random variate generation across multiple numerical backends:
- **NumPy** (CPU default)
- **PyTorch** (CPU / CUDA / MPS)
- **CuPy** (CUDA GPU)
- **JAX** (CPU / GPU / TPU)

Public API
----------
- :class:`BackendType` — enumeration of supported backend systems.
- :class:`TensorBackend` — abstract base class for tensor execution backends.
- :class:`NumPyBackend` — NumPy CPU tensor backend.
- :class:`PyTorchBackend` — PyTorch tensor backend.
- :class:`CuPyBackend` — CuPy GPU tensor backend.
- :class:`JAXBackend` — JAX tensor backend.
- :func:`available_backends` — list currently installed/available backends.
- :func:`get_backend` — retrieve the active or specified backend instance.
- :func:`is_backend_available` — check if a backend package is installed.
- :func:`set_backend` — set the global default backend and device.
- :func:`use_backend` — context manager for scoped backend execution.
"""

from __future__ import annotations

import abc
import contextlib
import enum
import importlib
import importlib.util
import threading
import typing

import numpy as np

__all__ = [
    "BackendType",
    "CuPyBackend",
    "JAXBackend",
    "NumPyBackend",
    "PyTorchBackend",
    "TensorBackend",
    "available_backends",
    "get_backend",
    "is_backend_available",
    "set_backend",
    "use_backend",
]


class BackendType(enum.Enum):
    """Enumeration of supported tensor calculation and hardware backends."""

    NUMPY = "numpy"
    TORCH = "torch"
    CUPY = "cupy"
    JAX = "jax"


def _parse_backend_type(backend: BackendType | str) -> BackendType:
    """Parse and validate a BackendType enum or string representation."""
    if isinstance(backend, BackendType):
        return backend
    if isinstance(backend, str):
        cleaned = backend.strip().lower()
        alias_map = {
            "numpy": BackendType.NUMPY,
            "np": BackendType.NUMPY,
            "torch": BackendType.TORCH,
            "pytorch": BackendType.TORCH,
            "cupy": BackendType.CUPY,
            "jax": BackendType.JAX,
        }
        if cleaned in alias_map:
            return alias_map[cleaned]
        msg = (
            f"Unsupported backend '{backend}'. Expected one of: "
            f"['numpy', 'torch', 'cupy', 'jax']."
        )
        raise ValueError(msg)
    msg = f"backend must be a BackendType or str, got {type(backend).__name__}"
    raise TypeError(msg)


def is_backend_available(backend: BackendType | str) -> bool:
    """Check whether the underlying library for the specified backend is installed.

    Args:
        backend: :class:`BackendType` or backend name string.

    Returns:
        True if the backend library is importable, False otherwise.
    """
    b_type = _parse_backend_type(backend)
    if b_type is BackendType.NUMPY:
        return True
    pkg_map = {
        BackendType.TORCH: "torch",
        BackendType.CUPY: "cupy",
        BackendType.JAX: "jax",
    }
    pkg_name = pkg_map[b_type]
    return importlib.util.find_spec(pkg_name) is not None


def available_backends() -> list[BackendType]:
    """Return a list of all currently installed and available backends."""
    return [b for b in BackendType if is_backend_available(b)]


class TensorBackend(abc.ABC):
    """Abstract base class defining tensor operations and array lifecycle."""

    @property
    @abc.abstractmethod
    def name(self) -> BackendType:
        """The identifier of this backend."""
        ...

    @property
    @abc.abstractmethod
    def device(self) -> str:
        """Device name (e.g. 'cpu', 'cuda', 'cuda:0', 'mps')."""
        ...

    @property
    def is_gpu(self) -> bool:
        """Return True if this backend executes on a GPU device."""
        dev = self.device.lower()
        return "cuda" in dev or "gpu" in dev

    @abc.abstractmethod
    def to_numpy(self, tensor: typing.Any) -> np.ndarray:
        """Convert a backend tensor or array to a contiguous NumPy ndarray."""
        ...

    @abc.abstractmethod
    def from_numpy(self, arr: np.ndarray) -> typing.Any:
        """Convert a NumPy ndarray to a native backend tensor on the target device."""
        ...

    @abc.abstractmethod
    def array(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        """Construct a new backend array/tensor."""
        ...

    @abc.abstractmethod
    def asarray(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        """Convert input to a backend array/tensor if not already one."""
        ...

    @abc.abstractmethod
    def zeros(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        """Construct an array of zeros."""
        ...

    @abc.abstractmethod
    def ones(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        """Construct an array of ones."""
        ...

    @abc.abstractmethod
    def linspace(
        self, start: float, stop: float, num: int, dtype: typing.Any = None
    ) -> typing.Any:
        """Construct evenly spaced values over a specified interval."""
        ...

    @abc.abstractmethod
    def diff(self, a: typing.Any, axis: int = -1) -> typing.Any:
        """Calculate the n-th discrete difference along the given axis."""
        ...

    @abc.abstractmethod
    def cumsum(self, a: typing.Any, axis: int = -1) -> typing.Any:
        """Return the cumulative sum of elements along a given axis."""
        ...

    @abc.abstractmethod
    def sum(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        """Sum array elements over a given axis."""
        ...

    @abc.abstractmethod
    def mean(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        """Compute the arithmetic mean along the specified axis."""
        ...

    @abc.abstractmethod
    def var(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> typing.Any:
        """Compute the variance along the specified axis."""
        ...

    @abc.abstractmethod
    def exp(self, a: typing.Any) -> typing.Any:
        """Calculate the exponential of all elements in the input array."""
        ...

    @abc.abstractmethod
    def log(self, a: typing.Any) -> typing.Any:
        """Natural logarithm, element-wise."""
        ...

    @abc.abstractmethod
    def sqrt(self, a: typing.Any) -> typing.Any:
        """Return the non-negative square-root of an array, element-wise."""
        ...

    @abc.abstractmethod
    def maximum(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        """Element-wise maximum of array elements."""
        ...

    @abc.abstractmethod
    def clip(self, a: typing.Any, a_min: typing.Any, a_max: typing.Any) -> typing.Any:
        """Clip (limit) values in an array."""
        ...

    @abc.abstractmethod
    def abs(self, a: typing.Any) -> typing.Any:
        """Calculate absolute values element-wise."""
        ...

    @abc.abstractmethod
    def matmul(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        """Matrix product of two arrays."""
        ...

    @abc.abstractmethod
    def cholesky(self, a: typing.Any) -> typing.Any:
        """Cholesky decomposition of a symmetric positive-definite matrix."""
        ...

    @abc.abstractmethod
    def standard_normal(
        self, shape: tuple[int, ...], seed: int | None = None
    ) -> typing.Any:
        """Generate standard normal random variates N(0, 1)."""
        ...


class NumPyBackend(TensorBackend):
    """Default CPU tensor backend using NumPy."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device

    @property
    def name(self) -> BackendType:
        return BackendType.NUMPY

    @property
    def device(self) -> str:
        return self._device

    def to_numpy(self, tensor: typing.Any) -> np.ndarray:
        return np.asarray(tensor, dtype=np.float64)

    def from_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64)

    def array(self, obj: typing.Any, dtype: typing.Any = None) -> np.ndarray:
        dt = np.float64 if dtype is None else dtype
        return np.array(obj, dtype=dt)

    def asarray(self, obj: typing.Any, dtype: typing.Any = None) -> np.ndarray:
        dt = np.float64 if dtype is None else dtype
        return np.asarray(obj, dtype=dt)

    def zeros(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> np.ndarray:
        dt = np.float64 if dtype is None else dtype
        return np.zeros(shape, dtype=dt)

    def ones(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> np.ndarray:
        dt = np.float64 if dtype is None else dtype
        return np.ones(shape, dtype=dt)

    def linspace(
        self, start: float, stop: float, num: int, dtype: typing.Any = None
    ) -> np.ndarray:
        dt = np.float64 if dtype is None else dtype
        return np.linspace(start, stop, num, dtype=dt)

    def diff(self, a: typing.Any, axis: int = -1) -> np.ndarray:
        return np.diff(np.asarray(a), axis=axis)

    def cumsum(self, a: typing.Any, axis: int = -1) -> np.ndarray:
        return np.cumsum(np.asarray(a), axis=axis)

    def sum(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        return np.sum(np.asarray(a), axis=axis, keepdims=keepdims)

    def mean(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        return np.mean(np.asarray(a), axis=axis, keepdims=keepdims)

    def var(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> typing.Any:
        return np.var(np.asarray(a), axis=axis, ddof=ddof, keepdims=keepdims)

    def exp(self, a: typing.Any) -> np.ndarray:
        return np.exp(np.asarray(a))

    def log(self, a: typing.Any) -> np.ndarray:
        return np.log(np.asarray(a))

    def sqrt(self, a: typing.Any) -> np.ndarray:
        return np.sqrt(np.asarray(a))

    def maximum(self, x1: typing.Any, x2: typing.Any) -> np.ndarray:
        return np.maximum(x1, x2)

    def clip(self, a: typing.Any, a_min: typing.Any, a_max: typing.Any) -> np.ndarray:
        return np.clip(a, a_min, a_max)

    def abs(self, a: typing.Any) -> np.ndarray:
        return np.abs(np.asarray(a))

    def matmul(self, x1: typing.Any, x2: typing.Any) -> np.ndarray:
        return np.asarray(x1) @ np.asarray(x2)

    def cholesky(self, a: typing.Any) -> np.ndarray:
        return np.linalg.cholesky(np.asarray(a, dtype=np.float64))

    def standard_normal(
        self, shape: tuple[int, ...], seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape, dtype=np.float64)


class PyTorchBackend(TensorBackend):
    """PyTorch tensor backend supporting CPU, CUDA, and Apple MPS devices."""

    def __init__(self, device: str = "cpu") -> None:
        if not is_backend_available(BackendType.TORCH):
            msg = "PyTorch is not installed. Install torch to use PyTorchBackend."
            raise RuntimeError(msg)
        torch = importlib.import_module("torch")
        self._torch = torch
        self._device = torch.device(device)

    @property
    def name(self) -> BackendType:
        return BackendType.TORCH

    @property
    def device(self) -> str:
        return str(self._device)

    def to_numpy(self, tensor: typing.Any) -> np.ndarray:
        if isinstance(tensor, self._torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor, dtype=np.float64)

    def from_numpy(self, arr: np.ndarray) -> typing.Any:
        return self._torch.from_numpy(arr).to(
            device=self._device, dtype=self._torch.float64
        )

    def array(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        dt = self._torch.float64 if dtype is None else dtype
        return self._torch.tensor(obj, dtype=dt, device=self._device)

    def asarray(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        dt = self._torch.float64 if dtype is None else dtype
        return self._torch.as_tensor(obj, dtype=dt, device=self._device)

    def zeros(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._torch.float64 if dtype is None else dtype
        sh = (shape,) if isinstance(shape, int) else shape
        return self._torch.zeros(sh, dtype=dt, device=self._device)

    def ones(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._torch.float64 if dtype is None else dtype
        sh = (shape,) if isinstance(shape, int) else shape
        return self._torch.ones(sh, dtype=dt, device=self._device)

    def linspace(
        self, start: float, stop: float, num: int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._torch.float64 if dtype is None else dtype
        return self._torch.linspace(start, stop, num, dtype=dt, device=self._device)

    def diff(self, a: typing.Any, axis: int = -1) -> typing.Any:
        t = self.asarray(a)
        return self._torch.diff(t, dim=axis)

    def cumsum(self, a: typing.Any, axis: int = -1) -> typing.Any:
        t = self.asarray(a)
        return self._torch.cumsum(t, dim=axis)

    def sum(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        t = self.asarray(a)
        if axis is None:
            return self._torch.sum(t)
        return self._torch.sum(t, dim=axis, keepdim=keepdims)

    def mean(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        t = self.asarray(a)
        if axis is None:
            return self._torch.mean(t)
        return self._torch.mean(t, dim=axis, keepdim=keepdims)

    def var(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> typing.Any:
        t = self.asarray(a)
        correction = ddof
        if axis is None:
            return self._torch.var(t, correction=correction)
        return self._torch.var(t, dim=axis, correction=correction, keepdim=keepdims)

    def exp(self, a: typing.Any) -> typing.Any:
        return self._torch.exp(self.asarray(a))

    def log(self, a: typing.Any) -> typing.Any:
        return self._torch.log(self.asarray(a))

    def sqrt(self, a: typing.Any) -> typing.Any:
        return self._torch.sqrt(self.asarray(a))

    def maximum(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        t1 = self.asarray(x1)
        t2 = self.asarray(x2)
        return self._torch.maximum(t1, t2)

    def clip(self, a: typing.Any, a_min: typing.Any, a_max: typing.Any) -> typing.Any:
        t = self.asarray(a)
        return self._torch.clamp(t, min=a_min, max=a_max)

    def abs(self, a: typing.Any) -> typing.Any:
        return self._torch.abs(self.asarray(a))

    def matmul(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        return self._torch.matmul(self.asarray(x1), self.asarray(x2))

    def cholesky(self, a: typing.Any) -> typing.Any:
        return self._torch.linalg.cholesky(self.asarray(a))

    def standard_normal(
        self, shape: tuple[int, ...], seed: int | None = None
    ) -> typing.Any:
        generator = None
        if seed is not None:
            generator = self._torch.Generator(device=self._device)
            generator.manual_seed(seed)
        return self._torch.randn(
            shape,
            generator=generator,
            dtype=self._torch.float64,
            device=self._device,
        )


class CuPyBackend(TensorBackend):
    """CuPy GPU tensor backend for native CUDA execution."""

    def __init__(self, device: str = "cuda:0") -> None:
        if not is_backend_available(BackendType.CUPY):
            msg = "CuPy is not installed. Install cupy to use CuPyBackend."
            raise RuntimeError(msg)
        cupy = importlib.import_module("cupy")
        self._cp = cupy
        self._device = device

    @property
    def name(self) -> BackendType:
        return BackendType.CUPY

    @property
    def device(self) -> str:
        return self._device

    def to_numpy(self, tensor: typing.Any) -> np.ndarray:
        if isinstance(tensor, self._cp.ndarray):
            return self._cp.asnumpy(tensor)
        return np.asarray(tensor, dtype=np.float64)

    def from_numpy(self, arr: np.ndarray) -> typing.Any:
        return self._cp.asarray(arr, dtype=self._cp.float64)

    def array(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        dt = self._cp.float64 if dtype is None else dtype
        return self._cp.array(obj, dtype=dt)

    def asarray(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        dt = self._cp.float64 if dtype is None else dtype
        return self._cp.asarray(obj, dtype=dt)

    def zeros(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._cp.float64 if dtype is None else dtype
        return self._cp.zeros(shape, dtype=dt)

    def ones(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._cp.float64 if dtype is None else dtype
        return self._cp.ones(shape, dtype=dt)

    def linspace(
        self, start: float, stop: float, num: int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._cp.float64 if dtype is None else dtype
        return self._cp.linspace(start, stop, num, dtype=dt)

    def diff(self, a: typing.Any, axis: int = -1) -> typing.Any:
        return self._cp.diff(self.asarray(a), axis=axis)

    def cumsum(self, a: typing.Any, axis: int = -1) -> typing.Any:
        return self._cp.cumsum(self.asarray(a), axis=axis)

    def sum(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        return self._cp.sum(self.asarray(a), axis=axis, keepdims=keepdims)

    def mean(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        return self._cp.mean(self.asarray(a), axis=axis, keepdims=keepdims)

    def var(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> typing.Any:
        return self._cp.var(self.asarray(a), axis=axis, ddof=ddof, keepdims=keepdims)

    def exp(self, a: typing.Any) -> typing.Any:
        return self._cp.exp(self.asarray(a))

    def log(self, a: typing.Any) -> typing.Any:
        return self._cp.log(self.asarray(a))

    def sqrt(self, a: typing.Any) -> typing.Any:
        return self._cp.sqrt(self.asarray(a))

    def maximum(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        return self._cp.maximum(self.asarray(x1), self.asarray(x2))

    def clip(self, a: typing.Any, a_min: typing.Any, a_max: typing.Any) -> typing.Any:
        return self._cp.clip(self.asarray(a), a_min, a_max)

    def abs(self, a: typing.Any) -> typing.Any:
        return self._cp.abs(self.asarray(a))

    def matmul(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        return self.asarray(x1) @ self.asarray(x2)

    def cholesky(self, a: typing.Any) -> typing.Any:
        return self._cp.linalg.cholesky(self.asarray(a))

    def standard_normal(
        self, shape: tuple[int, ...], seed: int | None = None
    ) -> typing.Any:
        rng = self._cp.random.default_rng(seed)
        return rng.standard_normal(shape, dtype=self._cp.float64)


class JAXBackend(TensorBackend):
    """JAX backend supporting XLA-compiled CPU, GPU, and TPU execution."""

    def __init__(self, device: str = "cpu") -> None:
        if not is_backend_available(BackendType.JAX):
            msg = "JAX is not installed. Install jax and jaxlib to use JAXBackend."
            raise RuntimeError(msg)
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        self._jax = jax
        self._jnp = jnp
        self._device = device

    @property
    def name(self) -> BackendType:
        return BackendType.JAX

    @property
    def device(self) -> str:
        return self._device

    def to_numpy(self, tensor: typing.Any) -> np.ndarray:
        return np.asarray(tensor, dtype=np.float64)

    def from_numpy(self, arr: np.ndarray) -> typing.Any:
        return self._jnp.asarray(arr, dtype=self._jnp.float64)

    def array(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        dt = self._jnp.float64 if dtype is None else dtype
        return self._jnp.array(obj, dtype=dt)

    def asarray(self, obj: typing.Any, dtype: typing.Any = None) -> typing.Any:
        dt = self._jnp.float64 if dtype is None else dtype
        return self._jnp.asarray(obj, dtype=dt)

    def zeros(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._jnp.float64 if dtype is None else dtype
        return self._jnp.zeros(shape, dtype=dt)

    def ones(
        self, shape: tuple[int, ...] | int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._jnp.float64 if dtype is None else dtype
        return self._jnp.ones(shape, dtype=dt)

    def linspace(
        self, start: float, stop: float, num: int, dtype: typing.Any = None
    ) -> typing.Any:
        dt = self._jnp.float64 if dtype is None else dtype
        return self._jnp.linspace(start, stop, num, dtype=dt)

    def diff(self, a: typing.Any, axis: int = -1) -> typing.Any:
        return self._jnp.diff(self.asarray(a), axis=axis)

    def cumsum(self, a: typing.Any, axis: int = -1) -> typing.Any:
        return self._jnp.cumsum(self.asarray(a), axis=axis)

    def sum(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        return self._jnp.sum(self.asarray(a), axis=axis, keepdims=keepdims)

    def mean(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> typing.Any:
        return self._jnp.mean(self.asarray(a), axis=axis, keepdims=keepdims)

    def var(
        self,
        a: typing.Any,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> typing.Any:
        return self._jnp.var(self.asarray(a), axis=axis, ddof=ddof, keepdims=keepdims)

    def exp(self, a: typing.Any) -> typing.Any:
        return self._jnp.exp(self.asarray(a))

    def log(self, a: typing.Any) -> typing.Any:
        return self._jnp.log(self.asarray(a))

    def sqrt(self, a: typing.Any) -> typing.Any:
        return self._jnp.sqrt(self.asarray(a))

    def maximum(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        return self._jnp.maximum(self.asarray(x1), self.asarray(x2))

    def clip(self, a: typing.Any, a_min: typing.Any, a_max: typing.Any) -> typing.Any:
        return self._jnp.clip(self.asarray(a), a_min, a_max)

    def abs(self, a: typing.Any) -> typing.Any:
        return self._jnp.abs(self.asarray(a))

    def matmul(self, x1: typing.Any, x2: typing.Any) -> typing.Any:
        return self.asarray(x1) @ self.asarray(x2)

    def cholesky(self, a: typing.Any) -> typing.Any:
        return self._jnp.linalg.cholesky(self.asarray(a))

    def standard_normal(
        self, shape: tuple[int, ...], seed: int | None = None
    ) -> typing.Any:
        s = seed if seed is not None else 0
        key = self._jax.random.PRNGKey(s)
        return self._jax.random.normal(key, shape, dtype=self._jnp.float64)


# ---------------------------------------------------------------------------
# Global backend management & thread-local state
# ---------------------------------------------------------------------------

_GLOBAL_BACKEND: TensorBackend = NumPyBackend()
_THREAD_LOCAL = threading.local()


def _get_active_backend() -> TensorBackend:
    """Retrieve the thread-local active backend if set, otherwise global backend."""
    return getattr(_THREAD_LOCAL, "backend", _GLOBAL_BACKEND)


def get_backend(
    name: BackendType | str | None = None,
    device: str | None = None,
) -> TensorBackend:
    """Retrieve an instantiated TensorBackend instance.

    If *name* is None, returns the currently active backend.

    Args:
        name: Optional backend type or name string (e.g. ``"numpy"``, ``"torch"``).
        device: Optional device target (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        A :class:`TensorBackend` instance.
    """
    if name is None:
        return _get_active_backend()

    b_type = _parse_backend_type(name)
    dev = device or "cpu"

    if b_type is BackendType.NUMPY:
        return NumPyBackend(device=dev)
    if b_type is BackendType.TORCH:
        return PyTorchBackend(device=dev)
    if b_type is BackendType.CUPY:
        return CuPyBackend(device=dev)
    if b_type is BackendType.JAX:
        return JAXBackend(device=dev)

    msg = f"Unknown backend type {b_type}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def set_backend(name: BackendType | str, device: str = "cpu") -> None:
    """Set the global default execution backend.

    Args:
        name: Backend type (:class:`BackendType` or str, e.g. ``"torch"``).
        device: Target device string (e.g. ``"cpu"``, ``"cuda"``).
    """
    global _GLOBAL_BACKEND
    _GLOBAL_BACKEND = get_backend(name=name, device=device)


@contextlib.contextmanager
def use_backend(
    name: BackendType | str, device: str = "cpu"
) -> typing.Iterator[TensorBackend]:
    """Context manager to execute a code block under a specified backend.

    Args:
        name: Backend type or name string.
        device: Target device string.

    Yields:
        The activated :class:`TensorBackend` instance.
    """
    prev_backend = getattr(_THREAD_LOCAL, "backend", None)
    new_backend = get_backend(name=name, device=device)
    _THREAD_LOCAL.backend = new_backend
    try:
        yield new_backend
    finally:
        if prev_backend is None:
            if hasattr(_THREAD_LOCAL, "backend"):
                del _THREAD_LOCAL.backend
        else:
            _THREAD_LOCAL.backend = prev_backend

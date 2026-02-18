from typing import Sequence

from mlx.core import Device, Dtype, Stream, array, scalar
from mlx.core.distributed import state as state

def bernoulli(
    p: scalar | array = ...,
    shape: Sequence[int] | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate Bernoulli random values.

    The values are sampled from the bernoulli distribution with parameter
    ``p``. The parameter ``p`` can be a :obj:`float` or :obj:`array` and
    must be broadcastable to ``shape``.

    Args:
        p (float or array, optional): Parameter of the Bernoulli
          distribution. Default: ``0.5``.
        shape (list(int), optional): Shape of the output.
          Default: ``p.shape``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The array of random integers.
    """

def categorical(
    logits: array,
    axis: int = ...,
    shape: Sequence[int] | None = ...,
    num_samples: int | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Sample from a categorical distribution.

    The values are sampled from the categorical distribution specified by
    the unnormalized values in ``logits``. Note, at most one of ``shape``
    or ``num_samples`` can be specified. If both are ``None``, the output
    has the same shape as ``logits`` with the ``axis`` dimension removed.

    Args:
        logits (array): The *unnormalized* categorical distribution(s).
        axis (int, optional): The axis which specifies the distribution.
           Default: ``-1``.
        shape (list(int), optional): The shape of the output. This must
           be broadcast compatible with ``logits.shape`` with the ``axis``
           dimension removed. Default: ``None``
        num_samples (int, optional): The number of samples to draw from each
          of the categorical distributions in ``logits``. The output will have
          ``num_samples`` in the last dimension. Default: ``None``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The ``shape``-sized output array with type ``uint32``.
    """

def gumbel(
    shape: Sequence[int] = ...,
    dtype: Dtype | None = ...,
    key: Stream | Device | None = ...,
    stream: array | None = ...,
) -> array:
    """
    Sample from the standard Gumbel distribution.

    The values are sampled from a standard Gumbel distribution
    which CDF ``exp(-exp(-x))``.

    Args:
        shape (list(int)): The shape of the output.
        dtype (Dtype, optional): The data type of the output.
          Default: ``float32``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array:
          The :class:`array` with shape ``shape`` and distributed according
          to the Gumbel distribution.
    """

def key(seed: int) -> array:
    """
    Get a PRNG key from a seed.

    Args:
        seed (int): Seed for the PRNG.

    Returns:
        array: The PRNG key array.
    """

def laplace(
    shape: Sequence[int] = ...,
    dtype: Dtype | None = ...,
    loc: float = ...,
    scale: float = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Sample numbers from a Laplace distribution.

    Args:
        shape (list(int), optional): Shape of the output. Default: ``()``.
        dtype (Dtype, optional): Type of the output. Default: ``float32``.
        loc (float, optional): Mean of the distribution. Default: ``0.0``.
        scale (float, optional): The scale "b" of the Laplace distribution.
          Default:``1.0``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The output array of random values.
    """

def multivariate_normal(
    mean: array,
    cov: array,
    shape: Sequence[int] = ...,
    dtype: Dtype | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate jointly-normal random samples given a mean and covariance.

    The matrix ``cov`` must be positive semi-definite. The behavior is
    undefined if it is not.  The only supported ``dtype`` is ``float32``.

    Args:
        mean (array): array of shape ``(..., n)``, the mean of the
          distribution.
        cov (array): array  of shape ``(..., n, n)``, the covariance
          matrix of the distribution. The batch shape ``...`` must be
          broadcast-compatible with that of ``mean``.
        shape (list(int), optional): The output shape must be
          broadcast-compatible with ``mean.shape[:-1]`` and ``cov.shape[:-2]``.
          If empty, the result shape is determined by broadcasting the batch
          shapes of ``mean`` and ``cov``. Default: ``[]``.
        dtype (Dtype, optional): The output type. Default: ``float32``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The output array of random values.
    """

def normal(
    shape: Sequence[int] = ...,
    dtype: Dtype | None = ...,
    loc: scalar | array | None = ...,
    scale: scalar | array | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    r"""
    Generate normally distributed random numbers.

    If ``loc`` and ``scale`` are not provided the "standard" normal
    distribution is used. That means $x \sim \mathcal{N}(0, 1)$ for
    real numbers and $\text{Re}(x),\text{Im}(x) \sim \mathcal{N}(0,
    \frac{1}{2})$ for complex numbers.

    Args:
        shape (list(int), optional): Shape of the output. Default: ``()``.
        dtype (Dtype, optional): Type of the output. Default: ``float32``.
        loc (scalar or array, optional): Mean of the distribution.
          Default: ``None``.
        scale (scalar or array, optional): Standard deviation of the
          distribution. Default: ``None``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The output array of random values.
    """

def permutation(
    x: int | array,
    axis: int = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate a random permutation or permute the entries of an array.

    Args:
        x (int or array, optional): If an integer is provided a random
          permtuation of ``mx.arange(x)`` is returned. Otherwise the entries
          of ``x`` along the given axis are randomly permuted.
        axis (int, optional): The axis to permute along. Default: ``0``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array:
          The generated random permutation or randomly permuted input array.
    """

def randint(
    low: scalar | array,
    high: scalar | array,
    shape: Sequence[int] = ...,
    dtype: Dtype | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate random integers from the given interval.

    The values are sampled with equal probability from the integers in
    half-open interval ``[low, high)``. The lower and upper bound can be
    scalars or arrays and must be broadcastable to ``shape``.

    Args:
        low (scalar or array): Lower bound of the interval.
        high (scalar or array): Upper bound of the interval.
        shape (list(int), optional): Shape of the output. Default: ``()``.
        dtype (Dtype, optional): Type of the output. Default: ``int32``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The array of random integers.
    """

def seed(seed: int) -> None:
    """
    Seed the global PRNG.

    Args:
        seed (int): Seed for the global PRNG.
    """

def split(key: array, num: int = ..., stream: Stream | Device | None = ...) -> array:
    """
    Split a PRNG key into sub keys.

    Args:
        key (array): Input key to split.
        num (int, optional): Number of sub keys. Default: ``2``.

    Returns:
        array: The array of sub keys with ``num`` as its first dimension.
    """

def truncated_normal(
    lower: scalar | array,
    upper: scalar | array,
    shape: Sequence[int] | None = ...,
    dtype: Dtype | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate values from a truncated normal distribution.

    The values are sampled from the truncated normal distribution
    on the domain ``(lower, upper)``. The bounds ``lower`` and ``upper``
    can be scalars or arrays and must be broadcastable to ``shape``.

    Args:
        lower (scalar or array): Lower bound of the domain.
        upper (scalar or array): Upper bound of the domain.
        shape (list(int), optional): The shape of the output.
          Default:``()``.
        dtype (Dtype, optional): The data type of the output.
          Default: ``float32``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The output array of random values.
    """

def uniform(
    low: scalar | array = ...,
    high: scalar | array = ...,
    shape: Sequence[int] = ...,
    dtype: Dtype | None = ...,
    key: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate uniformly distributed random numbers.

    The values are sampled uniformly in the half-open interval ``[low, high)``.
    The lower and upper bound can be scalars or arrays and must be
    broadcastable to ``shape``.

    Args:
        low (scalar or array, optional): Lower bound of the distribution.
          Default: ``0``.
        high (scalar or array, optional): Upper bound of the distribution.
          Default: ``1``.
        shape (list(int), optional): Shape of the output. Default:``()``.
        dtype (Dtype, optional): Type of the output. Default: ``float32``.
        key (array, optional): A PRNG key. Default: ``None``.

    Returns:
        array: The output array random values.
    """

import enum
import pathlib
import types
from typing import (
    Annotated,
    Callable,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    overload,
)

import numpy
from mlx.nn.layers import Module
from numpy.typing import ArrayLike as _ArrayLike

from . import cuda as cuda
from . import distributed as distributed
from . import metal as metal
from . import random as random

class ArrayAt:
    """A helper object to apply updates at specific indices."""
    def __getitem__(self, indices: object | None) -> ArrayAt: ...
    def add(
        self,
        value: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def subtract(
        self,
        value: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def multiply(
        self,
        value: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def divide(
        self,
        value: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def maximum(
        self,
        value: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def minimum(
        self,
        value: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...

class ArrayIterator:
    """A helper object to iterate over the 1st dimension of an array."""
    def __next__(self) -> array: ...
    def __iter__(self) -> ArrayIterator: ...

class ArrayLike:
    """
    Any Python object which has an ``__mlx__array__`` method that
    returns an :obj:`array`.
    """
    def __init__(self, arg: object, /) -> None: ...

class Device:
    """A device to run operations on."""
    def __init__(self, type: DeviceType, index: int = ...) -> None: ...
    @property
    def type(self) -> DeviceType: ...
    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...

class DeviceType(enum.Enum):
    cpu = ...  # type: ignore
    gpu = ...  #  type: ignore
    def __eq__(self, arg: object, /) -> bool: ...

class Dtype:
    """
    An object to hold the type of a :class:`array`.

    See the :ref:`list of types <data_types>` for more details
    on available data types.
    """
    @property
    def size(self) -> int:
        """Size of the type in bytes."""

    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class DtypeCategory(enum.Enum):
    """
    Type to hold categories of :class:`dtypes <Dtype>`.

    * :attr:`~mlx.core.generic`

      * :ref:`bool_ <data_types>`
      * :attr:`~mlx.core.number`

        * :attr:`~mlx.core.integer`

          * :attr:`~mlx.core.unsignedinteger`

            * :ref:`uint8 <data_types>`
            * :ref:`uint16 <data_types>`
            * :ref:`uint32 <data_types>`
            * :ref:`uint64 <data_types>`

          * :attr:`~mlx.core.signedinteger`

            * :ref:`int8 <data_types>`
            * :ref:`int32 <data_types>`
            * :ref:`int64 <data_types>`

        * :attr:`~mlx.core.inexact`

          * :attr:`~mlx.core.floating`

            * :ref:`float16 <data_types>`
            * :ref:`bfloat16 <data_types>`
            * :ref:`float32 <data_types>`
            * :ref:`float64 <data_types>`

          * :attr:`~mlx.core.complexfloating`

            * :ref:`complex64 <data_types>`

    See also :func:`~mlx.core.issubdtype`.
    """

    complexfloating = ...
    floating = ...
    inexact = ...
    signedinteger = ...
    unsignedinteger = ...
    integer = ...
    number = ...
    generic = ...

class FunctionExporter:
    """
    A context managing class for exporting multiple traces of the same
    function to a file.

    Make an instance of this class by calling fun:`mx.exporter`.
    """
    def close(self) -> None: ...
    def __enter__(self) -> FunctionExporter: ...
    def __exit__(
        self,
        exc_type: object | None = ...,
        exc_value: object | None = ...,
        traceback: object | None = ...,
    ) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...

class Stream:
    """A stream for running operations on a given device."""
    @property
    def device(self) -> Device: ...
    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...

class StreamContext:
    """
    A context manager for setting the current device and stream.

    See :func:`stream` for usage.

    Args:
        s: The stream or device to set as the default.
    """
    def __init__(self, s: Stream | Device) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type | None = ...,
        exc_value: object | None = ...,
        traceback: object | None = ...,
    ) -> None: ...

def abs(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise absolute value.

    Args:
        a (array): Input array.

    Returns:
        array: The absolute value of ``a``.
    """

def add(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise addition.

    Add two arrays with numpy-style broadcasting semantics. Either or both input arrays
    can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The sum of ``a`` and ``b``.
    """

def addmm(
    c: array,
    a: array,
    b: array,
    /,
    alpha: float = ...,
    beta: float = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Matrix multiplication with addition and optional scaling.

    Perform the (possibly batched) matrix multiplication of two arrays and add to the result
    with optional scaling factors.

    Args:
        c (array): Input array or scalar.
        a (array): Input array or scalar.
        b (array): Input array or scalar.
        alpha (float, optional): Scaling factor for the
            matrix product of ``a`` and ``b`` (default: ``1``)
        beta (float, optional): Scaling factor for ``c`` (default: ``1``)

    Returns:
        array: ``alpha * (a @ b)  + beta * c``
    """

def all(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    An `and` reduction over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

def allclose(
    a: array,
    b: array,
    /,
    rtol: float = ...,
    atol: float = ...,
    *,
    equal_nan: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Approximate comparison of two arrays.

    Infinite values are considered equal if they have the same sign, NaN values are not equal unless ``equal_nan`` is ``True``.

    The arrays are considered equal if:

    .. code-block::

     all(abs(a - b) <= (atol + rtol * abs(b)))

    Note unlike :func:`array_equal`, this function supports numpy-style
    broadcasting.

    Args:
        a (array): Input array.
        b (array): Input array.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, NaNs are considered equal.
          Defaults to ``False``.

    Returns:
        array: The boolean output scalar indicating if the arrays are close.
    """

def any(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    An `or` reduction over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

@overload
def arange(
    start: int | float,
    stop: int | float,
    step: int | float | None,
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generates ranges of numbers.

    Generate numbers in the half-open interval ``[start, stop)`` in
    increments of ``step``.

    Args:
        start (float or int, optional): Starting value which defaults to ``0``.
        stop (float or int): Stopping value.
        step (float or int, optional): Increment which defaults to ``1``.
        dtype (Dtype, optional): Specifies the data type of the output. If unspecified will default to ``float32`` if any of ``start``, ``stop``, or ``step`` are ``float``. Otherwise will default to ``int32``.

    Returns:
        array: The range of values.

    Note:
      Following the Numpy convention the actual increment used to
      generate numbers is ``dtype(start + step) - dtype(start)``.
      This can lead to unexpected results for example if `start + step`
      is a fractional value and the `dtype` is integral.
    """

@overload
def arange(
    stop: int | float,
    step: int | float | None = ...,
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array: ...
def arccos(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse cosine.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse cosine of ``a``.
    """

def arccosh(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse hyperbolic cosine.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse hyperbolic cosine of ``a``.
    """

def arcsin(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse sine.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse sine of ``a``.
    """

def arcsinh(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse hyperbolic sine.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse hyperbolic sine of ``a``.
    """

def arctan(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse tangent.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse tangent of ``a``.
    """

def arctan2(a: array, b: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse tangent of the ratio of two arrays.

    Args:
        a (array): Input array.
        b (array): Input array.

    Returns:
        array: The inverse tangent of the ratio of ``a`` and ``b``.
    """

def arctanh(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse hyperbolic tangent.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse hyperbolic tangent of ``a``.
    """

def argmax(
    a: array,
    /,
    axis: int | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Indices of the maximum values along the axis.

    Args:
        a (array): Input array.
        axis (int, optional): Optional axis to reduce over. If unspecified
          this defaults to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The ``uint32`` array with the indices of the maximum values.
    """

def argmin(
    a: array,
    /,
    axis: int | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Indices of the minimum values along the axis.

    Args:
        a (array): Input array.
        axis (int, optional): Optional axis to reduce over. If unspecified
          this defaults to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The ``uint32`` array with the indices of the minimum values.
    """

def argpartition(
    a: array,
    /,
    kth: int,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Returns the indices that partition the array.

    The ordering of the elements within a partition in given by the indices
    is undefined.

    Args:
        a (array): Input array.
        kth (int): Element index at the ``kth`` position in the output will
          give the sorted position. All indices before the ``kth`` position
          will be of elements less or equal to the element at the ``kth``
          index and all indices after will be of elements greater or equal
          to the element at the ``kth`` index.
        axis (int or None, optional): Optional axis to partition over.
          If ``None``, this partitions over the flattened array.
          If unspecified, it defaults to ``-1``.

    Returns:
        array: The ``uint32`` array containing indices that partition the input.
    """

def argsort(
    a: array,
    /,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Returns the indices that sort the array.

    Args:
        a (array): Input array.
        axis (int or None, optional): Optional axis to sort over.
          If ``None``, this sorts over the flattened array.
          If unspecified, it defaults to -1 (sorting over the last axis).

    Returns:
        array: The ``uint32`` array containing indices that sort the input.
    """

class array:
    """An N-dimensional array object."""
    def __init__(
        self: array,
        val: scalar | list | tuple | numpy.ndarray | array,
        dtype: Dtype | None = ...,
    ) -> None: ...
    def __buffer__(self, flags, /):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """

    def __release_buffer__(self, buffer, /):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """

    @property
    def size(self) -> int:
        """Number of elements in the array."""

    @property
    def ndim(self) -> int:
        """The array's dimension."""

    @property
    def itemsize(self) -> int:
        """The size of the array's datatype in bytes."""

    @property
    def nbytes(self) -> int:
        """The number of bytes in the array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of the array as a Python tuple.

        Returns:
          tuple(int): A tuple containing the sizes of each dimension.
        """

    @property
    def dtype(self) -> Dtype:
        """The array's :class:`Dtype`."""

    @property
    def real(self) -> array:
        """The real part of a complex array."""

    @property
    def imag(self) -> array:
        """The imaginary part of a complex array."""

    def item(self) -> scalar:
        """
        Access the value of a scalar array.

        Returns:
            Standard Python scalar.
        """

    def tolist(self) -> list_or_scalar:
        """
        Convert the array to a Python :class:`list`.

        Returns:
            list: The Python list.

            If the array is a scalar then a standard Python scalar is returned.

            If the array has more than one dimension then the result is a nested
            list of lists.

            The value type of the list corresponding to the last dimension is either
            ``bool``, ``int`` or ``float`` depending on the ``dtype`` of the array.
        """

    def astype(self, dtype: Dtype, stream: Stream | Device | None = ...) -> array:
        """
        Cast the array to a specified type.

        Args:
            dtype (Dtype): Type to which the array is cast.
            stream (Stream): Stream (or device) for the operation.

        Returns:
            array: The array with type ``dtype``.
        """

    def __array_namespace__(self, api_version: str | None = ...) -> types.ModuleType:
        """
        Returns an object that has all the array API functions on it.

        See the `Python array API <https://data-apis.org/array-api/latest/index.html>`_
        for more information.

        Args:
            api_version (str, optional): String representing the version
              of the array API spec to return. Default: ``None``.

        Returns:
            out (Any): An object representing the array API namespace.
        """

    def __getitem__(self, arg: object | None) -> array: ...
    def __setitem__(
        self,
        arg0: object | None,
        arg1: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> None: ...
    @property
    def at(self) -> ArrayAt:
        """
        Used to apply updates at the given indices.

        .. note::

           Regular in-place updates map to assignment. For instance ``x[idx] += y``
           maps to ``x[idx] = x[idx] + y``. As a result, assigning to the
           same index ignores all but one update. Using ``x.at[idx].add(y)``
           will correctly apply all updates to all indices.

        .. list-table::
           :header-rows: 1

           * - array.at syntax
             - In-place syntax
           * - ``x = x.at[idx].add(y)``
             - ``x[idx] += y``
           * - ``x = x.at[idx].subtract(y)``
             - ``x[idx] -= y``
           * - ``x = x.at[idx].multiply(y)``
             - ``x[idx] *= y``
           * - ``x = x.at[idx].divide(y)``
             - ``x[idx] /= y``
           * - ``x = x.at[idx].maximum(y)``
             - ``x[idx] = mx.maximum(x[idx], y)``
           * - ``x = x.at[idx].minimum(y)``
             - ``x[idx] = mx.minimum(x[idx], y)``

        Example:
            >>> a = mx.array([0, 0])
            >>> idx = mx.array([0, 1, 0, 1])
            >>> a[idx] += 1
            >>> a
            array([1, 1], dtype=int32)
            >>>
            >>> a = mx.array([0, 0])
            >>> a.at[idx].add(1)
            array([2, 2], dtype=int32)
        """

    def __len__(self) -> int: ...
    def __iter__(self) -> ArrayIterator: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg: tuple, /) -> None: ...
    def __dlpack__(self) -> _ArrayLike: ...
    def __dlpack_device__(self) -> tuple: ...
    def __copy__(self) -> array: ...
    def __deepcopy__(self, memo: dict) -> array: ...
    def __add__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __iadd__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __radd__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __sub__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __isub__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rsub__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __mul__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __imul__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rmul__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __truediv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __itruediv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rtruediv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __div__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rdiv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __floordiv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ifloordiv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rfloordiv__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __mod__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __imod__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rmod__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __eq__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array | bool: ...
    def __lt__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __le__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __gt__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ge__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ne__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array | bool: ...
    def __neg__(self) -> array: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __matmul__(self, other: array) -> array: ...
    def __imatmul__(self, other: array) -> array: ...
    def __pow__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rpow__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ipow__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __invert__(self) -> array: ...
    def __and__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __iand__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __or__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ior__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __lshift__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ilshift__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __rshift__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __irshift__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __xor__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __ixor__(
        self,
        other: bool
        | int
        | float
        | array
        | Annotated[_ArrayLike, dict(order="C", device="cpu", writable=False)]
        | complex
        | ArrayLike,
    ) -> array: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def flatten(
        self,
        start_axis: int = ...,
        end_axis: int = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`flatten`."""

    def reshape(self, *shape, stream: Stream | Device | None = ...) -> array:
        """
        Equivalent to :func:`reshape` but the shape can be passed either as a
        :obj:`tuple` or as separate arguments.

        See :func:`reshape` for full documentation.
        """

    def squeeze(
        self,
        axis: int | Sequence[int] | None = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`squeeze`."""

    def abs(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`abs`."""

    def __abs__(self) -> array:
        """See :func:`abs`."""

    def square(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`square`."""

    def sqrt(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`sqrt`."""

    def rsqrt(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`rsqrt`."""

    def reciprocal(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`reciprocal`."""

    def exp(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`exp`."""

    def log(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`log`."""

    def log2(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`log2`."""

    def log10(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`log10`."""

    def sin(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`sin`."""

    def cos(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`cos`."""

    def log1p(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`log1p`."""

    def all(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`all`."""

    def any(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`any`."""

    def moveaxis(
        self, source: int, destination: int, *, stream: Stream | Device | None = ...
    ) -> array:
        """See :func:`moveaxis`."""

    def swapaxes(
        self, axis1: int, axis2: int, *, stream: Stream | Device | None = ...
    ) -> array:
        """See :func:`swapaxes`."""

    def transpose(self, *axes, stream: Stream | Device | None = ...) -> array:
        """
        Equivalent to :func:`transpose` but the axes can be passed either as
        a tuple or as separate arguments.

        See :func:`transpose` for full documentation.
        """

    @property
    def T(self) -> array:
        """Equivalent to calling ``self.transpose()`` with no arguments."""

    def sum(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`sum`."""

    def prod(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`prod`."""

    def min(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`min`."""

    def max(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`max`."""

    def logcumsumexp(
        self,
        axis: int | None = ...,
        *,
        reverse: bool = ...,
        inclusive: bool = ...,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`logcumsumexp`."""

    def logsumexp(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`logsumexp`."""

    def mean(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`mean`."""

    def std(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        ddof: int = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`std`."""

    def var(
        self,
        axis: int | Sequence[int] | None = ...,
        keepdims: bool = ...,
        ddof: int = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`var`."""

    def split(
        self,
        indices_or_sections: int | tuple[int, ...],
        axis: int = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> list[array]:
        """See :func:`split`."""

    def argmin(
        self,
        axis: int | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`argmin`."""

    def argmax(
        self,
        axis: int | None = ...,
        keepdims: bool = ...,
        *,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`argmax`."""

    def cumsum(
        self,
        axis: int | None = ...,
        *,
        reverse: bool = ...,
        inclusive: bool = ...,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`cumsum`."""

    def cumprod(
        self,
        axis: int | None = ...,
        *,
        reverse: bool = ...,
        inclusive: bool = ...,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`cumprod`."""

    def cummax(
        self,
        axis: int | None = ...,
        *,
        reverse: bool = ...,
        inclusive: bool = ...,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`cummax`."""

    def cummin(
        self,
        axis: int | None = ...,
        *,
        reverse: bool = ...,
        inclusive: bool = ...,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`cummin`."""

    def round(
        self, decimals: int = ..., *, stream: Stream | Device | None = ...
    ) -> array:
        """See :func:`round`."""

    def diagonal(
        self,
        offset: int = ...,
        axis1: int = ...,
        axis2: int = ...,
        stream: Stream | Device | None = ...,
    ) -> array:
        """See :func:`diagonal`."""

    def diag(self, k: int = ..., *, stream: Stream | Device | None = ...) -> array:
        """Extract a diagonal or construct a diagonal matrix."""

    def conj(self, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`conj`."""

    def view(self, dtype: Dtype, *, stream: Stream | Device | None = ...) -> array:
        """See :func:`view`."""

def array_equal(
    a: scalar | array,
    b: scalar | array,
    equal_nan: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Array equality check.

    Compare two arrays for equality. Returns ``True`` if and only if the arrays
    have the same shape and their values are equal. The arrays need not have
    the same type to be considered equal.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.
        equal_nan (bool): If ``True``, NaNs are considered equal.
          Defaults to ``False``.

    Returns:
        array: A scalar boolean array.
    """

def as_strided(
    a: array,
    /,
    shape: Sequence[int] | None = ...,
    strides: Sequence[int] | None = ...,
    offset: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Create a view into the array with the given shape and strides.

    The resulting array will always be as if the provided array was row
    contiguous regardless of the provided arrays storage order and current
    strides.

    .. note::
       Note that this function should be used with caution as it changes
       the shape and strides of the array directly. This can lead to the
       resulting array pointing to invalid memory locations which can
       result into crashes.

    Args:
      a (array): Input array
      shape (list(int), optional): The shape of the resulting array. If
        None it defaults to ``a.shape()``.
      strides (list(int), optional): The strides of the resulting array. If
        None it defaults to the reverse exclusive cumulative product of
        ``a.shape()``.
      offset (int): Skip that many elements from the beginning of the input
        array.

    Returns:
      array: The output array which is the strided view of the input.
    """

def async_eval(*args: MX_ARRAY_TREE) -> None:
    """
    Asynchronously evaluate an :class:`array` or tree of :class:`array`.

    .. note::

      This is an experimental API and may change in future versions.

    Args:
        *args (arrays or trees of arrays): Each argument can be a single array
          or a tree of arrays. If a tree is given the nodes can be a Python
          :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
          arrays are ignored.

    Example:
        >>> x = mx.array(1.0)
        >>> y = mx.exp(x)
        >>> mx.async_eval(y)
        >>> print(y)
        >>>
        >>> y = mx.exp(x)
        >>> mx.async_eval(y)
        >>> z = y + 3
        >>> mx.async_eval(z)
        >>> print(z)
    """

def atleast_1d(
    *arys: array, stream: Stream | Device | None = ...
) -> array | list[array]:
    """
    Convert all arrays to have at least one dimension.

    Args:
        *arys: Input arrays.
        stream (Stream | Device | None, optional): The stream to execute the operation on.

    Returns:
        array or list(array): An array or list of arrays with at least one dimension.
    """

def atleast_2d(
    *arys: array, stream: Stream | Device | None = ...
) -> array | list[array]:
    """
    Convert all arrays to have at least two dimensions.

    Args:
        *arys: Input arrays.
        stream (Stream | Device | None, optional): The stream to execute the operation on.

    Returns:
        array or list(array): An array or list of arrays with at least two dimensions.
    """

def atleast_3d(
    *arys: array, stream: Stream | Device | None = ...
) -> array | list[array]:
    """
    Convert all arrays to have at least three dimensions.

    Args:
        *arys: Input arrays.
        stream (Stream | Device | None, optional): The stream to execute the operation on.

    Returns:
        array or list(array): An array or list of arrays with at least three dimensions.
    """

bfloat16: Dtype = ...

def bitwise_and(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise bitwise and.

    Take the bitwise and of two arrays with numpy-style broadcasting
    semantics. Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The bitwise and ``a & b``.
    """

def bitwise_invert(a: scalar | array, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise bitwise inverse.

    Take the bitwise complement of the input.

    Args:
        a (array): Input array or scalar.

    Returns:
        array: The bitwise inverse ``~a``.
    """

def bitwise_or(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise bitwise or.

    Take the bitwise or of two arrays with numpy-style broadcasting
    semantics. Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The bitwise or``a | b``.
    """

def bitwise_xor(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise bitwise xor.

    Take the bitwise exclusive or of two arrays with numpy-style
    broadcasting semantics. Either or both input arrays can also be
    scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The bitwise xor ``a ^ b``.
    """

def block_masked_mm(
    a: array,
    b: array,
    /,
    block_size: int = ...,
    mask_out: array | None = ...,
    mask_lhs: array | None = ...,
    mask_rhs: array | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    r"""
    Matrix multiplication with block masking.

    Perform the (possibly batched) matrix multiplication of two arrays and with blocks
    of size ``block_size x block_size`` optionally masked out.

    Assuming ``a`` with shape (..., `M`, `K`) and b with shape (..., `K`, `N`)

    * ``lhs_mask`` must have shape (..., :math:`\lceil` `M` / ``block_size`` :math:`\rceil`, :math:`\lceil` `K` / ``block_size`` :math:`\rceil`)

    * ``rhs_mask`` must have shape (..., :math:`\lceil` `K` / ``block_size`` :math:`\rceil`, :math:`\lceil` `N` / ``block_size`` :math:`\rceil`)

    * ``out_mask`` must have shape (..., :math:`\lceil` `M` / ``block_size`` :math:`\rceil`, :math:`\lceil` `N` / ``block_size`` :math:`\rceil`)

    Note: Only ``block_size=64`` and ``block_size=32`` are currently supported

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.
        block_size (int): Size of blocks to be masked. Must be ``32`` or ``64``. Default: ``64``.
        mask_out (array, optional): Mask for output. Default: ``None``.
        mask_lhs (array, optional): Mask for ``a``. Default: ``None``.
        mask_rhs (array, optional): Mask for ``b``. Default: ``None``.

    Returns:
        array: The output array.
    """

def broadcast_arrays(
    *arrays: array, stream: Stream | Device | None = ...
) -> tuple[array, ...]:
    """
    Broadcast arrays against one another.

    The broadcasting semantics are the same as Numpy.

    Args:
        *arrays (array): The input arrays.

    Returns:
        tuple(array): The output arrays with the broadcasted shape.
    """

def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int]:
    """
    Broadcast shapes.

    Returns the shape that results from broadcasting the supplied array shapes
    against each other.

    Args:
        *shapes (Sequence[int]): The shapes to broadcast.

    Returns:
        tuple: The broadcasted shape.

    Raises:
        ValueError: If the shapes cannot be broadcast.

    Example:
        >>> mx.broadcast_shapes((1,), (3, 1))
        (3, 1)
        >>> mx.broadcast_shapes((6, 7), (5, 6, 1), (7,))
        (5, 6, 7)
        >>> mx.broadcast_shapes((5, 1, 4), (1, 3, 1))
        (5, 3, 4)
    """

def broadcast_to(
    a: scalar | array,
    /,
    shape: Sequence[int],
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Broadcast an array to the given shape.

    The broadcasting semantics are the same as Numpy.

    Args:
        a (array): Input array.
        shape (list(int)): The shape to broadcast to.

    Returns:
        array: The output array with the new shape.
    """

def ceil(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise ceil.

    Args:
        a (array): Input array.

    Returns:
        array: The ceil of ``a``.
    """

def checkpoint(fun: Callable) -> Callable: ...
def clear_cache() -> None:
    """
    Clear the memory cache.

    After calling this, :func:`get_cache_memory` should return ``0``.
    """

def clip(
    a: array,
    /,
    a_min: scalar | array | None,
    a_max: scalar | array | None,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Clip the values of the array between the given minimum and maximum.

    If either ``a_min`` or ``a_max`` are ``None``, then corresponding edge
    is ignored. At least one of ``a_min`` and ``a_max`` cannot be ``None``.
    The input ``a`` and the limits must broadcast with one another.

    Args:
        a (array): Input array.
        a_min (scalar or array or None): Minimum value to clip to.
        a_max (scalar or array or None): Maximum value to clip to.

    Returns:
        array: The clipped array.
    """

def compile(
    fun: Callable,
    inputs: object | None = ...,
    outputs: object | None = ...,
    shapeless: bool = ...,
) -> Callable:
    """
    Returns a compiled function which produces the same output as ``fun``.

    Args:
        fun (Callable): A function which takes a variable number of
          :class:`array` or trees of :class:`array` and returns
          a variable number of :class:`array` or trees of :class:`array`.
        inputs (list or dict, optional): These inputs will be captured during
          the function compilation along with the inputs to ``fun``. The ``inputs``
          can be a :obj:`list` or a :obj:`dict` containing arbitrarily nested
          lists, dictionaries, or arrays. Leaf nodes that are not
          :obj:`array` are ignored. Default: ``None``
        outputs (list or dict, optional): These outputs will be captured and
          updated in a compiled function. The ``outputs`` can be a
          :obj:`list` or a :obj:`dict` containing arbitrarily nested lists,
          dictionaries, or arrays. Leaf nodes that are not :obj:`array` are ignored.
          Default: ``None``
        shapeless (bool, optional): A function compiled with the ``shapeless``
          option enabled will not be recompiled when the input shape changes. Not all
          functions can be compiled with ``shapeless`` enabled. Attempting to compile
          such functions with shapeless enabled will throw. Note, changing the number
          of dimensions or type of any input will result in a recompilation even with
          ``shapeless`` set to ``True``. Default: ``False``

    Returns:
        Callable: A compiled function which has the same input arguments
        as ``fun`` and returns the the same output(s).
    """

complex64: Dtype = ...
complexfloating: DtypeCategory = ...

def concat(
    arrays: list[array],
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """See :func:`concatenate`."""

def concatenate(
    arrays: list[array],
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Concatenate the arrays along the given axis.

    Args:
        arrays (list(array)): Input :obj:`list` or :obj:`tuple` of arrays.
        axis (int, optional): Optional axis to concatenate along. If
          unspecified defaults to ``0``.

    Returns:
        array: The concatenated array.
    """

def conj(a: array, *, stream: Stream | Device | None = ...) -> array:
    """
    Return the elementwise complex conjugate of the input.
    Alias for `mx.conjugate`.

    Args:
      a (array): Input array

    Returns:
      array: The output array.
    """

def conjugate(a: array, *, stream: Stream | Device | None = ...) -> array:
    """
    Return the elementwise complex conjugate of the input.
    Alias for `mx.conj`.

    Args:
      a (array): Input array

    Returns:
      array: The output array.
    """

def contiguous(
    a: array,
    /,
    allow_col_major: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Force an array to be row contiguous. Copy if necessary.

    Args:
      a (array): The input to make contiguous
      allow_col_major (bool): Consider column major as contiguous and don't copy

    Returns:
      array: The row or col contiguous output.
    """

def conv1d(
    input: array,
    weight: array,
    /,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    groups: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    1D convolution over an input with several channels

    Args:
        input (array): Input array of shape ``(N, L, C_in)``.
        weight (array): Weight array of shape ``(C_out, K, C_in)``.
        stride (int, optional): Kernel stride. Default: ``1``.
        padding (int, optional): Input padding. Default: ``0``.
        dilation (int, optional): Kernel dilation. Default: ``1``.
        groups (int, optional): Input feature groups. Default: ``1``.

    Returns:
        array: The convolved array.
    """

def conv2d(
    input: array,
    weight: array,
    /,
    stride: int | tuple[int, int] = ...,
    padding: int | tuple[int, int] = ...,
    dilation: int | tuple[int, int] = ...,
    groups: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    2D convolution over an input with several channels

    Args:
        input (array): Input array of shape ``(N, H, W, C_in)``.
        weight (array): Weight array of shape ``(C_out, KH, KW, C_in)``.
        stride (int or tuple(int), optional): :obj:`tuple` of size 2 with
            kernel strides. All spatial dimensions get the same stride if
            only one number is specified. Default: ``1``.
        padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
            symmetric input padding. All spatial dimensions get the same
            padding if only one number is specified. Default: ``0``.
        dilation (int or tuple(int), optional): :obj:`tuple` of size 2 with
            kernel dilation. All spatial dimensions get the same dilation
            if only one number is specified. Default: ``1``
        groups (int, optional): input feature groups. Default: ``1``.

    Returns:
        array: The convolved array.
    """

def conv3d(
    input: array,
    weight: array,
    /,
    stride: int | tuple[int, int, int] = ...,
    padding: int | tuple[int, int, int] = ...,
    dilation: int | tuple[int, int, int] = ...,
    groups: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    3D convolution over an input with several channels

    Note: Only the default ``groups=1`` is currently supported.

    Args:
        input (array): Input array of shape ``(N, D, H, W, C_in)``.
        weight (array): Weight array of shape ``(C_out, KD, KH, KW, C_in)``.
        stride (int or tuple(int), optional): :obj:`tuple` of size 3 with
            kernel strides. All spatial dimensions get the same stride if
            only one number is specified. Default: ``1``.
        padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
            symmetric input padding. All spatial dimensions get the same
            padding if only one number is specified. Default: ``0``.
        dilation (int or tuple(int), optional): :obj:`tuple` of size 3 with
            kernel dilation. All spatial dimensions get the same dilation
            if only one number is specified. Default: ``1``
        groups (int, optional): input feature groups. Default: ``1``.

    Returns:
        array: The convolved array.
    """

def conv_general(
    input: array,
    weight: array,
    /,
    stride: int | Sequence[int] = ...,
    padding: int | Sequence[int] | tuple[Sequence[int] | Sequence[int]] = ...,
    kernel_dilation: int | Sequence[int] = ...,
    input_dilation: int | Sequence[int] = ...,
    groups: int = ...,
    flip: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    General convolution over an input with several channels

    Args:
        input (array): Input array of shape ``(N, ..., C_in)``.
        weight (array): Weight array of shape ``(C_out, ..., C_in)``.
        stride (int or list(int), optional): :obj:`list` with kernel strides.
            All spatial dimensions get the same stride if
            only one number is specified. Default: ``1``.
        padding (int, list(int), or tuple(list(int), list(int)), optional):
            :obj:`list` with input padding. All spatial dimensions get the same
            padding if only one number is specified. Default: ``0``.
        kernel_dilation (int or list(int), optional): :obj:`list` with
            kernel dilation. All spatial dimensions get the same dilation
            if only one number is specified. Default: ``1``
        input_dilation (int or list(int), optional): :obj:`list` with
            input dilation. All spatial dimensions get the same dilation
            if only one number is specified. Default: ``1``
        groups (int, optional): Input feature groups. Default: ``1``.
        flip (bool, optional): Flip the order in which the spatial dimensions of
            the weights are processed. Performs the cross-correlation operator when
            ``flip`` is ``False`` and the convolution operator otherwise.
            Default: ``False``.

    Returns:
        array: The convolved array.
    """

def conv_transpose1d(
    input: array,
    weight: array,
    /,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    output_padding: int = ...,
    groups: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    1D transposed convolution over an input with several channels

    Args:
        input (array): Input array of shape ``(N, L, C_in)``.
        weight (array): Weight array of shape ``(C_out, K, C_in)``.
        stride (int, optional): Kernel stride. Default: ``1``.
        padding (int, optional): Input padding. Default: ``0``.
        dilation (int, optional): Kernel dilation. Default: ``1``.
        output_padding (int, optional): Output padding. Default: ``0``.
        groups (int, optional): Input feature groups. Default: ``1``.

    Returns:
        array: The convolved array.
    """

def conv_transpose2d(
    input: array,
    weight: array,
    /,
    stride: int | tuple[int, int] = ...,
    padding: int | tuple[int, int] = ...,
    dilation: int | tuple[int, int] = ...,
    output_padding: int | tuple[int, int] = ...,
    groups: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    2D transposed convolution over an input with several channels

    Note: Only the default ``groups=1`` is currently supported.

    Args:
        input (array): Input array of shape ``(N, H, W, C_in)``.
        weight (array): Weight array of shape ``(C_out, KH, KW, C_in)``.
        stride (int or tuple(int), optional): :obj:`tuple` of size 2 with
            kernel strides. All spatial dimensions get the same stride if
            only one number is specified. Default: ``1``.
        padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
            symmetric input padding. All spatial dimensions get the same
            padding if only one number is specified. Default: ``0``.
        dilation (int or tuple(int), optional): :obj:`tuple` of size 2 with
            kernel dilation. All spatial dimensions get the same dilation
            if only one number is specified. Default: ``1``
        output_padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
            output padding. All spatial dimensions get the same output
            padding if only one number is specified. Default: ``0``.
        groups (int, optional): input feature groups. Default: ``1``.

    Returns:
        array: The convolved array.
    """

def conv_transpose3d(
    input: array,
    weight: array,
    /,
    stride: int | tuple[int, int, int] = ...,
    padding: int | tuple[int, int, int] = ...,
    dilation: int | tuple[int, int, int] = ...,
    output_padding: int | tuple[int, int, int] = ...,
    groups: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    3D transposed convolution over an input with several channels

    Note: Only the default ``groups=1`` is currently supported.

    Args:
        input (array): Input array of shape ``(N, D, H, W, C_in)``.
        weight (array): Weight array of shape ``(C_out, KD, KH, KW, C_in)``.
        stride (int or tuple(int), optional): :obj:`tuple` of size 3 with
            kernel strides. All spatial dimensions get the same stride if
            only one number is specified. Default: ``1``.
        padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
            symmetric input padding. All spatial dimensions get the same
            padding if only one number is specified. Default: ``0``.
        dilation (int or tuple(int), optional): :obj:`tuple` of size 3 with
            kernel dilation. All spatial dimensions get the same dilation
            if only one number is specified. Default: ``1``
        output_padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
            output padding. All spatial dimensions get the same output
            padding if only one number is specified. Default: ``0``.
        groups (int, optional): input feature groups. Default: ``1``.

    Returns:
        array: The convolved array.
    """

def convolve(
    a: array, v: array, /, mode: str = ..., *, stream: Stream | Device | None = ...
) -> array:
    """
    The discrete convolution of 1D arrays.

    If ``v`` is longer than ``a``, then they are swapped.
    The conv filter is flipped following signal processing convention.

    Args:
        a (array): 1D Input array.
        v (array): 1D Input array.
        mode (str, optional): {'full', 'valid', 'same'}

    Returns:
        array: The convolved array.
    """

def cos(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise cosine.

    Args:
        a (array): Input array.

    Returns:
        array: The cosine of ``a``.
    """

def cosh(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise hyperbolic cosine.

    Args:
        a (array): Input array.

    Returns:
        array: The hyperbolic cosine of ``a``.
    """

cpu: DeviceType = ...

def cummax(
    a: array,
    /,
    axis: int | None = ...,
    *,
    reverse: bool = ...,
    inclusive: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return the cumulative maximum of the elements along the given axis.

    Args:
      a (array): Input array
      axis (int, optional): Optional axis to compute the cumulative maximum
        over. If unspecified the cumulative maximum of the flattened array is
        returned.
      reverse (bool): Perform the cumulative maximum in reverse.
      inclusive (bool): The i-th element of the output includes the i-th
        element of the input.

    Returns:
      array: The output array.
    """

def cummin(
    a: array,
    /,
    axis: int | None = ...,
    *,
    reverse: bool = ...,
    inclusive: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return the cumulative minimum of the elements along the given axis.

    Args:
      a (array): Input array
      axis (int, optional): Optional axis to compute the cumulative minimum
        over. If unspecified the cumulative minimum of the flattened array is
        returned.
      reverse (bool): Perform the cumulative minimum in reverse.
      inclusive (bool): The i-th element of the output includes the i-th
        element of the input.

    Returns:
      array: The output array.
    """

def cumprod(
    a: array,
    /,
    axis: int | None = ...,
    *,
    reverse: bool = ...,
    inclusive: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return the cumulative product of the elements along the given axis.

    Args:
      a (array): Input array
      axis (int, optional): Optional axis to compute the cumulative product
        over. If unspecified the cumulative product of the flattened array is
        returned.
      reverse (bool): Perform the cumulative product in reverse.
      inclusive (bool): The i-th element of the output includes the i-th
        element of the input.

    Returns:
      array: The output array.
    """

def cumsum(
    a: array,
    /,
    axis: int | None = ...,
    *,
    reverse: bool = ...,
    inclusive: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return the cumulative sum of the elements along the given axis.

    Args:
      a (array): Input array
      axis (int, optional): Optional axis to compute the cumulative sum
        over. If unspecified the cumulative sum of the flattened array is
        returned.
      reverse (bool): Perform the cumulative sum in reverse.
      inclusive (bool): The i-th element of the output includes the i-th
        element of the input.

    Returns:
      array: The output array.
    """

class custom_function:
    """
    Set up a function for custom gradient and vmap definitions.

    This class is meant to be used as a function decorator. Instances are
    callables that behave identically to the wrapped function. However, when
    a function transformation is used (e.g. computing gradients using
    :func:`value_and_grad`) then the functions defined via
    :meth:`custom_function.vjp`, :meth:`custom_function.jvp` and
    :meth:`custom_function.vmap` are used instead of the default transformation.

    Note, all custom transformations are optional. Undefined transformations
    fall back to the default behaviour.

    Example:

      .. code-block:: python

          import mlx.core as mx

          @mx.custom_function
          def f(x, y):
              return mx.sin(x) * y

          @f.vjp
          def f_vjp(primals, cotangent, output):
              x, y = primals
              return cotan * mx.cos(x) * y, cotan * mx.sin(x)

          @f.jvp
          def f_jvp(primals, tangents):
            x, y = primals
            dx, dy = tangents
            return dx * mx.cos(x) * y + dy * mx.sin(x)

          @f.vmap
          def f_vmap(inputs, axes):
            x, y = inputs
            ax, ay = axes
            if ay != ax and ax is not None:
                y = y.swapaxes(ay, ax)
            return mx.sin(x) * y, (ax or ay)

    All ``custom_function`` instances behave as pure functions. Namely, any
    variables captured will be treated as constants and no gradients will be
    computed with respect to the captured arrays. For instance:

      .. code-block:: python

        import mlx.core as mx

        def g(x, y):
          @mx.custom_function
          def f(x):
            return x * y

          @f.vjp
          def f_vjp(x, dx, fx):
            # Note that we have only x, dx and fx and nothing with respect to y
            raise ValueError("Abort!")

          return f(x)

        x = mx.array(2.0)
        y = mx.array(3.0)
        print(g(x, y))                     # prints 6.0
        print(mx.grad(g)(x, y))            # Raises exception
        print(mx.grad(g, argnums=1)(x, y)) # prints 0.0
    """
    def __init__(self, f: Callable) -> None: ...
    def __call__(self, *args, **kwargs) -> object: ...
    def vjp(self, f: Callable):
        """
        Define a custom vjp for the wrapped function.

        The vjp function takes three arguments:

        - *primals*: A pytree that contains all the positional arguments to
          the function. It could be a single array, a tuple of arrays or a
          full blown tuple of dicts of arrays etc.
        - *cotangents*: A pytree that matches the structure of the output
          but contains the cotangents (usually the gradients of the loss
          function with respect to the outputs).
        - *outputs*: The outputs of the function to be used to avoid
          recomputing them for the gradient computation.

        The vjp function should return the same pytree structure as the
        primals but containing the corresponding computed cotangents.
        """

    def jvp(self, f: Callable):
        """
        Define a custom jvp for the wrapped function.

        The jvp function takes two arguments:

        - *primals*: A pytree that contains all the positional arguments to
          the function. It could be a single array, a tuple of arrays or a
          full blown tuple of dicts of arrays etc.
        - *tangents*: A pytree that matches the structure of the inputs but
          instead contains the gradients wrt to each input. Tangents could
          be ``None`` if some inputs don't have an associated gradient.

        The jvp function should return the same pytree structure as the
        outputs of the function but containing the tangents.
        """

    def vmap(self, f: Callable):
        """
        Define a custom vectorization transformation for the wrapped function.

        The vmap function takes two arguments:

        - *inputs*: A pytree that contains all the positional arguments to
          the function. It could be a single array, a tuple of arrays or a
          full blown tuple of dicts of arrays etc.
        - *axes*: A pytree that matches the structure of the inputs but
          instead contains the vectorization axis for each input or
          ``None`` if an input is not vectorized.

        The vmap function should return the outputs of the original
        function but vectorized over the provided axes. It should also
        return a pytree with the vectorization axes of each output. If some
        outputs are no longer vectorized, then their vectorization axis
        should be ``None``.
        """

def default_device() -> Device:
    """Get the default device."""

def default_stream(device: Device) -> Stream:
    """Get the device's default stream."""

def degrees(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Convert angles from radians to degrees.

    Args:
        a (array): Input array.

    Returns:
        array: The angles in degrees.
    """

def depends(inputs: array | Sequence[array], dependencies: array | Sequence[array]):
    """
    Insert dependencies between arrays in the graph. The outputs are
    identical to ``inputs`` but with dependencies on ``dependencies``.

    Args:
        inputs (array or Sequence[array]): The input array or arrays.
        dependencies (array or Sequence[array]): The array or arrays
          to insert dependencies on.

    Returns:
        array or Sequence[array]: The outputs which depend on dependencies.
    """

def dequantize(
    w: array,
    /,
    scales: array,
    biases: array | None = ...,
    group_size: int = ...,
    bits: int = ...,
    mode: str = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    r"""
    Dequantize the matrix ``w`` using quantization parameters.

    Args:
      w (array): Matrix to be dequantized
      scales (array): The scales to use per ``group_size`` elements of ``w``.
      biases (array, optional): The biases to use per ``group_size``
         elements of ``w``. Default: ``None``.
      group_size (int, optional): The size of the group in ``w`` that shares a
        scale and bias. Default: ``64``.
      bits (int, optional): The number of bits occupied by each element in
        ``w``. Default: ``4``.
      mode (str, optional): The quantization mode. Default: ``"affine"``.

    Returns:
      array: The dequantized version of ``w``

    Notes:
      The currently supported quantization modes are ``"affine"`` and ``mxfp4``.

      For ``affine`` quantization, given the notation in :func:`quantize`,
      we compute :math:`w_i` from :math:`\hat{w_i}` and corresponding :math:`s`
      and :math:`\beta` as follows

      .. math::

        w_i = s \hat{w_i} + \beta
    """

def diag(a: array, /, k: int = ..., *, stream: Stream | Device | None = ...) -> array:
    """
    Extract a diagonal or construct a diagonal matrix.
    If ``a`` is 1-D then a diagonal matrix is constructed with ``a`` on the
    :math:`k`-th diagonal. If ``a`` is 2-D then the :math:`k`-th diagonal is
    returned.

    Args:
        a (array): 1-D or 2-D input array.
        k (int, optional): The diagonal to extract or construct.
            Default: ``0``.

    Returns:
        array: The extracted diagonal or the constructed diagonal matrix.
    """

def diagonal(
    a: array,
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return specified diagonals.

    If ``a`` is 2-D, then a 1-D array containing the diagonal at the given
    ``offset`` is returned.

    If ``a`` has more than two dimensions, then ``axis1`` and ``axis2``
    determine the 2D subarrays from which diagonals are extracted. The new
    shape is the original shape with ``axis1`` and ``axis2`` removed and a
    new dimension inserted at the end corresponding to the diagonal.

    Args:
      a (array): Input array
      offset (int, optional): Offset of the diagonal from the main diagonal.
        Can be positive or negative. Default: ``0``.
      axis1 (int, optional): The first axis of the 2-D sub-arrays from which
          the diagonals should be taken. Default: ``0``.
      axis2 (int, optional): The second axis of the 2-D sub-arrays from which
          the diagonals should be taken. Default: ``1``.

    Returns:
        array: The diagonals of the array.
    """

def disable_compile() -> None:
    """
    Globally disable compilation. Setting the environment variable
    ``MLX_DISABLE_COMPILE`` can also be used to disable compilation.
    """

def divide(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise division.

    Divide two arrays with numpy-style broadcasting semantics. Either or both
    input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The quotient ``a / b``.
    """

def divmod(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise quotient and remainder.

    The fuction ``divmod(a, b)`` is equivalent to but faster than
    ``(a // b, a % b)``. The function uses numpy-style broadcasting
    semantics. Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        tuple(array, array): The quotient ``a // b`` and remainder ``a % b``.
    """

e: float = ...

def einsum(subscripts: str, *operands, stream: Stream | Device | None = ...) -> array:
    """
    Perform the Einstein summation convention on the operands.

    Args:
      subscripts (str): The Einstein summation convention equation.
      *operands (array): The input arrays.

    Returns:
      array: The output array.
    """

def einsum_path(subscripts: str, *operands):
    """
    Compute the contraction order for the given Einstein summation.

    Args:
      subscripts (str): The Einstein summation convention equation.
      *operands (array): The input arrays.

    Returns:
      tuple(list(tuple(int, int)), str):
        The einsum path and a string containing information about the
        chosen path.
    """

def enable_compile() -> None:
    """
    Globally enable compilation. This will override the environment
    variable ``MLX_DISABLE_COMPILE`` if set.
    """

def equal(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise equality.

    Equality comparison on two arrays with numpy-style broadcasting semantics.
    Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The element-wise comparison ``a == b``.
    """

def erf(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    r"""
    Element-wise error function.

    .. math::
      \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt

    Args:
        a (array): Input array.

    Returns:
        array: The error function of ``a``.
    """

def erfinv(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise inverse of :func:`erf`.

    Args:
        a (array): Input array.

    Returns:
        array: The inverse error function of ``a``.
    """

euler_gamma: float = ...

type MX_ARRAY_TREE = (
    array
    | Module
    | list[MX_ARRAY_TREE]
    | tuple[MX_ARRAY_TREE, ...]
    | Mapping[str, MX_ARRAY_TREE]
)

def eval(*args: MX_ARRAY_TREE | None) -> None:
    """
    Evaluate an :class:`array` or tree of :class:`array`.

    Args:
        *args (arrays or trees of arrays): Each argument can be a single array
          or a tree of arrays. If a tree is given the nodes can be a Python
          :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
          arrays are ignored.
    """

def exp(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise exponential.

    Args:
        a (array): Input array.

    Returns:
        array: The exponential of ``a``.
    """

def expand_dims(
    a: array,
    /,
    axis: int | Sequence[int],
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Add a size one dimension at the given axis.

    Args:
        a (array): Input array.
        axes (int or tuple(int)): The index of the inserted dimensions.

    Returns:
        array: The array with inserted dimensions.
    """

def expm1(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise exponential minus 1.

    Computes ``exp(x) - 1`` with greater precision for small ``x``.

    Args:
        a (array): Input array.

    Returns:
        array: The expm1 of ``a``.
    """

def export_function(
    arg0: object, fun: Callable, *args, shapeless: bool = ..., **kwargs
) -> None:
    """
    Export an MLX function.

    Example input arrays must be provided to export a function. The example
    inputs can be variable ``*args`` and ``**kwargs`` or a tuple of arrays
    and/or dictionary of string keys with array values.

    .. warning::

      This is part of an experimental API which is likely to
      change in future versions of MLX. Functions exported with older
      versions of MLX may not be compatible with future versions.

    Args:
        file (str or Callable): Either a file path to export the function
          to or a callback.
        fun (Callable): A function which takes as input zero or more
          :class:`array` and returns one or more :class:`array`.
        *args (array): Example array inputs to the function.
        shapeless (bool, optional): Whether or not the function allows
          inputs with variable shapes. Default: ``False``.
        **kwargs (array): Additional example keyword array inputs to the
          function.

    Example:

      .. code-block:: python

        def fun(x, y):
            return x + y

        x = mx.array(1)
        y = mx.array([1, 2, 3])
        mx.export_function("fun.mlxfn", fun, x, y=y)
    """

def export_to_dot(file: object, *args, **kwargs) -> None:
    """
    Export a graph to DOT format for visualization.

    A variable number of output arrays can be provided for exporting
    The graph exported will recursively include all unevaluated inputs of
    the provided outputs.

    Args:
        file (str): The file path to export to.
        *args (array): The output arrays.
        **kwargs (dict[str, array]): Provide some names for arrays in the
          graph to make the result easier to parse.

    Example:
      >>> a = mx.array(1) + mx.array(2)
      >>> mx.export_to_dot("graph.dot", a)
      >>> x = mx.array(1)
      >>> y = mx.array(2)
      >>> mx.export_to_dot("graph.dot", x + y, x=x, y=y)
    """

def exporter(file: str, fun: Callable, *, shapeless: bool = ...) -> FunctionExporter:
    """
    Make a callable object to export multiple traces of a function to a file.

    .. warning::

      This is part of an experimental API which is likely to
      change in future versions of MLX. Functions exported with older
      versions of MLX may not be compatible with future versions.

    Args:
        file (str): File path to export the function to.
        shapeless (bool, optional): Whether or not the function allows
          inputs with variable shapes. Default: ``False``.

    Example:

      .. code-block:: python

        def fun(*args):
            return sum(args)

        with mx.exporter("fun.mlxfn", fun) as exporter:
            exporter(mx.array(1))
            exporter(mx.array(1), mx.array(2))
            exporter(mx.array(1), mx.array(2), mx.array(3))
    """

def eye(
    n: int,
    m: int | None = ...,
    k: int = ...,
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Create an identity matrix or a general diagonal matrix.

    Args:
        n (int): The number of rows in the output.
        m (int, optional): The number of columns in the output. Defaults to n.
        k (int, optional): Index of the diagonal. Defaults to 0 (main diagonal).
        dtype (Dtype, optional): Data type of the output array. Defaults to float32.
        stream (Stream, optional): Stream or device. Defaults to None.

    Returns:
        array: An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.
    """

class finfo:
    """Get information on floating-point types."""
    def __init__(self, arg: Dtype, /) -> None: ...
    @property
    def min(self) -> float:
        """The smallest representable number."""

    @property
    def max(self) -> float:
        """The largest representable number."""

    @property
    def eps(self) -> float:
        """
        The difference between 1.0 and the next smallest
        representable number larger than 1.0.
        """

    @property
    def dtype(self) -> Dtype:
        """The :obj:`Dtype`."""

    def __repr__(self) -> str: ...

def flatten(
    a: array,
    /,
    start_axis: int = ...,
    end_axis: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Flatten an array.

    The axes flattened will be between ``start_axis`` and ``end_axis``,
    inclusive. Negative axes are supported. After converting negative axis to
    positive, axes outside the valid range will be clamped to a valid value,
    ``start_axis`` to ``0`` and ``end_axis`` to ``ndim - 1``.

    Args:
        a (array): Input array.
        start_axis (int, optional): The first dimension to flatten. Defaults to ``0``.
        end_axis (int, optional): The last dimension to flatten. Defaults to ``-1``.
        stream (Stream, optional): Stream or device. Defaults to ``None``
          in which case the default stream of the default device is used.

    Returns:
        array: The flattened array.

    Example:
        >>> a = mx.array([[1, 2], [3, 4]])
        >>> mx.flatten(a)
        array([1, 2, 3, 4], dtype=int32)
        >>>
        >>> mx.flatten(a, start_axis=0, end_axis=-1)
        array([1, 2, 3, 4], dtype=int32)
    """

float16: Dtype = ...
float32: Dtype = ...
float64: Dtype = ...
floating: DtypeCategory = ...

def floor(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise floor.

    Args:
        a (array): Input array.

    Returns:
        array: The floor of ``a``.
    """

def floor_divide(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise integer division.

    If either array is a floating point type then it is equivalent to
    calling :func:`floor` after :func:`divide`.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The quotient ``a // b``.
    """

def full(
    shape: int | Sequence[int],
    vals: scalar | array,
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Construct an array with the given value.

    Constructs an array of size ``shape`` filled with ``vals``. If ``vals``
    is an :obj:`array` it must be broadcastable to the given ``shape``.

    Args:
        shape (int or list(int)): The shape of the output array.
        vals (float or int or array): Values to fill the array with.
        dtype (Dtype, optional): Data type of the output array. If
          unspecified the output type is inferred from ``vals``.

    Returns:
        array: The output array with the specified shape and values.
    """

def gather_mm(
    a: array,
    b: array,
    /,
    lhs_indices: array,
    rhs_indices: array,
    *,
    sorted_indices: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Matrix multiplication with matrix-level gather.

    Performs a gather of the operands with the given indices followed by a
    (possibly batched) matrix multiplication of two arrays.  This operation
    is more efficient than explicitly applying a :func:`take` followed by a
    :func:`matmul`.

    The indices ``lhs_indices`` and ``rhs_indices`` contain flat indices
    along the batch dimensions (i.e. all but the last two dimensions) of
    ``a`` and ``b`` respectively.

    For ``a`` with shape ``(A1, A2, ..., AS, M, K)``, ``lhs_indices``
    contains indices from the range ``[0, A1 * A2 * ... * AS)``

    For ``b`` with shape ``(B1, B2, ..., BS, M, K)``, ``rhs_indices``
    contains indices from the range ``[0, B1 * B2 * ... * BS)``

    If only one index is passed and it is sorted, the ``sorted_indices``
    flag can be passed for a possible faster implementation.

    Args:
        a (array): Input array.
        b (array): Input array.
        lhs_indices (array, optional): Integer indices for ``a``. Default: ``None``
        rhs_indices (array, optional): Integer indices for ``b``. Default: ``None``
        sorted_indices (bool, optional): May allow a faster implementation
          if the passed indices are sorted. Default: ``False``.

    Returns:
        array: The output array.
    """

def gather_qmm(
    x: array,
    w: array,
    /,
    scales: array,
    biases: array | None = ...,
    lhs_indices: array | None = ...,
    rhs_indices: array | None = ...,
    transpose: bool = ...,
    group_size: int = ...,
    bits: int = ...,
    mode: str = ...,
    *,
    sorted_indices: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Perform quantized matrix multiplication with matrix-level gather.

    This operation is the quantized equivalent to :func:`gather_mm`.
    Similar to :func:`gather_mm`, the indices ``lhs_indices`` and
    ``rhs_indices`` contain flat indices along the batch dimensions (i.e.
    all but the last two dimensions) of ``x`` and ``w`` respectively.

    Note that ``scales`` and ``biases`` must have the same batch dimensions
    as ``w`` since they represent the same quantized matrix.

    Args:
        x (array): Input array
        w (array): Quantized matrix packed in unsigned integers
        scales (array): The scales to use per ``group_size`` elements of ``w``
        biases (array, optional): The biases to use per ``group_size``
          elements of ``w``. Default: ``None``.
        lhs_indices (array, optional): Integer indices for ``x``. Default: ``None``.
        rhs_indices (array, optional): Integer indices for ``w``. Default: ``None``.
        transpose (bool, optional): Defines whether to multiply with the
          transposed ``w`` or not, namely whether we are performing
          ``x @ w.T`` or ``x @ w``. Default: ``True``.
        group_size (int, optional): The size of the group in ``w`` that
          shares a scale and bias. Default: ``64``.
        bits (int, optional): The number of bits occupied by each element in
          ``w``. Default: ``4``.
        mode (str, optional): The quantization mode. Default: ``"affine"``.
        sorted_indices (bool, optional): May allow a faster implementation
          if the passed indices are sorted. Default: ``False``.

    Returns:
        array: The result of the multiplication of ``x`` with ``w``
          after gathering using ``lhs_indices`` and ``rhs_indices``.
    """

generic: DtypeCategory = ...

def get_active_memory() -> int:
    """
    Get the actively used memory in bytes.

    Note, this will not always match memory use reported by the system because
    it does not include cached memory buffers.
    """

def get_cache_memory() -> int:
    """
    Get the cache size in bytes.

    The cache includes memory not currently used that has not been returned
    to the system allocator.
    """

def get_peak_memory() -> int:
    """
    Get the peak amount of used memory in bytes.

    The maximum memory used recorded from the beginning of the program
    execution or since the last call to :func:`reset_peak_memory`.
    """

gpu: DeviceType = ...

def grad(
    fun: Callable,
    argnums: int | Sequence[int] | None = ...,
    argnames: str | Sequence[str] = ...,
) -> Callable:
    """
    Returns a function which computes the gradient of ``fun``.

    Args:
        fun (Callable): A function which takes a variable number of
          :class:`array` or trees of :class:`array` and returns
          a scalar output :class:`array`.
        argnums (int or list(int), optional): Specify the index (or indices)
          of the positional arguments of ``fun`` to compute the gradient
          with respect to. If neither ``argnums`` nor ``argnames`` are
          provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
          argument.
        argnames (str or list(str), optional): Specify keyword arguments of
          ``fun`` to compute gradients with respect to. It defaults to [] so
          no gradients for keyword arguments by default.

    Returns:
        Callable: A function which has the same input arguments as ``fun`` and
        returns the gradient(s).
    """

def greater(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise greater than.

    Strict greater than on two arrays with numpy-style broadcasting semantics.
    Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The element-wise comparison ``a > b``.
    """

def greater_equal(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise greater or equal.

    Greater than or equal on two arrays with numpy-style broadcasting semantics.
    Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The element-wise comparison ``a >= b``.
    """

def hadamard_transform(
    a: array, scale: float | None = ..., stream: Stream | Device | None = ...
) -> array:
    """
    Perform the Walsh-Hadamard transform along the final axis.

    Equivalent to:

    .. code-block:: python

       from scipy.linalg import hadamard

       y = (hadamard(len(x)) @ x) * scale

    Supports sizes ``n = m*2^k`` for ``m`` in ``(1, 12, 20, 28)`` and ``2^k
    <= 8192`` for float32 and ``2^k <= 16384`` for float16/bfloat16.

    Args:
        a (array): Input array or scalar.
        scale (float): Scale the output by this factor.
          Defaults to ``1/sqrt(a.shape[-1])`` so that the Hadamard matrix is orthonormal.

    Returns:
        array: The transformed array.
    """

def identity(
    n: int, dtype: Dtype | None = ..., *, stream: Stream | Device | None = ...
) -> array:
    """
    Create a square identity matrix.

    Args:
        n (int): The number of rows and columns in the output.
        dtype (Dtype, optional): Data type of the output array. Defaults to float32.
        stream (Stream, optional): Stream or device. Defaults to None.

    Returns:
        array: An identity matrix of size n x n.
    """

class iinfo:
    """Get information on integer types."""
    def __init__(self, arg: Dtype, /) -> None: ...
    @property
    def min(self) -> int:
        """The smallest representable number."""

    @property
    def max(self) -> int:
        """The largest representable number."""

    @property
    def dtype(self) -> Dtype:
        """The :obj:`Dtype`."""

    def __repr__(self) -> str: ...

def imag(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Returns the imaginary part of a complex array.

    Args:
        a (array): Input array.

    Returns:
        array: The imaginary part of ``a``.
    """

def import_function(file: str) -> Callable:
    """
    Import a function from a file.

    The imported function can be called either with ``*args`` and
    ``**kwargs`` or with a tuple of arrays and/or dictionary of string
    keys with array values. Imported functions always return a tuple of
    arrays.

    .. warning::

      This is part of an experimental API which is likely to
      change in future versions of MLX. Functions exported with older
      versions of MLX may not be compatible with future versions.

    Args:
        file (str): The file path to import the function from.

    Returns:
        Callable: The imported function.

    Example:
      >>> fn = mx.import_function("function.mlxfn")
      >>> out = fn(a, b, x=x, y=y)[0]
      >>>
      >>> out = fn((a, b), {"x": x, "y": y}[0]
    """

inexact: DtypeCategory = ...
inf: float = ...

def inner(a: array, b: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the last axes.

    Args:
      a (array): Input array
      b (array): Input array

    Returns:
      array: The inner product.
    """

int16: Dtype = ...
int32: Dtype = ...
int64: Dtype = ...
int8: Dtype = ...
integer: DtypeCategory = ...

def is_available(device: Device) -> bool:
    """Check if a back-end is available for the given device."""

def isclose(
    a: array,
    b: array,
    /,
    rtol: float = ...,
    atol: float = ...,
    *,
    equal_nan: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    Infinite values are considered equal if they have the same sign, NaN values are
    not equal unless ``equal_nan`` is ``True``.

    Two values are considered equal if:

    .. code-block::

     abs(a - b) <= (atol + rtol * abs(b))

    Note unlike :func:`array_equal`, this function supports numpy-style
    broadcasting.

    Args:
        a (array): Input array.
        b (array): Input array.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, NaNs are considered equal.
          Defaults to ``False``.

    Returns:
        array: The boolean output scalar indicating if the arrays are close.
    """

def isfinite(a: array, stream: Stream | Device | None = ...) -> array:
    """
    Return a boolean array indicating which elements are finite.

    An element is finite if it is not infinite or NaN.

    Args:
        a (array): Input array.

    Returns:
        array: The boolean array indicating which elements are finite.
    """

def isinf(a: array, stream: Stream | Device | None = ...) -> array:
    """
    Return a boolean array indicating which elements are +/- inifnity.

    Args:
        a (array): Input array.

    Returns:
        array: The boolean array indicating which elements are +/- infinity.
    """

def isnan(a: array, stream: Stream | Device | None = ...) -> array:
    """
    Return a boolean array indicating which elements are NaN.

    Args:
        a (array): Input array.

    Returns:
        array: The boolean array indicating which elements are NaN.
    """

def isneginf(a: array, stream: Stream | Device | None = ...) -> array:
    """
    Return a boolean array indicating which elements are negative infinity.

    Args:
        a (array): Input array.
        stream (Stream | Device | None): Optional stream or device.

    Returns:
        array: The boolean array indicating which elements are negative infinity.
    """

def isposinf(a: array, stream: Stream | Device | None = ...) -> array:
    """
    Return a boolean array indicating which elements are positive infinity.

    Args:
        a (array): Input array.
        stream (Stream | Device | None): Optional stream or device.

    Returns:
        array: The boolean array indicating which elements are positive infinity.
    """

def issubdtype(arg1: Dtype | DtypeCategory, arg2: Dtype | DtypeCategory) -> bool:
    """
    Check if a :obj:`Dtype` or :obj:`DtypeCategory` is a subtype
    of another.

    Args:
        arg1 (Dtype | DtypeCategory: First dtype or category.
        arg2 (Dtype | DtypeCategory: Second dtype or category.

    Returns:
        bool:
           A boolean indicating if the first input is a subtype of the
           second input.

    Example:

      >>> ints = mx.array([1, 2, 3], dtype=mx.int32)
      >>> mx.issubdtype(ints.dtype, mx.integer)
      True
      >>> mx.issubdtype(ints.dtype, mx.floating)
      False

      >>> floats = mx.array([1, 2, 3], dtype=mx.float32)
      >>> mx.issubdtype(floats.dtype, mx.integer)
      False
      >>> mx.issubdtype(floats.dtype, mx.floating)
      True

      Similar types of different sizes are not subdtypes of each other:

      >>> mx.issubdtype(mx.float64, mx.float32)
      False
      >>> mx.issubdtype(mx.float32, mx.float64)
      False

      but both are subtypes of `floating`:

      >>> mx.issubdtype(mx.float64, mx.floating)
      True
      >>> mx.issubdtype(mx.float32, mx.floating)
      True

      For convenience, dtype-like objects are allowed too:

      >>> mx.issubdtype(mx.float32, mx.inexact)
      True
      >>> mx.issubdtype(mx.signedinteger, mx.floating)
      False
    """

def jvp(
    fun: Callable, primals: list[array], tangents: list[array]
) -> tuple[list[array], list[array]]:
    """
    Compute the Jacobian-vector product.

    This computes the product of the Jacobian of a function ``fun`` evaluated
    at ``primals`` with the ``tangents``.

    Args:
        fun (Callable): A function which takes a variable number of :class:`array`
          and returns a single :class:`array` or list of :class:`array`.
        primals (list(array)): A list of :class:`array` at which to
          evaluate the Jacobian.
        tangents (list(array)): A list of :class:`array` which are the
          "vector" in the Jacobian-vector product. The ``tangents`` should be the
          same in number, shape, and type as the inputs of ``fun`` (i.e. the ``primals``).

    Returns:
        list(array): A list of the Jacobian-vector products which
        is the same in number, shape, and type of the inputs to ``fun``.
    """

def kron(a: array, b: array, *, stream: Stream | Device | None = ...) -> array:
    """
    Compute the Kronecker product of two arrays ``a`` and ``b``.

    Args:
      a (array): The first input array.
      b (array): The second input array.
      stream (Stream | Device | None, optional): Optional stream or
        device for execution. Default: ``None``.

    Returns:
      array: The Kronecker product of ``a`` and ``b``.

    Examples:
      >>> a = mx.array([[1, 2], [3, 4]])
      >>> b = mx.array([[0, 5], [6, 7]])
      >>> result = mx.kron(a, b)
      >>> print(result)
      array([[0, 5, 0, 10],
             [6, 7, 12, 14],
             [0, 15, 0, 20],
             [18, 21, 24, 28]], dtype=int32)
    """

def left_shift(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise left shift.

    Shift the bits of the first input to the left by the second using
    numpy-style broadcasting semantics. Either or both input arrays can
    also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The bitwise left shift ``a << b``.
    """

def less(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise less than.

    Strict less than on two arrays with numpy-style broadcasting semantics.
    Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The element-wise comparison ``a < b``.
    """

def less_equal(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise less than or equal.

    Less than or equal on two arrays with numpy-style broadcasting semantics.
    Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The element-wise comparison ``a <= b``.
    """

def linspace(
    start,
    stop,
    num: int | None = ...,
    dtype: Dtype | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate ``num`` evenly spaced numbers over interval ``[start, stop]``.

    Args:
        start (scalar): Starting value.
        stop (scalar): Stopping value.
        num (int, optional): Number of samples, defaults to ``50``.
        dtype (Dtype, optional): Specifies the data type of the output,
          default to ``float32``.

    Returns:
        array: The range of values.
    """

def load(
    file: str | pathlib.Path,
    /,
    format: str | None = ...,
    return_metadata: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array | dict[str, array]:
    """
    Load array(s) from a binary file.

    The supported formats are ``.npy``, ``.npz``, ``.safetensors``, and
    ``.gguf``.

    Args:
        file (str, pathlib.Path): File in which the array is saved.
        format (str, optional): Format of the file. If ``None``, the
          format is inferred from the file extension. Supported formats:
          ``npy``, ``npz``, and ``safetensors``. Default: ``None``.
        return_metadata (bool, optional): Load the metadata for formats
          which support matadata. The metadata will be returned as an
          additional dictionary. Default: ``False``.
    Returns:
        array or dict:
            A single array if loading from a ``.npy`` file or a dict
            mapping names to arrays if loading from a ``.npz`` or
            ``.safetensors`` file. If ``return_metadata`` is ``True`` an
            additional dictionary of metadata will be returned.

    Warning:

      When loading unsupported quantization formats from GGUF, tensors
      will automatically cast to ``mx.float16``
    """

def log(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise natural logarithm.

    Args:
        a (array): Input array.

    Returns:
        array: The natural logarithm of ``a``.
    """

def log10(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise base-10 logarithm.

    Args:
        a (array): Input array.

    Returns:
        array: The base-10 logarithm of ``a``.
    """

def log1p(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise natural log of one plus the array.

    Args:
        a (array): Input array.

    Returns:
        array: The natural logarithm of one plus ``a``.
    """

def log2(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise base-2 logarithm.

    Args:
        a (array): Input array.

    Returns:
        array: The base-2 logarithm of ``a``.
    """

def logaddexp(
    a: scalar | array,
    b: scalar | array,
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise log-add-exp.

    This is a numerically stable log-add-exp of two arrays with numpy-style
    broadcasting semantics. Either or both input arrays can also be scalars.

    The computation is is a numerically stable version of ``log(exp(a) + exp(b))``.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The log-add-exp of ``a`` and ``b``.
    """

def logcumsumexp(
    a: array,
    /,
    axis: int | None = ...,
    *,
    reverse: bool = ...,
    inclusive: bool = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return the cumulative logsumexp of the elements along the given axis.

    Args:
      a (array): Input array
      axis (int, optional): Optional axis to compute the cumulative logsumexp
        over. If unspecified the cumulative logsumexp of the flattened array is
        returned.
      reverse (bool): Perform the cumulative logsumexp in reverse.
      inclusive (bool): The i-th element of the output includes the i-th
        element of the input.

    Returns:
      array: The output array.
    """

def logical_and(
    a: array, b: array, /, *, stream: Stream | Device | None = ...
) -> array:
    """
    Element-wise logical and.

    Args:
        a (array): First input array or scalar.
        b (array): Second input array or scalar.

    Returns:
        array: The boolean array containing the logical and of ``a`` and ``b``.
    """

def logical_not(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise logical not.

    Args:
        a (array): Input array or scalar.

    Returns:
        array: The boolean array containing the logical not of ``a``.
    """

def logical_or(a: array, b: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise logical or.

    Args:
        a (array): First input array or scalar.
        b (array): Second input array or scalar.

    Returns:
        array: The boolean array containing the logical or of ``a`` and ``b``.
    """

def logsumexp(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    A `log-sum-exp` reduction over the given axes.

    The log-sum-exp reduction is a numerically stable version of:

    .. code-block::

      log(sum(exp(a), axis))

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

def matmul(a: array, b: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Matrix multiplication.

    Perform the (possibly batched) matrix multiplication of two arrays. This function supports
    broadcasting for arrays with more than two dimensions.

    - If the first array is 1-D then a 1 is prepended to its shape to make it
      a matrix. Similarly if the second array is 1-D then a 1 is appended to its
      shape to make it a matrix. In either case the singleton dimension is removed
      from the result.
    - A batched matrix multiplication is performed if the arrays have more than
      2 dimensions.  The matrix dimensions for the matrix product are the last
      two dimensions of each input.
    - All but the last two dimensions of each input are broadcast with one another using
      standard numpy-style broadcasting semantics.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The matrix product of ``a`` and ``b``.
    """

def max(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    A `max` reduction over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

def maximum(
    a: scalar | array,
    b: scalar | array,
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise maximum.

    Take the element-wise max of two arrays with numpy-style broadcasting
    semantics. Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The max of ``a`` and ``b``.
    """

def mean(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Compute the mean(s) over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array of means.
    """

def meshgrid(
    *arrays: array,
    sparse: bool | None = ...,
    indexing: str | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Generate multidimensional coordinate grids from 1-D coordinate arrays

    Args:
        *arrays (array): Input arrays.
        sparse (bool, optional): If ``True``, a sparse grid is returned in which each output
          array has a single non-zero element. If ``False``, a dense grid is returned.
          Defaults to ``False``.
        indexing (str, optional): Cartesian ('xy') or matrix ('ij') indexing of the output arrays.
          Defaults to ``'xy'``.

    Returns:
        list(array): The output arrays.
    """

def min(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    A `min` reduction over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

def minimum(
    a: scalar | array,
    b: scalar | array,
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise minimum.

    Take the element-wise min of two arrays with numpy-style broadcasting
    semantics. Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The min of ``a`` and ``b``.
    """

def moveaxis(
    a: array,
    /,
    source: int,
    destination: int,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Move an axis to a new position.

    Args:
        a (array): Input array.
        source (int): Specifies the source axis.
        destination (int): Specifies the destination axis.

    Returns:
        array: The array with the axis moved.
    """

def multiply(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise multiplication.

    Multiply two arrays with numpy-style broadcasting semantics. Either or both
    input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The multiplication ``a * b``.
    """

nan: float = ...

def nan_to_num(
    a: scalar | array,
    nan: float = ...,
    posinf: float | None = ...,
    neginf: float | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Replace NaN and Inf values with finite numbers.

    Args:
        a (array): Input array
        nan (float, optional): Value to replace NaN with. Default: ``0``.
        posinf (float, optional): Value to replace positive infinities
          with. If ``None``, defaults to largest finite value for the
          given data type. Default: ``None``.
        neginf (float, optional): Value to replace negative infinities
          with. If ``None``, defaults to the negative of the largest
          finite value for the given data type. Default: ``None``.

    Returns:
        array: Output array with NaN and Inf replaced.
    """

def negative(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise negation.

    Args:
        a (array): Input array.

    Returns:
        array: The negative of ``a``.
    """

def new_stream(device: Device) -> Stream:
    """Make a new stream on the given device."""

newaxis: None = ...

def not_equal(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise not equal.

    Not equal comparison on two arrays with numpy-style broadcasting semantics.
    Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The element-wise comparison ``a != b``.
    """

number: DtypeCategory = ...

def ones(
    shape: int | Sequence[int],
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Construct an array of ones.

    Args:
        shape (int or list(int)): The shape of the output array.
        dtype (Dtype, optional): Data type of the output array. If
          unspecified the output type defaults to ``float32``.

    Returns:
        array: The array of ones with the specified shape.
    """

def ones_like(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    An array of ones like the input.

    Args:
        a (array): The input to take the shape and type from.

    Returns:
        array: The output array filled with ones.
    """

def outer(a: array, b: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand.

    Args:
      a (array): Input array
      b (array): Input array

    Returns:
      array: The outer product.
    """

def pad(
    a: array,
    pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
    mode: Literal["constant", "edge"] = ...,
    constant_values: scalar | array = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Pad an array with a constant value

    Args:
        a (array): Input array.
        pad_width (int, tuple(int), tuple(int, int) or list(tuple(int, int))): Number of padded
          values to add to the edges of each axis:``((before_1, after_1),
          (before_2, after_2), ..., (before_N, after_N))``. If a single pair
          of integers is passed then ``(before_i, after_i)`` are all the same.
          If a single integer or tuple with a single integer is passed then
          all axes are extended by the same number on each side.
        mode: Padding mode. One of the following strings:
          "constant" (default): Pads with a constant value.
          "edge": Pads with the edge values of array.
        constant_value (array or scalar, optional): Optional constant value
          to pad the edges of the array with.

    Returns:
        array: The padded array.
    """

def partition(
    a: array,
    /,
    kth: int,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Returns a partitioned copy of the array such that the smaller ``kth``
    elements are first.

    The ordering of the elements in partitions is undefined.

    Args:
        a (array): Input array.
        kth (int): Element at the ``kth`` index will be in its sorted
          position in the output. All elements before the kth index will
          be less or equal to the ``kth`` element and all elements after
          will be greater or equal to the ``kth`` element in the output.
        axis (int or None, optional): Optional axis to partition over.
          If ``None``, this partitions over the flattened array.
          If unspecified, it defaults to ``-1``.

    Returns:
        array: The partitioned array.
    """

def permute_dims(
    a: array,
    /,
    axes: Sequence[int] | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """See :func:`transpose`."""

pi: float = ...

def power(
    a: scalar | array,
    b: scalar | array,
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise power operation.

    Raise the elements of a to the powers in elements of b with numpy-style
    broadcasting semantics. Either or both input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: Bases of ``a`` raised to powers in ``b``.
    """

def prod(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    An product reduction over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

def put_along_axis(
    a: array,
    /,
    indices: array,
    values: array,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Put values along an axis at the specified indices.

    Args:
        a (array): Destination array.
        indices (array): Indices array. These should be broadcastable with
          the input array excluding the `axis` dimension.
        values (array): Values array. These should be broadcastable with
          the indices.

        axis (int or None): Axis in the destination to put the values to. If
          ``axis == None`` the destination is flattened prior to the put
          operation.

    Returns:
        array: The output array.
    """

def quantize(
    w: array,
    /,
    group_size: int = ...,
    bits: int = ...,
    mode: str = ...,
    *,
    stream: Stream | Device | None = ...,
) -> tuple[array, array, array]:
    r"""
    Quantize the matrix ``w`` using ``bits`` bits per element.

    Note, every ``group_size`` elements in a row of ``w`` are quantized
    together. Hence, number of columns of ``w`` should be divisible by
    ``group_size``. In particular, the rows of ``w`` are divided into groups of
    size ``group_size`` which are quantized together.

    .. warning::

      ``quantize`` currently only supports 2D inputs with the second
      dimension divisible by ``group_size``

    The supported quantization modes are ``"affine"`` and ``"mxfp4"``. They
    are described in more detail below.

    Args:
      w (array): Matrix to be quantized
      group_size (int, optional): The size of the group in ``w`` that shares a
        scale and bias. Default: ``64``.
      bits (int, optional): The number of bits occupied by each element of
        ``w`` in the returned quantized matrix. Default: ``4``.
      mode (str, optional): The quantization mode. Default: ``"affine"``.

    Returns:
      tuple: A tuple with either two or three elements containing:

      * w_q (array): The quantized version of ``w``
      * scales (array): The quantization scales
      * biases (array): The quantization biases (returned for ``mode=="affine"``).

    Notes:
      The ``affine`` mode quantizes groups of :math:`g` consecutive
      elements in a row of ``w``. For each group the quantized
      representation of each element :math:`\hat{w_i}` is computed as follows:

      .. math::

        \begin{aligned}
          \alpha &= \max_i w_i \\
          \beta &= \min_i w_i \\
          s &= \frac{\alpha - \beta}{2^b - 1} \\
          \hat{w_i} &= \textrm{round}\left( \frac{w_i - \beta}{s}\right).
        \end{aligned}

      After the above computation, :math:`\hat{w_i}` fits in :math:`b` bits
      and is packed in an unsigned 32-bit integer from the lower to upper
      bits. For instance, for 4-bit quantization we fit 8 elements in an
      unsigned 32 bit integer where the 1st element occupies the 4 least
      significant bits, the 2nd bits 4-7 etc.

      To dequantize the elements of ``w``, we also save :math:`s` and
      :math:`\beta` which are the returned ``scales`` and
      ``biases`` respectively.

      The ``mxfp4`` mode similarly quantizes groups of :math:`g` elements
      of ``w``. For ``mxfp4`` the group size must be ``32``. The elements
      are quantized to 4-bit precision floating-point values (E2M1) with a
      shared 8-bit scale per group. Unlike ``affine`` quantization,
      ``mxfp4`` does not have a bias value. More details on the format can
      be found in the `specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_.
    """

def quantized_matmul(
    x: array,
    w: array,
    /,
    scales: array,
    biases: array | None = ...,
    transpose: bool = ...,
    group_size: int = ...,
    bits: int = ...,
    mode: str = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Perform the matrix multiplication with the quantized matrix ``w``. The
    quantization uses one floating point scale and bias per ``group_size`` of
    elements. Each element in ``w`` takes ``bits`` bits and is packed in an
    unsigned 32 bit integer.

    Args:
      x (array): Input array
      w (array): Quantized matrix packed in unsigned integers
      scales (array): The scales to use per ``group_size`` elements of ``w``
      biases (array, optional): The biases to use per ``group_size``
        elements of ``w``. Default: ``None``.
      transpose (bool, optional): Defines whether to multiply with the
        transposed ``w`` or not, namely whether we are performing
        ``x @ w.T`` or ``x @ w``. Default: ``True``.
      group_size (int, optional): The size of the group in ``w`` that
        shares a scale and bias. Default: ``64``.
      bits (int, optional): The number of bits occupied by each element in
        ``w``. Default: ``4``.
      mode (str, optional): The quantization mode. Default: ``"affine"``.

    Returns:
      array: The result of the multiplication of ``x`` with ``w``.
    """

def radians(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Convert angles from degrees to radians.

    Args:
        a (array): Input array.

    Returns:
        array: The angles in radians.
    """

def real(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Returns the real part of a complex array.

    Args:
        a (array): Input array.

    Returns:
        array: The real part of ``a``.
    """

def reciprocal(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise reciprocal.

    Args:
        a (array): Input array.

    Returns:
        array: The reciprocal of ``a``.
    """

def remainder(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise remainder of division.

    Computes the remainder of dividing a with b with numpy-style
    broadcasting semantics. Either or both input arrays can also be
    scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The remainder of ``a // b``.
    """

def repeat(
    array: array,
    repeats: int,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Repeat an array along a specified axis.

    Args:
        array (array): Input array.
        repeats (int): The number of repetitions for each element.
        axis (int, optional): The axis in which to repeat the array along. If
          unspecified it uses the flattened array of the input and repeats
          along axis 0.
        stream (Stream, optional): Stream or device. Defaults to ``None``.

    Returns:
        array: The resulting repeated array.
    """

def reset_peak_memory() -> None:
    """Reset the peak memory to zero."""

def reshape(
    a: array, /, shape: Sequence[int], *, stream: Stream | Device | None = ...
) -> array:
    """
    Reshape an array while preserving the size.

    Args:
        a (array): Input array.
        shape (tuple(int)): New shape.
        stream (Stream, optional): Stream or device. Defaults to ``None``
          in which case the default stream of the default device is used.

    Returns:
        array: The reshaped array.
    """

def right_shift(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise right shift.

    Shift the bits of the first input to the right by the second using
    numpy-style broadcasting semantics. Either or both input arrays can
    also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The bitwise right shift ``a >> b``.
    """

def roll(
    a: array,
    shift: int | tuple[int],
    axis: int | tuple[int] | None = ...,
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Roll array elements along a given axis.

    Elements that are rolled beyond the end of the array are introduced at
    the beggining and vice-versa.

    If the axis is not provided the array is flattened, rolled and then the
    shape is restored.

    Args:
      a (array): Input array
      shift (int or tuple(int)): The number of places by which elements
        are shifted. If positive the array is rolled to the right, if
        negative it is rolled to the left. If an int is provided but the
        axis is a tuple then the same value is used for all axes.
      axis (int or tuple(int), optional): The axis or axes along which to
        roll the elements.
    """

def round(
    a: array, /, decimals: int = ..., stream: Stream | Device | None = ...
) -> array:
    """
    Round to the given number of decimals.

    Basically performs:

    .. code-block:: python

      s = 10**decimals
      x = round(x * s) / s

    Args:
      a (array): Input array
      decimals (int): Number of decimal places to round to. (default: 0)

    Returns:
      array: An array of the same type as ``a`` rounded to the
      given number of decimals.
    """

def rsqrt(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise reciprocal and square root.

    Args:
        a (array): Input array.

    Returns:
        array: One over the square root of ``a``.
    """

def save(file: str | pathlib.Path, arr: array) -> None:
    """
    Save the array to a binary file in ``.npy`` format.

    Args:
        file (str, pathlib.Path): File to which the array is saved
        arr (array): Array to be saved.
    """

def save_gguf(
    file: str | pathlib.Path,
    arrays: dict[str, array],
    metadata: dict[str, array | str | list[str]],
):
    """
    Save array(s) to a binary file in ``.gguf`` format.

    See the `GGUF documentation
    <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>`_ for
    more information on the format.

    Args:
        file (file, str, pathlib.Path): File in which the array is saved.
        arrays (dict(str, array)): The dictionary of names to arrays to
          be saved.
        metadata (dict(str, array | str | list(str))): The dictionary
           of metadata to be saved. The values can be a scalar or 1D
           obj:`array`, a :obj:`str`, or a :obj:`list` of :obj:`str`.
    """

def save_safetensors(
    file: str | pathlib.Path,
    arrays: dict[str, array],
    metadata: dict[str, str] | None = ...,
):
    """
    Save array(s) to a binary file in ``.safetensors`` format.

    See the `Safetensors documentation
    <https://huggingface.co/docs/safetensors/index>`_ for more
    information on the format.

    Args:
        file (file, str, pathlib.Path): File in which the array is saved.
        arrays (dict(str, array)): The dictionary of names to arrays to
          be saved.
        metadata (dict(str, str), optional): The dictionary of
          metadata to be saved.
    """

def savez(file: str | pathlib.Path, *args, **kwargs):
    """
    Save several arrays to a binary file in uncompressed ``.npz``
    format.

    .. code-block:: python

        import mlx.core as mx

        x = mx.ones((10, 10))
        mx.savez("my_path.npz", x=x)

        import mlx.nn as nn
        from mlx.utils import tree_flatten

        model = nn.TransformerEncoder(6, 128, 4)
        flat_params = tree_flatten(model.parameters())
        mx.savez("model.npz", **dict(flat_params))

    Args:
        file (file, str, pathlib.Path): Path to file to which the arrays are saved.
        *args (arrays): Arrays to be saved.
        **kwargs (arrays): Arrays to be saved. Each array will be saved
          with the associated keyword as the output file name.
    """

def savez_compressed(file: str | pathlib.Path, *args, **kwargs):
    """
    Save several arrays to a binary file in compressed ``.npz`` format.

    Args:
        file (file, str, pathlib.Path): Path to file to which the arrays are saved.
        *args (arrays): Arrays to be saved.
        **kwargs (arrays): Arrays to be saved. Each array will be saved
          with the associated keyword as the output file name.
    """

def segmented_mm(
    a: array, b: array, /, segments: array, *, stream: Stream | Device | None = ...
) -> array:
    """
    Perform a matrix multiplication but segment the inner dimension and
    save the result for each segment separately.

    Args:
      a (array): Input array of shape ``MxK``.
      b (array): Input array of shape ``KxN``.
      segments (array): The offsets into the inner dimension for each segment.

    Returns:
      array: The result per segment of shape ``MxN``.
    """

def set_cache_limit(limit: int) -> int:
    """
    Set the free cache limit.

    If using more than the given limit, free memory will be reclaimed
    from the cache on the next allocation. To disable the cache, set
    the limit to ``0``.

    The cache limit defaults to the memory limit. See
    :func:`set_memory_limit` for more details.

    Args:
      limit (int): The cache limit in bytes.

    Returns:
      int: The previous cache limit in bytes.
    """

def set_default_device(device: Device | DeviceType) -> None:
    """Set the default device."""

def set_default_stream(stream: Stream) -> None:
    """
    Set the default stream.

    This will make the given stream the default for the
    streams device. It will not change the default device.

    Args:
      stream (stream): Stream to make the default.
    """

def set_memory_limit(limit: int) -> int:
    """
    Set the memory limit.

    The memory limit is a guideline for the maximum amount of memory to use
    during graph evaluation. If the memory limit is exceeded and there is no
    more RAM (including swap when available) allocations will result in an
    exception.

    When metal is available the memory limit defaults to 1.5 times the
    maximum recommended working set size reported by the device.

    Args:
      limit (int): Memory limit in bytes.

    Returns:
      int: The previous memory limit in bytes.
    """

def set_wired_limit(limit: int) -> int:
    """
    Set the wired size limit.

    .. note::
       * This function is only useful on macOS 15.0 or higher.
       * The wired limit should remain strictly less than the total
         memory size.

    The wired limit is the total size in bytes of memory that will be kept
    resident. The default value is ``0``.

    Setting a wired limit larger than system wired limit is an error. You can
    increase the system wired limit with:

    .. code-block::

      sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>

    Use :func:`device_info` to query the system wired limit
    (``"max_recommended_working_set_size"``) and the total memory size
    (``"memory_size"``).

    Args:
      limit (int): The wired limit in bytes.

    Returns:
      int: The previous wired limit in bytes.
    """

def sigmoid(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    r"""
    Element-wise logistic sigmoid.

    The logistic sigmoid function is:

    .. math::
      \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

    Args:
        a (array): Input array.

    Returns:
        array: The logistic sigmoid of ``a``.
    """

def sign(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise sign.

    Args:
        a (array): Input array.

    Returns:
        array: The sign of ``a``.
    """

signedinteger: DtypeCategory = ...

def sin(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise sine.

    Args:
        a (array): Input array.

    Returns:
        array: The sine of ``a``.
    """

def sinh(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise hyperbolic sine.

    Args:
        a (array): Input array.

    Returns:
        array: The hyperbolic sine of ``a``.
    """

def slice(
    a: array,
    start_indices: array,
    axes: Sequence[int],
    slice_size: Sequence[int],
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Extract a sub-array from the input array.

    Args:
      a (array): Input array
      start_indices (array): The index location to start the slice at.
      axes (tuple(int)): The axes corresponding to the indices in ``start_indices``.
      slice_size (tuple(int)): The size of the slice.

    Returns:
      array: The sliced output array.

    Example:

      >>> a = mx.array([[1, 2, 3], [4, 5, 6]])
      >>> mx.slice(a, start_indices=mx.array(1), axes=(0,), slice_size=(1, 2))
      array([[4, 5]], dtype=int32)
      >>>
      >>> mx.slice(a, start_indices=mx.array(1), axes=(1,), slice_size=(2, 1))
      array([[2],
             [5]], dtype=int32)
    """

def slice_update(
    a: array,
    update: array,
    start_indices: array,
    axes: Sequence[int],
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Update a sub-array of the input array.

    Args:
      a (array): The input array to update
      update (array): The update array.
      start_indices (array): The index location to start the slice at.
      axes (tuple(int)): The axes corresponding to the indices in ``start_indices``.

    Returns:
      array: The output array with the same shape and type as the input.

    Example:

      >>> a = mx.zeros((3, 3))
      >>> mx.slice_update(a, mx.ones((1, 2)), start_indices=mx.array(1, 1), axes=(0, 1))
      array([[0, 0, 0],
             [0, 1, 0],
             [0, 1, 0]], dtype=float32)
    """

def softmax(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Perform the softmax along the given axis.

    This operation is a numerically stable version of:

    .. code-block::

      exp(a) / sum(exp(a), axis, keepdims=True)

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or axes to compute
         the softmax over. If unspecified this performs the softmax over
         the full array.

    Returns:
        array: The output of the softmax.
    """

def sort(
    a: array,
    /,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Returns a sorted copy of the array.

    Args:
        a (array): Input array.
        axis (int or None, optional): Optional axis to sort over.
          If ``None``, this sorts over the flattened array.
          If unspecified, it defaults to -1 (sorting over the last axis).

    Returns:
        array: The sorted array.
    """

def split(
    a: array,
    /,
    indices_or_sections: int | Sequence[int],
    axis: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Split an array along a given axis.

    Args:
        a (array): Input array.
        indices_or_sections (int or list(int)): If ``indices_or_sections``
          is an integer the array is split into that many sections of equal
          size. An error is raised if this is not possible. If ``indices_or_sections``
          is a list, the list contains the indices of the start of each subarray
          along the given axis.
        axis (int, optional): Axis to split along, defaults to `0`.

    Returns:
        list(array): A list of split arrays.
    """

def sqrt(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise square root.

    Args:
        a (array): Input array.

    Returns:
        array: The square root of ``a``.
    """

def square(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise square.

    Args:
        a (array): Input array.

    Returns:
        array: The square of ``a``.
    """

def squeeze(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Remove length one axes from an array.

    Args:
        a (array): Input array.
        axis (int or tuple(int), optional): Axes to remove. Defaults
          to ``None`` in which case all size one axes are removed.

    Returns:
        array: The output array with size one axes removed.
    """

def stack(
    arrays: list[array],
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Stacks the arrays along a new axis.

    Args:
        arrays (list(array)): A list of arrays to stack.
        axis (int, optional): The axis in the result array along which the
          input arrays are stacked. Defaults to ``0``.
        stream (Stream, optional): Stream or device. Defaults to ``None``.

    Returns:
        array: The resulting stacked array.
    """

def std(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    ddof: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Compute the standard deviation(s) over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.
        ddof (int, optional): The divisor to compute the variance
          is ``N - ddof``, defaults to 0.

    Returns:
        array: The output array of standard deviations.
    """

def stop_gradient(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Stop gradients from being computed.

    The operation is the identity but it prevents gradients from flowing
    through the array.

    Args:
        a (array): Input array.

    Returns:
        array:
          The unchanged input ``a`` but without gradient flowing
          through it.
    """

def stream(s: Stream | Device) -> StreamContext:
    """
    Create a context manager to set the default device and stream.

    Args:
        s: The :obj:`Stream` or :obj:`Device` to set as the default.

    Returns:
        A context manager that sets the default device and stream.

    Example:

    .. code-block::python

      import mlx.core as mx

      # Create a context manager for the default device and stream.
      with mx.stream(mx.cpu):
          # Operations here will use mx.cpu by default.
          pass
    """

def subtract(
    a: scalar | array,
    b: scalar | array,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Element-wise subtraction.

    Subtract one array from another with numpy-style broadcasting semantics. Either or both
    input arrays can also be scalars.

    Args:
        a (array): Input array or scalar.
        b (array): Input array or scalar.

    Returns:
        array: The difference ``a - b``.
    """

def sum(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Sum reduce the array over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.

    Returns:
        array: The output array with the corresponding axes reduced.
    """

def swapaxes(
    a: array, /, axis1: int, axis2: int, *, stream: Stream | Device | None = ...
) -> array:
    """
    Swap two axes of an array.

    Args:
        a (array): Input array.
        axis1 (int): Specifies the first axis.
        axis2 (int): Specifies the second axis.

    Returns:
        array: The array with swapped axes.
    """

def synchronize(stream: Stream | None = ...) -> None:
    """
    Synchronize with the given stream.

    Args:
      stream (Stream, optional): The stream to synchronize with. If ``None``
         then the default stream of the default device is used.
         Default: ``None``.
    """

def take(
    a: array,
    /,
    indices: int | array,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Take elements along an axis.

    The elements are taken from ``indices`` along the specified axis.
    If the axis is not specified the array is treated as a flattened
    1-D array prior to performing the take.

    As an example, if the ``axis=1`` this is equivalent to ``a[:, indices, ...]``.

    Args:
        a (array): Input array.
        indices (int or array): Integer index or input array with integral type.
        axis (int, optional): Axis along which to perform the take. If unspecified
          the array is treated as a flattened 1-D vector.

    Returns:
        array: The indexed values of ``a``.
    """

def take_along_axis(
    a: array,
    /,
    indices: array,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Take values along an axis at the specified indices.

    Args:
        a (array): Input array.
        indices (array): Indices array. These should be broadcastable with
          the input array excluding the `axis` dimension.
        axis (int or None): Axis in the input to take the values from. If
          ``axis == None`` the array is flattened to 1D prior to the indexing
          operation.

    Returns:
        array: The output array.
    """

def tan(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise tangent.

    Args:
        a (array): Input array.

    Returns:
        array: The tangent of ``a``.
    """

def tanh(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    Element-wise hyperbolic tangent.

    Args:
        a (array): Input array.

    Returns:
        array: The hyperbolic tangent of ``a``.
    """

def tensordot(
    a: array,
    b: array,
    /,
    axes: int | list[Sequence[int]] = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Compute the tensor dot product along the specified axes.

    Args:
        a (array): Input array
        b (array): Input array
        axes (int or list(list(int)), optional): The number of dimensions to
          sum over. If an integer is provided, then sum over the last
          ``axes`` dimensions of ``a`` and the first ``axes`` dimensions of
          ``b``. If a list of lists is provided, then sum over the
          corresponding dimensions of ``a`` and ``b``. Default: 2.

    Returns:
        array: The tensor dot product.
    """

def tile(
    a: array,
    reps: int | Sequence[int],
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Construct an array by repeating ``a`` the number of times given by ``reps``.

    Args:
      a (array): Input array
      reps (int or list(int)): The number of times to repeat ``a`` along each axis.

    Returns:
      array: The tiled array.
    """

def topk(
    a: array,
    /,
    k: int,
    axis: int | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Returns the ``k`` largest elements from the input along a given axis.

    The elements will not necessarily be in sorted order.

    Args:
        a (array): Input array.
        k (int): ``k`` top elements to be returned
        axis (int or None, optional): Optional axis to select over.
          If ``None``, this selects the top ``k`` elements over the
          flattened array. If unspecified, it defaults to ``-1``.

    Returns:
        array: The top ``k`` elements from the input.
    """

def trace(
    a: array,
    /,
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Return the sum along a specified diagonal in the given array.

    Args:
      a (array): Input array
      offset (int, optional): Offset of the diagonal from the main diagonal.
        Can be positive or negative. Default: ``0``.
      axis1 (int, optional): The first axis of the 2-D sub-arrays from which
          the diagonals should be taken. Default: ``0``.
      axis2 (int, optional): The second axis of the 2-D sub-arrays from which
          the diagonals should be taken. Default: ``1``.
      dtype (Dtype, optional): Data type of the output array. If
          unspecified the output type is inferred from the input array.

    Returns:
        array: Sum of specified diagonal.
    """

def transpose(
    a: array,
    /,
    axes: Sequence[int] | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Transpose the dimensions of the array.

    Args:
        a (array): Input array.
        axes (list(int), optional): Specifies the source axis for each axis
          in the new array. The default is to reverse the axes.

    Returns:
        array: The transposed array.
    """

def tri(
    n: int,
    m: int,
    k: int,
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Args:
      n (int): The number of rows in the output.
      m (int, optional): The number of cols in the output. Defaults to ``None``.
      k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
      dtype (Dtype, optional): Data type of the output array. Defaults to ``float32``.
      stream (Stream, optional): Stream or device. Defaults to ``None``.

    Returns:
      array: Array with its lower triangle filled with ones and zeros elsewhere
    """

def tril(x: array, k: int, *, stream: Stream | Device | None = ...) -> array:
    """
    Zeros the array above the given diagonal.

    Args:
      x (array): input array.
      k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
      stream (Stream, optional): Stream or device. Defaults to ``None``.

    Returns:
      array: Array zeroed above the given diagonal
    """

def triu(x: array, k: int, *, stream: Stream | Device | None = ...) -> array:
    """
    Zeros the array below the given diagonal.

    Args:
      x (array): input array.
      k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
      stream (Stream, optional): Stream or device. Defaults to ``None``.

    Returns:
      array: Array zeroed below the given diagonal
    """

uint16: Dtype = ...
uint32: Dtype = ...
uint64: Dtype = ...
uint8: Dtype = ...

def unflatten(
    a: array,
    /,
    axis: int,
    shape: Sequence[int],
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Unflatten an axis of an array to a shape.

    Args:
        a (array): Input array.
        axis (int): The axis to unflatten.
        shape (tuple(int)): The shape to unflatten to. At most one
          entry can be ``-1`` in which case the corresponding size will be
          inferred.
        stream (Stream, optional): Stream or device. Defaults to ``None``
          in which case the default stream of the default device is used.

    Returns:
        array: The unflattened array.

    Example:
        >>> a = mx.array([1, 2, 3, 4])
        >>> mx.unflatten(a, 0, (2, -1))
        array([[1, 2], [3, 4]], dtype=int32)
    """

unsignedinteger: DtypeCategory = ...

def value_and_grad(
    fun: Callable,
    argnums: int | Sequence[int] | None = ...,
    argnames: str | Sequence[str] = ...,
) -> Callable:
    """
    Returns a function which computes the value and gradient of ``fun``.

    The function passed to :func:`value_and_grad` should return either
    a scalar loss or a tuple in which the first element is a scalar
    loss and the remaining elements can be anything.

    .. code-block:: python

        import mlx.core as mx

        def mse(params, inputs, targets):
            outputs = forward(params, inputs)
            lvalue = (outputs - targets).square().mean()
            return lvalue

        # Returns lvalue, dlvalue/dparams
        lvalue, grads = mx.value_and_grad(mse)(params, inputs, targets)

        def lasso(params, inputs, targets, a=1.0, b=1.0):
            outputs = forward(params, inputs)
            mse = (outputs - targets).square().mean()
            l1 = mx.abs(outputs - targets).mean()

            loss = a*mse + b*l1

            return loss, mse, l1

        (loss, mse, l1), grads = mx.value_and_grad(lasso)(params, inputs, targets)

    Args:
        fun (Callable): A function which takes a variable number of
          :class:`array` or trees of :class:`array` and returns
          a scalar output :class:`array` or a tuple the first element
          of which should be a scalar :class:`array`.
        argnums (int or list(int), optional): Specify the index (or indices)
          of the positional arguments of ``fun`` to compute the gradient
          with respect to. If neither ``argnums`` nor ``argnames`` are
          provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
          argument.
        argnames (str or list(str), optional): Specify keyword arguments of
          ``fun`` to compute gradients with respect to. It defaults to [] so
          no gradients for keyword arguments by default.

    Returns:
        Callable: A function which returns a tuple where the first element
        is the output of `fun` and the second element is the gradients w.r.t.
        the loss.
    """

def var(
    a: array,
    /,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    ddof: int = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Compute the variance(s) over the given axes.

    Args:
        a (array): Input array.
        axis (int or list(int), optional): Optional axis or
          axes to reduce over. If unspecified this defaults
          to reducing over the entire array.
        keepdims (bool, optional): Keep reduced axes as
          singleton dimensions, defaults to `False`.
        ddof (int, optional): The divisor to compute the variance
          is ``N - ddof``, defaults to 0.

    Returns:
        array: The output array of variances.
    """

def view(
    a: scalar | array, dtype: Dtype, stream: Stream | Device | None = ...
) -> array:
    """
    View the array as a different type.

    The output shape changes along the last axis if the input array's
    type and the input ``dtype`` do not have the same size.

    Note: the view op does not imply that the input and output arrays share
    their underlying data. The view only gaurantees that the binary
    representation of each element (or group of elements) is the same.

    Args:
        a (array): Input array or scalar.
        dtype (Dtype): The data type to change to.

    Returns:
        array: The array with the new type.
    """

def vjp(
    fun: Callable, primals: list[array], cotangents: list[array]
) -> tuple[list[array], list[array]]:
    """
    Compute the vector-Jacobian product.

    Computes the product of the ``cotangents`` with the Jacobian of a
    function ``fun`` evaluated at ``primals``.

    Args:
      fun (Callable): A function which takes a variable number of :class:`array`
        and returns a single :class:`array` or list of :class:`array`.
      primals (list(array)): A list of :class:`array` at which to
        evaluate the Jacobian.
      cotangents (list(array)): A list of :class:`array` which are the
        "vector" in the vector-Jacobian product. The ``cotangents`` should be the
        same in number, shape, and type as the outputs of ``fun``.

    Returns:
        list(array): A list of the vector-Jacobian products which
        is the same in number, shape, and type of the outputs of ``fun``.
    """

def vmap(fun: Callable, in_axes: object = ..., out_axes: object = ...) -> Callable:
    """
    Returns a vectorized version of ``fun``.

    Args:
        fun (Callable): A function which takes a variable number of
          :class:`array` or a tree of :class:`array` and returns
          a variable number of :class:`array` or a tree of :class:`array`.
        in_axes (int, optional): An integer or a valid prefix tree of the
          inputs to ``fun`` where each node specifies the vmapped axis. If
          the value is ``None`` then the corresponding input(s) are not vmapped.
          Defaults to ``0``.
        out_axes (int, optional): An integer or a valid prefix tree of the
          outputs of ``fun`` where each node specifies the vmapped axis. If
          the value is ``None`` then the corresponding outputs(s) are not vmapped.
          Defaults to ``0``.

    Returns:
        Callable: The vectorized function.
    """

def where(
    condition: scalar | array,
    x: scalar | array,
    y: scalar | array,
    /,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Select from ``x`` or ``y`` according to ``condition``.

    The condition and input arrays must be the same shape or
    broadcastable with each another.

    Args:
      condition (array): The condition array.
      x (array): The input selected from where condition is ``True``.
      y (array): The input selected from where condition is ``False``.

    Returns:
        array: The output containing elements selected from
        ``x`` and ``y``.
    """

def zeros(
    shape: int | Sequence[int],
    dtype: Dtype | None = ...,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Construct an array of zeros.

    Args:
        shape (int or list(int)): The shape of the output array.
        dtype (Dtype, optional): Data type of the output array. If
          unspecified the output type defaults to ``float32``.

    Returns:
        array: The array of zeros with the specified shape.
    """

def zeros_like(a: array, /, *, stream: Stream | Device | None = ...) -> array:
    """
    An array of zeros like the input.

    Args:
        a (array): The input to take the shape and type from.

    Returns:
        array: The output array filled with zeros.
    """

scalar: TypeAlias = int | float | bool
list_or_scalar: TypeAlias = scalar | list["list_or_scalar"]
bool_: Dtype = ...

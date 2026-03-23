from typing import Sequence

from mlx.core import Device, Dtype, Stream, array

class Group:
    """
    An :class:`mlx.core.distributed.Group` represents a group of independent mlx
    processes that can communicate.
    """
    def rank(self) -> int:
        """Get the rank of this process"""

    def size(self) -> int:
        """Get the size of the group"""

    def split(self, color: int, key: int = ...) -> Group:
        """
        Split the group to subgroups based on the provided color.

        Processes that use the same color go to the same group. The ``key``
        argument defines the rank in the new group. The smaller the key the
        smaller the rank. If the key is negative then the rank in the
        current group is used.

        Args:
          color (int): A value to group processes into subgroups.
          key (int, optional): A key to optionally change the rank ordering
            of the processes.
        """

def all_gather(
    x: array, *, group: Group | None = ..., stream: Stream | Device | None = ...
) -> array:
    """
    Gather arrays from all processes.

    Gather the ``x`` arrays from all processes in the group and concatenate
    them along the first axis. The arrays should all have the same shape.

    Args:
      x (array): Input array.
      group (Group): The group of processes that will participate in the
        gather. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: The concatenation of all ``x`` arrays.
    """

def all_max(
    x: array, *, group: Group | None = ..., stream: Stream | Device | None = ...
) -> array:
    """
    All reduce max.

    Find the maximum of the ``x`` arrays from all processes in the group.

    Args:
      x (array): Input array.
      group (Group): The group of processes that will participate in the
        reduction. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: The maximum of all ``x`` arrays.
    """

def all_min(
    x: array, *, group: Group | None = ..., stream: Stream | Device | None = ...
) -> array:
    """
    All reduce min.

    Find the minimum of the ``x`` arrays from all processes in the group.

    Args:
      x (array): Input array.
      group (Group): The group of processes that will participate in the
        reduction. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: The minimum of all ``x`` arrays.
    """

def all_sum(
    x: array, *, group: Group | None = ..., stream: Stream | Device | None = ...
) -> array:
    """
    All reduce sum.

    Sum the ``x`` arrays from all processes in the group.

    Args:
      x (array): Input array.
      group (Group): The group of processes that will participate in the
        reduction. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: The sum of all ``x`` arrays.
    """

def init(strict: bool = ..., backend: str = ...) -> Group:
    """
    Initialize the communication backend and create the global communication group.

    Example:

      .. code:: python

        import mlx.core as mx

        group = mx.distributed.init(backend="ring")

    Args:
      strict (bool, optional): If set to False it returns a singleton group
        in case ``mx.distributed.is_available()`` returns False otherwise
        it throws a runtime error. Default: ``False``
      backend (str, optional): Which distributed backend to initialize.
        Possible values ``mpi``, ``ring``, ``nccl``, ``any``. If set to ``any`` all
        available backends are tried and the first one that succeeds
        becomes the global group which will be returned in subsequent
        calls. Default: ``any``

    Returns:
      Group: The group representing all the launched processes.
    """

def is_available() -> bool:
    """Check if a communication backend is available."""

def recv(
    shape: Sequence[int],
    dtype: Dtype,
    src: int,
    *,
    group: Group | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Recv an array with shape ``shape`` and dtype ``dtype`` from process
    with rank ``src``.

    Args:
      shape (tuple[int]): The shape of the array we are receiving.
      dtype (Dtype): The data type of the array we are receiving.
      src (int): Rank of the source process in the group.
      group (Group): The group of processes that will participate in the
        recv. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: The array that was received from ``src``.
    """

def recv_like(
    x: array,
    src: int,
    *,
    group: Group | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Recv an array with shape and type like ``x`` from process with rank
    ``src``.

    It is equivalent to calling ``mx.distributed.recv(x.shape, x.dtype, src)``.

    Args:
      x (array): An array defining the shape and dtype of the array we are
        receiving.
      src (int): Rank of the source process in the group.
      group (Group): The group of processes that will participate in the
        recv. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: The array that was received from ``src``.
    """

def send(
    x: array,
    dst: int,
    *,
    group: Group | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """
    Send an array from the current process to the process that has rank
    ``dst`` in the group.

    Args:
      x (array): Input array.
      dst (int): Rank of the destination process in the group.
      group (Group): The group of processes that will participate in the
        sned. If set to ``None`` the global group is used. Default:
        ``None``.
      stream (Stream, optional): Stream or device. Defaults to ``None``
        in which case the default stream of the default device is used.

    Returns:
      array: An array identical to ``x`` which when evaluated the send is performed.
    """

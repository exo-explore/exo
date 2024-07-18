# HCQ Compatible Runtime

## Overview

The main aspect of HCQ-compatible runtimes is how they interact with devices. In HCQ, all interactions with devices occur in a hardware-friendly manner using [command queues](#commandqueues). This approach allows commands to be issued directly to devices, bypassing runtime overhead such as HIP or CUDA. Additionally, by using the HCQ API, these runtimes can benefit from various optimizations and features, including [HCQGraph](#hcqgraph) and built-in profiling capabilities.

### Command Queues

To interact with devices, there are 2 types of queues: `HWComputeQueue` and `HWCopyQueue`. Commands which are defined in a base `HWCommandQueue` class should be supported by both queues. These methods are timestamp and synchronization methods like [signal](#tinygrad.device.HWCommandQueue.signal) and [wait](#tinygrad.device.HWCommandQueue.wait).

For example, the following Python code enqueues a wait, execute, and signal command on the HCQ-compatible device:
```python
HWComputeQueue().wait(signal_to_wait, value_to_wait) \
                .exec(program, kernargs_ptr, global_dims, local_dims) \
                .signal(signal_to_fire, value_to_fire) \
                .submit(your_device)
```

Each runtime should implement the required functions that are defined in the `HWCommandQueue`, `HWComputeQueue`, and `HWCopyQueue` classes.

::: tinygrad.device.HWCommandQueue
    options:
        members: [
            "signal",
            "wait",
            "timestamp",
            "update_signal",
            "update_wait",
            "submit",
        ]
        show_source: false

::: tinygrad.device.HWComputeQueue
    options:
        members: [
            "memory_barrier",
            "exec",
            "update_exec",
        ]
        show_source: false

::: tinygrad.device.HWCopyQueue
    options:
        members: [
            "copy",
            "update_copy",
        ]
        show_source: false

#### Implementing custom commands

To implement custom commands in the queue, use the @hcq_command decorator for your command implementations.

::: tinygrad.device.hcq_command
    options:
        members: [
            "copy",
            "update_copy",
        ]
        show_source: false

### HCQ Compatible Device

The `HCQCompatCompiled` class defines the API for HCQ-compatible devices. This class serves as an abstract base class that device-specific implementations should inherit from and implement.

::: tinygrad.device.HCQCompatCompiled
    options:
        members: [
            "_alloc_signal",
            "_free_signal",
            "_read_signal",
            "_read_timestamp",
            "_set_signal",
            "_wait_signal",
            "_gpu2cpu_time",
        ]
        show_source: false

#### Signals

Signals are device-dependent structures used for synchronization and timing in HCQ-compatible devices. They should be designed to record both a `value` and a `timestamp` within the same signal. The following Python code demonstrates the usage of signals:

```python
signal = your_device._alloc_signal()

HWComputeQueue().timestamp(signal) \
                .signal(signal, value_to_fire) \
                .submit(your_device)

your_device._wait_signal(signal, value_to_fire)
timestamp = your_device._read_timestamp()
```

##### Synchronization signals

Each HCQ-compatible device must allocate two signals for global synchronization purposes. These signals are passed to the `HCQCompatCompiled` base class during initialization: an active timeline signal `self.timeline_signal` and a shadow timeline signal `self._shadow_timeline_signal` which helps to handle signal value overflow issues. You can find more about synchronization in the [synchronization section](#synchronization)

### HCQ Compatible Allocator

The `HCQCompatAllocator` base class simplifies allocator logic by leveraging [command queues](#commandqueues) abstractions. This class efficiently handles copy and transfer operations, leaving only the alloc and free functions to be implemented by individual backends.

::: tinygrad.device.HCQCompatAllocator
    options:
        members: [
            "_alloc",
            "_free",
        ]
        show_source: false

#### HCQ Allocator Result Protocol

Backends must adhere to the `HCQCompatAllocRes` protocol when returning allocation results.

::: tinygrad.device.HCQCompatAllocRes
    options:
        members: true
        show_source: false

### HCQ Compatible Program

The `HCQCompatProgram` is a helper base class for defining programs compatible with HCQ-compatible devices. Currently, the arguments consist of pointers to buffers, followed by `vals` fields. The convention expects a packed struct containing the passed pointers, followed by `vals` located at `kernargs_args_offset`.

::: tinygrad.device.HCQCompatProgram
    options:
        members: true
        show_source: false

### Synchronization

HCQ-compatible devices use a global timeline signal for synchronizing all operations. This mechanism ensures proper ordering and completion of tasks across the device. By convention, `self.timeline_value` points to the next value to signal. So, to wait for all previous operations on the device to complete, wait for `self.timeline_value - 1` value. The following Python code demonstrates the typical usage of signals to synchronize execution to other operations on the device:

```python
HWComputeQueue().wait(your_device.timeline_signal, your_device.timeline_value - 1) \
                .exec(...)
                .signal(your_device.timeline_signal, your_device.timeline_value) \
                .submit(your_device)
your_device.timeline_value += 1

# Optionally wait for execution
your_device._wait_signal(your_device.timeline_signal, your_device.timeline_value - 1)
```

## HCQGraph

[HCQGraph](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/graph/hcq.py) is a core feature that implements `GraphRunner` for HCQ-compatible devices. `HCQGraph` builds a static `HWComputeQueue` and `HWCopyQueue` for all operations per device. To optimize enqueue time, only the necessary parts of the queues are updated for each run using the update APIs of the queues, avoiding a complete rebuild.
Optionally, queues can implement a `bind` API, which allows further optimization by eliminating the need to copy the queues into the device ring.

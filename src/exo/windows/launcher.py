"""Windows window that starts exo with a device and a memory limit.

The Mac app injects settings through the child environment. This launcher
does the same for ``EXO_TINYGRAD_DEVICES`` and ``OVERRIDE_MEMORY_MB``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from typing import Literal, final

from exo.backends.tinygrad_engine import windows_backend_for_adapter_name
from exo.shared.types.backends import Backend
from exo.utils.pydantic_ext import FrozenModel

type WindowsDeviceName = Literal["AMD", "CUDA", "CPU"]
type MemoryKind = Literal["vram", "ram"]


@final
class WindowsDeviceChoice(FrozenModel):
    """One row in the device dropdown.

    ``device_name`` is the value written to ``EXO_TINYGRAD_DEVICES``.
    ``memory_kind`` selects the megabyte label: VRAM for a GPU, RAM for CPU.
    """

    device_name: WindowsDeviceName
    label: str
    memory_kind: MemoryKind


def windows_device_choices(
    adapter_names: Sequence[str],
    has_nvidia_gpu: bool,
) -> list[WindowsDeviceChoice]:
    """List the devices the launcher can start.

    Each AMD or NVIDIA adapter is its own row, labeled with the adapter name
    and VRAM. NVML with no NVIDIA name adds one CUDA row. CPU is always last
    and labeled RAM, so a GPU machine can still join as ``WinCPU``.
    """
    choices: list[WindowsDeviceChoice] = []
    saw_cuda = False
    for adapter_name in adapter_names:
        stripped_name = adapter_name.strip()
        if stripped_name == "":
            continue
        identity = windows_backend_for_adapter_name(stripped_name)
        if identity is Backend.WinAMD:
            choices.append(
                WindowsDeviceChoice(
                    device_name="AMD",
                    label=f"{stripped_name} (VRAM)",
                    memory_kind="vram",
                )
            )
        elif identity is Backend.WinCUDA:
            saw_cuda = True
            choices.append(
                WindowsDeviceChoice(
                    device_name="CUDA",
                    label=f"{stripped_name} (VRAM)",
                    memory_kind="vram",
                )
            )
    if has_nvidia_gpu and not saw_cuda:
        choices.append(
            WindowsDeviceChoice(
                device_name="CUDA",
                label="CUDA (VRAM)",
                memory_kind="vram",
            )
        )
    choices.append(
        WindowsDeviceChoice(
            device_name="CPU",
            label="CPU (RAM)",
            memory_kind="ram",
        )
    )
    return choices


def memory_field_label(choice: WindowsDeviceChoice) -> str:
    """Return the megabyte caption for the selected device."""
    if choice.memory_kind == "vram":
        return "VRAM (MB)"
    return "RAM (MB)"


def parse_memory_limit_megabytes(text: str) -> int | None:
    """Parse the memory box.

    A blank box returns ``None`` so the existing probe is left unchanged.

    Raises:
        ValueError: The window shows this on the status line when the text
            is not a positive number of megabytes.
    """
    stripped = text.strip()
    if stripped == "":
        return None
    if not stripped.isdecimal():
        raise ValueError("Memory limit must be a positive number of megabytes")
    return _require_positive_megabytes(int(stripped))


def launch_environment(
    base: Mapping[str, str],
    device_name: WindowsDeviceName,
    memory_limit_megabytes: int | None,
) -> dict[str, str]:
    """Return the environment for the exo child.

    ``device_name`` becomes ``EXO_TINYGRAD_DEVICES``. A positive memory limit
    becomes ``OVERRIDE_MEMORY_MB``. ``None`` removes that variable so a blank
    box does not keep a stale override from the parent.

    Raises:
        ValueError: The window shows this on the status line when
            ``memory_limit_megabytes`` is below 1.
    """
    environment = dict(base)
    environment["EXO_TINYGRAD_DEVICES"] = device_name
    if memory_limit_megabytes is None:
        environment.pop("OVERRIDE_MEMORY_MB", None)
    else:
        environment["OVERRIDE_MEMORY_MB"] = str(
            _require_positive_megabytes(memory_limit_megabytes)
        )
    return environment


def exit_status(exit_code: int, output_lines: Sequence[str]) -> str:
    """Return the status line after the child leaves.

    A non-zero exit keeps the last output line so a missing extension is
    visible. A clean exit shows only the code.
    """
    if exit_code != 0 and output_lines:
        return f"exited {exit_code}: {output_lines[-1]}"
    return f"exited {exit_code}"


def _require_positive_megabytes(memory_limit_megabytes: int) -> int:
    if memory_limit_megabytes < 1:
        raise ValueError("Memory limit must be a positive number of megabytes")
    return memory_limit_megabytes


def start_exo_process(
    device_name: WindowsDeviceName,
    memory_limit_megabytes: int | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Spawn ``python -m exo`` with the launcher environment.

    Raises:
        ValueError: ``launch_environment`` raises when the memory limit is
            below 1. The window catches that and shows it on the status line
            before this function is called.
    """
    base = os.environ if environment is None else environment
    child_environment = launch_environment(base, device_name, memory_limit_megabytes)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        [sys.executable, "-m", "exo"],
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )


def _drain_process_output(
    process: subprocess.Popen[str],
    output_lines: list[str],
) -> None:
    stdout = process.stdout
    if stdout is None:
        return
    for line in stdout:
        output_lines.append(line.rstrip())
        del output_lines[:-20]


def open_launcher(choices: Sequence[WindowsDeviceChoice]) -> None:
    """Show the device dropdown, memory box, and Start button.

    Tk is imported here so tests can import the launch helpers without a
    display. Start does nothing while the child is still running.
    """
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("exo")
    root.resizable(False, False)

    labels = [choice.label for choice in choices]
    selected_label = tk.StringVar(value=labels[0])
    memory_label = tk.StringVar(value=memory_field_label(choices[0]))
    memory_text = tk.StringVar()
    status_text = tk.StringVar(value="stopped")

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0)

    ttk.Label(frame, text="Device").grid(row=0, column=0, sticky="w")
    device_menu = ttk.Combobox(
        frame,
        textvariable=selected_label,
        values=labels,
        state="readonly",
        width=42,
    )
    device_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

    ttk.Label(frame, textvariable=memory_label).grid(
        row=1, column=0, sticky="w", pady=(8, 0)
    )
    ttk.Entry(frame, textvariable=memory_text, width=12).grid(
        row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
    )

    process_holder: list[subprocess.Popen[str] | None] = [None]
    output_lines: list[str] = []
    validation_error = False

    def selected_choice() -> WindowsDeviceChoice:
        label = selected_label.get()
        for choice in choices:
            if choice.label == label:
                return choice
        return choices[0]

    def on_device_changed(_event: object) -> None:
        memory_label.set(memory_field_label(selected_choice()))

    device_menu.bind("<<ComboboxSelected>>", on_device_changed)

    def start() -> None:
        nonlocal validation_error
        current = process_holder[0]
        if current is not None and current.poll() is None:
            return
        try:
            memory_limit = parse_memory_limit_megabytes(memory_text.get())
        except ValueError as error:
            validation_error = True
            status_text.set(str(error))
            return
        validation_error = False
        output_lines.clear()
        process = start_exo_process(selected_choice().device_name, memory_limit)
        process_holder[0] = process
        threading.Thread(
            target=_drain_process_output,
            args=(process, output_lines),
            daemon=True,
        ).start()
        status_text.set("running")

    ttk.Button(frame, text="Start", command=start).grid(
        row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0)
    )
    ttk.Label(frame, textvariable=status_text).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
    )

    def refresh_status() -> None:
        current = process_holder[0]
        if current is not None and not validation_error:
            code = current.poll()
            if code is None:
                status_text.set("running")
            else:
                status_text.set(exit_status(code, output_lines))
        root.after(500, refresh_status)

    root.after(500, refresh_status)
    root.mainloop()


def main() -> None:
    """Open the Windows launcher.

    Raises:
        SystemExit: The console script exits on macOS and Linux. The message
            is written to stderr.
    """
    if sys.platform != "win32":
        print("exo-windows runs on Windows.", file=sys.stderr)
        raise SystemExit(1)
    from exo.utils.info_gatherer.info_gatherer import windows_device_probe

    adapter_names, has_nvidia_gpu = windows_device_probe()
    open_launcher(windows_device_choices(adapter_names, has_nvidia_gpu))

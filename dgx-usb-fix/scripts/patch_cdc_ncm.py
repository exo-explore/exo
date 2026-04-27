#!/usr/bin/env python3
"""Patch Linux cdc_ncm.c for Apple 05ac:1905 direct USB-C networking."""

from __future__ import annotations

import re
import sys
from pathlib import Path


APPLE_MARKER = "Apple Mac direct USB-C networking quirk"
MODULE_MARKER = 'MODULE_INFO(dgx_usb_fix, "apple-05ac-1905-cdc-ncm");'


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def insert_apple_device_matches(source: str) -> str:
    if "0x05ac, 0x1905, 0" in source and "0x05ac, 0x1905, 2" in source:
        return source

    if "apple_private_interface_info" not in source:
        die("apple_private_interface_info is missing from cdc_ncm.c")

    apple_entries = (
        f"\t/* {APPLE_MARKER}: no status endpoint on control interfaces. */\n"
        "\t{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0),\n"
        "\t  .driver_info = (unsigned long)&apple_private_interface_info,\n"
        "\t},\n"
        "\t{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2),\n"
        "\t  .driver_info = (unsigned long)&apple_private_interface_info,\n"
        "\t},\n"
        "\n"
    )

    generic_match = re.compile(
        r"(\t/\* Generic CDC-NCM devices \*/\n"
        r"\t\{ USB_INTERFACE_INFO\(USB_CLASS_COMM,\s*\n"
        r"\t\tUSB_CDC_SUBCLASS_NCM, USB_CDC_PROTO_NONE\),\s*\n"
        r"\t\t\.driver_info = \(unsigned long\)&cdc_ncm_info,\s*\n"
        r"\t\},)",
        re.MULTILINE,
    )
    patched, count = generic_match.subn(apple_entries + r"\1", source, count=1)
    if count == 1:
        return patched

    fallback = re.compile(
        r"(\{\s*USB_INTERFACE_INFO\s*\(\s*USB_CLASS_COMM\s*,\s*"
        r"USB_CDC_SUBCLASS_NCM\s*,\s*USB_CDC_PROTO_NONE\s*\)\s*,\s*"
        r"\.driver_info\s*=\s*\(unsigned long\)&cdc_ncm_info\s*,\s*\},)",
        re.DOTALL,
    )
    patched, count = fallback.subn(apple_entries + r"\1", source, count=1)
    if count != 1:
        die("could not find generic CDC-NCM class match entry")
    return patched


def patch_legacy_status_endpoint_check(source: str) -> str:
    if "FLAG_LINK_INTR" in source:
        return source

    old_check = re.compile(
        r"if\s*\(\s*!dev->in\s*\|\|\s*!dev->out\s*\|\|\s*!dev->status\s*\)",
        re.DOTALL,
    )
    new_check = (
        "if (!dev->in || !dev->out || "
        "(!dev->status && "
        "!(le16_to_cpu(dev->udev->descriptor.idVendor) == 0x05ac && "
        "le16_to_cpu(dev->udev->descriptor.idProduct) == 0x1905)))"
    )
    patched, count = old_check.subn(new_check, source, count=1)
    if count != 1:
        die("could not patch legacy missing-status-endpoint check")
    return patched


def add_module_marker(source: str) -> str:
    if MODULE_MARKER in source:
        return source

    marker_patterns = [
        r"(MODULE_DESCRIPTION\([^\n]+\);\n)",
        r"(MODULE_LICENSE\([^\n]+\);\n)",
    ]
    for pattern in marker_patterns:
        patched, count = re.subn(pattern, MODULE_MARKER + "\n" + r"\1", source, count=1)
        if count == 1:
            return patched

    die("could not find MODULE_DESCRIPTION or MODULE_LICENSE for module marker")


def verify(source: str) -> None:
    required = [
        "USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0)",
        "USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2)",
        ".driver_info = (unsigned long)&apple_private_interface_info",
        MODULE_MARKER,
    ]
    for token in required:
        if token not in source:
            die(f"verification failed, missing token: {token}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} /path/to/drivers/net/usb/cdc_ncm.c", file=sys.stderr)
        return 2

    path = Path(argv[1])
    source = path.read_text()
    source = insert_apple_device_matches(source)
    source = patch_legacy_status_endpoint_check(source)
    source = add_module_marker(source)
    verify(source)
    path.write_text(source)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

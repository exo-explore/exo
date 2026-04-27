# Motivation

Last updated: 2026-04-27.

## Problem

We want direct Mac mini to Spark networking over a USB-C-looking cable plugged into USB-C-looking ports. Ethernet adapters, RJ45 cables, and other physical workarounds are out of scope for this path. Thunderbolt would be acceptable physically, but the observed connection is USB/xHCI rather than Thunderbolt networking.

Spark sees the Mac as an Apple USB device:

```text
05ac:1905 Apple, Inc. Mac
```

The Mac exposes two CDC-NCM network functions over USB. The descriptors are close enough to standard CDC-NCM that Linux's `cdc_ncm` driver tries to bind, but both Apple NCM control interfaces have zero endpoints:

```text
Interface 0: NCM Control, bNumEndpoints 0
Interface 1: NCM Data, bulk IN/OUT endpoints
Interface 2: NCM Control, bNumEndpoints 0
Interface 3: NCM Data, bulk IN/OUT endpoints
```

The stock Spark kernel driver follows the generic CDC-NCM path, expects a status/interrupt endpoint, and rejects the device:

```text
cdc_ncm 4-1:1.0: bind() failure
cdc_ncm 4-1:1.2: bind() failure
```

That means Linux never creates a network interface for the USB-C link.

## Cleanest Technical Fix

The cleanest technical fix is a small kernel-driver quirk in `drivers/net/usb/cdc_ncm.c`.

Conceptually, add explicit matches for Apple's Mac USB-C networking device:

```c
{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0),
  .driver_info = (unsigned long)&apple_private_interface_info,
},
{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2),
  .driver_info = (unsigned long)&apple_private_interface_info,
},
```

Why this works:

- The Mac's data interfaces do have usable bulk IN/OUT endpoints.
- The missing piece is only the control/status interrupt endpoint expected by the generic NCM path.
- Linux already has Apple-specific NCM handling for other Apple devices.
- Mapping `05ac:1905` to `apple_private_interface_info` makes the driver treat this as an Apple private-interface case that does not require the missing link-status interrupt endpoint.

The prior script in `tmp/spark/spark-build-apple-cdc-ncm.sh` is attempting exactly this kind of patch. It also includes a fallback edit for older kernel trees where the missing-status-endpoint check is hardcoded differently.

Long term, this should ideally be shipped by NVIDIA/Ubuntu/upstream Linux so the Spark's stock kernel can bind the Mac USB-C networking device without local hacks.

## Why Not Do The Kernel Patch Now

Spark currently has Secure Boot enabled. A locally built out-of-tree `cdc_ncm.ko` is likely rejected unless it is signed by a key trusted by the running kernel.

The practical kernel-module options are:

- enroll a local signing key into the machine's Secure Boot trust path, typically through MOK/UEFI boot-time enrollment
- disable Secure Boot
- wait for a signed vendor kernel/module update

For this task, changing boot security state, enrolling signing keys through firmware/boot UI, or disabling Secure Boot is not acceptable for now. Even if MOK enrollment is not literally a BIOS menu on all systems, it is still a boot-security trust change and usually requires an interactive reboot flow.

That is why `dgxusbd` is exploring a userspace workaround.

## Userspace Workaround

The userspace idea is to avoid the kernel `cdc_ncm` host driver entirely:

```text
Apple Mac USB CDC-NCM function
  <-> userspace USB control/bulk transfers
  <-> userspace CDC-NCM NTB parser/builder
  <-> Linux TAP interface
  <-> Linux IP stack
```

Because the stock kernel driver fails to bind, the USB interfaces remain unbound (`Driver=[none]`) and can potentially be claimed by a userspace USB driver. The userspace process then creates a TAP interface and forwards raw Ethernet frames between TAP and Apple's CDC-NCM bulk endpoints.

This keeps the physical USB-C-to-USB-C cable requirement while avoiding:

- custom kernel modules
- Secure Boot key enrollment
- disabling Secure Boot
- BIOS/firmware changes

Tradeoff: it is more code than the kernel quirk and may perform worse than an in-kernel driver. It is still useful as a proof-of-viability path and a debugging tool while waiting for a vendor or upstream kernel fix.


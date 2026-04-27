# Motivation

Last updated: 2026-04-27.

## Problem

We want direct Mac mini to Spark networking over a USB-C-looking cable plugged
into USB-C-looking ports. Ethernet adapters, RJ45 cables, and other physical
workarounds are out of scope for this path. Thunderbolt would be acceptable
physically, but the observed connection is USB/xHCI rather than Thunderbolt
networking.

Spark sees the Mac as:

```text
05ac:1905 Apple, Inc. Mac
```

The Mac exposes two CDC-NCM network functions:

```text
Interface 0: NCM control, bNumEndpoints 0
Interface 1: NCM data, bulk IN/OUT endpoints
Interface 2: NCM control, bNumEndpoints 0
Interface 3: NCM data, bulk IN/OUT endpoints
```

The functions are close enough to generic CDC-NCM that Linux's `cdc_ncm` driver
tries to bind. The generic match path expects a link-status interrupt endpoint,
so binding fails:

```text
cdc_ncm 4-1:1.0: bind() failure
cdc_ncm 4-1:1.2: bind() failure
```

Linux therefore never creates a normal network interface for the USB-C link.

## Why Route A Is Clean

The smallest technical fix is a `cdc_ncm.c` device quirk:

```c
{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0),
  .driver_info = (unsigned long)&apple_private_interface_info,
},
{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2),
  .driver_info = (unsigned long)&apple_private_interface_info,
},
```

Linux already has Apple-specific CDC-NCM driver-info records for Apple devices.
`apple_private_interface_info` is specifically the Apple private path without
`FLAG_LINK_INTR`, which means the missing status endpoint is not fatal. It also
keeps `FLAG_SEND_ZLP`, so the mature Linux transmit path handles short-packet
and zero-length-packet behavior that userspace route B had to rediscover.

## Why Route B Was Explored

Spark has Secure Boot enabled. Loading a locally built `cdc_ncm.ko` requires one
of:

- a signing key already trusted by the running kernel
- enrolling a local Machine Owner Key through the MOK flow
- disabling Secure Boot
- a signed vendor kernel/module update

Route B (`rust/dgxusbd`) avoided that boot-trust problem by claiming the unbound
USB interfaces in userspace and bridging them to TAP. It proved the descriptors
and hardware path are viable, and it carried packets. Its remaining blocker was
Spark-to-Mac CDC-NCM transmit compatibility under sustained traffic. Route A
should avoid that class of bug by using Linux's in-kernel CDC-NCM implementation.

## Long-Term Target

The local module is a lab workaround. The durable fix should be shipped by
NVIDIA/Ubuntu or upstream Linux as a small `cdc_ncm` quirk for Apple `05ac:1905`.

# Protocol Notes

Last updated: 2026-04-27.

## Observed Apple USB Device

Spark sees:

```text
ID 05ac:1905 Apple, Inc. Mac
Product: Mac
Manufacturer: Apple Inc.
```

Observed CDC-NCM pairs:

```text
Pair A:
  control interface: 0
  data interface:    1
  bulk IN endpoint:  0x81
  bulk OUT endpoint: 0x01
  MAC string:        D2DCB4CCF72D

Pair B:
  control interface: 2
  data interface:    3
  bulk IN endpoint:  0x82
  bulk OUT endpoint: 0x02
  MAC string:        D2DCB4CCF70D
```

Both data interfaces expose bulk endpoints on alternate setting 1.

The route-A-critical descriptor fact:

```text
Control interfaces 0 and 2 have bNumEndpoints 0.
```

## Linux `cdc_ncm` Behavior

Linux 6.17 has this endpoint guard in `cdc_ncm_bind_common`:

```c
if (!dev->in || !dev->out ||
    (!dev->status && dev->driver_info->flags & FLAG_LINK_INTR)) {
        ...
}
```

The generic CDC-NCM class match uses `cdc_ncm_info`, whose flags include
`FLAG_LINK_INTR`. With no status endpoint, the Apple device fails bind.

Linux already has:

```c
static const struct driver_info apple_private_interface_info = {
  .description = "CDC NCM (Apple Private)",
  .flags = FLAG_POINTTOPOINT | FLAG_NO_SETINT | FLAG_MULTI_PACKET
           | FLAG_ETHER | FLAG_SEND_ZLP,
  ...
};
```

That driver info omits `FLAG_LINK_INTR`, so a missing status endpoint is not
fatal. It also includes `FLAG_SEND_ZLP`, which matters for Apple transmit
compatibility.

## Route-A Patch

Add explicit matches before the generic CDC-NCM class match:

```c
{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0),
  .driver_info = (unsigned long)&apple_private_interface_info,
},
{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2),
  .driver_info = (unsigned long)&apple_private_interface_info,
},
```

This changes only the match path for the Apple Mac direct USB-C networking
function. It does not implement a new CDC-NCM data path.

## Why This Helps Route B's TX Problem

Route B proved Mac-to-Spark RX and basic bidirectional ICMP. Its remaining
problem was Spark-to-Mac sustained TCP/UDP, likely tied to exact CDC-NCM transmit
shape: NDP placement, ZLP/short-packet behavior, sequence behavior, alignment,
and batching.

Route A delegates all of that to Linux `cdc_ncm_tx_fixup`, including:

- multi-frame NTB batching
- timeout-based NTB flush
- negotiated input/output sizing
- packet filter setup
- optional ZLP/short-packet behavior through driver flags

## Legacy Kernel Fallback

Older kernel trees may have a direct endpoint guard:

```c
if (!dev->in || !dev->out || !dev->status)
```

The Python patcher contains a fallback to relax that check specifically for
`05ac:1905` if the newer `FLAG_LINK_INTR` logic is absent. On observed Linux
6.17, the explicit Apple private match is the important change.

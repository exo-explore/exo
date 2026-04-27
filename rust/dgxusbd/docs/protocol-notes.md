# Protocol Notes

Last updated: 2026-04-27.

## Observed Apple USB Device

Spark sees the Mac as:

```text
ID 05ac:1905 Apple, Inc. Mac
Product: Mac
Manufacturer: Apple Inc.
```

The observed configuration has four interfaces forming two CDC-NCM functions.

Pair A:

```text
Control interface: 0
Data interface:    1
Bulk IN endpoint:  0x81
Bulk OUT endpoint: 0x01
MAC string:        D2DCB4CCF72D
```

Pair B:

```text
Control interface: 2
Data interface:    3
Bulk IN endpoint:  0x82
Bulk OUT endpoint: 0x02
MAC string:        D2DCB4CCF70D
```

Both data interfaces expose their bulk endpoints on alternate setting 1.

Important descriptor detail:

```text
Control interfaces 0 and 2 have bNumEndpoints 0.
```

The Linux generic `cdc_ncm` path expects a status/interrupt endpoint and fails to bind. The existing kernel-module patch maps `05ac:1905` to Apple's private-interface quirk, which skips that requirement.

## CDC-NCM Data Shape

CDC-NCM does not send one Ethernet frame per USB transfer. It sends one Network Transfer Block (NTB) containing:

```text
NTH16 header
NDP16 table
DPE16 entries: Ethernet frame offset and length
Ethernet frame bytes
padding/alignment
```

MVP assumptions:

- NTB16 only.
- No CRC.
- One selected interface pair.
- Ignore interrupt/status endpoint because Apple does not provide one here.
- Treat the USB device being present/open as link carrier.

## Control Requests To Investigate

Likely needed or useful:

- `GET_NTB_PARAMETERS`
- `SET_NTB_INPUT_SIZE`
- `SET_ETHERNET_PACKET_FILTER`
- possibly `GET_NTB_INPUT_SIZE`

Do not assume all control requests are required for the first proof of life. Start by reading parameters and setting packet filter/input size if the device accepts those requests.

## TAP Direction Mapping

USB bulk IN to TAP:

```text
read USB transfer
parse NTB
for each datagram, write raw Ethernet frame to TAP
```

TAP to USB bulk OUT:

```text
read raw Ethernet frame from TAP
pack into NTB
write USB transfer
```

The Linux IP stack owns ARP, IPv4, IPv6, TCP, UDP, DHCP, etc. `dgxusbd` should forward opaque Ethernet frames and avoid becoming an IP stack.

## Useful References

- Linux `cdc_ncm.c`: `drivers/net/usb/cdc_ncm.c`
- Linux USB CDC UAPI constants/structs: `include/uapi/linux/usb/cdc.h`
- Linux TAP docs: kernel `networking/tuntap`


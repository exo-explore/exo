# Quickstart And Handoff

Last updated: 2026-04-27.

This is the shortest path for a new session picking up route A.

## Current Working State

Route A is working on Spark with the patched kernel module installed:

```text
Spark host: jensen@gx10-a174
Spark repo: ~/Desktop/exo
Spark kernel: 6.17.0-1014-nvidia
Patched module: /lib/modules/6.17.0-1014-nvidia/updates/dgx-usb-fix/cdc_ncm.ko
Module marker: dgx_usb_fix: apple-05ac-1905-cdc-ncm
Module signer: DGX USB Fix Module Signing
```

The Apple `05ac:1905` USB device exposes two CDC-NCM functions. Both now bind
to Linux `cdc_ncm`:

```text
USB interface 00 -> Spark enxd2dcb4ccf72d
USB interface 02 -> Spark enxd2dcb4ccf70d
```

Current verified IPv6 link-local pairs:

```text
spark@fe80::5786:d998:b0b6:9041%enxd2dcb4ccf72d
  <-> mac@fe80::8a2:83dc:50cd:d9a%en5

spark@fe80::dc3d:13fd:b127:97ac%enxd2dcb4ccf70d
  <-> mac@fe80::d0dc:b4ff:fecc:f7f2%anpi0
```

These addresses are link-local. They may change if NetworkManager profiles,
stable secrets, or the Mac-side interface state changes; remap with the commands
below before relying on them.

## Start A Session

```sh
ssh jensen@gx10-a174
cd ~/Desktop/exo
git pull --ff-only
nix develop .#dgx-usb-fix
```

Useful wrapped commands from the dev shell:

```sh
dgx-usb-fix-diagnose
dgx-usb-fix-install
dgx-usb-fix-configure-link-local
dgx-usb-fix-create-mok-key
```

The wrappers preserve the Nix `PATH` through `sudo`.

## Verify Route A

```sh
modinfo -n cdc_ncm
modinfo cdc_ncm | grep -Ei 'dgx_usb_fix|v05ACp1905|signer'
lsusb -t
ip -br link
ip -br -6 addr
```

Expected:

```text
modinfo -n cdc_ncm -> /lib/modules/.../updates/dgx-usb-fix/cdc_ncm.ko
lsusb -t -> interfaces 0, 1, 2, 3 use Driver=cdc_ncm
ip -br -6 addr -> both enxd2... interfaces have fe80::/64 addresses
```

## Keep Link-Local Addresses Stable

Spark NetworkManager initially auto-created generic wired profiles that tried
DHCP/RA. When no DHCP/RA appeared, NM marked activation failed and withdrew
otherwise valid `fe80::` addresses. The fix is:

```sh
dgx-usb-fix-configure-link-local
```

It matches only Apple `05ac:1905` `cdc_ncm` interfaces and sets:

```text
ipv4.method disabled
ipv6.method link-local
ipv6.addr-gen-mode stable-privacy
connection.autoconnect yes
```

## Map Interfaces

Spark side:

```sh
ip -br -6 addr show enxd2dcb4ccf72d
ip -br -6 addr show enxd2dcb4ccf70d
for i in enxd2dcb4ccf72d enxd2dcb4ccf70d; do
  udevadm info -q property -p /sys/class/net/$i | grep -E 'ID_USB_INTERFACE_NUM|ID_PATH='
done
```

Mac side:

```sh
ssh e2@e2 'for i in en5 anpi0 en6 en7 anpi1 anpi2; do printf "==%s==\n" $i; ifconfig $i 2>/dev/null | grep -E "ether |inet6 |status:"; done'
```

Neighbor tables after pinging are also useful:

```sh
ip -6 neigh show dev enxd2dcb4ccf72d
ip -6 neigh show dev enxd2dcb4ccf70d
ssh e2@e2 'ndp -an | grep -E "%en5|%anpi0"'
```

## Ping Current Pairs

From Spark:

```sh
ping -6 "fe80::8a2:83dc:50cd:d9a%enxd2dcb4ccf72d"
ping -6 "fe80::d0dc:b4ff:fecc:f7f2%enxd2dcb4ccf70d"
```

From the Mac:

```sh
ssh e2@e2 'ping6 "fe80::5786:d998:b0b6:9041%en5"'
ssh e2@e2 'ping6 "fe80::dc3d:13fd:b127:97ac%anpi0"'
```

## Next Work

- Run TCP/UDP throughput tests over each link-local pair.
- Decide whether both CDC-NCM functions are useful or whether route A should
  bind only one function.
- If a new Spark kernel boots, rerun `dgx-usb-fix-install`; MOK enrollment can
  be reused if `/root/MOK.priv` and `/root/MOK.der` are still the enrolled key.

# Current Status

Last updated: 2026-04-27.

## Repository State

Local repo:

- Path: `/home/royalguard/Desktop/exo-all/networking-related/exo-dgx-mac-usb-fix`
- Branch: `andrei/dgx-mac-usb-fix`
- Route-A tooling path: `dgx-usb-fix/`

Spark repo:

- Host: `jensen@gx10-a174`
- Path: `~/Desktop/exo`
- Branch: `andrei/dgx-mac-usb-fix`
- Latest route-A tooling, including the link-local helper, has been pulled and
  validated on Spark. Run `git rev-parse --short HEAD` in `~/Desktop/exo` for
  the exact current commit.

## Hardware And Kernel

Spark:

- Kernel observed: `Linux 6.17.0-1014-nvidia aarch64`
- Kernel package observed: `linux-image-6.17.0-1014-nvidia 6.17.0-1014.14`
- Source package mapping observed:
  - binary source wrapper: `linux-signed-nvidia-6.17 6.17.0-1014.14`
  - real C source package: `linux-nvidia-6.17 6.17.0-1014.14`
- Headers present at `/lib/modules/6.17.0-1014-nvidia/build`.
- Secure Boot enabled.
- `enP7s7` is Realtek PCI Ethernet via `r8127`, not the USB-C Mac link.

Mac:

- SSH target: `e2@e2`
- OS observed: Darwin 25.4.0
- macOS reported no Thunderbolt device connected on its Thunderbolt/USB4 buses.

USB-C link:

- Spark sees `05ac:1905 Apple, Inc. Mac`.
- `lsusb -t` showed a 10000M USB path.
- The Mac exposes two CDC-NCM pairs:
  - control interface 0, data interface 1, IN `0x81`, OUT `0x01`,
    MAC string `D2DCB4CCF72D`
  - control interface 2, data interface 3, IN `0x82`, OUT `0x02`,
    MAC string `D2DCB4CCF70D`
- Control interfaces 0 and 2 have no status/interrupt endpoint.

## Route-A Module Status

Route A has been built, signed, installed, and loaded on Spark.

Observed after install:

```text
modinfo -n cdc_ncm
/lib/modules/6.17.0-1014-nvidia/updates/dgx-usb-fix/cdc_ncm.ko

dgx_usb_fix:    apple-05ac-1905-cdc-ncm
alias:          usb:v05ACp1905d*dc*dsc*dp*ic*isc*ip*in02*
alias:          usb:v05ACp1905d*dc*dsc*dp*ic*isc*ip*in00*
signer:         DGX USB Fix Module Signing
```

The pre-patch failure was:

```text
cdc_ncm 4-1:1.0: bind() failure
cdc_ncm 4-1:1.2: bind() failure
```

After route A, both functions bound:

```text
4-1:1.0 + 4-1:1.1 -> Spark enxd2dcb4ccf72d
4-1:1.2 + 4-1:1.3 -> Spark enxd2dcb4ccf70d
```

## Current IPv6 Link-Local Map

Current verified pairs:

```text
spark@fe80::5786:d998:b0b6:9041%enxd2dcb4ccf72d
  <-> mac@fe80::8a2:83dc:50cd:d9a%en5

spark@fe80::dc3d:13fd:b127:97ac%enxd2dcb4ccf70d
  <-> mac@fe80::d0dc:b4ff:fecc:f7f2%anpi0
```

Both directions were verified with scoped IPv6 ping.

Notes:

- `fe80::/64` here is normal IPv6 link-local, not a global routed prefix.
- Spark uses NetworkManager stable/private link-local addresses, so they are
  not derived from the Spark interface MACs.
- Mac `en5` has an IPv4 APIPA address too, but route-A validation currently
  only needs IPv6 link-local.

## NetworkManager Link-Local Fix

Problem observed after the module started working:

- NetworkManager auto-created `Wired connection 1` and `Wired connection 2`.
- The profiles used automatic IPv4/IPv6 config.
- With no DHCP/RA on the direct link, NetworkManager eventually marked
  activation failed and withdrew otherwise valid `fe80::` addresses.

Fix:

```sh
dgx-usb-fix-configure-link-local
```

That helper matches only Apple `05ac:1905` interfaces driven by `cdc_ncm` and
sets:

```text
ipv4.method disabled
ipv6.method link-local
ipv6.addr-gen-mode stable-privacy
connection.autoconnect yes
```

The user reported this now works.

## Route-A Tooling Status

Implemented:

- `dgx-usb-fix/build-and-install.sh`
  - derives the matching Ubuntu source package from `linux-image-$(uname -r)`
  - downloads matching source files from Launchpad
  - runs `scripts/patch_cdc_ncm.py`
  - builds `cdc_ncm.ko` as an external module against running kernel headers
  - signs the module when Secure Boot is enabled
  - installs into `/lib/modules/$(uname -r)/updates/dgx-usb-fix/cdc_ncm.ko`
  - runs `depmod`
  - optionally reloads `cdc_ncm`
- `dgx-usb-fix/create-mok-key.sh`
  - creates `/root/MOK.priv` and `/root/MOK.der`
  - intentionally does not import or enroll the cert
- `dgx-usb-fix/configure-link-local.sh`
  - configures Apple `05ac:1905` `cdc_ncm` NetworkManager profiles for
    IPv6 link-local-only operation
- `dgx-usb-fix/diagnose.sh`
  - read-only diagnostics for kernel, Secure Boot, module aliases, USB state,
    and kernel logs
- `dgx-usb-fix/parts.nix`
  - provides the build shell as `nix develop .#dgx-usb-fix`
  - exports default MOK paths
  - provides root wrappers: `dgx-usb-fix-diagnose`,
    `dgx-usb-fix-create-mok-key`, `dgx-usb-fix-configure-link-local`,
    and `dgx-usb-fix-install`

Validation run locally and on Spark:

```sh
bash -n dgx-usb-fix/build-and-install.sh \
  dgx-usb-fix/create-mok-key.sh \
  dgx-usb-fix/configure-link-local.sh \
  dgx-usb-fix/diagnose.sh \
  tmp/spark/spark-build-apple-cdc-ncm-clean.sh \
  tmp/spark/spark-build-apple-cdc-ncm.sh
python3 -m py_compile dgx-usb-fix/scripts/patch_cdc_ncm.py
nix eval .#devShells.x86_64-linux.dgx-usb-fix.drvPath
```

## Current Next Step

The kernel bind problem is fixed for the current Spark/Mac link. The next useful
work is throughput and stability validation:

- Run `iperf3` TCP and UDP over each link-local pair.
- Decide whether both CDC-NCM functions are useful or whether route A should
  bind only one function.
- Record any throughput and loss findings in this file.
- If Spark boots a new kernel, rerun `dgx-usb-fix-install`; the MOK enrollment
  can be reused if `/root/MOK.priv` and `/root/MOK.der` remain available.

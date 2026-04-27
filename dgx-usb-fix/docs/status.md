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
- Latest route-A tooling pulled and validated on Spark through commit `9658406d`.

## Hardware Findings Shared With Route B

Spark:

- Kernel observed: `Linux 6.17.0-1014-nvidia aarch64`
- Kernel package observed: `linux-image-6.17.0-1014-nvidia 6.17.0-1014.14`
- Source package mapping observed:
  - binary source wrapper: `linux-signed-nvidia-6.17 6.17.0-1014.14`
  - real C source package: `linux-nvidia-6.17 6.17.0-1014.14`
- Headers present at `/lib/modules/6.17.0-1014-nvidia/build`.
- Secure Boot enabled.
- No `/sys/class/udc` was present during investigation.
- No Thunderbolt bus was observed on Spark.
- `enP7s7` is Realtek PCI Ethernet via `r8127`, not the USB-C Mac link.

Mac:

- SSH target: `e2@e2`
- OS observed: Darwin 25.4.0
- macOS reported no Thunderbolt device connected on its Thunderbolt/USB4 buses.
- USB peer interface observed as `Ethernet Adapter (en5)` with address
  `d2:dc:b4:cc:f7:d2`.

USB-C link:

- Spark sees `05ac:1905 Apple, Inc. Mac`.
- `lsusb -t` showed a 10000M USB path.
- The Mac exposes two CDC-NCM pairs:
  - control interface 0, data interface 1, IN `0x81`, OUT `0x01`,
    MAC string `D2DCB4CCF72D`
  - control interface 2, data interface 3, IN `0x82`, OUT `0x02`,
    MAC string `D2DCB4CCF70D`
- Control interfaces 0 and 2 have no status/interrupt endpoint.
- Stock `cdc_ncm` failed with:

```text
cdc_ncm 4-1:1.0: bind() failure
cdc_ncm 4-1:1.2: bind() failure
```

## Route-A Tooling Status

Implemented:

- `dgx-usb-fix/build-and-install.sh`
  - derives the matching Ubuntu source package from `linux-image-$(uname -r)`
  - downloads the matching source files from Launchpad
  - runs `scripts/patch_cdc_ncm.py`
  - builds `cdc_ncm.ko` as an external module against running kernel headers
  - signs the module when Secure Boot is enabled
  - installs into `/lib/modules/$(uname -r)/updates/dgx-usb-fix/cdc_ncm.ko`
  - runs `depmod`
  - optionally reloads `cdc_ncm`
- `dgx-usb-fix/create-mok-key.sh`
  - creates `/root/MOK.priv` and `/root/MOK.der`
  - intentionally does not import or enroll the cert
- `dgx-usb-fix/diagnose.sh`
  - read-only diagnostics for kernel, Secure Boot, module aliases, USB state,
    and kernel logs
- `dgx-usb-fix/configure-link-local.sh`
  - configures Apple `05ac:1905` `cdc_ncm` NetworkManager profiles for
    IPv6 link-local-only operation
  - prevents DHCP/RA activation failure from withdrawing `fe80::` addresses
- `dgx-usb-fix/parts.nix`
  - provides the build shell as `nix develop .#dgx-usb-fix`
  - exports the default MOK paths
  - provides root wrappers: `dgx-usb-fix-diagnose`,
    `dgx-usb-fix-create-mok-key`, `dgx-usb-fix-configure-link-local`,
    and `dgx-usb-fix-install`
- `tmp/spark/spark-build-apple-cdc-ncm*.sh`
  - now compatibility wrappers around `dgx-usb-fix/build-and-install.sh`

Validation run on Spark:

```sh
bash -n dgx-usb-fix/build-and-install.sh \
  dgx-usb-fix/create-mok-key.sh \
  dgx-usb-fix/configure-link-local.sh \
  dgx-usb-fix/diagnose.sh \
  tmp/spark/spark-build-apple-cdc-ncm-clean.sh \
  tmp/spark/spark-build-apple-cdc-ncm.sh
python3 -m py_compile dgx-usb-fix/scripts/patch_cdc_ncm.py
```

Result: success.

Remote source patch validation:

- The Python patcher was run against upstream Linux 6.17 `drivers/net/usb/cdc_ncm.c`.
- It inserted explicit `05ac:1905` matches for interfaces `0` and `2`.
- It inserted `MODULE_INFO(dgx_usb_fix, "apple-05ac-1905-cdc-ncm");`.
- The reference patch applies with `patch --dry-run` to the same source shape.

## Not Yet Run

The actual route-A module build/install/load has not been run in this session.
Reasons:

- Secure Boot is enabled.
- No current proof of enrolled `/root/MOK.priv` and `/root/MOK.der` was available
  without root.
- Creating and enrolling MOK keys changes machine boot trust and requires an
  interactive reboot-time flow.

## Current Expected Next Step

If no MOK key is already enrolled, follow [lab-workflow.md](lab-workflow.md):

1. Enter the Nix shell.
2. Create `/root/MOK.priv` and `/root/MOK.der`.
3. Import `/root/MOK.der` with `mokutil`.
4. Reboot once and enroll the cert in the MOK Manager pre-boot UI.
5. Build, sign, install, and reload `cdc_ncm`.
6. Replug the USB-C cable and validate a Linux netdev appears.

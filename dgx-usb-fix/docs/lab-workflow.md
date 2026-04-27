# Lab Workflow

Last updated: 2026-04-27.

## Machines

Spark / DGX:

- SSH: `ssh jensen@gx10-a174`
- Repo: `~/Desktop/exo`
- Branch: `andrei/dgx-mac-usb-fix`
- Nix installed.
- Runtime host for the route-A kernel module.

Mac mini:

- SSH: `ssh e2@e2`
- Used for observing macOS interface state and testing peer connectivity.
- Nix installed.
- Observed USB peer interface: `en5`.

Local workstation:

- Repo: `/home/royalguard/Desktop/exo-all/networking-related/exo-dgx-mac-usb-fix`
- Branch: `andrei/dgx-mac-usb-fix`
- Main editing location.

## Build Environment

Use the repo dev shell. Do not install compilers or build dependencies from the
script.

```sh
cd ~/Desktop/exo
nix develop .#dgx-usb-fix
```

The shell also provides root wrappers that preserve the Nix `PATH` across
`sudo`:

```sh
dgx-usb-fix-diagnose
dgx-usb-fix-create-mok-key
dgx-usb-fix-configure-link-local
dgx-usb-fix-install
```

The shell provides compiler/build tooling (`gcc`, `make`, `binutils`, `bc`,
`bison`, `flex`, `perl`, `elfutils`, `pahole` when available) and route-A utilities
(`curl`, `dpkg-source`, `kmod`, `mokutil`, `openssl`, `python3`, `zstd`,
`usbutils`, `iproute2`).

The Spark is currently Ubuntu-kernel based even though Nix is installed. The
script still uses `dpkg-query` to derive the matching Ubuntu kernel source
package for the running `linux-image-*` package.

## Exact Secure Boot / MOK Flow With No Existing Keys

This is not a normal BIOS settings flow. On Ubuntu Secure Boot systems, `mokutil`
registers a pending Machine Owner Key request. Enrollment happens in a blue
shim/MOK Manager pre-boot UI on the next reboot.

1. Enter the Nix shell:

```sh
cd ~/Desktop/exo
nix develop .#dgx-usb-fix
```

2. Check the current state:

```sh
dgx-usb-fix-diagnose
```

3. Create local signing key files:

```sh
dgx-usb-fix-create-mok-key
```

This creates:

```text
/root/MOK.priv
/root/MOK.der
```

It does not enroll anything.

4. Import the public cert into the pending MOK enrollment queue:

```sh
sudo mokutil --import /root/MOK.der
```

`mokutil` asks for a one-time enrollment password. This password is used only in
the pre-boot MOK Manager UI.

5. Reboot:

```sh
sudo reboot
```

6. In the MOK Manager pre-boot UI:

```text
Enroll MOK
Continue
Yes
Enter the one-time password from mokutil --import
Reboot
```

7. After Linux boots again, verify the key is enrolled:

```sh
sudo mokutil --test-key /root/MOK.der
mokutil --sb-state
```

Expected: Secure Boot remains enabled, and the cert is reported as already
enrolled or accepted.

## Build, Sign, Install, And Load

After MOK enrollment:

```sh
cd ~/Desktop/exo
nix develop .#dgx-usb-fix
dgx-usb-fix-install
```

The script:

1. Derives the matching source package from `linux-image-$(uname -r)`.
2. Downloads the matching source tarball, diff, and `.dsc` from Launchpad.
3. Patches `drivers/net/usb/cdc_ncm.c`.
4. Builds external module `cdc_ncm.ko` against `/lib/modules/$(uname -r)/build`.
5. Signs the module with `/root/MOK.priv` and `/root/MOK.der` if Secure Boot is enabled.
6. Installs it to `/lib/modules/$(uname -r)/updates/dgx-usb-fix/cdc_ncm.ko`.
7. Runs `depmod`.
8. Reloads `cdc_ncm` unless `--skip-load` was passed.

No second machine reboot is required if `cdc_ncm` unloads and reloads cleanly.
After module reload, unplug and replug the USB-C cable to make the Apple device
bind again. If `cdc_ncm` cannot be unloaded because another device is using it,
install with `--skip-load` and reboot once after installation so the kernel loads
the patched module on boot.

## Validation

Run:

```sh
./dgx-usb-fix/diagnose.sh
modinfo -n cdc_ncm
modinfo cdc_ncm | grep -E 'dgx_usb_fix|v05ACp1905'
lsusb -t
ip -br link
journalctl -k --no-pager | grep -E 'cdc_ncm|05ac|1905|bind'
```

Expected:

- `modinfo -n cdc_ncm` resolves to
  `/lib/modules/$(uname -r)/updates/dgx-usb-fix/cdc_ncm.ko`.
- `modinfo cdc_ncm` contains `dgx_usb_fix: apple-05ac-1905-cdc-ncm`.
- `modinfo cdc_ncm` contains `v05ACp1905` aliases for interfaces `00` and `02`.
- No fresh `bind() failure` appears for `05ac:1905` after replug.
- A Linux network interface appears for the Mac USB-C link.

## Network Test Shape

Once the kernel creates a netdev, prefer IPv6 link-local tests first:

```sh
dgx-usb-fix-configure-link-local
ip -6 addr
ssh e2@e2 "ifconfig en5"
ping -6 -c 3 'fe80::<mac-suffix>%<spark-usb-iface>'
ssh e2@e2 "ping6 -c 3 fe80::<spark-suffix>%en5"
```

On Spark, NetworkManager's default auto-created wired profiles may withdraw
IPv6 link-local addresses after DHCP/RA timeout. `dgx-usb-fix-configure-link-local`
changes the Apple `05ac:1905` `cdc_ncm` profiles to:

```text
ipv4.method disabled
ipv6.method link-local
ipv6.addr-gen-mode stable-privacy
```

Expected current pairing:

```text
Spark enxd2dcb4ccf72d <-> Mac en5
Spark enxd2dcb4ccf70d <-> Mac anpi0
```

Static IPv4 can be used later for final validation if needed:

- Spark USB netdev: `192.168.254.2/30`
- Mac `en5`: `192.168.254.1/30`

## Kernel Updates

Any kernel update changes `uname -r`, `vermagic`, and the module install path.
Rerun `build-and-install.sh` after booting a new kernel. The MOK key enrollment
does not need to be repeated if `/root/MOK.priv` and `/root/MOK.der` are still
the enrolled key pair.

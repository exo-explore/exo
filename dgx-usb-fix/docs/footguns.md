# Footguns And Shortcuts

Last updated: 2026-04-27.

## MOK Is Not A Normal BIOS Setting

The local signing cert is enrolled through shim's MOK Manager, a pre-boot UI
triggered by `mokutil --import`. It may look firmware-like, but the flow is not
"go into BIOS and toggle a setting." The normal sequence is:

```text
create key files -> mokutil --import cert -> reboot -> MOK Manager -> enroll -> reboot
```

Do not hide this inside a build script. It is a boot-trust change.

## The Key Script Does Not Enroll Anything

`create-mok-key.sh` creates only:

```text
/root/MOK.priv
/root/MOK.der
```

It does not run `mokutil --import`, does not reboot, and does not enroll a key.

## Preserve Nix PATH Through sudo

The build dependencies come from `nix develop .#dgx-usb-fix`, not apt. A plain
`sudo ./dgx-usb-fix/build-and-install.sh` may lose the Nix PATH and fail to find
`dpkg-source`, `gcc`, `make`, `mokutil`, or other tools.

Use:

```sh
sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/build-and-install.sh
```

## Spark Is Ubuntu-Kernel Based In Current Observations

Even if the workstation is NixOS and both remotes have Nix installed, the Spark
currently reports Ubuntu-style kernel packages:

```text
linux-image-6.17.0-1014-nvidia 6.17.0-1014.14
source: linux-signed-nvidia-6.17 6.17.0-1014.14
```

The script intentionally uses `dpkg-query` to find the matching kernel source.
Nix supplies build tools; Ubuntu package metadata supplies the exact source
identity for the running kernel.

## No Second Reboot Unless Module Reload Fails

With no enrolled MOK key, one reboot is required for MOK enrollment. After that,
`build-and-install.sh` can build, sign, install, and reload the module in the
running system.

A second reboot is only a fallback if `cdc_ncm` cannot be unloaded because
another device is using it. In that case:

```sh
sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/build-and-install.sh --skip-load
sudo reboot
```

After reboot, `depmod` should select the installed `updates/dgx-usb-fix` module.

## Userspace Claims Can Block Kernel Bind

If `dgxusbd` or another userspace USB process has claimed interfaces, `cdc_ncm`
may log `failed to claim data intf` and bind failure even after the module is
patched. Kill route-B test processes before route-A bind tests:

```sh
pkill -x dgxusbd || true
```

Then unplug/replug the USB-C cable.

## Do Not Confuse Interfaces

`enP7s7` on Spark is Realtek PCI Ethernet via `r8127`. It is not the USB-C Mac
link. Validate route A with `lsusb -t`, `modinfo`, and kernel logs before
looking for a netdev.

## Kernel Updates Require Rebuild

The module is built for one `uname -r`. If Spark boots a new kernel, rerun
`build-and-install.sh`. The enrolled MOK can be reused as long as the same
`/root/MOK.priv` and `/root/MOK.der` remain available.

## Do Not Commit Secrets

Never commit generated MOK private keys, sudo passwords, SSH keys, or host
private credentials. The repo should contain scripts and docs only.

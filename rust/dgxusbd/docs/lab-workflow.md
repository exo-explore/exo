# Lab Workflow

Last updated: 2026-04-27.

## Machines

Spark / DGX:

- SSH: `ssh jensen@gx10-a174`
- Repo: `~/Desktop/exo`
- Branch: `andrei/dgx-mac-usb-fix`
- Nix is installed.
- Primary runtime/test host for `dgxusbd`.

Mac mini:

- SSH: `ssh e2@e2`
- Used for observing macOS interface state and USB/Thunderbolt status.
- Nix is installed.

Local workstation:

- Repo: `/home/royalguard/Desktop/exo-all/networking-related/exo-dgx-mac-usb-fix`
- Branch: `andrei/dgx-mac-usb-fix`
- Main editing location.

Do not commit plaintext sudo passwords or other secrets. Use the handoff/conversation for credentials when needed.

## Physical Connection

Goal: direct USB-C-looking cable to USB-C-looking cable between Mac mini and Spark. Thunderbolt is acceptable physically because it uses USB-C connectors, but the currently observed Spark path is USB 3.x/xHCI, not Thunderbolt networking.

Current observed state:

- Spark enumerates the Mac as USB device `05ac:1905`.
- Spark does not currently create a Linux netdev for this USB link because stock `cdc_ncm` fails to bind.
- Mac Thunderbolt Bridge is not the active path in the observed setup.

## Local Development Loop

Before code changes:

```sh
git status --short
nix develop -c cargo check -p dgxusbd
```

Usual local verification after code changes:

```sh
nix develop -c cargo fmt --check -p dgxusbd
nix develop -c cargo clippy -p dgxusbd --all-targets
nix develop -c cargo test -p dgxusbd
nix develop -c cargo check -p dgxusbd
```

Adjust exact commands if workspace lints or the flake wiring require it.

## Commit / Push / Pull / Test Loop

1. Implement locally in the branch.
2. Run local Rust checks.
3. Commit locally.
4. Push to `origin andrei/dgx-mac-usb-fix`.
5. On Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && git pull --ff-only"
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo check -p dgxusbd"
```

6. For hardware tests on Spark, prefer an explicit, logged command. Example shape once a real CLI exists:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && RUST_LOG=info sudo -E nix run .#dgxusbd -- --help"
```

Some operations will require root or capabilities:

- claiming USB interfaces if permission rules are not installed
- creating/configuring TAP devices
- changing routes or IP addresses

## Useful Observation Commands

Spark:

```sh
ip -br link
ip -br addr
lsusb
lsusb -t
sudo lsusb -v -d 05ac:1905
journalctl -k --no-pager | grep -E 'cdc_ncm|05ac|1905|bind'
```

Mac:

```sh
ifconfig
networksetup -listallhardwareports
networksetup -listnetworkserviceorder
system_profiler SPThunderboltDataType
system_profiler SPUSBDataType
```

## Interrupt Safety

Avoid commands that unload networking modules, reset interfaces, or alter routes unless the current iteration explicitly calls for it. SSH access is over Wi-Fi/Tailscale-style paths today, but still treat network changes as potentially disruptive.


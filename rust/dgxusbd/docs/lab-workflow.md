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
- During the successful bridge ping test, macOS hardware port `Ethernet Adapter (en5)` was the usable peer interface.

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

## Current Bridge Test Commands

USB endpoint smoke on Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- usb-smoke --read-timeout-ms 250"
```

TAP smoke on Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- tap-smoke --name dgxusb0 --mtu 1500"
```

Short bridge smoke on Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- bridge --tap-name dgxusb0 --mtu 1500 --duration-seconds 3 --usb-timeout-ms 100 --usb-read-timeout-ms 1 --tap-budget-frames 32 --usb-budget-ntbs 8"
```

Temporary static-IP ping shape:

1. Run bridge long enough for testing on Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- bridge --tap-name dgxusb0 --mtu 1500 --duration-seconds 180"
```

2. Add `192.168.254.2/30` to Spark `dgxusb0`.
3. Add temporary `192.168.254.1/30` to Mac `en5`.
4. Ping `192.168.254.2` from the Mac.
5. Remove the Mac alias and let the bridge exit so Spark `dgxusb0` disappears.

The successful test used exactly that shape and produced 3/3 ICMP replies.

## Throughput Testing

Use iperf3 only after the bridge is running and both sides have temporary static IPs on the USB-C link. The known-good addressing shape is:

- Spark TAP `dgxusb0`: `192.168.254.2/30`
- Mac peer interface `en5`: `192.168.254.1/30`

On the Mac, use `/opt/homebrew/bin/iperf3` in noninteractive SSH sessions unless the PATH has been adjusted. If port 5201 is already occupied, use a matched `-p <port>` on both server and client.

TCP from Mac to Spark:

```sh
ssh jensen@gx10-a174 "iperf3 -s -1"
ssh e2@e2 "/opt/homebrew/bin/iperf3 -c 192.168.254.2"
```

UDP from Mac to Spark, uncapped by iperf's target bitrate:

```sh
ssh jensen@gx10-a174 "iperf3 -s -1"
ssh e2@e2 "/opt/homebrew/bin/iperf3 -c 192.168.254.2 -b 0 -u"
```

Reverse direction can be tested by starting the server on the Mac and the client on the Spark:

```sh
ssh e2@e2 "/opt/homebrew/bin/iperf3 -s -1"
ssh jensen@gx10-a174 "iperf3 -c 192.168.254.1"
ssh e2@e2 "/opt/homebrew/bin/iperf3 -s -1"
ssh jensen@gx10-a174 "iperf3 -c 192.168.254.1 -b 0 -u"
```

For Spark-to-Mac UDP, also run capped tests such as `-b 50M -u` and `-b 100M -u`. On Linux, `iperf3 -u -b 0` can inject far more traffic than the current userspace bridge can drain, so heavy loss in that specific test is an overload datapoint rather than a clean throughput ceiling.

Expected interpretation for the current MVP: successful packet movement and low malformed-NTB counts matter first. The bridge now has bounded per-loop TAP and USB budgets, but it still uses one synchronous loop, sends one Ethernet frame per NTB, allocates a fresh NTB for each TAP frame, and does no batching or deeper USB queueing. Asymmetric and lower-than-link-rate throughput is expected until those pieces change.

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

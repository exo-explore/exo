# Current Status

Last updated: 2026-04-27.

## Repositories

Local repo:

- Path: `/home/royalguard/Desktop/exo-all/networking-related/exo-dgx-mac-usb-fix`
- Branch: `andrei/dgx-mac-usb-fix`
- HEAD before docs were added: `189d88ea`
- Initial worktree state: clean

Spark repo:

- Host: `jensen@gx10-a174`
- Path: `~/Desktop/exo`
- Branch: `andrei/dgx-mac-usb-fix`
- HEAD when checked: `189d88ea`

## Rust Stub

Initial files before the implementation pass:

- `rust/dgxusbd/Cargo.toml`
- `rust/dgxusbd/src/main.rs`
- `rust/dgxusbd/src/lib.rs`

Current binary behavior: provides a thin CLI wrapper around library logic.
Implemented subcommands:

- `list`: list candidate USB devices, currently defaulting to Apple `05ac:1905`.
- `probe`: open the candidate device, inspect descriptors, detect CDC-NCM pairs, and optionally claim interfaces.
- `usb-smoke`: claim one NCM pair, run CDC-NCM class setup, select the data altsetting, open bulk endpoints, and do one timed IN read.
- `tap-smoke`: create a Linux TAP interface through `tun-rs`.
- `bridge`: run a conservative one-pair CDC-NCM-to-TAP bridge.

`dgxusbd` is already registered as a workspace member and flake app/package.

## Build Checks Run

Local:

```sh
nix develop -c cargo check -p dgxusbd
```

Result: success.

Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo check -p dgxusbd"
```

Result: success.

Spark flake app smoke run:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix run .#dgxusbd -- list"
```

Result: success. It saw `05ac:1905` at SuperSpeedPlus with four interfaces.

Spark descriptor probe:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- probe"
```

Result: success under sudo. Non-root failed with USB open `errno 13`, as expected.

Spark claim dry-run:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- probe --claim"
```

Result: success. Userspace claimed interfaces 0, 1, 2, and 3 without requiring a kernel module patch.

Spark USB endpoint smoke:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- usb-smoke --read-timeout-ms 250"
```

Result: success. Pair 0/1 was selected, data altsetting 1 was set, bulk endpoints `0x81` and `0x01` opened with 1024-byte max packets, and a timed IN read returned 140 bytes.

Spark TAP smoke:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- tap-smoke --name dgxusb0 --mtu 1500"
```

Result: success. `dgxusb0` was created and removed when the process exited.

Spark bridge smoke:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && sudo -E nix run .#dgxusbd -- bridge --tap-name dgxusb0 --mtu 1500 --duration-seconds 3 --usb-timeout-ms 100"
```

Result after RX alignment fix: success. Counters showed Spark-to-Mac and Mac-to-Spark frame movement with zero malformed NTBs:

```text
tap_rx=17 tap_drop=0 usb_tx_ntb=17 usb_rx_ntb=21 usb_timeout=23 usb_rx_frames=21 tap_tx=21 malformed_ntb=0
```

Temporary static-IP ping:

- Spark bridge/TAP: `dgxusb0` with `192.168.254.2/30`
- Mac interface: `en5` with temporary `192.168.254.1/30`
- Result: `3 packets transmitted, 3 packets received, 0.0% packet loss`
- Final bridge counters for that run:

```text
tap_rx=58 tap_drop=0 usb_tx_ntb=58 usb_rx_ntb=102 usb_timeout=362 usb_rx_frames=103 tap_tx=103 malformed_ntb=0
```

The temporary Mac IP alias was removed after the ping. The Spark TAP was removed when the bridge process exited.

## Hardware Findings

Spark:

- Hostname: `gx10-a174`
- Kernel: `Linux 6.17.0-1014-nvidia aarch64`
- Secure Boot: enabled
- No `/sys/class/udc` was present, so Spark does not currently look able to act as a Linux USB gadget device.
- No `/sys/bus/thunderbolt/devices` was present, and no Thunderbolt bus was observed on Spark.
- `enP7s7` is Realtek PCI Ethernet via `r8127`, not the USB-C Mac link.

Mac:

- SSH target: `e2@e2`
- OS when checked: Darwin 25.4.0
- macOS reported no Thunderbolt device connected on its Thunderbolt/USB4 buses during the investigation.

USB-C link:

- Spark sees the Mac as `05ac:1905 Apple, Inc. Mac`.
- `lsusb -t` showed the device on a 10000M USB path with all four interfaces unbound after `cdc_ncm` bind failure.
- `dgxusbd list` reported the link at `SuperPlus`.
- `dgxusbd probe` detected two CDC-NCM pairs:
  - control interface 0, data interface 1, IN `0x81`, OUT `0x01`, MAC string `D2DCB4CCF72D`
  - control interface 2, data interface 3, IN `0x82`, OUT `0x02`, MAC string `D2DCB4CCF70D`
- Both NCM control interfaces have no status/interrupt endpoint, which explains why the generic Linux `cdc_ncm` path rejects the device.
- Kernel log showed:

```text
cdc_ncm 4-1:1.0: bind() failure
cdc_ncm 4-1:1.2: bind() failure
```

The kernel-driver workaround script in `tmp/spark` patches `cdc_ncm` to add a device-specific `05ac:1905` quirk. That remains the cleaner long-term fix if shipped by NVIDIA/Ubuntu, but `dgxusbd` is exploring a no-kernel-module userspace path.

## Implementation Checkpoints

Iteration 1 commit:

- `df1fa775 Add dgxusbd USB probe CLI`

Iteration 1 status:

- Implemented and pushed.
- Pulled and tested on Spark.
- Hardware probe confirmed the prior descriptor hypothesis.
- Userspace can claim the unbound NCM interfaces.

Iteration 2-4 commits:

- `ed1cfe49 Add dgxusbd NCM bridge MVP`
- `d2a87e6b Accept unaligned RX datagrams`

Iteration 2-4 status:

- Implemented CDC-NCM NTB16 parser/builder with unit tests.
- Implemented NCM class setup based on Linux `cdc_ncm` behavior:
  - `GET_NTB_PARAMETERS`
  - `SET_CRC_MODE`
  - `SET_NTB_FORMAT`
  - `SET_NTB_INPUT_SIZE`
  - `GET/SET_MAX_DATAGRAM_SIZE`
  - `SET_ETHERNET_PACKET_FILTER`
- Implemented TAP creation through `tun-rs`.
- Implemented a one-pair bridge defaulting to control interface 0, data interface 1, IN `0x81`, OUT `0x01`.
- Hardware bridge successfully carried ICMP over the USB-C cable between macOS `en5` and Spark `dgxusb0`.

Important bug found and fixed:

- The first bridge run rejected all Mac-to-Spark NTBs because the parser required DPE datagram indexes to be 4-byte aligned.
- Live Mac NTBs used datagram index 30. This is plausible because NCM payload alignment is about payload placement, not necessarily DPE datagram-index alignment.
- RX index alignment validation was relaxed in `d2a87e6b`; malformed count then dropped to zero.

Next proposed iteration:

- Add operational polish around the working MVP: clearer command docs, optional IP assignment helper, pcap/debug dump, reconnect handling, and bridge shutdown/cleanup behavior.

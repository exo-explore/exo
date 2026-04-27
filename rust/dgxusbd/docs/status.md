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

Iteration 3a/3b in the conversation, Iteration 5 in `implementation-plan.md`:

- `58395120 Refactor dgxusbd CLI options`
- `436c521c Improve dgxusbd bridge scheduling`

Status:

- Implemented `clap(flatten)` option structs and a `UsbId` parser.
- Added separate USB read/write timeout configuration.
- Added bounded per-loop TAP-to-USB and USB-to-TAP budgets so one direction does not drain indefinitely before the other direction is checked.
- Made bridge mode require successful CDC-NCM setup requests. `usb-smoke` still reports setup failures without making every probe fatal.
- Reworked NTB16 builder sizing to use checked accumulation and reject oversized output before allocating the final buffer.
- Removed the custom `U16_MAX_USIZE` alias; NTB16 limits now use the standard `u16::MAX`.
- Added repeatable iperf3 commands and recorded a baseline.

Local checks after this pass:

```sh
nix develop -c cargo fmt -p dgxusbd --check
nix develop -c cargo check -p dgxusbd
nix develop -c cargo test -p dgxusbd
nix develop -c cargo clippy -p dgxusbd --all-targets
```

Result: success. Tests: 9 passed. Clippy still emits broad workspace warning noise but exits 0.

Spark checks after pull:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo check -p dgxusbd"
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo test -p dgxusbd"
```

Result: success. Tests: 9 passed.

Hardware checks after this pass:

- `usb-smoke` still selected pair 0/1, selected data altsetting 1, opened bulk endpoints `0x81` and `0x01`, completed all NCM setup requests, and completed timed NTB reads. The baseline smoke read 140 bytes; the final pulled-commit smoke read 232 bytes.
- Short bridge smoke with scheduling flags moved frames both directions with zero malformed NTBs:

```text
tap_rx=18 tap_drop=0 usb_tx_ntb=18 usb_rx_ntb=16 usb_timeout=1780 usb_rx_frames=16 tap_tx=16 malformed_ntb=0
```

- Temporary static-IP ping from Mac to Spark: 3/3 replies, 0% loss, about 3.4 ms average.
- Temporary static-IP ping from Spark to Mac: 3/3 replies, 0% loss, about 2.0 ms average.
- Longer bridge run covering TCP and UDP iperf traffic had zero malformed NTBs:

```text
tap_rx=140625 tap_drop=0 usb_tx_ntb=140625 usb_rx_ntb=101507 usb_timeout=111006 usb_rx_frames=804611 tap_tx=804611 malformed_ntb=0
```

Throughput baseline:

| Direction | Test | Receiver result |
| --- | --- | --- |
| Mac -> Spark | TCP, 5s | 387 Mbit/s |
| Mac -> Spark | UDP `-b 0`, 5s | 1.41 Gbit/s, 0.47% loss |
| Spark -> Mac | TCP, 5s | 207 Mbit/s, client reported 58 retransmits |
| Spark -> Mac | UDP `-b 0`, 5s | 34.9 Mbit/s received; sender attempted 14.5 Gbit/s and dropped almost everything |
| Spark -> Mac | UDP `-b 100M`, 5s | 55.6 Mbit/s, 39% loss |
| Spark -> Mac | UDP `-b 50M`, 5s | 48.9 Mbit/s, 0% loss |

Interpretation:

- The bridge is viable and no longer just a packet-movement proof.
- Low/asymmetric throughput is still expected for this userspace MVP because it uses one synchronous loop, sends one Ethernet frame per NTB, allocates per TAP frame, and does not batch or pipeline USB transfers.
- Spark-to-Mac `iperf3 -u -b 0` is not a useful capacity test. It is an overload test where Linux injects traffic far faster than the current bridge can drain it.

Review response:

- The timeout-driven fairness concern is valid. The bridge now avoids unlimited TAP draining, but every full TAP budget is still followed by a USB-IN read attempt. With the default 1 ms read timeout, one-way Spark-to-Mac traffic can be shaped by idle waits for traffic that is not arriving.
- The `--max-events` concern is valid. The current loop can overshoot by up to a TAP/USB budget because it checks the limit only at the top of the outer loop.
- The USB timeout help text is stale. `--usb-timeout-ms` is now the write timeout in bridge mode; read timeout is controlled separately by `--usb-read-timeout-ms`.
- The NTB builder preflight needs broader tests before batching work grows: multi-frame alignment padding and too-large aggregate output should be covered explicitly.
- Babblerd has the better runtime shape for a hot dataplane: a dedicated dataplane object/thread, readiness-driven drains, precompiled hot-path state, and low-churn storage. `dgxusbd` should borrow that shape, but not babblerd's one-packet-per-datagram behavior because CDC-NCM throughput depends heavily on NTB batching.

Previous proposal for user-facing Iteration 4, now implemented:

- Treat Iteration 4 as dataplane runtime and throughput correctness, not generic operational polish.
- Start with the small review fixes: exact `--max-events`, write-timeout naming/help, and the missing NTB builder tests.
- Move forwarding into a dataplane module/object with explicit start, stop, and counters.
- Replace timeout fairness with readiness or real concurrency. If `nusb` endpoints cannot be readiness-polled, use independent TAP-to-USB and USB-to-TAP workers with blocking endpoint operations.
- Add reusable buffers and multi-frame NTB batching before deeper USB pipelining.
- Rerun the same iperf TCP/UDP matrix and compare against the baseline above.

Deferred after user-facing Iteration 4:

- Reconnect handling, pair-selection polish, optional pair 2/3 support, pcap/debug dump, IP assignment helpers, and general shutdown/cleanup ergonomics.

## User-Facing Iteration 4: Dataplane Runtime

Commits:

- `435a8c00 Refactor dgxusbd bridge dataplane`
- `34fb3105 Enable multi-queue TAP for dgxusbd dataplane`
- `365dd8ed Share single TAP handle in dgxusbd dataplane`
- `8e98295e Queue USB OUT transfers in dgxusbd dataplane`

Status:

- Implemented the review fixes from the prior iteration:
  - added exact event-budget accounting for the event counters
  - renamed the bridge write timeout to `--usb-write-timeout-ms` while keeping `--usb-timeout-ms` as an alias
  - added NTB builder tests for multi-frame padding, aggregate-size rejection, and max datagram count
  - kept NTB16 limits on standard `u16::MAX`
- Moved bridge forwarding into `src/dataplane.rs`.
- Split forwarding into two OS workers:
  - TAP-to-USB uses `mio::Poll` for TAP readability.
  - USB-to-TAP keeps multiple `nusb` bulk-IN transfers queued.
- Added multi-frame NTB batching for TAP-to-USB, bounded by negotiated max NTB size and datagram count.
- Added queued `nusb` bulk-OUT transfers for TAP-to-USB. This is still architecturally desirable, but it did not solve the remaining Spark-to-Mac bottleneck by itself.
- Tried Linux TAP multi-queue, then backed it out for the bridge path. Cloning a multi-queue TAP created separate queues; because the bridge did not drain every queue, IPv6 ping stopped working. The bridge now shares one TAP handle between workers instead.

Checks:

```sh
nix develop -c cargo fmt -p dgxusbd --check
nix develop -c cargo check -p dgxusbd
nix develop -c cargo test -p dgxusbd
nix develop -c cargo clippy -p dgxusbd --all-targets
```

Result: success locally. Tests: 14 passed. Clippy still exits 0 with broad warning noise.

Spark after pull:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo check -p dgxusbd"
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo test -p dgxusbd"
```

Result: success. Tests: 14 passed.

Hardware test policy update:

- Current lab throughput tests should use IPv6 link-local addresses scoped to the USB interfaces.
- Do not add temporary IPv4 addresses until the end-state validation pass needs them.
- Example observed addresses during the final runs:
  - Spark: `fe80::8c02:56ff:feef:8a19%dgxusb0`, later `fe80::9c75:5cff:fe02:4dae%dgxusb0`
  - Mac: `fe80::8a2:83dc:50cd:d9a%en5`

Link-local ping after the shared-TAP fix:

- Mac to Spark: 3/3 replies, 0% loss, about 2.1 ms average.
- Spark to Mac: 3/3 replies, 0% loss, about 1.5 ms average.

Throughput observations over IPv6 link-local:

| Code state | Direction | Test | Receiver result |
| --- | --- | --- | --- |
| `365dd8ed` shared TAP | Mac -> Spark | TCP, 5s | 4.49 Gbit/s |
| `365dd8ed` shared TAP | Mac -> Spark | UDP `-b 0`, 5s | 5.71 Gbit/s, 0.095% loss |
| `365dd8ed` shared TAP | Spark -> Mac | TCP, 5s | 14.4 Mbit/s and failed iperf result exchange |
| `365dd8ed` shared TAP | Spark -> Mac | UDP `-b 500M`, 5s | about 458-479 Mbit/s by interval, 2-7.5% loss, failed result exchange |
| `8e98295e` queued OUT | Mac -> Spark | TCP, 5s | 3.36 Gbit/s |
| `8e98295e` queued OUT | Spark -> Mac | TCP, 5s | 25.1 Mbit/s and failed iperf result exchange |

Negative tests:

- `--tap-budget-frames 1` did not improve Spark-to-Mac TCP. It often made the link wedging easier to trigger.
- Disabling TAP offloads with `ethtool -K dgxusb0 gro off gso off sg off txvlan off rxvlan off` immediately after TAP creation did not fix Spark-to-Mac TCP. A clean run still received only about 4.4 Mbit/s.

Current interpretation:

- The userspace path is viable for Mac-to-Spark traffic and basic bidirectional ICMP.
- The main remaining correctness/performance blocker is Spark-to-Mac TCP. Spark-to-Mac paced UDP is now clean at 50 Mbit/s and 100 Mbit/s after the Iteration 5 transmit-shape work.
- The blocker is probably not the old timeout scheduler, TAP multi-queue, simple Linux TAP offloads, lack of USB OUT queueing, or a basic malformed-NTB issue.
- Next investigation should focus on TCP burst/backpressure behavior: queued OUT pressure, NTB batch size, explicit pacing, ACK ingress, and packet capture/sampling.

## Iteration 5 Results

Commits:

- `6de9a76c` `Mirror Linux Apple NCM transmit framing`
- `f5a4ceac` `Keep NCM datagram size for smaller TAP MTUs`

Local validation:

- `nix develop -c cargo test -p dgxusbd`: passed, 19 tests.
- `nix develop -c cargo fmt --check -p dgxusbd`: passed.
- `nix develop -c cargo clippy -p dgxusbd --all-targets`: passed with broad existing warning noise.

Implemented behavior:

- Default TX NTBs now mirror Linux's Apple private CDC-NCM path:
  - front NDP, not NDP-at-end
  - 40 datagrams per NDP max
  - full reserved NDP table before data
  - 16 KiB conservative TX NTB size, bumped to 16385 bytes for 1024-byte USB packet short-packet behavior
  - one zero byte appended when needed to avoid ending a short NTB exactly on a USB bulk packet boundary
- NTBs are built directly into reusable `nusb::Buffer` objects.
- TAP batch frame storage is reused instead of allocating a fresh `Vec` for every TAP frame.
- Bridge report now includes byte counters and USB write completions.
- Worker errors are drained after shutdown so late worker failures are not hidden.
- TAP reads no longer consume `--max-events` before packets are committed to USB.
- Smaller TAP MTUs no longer force smaller CDC-NCM max datagram setup; the Mac stalls if asked to set max datagram below 1514.

Hardware observations over IPv6 link-local:

| Commit/config | Direction | Test | Result |
| --- | --- | --- | --- |
| `6de9a76c`, default TX | Spark -> Mac | ping | 3/3 replies, 0% loss |
| `6de9a76c`, default TX | Mac -> Spark | ping | 3/3 replies, 0% loss |
| `6de9a76c`, default TX | Spark -> Mac | UDP `-b 50M`, 5s | 50.0 Mbit/s, 0% loss, clean result exchange |
| `6de9a76c`, default TX | Spark -> Mac | UDP `-b 100M`, 5s | 99.9-100 Mbit/s, 0% loss, clean result exchange |
| `6de9a76c`, default TX | Spark -> Mac | UDP `-b 250M`, 5s | overload; Mac receiver stopped after initial burst, result exchange did not complete cleanly |
| `6de9a76c`, default TX | Spark -> Mac | TCP, 8s | still unstable, single-digit Mbit/s receiver rate and frequent result-exchange failures |
| `6de9a76c`, `--tx-ndp-placement end` | Spark -> Mac | ping and UDP `-b 100M` | works, but no improvement over front NDP |
| `f5a4ceac`, `--mtu 1280` | Spark -> Mac | TCP, 8s | still unstable; smaller TAP MTU did not fix TCP |
| `f5a4ceac`, `--mtu 1280` | Mac -> Spark | TCP, 5s | about 3.53 Gbit/s receiver rate |

Representative bridge counters from a default-TX run that included Spark-to-Mac UDP/TCP tests:

```text
tap_rx=83075 tap_rx_bytes=123774726 tap_drop=0 usb_tx_ntb=11920 usb_tx_bytes=126134132 usb_rx_ntb=1778 usb_rx_bytes=240796 usb_timeout=2276 usb_rx_frames=1793 tap_tx=1793 tap_tx_bytes=187373 tap_tx_drop=0 malformed_ntb=0 usb_write_done=11920
```

Representative bridge counters from a later `--mtu 1280` run that included Mac-to-Spark TCP:

```text
tap_rx=637102 tap_rx_bytes=60742343 tap_drop=0 usb_tx_ntb=165210 usb_tx_bytes=92415182 usb_rx_ntb=154209 usb_rx_bytes=2388581318 usb_timeout=1383 usb_rx_frames=1834601 tap_tx=1834601 tap_tx_bytes=2373261985 tap_tx_drop=0 malformed_ntb=0 usb_write_done=165210
```

Interpretation:

- The Linux Apple transmit-format changes are directionally correct for UDP and no longer look like a malformed-NTB problem.
- The remaining TCP failure is likely burst/backpressure or ACK/data scheduling behavior under TCP, not basic CDC-NCM descriptor parsing or NDP placement.
- Do not make NDP-at-end the default based on current evidence.
- Do not treat `iperf3 -u -b 250M` or `-b 0` as a clean capacity test yet; they are overload tests.

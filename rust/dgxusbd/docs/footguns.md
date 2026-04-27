# Footguns And Shortcuts

Last updated: 2026-04-27.

## Do Not Confuse Interfaces

`enP7s7` on Spark is Realtek PCI Ethernet using `r8127`. It is not the USB-C Mac link.

The USB-C Mac link currently creates no Linux netdev because `cdc_ncm` fails to bind. Look at `lsusb`, `lsusb -t`, and kernel logs rather than `ip link` alone.

## Kernel Driver State

Current `cdc_ncm` failure is useful for userspace work because the interfaces remain unbound:

```text
Driver=[none]
```

If a future kernel or local patch successfully binds `cdc_ncm`, a userspace USB program may fail to claim the interfaces unless it detaches the kernel driver or the driver is prevented from binding. Be careful with detach behavior because it changes system state.

## Secure Boot And Kernel Modules

Spark has Secure Boot enabled. Loading a custom kernel module probably requires signing with an enrolled key. No `/root/MOK.priv` or `/root/MOK.der` files were found during the initial check.

The userspace bridge is specifically meant to avoid this path.

## USB Gadget Mode

No `/sys/class/udc` was present on Spark during investigation. Do not assume Spark can act as a USB Ethernet gadget unless a later kernel/firmware exposes a UDC.

## Thunderbolt

The physical connector may be USB-C/Thunderbolt-looking, but the observed path is not Thunderbolt networking:

- Spark had no Thunderbolt bus devices.
- macOS reported no Thunderbolt device connected.
- Spark saw the Mac as a USB device on xHCI.

Do not spend implementation time on `thunderbolt_net` unless the hardware state changes.

## Running With sudo

Hardware runs will likely need root for USB claim and TAP creation. Avoid accidentally creating root-owned source-tree build artifacts.

Safer pattern when possible:

```sh
nix build .#dgxusbd
sudo -E ./result/bin/dgxusbd ...
```

For quick Spark tests, this also works but can be slower because it may build through Nix:

```sh
RUST_LOG=info sudo -E nix run .#dgxusbd -- ...
```

## Nix Flake Source And New Files

`nix run .#dgxusbd` builds from the flake source, which follows Git state. New Rust source files must be at least staged with `git add` before `nix run` can see them. Otherwise the normal Cargo build may pass locally while the Nix build fails with missing module files.

## Avoid Reinventing The Wheel

Treat handrolled infrastructure as a last resort. Before implementing parsing, formatting, byte-level conversions, CLI value parsing, protocol codecs, packet manipulation, async orchestration, Linux interface management, logging/reporting, test fixtures, or unsafe code, first check whether the standard library, an existing dependency, a kernel/userspace API wrapper, or a well-maintained crate already solves the problem.

Use this process before adding nontrivial custom machinery:

1. Search the current dependency graph and existing source for a helper that already fits the need.
2. Check the upstream API of the crate closest to the problem domain. Prefer domain-specific APIs over generic byte/string manipulation.
3. Search for maintained crates if the current dependency set does not cover the need. Favor crates with recent releases, docs, tests, sensible ownership, and narrow scope.
4. Compare the cost of adding the dependency against the long-term cost of maintaining custom code. For protocol parsing, binary layouts, network interface setup, packet formats, and async IO, the dependency usually wins.
5. If custom code remains necessary, keep it small, isolated, heavily tested, and documented with the reason an off-the-shelf option was not used.

It is acceptable to add a decently maintained dependency when it removes meaningful custom parsing, binary layout handling, unsafe code, platform-specific syscall wrappers, or complex formatting/reporting logic. Do not avoid a dependency just to save a few lines in `Cargo.toml` while growing bespoke code in `src/`.

## Prefer Rusty Structure Over Free Functions

When a module starts collecting loosely related helper functions, repeated option fields, primitive IDs, or handwritten forwarding code, stop and ask whether the code wants a Rust type, trait, derive macro, or focused dependency. Do not wait until the boilerplate is severe. Adopt the Rustier shape as soon as it is viable and makes the next change easier to express.

Use domain newtypes when raw primitives carry protocol meaning, such as USB IDs, interface numbers, endpoint addresses, sequence numbers, MTUs, NTB sizes, and Ethernet frame lengths. Put parsing, formatting, validation, and conversions on those types with `FromStr`, `TryFrom`, `Display`, `From`, or extension traits instead of scattering free functions around the module.

Prefer extension-method APIs over free-function sprawl when the behavior naturally belongs to a type from another crate or a primitive/domain type that cannot directly receive inherent methods. An extension trait is often clearer than passing the same values through helper functions. Crates such as `extend` are acceptable when they make extension traits lighter and more consistent.

Prefer derive and macro support when it removes boring, easy-to-get-wrong boilerplate without hiding important behavior. Examples of useful patterns include `clap` flattened option structs for repeated CLI arguments, `thiserror` for typed library errors, `bon` or similar builder derives for configuration structs with many optional fields, `delegate` or `ambassador` for wrapper forwarding, `itertools` for clearer grouping/chunking, `enum_primitive`, `num_enum`, or similar enum conversion helpers for protocol constants, and display/conversion derives for simple enums and newtypes.

More advanced crates are also on the table when they fit the actual shape of the code. `impl-trait-for-tuples` can remove repeated tuple impls, and `frunk` or other typed-structure crates can be reasonable if they clearly simplify repeated typed plumbing. Reach for these early when they delete real mechanical code and leave the behavior easier to review.

Dependency choice should still be boring and defensible. Check maintenance, documentation, transitive dependencies, platform assumptions, Nix build behavior, and whether the generated code will be obvious to the next implementor. The goal is idiomatic, maintainable Rust. Avoid macros only when they obscure behavior or replace code that was already clearer without them.

## Keep Early Tests Narrow

Initial target should be one NCM pair only:

```text
control interface 0
data interface 1
bulk IN 0x81
bulk OUT 0x01
```

Adding pair 2/3, aggregation, DHCP integration, routing, and reconnect loops should wait until one-pair packet movement is proven.

## NCM RX Alignment

Do not require received DPE datagram indexes to be 4-byte aligned. Live Mac NTBs used datagram index 30. The bridge should still enforce bounds, terminators, signatures, and minimum Ethernet-frame size, but a strict DPE-index alignment check drops valid traffic from this Mac.

## Do Not Store Secrets

Do not commit sudo passwords, SSH keys, host-private credentials, or generated signing keys. Lab credentials should stay in the secure handoff/channel, not in repository docs.

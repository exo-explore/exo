# DGX USB Kernel Fix Notes

Last updated: 2026-04-27.

This folder tracks route A: patch Linux's `cdc_ncm` kernel module so DGX
Spark / GX10 can bind Apple's `05ac:1905` direct USB-C CDC-NCM device.

Start here:

- [quickstart.md](quickstart.md): shortest current handoff, commands, and live interface pairs.
- [status.md](status.md): current route-A repository, hardware, and validation status.
- [motivation.md](motivation.md): why this kernel quirk is the cleanest technical fix.
- [lab-workflow.md](lab-workflow.md): exact build, MOK enrollment, install, and test flow.
- [architecture.md](architecture.md): file layout and runtime shape for this approach.
- [implementation-plan.md](implementation-plan.md): small route-A milestones.
- [protocol-notes.md](protocol-notes.md): USB descriptors and Linux `cdc_ncm` quirk details.
- [footguns.md](footguns.md): Secure Boot, module loading, and lab risks.

Route-A working hypothesis: Spark sees the Mac as USB device `05ac:1905` with
two CDC-NCM functions. Stock `cdc_ncm` matches the generic class entry, expects
a link-status interrupt endpoint, and fails to bind because Apple's control
interfaces have zero endpoints. Mapping interfaces `0` and `2` to Linux's
existing `apple_private_interface_info` should make the in-kernel driver accept
the device and use Linux's mature CDC-NCM transmit path.

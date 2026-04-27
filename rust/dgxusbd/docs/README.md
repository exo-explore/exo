# dgxusbd Task Notes

Last updated: 2026-04-27.

`dgxusbd` is intended to explore a userspace CDC-NCM-to-TAP bridge for direct USB-C networking between an Apple Mac mini and an ASUS/NVIDIA GX10 / DGX Spark. The current repository contains only a Rust stub binary and library; implementation has not started.

Start here:

- [status.md](status.md): current repository, hardware, and build status.
- [motivation.md](motivation.md): why this project exists, what the kernel fix would be, and why userspace is being explored first.
- [lab-workflow.md](lab-workflow.md): hosts, SSH targets, repo paths, and commit/push/pull/test loop.
- [architecture.md](architecture.md): intended library/module boundaries and runtime shape.
- [implementation-plan.md](implementation-plan.md): proposed implementation sequence for review before code changes.
- [protocol-notes.md](protocol-notes.md): observed USB descriptors and CDC-NCM framing notes.
- [footguns.md](footguns.md): known risks, shortcuts, and things to avoid.

Current working hypothesis: Spark sees the Mac as USB device `05ac:1905` with two CDC-NCM functions, but Linux's stock `cdc_ncm` host driver rejects both because Apple's NCM control interfaces omit the interrupt/status endpoint expected by the generic driver path. A userspace bridge can bypass that kernel driver path by claiming the unbound USB interfaces directly and moving Ethernet frames between USB bulk endpoints and a Linux TAP device.

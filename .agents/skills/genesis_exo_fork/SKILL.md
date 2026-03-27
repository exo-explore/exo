---
name: genesis-exo-fork
description: Use when working inside tools/exo on the maintained Genesis EXO fork. Covers Rust, Python, dashboard, placement, and long-context work while preserving Genesis cluster canon and avoiding upstream-style assumptions that conflict with this repo.
---

# Genesis EXO Fork

Use this skill when the current work is inside `tools/exo/`.

## Required context

1. Read `docs/engineering/EXO_AGENT_BRIEF.md`.
2. Load the specific EXO canon docs relevant to the change:
   - `docs/engineering/EXO_CLUSTER_SPEC.md`
   - `docs/engineering/EXO_1M_ARCHITECTURE.md`
   - `docs/engineering/LONG_CONTEXT_1M_SPEC.md`
   - `docs/engineering/LONG_CONTEXT_1M_EARLY_CHECKS.md`

## Local assumptions

- `tools/exo/` is a maintained fork, not a pristine upstream checkout.
- Genesis uses full-model-per-node placement, not tensor sharding.
- mDNS names and canonical config are authoritative; do not hardcode DHCP IPs.
- Placement and API defaults must remain compatible with Genesis scripts, especially `./scripts/exo-cluster.sh`.

## Editing guidance

- Check how a change affects:
  - Rust discovery and networking
  - Python runtime and MLX wiring
  - dashboard behavior
  - model placement and cluster startup scripts
- Keep Genesis-specific environment variables and guardrails intact.
- If the change affects long-context behavior, update the corresponding canon docs.

## Verification

- Prefer the maintained Genesis cluster scripts and validation entrypoints over ad hoc commands.
- When a code change alters runtime behavior, state the expected impact on:
  - API reachability
  - placement
  - long-context readiness
  - dashboard visibility

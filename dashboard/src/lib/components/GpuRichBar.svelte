<script lang="ts">
  import { nodeGpuProfile, topologyData } from "$lib/stores/app.svelte";

  interface Props {
    class?: string;
  }

  let { class: className = "" }: Props = $props();

  const profiles = $derived(nodeGpuProfile());
  const topology = $derived(topologyData());

  const totalTflops = $derived(
    Object.values(profiles).reduce((sum, p) => sum + (p?.tflopsFp16 ?? 0), 0),
  );
  const totalBandwidthGbps = $derived(
    Object.values(profiles).reduce(
      (sum, p) => sum + (p?.memoryBandwidthGbps ?? 0),
      0,
    ),
  );
  const totalMemoryBytes = $derived(
    Object.values(topology?.nodes ?? {}).reduce(
      (sum, n) => sum + (n.system_info?.memory ?? 0),
      0,
    ),
  );
  const hasAnyProfile = $derived(Object.keys(profiles).length > 0);

  // ── GPU-rich/poor scoring ────────────────────────────────────────────────
  //
  // We score each cluster against a reference "rich" rig — 8× H100 — and
  // blend the three normalized scores. TFLOPS dominate the blend, memory
  // bandwidth contributes meaningfully, total memory matters least. To honor
  // "if one is insanely high you're rich" we also take the per-dimension
  // best after weight, so a cluster that's outrageous in any single axis
  // still pulls the thumb noticeably right.
  //
  // Anchors:
  //   TFLOPS:    2140  — 8× H100 sparse FP16 (matches HF reference)
  //   Bandwidth: 26_800 GB/s — 8× HBM3 (~3.35 TB/s each)
  //   Memory:    640 GB — 8× 80 GB
  const TFLOPS_ANCHOR = 2140;
  const BANDWIDTH_ANCHOR_GBPS = 26_800;
  const MEMORY_ANCHOR_BYTES = 640 * 1024 * 1024 * 1024;

  // Blend weights (sum to 1). TFLOPS most, memory least.
  const W_TFLOPS = 0.6;
  const W_BANDWIDTH = 0.3;
  const W_MEMORY = 0.1;

  // Per-dimension caps for the "any one insanely high → rich" leg. A
  // dimension that's fully maxed contributes at most this much. We allow
  // FLOPS alone to nearly fill the bar; bandwidth alone gets you to ~80%;
  // memory alone caps at ~50% because lots of unused RAM doesn't make a
  // cluster generally GPU-rich.
  const SINGLE_DIM_CAP_TFLOPS = 0.95;
  const SINGLE_DIM_CAP_BANDWIDTH = 0.8;
  const SINGLE_DIM_CAP_MEMORY = 0.5;

  const fillPercent = $derived.by(() => {
    if (!hasAnyProfile && totalMemoryBytes <= 0) return 0;
    const tflopsScore = Math.min(1, totalTflops / TFLOPS_ANCHOR);
    const bandwidthScore = Math.min(
      1,
      totalBandwidthGbps / BANDWIDTH_ANCHOR_GBPS,
    );
    const memoryScore = Math.min(1, totalMemoryBytes / MEMORY_ANCHOR_BYTES);

    const blend =
      W_TFLOPS * tflopsScore +
      W_BANDWIDTH * bandwidthScore +
      W_MEMORY * memoryScore;

    const singleDim = Math.max(
      SINGLE_DIM_CAP_TFLOPS * tflopsScore,
      SINGLE_DIM_CAP_BANDWIDTH * bandwidthScore,
      SINGLE_DIM_CAP_MEMORY * memoryScore,
    );

    return Math.min(100, Math.max(0, Math.max(blend, singleDim) * 100));
  });

  function formatTflops(value: number): string {
    if (value >= 1000) return `${(value / 1000).toFixed(2)} PFLOPS`;
    return `${value.toFixed(1)} TFLOPS`;
  }

  function formatBandwidth(value: number): string {
    if (value >= 1000) return `${(value / 1000).toFixed(2)} TB/s`;
    return `${value.toFixed(0)} GB/s`;
  }

  function formatMemory(bytes: number): string {
    if (bytes <= 0) return "—";
    if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(2)} TB`;
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(0)} GB`;
    return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  }
</script>

<div class={`gpu-rich-bar ${className}`}>
  <div class="header">
    <span class="poor-label">GPU poor</span>
    <span class="rich-label">GPU rich</span>
  </div>

  <div class="track" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow={fillPercent.toFixed(0)}>
    <div class="gradient"></div>
    {#if hasAnyProfile}
      <div class="thumb" style="left: {fillPercent}%;"></div>
    {/if}
  </div>

  <div class="footer">
    <div class="stat-block">
      <span class="stat-value">{formatTflops(totalTflops)}</span>
      <span class="stat-label">FP16 compute</span>
    </div>
    <div class="stat-block">
      <span class="stat-value">{formatBandwidth(totalBandwidthGbps)}</span>
      <span class="stat-label">Memory bandwidth</span>
    </div>
    <div class="stat-block">
      <span class="stat-value">{formatMemory(totalMemoryBytes)}</span>
      <span class="stat-label">Memory</span>
    </div>
  </div>
</div>

<style>
  .gpu-rich-bar {
    width: 100%;
    max-width: 560px;
    margin: 0 auto;
    padding: 4px 4px 2px;
    background: transparent;
    font-family: "SF Mono", Monaco, monospace;
  }

  .header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.85);
    letter-spacing: 0.06em;
  }

  .poor-label {
    color: rgba(244, 67, 54, 0.9);
  }
  .rich-label {
    color: rgba(74, 222, 128, 0.95);
  }

  .track {
    position: relative;
    height: 8px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.04);
    overflow: visible;
  }

  .gradient {
    position: absolute;
    inset: 0;
    border-radius: 999px;
    background: linear-gradient(
      to right,
      #f44336 0%,
      #ff9800 35%,
      #ffd700 60%,
      #4ade80 100%
    );
    opacity: 0.95;
  }

  .thumb {
    position: absolute;
    top: 50%;
    width: 14px;
    height: 14px;
    background: #ffffff;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    box-shadow:
      0 0 0 2px rgba(0, 0, 0, 0.4),
      0 2px 4px rgba(0, 0, 0, 0.5);
  }

  .footer {
    display: flex;
    justify-content: center;
    margin-top: 8px;
    gap: 18px;
    flex-wrap: wrap;
  }

  .stat-block {
    display: flex;
    flex-direction: row;
    align-items: baseline;
    gap: 6px;
  }

  .stat-value {
    color: rgba(255, 215, 0, 0.95);
    font-size: 13px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .stat-label {
    color: rgba(179, 179, 179, 0.55);
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
</style>

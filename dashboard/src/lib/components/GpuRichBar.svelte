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

<div class={`cluster-stats ${className}`}>
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

<style>
  .cluster-stats {
    width: 100%;
    max-width: 560px;
    margin: 0 auto;
    padding: 4px 4px 2px;
    background: transparent;
    font-family: "SF Mono", Monaco, monospace;
    display: flex;
    justify-content: center;
    gap: 24px;
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

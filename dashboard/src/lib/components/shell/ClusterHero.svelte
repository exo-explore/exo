<script lang="ts">
  import { cluster, type ClusterNode } from "$lib/api/cluster.svelte";

  let snapshot = $derived(cluster.value);
  let nodes = $derived(snapshot.nodes);

  // Layout: place nodes evenly along a horizontal line in the SVG.
  // Center node is the local one if present, otherwise the first.
  function layout(nodes: ClusterNode[]) {
    const cy = 180;
    if (nodes.length === 0) return [];
    const ordered = [...nodes];
    // Put local in the middle if more than one node
    const localIdx = ordered.findIndex((n) => n.isLocal);
    if (localIdx > 0 && ordered.length > 1) {
      const local = ordered.splice(localIdx, 1)[0]!;
      const mid = Math.floor(ordered.length / 2);
      ordered.splice(mid, 0, local);
    }
    const xs = computeXs(ordered.length);
    return ordered.map((n, i) => ({ ...n, cx: xs[i]!, cy, isCenter: i === Math.floor(ordered.length / 2) }));
  }

  function computeXs(n: number): number[] {
    if (n === 1) return [310];
    if (n === 2) return [200, 420];
    if (n === 3) return [140, 310, 480];
    if (n === 4) return [100, 240, 380, 520];
    // fallback: even spacing 80..540
    const step = (540 - 80) / (n - 1);
    return Array.from({ length: n }, (_, i) => 80 + step * i);
  }

  let placed = $derived(layout(nodes));
  let center = $derived(placed.find((n) => n.isCenter) ?? null);
  let nodeCount = $derived(snapshot.nodes.length);

  // Memory bar geometry
  function memBarFill(node: { memoryFraction: number; isCenter: boolean }) {
    const width = node.isCenter ? 72 : 56;
    return Math.max(2, width * node.memoryFraction);
  }
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">Cluster topology</span>
    <span class="card-sublabel">
      {nodeCount} {nodeCount === 1 ? "NODE" : "NODES"}
    </span>
  </div>
  <div class="cluster-hero">
    {#if nodes.length === 0}
      <div class="empty">
        <div class="empty-eyebrow">NO PEERS</div>
        <div class="empty-title">Single-node cluster</div>
        <div class="empty-sub">Connect another Mac on the same network to expand.</div>
      </div>
    {:else}
      <svg viewBox="0 0 620 360" xmlns="http://www.w3.org/2000/svg">
        <!-- Link halos -->
        {#each placed as node, i}
          {#if i > 0}
            {@const prev = placed[i - 1]}
            <path
              class="link-halo"
              d="M{prev?.cx} {prev?.cy} L {node.cx} {node.cy}"
            />
          {/if}
        {/each}
        <!-- Links -->
        {#each placed as node, i}
          {#if i > 0}
            {@const prev = placed[i - 1]}
            <path
              class="link active"
              d="M{prev?.cx} {prev?.cy} L {node.cx} {node.cy}"
            />
          {/if}
        {/each}

        <!-- Nodes -->
        {#each placed as node}
          {@const r = node.isCenter ? 38 : 30}
          {@const ringR = node.isCenter ? 54 : 44}
          {@const memX = node.cx - (node.isCenter ? 36 : 28)}
          {@const memY = node.cy + (node.isCenter ? 50 : 42)}
          {@const memW = node.isCenter ? 72 : 56}
          <g class="node-group">
            <circle class="glow" cx={node.cx} cy={node.cy} r="50" />
            <circle class="ring-bg" cx={node.cx} cy={node.cy} r={ringR} />
            <circle class="core" cx={node.cx} cy={node.cy} r={r} />

            {#if node.isLeader}
              <rect
                x={node.cx - 19}
                y={node.cy - r - 16}
                width="38"
                height="10"
                rx="5"
                fill="var(--ux-accent)"
              />
              <text class="badge" x={node.cx} y={node.cy - r - 9}>LEADER</text>
            {/if}

            <text class="label" x={node.cx} y={node.cy - 3}>
              {node.friendlyName.length > 14 ? node.friendlyName.slice(0, 13) + "…" : node.friendlyName}
            </text>
            <text class="inner-stat" x={node.cx} y={node.cy + 12}>
              {Math.round(node.memoryFraction * 100)}%
            </text>

            <rect class="mem-track" x={memX} y={memY} width={memW} height="4" rx="1.5" />
            <rect
              class="mem-fill"
              x={memX}
              y={memY}
              width={memBarFill({ memoryFraction: node.memoryFraction, isCenter: node.isCenter })}
              height="4"
              rx="1.5"
            />

            <text class="spec" x={node.cx} y={memY + 18}>
              {node.chip.toUpperCase()} · {node.totalMemoryGB > 0 ? `${node.totalMemoryGB.toFixed(0)} GB` : "—"}
            </text>
            {#if node.isLocal}
              <text class="spec accent" x={node.cx} y={memY + 32}>LOCAL</text>
            {:else if node.tempC}
              <text class="spec" x={node.cx} y={memY + 32}>{node.tempC.toFixed(0)}°C</text>
            {/if}
          </g>
        {/each}
      </svg>
    {/if}

    <div class="legend">
      <span class="legend-item"><span class="swatch active"></span>active</span>
      <span class="legend-item"><span class="swatch idle"></span>idle</span>
    </div>
  </div>
</div>

<style>
  .card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border-bottom: 1px solid var(--ux-border);
  }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: -0.005em;
    color: var(--ux-text);
  }
  .card-sublabel {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .cluster-hero {
    position: relative;
    min-height: 380px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
      radial-gradient(ellipse 60% 50% at center, var(--ux-accent-bg) 0%, transparent 70%),
      var(--ux-card);
    border-radius: 0 0 var(--ux-radius) var(--ux-radius);
    overflow: hidden;
  }
  .cluster-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(
      circle at 1px 1px,
      var(--ux-text-faint) 1px,
      transparent 0
    );
    opacity: 0.18;
    background-size: 24px 24px;
    mask-image: radial-gradient(ellipse 70% 60% at center, #000 0%, transparent 100%);
  }
  svg {
    width: 100%;
    height: auto;
    max-width: 620px;
    position: relative;
    z-index: 2;
  }
  .glow {
    fill: var(--ux-accent);
    opacity: 0.18;
    animation: uxNodeGlow 3.2s ease-in-out infinite;
  }
  .ring-bg {
    fill: none;
    stroke: var(--ux-border-strong);
    stroke-width: 1;
    stroke-dasharray: 1 3;
  }
  .core {
    fill: var(--ux-surface-deep);
    stroke: var(--ux-accent);
    stroke-width: 1;
  }
  .link {
    stroke: var(--ux-border-strong);
    stroke-width: 1.2;
    fill: none;
  }
  .link-halo {
    stroke: var(--ux-accent);
    stroke-width: 4;
    fill: none;
    opacity: 0.07;
  }
  .link.active {
    stroke: var(--ux-accent);
    stroke-dasharray: 3 5;
    animation: uxDash 1.8s linear infinite;
    opacity: 0.85;
  }
  .label {
    font-family: var(--ux-mono);
    font-size: 10px;
    font-weight: 500;
    fill: var(--ux-text);
    text-anchor: middle;
    letter-spacing: 0.03em;
  }
  .inner-stat {
    font-family: var(--ux-mono);
    font-size: 9.5px;
    font-weight: 500;
    fill: var(--ux-accent);
    text-anchor: middle;
  }
  .badge {
    font-family: var(--ux-mono);
    font-size: 7px;
    font-weight: 600;
    fill: var(--ux-text-invert);
    text-anchor: middle;
    letter-spacing: 0.08em;
  }
  .mem-track {
    fill: var(--ux-bg-raised);
    stroke: var(--ux-border-strong);
    stroke-width: 0.6;
  }
  .mem-fill {
    fill: var(--ux-accent);
  }
  .spec {
    font-family: var(--ux-mono);
    font-size: 8.5px;
    fill: var(--ux-text-faint);
    text-anchor: middle;
    letter-spacing: 0.04em;
  }
  .spec.accent {
    fill: var(--ux-accent);
  }
  .legend {
    position: absolute;
    bottom: 14px;
    left: 18px;
    display: flex;
    gap: 18px;
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .swatch {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    border: 1px solid var(--ux-border-strong);
  }
  .swatch.active {
    background: var(--ux-accent);
    border-color: var(--ux-accent);
  }
  .swatch.idle {
    background: var(--ux-bg-raised);
  }
  .empty {
    text-align: center;
    padding: 40px;
    z-index: 2;
    position: relative;
  }
  .empty-eyebrow {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.14em;
    margin-bottom: 8px;
  }
  .empty-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--ux-text);
    margin-bottom: 6px;
  }
  .empty-sub {
    font-size: 13px;
    color: var(--ux-text-dim);
  }
</style>

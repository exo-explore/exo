<script lang="ts">
  import type { ClusterNode } from "$lib/api/cluster.svelte";

  interface Props {
    node: ClusterNode;
  }
  let { node }: Props = $props();

  let memPct = $derived(Math.round(node.memoryFraction * 100));
</script>

<div class="card">
  <div class="card-header">
    <div>
      <div class="name">
        {node.friendlyName}
        {#if node.isLocal}<span class="tag local">LOCAL</span>{/if}
        {#if node.isLeader}<span class="tag leader">LEADER</span>{/if}
      </div>
      <div class="sub">
        <span class="mono">{node.shortId}</span>
        · {node.chip}
      </div>
    </div>
    <div class="status">
      <span class="dot"></span>
      <span>ONLINE</span>
    </div>
  </div>

  <div class="grid">
    <div class="stat">
      <div class="lbl">MEMORY</div>
      <div class="val">
        {node.usedMemoryGB.toFixed(1)}
        <span class="unit">/ {node.totalMemoryGB.toFixed(0)} GB</span>
      </div>
      <div class="bar">
        <div class="bar-fill" style="width: {memPct}%"></div>
      </div>
      <div class="foot">{memPct}% used</div>
    </div>
    {#if node.tempC !== undefined}
      <div class="stat">
        <div class="lbl">TEMPERATURE</div>
        <div class="val">
          {node.tempC.toFixed(0)}
          <span class="unit">°C</span>
        </div>
        <div class="foot">
          {node.tempC < 50 ? "cool" : node.tempC < 70 ? "warm" : "hot"}
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 18px 20px;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
  }
  .name {
    font-size: 15px;
    font-weight: 600;
    color: var(--ux-text);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .tag {
    font-family: var(--ux-mono);
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 3px;
    letter-spacing: 0.08em;
    font-weight: 600;
  }
  .tag.local {
    background: var(--ux-blue-bg);
    color: var(--ux-blue);
  }
  .tag.leader {
    background: var(--ux-accent);
    color: var(--ux-text-invert);
  }
  .sub {
    font-size: 11px;
    color: var(--ux-text-faint);
    margin-top: 4px;
  }
  .mono {
    font-family: var(--ux-mono);
  }
  .status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-green);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .status .dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--ux-green);
    box-shadow: 0 0 0 2px var(--ux-green-bg);
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }
  .stat .lbl {
    font-family: var(--ux-mono);
    font-size: 9.5px;
    color: var(--ux-text-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
  }
  .stat .val {
    font-family: var(--ux-mono);
    font-size: 22px;
    font-weight: 500;
    color: var(--ux-text);
    line-height: 1;
  }
  .stat .unit {
    color: var(--ux-text-faint);
    font-size: 12px;
    margin-left: 2px;
    font-weight: 400;
  }
  .bar {
    margin-top: 10px;
    height: 4px;
    background: var(--ux-bg-raised);
    border-radius: 2px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--ux-accent) 0%, var(--ux-accent-strong) 100%);
    border-radius: 2px;
  }
  .foot {
    margin-top: 8px;
    font-size: 11px;
    color: var(--ux-text-faint);
    font-family: var(--ux-mono);
  }
</style>

<script lang="ts">
  import ClusterHero from "$lib/components/shell/ClusterHero.svelte";
  import NodeDetailCard from "$lib/components/shell/NodeDetailCard.svelte";
  import { cluster } from "$lib/api/cluster.svelte";

  let snapshot = $derived(cluster.value);
  let nodeCount = $derived(snapshot.nodes.length);
</script>

<div class="page-header">
  <div>
    <div class="eyebrow">CLUSTER</div>
    <h1>
      {nodeCount}
      {nodeCount === 1 ? "node" : "nodes"} on this cluster.
    </h1>
    <div class="subtitle">
      Topology, health, and per-node detail.
    </div>
  </div>
</div>

<ClusterHero />

<div class="nodes" class:single={nodeCount === 1}>
  {#each snapshot.nodes as node (node.id)}
    <NodeDetailCard {node} />
  {/each}
</div>

<style>
  .page-header {
    margin-bottom: 36px;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ux-text-faint);
    font-size: 10px;
    font-weight: 600;
    font-family: var(--ux-mono);
    margin-bottom: 6px;
  }
  h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ux-text);
  }
  .subtitle {
    color: var(--ux-text-dim);
    font-size: 13px;
    margin-top: 6px;
  }
  .nodes {
    margin-top: 14px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 14px;
  }
  .nodes.single {
    grid-template-columns: 1fr;
  }
</style>

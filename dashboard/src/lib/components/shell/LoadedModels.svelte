<script lang="ts">
  import { cluster } from "$lib/api/cluster.svelte";

  let snapshot = $derived(cluster.value);
  let instances = $derived(snapshot.instances);
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">Loaded models</span>
    <span class="card-sublabel">{instances.length} ACTIVE</span>
  </div>
  <div class="list">
    {#if instances.length === 0}
      <div class="empty">
        No models loaded.
        <a href="#/models" class="empty-link">Browse models →</a>
      </div>
    {:else}
      {#each instances as inst}
        <div class="row">
          <div>
            <div class="name">{inst.modelId}</div>
            <div class="meta">
              {inst.shardCount > 1 ? `${inst.shardCount} shards · ` : ""}{inst.nodes.length}
              {inst.nodes.length === 1 ? "node" : "nodes"}
            </div>
          </div>
          <span class="state active">
            <span class="dot"></span>ACTIVE
          </span>
        </div>
      {/each}
    {/if}
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
    color: var(--ux-text);
  }
  .card-sublabel {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .list {
    padding: 6px 0;
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid var(--ux-border);
  }
  .row:last-child {
    border-bottom: none;
  }
  .name {
    font-family: var(--ux-mono);
    font-size: 12px;
    color: var(--ux-text);
    letter-spacing: -0.01em;
  }
  .meta {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    margin-top: 2px;
  }
  .state {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--ux-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .state.active {
    color: var(--ux-accent);
  }
  .state .dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--ux-accent);
  }
  .empty {
    padding: 30px 18px;
    text-align: center;
    color: var(--ux-text-dim);
    font-size: 13px;
  }
  .empty-link {
    display: block;
    margin-top: 8px;
    color: var(--ux-text);
    text-decoration: underline;
    text-decoration-color: var(--ux-border-strong);
    font-size: 12px;
  }
  .empty-link:hover {
    text-decoration-color: var(--ux-text);
  }
</style>

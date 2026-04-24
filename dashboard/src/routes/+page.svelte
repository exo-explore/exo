<script lang="ts">
  import { serverStats } from "$lib/api/stats.svelte";
  import StatCard from "$lib/components/shell/StatCard.svelte";
  import ClusterHero from "$lib/components/shell/ClusterHero.svelte";
  import LoadedModels from "$lib/components/shell/LoadedModels.svelte";
  import EndpointsCard from "$lib/components/shell/EndpointsCard.svelte";
  import RecentRequests from "$lib/components/shell/RecentRequests.svelte";

  let stats = $derived(serverStats.value);

  function fmt(n: number | undefined | null): string {
    if (n === undefined || n === null) return "—";
    return n.toLocaleString();
  }

  function uptime(seconds: number | undefined): string {
    if (!seconds || !isFinite(seconds)) return "—";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h > 0) return `${h}h ${m}m uptime`;
    return `${m}m uptime`;
  }

  let headline = $derived(
    stats
      ? `Serving ${stats.nodeCount} ${stats.nodeCount === 1 ? "node" : "nodes"}, ${stats.instanceCount} ${stats.instanceCount === 1 ? "model" : "models"} live.`
      : "Connecting to your cluster…"
  );
</script>

<div class="page-header">
  <div>
    <div class="eyebrow">STATUS</div>
    <h1>{headline}</h1>
    <div class="subtitle">
      {#if stats}
        {fmt(stats.totalRequests)} requests served · {uptime(stats.uptimeSeconds)}
      {:else}
        Waiting for the API server to respond.
      {/if}
    </div>
  </div>
  <div class="page-actions">
    <a class="btn" href="#/legacy">Open legacy view</a>
    <a class="btn btn-primary" href="#/models">+ Launch model</a>
  </div>
</div>

<div class="stat-grid">
  <StatCard
    label="Total requests"
    value={fmt(stats?.totalRequests)}
    foot={stats ? `since ${uptime(stats.uptimeSeconds)}` : ""}
  />
  <StatCard
    label="Loaded instances"
    value={fmt(stats?.instanceCount)}
    foot={stats ? `${stats.instanceCount > 0 ? "running" : "no models loaded"}` : ""}
  />
  <StatCard
    label="Active commands"
    value={`${stats?.activeCommands ?? 0}`}
    foot={(stats?.activeCommands ?? 0) > 0 ? "in flight" : "idle · ready for work"}
    accent={(stats?.activeCommands ?? 0) > 0}
  />
</div>

<div class="hero-grid">
  <ClusterHero />
  <LoadedModels />
</div>

<div class="two-col">
  <EndpointsCard />
  <RecentRequests />
</div>

<style>
  .page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 36px;
    gap: 16px;
    flex-wrap: wrap;
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
  .page-actions {
    display: flex;
    gap: 8px;
  }
  .btn {
    font-family: var(--ux-sans);
    font-size: 13px;
    font-weight: 500;
    padding: 8px 14px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: var(--ux-card);
    color: var(--ux-text);
    cursor: pointer;
    transition: background 120ms, border-color 120ms;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
  }
  .btn:hover {
    background: var(--ux-bg-hover);
    border-color: #353535;
  }
  .btn-primary {
    background: var(--ux-text);
    color: var(--ux-bg);
    border-color: var(--ux-text);
    font-weight: 600;
  }
  .btn-primary:hover {
    background: #fff;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 14px;
  }
  .hero-grid {
    display: grid;
    grid-template-columns: 1.7fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  @media (max-width: 900px) {
    .stat-grid,
    .hero-grid,
    .two-col {
      grid-template-columns: 1fr;
    }
  }
</style>

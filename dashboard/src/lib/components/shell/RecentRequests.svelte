<script lang="ts">
  import { browser } from "$app/environment";

  interface TraceItem {
    task_id: string;
    created_at: string;
    file_size: number;
  }

  let traces = $state<TraceItem[]>([]);

  async function fetchTraces() {
    try {
      const res = await fetch("/v1/traces");
      if (!res.ok) return;
      const data = (await res.json()) as { traces: TraceItem[] };
      traces = (data.traces ?? []).slice(0, 6);
    } catch {
      // ignore
    }
  }

  if (browser) {
    fetchTraces();
    setInterval(fetchTraces, 4000);
  }

  function timeOf(iso: string): string {
    try {
      return new Date(iso).toTimeString().slice(0, 8);
    } catch {
      return "—";
    }
  }
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">Recent requests</span>
    <a class="card-sublabel link" href="#/legacy">VIEW ALL →</a>
  </div>
  <div class="list">
    {#if traces.length === 0}
      <div class="empty">No traces yet. Send a request to see activity here.</div>
    {:else}
      {#each traces as t}
        <div class="row">
          <span class="status"></span>
          <span class="time">{timeOf(t.created_at)}</span>
          <span class="task">trace · {t.task_id.slice(0, 10)}</span>
          <span class="size">{(t.file_size / 1024).toFixed(1)} KB</span>
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
    text-decoration: none;
  }
  .card-sublabel.link:hover {
    color: var(--ux-text-dim);
  }
  .list {
    padding: 4px 0;
  }
  .row {
    display: grid;
    grid-template-columns: auto auto 1fr auto;
    align-items: center;
    gap: 12px;
    padding: 11px 18px;
    border-bottom: 1px solid var(--ux-border);
    font-family: var(--ux-mono);
    font-size: 11.5px;
  }
  .row:last-child {
    border-bottom: none;
  }
  .status {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ux-green);
    box-shadow: 0 0 0 2px var(--ux-green-bg);
  }
  .time {
    color: var(--ux-text-faint);
    font-size: 10.5px;
  }
  .task {
    color: var(--ux-text);
  }
  .size {
    color: var(--ux-text-dim);
    text-align: right;
  }
  .empty {
    padding: 30px 18px;
    text-align: center;
    color: var(--ux-text-dim);
    font-size: 13px;
  }
</style>

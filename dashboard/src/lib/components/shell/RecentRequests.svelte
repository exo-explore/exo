<script lang="ts">
  import { browser } from "$app/environment";

  type Status = "pending" | "running" | "complete" | "failed" | "cancelled";

  interface RecentRequest {
    taskId: string;
    modelId: string;
    createdAt: number; // unix seconds
    status: Status;
    outputTokens: number | null;
  }

  let requests = $state<RecentRequest[]>([]);

  async function fetchRecent() {
    try {
      const res = await fetch("/v1/recent-requests?limit=6");
      if (!res.ok) return;
      const data = (await res.json()) as { requests: RecentRequest[] };
      requests = data.requests ?? [];
    } catch {
      // ignore
    }
  }

  if (browser) {
    fetchRecent();
    setInterval(fetchRecent, 3000);
  }

  function relativeTime(seconds: number): string {
    const ms = Date.now() - seconds * 1000;
    if (!isFinite(ms) || ms < 0) return "now";
    const s = Math.round(ms / 1000);
    if (s < 5) return "now";
    if (s < 60) return `${s}s ago`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.round(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.round(h / 24);
    return `${d}d ago`;
  }

  function shortModel(id: string): string {
    return id.split("/").pop() ?? id;
  }
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">Recent requests</span>
    <span class="card-sublabel">{requests.length} SHOWN</span>
  </div>
  <div class="list">
    {#if requests.length === 0}
      <div class="empty">
        No requests yet. Send a chat message to see activity here.
      </div>
    {:else}
      {#each requests as r}
        <div class="row" data-status={r.status}>
          <span class="status"></span>
          <div class="middle">
            <div class="model">{shortModel(r.modelId)}</div>
            <div class="meta">
              <span class="time">{relativeTime(r.createdAt)}</span>
              <span class="sep">·</span>
              <span class="state-text">{r.status}</span>
              <span class="sep">·</span>
              <span class="tid">{r.taskId.slice(0, 8)}</span>
            </div>
          </div>
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
    grid-template-columns: auto 1fr;
    align-items: center;
    gap: 12px;
    padding: 10px 18px;
    border-bottom: 1px solid var(--ux-border);
  }
  .row:last-child {
    border-bottom: none;
  }
  .status {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ux-text-faint);
  }
  .row[data-status="complete"] .status {
    background: var(--ux-green);
    box-shadow: 0 0 0 2px var(--ux-green-bg);
  }
  .row[data-status="running"] .status,
  .row[data-status="pending"] .status {
    background: var(--ux-accent);
    box-shadow: 0 0 0 2px var(--ux-accent-bg);
    animation: uxPulse 1.6s ease-in-out infinite;
  }
  .row[data-status="failed"] .status {
    background: var(--ux-red);
  }
  .row[data-status="cancelled"] .status {
    background: var(--ux-text-faint);
  }
  .middle {
    min-width: 0;
  }
  .model {
    font-family: var(--ux-mono);
    font-size: 11.5px;
    color: var(--ux-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    margin-top: 2px;
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sep {
    opacity: 0.5;
  }
  .state-text {
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .row[data-status="complete"] .state-text {
    color: var(--ux-green);
  }
  .row[data-status="running"] .state-text,
  .row[data-status="pending"] .state-text {
    color: var(--ux-accent);
  }
  .row[data-status="failed"] .state-text {
    color: var(--ux-red);
  }
  .tid {
    color: var(--ux-text-faint);
  }
  .empty {
    padding: 30px 18px;
    text-align: center;
    color: var(--ux-text-dim);
    font-size: 13px;
  }
</style>

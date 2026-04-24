<script lang="ts">
  import { page } from "$app/state";
  import { serverStats } from "$lib/api/stats.svelte";

  const navItems: Array<{ href: string; label: string }> = [
    { href: "#/", label: "Status" },
    { href: "#/chat", label: "Chat" },
    { href: "#/models", label: "Models" },
    { href: "#/cluster", label: "Cluster" },
    { href: "#/integrations", label: "Integrations" },
    { href: "#/benchmark", label: "Benchmark" },
    { href: "#/settings", label: "Settings" },
  ];

  function isActive(href: string) {
    const path = page.url.hash || "#/";
    if (href === "#/") return path === "#/" || path === "";
    return path === href || path.startsWith(href + "/");
  }

  function uptimeDisplay(seconds: number): string {
    if (!seconds || !isFinite(seconds)) return "—";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  }

  let stats = $derived(serverStats.value);
  let nodeLabel = $derived(
    stats ? `${stats.nodeCount} node${stats.nodeCount === 1 ? "" : "s"}` : "—"
  );
  let isRunning = $derived(stats !== null);
  let statusText = $derived(
    isRunning ? `RUNNING · ${uptimeDisplay(stats?.uptimeSeconds ?? 0)}` : "OFFLINE"
  );
</script>

<header class="topbar">
  <a class="brand" href="#/">
    <span class="brand-mark">
      <svg width="12" height="12" viewBox="0 0 12 12">
        <path d="M3 6L9 3M3 6L9 9" stroke="rgba(237,237,237,0.35)" stroke-width="0.6" />
        <circle cx="3" cy="6" r="1.4" fill="#ededed" />
        <circle cx="9" cy="3" r="1.4" fill="#ededed" />
        <circle cx="9" cy="9" r="1.4" fill="var(--ux-accent)" />
      </svg>
    </span>
    <span class="brand-name">exo</span>
    <span class="brand-cluster">{nodeLabel}</span>
  </a>

  <nav class="nav">
    {#each navItems as item}
      <a href={item.href} class:active={isActive(item.href)}>{item.label}</a>
    {/each}
  </nav>

  <div class="topbar-right">
    <span class="status-pill" class:offline={!isRunning}>
      <span class="dot"></span>
      <span>{statusText}</span>
    </span>
    <a href="#/legacy" class="legacy-link" title="Open legacy command-center view">
      legacy →
    </a>
  </div>
</header>

<style>
  .topbar {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    padding: 14px 22px;
    border-bottom: 1px solid var(--ux-border);
    background: var(--ux-bg);
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(12px);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: -0.01em;
    color: var(--ux-text);
    text-decoration: none;
  }
  .brand-name {
    color: var(--ux-text);
  }
  .brand-mark {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: linear-gradient(135deg, #1c1c1c 0%, #0a0a0a 100%);
    border: 1px solid var(--ux-border-strong);
    display: grid;
    place-items: center;
    position: relative;
    overflow: hidden;
  }
  .brand-mark::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(
      circle at 70% 30%,
      rgba(245, 166, 35, 0.35) 0%,
      transparent 60%
    );
  }
  .brand-mark svg {
    position: relative;
    z-index: 1;
  }
  .brand-cluster {
    font-family: var(--ux-mono);
    font-size: 11px;
    color: var(--ux-text-faint);
    padding-left: 10px;
    margin-left: 10px;
    border-left: 1px solid var(--ux-border);
    letter-spacing: 0.02em;
  }
  .nav {
    display: flex;
    justify-content: center;
    gap: 4px;
    flex-wrap: wrap;
  }
  .nav a {
    color: var(--ux-text-dim);
    font-size: 13px;
    font-weight: 500;
    padding: 7px 12px;
    border-radius: var(--ux-radius-sm);
    text-decoration: none;
    transition: background 120ms, color 120ms;
    letter-spacing: -0.005em;
  }
  .nav a:hover {
    color: var(--ux-text);
    background: var(--ux-bg-hover);
  }
  .nav a.active {
    color: var(--ux-text);
    background: var(--ux-bg-raised);
    box-shadow: inset 0 0 0 1px var(--ux-border-strong);
  }
  .topbar-right {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: var(--ux-mono);
    font-size: 11px;
    color: var(--ux-text-dim);
    padding: 5px 10px 5px 8px;
    border: 1px solid var(--ux-border);
    border-radius: 999px;
    background: var(--ux-card);
  }
  .status-pill .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ux-green);
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.15);
    animation: uxPulse 2.4s ease-in-out infinite;
  }
  .status-pill.offline .dot {
    background: var(--ux-text-faint);
    box-shadow: none;
    animation: none;
  }
  .legacy-link {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    text-decoration: none;
    letter-spacing: 0.05em;
    transition: color 120ms;
  }
  .legacy-link:hover {
    color: var(--ux-text-dim);
  }
  @media (max-width: 900px) {
    .topbar {
      grid-template-columns: auto 1fr;
      grid-template-rows: auto auto;
      gap: 8px;
    }
    .nav {
      grid-column: 1 / -1;
      justify-content: flex-start;
      overflow-x: auto;
      flex-wrap: nowrap;
    }
  }
</style>

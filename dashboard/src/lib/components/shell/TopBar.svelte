<script lang="ts">
  import { page } from "$app/state";
  import { serverStats } from "$lib/api/stats.svelte";
  import { theme, type ThemeMode } from "$lib/theme.svelte";

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

  let themeMode = $derived<ThemeMode>(theme.mode);
  let themeTooltip = $derived.by(() => {
    const labels: Record<ThemeMode, string> = {
      light: "Light",
      dark: "Dark",
      solar: "Solar",
      system: "System",
    };
    return `Theme: ${labels[themeMode]} (click to cycle)`;
  });
</script>

<header class="topbar">
  <a class="brand" href="#/">
    <span class="brand-mark">
      <svg width="12" height="12" viewBox="0 0 12 12">
        <path d="M3 6L9 3M3 6L9 9" stroke="currentColor" stroke-opacity="0.35" stroke-width="0.6" />
        <circle cx="3" cy="6" r="1.4" fill="currentColor" />
        <circle cx="9" cy="3" r="1.4" fill="currentColor" />
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
    <button
      class="theme-toggle"
      type="button"
      onclick={() => theme.cycle()}
      title={themeTooltip}
      aria-label={themeTooltip}
    >
      {#if themeMode === "light"}
        <!-- sun -->
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      {:else if themeMode === "dark"}
        <!-- moon -->
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      {:else if themeMode === "solar"}
        <!-- paper / book -->
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 4h12a4 4 0 0 1 4 4v12H8a4 4 0 0 1-4-4V4z" />
          <path d="M4 4v16M8 8h8M8 12h8M8 16h6" />
        </svg>
      {:else}
        <!-- system / display -->
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="4" width="18" height="13" rx="2" />
          <path d="M8 21h8M12 17v4" />
        </svg>
      {/if}
    </button>
    <span class="status-pill" class:offline={!isRunning}>
      <span class="dot"></span>
      <span>{statusText}</span>
    </span>
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
    background: linear-gradient(
      135deg,
      var(--ux-bg-raised) 0%,
      var(--ux-surface-deep) 100%
    );
    border: 1px solid var(--ux-border-strong);
    color: var(--ux-text);
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
      var(--ux-accent-bg-strong) 0%,
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
    box-shadow: 0 0 0 3px var(--ux-green-bg);
    animation: uxPulse 2.4s ease-in-out infinite;
  }
  .status-pill.offline .dot {
    background: var(--ux-text-faint);
    box-shadow: none;
    animation: none;
  }
  .theme-toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: var(--ux-radius-sm);
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    color: var(--ux-text-dim);
    cursor: pointer;
    padding: 0;
    transition: color 120ms, border-color 120ms, background 120ms;
  }
  .theme-toggle:hover {
    color: var(--ux-text);
    border-color: var(--ux-border-strong);
    background: var(--ux-bg-hover);
  }
  .theme-toggle:focus-visible {
    outline: 2px solid var(--ux-accent);
    outline-offset: 1px;
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

<script lang="ts">
  import { browser } from "$app/environment";
  import { serverStats } from "$lib/api/stats.svelte";
  import { cluster } from "$lib/api/cluster.svelte";
  import { onMount } from "svelte";

  const apiUrl = browser
    ? window.location.origin.replace("localhost", "127.0.0.1")
    : "http://127.0.0.1:52415";

  let stats = $derived(serverStats.value);
  let clusterSnap = $derived(cluster.value);

  let onboardingDone = $state<boolean | null>(null);
  let copied = $state<string | null>(null);

  onMount(async () => {
    try {
      const res = await fetch("/onboarding");
      if (res.ok) {
        const data = (await res.json()) as { completed?: boolean };
        onboardingDone = data.completed ?? null;
      }
    } catch {
      /* ignore */
    }
  });

  async function copy(value: string, key: string) {
    try {
      await navigator.clipboard.writeText(value);
      copied = key;
      setTimeout(() => {
        if (copied === key) copied = null;
      }, 1500);
    } catch {
      /* ignore */
    }
  }

  async function resetOnboarding() {
    onboardingDone = null;
    if (browser) {
      // Reload to trigger first-launch flow if it exists.
      window.location.hash = "#/";
      window.location.reload();
    }
  }

  function fmtMem(gb: number): string {
    if (gb >= 1) return `${gb.toFixed(0)} GB`;
    return `${(gb * 1024).toFixed(0)} MB`;
  }

  let totalMemoryGb = $derived(
    clusterSnap.nodes.reduce((sum, n) => sum + n.totalMemoryGB, 0),
  );
  let usedMemoryGb = $derived(
    clusterSnap.nodes.reduce((sum, n) => sum + n.usedMemoryGB, 0),
  );

  const endpoints = [
    {
      key: "openai",
      label: "OpenAI-compatible",
      hint: "Drop-in replacement for the OpenAI SDK base_url",
      get url() {
        return `${apiUrl}/v1`;
      },
    },
    {
      key: "claude",
      label: "Anthropic-compatible",
      hint: "Set ANTHROPIC_BASE_URL to this value",
      get url() {
        return apiUrl;
      },
    },
    {
      key: "ollama",
      label: "Ollama-compatible",
      hint: "Set OLLAMA_HOST to this value",
      get url() {
        return `${apiUrl}/ollama`;
      },
    },
  ];
</script>

<div class="page-header">
  <div>
    <div class="eyebrow">SETTINGS</div>
    <h1>Server &amp; cluster.</h1>
    <div class="subtitle">
      Endpoints, cluster overview, and one-time setup state.
    </div>
  </div>
</div>

<section class="block">
  <div class="block-header">
    <div class="block-title">API endpoints</div>
    <div class="block-hint">Click any URL to copy.</div>
  </div>
  <div class="endpoints">
    {#each endpoints as ep}
      <button class="endpoint" onclick={() => copy(ep.url, ep.key)}>
        <div class="endpoint-label">{ep.label}</div>
        <div class="endpoint-url">{ep.url}</div>
        <div class="endpoint-hint">{ep.hint}</div>
        <div class="endpoint-copy">{copied === ep.key ? "COPIED" : "COPY"}</div>
      </button>
    {/each}
  </div>
</section>

<div class="two-col">
  <section class="block">
    <div class="block-header">
      <div class="block-title">Cluster</div>
      <div class="block-hint">Live snapshot.</div>
    </div>
    <dl class="kv">
      <div>
        <dt>Nodes</dt>
        <dd>{clusterSnap.nodes.length}</dd>
      </div>
      <div>
        <dt>Loaded models</dt>
        <dd>{stats?.instanceCount ?? 0}</dd>
      </div>
      <div>
        <dt>Memory</dt>
        <dd>
          {usedMemoryGb > 0 ? fmtMem(usedMemoryGb) : "—"}
          <span class="muted"> / {fmtMem(totalMemoryGb)}</span>
        </dd>
      </div>
      <div>
        <dt>Total requests</dt>
        <dd>{stats?.totalRequests?.toLocaleString() ?? "—"}</dd>
      </div>
      <div>
        <dt>Active commands</dt>
        <dd>{stats?.activeCommands ?? 0}</dd>
      </div>
    </dl>
  </section>

  <section class="block">
    <div class="block-header">
      <div class="block-title">First-launch state</div>
      <div class="block-hint">Reset to walk through setup again.</div>
    </div>
    <div class="onboarding-row">
      <div>
        <div class="status-line">
          <span class="dot" class:green={onboardingDone === true} class:amber={onboardingDone === false}></span>
          <span>
            {#if onboardingDone === true}
              Onboarding completed.
            {:else if onboardingDone === false}
              Onboarding not yet complete.
            {:else}
              Checking…
            {/if}
          </span>
        </div>
        <div class="muted spacer">
          The macOS app shows this on first run. Reset re-triggers it on next launch.
        </div>
      </div>
      <button class="btn" onclick={resetOnboarding}>Re-run onboarding</button>
    </div>
  </section>
</div>

<section class="block">
  <div class="block-header">
    <div class="block-title">Advanced</div>
    <div class="block-hint">Things still living in the legacy surfaces.</div>
  </div>
  <div class="link-list">
    <a class="link-row" href="#/legacy">
      <div>
        <div class="link-title">Legacy dashboard</div>
        <div class="link-desc">Detailed topology, downloads, RDMA, debugging panels.</div>
      </div>
      <div class="link-arrow">→</div>
    </a>
    <a class="link-row" href="#/integrations">
      <div>
        <div class="link-title">Integrations</div>
        <div class="link-desc">Generated config files for Claude Code, Codex, Pi, and more.</div>
      </div>
      <div class="link-arrow">→</div>
    </a>
    <a class="link-row" href="#/cluster">
      <div>
        <div class="link-title">Cluster detail</div>
        <div class="link-desc">Per-node memory, temperature, IDs.</div>
      </div>
      <div class="link-arrow">→</div>
    </a>
  </div>
</section>

<style>
  .page-header {
    margin-bottom: 28px;
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

  .block {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 18px 20px 20px;
    margin-bottom: 16px;
  }
  .block-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 14px;
    gap: 10px;
    flex-wrap: wrap;
  }
  .block-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--ux-text);
  }
  .block-hint {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .endpoints {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 10px;
  }
  .endpoint {
    background: var(--ux-bg-raised);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius-sm);
    padding: 12px 14px;
    cursor: pointer;
    text-align: left;
    color: var(--ux-text);
    font: inherit;
    position: relative;
    transition: border-color 120ms;
  }
  .endpoint:hover {
    border-color: var(--ux-border-strong);
  }
  .endpoint-label {
    font-family: var(--ux-mono);
    font-size: 9.5px;
    color: var(--ux-text-faint);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .endpoint-url {
    font-family: var(--ux-mono);
    font-size: 12.5px;
    color: var(--ux-text);
    word-break: break-all;
    line-height: 1.3;
  }
  .endpoint-hint {
    font-size: 11px;
    color: var(--ux-text-dim);
    margin-top: 6px;
    line-height: 1.4;
  }
  .endpoint-copy {
    position: absolute;
    top: 10px;
    right: 12px;
    font-family: var(--ux-mono);
    font-size: 9px;
    color: var(--ux-accent);
    letter-spacing: 0.12em;
    opacity: 0;
    transition: opacity 120ms;
  }
  .endpoint:hover .endpoint-copy {
    opacity: 1;
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
  }
  @media (max-width: 760px) {
    .two-col {
      grid-template-columns: 1fr;
    }
  }
  .two-col .block {
    margin-bottom: 0;
  }

  .kv {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    margin: 0;
  }
  .kv > div {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 6px 0;
    border-bottom: 1px dashed var(--ux-border);
  }
  .kv > div:last-child {
    border-bottom: none;
  }
  .kv dt {
    font-family: var(--ux-mono);
    font-size: 11px;
    color: var(--ux-text-faint);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .kv dd {
    margin: 0;
    font-family: var(--ux-mono);
    font-size: 13px;
    color: var(--ux-text);
  }
  .muted {
    color: var(--ux-text-faint);
  }

  .onboarding-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  .status-line {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--ux-text);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ux-text-faint);
  }
  .dot.green {
    background: var(--ux-green);
  }
  .dot.amber {
    background: var(--ux-accent);
  }
  .spacer {
    margin-top: 6px;
    font-size: 11px;
  }
  .btn {
    font-family: var(--ux-sans);
    font-size: 12px;
    font-weight: 500;
    padding: 8px 14px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: var(--ux-card);
    color: var(--ux-text);
    cursor: pointer;
    text-decoration: none;
  }
  .btn:hover {
    background: var(--ux-bg-hover);
  }

  .link-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: var(--ux-border);
    border-radius: var(--ux-radius-sm);
    overflow: hidden;
  }
  .link-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    background: var(--ux-bg-raised);
    color: inherit;
    text-decoration: none;
    transition: background 120ms;
  }
  .link-row:hover {
    background: var(--ux-bg-hover);
  }
  .link-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--ux-text);
    margin-bottom: 2px;
  }
  .link-desc {
    font-size: 11.5px;
    color: var(--ux-text-dim);
  }
  .link-arrow {
    font-family: var(--ux-mono);
    color: var(--ux-text-faint);
    font-size: 16px;
  }
</style>

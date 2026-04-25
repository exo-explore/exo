<script lang="ts">
  import { models, type ModelEntry } from "$lib/api/models.svelte";

  type Tab = "downloaded" | "available";
  let tab = $state<Tab>("downloaded");
  let query = $state("");
  let launchingId = $state<string | null>(null);
  let lastError = $state<string | null>(null);

  let snap = $derived(models.value);
  let downloadedCount = $derived(snap.downloadedIds.size);
  let runningCount = $derived(snap.runningModelIds.size);

  let filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    let list = snap.all;
    if (tab === "downloaded") {
      list = list.filter((m) => snap.downloadedIds.has(m.id));
    }
    if (q) {
      list = list.filter(
        (m) =>
          m.id.toLowerCase().includes(q) ||
          m.name.toLowerCase().includes(q) ||
          (m.family ?? "").toLowerCase().includes(q),
      );
    }
    return [...list].sort((a, b) => {
      const ar = snap.runningModelIds.has(a.id) ? 0 : 1;
      const br = snap.runningModelIds.has(b.id) ? 0 : 1;
      if (ar !== br) return ar - br;
      const ad = snap.downloadedIds.has(a.id) ? 0 : 1;
      const bd = snap.downloadedIds.has(b.id) ? 0 : 1;
      if (ad !== bd) return ad - bd;
      return a.name.localeCompare(b.name);
    });
  });

  function fmtSize(mb: number): string {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(0)} MB`;
  }

  function fmtCtx(n: number | null): string {
    if (!n) return "—";
    if (n >= 1000) return `${(n / 1000).toFixed(0)}k ctx`;
    return `${n} ctx`;
  }

  function statusFor(m: ModelEntry): "running" | "downloaded" | "available" {
    if (snap.runningModelIds.has(m.id)) return "running";
    if (snap.downloadedIds.has(m.id)) return "downloaded";
    return "available";
  }

  async function launch(model: ModelEntry) {
    launchingId = model.id;
    lastError = null;
    try {
      const placementRes = await fetch(
        `/placement?model_id=${encodeURIComponent(model.id)}`,
      );
      if (!placementRes.ok) {
        const text = await placementRes.text();
        throw new Error(`placement: ${text || placementRes.status}`);
      }
      const instance = await placementRes.json();
      const createRes = await fetch("/instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance }),
      });
      if (!createRes.ok) {
        const text = await createRes.text();
        throw new Error(`create: ${text || createRes.status}`);
      }
      await models.refresh();
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      launchingId = null;
    }
  }

  async function startDownload(model: ModelEntry) {
    launchingId = model.id;
    lastError = null;
    try {
      const res = await fetch("/download/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: model.id }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`download: ${text || res.status}`);
      }
      await models.refresh();
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      launchingId = null;
    }
  }

  function familyMark(family: string | null): string {
    return (family ?? "?").toUpperCase().slice(0, 1);
  }

  function familyColor(family: string | null): string {
    const f = family ?? "model";
    let h = 0;
    for (let i = 0; i < f.length; i++) h = (h * 31 + f.charCodeAt(i)) | 0;
    const hue = Math.abs(h) % 360;
    return `hsl(${hue}, 60%, 55%)`;
  }
</script>

<div class="page-header">
  <div>
    <div class="eyebrow">MODELS</div>
    <h1>
      {#if downloadedCount === 0 && snap.loaded}
        No models on disk yet — browse below.
      {:else}
        {downloadedCount}
        {downloadedCount === 1 ? "model" : "models"} on disk · {runningCount} running.
      {/if}
    </h1>
    <div class="subtitle">
      Launch from disk, browse the catalog, watch downloads land.
    </div>
  </div>
  <div class="actions">
    <a class="btn" href="#/legacy">Open legacy picker</a>
  </div>
</div>

<div class="controls">
  <div class="tabs" role="tablist">
    <button
      role="tab"
      aria-selected={tab === "downloaded"}
      class:active={tab === "downloaded"}
      onclick={() => (tab = "downloaded")}
    >
      On disk <span class="count">{downloadedCount}</span>
    </button>
    <button
      role="tab"
      aria-selected={tab === "available"}
      class:active={tab === "available"}
      onclick={() => (tab = "available")}
    >
      Catalog <span class="count">{snap.all.length}</span>
    </button>
  </div>
  <div class="search">
    <input
      type="search"
      placeholder="Filter by name, family, ID…"
      bind:value={query}
    />
  </div>
</div>

{#if lastError}
  <div class="error-banner">
    <span class="error-tag">ERROR</span>
    {lastError}
    <button class="dismiss" onclick={() => (lastError = null)}>×</button>
  </div>
{/if}

{#if !snap.loaded}
  <div class="empty">
    <div class="empty-eyebrow">LOADING</div>
    <div class="empty-title">Reading model catalog…</div>
  </div>
{:else if filtered.length === 0}
  <div class="empty">
    <div class="empty-eyebrow">EMPTY</div>
    <div class="empty-title">
      {tab === "downloaded"
        ? "Nothing installed yet."
        : "No models match that filter."}
    </div>
    <div class="empty-sub">
      {#if tab === "downloaded"}
        Switch to <button class="link" onclick={() => (tab = "available")}>
          Catalog
        </button> to find one to download.
      {:else}
        Try a different search term.
      {/if}
    </div>
  </div>
{:else}
  <div class="grid">
    {#each filtered as model (model.id)}
      {@const status = statusFor(model)}
      <div class="card" class:running={status === "running"}>
        <div class="card-top">
          <div
            class="mark"
            style="background: {familyColor(model.family)}1f; color: {familyColor(model.family)}; border-color: {familyColor(model.family)}33;"
          >
            {familyMark(model.family)}
          </div>
          <div class="card-meta">
            <div class="card-title">{model.name}</div>
            <div class="card-id">{model.id}</div>
          </div>
          <div class="status-pill" data-status={status}>
            <span class="dot"></span>
            <span>
              {status === "running"
                ? "RUNNING"
                : status === "downloaded"
                  ? "ON DISK"
                  : "AVAILABLE"}
            </span>
          </div>
        </div>

        <div class="specs">
          {#if model.family}
            <span class="spec">{model.family}</span>
          {/if}
          {#if model.quantization}
            <span class="spec">{model.quantization}</span>
          {/if}
          <span class="spec">{fmtSize(model.storageMb)}</span>
          <span class="spec">{fmtCtx(model.contextLength)}</span>
          {#if model.supportsTensor}
            <span class="spec accent">TP</span>
          {/if}
          {#each model.capabilities.slice(0, 3) as cap}
            <span class="spec faint">{cap}</span>
          {/each}
        </div>

        <div class="card-actions">
          {#if status === "running"}
            <a class="btn-action primary" href="#/chat">Open chat →</a>
            <span class="muted">Already loaded across the cluster.</span>
          {:else if status === "downloaded"}
            <button
              class="btn-action primary"
              disabled={launchingId === model.id}
              onclick={() => launch(model)}
            >
              {launchingId === model.id ? "Launching…" : "Launch"}
            </button>
            <span class="muted">Will load onto best-fit node.</span>
          {:else}
            <button
              class="btn-action"
              disabled={launchingId === model.id}
              onclick={() => startDownload(model)}
            >
              {launchingId === model.id
                ? "Starting…"
                : `Download · ${fmtSize(model.storageMb)}`}
            </button>
          {/if}
        </div>
      </div>
    {/each}
  </div>
{/if}

<style>
  .page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 28px;
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
  .actions {
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
    text-decoration: none;
  }
  .btn:hover {
    background: var(--ux-bg-hover);
  }

  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
    flex-wrap: wrap;
  }
  .tabs {
    display: inline-flex;
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius-sm);
    padding: 3px;
    gap: 2px;
  }
  .tabs button {
    background: transparent;
    border: none;
    color: var(--ux-text-dim);
    font-family: var(--ux-sans);
    font-size: 12px;
    font-weight: 500;
    padding: 7px 14px;
    border-radius: 4px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: background 100ms, color 100ms;
  }
  .tabs button:hover {
    color: var(--ux-text);
  }
  .tabs button.active {
    background: var(--ux-bg-raised);
    color: var(--ux-text);
  }
  .count {
    font-family: var(--ux-mono);
    font-size: 10px;
    background: var(--ux-bg);
    padding: 1px 6px;
    border-radius: 3px;
    color: var(--ux-text-faint);
  }
  .tabs button.active .count {
    color: var(--ux-accent);
  }
  .search input {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius-sm);
    color: var(--ux-text);
    font-family: var(--ux-sans);
    font-size: 13px;
    padding: 8px 12px;
    min-width: 280px;
    outline: none;
    transition: border-color 100ms;
  }
  .search input:focus {
    border-color: var(--ux-border-strong);
  }
  .search input::placeholder {
    color: var(--ux-text-faint);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(245, 100, 80, 0.08);
    border: 1px solid rgba(245, 100, 80, 0.4);
    border-radius: var(--ux-radius-sm);
    margin-bottom: 16px;
    font-family: var(--ux-mono);
    font-size: 12px;
    color: #f5b6a8;
  }
  .error-tag {
    background: rgba(245, 100, 80, 0.2);
    color: #ffb59f;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.08em;
  }
  .dismiss {
    margin-left: auto;
    background: transparent;
    border: none;
    color: #f5b6a8;
    cursor: pointer;
    font-size: 18px;
    line-height: 1;
    padding: 0 4px;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    transition: border-color 120ms;
  }
  .card:hover {
    border-color: var(--ux-border-strong);
  }
  .card.running {
    border-color: rgba(74, 222, 128, 0.35);
    background: linear-gradient(
      180deg,
      rgba(74, 222, 128, 0.04) 0%,
      var(--ux-card) 60%
    );
  }
  .card-top {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }
  .mark {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    border: 1px solid;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--ux-mono);
    font-weight: 600;
    font-size: 16px;
    flex-shrink: 0;
  }
  .card-meta {
    flex: 1;
    min-width: 0;
  }
  .card-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--ux-text);
    word-break: break-word;
    line-height: 1.3;
  }
  .card-id {
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
    margin-top: 2px;
    word-break: break-all;
    line-height: 1.3;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--ux-mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--ux-text-faint);
    padding: 4px 7px;
    border-radius: 3px;
    background: var(--ux-bg-raised);
    flex-shrink: 0;
  }
  .status-pill .dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
  }
  .status-pill[data-status="running"] {
    color: var(--ux-green);
    background: rgba(74, 222, 128, 0.1);
  }
  .status-pill[data-status="running"] .dot {
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.15);
  }
  .status-pill[data-status="downloaded"] {
    color: var(--ux-accent);
    background: rgba(245, 166, 35, 0.08);
  }

  .specs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .spec {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-dim);
    background: var(--ux-bg-raised);
    padding: 3px 7px;
    border-radius: 3px;
    text-transform: lowercase;
    letter-spacing: 0.02em;
  }
  .spec.accent {
    color: var(--ux-accent);
    background: rgba(245, 166, 35, 0.1);
  }
  .spec.faint {
    color: var(--ux-text-faint);
  }

  .card-actions {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .btn-action {
    font-family: var(--ux-sans);
    font-size: 12px;
    font-weight: 500;
    padding: 7px 14px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: transparent;
    color: var(--ux-text);
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: background 120ms, border-color 120ms;
  }
  .btn-action:hover:not(:disabled) {
    background: var(--ux-bg-hover);
    border-color: #353535;
  }
  .btn-action:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .btn-action.primary {
    background: var(--ux-text);
    color: var(--ux-bg);
    border-color: var(--ux-text);
    font-weight: 600;
  }
  .btn-action.primary:hover:not(:disabled) {
    background: #fff;
  }
  .muted {
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
    letter-spacing: 0.02em;
  }

  .empty {
    text-align: center;
    padding: 80px 24px;
  }
  .empty-eyebrow {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.14em;
    margin-bottom: 10px;
  }
  .empty-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--ux-text);
    margin-bottom: 8px;
  }
  .empty-sub {
    font-size: 13px;
    color: var(--ux-text-dim);
  }
  .link {
    background: transparent;
    border: none;
    color: var(--ux-accent);
    cursor: pointer;
    text-decoration: underline;
    font: inherit;
    padding: 0;
  }
</style>

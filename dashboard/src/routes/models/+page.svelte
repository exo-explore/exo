<script lang="ts">
  import {
    models,
    type ModelEntry,
    type NodeDownloadProgress,
    type NodeInfo,
  } from "$lib/api/models.svelte";

  type Tab = "downloaded" | "available";
  let tab = $state<Tab>("downloaded");
  let query = $state("");
  let busyId = $state<string | null>(null);
  let lastError = $state<string | null>(null);
  let nodePickerForModel = $state<string | null>(null);

  let snap = $derived(models.value);
  let downloadedCount = $derived(snap.downloadedIds.size);
  let runningCount = $derived(snap.runningModelIds.size);
  let nodeCount = $derived(snap.nodes.length);
  let activeDownloadCount = $derived.by(() => {
    let n = 0;
    for (const inner of snap.perModelNodes.values()) {
      for (const p of inner.values()) {
        if (p.status === "ongoing" || p.status === "pending") n++;
      }
    }
    return n;
  });

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

  function fmtBytes(b: number): string {
    if (b >= 1024 ** 3) return `${(b / 1024 ** 3).toFixed(1)} GB`;
    if (b >= 1024 ** 2) return `${(b / 1024 ** 2).toFixed(0)} MB`;
    if (b >= 1024) return `${(b / 1024).toFixed(0)} KB`;
    return `${b} B`;
  }

  function fmtSpeed(bps: number): string {
    if (bps <= 0) return "";
    return `${fmtBytes(bps)}/s`;
  }

  function fmtEta(ms: number): string {
    if (!ms || !isFinite(ms) || ms <= 0) return "";
    const s = Math.round(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m}m`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m`;
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

  /** Per-node disposition for a model: which nodes have it, are downloading, missing. */
  function nodeDispositions(m: ModelEntry): Array<{
    node: NodeInfo;
    progress: NodeDownloadProgress | null;
    isRunning: boolean;
  }> {
    const inner = snap.perModelNodes.get(m.id);
    const runningSet = snap.runningOnNodes.get(m.id) ?? new Set<string>();
    return snap.nodes.map((node) => ({
      node,
      progress: inner?.get(node.nodeId) ?? null,
      isRunning: runningSet.has(node.nodeId),
    }));
  }

  /** A model can be launched as-is when at least one node has the bytes. */
  function canLaunch(m: ModelEntry): boolean {
    return snap.downloadedIds.has(m.id);
  }

  /** Best-fit launch: ask the placement engine, post the resulting Instance. */
  async function launch(model: ModelEntry) {
    busyId = model.id;
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
      busyId = null;
    }
  }

  /**
   * Start a download on a specific node. Requires a shard_metadata blob, which
   * we get from any existing download record for this model. If none exists
   * (catalog model never seen), we have to ask the placement preview API for
   * one — the preview's Instance carries shard metadata in nodeToShard.
   */
  async function startDownloadOnNode(model: ModelEntry, nodeId: string) {
    busyId = model.id;
    lastError = null;
    nodePickerForModel = null;
    try {
      let shardMetadata = await models.getShardMetadata(model.id);
      if (!shardMetadata) {
        // Bootstrap shard metadata from a placement preview (it carries one
        // shard per assigned node — we just need any).
        const previewRes = await fetch(
          `/instance/previews?model_id=${encodeURIComponent(model.id)}`,
        );
        if (!previewRes.ok) {
          const text = await previewRes.text();
          throw new Error(`previews: ${text || previewRes.status}`);
        }
        const previewBody = (await previewRes.json()) as {
          previews?: Array<{
            instance?: {
              shardAssignments?: {
                nodeToShard?: Record<string, Record<string, unknown>>;
              };
            };
          }>;
        };
        const shards = previewBody.previews?.[0]?.instance?.shardAssignments
          ?.nodeToShard;
        if (shards) {
          // Wrap the shard payload back into the tagged-union envelope the
          // download API expects.
          const first = Object.values(shards)[0];
          if (first) {
            const tag = Object.keys(first)[0];
            shardMetadata = first as Record<string, unknown>;
            void tag;
          }
        }
        if (!shardMetadata) {
          throw new Error("no shard metadata for this model");
        }
      }

      const res = await fetch("/download/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targetNodeId: nodeId,
          shardMetadata,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`download: ${text || res.status}`);
      }
      await models.refresh();
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      busyId = null;
    }
  }

  async function cancelDownload(modelId: string, nodeId: string) {
    busyId = modelId;
    lastError = null;
    try {
      const res = await fetch("/download/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetNodeId: nodeId, modelId }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`cancel: ${text || res.status}`);
      }
      await models.refresh();
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      busyId = null;
    }
  }

  async function deleteFromNode(modelId: string, nodeId: string) {
    if (!confirm(`Delete this model from ${shortNode(nodeId)}?`)) return;
    busyId = modelId;
    lastError = null;
    try {
      const res = await fetch(
        `/download/${encodeURIComponent(nodeId)}/${encodeURIComponent(modelId)}`,
        { method: "DELETE" },
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`delete: ${text || res.status}`);
      }
      await models.refresh();
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      busyId = null;
    }
  }

  function shortNode(nodeId: string): string {
    const n = snap.nodes.find((nn) => nn.nodeId === nodeId);
    return n?.friendlyName ?? nodeId.slice(-8);
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

  /** Default download target when there's exactly one node. */
  function defaultDownloadTarget(): NodeInfo | null {
    if (snap.nodes.length === 1) return snap.nodes[0]!;
    const local = snap.nodes.find((n) => n.isLocal);
    return local ?? snap.nodes[0] ?? null;
  }

  function handleDownloadClick(model: ModelEntry) {
    if (snap.nodes.length <= 1) {
      const target = defaultDownloadTarget();
      if (target) startDownloadOnNode(model, target.nodeId);
      return;
    }
    nodePickerForModel = nodePickerForModel === model.id ? null : model.id;
  }

  function closeNodePicker() {
    nodePickerForModel = null;
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
      {@const dispositions = nodeDispositions(model)}
      {@const completedNodes = dispositions.filter((d) => d.progress?.status === "completed")}
      {@const ongoingNodes = dispositions.filter((d) => d.progress?.status === "ongoing" || d.progress?.status === "pending")}
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
              {#if status === "running"}RUNNING
              {:else if status === "downloaded"}
                {#if nodeCount > 1}
                  {completedNodes.length}/{nodeCount} NODES
                {:else}
                  ON DISK
                {/if}
              {:else if ongoingNodes.length > 0}
                DOWNLOADING
              {:else}
                AVAILABLE
              {/if}
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

        {#if nodeCount > 0 && (status !== "available" || ongoingNodes.length > 0)}
          <div class="node-row">
            {#each dispositions as d}
              {@const p = d.progress}
              {@const dispStatus = d.isRunning
                ? "running"
                : p?.status ?? "missing"}
              <div
                class="node-tag"
                data-status={dispStatus}
                title={`${d.node.friendlyName} · ${dispStatus}${p && p.totalBytes > 0 ? ` · ${fmtBytes(p.downloadedBytes)} / ${fmtBytes(p.totalBytes)}` : ""}`}
              >
                <span class="node-dot"></span>
                <span class="node-name">{d.node.friendlyName}</span>
                {#if d.isRunning}
                  <span class="node-extra">running</span>
                {:else if p?.status === "completed"}
                  <span class="node-extra">on disk</span>
                {:else if p?.status === "ongoing"}
                  <span class="node-extra">{p.percent.toFixed(0)}%</span>
                {:else if p?.status === "pending"}
                  <span class="node-extra">queued</span>
                {:else if p?.status === "failed"}
                  <span class="node-extra">failed</span>
                {/if}
                {#if p?.status === "ongoing" || p?.status === "pending"}
                  <button
                    class="node-action cancel"
                    title="Cancel this download"
                    onclick={(e) => {
                      e.stopPropagation();
                      cancelDownload(model.id, d.node.nodeId);
                    }}
                  >×</button>
                {:else if p?.status === "completed" && !d.isRunning}
                  <button
                    class="node-action delete"
                    title={`Delete from ${d.node.friendlyName}`}
                    onclick={(e) => {
                      e.stopPropagation();
                      deleteFromNode(model.id, d.node.nodeId);
                    }}
                  >×</button>
                {/if}
              </div>
            {/each}
          </div>

          {#if ongoingNodes.length > 0}
            {#each ongoingNodes as on}
              {@const p = on.progress!}
              <div class="progress-row">
                <div class="progress-meta">
                  <span class="progress-node">{on.node.friendlyName}</span>
                  <span class="progress-stats">
                    {fmtBytes(p.downloadedBytes)} / {fmtBytes(p.totalBytes)}
                    {#if p.speedBps > 0} · {fmtSpeed(p.speedBps)}{/if}
                    {#if p.etaMs > 0} · ETA {fmtEta(p.etaMs)}{/if}
                  </span>
                </div>
                <div class="progress-bar">
                  <div class="progress-fill" style="width: {p.percent}%"></div>
                </div>
              </div>
            {/each}
          {/if}
        {/if}

        <div class="card-actions">
          {#if status === "running"}
            <a class="btn-action primary" href="#/chat">Open chat →</a>
            {#if completedNodes.length > 0}
              <span class="muted">
                Loaded · {completedNodes.length}/{nodeCount} {nodeCount === 1 ? "node" : "nodes"} on disk
              </span>
            {/if}
          {:else if status === "downloaded"}
            <button
              class="btn-action primary"
              disabled={busyId === model.id}
              onclick={() => launch(model)}
            >
              {busyId === model.id ? "Launching…" : "Launch"}
            </button>
            {#if completedNodes.length < nodeCount && nodeCount > 1}
              <button
                class="btn-action"
                disabled={busyId === model.id}
                onclick={() => handleDownloadClick(model)}
              >
                + Download to another node
              </button>
            {/if}
          {:else}
            <button
              class="btn-action"
              disabled={busyId === model.id || nodeCount === 0}
              onclick={() => handleDownloadClick(model)}
            >
              {#if busyId === model.id}
                Starting…
              {:else if nodeCount > 1}
                Download · {fmtSize(model.storageMb)}
              {:else}
                Download · {fmtSize(model.storageMb)}
              {/if}
            </button>
          {/if}
        </div>

        {#if nodePickerForModel === model.id}
          <button
            class="picker-backdrop"
            type="button"
            onclick={closeNodePicker}
            aria-label="Close node picker"
          ></button>
          <div class="node-picker" role="menu">
            <div class="node-picker-head">Download to which node?</div>
            {#each snap.nodes as n}
              {@const existing = snap.perModelNodes.get(model.id)?.get(n.nodeId)}
              {@const already = existing?.status === "completed"}
              {@const inProgress =
                existing?.status === "ongoing" ||
                existing?.status === "pending"}
              {@const fits =
                n.diskAvailableBytes >= model.storageMb * 1024 * 1024}
              <button
                type="button"
                class="node-picker-item"
                disabled={already || inProgress || !fits}
                onclick={() => startDownloadOnNode(model, n.nodeId)}
              >
                <span class="np-name">
                  {n.friendlyName}
                  {#if n.isLocal}<span class="np-tag">LOCAL</span>{/if}
                </span>
                <span class="np-disk">
                  {fmtBytes(n.diskAvailableBytes)} free
                </span>
                {#if already}
                  <span class="np-status">on disk ✓</span>
                {:else if inProgress}
                  <span class="np-status">downloading…</span>
                {:else if !fits}
                  <span class="np-status">not enough disk</span>
                {:else}
                  <span class="np-status accent">→</span>
                {/if}
              </button>
            {/each}
          </div>
        {/if}
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
    background: var(--ux-red-bg);
    border: 1px solid var(--ux-red-border);
    border-radius: var(--ux-radius-sm);
    margin-bottom: 16px;
    font-family: var(--ux-mono);
    font-size: 12px;
    color: var(--ux-red-text);
  }
  .error-tag {
    background: var(--ux-red-bg);
    color: var(--ux-red);
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
    color: var(--ux-red-text);
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
    position: relative;
  }
  .card:hover {
    border-color: var(--ux-border-strong);
  }
  .card.running {
    border-color: var(--ux-green-border);
    background: linear-gradient(
      180deg,
      var(--ux-green-bg) 0%,
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
    background: var(--ux-green-bg);
  }
  .status-pill[data-status="running"] .dot {
    box-shadow: 0 0 0 2px var(--ux-green-bg);
  }
  .status-pill[data-status="downloaded"] {
    color: var(--ux-accent);
    background: var(--ux-accent-bg);
  }
  .status-pill[data-status="available"] {
    color: var(--ux-text-dim);
    background: var(--ux-bg-raised);
  }
  .status-pill[data-status="available"] .dot {
    background: var(--ux-text-faint);
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
    background: var(--ux-accent-bg);
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
    border-color: var(--ux-border-stronger);
  }
  .btn-action:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .btn-action.primary {
    background: var(--ux-text);
    color: var(--ux-text-invert);
    border-color: var(--ux-text);
    font-weight: 600;
  }
  .btn-action.primary:hover:not(:disabled) {
    background: var(--ux-primary-hover);
    border-color: var(--ux-primary-hover);
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

  /* Per-node disposition row */
  .node-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .node-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 6px 4px 8px;
    background: var(--ux-bg-raised);
    border: 1px solid var(--ux-border);
    border-radius: 999px;
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-dim);
  }
  .node-tag .node-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ux-text-faint);
  }
  .node-tag[data-status="completed"] {
    background: var(--ux-accent-bg);
    border-color: var(--ux-accent);
    color: var(--ux-accent);
  }
  .node-tag[data-status="completed"] .node-dot {
    background: var(--ux-accent);
  }
  .node-tag[data-status="running"] {
    background: var(--ux-green-bg);
    border-color: var(--ux-green-border);
    color: var(--ux-green);
  }
  .node-tag[data-status="running"] .node-dot {
    background: var(--ux-green);
    box-shadow: 0 0 0 2px var(--ux-green-bg);
    animation: uxPulse 2.4s ease-in-out infinite;
  }
  .node-tag[data-status="ongoing"],
  .node-tag[data-status="pending"] {
    background: var(--ux-accent-bg);
    border-color: var(--ux-accent-bg-strong);
    color: var(--ux-accent-strong);
  }
  .node-tag[data-status="ongoing"] .node-dot {
    background: var(--ux-accent);
    animation: uxPulse 1.6s ease-in-out infinite;
  }
  .node-tag[data-status="failed"] {
    background: var(--ux-red-bg);
    border-color: var(--ux-red-border);
    color: var(--ux-red-text);
  }
  .node-tag[data-status="failed"] .node-dot {
    background: var(--ux-red);
  }
  .node-extra {
    color: inherit;
    opacity: 0.85;
  }
  .node-action {
    background: transparent;
    border: none;
    color: inherit;
    opacity: 0.55;
    cursor: pointer;
    padding: 0 2px;
    margin-left: 2px;
    font-size: 13px;
    line-height: 1;
  }
  .node-action:hover {
    opacity: 1;
  }

  /* Per-node progress bar */
  .progress-row {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .progress-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--ux-mono);
    font-size: 10.5px;
  }
  .progress-node {
    color: var(--ux-text);
  }
  .progress-stats {
    color: var(--ux-text-faint);
  }
  .progress-bar {
    height: 4px;
    background: var(--ux-bg-raised);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(
      90deg,
      var(--ux-accent),
      var(--ux-accent-strong)
    );
    border-radius: 2px;
    transition: width 400ms ease-out;
  }

  /* Node picker popover (multi-node clusters) */
  .picker-backdrop {
    position: fixed;
    inset: 0;
    background: transparent;
    border: none;
    padding: 0;
    cursor: default;
    z-index: 9;
  }
  .node-picker {
    position: absolute;
    bottom: 16px;
    right: 18px;
    min-width: 280px;
    background: var(--ux-card);
    border: 1px solid var(--ux-border-strong);
    border-radius: var(--ux-radius);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    z-index: 10;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .node-picker-head {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.12em;
    padding: 6px 10px 8px;
    border-bottom: 1px solid var(--ux-border);
    margin-bottom: 4px;
  }
  .node-picker-item {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 8px;
    align-items: center;
    padding: 8px 10px;
    background: transparent;
    border: none;
    border-radius: var(--ux-radius-sm);
    cursor: pointer;
    color: var(--ux-text);
    font-family: var(--ux-sans);
    font-size: 12.5px;
    text-align: left;
    transition: background 120ms;
  }
  .node-picker-item:hover:not(:disabled) {
    background: var(--ux-bg-hover);
  }
  .node-picker-item:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }
  .np-name {
    color: var(--ux-text);
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .np-tag {
    font-family: var(--ux-mono);
    font-size: 8.5px;
    background: var(--ux-blue-bg);
    color: var(--ux-blue);
    padding: 1px 4px;
    border-radius: 2px;
    letter-spacing: 0.06em;
  }
  .np-disk {
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
  }
  .np-status {
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
  }
  .np-status.accent {
    color: var(--ux-accent);
  }
</style>

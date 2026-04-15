<script lang="ts">
  import { onMount } from "svelte";
  import {
    listTrajectories,
    deleteTrajectories,
    getTrajectoryRawUrl,
    type TrajectoryListItem,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  let trajectories = $state<TrajectoryListItem[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let selectedIds = $state<Set<string>>(new Set());
  let deleting = $state(false);
  let compareA = $state<string | null>(null);
  let compareB = $state<string | null>(null);

  let allSelected = $derived(
    trajectories.length > 0 && selectedIds.size === trajectories.length,
  );

  const summary = $derived.by(() => {
    if (trajectories.length === 0) {
      return null;
    }
    let totalRequests = 0;
    let totalPrompt = 0;
    let totalCompletion = 0;
    let totalCached = 0;
    let totalTools = 0;
    const ttft: number[] = [];
    const ptps: number[] = [];
    const gtps: number[] = [];
    let hitsNone = 0;
    let hitsPartial = 0;
    let hitsExact = 0;
    const models = new Set<string>();
    for (const t of trajectories) {
      totalRequests += t.agentStepCount;
      totalPrompt += t.totalPromptTokens;
      totalCompletion += t.totalCompletionTokens;
      totalCached += t.totalCachedTokens;
      totalTools += t.toolCallCount;
      if (t.avgTtftMs !== null) ttft.push(t.avgTtftMs);
      if (t.avgPromptTps !== null) ptps.push(t.avgPromptTps);
      if (t.avgGenerationTps !== null) gtps.push(t.avgGenerationTps);
      hitsNone += t.cacheHitNone;
      hitsPartial += t.cacheHitPartial;
      hitsExact += t.cacheHitExact;
      if (t.model) models.add(t.model);
    }
    const totalHits = hitsNone + hitsPartial + hitsExact;
    const cacheHitRate =
      totalHits > 0 ? ((hitsPartial + hitsExact) / totalHits) * 100 : null;
    const cachedRatio =
      totalPrompt > 0 ? (totalCached / totalPrompt) * 100 : null;
    const mean = (xs: number[]) =>
      xs.length > 0 ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
    return {
      trajectoryCount: trajectories.length,
      totalRequests,
      totalPrompt,
      totalCompletion,
      totalCached,
      totalTools,
      avgTtftMs: mean(ttft),
      avgPromptTps: mean(ptps),
      avgGenerationTps: mean(gtps),
      cacheHitRate,
      cachedRatio,
      hitsNone,
      hitsPartial,
      hitsExact,
      models: [...models],
    };
  });

  function toggleSelect(sessionId: string) {
    const next = new Set(selectedIds);
    if (next.has(sessionId)) next.delete(sessionId);
    else next.add(sessionId);
    selectedIds = next;
  }

  function toggleSelectAll() {
    selectedIds = allSelected
      ? new Set()
      : new Set(trajectories.map((t) => t.sessionId));
  }

  async function handleDelete() {
    if (selectedIds.size === 0) return;
    const count = selectedIds.size;
    if (
      !confirm(
        `Delete ${count} trajector${count === 1 ? "y" : "ies"}? This cannot be undone.`,
      )
    )
      return;
    deleting = true;
    try {
      await deleteTrajectories([...selectedIds]);
      selectedIds = new Set();
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to delete trajectories";
    } finally {
      deleting = false;
    }
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleString();
  }

  function formatTokens(n: number): string {
    if (n < 1000) return String(n);
    if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
    return `${(n / 1_000_000).toFixed(2)}M`;
  }

  function formatMs(n: number | null): string {
    if (n === null) return "—";
    if (n < 1000) return `${n.toFixed(0)}ms`;
    return `${(n / 1000).toFixed(2)}s`;
  }

  function formatPct(n: number | null): string {
    return n === null ? "—" : `${n.toFixed(1)}%`;
  }

  function formatTps(n: number | null): string {
    return n === null ? "—" : n.toFixed(1);
  }

  async function downloadTrajectory(sessionId: string) {
    const response = await fetch(getTrajectoryRawUrl(sessionId));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function pickForCompare(sessionId: string) {
    if (compareA === null) compareA = sessionId;
    else if (compareB === null && sessionId !== compareA) compareB = sessionId;
    else {
      compareA = sessionId;
      compareB = null;
    }
  }

  function clearCompare() {
    compareA = null;
    compareB = null;
  }

  async function refresh() {
    loading = true;
    error = null;
    try {
      const response = await listTrajectories();
      trajectories = response.trajectories;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load trajectories";
    } finally {
      loading = false;
    }
  }

  onMount(refresh);
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <h1 class="text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow">
        Trajectories
      </h1>
      <div class="flex items-center gap-3">
        {#if compareA && compareB}
          <a
            href="#/trajectories/compare?a={encodeURIComponent(compareA)}&b={encodeURIComponent(compareB)}"
            class="text-xs font-mono text-exo-dark-gray bg-exo-yellow hover:bg-exo-yellow/90 transition-colors uppercase px-2 py-1 rounded font-semibold"
          >
            Compare
          </a>
          <button
            type="button"
            class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
            onclick={clearCompare}
          >
            Clear
          </button>
        {:else if compareA}
          <span class="text-xs font-mono text-exo-light-gray uppercase">
            Pick another to compare
          </span>
          <button
            type="button"
            class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
            onclick={clearCompare}
          >
            Clear
          </button>
        {/if}
        {#if selectedIds.size > 0}
          <button
            type="button"
            class="text-xs font-mono text-red-400 hover:text-red-300 transition-colors uppercase border border-red-500/40 px-2 py-1 rounded"
            onclick={handleDelete}
            disabled={deleting}
          >
            {deleting ? "Deleting..." : `Delete (${selectedIds.size})`}
          </button>
        {/if}
        <button
          type="button"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
          onclick={refresh}
          disabled={loading}
        >
          Refresh
        </button>
      </div>
    </div>

    {#if loading}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray text-sm"
      >
        Loading trajectories...
      </div>
    {:else if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-6 text-center text-red-400 text-sm"
      >
        {error}
      </div>
    {:else if trajectories.length === 0}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray space-y-2"
      >
        <div class="text-sm">No trajectories found.</div>
        <div class="text-xs text-exo-light-gray/70">
          Run exo with --trajectories or EXO_TRAJECTORIES=1 to record live chat
          completions.
        </div>
      </div>
    {:else if summary}
      <!-- Summary dashboard -->
      <div
        class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs font-mono"
      >
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3"
        >
          <div class="text-exo-light-gray uppercase tracking-wider">
            Trajectories
          </div>
          <div class="text-xl text-exo-yellow mt-1">
            {summary.trajectoryCount}
          </div>
          <div class="text-[10px] text-exo-light-gray/70 mt-1">
            {summary.totalRequests} requests
          </div>
        </div>
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3"
        >
          <div class="text-exo-light-gray uppercase tracking-wider">
            Prompt tokens
          </div>
          <div class="text-xl text-exo-yellow mt-1">
            {formatTokens(summary.totalPrompt)}
          </div>
          <div class="text-[10px] text-exo-light-gray/70 mt-1">
            {formatTokens(summary.totalCached)} cached
            ({formatPct(summary.cachedRatio)})
          </div>
        </div>
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3"
        >
          <div class="text-exo-light-gray uppercase tracking-wider">
            Completion tokens
          </div>
          <div class="text-xl text-exo-yellow mt-1">
            {formatTokens(summary.totalCompletion)}
          </div>
          <div class="text-[10px] text-exo-light-gray/70 mt-1">
            {summary.totalTools} tool calls
          </div>
        </div>
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3"
        >
          <div class="text-exo-light-gray uppercase tracking-wider">TTFT</div>
          <div class="text-xl text-exo-yellow mt-1">
            {formatMs(summary.avgTtftMs)}
          </div>
          <div class="text-[10px] text-exo-light-gray/70 mt-1">avg</div>
        </div>
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3"
        >
          <div class="text-exo-light-gray uppercase tracking-wider">
            Prefill / gen tok/s
          </div>
          <div class="text-xl text-exo-yellow mt-1">
            {formatTps(summary.avgPromptTps)} / {formatTps(summary.avgGenerationTps)}
          </div>
          <div class="text-[10px] text-exo-light-gray/70 mt-1">avg</div>
        </div>
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3"
        >
          <div class="text-exo-light-gray uppercase tracking-wider">
            Prefix cache hit
          </div>
          <div class="text-xl text-exo-yellow mt-1">
            {formatPct(summary.cacheHitRate)}
          </div>
          <div class="text-[10px] text-exo-light-gray/70 mt-1">
            {summary.hitsExact} exact &bull; {summary.hitsPartial} partial
            &bull; {summary.hitsNone} miss
          </div>
        </div>
      </div>

      {#if summary.models.length > 0}
        <div class="text-xs font-mono text-exo-light-gray/70">
          Models: {summary.models.join(", ")}
        </div>
      {/if}

      <!-- Table -->
      <div class="space-y-2">
        <div class="flex items-center gap-2 px-1">
          <button
            type="button"
            class="text-xs font-mono uppercase transition-colors {allSelected
              ? 'text-exo-yellow'
              : 'text-exo-light-gray hover:text-exo-yellow'}"
            onclick={toggleSelectAll}
          >
            {allSelected ? "Deselect all" : "Select all"}
          </button>
        </div>
        {#each trajectories as t}
          {@const isSelected = selectedIds.has(t.sessionId)}
          {@const isCompareA = compareA === t.sessionId}
          {@const isCompareB = compareB === t.sessionId}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            role="button"
            tabindex="0"
            class="w-full text-left rounded border-l-2 border-r border-t border-b transition-all p-3 cursor-pointer {isSelected
              ? 'bg-exo-yellow/10 border-l-exo-yellow border-r-exo-medium-gray/30 border-t-exo-medium-gray/30 border-b-exo-medium-gray/30'
              : 'bg-exo-black/30 border-l-transparent border-r-exo-medium-gray/30 border-t-exo-medium-gray/30 border-b-exo-medium-gray/30 hover:bg-white/[0.03]'}"
            onclick={() => toggleSelect(t.sessionId)}
            onkeydown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                toggleSelect(t.sessionId);
              }
            }}
          >
            <div class="flex items-center justify-between gap-4 mb-2">
              <div class="min-w-0 flex-1">
                <a
                  href="#/trajectories/{t.sessionId}"
                  class="text-sm font-mono truncate block transition-colors {isSelected
                    ? 'text-exo-yellow'
                    : 'text-white hover:text-exo-yellow'}"
                  onclick={(e) => e.stopPropagation()}
                >
                  {t.sessionId}
                  {#if isCompareA}
                    <span class="ml-2 text-xs text-exo-yellow">[A]</span>
                  {:else if isCompareB}
                    <span class="ml-2 text-xs text-exo-yellow">[B]</span>
                  {/if}
                </a>
                <div class="text-[10px] text-exo-light-gray font-mono mt-1">
                  {formatDate(t.updatedAt)} &bull; {t.model}
                </div>
              </div>
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <div
                class="flex items-center gap-2 shrink-0"
                onclick={(e) => e.stopPropagation()}
              >
                <button
                  type="button"
                  class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
                  onclick={() => pickForCompare(t.sessionId)}
                >
                  {isCompareA ? "Set B" : "Compare"}
                </button>
                <a
                  href="#/trajectories/{t.sessionId}"
                  class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
                >
                  View
                </a>
                <button
                  type="button"
                  class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
                  onclick={() => downloadTrajectory(t.sessionId)}
                >
                  Download
                </button>
              </div>
            </div>
            <div
              class="grid grid-cols-3 sm:grid-cols-6 gap-2 text-[10px] font-mono"
            >
              <div
                class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
              >
                steps <span class="text-white">{t.totalSteps}</span>
              </div>
              <div
                class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
              >
                pt <span class="text-white">{formatTokens(t.totalPromptTokens)}</span>
              </div>
              <div
                class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
              >
                ct <span class="text-white">{formatTokens(t.totalCompletionTokens)}</span>
              </div>
              <div
                class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
              >
                cached <span class="text-white">{formatTokens(t.totalCachedTokens)}</span>
              </div>
              <div
                class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
              >
                ttft <span class="text-white">{formatMs(t.avgTtftMs)}</span>
              </div>
              <div
                class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
              >
                gen <span class="text-white">{formatTps(t.avgGenerationTps)}</span>
                tok/s
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

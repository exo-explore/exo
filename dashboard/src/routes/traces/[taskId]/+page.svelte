<script lang="ts">
  import { page } from "$app/stores";
  import { onMount } from "svelte";
  import {
    fetchTraceStats,
    getTraceRawUrl,
    type TraceStatsResponse,
    type TraceCategoryStats,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  const taskId = $derived($page.params.taskId);

  let stats = $state<TraceStatsResponse | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  function formatDuration(us: number): string {
    if (us < 1000) return `${us.toFixed(0)}us`;
    if (us < 1_000_000) return `${(us / 1000).toFixed(2)}ms`;
    return `${(us / 1_000_000).toFixed(2)}s`;
  }

  function formatPercentage(part: number, total: number): string {
    if (total === 0) return "0.0%";
    return `${((part / total) * 100).toFixed(1)}%`;
  }

  // Parse hierarchical categories like "sync/compute" into phases
  type PhaseData = {
    name: string;
    subcategories: { name: string; stats: TraceCategoryStats }[];
    totalUs: number; // From outer span (e.g., "sync" category)
    stepCount: number; // Count of outer span events
  };

  function parsePhases(
    byCategory: Record<string, TraceCategoryStats>,
  ): PhaseData[] {
    const phases = new Map<
      string,
      {
        subcats: Map<string, TraceCategoryStats>;
        outerStats: TraceCategoryStats | null;
      }
    >();

    for (const [category, catStats] of Object.entries(byCategory)) {
      if (category.includes("/")) {
        const [phase, subcat] = category.split("/", 2);
        if (!phases.has(phase)) {
          phases.set(phase, { subcats: new Map(), outerStats: null });
        }
        phases.get(phase)!.subcats.set(subcat, catStats);
      } else {
        // Outer span - this IS the phase total
        if (!phases.has(category)) {
          phases.set(category, { subcats: new Map(), outerStats: null });
        }
        phases.get(category)!.outerStats = catStats;
      }
    }

    return Array.from(phases.entries())
      .filter(([_, data]) => data.outerStats !== null) // Only phases with outer spans
      .map(([name, data]) => ({
        name,
        subcategories: Array.from(data.subcats.entries())
          .map(([subName, subStats]) => ({ name: subName, stats: subStats }))
          .sort((a, b) => b.stats.totalUs - a.stats.totalUs),
        totalUs: data.outerStats!.totalUs, // Outer span total
        stepCount: data.outerStats!.count, // Number of steps
      }))
      .sort((a, b) => b.totalUs - a.totalUs);
  }

  async function downloadTrace() {
    if (!taskId) return;
    const response = await fetch(getTraceRawUrl(taskId));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trace_${taskId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function openInPerfetto() {
    if (!taskId) return;

    // Fetch trace data from our local API
    const response = await fetch(getTraceRawUrl(taskId));
    const traceData = await response.arrayBuffer();

    // Open Perfetto UI
    const perfettoWindow = window.open("https://ui.perfetto.dev");
    if (!perfettoWindow) {
      alert("Failed to open Perfetto. Please allow popups.");
      return;
    }

    // Wait for Perfetto to be ready, then send trace via postMessage
    const onMessage = (e: MessageEvent) => {
      if (e.data === "PONG") {
        window.removeEventListener("message", onMessage);
        perfettoWindow.postMessage(
          {
            perfetto: {
              buffer: traceData,
              title: `Trace ${taskId}`,
            },
          },
          "https://ui.perfetto.dev",
        );
      }
    };
    window.addEventListener("message", onMessage);

    // Ping Perfetto until it responds
    const pingInterval = setInterval(() => {
      perfettoWindow.postMessage("PING", "https://ui.perfetto.dev");
    }, 50);

    // Clean up after 10 seconds
    setTimeout(() => {
      clearInterval(pingInterval);
      window.removeEventListener("message", onMessage);
    }, 10000);
  }

  onMount(async () => {
    if (!taskId) {
      error = "No task ID provided";
      loading = false;
      return;
    }

    try {
      stats = await fetchTraceStats(taskId);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load trace";
    } finally {
      loading = false;
    }
  });

  const phases = $derived(stats ? parsePhases(stats.byCategory) : []);
  const sortedRanks = $derived(
    stats
      ? Object.keys(stats.byRank)
          .map(Number)
          .sort((a, b) => a - b)
      : [],
  );
  const nodeCount = $derived(sortedRanks.length || 1);
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <div>
        <h1
          class="text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow"
        >
          Trace
        </h1>
        <p class="text-sm text-exo-light-gray font-mono truncate max-w-lg">
          {taskId}
        </p>
      </div>
      <div class="flex items-center gap-3">
        <a
          href="#/traces"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-3 py-1.5 rounded"
        >
          All Traces
        </a>
        <button
          type="button"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-3 py-1.5 rounded"
          onclick={downloadTrace}
          disabled={loading || !!error}
        >
          Download
        </button>
        <button
          type="button"
          class="text-xs font-mono text-exo-dark-gray bg-exo-yellow hover:bg-exo-yellow/90 transition-colors uppercase px-3 py-1.5 rounded font-semibold"
          onclick={openInPerfetto}
          disabled={loading || !!error}
        >
          View Trace
        </button>
      </div>
    </div>

    {#if loading}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">Loading trace data...</div>
      </div>
    {:else if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-6 text-center text-red-400"
      >
        <div class="text-sm">{error}</div>
      </div>
    {:else if stats}
      <!-- Wall Time Summary -->
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-2"
      >
        <h2
          class="text-sm font-mono uppercase tracking-wider text-exo-light-gray"
        >
          Summary
        </h2>
        <div class="text-3xl font-mono text-exo-yellow">
          {formatDuration(stats.totalWallTimeUs)}
        </div>
        <div class="text-xs text-exo-light-gray">Total wall time</div>
      </div>

      <!-- By Phase -->
      {#if phases.length > 0}
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-4"
        >
          <h2
            class="text-sm font-mono uppercase tracking-wider text-exo-light-gray"
          >
            By Phase <span class="text-exo-light-gray/50">(avg per node)</span>
          </h2>
          <div class="space-y-4">
            {#each phases as phase}
              {@const normalizedTotal = phase.totalUs / nodeCount}
              {@const normalizedStepCount = phase.stepCount / nodeCount}
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-sm font-mono text-white">{phase.name}</span>
                  <span class="text-sm font-mono">
                    <span class="text-exo-yellow"
                      >{formatDuration(normalizedTotal)}</span
                    >
                    <span class="text-exo-light-gray ml-2">
                      ({normalizedStepCount} steps, {formatDuration(
                        normalizedTotal / normalizedStepCount,
                      )}/step)
                    </span>
                  </span>
                </div>
                {#if phase.subcategories.length > 0}
                  <div class="pl-4 space-y-1.5">
                    {#each phase.subcategories as subcat}
                      {@const normalizedSubcat =
                        subcat.stats.totalUs / nodeCount}
                      {@const pct = formatPercentage(
                        normalizedSubcat,
                        normalizedTotal,
                      )}
                      {@const perStep = normalizedSubcat / normalizedStepCount}
                      <div
                        class="flex items-center justify-between text-xs font-mono"
                      >
                        <span class="text-exo-light-gray">{subcat.name}</span>
                        <span class="text-white">
                          {formatDuration(normalizedSubcat)}
                          <span class="text-exo-light-gray ml-2">({pct})</span>
                          <span class="text-exo-light-gray/60 ml-2"
                            >{formatDuration(perStep)}/step</span
                          >
                        </span>
                      </div>
                      <!-- Progress bar -->
                      <div
                        class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                      >
                        <div
                          class="absolute inset-y-0 left-0 bg-gradient-to-r from-exo-yellow to-exo-yellow/70 transition-all duration-300"
                          style="width: {pct}"
                        ></div>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- By Rank -->
      {#if sortedRanks.length > 0}
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-4"
        >
          <h2
            class="text-sm font-mono uppercase tracking-wider text-exo-light-gray"
          >
            By Rank
          </h2>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each sortedRanks as rank}
              {@const rankStats = stats.byRank[rank]}
              {@const rankPhases = parsePhases(rankStats.byCategory)}
              <div
                class="rounded border border-exo-medium-gray/20 bg-exo-dark-gray/60 p-3 space-y-3"
              >
                <div class="text-sm font-mono text-exo-yellow">
                  Rank {rank}
                </div>
                <div class="space-y-2">
                  {#each rankPhases as phase}
                    <div class="space-y-1">
                      <div class="flex items-center justify-between text-xs">
                        <span class="font-mono text-exo-light-gray"
                          >{phase.name}</span
                        >
                        <span class="font-mono text-white">
                          {formatDuration(phase.totalUs)}
                          <span class="text-exo-light-gray/50 ml-1">
                            ({phase.stepCount}x)
                          </span>
                        </span>
                      </div>
                      {#if phase.subcategories.length > 0}
                        <div class="pl-2 space-y-0.5">
                          {#each phase.subcategories as subcat}
                            {@const pct = formatPercentage(
                              subcat.stats.totalUs,
                              phase.totalUs,
                            )}
                            {@const perStep =
                              subcat.stats.totalUs / phase.stepCount}
                            <div
                              class="flex items-center justify-between text-[10px] font-mono"
                            >
                              <span class="text-exo-light-gray/70"
                                >{subcat.name}</span
                              >
                              <span class="text-exo-light-gray">
                                {formatDuration(subcat.stats.totalUs)}
                                <span class="text-exo-light-gray/50"
                                  >({pct})</span
                                >
                                <span class="text-exo-light-gray/30 ml-1"
                                  >{formatDuration(perStep)}/step</span
                                >
                              </span>
                            </div>
                          {/each}
                        </div>
                      {/if}
                    </div>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>

<script lang="ts">
  import { onMount } from "svelte";
  import {
    listTraces,
    getTraceRawUrl,
    type TraceListItem,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  let traces = $state<TraceListItem[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);

  function formatBytes(bytes: number): string {
    if (!bytes || bytes <= 0) return "0B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.min(
      Math.floor(Math.log(bytes) / Math.log(1024)),
      units.length - 1,
    );
    const val = bytes / Math.pow(1024, i);
    return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
  }

  function formatDate(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleString();
  }

  async function downloadTrace(taskId: string) {
    const response = await fetch(getTraceRawUrl(taskId));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trace_${taskId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function openInPerfetto(taskId: string) {
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

  async function refresh() {
    loading = true;
    error = null;
    try {
      const response = await listTraces();
      traces = response.traces;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load traces";
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    refresh();
  });
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <div>
        <h1
          class="text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow"
        >
          Traces
        </h1>
      </div>
      <div class="flex items-center gap-3">
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
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">Loading traces...</div>
      </div>
    {:else if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-6 text-center text-red-400"
      >
        <div class="text-sm">{error}</div>
      </div>
    {:else if traces.length === 0}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray space-y-2"
      >
        <div class="text-sm">No traces found.</div>
        <div class="text-xs text-exo-light-gray/70">
          Run exo with EXO_TRACING_ENABLED=1 to collect traces.
        </div>
      </div>
    {:else}
      <div class="space-y-3">
        {#each traces as trace}
          <div
            class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 flex items-center justify-between gap-4"
          >
            <div class="min-w-0 flex-1">
              <a
                href="#/traces/{trace.taskId}"
                class="text-sm font-mono text-white hover:text-exo-yellow transition-colors truncate block"
              >
                {trace.taskId}
              </a>
              <div class="text-xs text-exo-light-gray font-mono mt-1">
                {formatDate(trace.createdAt)} &bull; {formatBytes(
                  trace.fileSize,
                )}
              </div>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <a
                href="#/traces/{trace.taskId}"
                class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
              >
                View Stats
              </a>
              <button
                type="button"
                class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
                onclick={() => downloadTrace(trace.taskId)}
              >
                Download
              </button>
              <button
                type="button"
                class="text-xs font-mono text-exo-dark-gray bg-exo-yellow hover:bg-exo-yellow/90 transition-colors uppercase px-2 py-1 rounded font-semibold"
                onclick={() => openInPerfetto(trace.taskId)}
              >
                View Trace
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

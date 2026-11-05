<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import {
    listLogs,
    getLogTail,
    getLogRawUrl,
    type LogFileListItem,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  const LOG_LABELS: Record<string, string> = {
    main: "Main Log",
    runner_stdout: "Runner Stdout",
    runner_stderr: "Runner Stderr",
  };

  let logs = $state<LogFileListItem[]>([]);
  let selectedName = $state<string | null>(null);
  let content = $state<string>("");
  let truncated = $state(false);
  let loadingList = $state(true);
  let loadingContent = $state(false);
  let error = $state<string | null>(null);
  let autoRefresh = $state(true);

  let refreshTimer: ReturnType<typeof setInterval> | undefined;

  function labelFor(name: string): string {
    return LOG_LABELS[name] ?? name;
  }

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
    return new Date(isoString).toLocaleString();
  }

  async function refreshList() {
    loadingList = true;
    error = null;
    try {
      const response = await listLogs();
      logs = response.logs;
      if (!selectedName && logs.length > 0) {
        selectLog(logs[0].name);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load logs";
    } finally {
      loadingList = false;
    }
  }

  async function refreshContent() {
    if (!selectedName) return;
    loadingContent = true;
    error = null;
    try {
      const response = await getLogTail(selectedName);
      content = response.content;
      truncated = response.truncated;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load log content";
    } finally {
      loadingContent = false;
    }
  }

  function selectLog(name: string) {
    selectedName = name;
    refreshContent();
  }

  async function downloadLog(name: string) {
    const response = await fetch(getLogRawUrl(name));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
  }

  $effect(() => {
    if (refreshTimer) clearInterval(refreshTimer);
    if (autoRefresh) {
      refreshTimer = setInterval(refreshContent, 3000);
    }
  });

  onMount(() => {
    refreshList();
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
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
          Logs
        </h1>
        <div class="text-xs text-exo-light-gray/70 font-mono mt-1">
          Logs from this node only.
        </div>
      </div>
      <div class="flex items-center gap-3">
        <button
          type="button"
          class="text-xs font-mono uppercase transition-colors border px-2 py-1 rounded {autoRefresh
            ? 'text-exo-yellow border-exo-yellow/40'
            : 'text-exo-light-gray hover:text-exo-yellow border-exo-medium-gray/40'}"
          onclick={toggleAutoRefresh}
        >
          Auto-refresh {autoRefresh ? "on" : "off"}
        </button>
        <button
          type="button"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
          onclick={refreshContent}
          disabled={loadingContent || !selectedName}
        >
          Refresh
        </button>
      </div>
    </div>

    {#if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400"
      >
        {error}
      </div>
    {/if}

    {#if loadingList}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">Loading logs...</div>
      </div>
    {:else if logs.length === 0}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">No log files found on this node.</div>
      </div>
    {:else}
      <div class="flex gap-2 flex-wrap">
        {#each logs as log}
          <button
            type="button"
            class="text-xs font-mono uppercase transition-colors border px-3 py-1.5 rounded {selectedName ===
            log.name
              ? 'text-exo-yellow border-exo-yellow/40 bg-exo-yellow/10'
              : 'text-exo-light-gray hover:text-exo-yellow border-exo-medium-gray/40'}"
            onclick={() => selectLog(log.name)}
          >
            {labelFor(log.name)}
            <span class="text-exo-light-gray/60 normal-case"
              >&nbsp;&bull; {formatBytes(log.fileSize)} &bull; {formatDate(
                log.modifiedAt,
              )}</span
            >
          </button>
        {/each}
        {#if selectedName}
          <button
            type="button"
            class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-3 py-1.5 rounded"
            onclick={() => downloadLog(selectedName!)}
          >
            Download full log
          </button>
        {/if}
      </div>

      {#if truncated}
        <div class="text-xs text-exo-light-gray/70 font-mono">
          Showing the tail of this log. Use "Download full log" for the complete
          file.
        </div>
      {/if}

      <pre
        class="rounded border border-exo-medium-gray/30 bg-exo-black/50 p-4 text-xs font-mono text-exo-light-gray whitespace-pre-wrap break-words overflow-y-auto max-h-[70vh]">{content ||
          (loadingContent ? "Loading..." : "No content.")}</pre>
    {/if}
  </div>
</div>

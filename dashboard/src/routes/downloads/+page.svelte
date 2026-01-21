<script lang="ts">
  import { onMount } from "svelte";
  import {
    topologyData,
    downloads,
    type DownloadProgress,
    refreshState,
    lastUpdate as lastUpdateStore,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  type FileProgress = {
    name: string;
    totalBytes: number;
    downloadedBytes: number;
    speed: number;
    etaMs: number;
    percentage: number;
  };

  type ModelEntry = {
    modelId: string;
    prettyName?: string | null;
    percentage: number;
    downloadedBytes: number;
    totalBytes: number;
    speed: number;
    etaMs: number;
    status: "completed" | "downloading";
    files: FileProgress[];
  };

  type NodeEntry = {
    nodeId: string;
    nodeName: string;
    models: ModelEntry[];
  };

  const data = $derived(topologyData());
  const downloadsData = $derived(downloads());

  function getNodeLabel(nodeId: string): string {
    const node = data?.nodes?.[nodeId];
    if (!node) return nodeId.slice(0, 8);
    return (
      node.friendly_name || node.system_info?.model_id || nodeId.slice(0, 8)
    );
  }

  function getBytes(value: unknown): number {
    if (typeof value === "number") return value;
    if (value && typeof value === "object") {
      const v = value as Record<string, unknown>;
      if (typeof v.in_bytes === "number") return v.in_bytes;
      if (typeof v.inBytes === "number") return v.inBytes;
    }
    return 0;
  }

  function formatBytes(bytes: number): string {
    if (!bytes || bytes <= 0) return "0B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(
      Math.floor(Math.log(bytes) / Math.log(1024)),
      units.length - 1,
    );
    const val = bytes / Math.pow(1024, i);
    return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
  }

  function formatEta(ms: number): string {
    if (!ms || ms <= 0) return "--";
    const totalSeconds = Math.round(ms / 1000);
    const s = totalSeconds % 60;
    const m = Math.floor(totalSeconds / 60) % 60;
    const h = Math.floor(totalSeconds / 3600);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function formatSpeed(bytesPerSecond: number): string {
    if (!bytesPerSecond || bytesPerSecond <= 0) return "--";
    const units = ["B/s", "KB/s", "MB/s", "GB/s"];
    const i = Math.min(
      Math.floor(Math.log(bytesPerSecond) / Math.log(1024)),
      units.length - 1,
    );
    const val = bytesPerSecond / Math.pow(1024, i);
    return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
  }

  function clampPercent(value: number | undefined): number {
    if (!Number.isFinite(value)) return 0;
    return Math.min(100, Math.max(0, value as number));
  }

  function extractModelIdFromDownload(
    downloadPayload: Record<string, unknown>,
  ): string | null {
    const shardMetadata =
      downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
    if (!shardMetadata || typeof shardMetadata !== "object") return null;

    const shardObj = shardMetadata as Record<string, unknown>;
    const shardKeys = Object.keys(shardObj);
    if (shardKeys.length !== 1) return null;

    const shardData = shardObj[shardKeys[0]] as Record<string, unknown>;
    if (!shardData) return null;

    const modelMeta = shardData.model_card ?? shardData.modelCard;
    if (!modelMeta || typeof modelMeta !== "object") return null;

    const meta = modelMeta as Record<string, unknown>;
    return (meta.model_id as string) ?? (meta.modelId as string) ?? null;
  }

  function parseDownloadProgress(
    payload: Record<string, unknown>,
  ): DownloadProgress | null {
    const progress = payload.download_progress ?? payload.downloadProgress;
    if (!progress || typeof progress !== "object") return null;

    const prog = progress as Record<string, unknown>;
    const totalBytes = getBytes(prog.total_bytes ?? prog.totalBytes);
    const downloadedBytes = getBytes(
      prog.downloaded_bytes ?? prog.downloadedBytes,
    );
    const speed = (prog.speed as number) ?? 0;
    const completedFiles =
      (prog.completed_files as number) ?? (prog.completedFiles as number) ?? 0;
    const totalFiles =
      (prog.total_files as number) ?? (prog.totalFiles as number) ?? 0;
    const etaMs = (prog.eta_ms as number) ?? (prog.etaMs as number) ?? 0;

    const files: DownloadProgress["files"] = [];
    const filesObj = (prog.files ?? {}) as Record<string, unknown>;
    for (const [fileName, fileData] of Object.entries(filesObj)) {
      if (!fileData || typeof fileData !== "object") continue;
      const fd = fileData as Record<string, unknown>;
      const fTotal = getBytes(fd.total_bytes ?? fd.totalBytes);
      const fDownloaded = getBytes(fd.downloaded_bytes ?? fd.downloadedBytes);
      files.push({
        name: fileName,
        totalBytes: fTotal,
        downloadedBytes: fDownloaded,
        speed: (fd.speed as number) ?? 0,
        etaMs: (fd.eta_ms as number) ?? (fd.etaMs as number) ?? 0,
        percentage: fTotal > 0 ? (fDownloaded / fTotal) * 100 : 0,
      });
    }

    return {
      totalBytes,
      downloadedBytes,
      speed,
      etaMs:
        etaMs ||
        (speed > 0 ? ((totalBytes - downloadedBytes) / speed) * 1000 : 0),
      percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
      completedFiles,
      totalFiles,
      files,
    };
  }

  function getBarGradient(percentage: number): string {
    if (percentage >= 100) return "from-green-500 to-green-400";
    if (percentage <= 0) return "from-red-500 to-red-400";
    return "from-exo-yellow to-exo-yellow/70";
  }

  let downloadOverview = $state<NodeEntry[]>([]);
  let models = $state<Array<{ id: string; storage_size_megabytes?: number }>>(
    [],
  );

  async function fetchModels() {
    try {
      const response = await fetch("/models");
      if (response.ok) {
        const data = await response.json();
        models = data.data || [];
      }
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  }

  function getModelTotalBytes(
    modelId: string,
    downloadTotalBytes: number,
  ): number {
    if (downloadTotalBytes > 0) return downloadTotalBytes;
    const model = models.find((m) => m.id === modelId);
    if (model?.storage_size_megabytes) {
      return model.storage_size_megabytes * 1024 * 1024;
    }
    return 0;
  }

  $effect(() => {
    try {
      if (!downloadsData || Object.keys(downloadsData).length === 0) {
        downloadOverview = [];
        return;
      }

      const entries = Object.entries(downloadsData);
      const built: NodeEntry[] = [];

      for (const [nodeId, nodeDownloads] of entries) {
        const modelMap = new Map<string, ModelEntry>();
        const nodeEntries = Array.isArray(nodeDownloads)
          ? nodeDownloads
          : nodeDownloads && typeof nodeDownloads === "object"
            ? Object.values(nodeDownloads as Record<string, unknown>)
            : [];

        for (const downloadWrapped of nodeEntries) {
          if (!downloadWrapped || typeof downloadWrapped !== "object") continue;

          const keys = Object.keys(downloadWrapped as Record<string, unknown>);
          if (keys.length !== 1) continue;

          const downloadKind = keys[0];
          const downloadPayload = (downloadWrapped as Record<string, unknown>)[
            downloadKind
          ] as Record<string, unknown>;
          if (!downloadPayload) continue;

          const modelId =
            extractModelIdFromDownload(downloadPayload) ?? "unknown-model";
          const prettyName = (() => {
            const shardMetadata =
              downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
            if (!shardMetadata || typeof shardMetadata !== "object")
              return null;
            const shardObj = shardMetadata as Record<string, unknown>;
            const shardKeys = Object.keys(shardObj);
            if (shardKeys.length !== 1) return null;
            const shardData = shardObj[shardKeys[0]] as Record<string, unknown>;
            const modelMeta = shardData?.model_card ?? shardData?.modelCard;
            if (!modelMeta || typeof modelMeta !== "object") return null;
            const meta = modelMeta as Record<string, unknown>;
            return (meta.prettyName as string) ?? null;
          })();

          const rawProgress =
            (downloadPayload as Record<string, unknown>).download_progress ??
            (downloadPayload as Record<string, unknown>).downloadProgress ??
            {};
          // For DownloadCompleted, total_bytes is at top level; for DownloadOngoing, it's inside download_progress
          const totalBytes = getBytes(
            (downloadPayload as Record<string, unknown>).total_bytes ??
              (downloadPayload as Record<string, unknown>).totalBytes ??
              (rawProgress as Record<string, unknown>).total_bytes ??
              (rawProgress as Record<string, unknown>).totalBytes,
          );
          const downloadedBytes = getBytes(
            (rawProgress as Record<string, unknown>).downloaded_bytes ??
              (rawProgress as Record<string, unknown>).downloadedBytes,
          );
          const speed =
            ((rawProgress as Record<string, unknown>).speed as number) ?? 0;
          const etaMs =
            ((rawProgress as Record<string, unknown>).eta_ms as number) ??
            ((rawProgress as Record<string, unknown>).etaMs as number) ??
            0;
          const percentage =
            totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0;

          const files: FileProgress[] = [];
          const filesObj = (rawProgress as Record<string, unknown>).files as
            | Record<string, unknown>
            | undefined;
          if (filesObj && typeof filesObj === "object") {
            for (const [fileName, fileData] of Object.entries(filesObj)) {
              if (!fileData || typeof fileData !== "object") continue;
              const fd = fileData as Record<string, unknown>;
              const fTotal = getBytes(fd.total_bytes ?? fd.totalBytes);
              const fDownloaded = getBytes(
                fd.downloaded_bytes ?? fd.downloadedBytes,
              );
              files.push({
                name: fileName,
                totalBytes: fTotal,
                downloadedBytes: fDownloaded,
                speed: (fd.speed as number) ?? 0,
                etaMs: (fd.eta_ms as number) ?? (fd.etaMs as number) ?? 0,
                percentage: clampPercent(
                  fTotal > 0 ? (fDownloaded / fTotal) * 100 : 0,
                ),
              });
            }
          }

          const entry: ModelEntry = {
            modelId,
            prettyName,
            percentage:
              downloadKind === "DownloadCompleted"
                ? 100
                : clampPercent(percentage),
            downloadedBytes,
            totalBytes,
            speed,
            etaMs,
            status:
              downloadKind === "DownloadCompleted"
                ? "completed"
                : "downloading",
            files,
          };

          const existing = modelMap.get(modelId);
          if (!existing) {
            modelMap.set(modelId, entry);
          } else if (
            (entry.status === "completed" && existing.status !== "completed") ||
            (entry.status === existing.status &&
              entry.downloadedBytes > existing.downloadedBytes)
          ) {
            modelMap.set(modelId, entry);
          }
        }

        let models = Array.from(modelMap.values()).sort(
          (a, b) => b.percentage - a.percentage,
        );
        if (models.length === 0 && nodeEntries.length > 0) {
          models = [
            {
              modelId: "Unknown download",
              percentage: 0,
              downloadedBytes: 0,
              totalBytes: 0,
              speed: 0,
              etaMs: 0,
              status: "downloading",
              files: [],
            },
          ];
        }

        built.push({
          nodeId,
          nodeName: getNodeLabel(nodeId),
          models,
        });
      }

      downloadOverview = built;
    } catch (err) {
      console.error("Parse downloads error", err);
      downloadOverview = [];
    }
  });

  const hasDownloads = $derived(downloadOverview.length > 0);
  const lastUpdateTs = $derived(lastUpdateStore());
  const downloadKeys = $derived(Object.keys(downloadsData || {}));

  let expanded = $state<Set<string>>(new Set());
  function toggleExpand(key: string): void {
    const next = new Set(expanded);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    expanded = next;
  }

  onMount(() => {
    // Ensure we fetch at least once when visiting downloads directly
    refreshState();
    fetchModels();
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
          Downloads
        </h1>
        <p class="text-sm text-exo-light-gray">
          Overview of models on each node
        </p>
      </div>
      <div class="flex items-center gap-3">
        <button
          type="button"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
          onclick={() => refreshState()}
          title="Force refresh from /state"
        >
          Refresh
        </button>
        <div class="text-[11px] font-mono text-exo-light-gray">
          Last update: {lastUpdateTs
            ? new Date(lastUpdateTs).toLocaleTimeString()
            : "n/a"}
        </div>
      </div>
    </div>

    {#if !hasDownloads}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray space-y-2"
      >
        <div class="text-sm">
          No downloads found. Start a model download to see progress here.
        </div>
        <div class="text-[11px] text-exo-light-gray/70">
          Download keys detected: {downloadKeys.length === 0
            ? "none"
            : downloadKeys.join(", ")}
        </div>
      </div>
    {:else}
      <div class="downloads-grid gap-4">
        {#each downloadOverview as node}
          <div
            class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-3 flex flex-col"
          >
            <div class="flex items-center justify-between gap-3">
              <div class="min-w-0 flex-1">
                <div class="text-lg font-mono text-white truncate">
                  {node.nodeName}
                </div>
                <div class="text-xs text-exo-light-gray font-mono truncate">
                  {node.nodeId}
                </div>
              </div>
              <div
                class="text-xs font-mono uppercase tracking-wider whitespace-nowrap shrink-0 text-right"
              >
                <div>
                  <span class="text-green-400"
                    >{node.models.filter((m) => m.status === "completed")
                      .length}</span
                  ><span class="text-exo-yellow">
                    / {node.models.length} models</span
                  >
                </div>
                <div class="text-exo-light-gray normal-case tracking-normal">
                  {formatBytes(
                    node.models
                      .filter((m) => m.status === "completed")
                      .reduce((sum, m) => sum + m.totalBytes, 0),
                  )} on disk
                </div>
              </div>
            </div>

            {#each node.models as model}
              {@const key = `${node.nodeId}|${model.modelId}`}
              {@const pct = clampPercent(model.percentage)}
              {@const gradient = getBarGradient(pct)}
              {@const isExpanded = expanded.has(key)}
              <div
                class="rounded border border-exo-medium-gray/30 bg-exo-dark-gray/60 p-3 space-y-2"
              >
                <div class="flex items-center justify-between gap-3">
                  <div class="min-w-0 space-y-0.5">
                    <div
                      class="text-xs font-mono text-white truncate"
                      title={model.prettyName ?? model.modelId}
                    >
                      {model.prettyName ?? model.modelId}
                    </div>
                    <div
                      class="text-[10px] text-exo-light-gray font-mono truncate"
                      title={model.modelId}
                    >
                      {model.modelId}
                    </div>
                    {#if model.status !== "completed"}
                      <div class="text-[11px] text-exo-light-gray font-mono">
                        {formatBytes(model.downloadedBytes)} / {formatBytes(
                          getModelTotalBytes(model.modelId, model.totalBytes),
                        )}
                      </div>
                    {/if}
                  </div>
                  <div class="flex items-center gap-2">
                    <span
                      class="text-xs font-mono {pct >= 100
                        ? 'text-green-400'
                        : pct <= 0
                          ? 'text-red-400'
                          : 'text-exo-yellow'}"
                    >
                      {pct.toFixed(1)}%
                    </span>
                    <button
                      type="button"
                      class="text-exo-light-gray hover:text-exo-yellow transition-colors"
                      onclick={() => toggleExpand(key)}
                      aria-expanded={isExpanded}
                      title="Toggle file details"
                    >
                      <svg
                        class="w-4 h-4"
                        viewBox="0 0 20 20"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <path
                          d="M6 8l4 4 4-4"
                          class={isExpanded
                            ? "transform rotate-180 origin-center transition-transform duration-150"
                            : "transition-transform duration-150"}
                        ></path>
                      </svg>
                    </button>
                  </div>
                </div>

                <div
                  class="relative h-2 bg-exo-black/60 rounded-sm overflow-hidden"
                >
                  <div
                    class={`absolute inset-y-0 left-0 bg-gradient-to-r ${gradient} transition-all duration-300`}
                    style={`width: ${pct.toFixed(1)}%`}
                  ></div>
                </div>

                <div
                  class="flex items-center justify-between text-xs font-mono text-exo-light-gray"
                >
                  <span
                    >{model.status === "completed"
                      ? `Completed (${formatBytes(model.totalBytes)})`
                      : `${formatSpeed(model.speed)} • ETA ${formatEta(model.etaMs)}`}</span
                  >
                  {#if model.status !== "completed"}
                    <span
                      >{model.files.length} file{model.files.length === 1
                        ? ""
                        : "s"}</span
                    >
                  {/if}
                </div>

                {#if isExpanded}
                  <div class="mt-2 space-y-1.5">
                    {#if model.files.length === 0}
                      <div class="text-[11px] font-mono text-exo-light-gray/70">
                        No file details reported.
                      </div>
                    {:else}
                      {#each model.files as f}
                        {@const fpct = clampPercent(f.percentage)}
                        {@const fgradient = getBarGradient(fpct)}
                        <div
                          class="rounded border border-exo-medium-gray/20 bg-exo-black/40 p-2 space-y-1"
                        >
                          <div
                            class="flex items-center justify-between text-[11px] font-mono text-exo-light-gray/90"
                          >
                            <span class="truncate pr-2">{f.name}</span>
                            <span
                              class={fpct >= 100
                                ? "text-green-400"
                                : fpct <= 0
                                  ? "text-red-400"
                                  : "text-exo-yellow"}>{fpct.toFixed(1)}%</span
                            >
                          </div>
                          <div
                            class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                          >
                            <div
                              class={`absolute inset-y-0 left-0 bg-gradient-to-r ${fgradient} transition-all duration-300`}
                              style={`width: ${fpct.toFixed(1)}%`}
                            ></div>
                          </div>
                          <div
                            class="flex items-center justify-between text-[10px] text-exo-light-gray/70"
                          >
                            <span
                              >{formatBytes(f.downloadedBytes)} / {formatBytes(
                                f.totalBytes,
                              )}</span
                            >
                            <span
                              >{formatSpeed(f.speed)} • ETA {formatEta(
                                f.etaMs,
                              )}</span
                            >
                          </div>
                        </div>
                      {/each}
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .downloads-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  }
  @media (min-width: 1024px) {
    .downloads-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
  @media (min-width: 1600px) {
    .downloads-grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }
</style>

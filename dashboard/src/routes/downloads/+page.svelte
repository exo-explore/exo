<script lang="ts">
  import { onMount } from "svelte";
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import {
    topologyData,
    downloads,
    nodeDisk,
    nodeStorageConfig,
    refreshState,
    lastUpdate as lastUpdateStore,
    startDownload,
    deleteDownload,
    setStorageConfig,
  } from "$lib/stores/app.svelte";
  import {
    getDownloadTag,
    extractModelIdFromDownload,
    extractShardMetadata,
  } from "$lib/utils/downloads";
  import { addToast } from "$lib/stores/toast.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  type CellStatus =
    | { kind: "completed"; totalBytes: number; modelDirectory?: string }
    | {
        kind: "downloading";
        percentage: number;
        downloadedBytes: number;
        totalBytes: number;
        speed: number;
        etaMs: number;
        modelDirectory?: string;
      }
    | {
        kind: "pending";
        downloaded: number;
        total: number;
        modelDirectory?: string;
      }
    | { kind: "failed"; modelDirectory?: string }
    | {
        kind: "rejected";
        reason: string;
        requiredBytes: number;
        availableBytes: number;
        limitBytes: number;
        modelDirectory?: string;
      }
    | { kind: "not_present" };

  type ModelCardInfo = {
    family: string;
    quantization: string;
    baseModel: string;
    capabilities: string[];
    storageSize: number;
    nLayers: number;
    supportsTensor: boolean;
  };

  type ModelRow = {
    modelId: string;
    prettyName: string | null;
    cells: Record<string, CellStatus>;
    shardMetadata: Record<string, unknown> | null;
    modelCard: ModelCardInfo | null;
  };

  type NodeColumn = {
    nodeId: string;
    label: string;
    diskAvailable?: number;
    diskTotal?: number;
    storageLimit?: number;
    storagePolicy?: "manual" | "auto-evict";
  };

  const data = $derived(topologyData());
  const downloadsData = $derived(downloads());
  const nodeDiskData = $derived(nodeDisk());
  const storageConfigData = $derived(nodeStorageConfig());

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

  function getNodeUsedStorage(nodeId: string): number {
    const nodeDownloads = downloadsData?.[nodeId];
    if (!nodeDownloads || !Array.isArray(nodeDownloads)) return 0;
    let total = 0;
    for (const entry of nodeDownloads) {
      const tagged = getDownloadTag(entry);
      if (!tagged) continue;
      const [tag, payload] = tagged;
      if (tag === "ModelReady") {
        total += getBytes(payload.total);
      } else if (tag === "ModelDownloading") {
        const prog = (payload.download_progress ?? payload.downloadProgress) as
          | Record<string, unknown>
          | undefined;
        if (prog) total += getBytes(prog.downloaded);
      }
    }
    return total;
  }

  function storageBarColor(percent: number): string {
    if (percent >= 90) return "bg-red-500";
    if (percent >= 70) return "bg-yellow-500";
    return "bg-green-500";
  }

  const CELL_PRIORITY: Record<CellStatus["kind"], number> = {
    completed: 5,
    downloading: 4,
    pending: 3,
    rejected: 2,
    failed: 1,
    not_present: 0,
  };

  function shouldUpgradeCell(
    existing: CellStatus,
    candidate: CellStatus,
  ): boolean {
    return CELL_PRIORITY[candidate.kind] > CELL_PRIORITY[existing.kind];
  }

  function extractModelCard(payload: Record<string, unknown>): {
    prettyName: string | null;
    card: ModelCardInfo | null;
  } {
    const shardMetadata = payload.shard_metadata ?? payload.shardMetadata;
    if (!shardMetadata || typeof shardMetadata !== "object")
      return { prettyName: null, card: null };
    const shardObj = shardMetadata as Record<string, unknown>;
    const shardKeys = Object.keys(shardObj);
    if (shardKeys.length !== 1) return { prettyName: null, card: null };
    const shardData = shardObj[shardKeys[0]] as Record<string, unknown>;
    const modelMeta = shardData?.model_card ?? shardData?.modelCard;
    if (!modelMeta || typeof modelMeta !== "object")
      return { prettyName: null, card: null };
    const meta = modelMeta as Record<string, unknown>;

    const prettyName = (meta.prettyName as string) ?? null;

    const card: ModelCardInfo = {
      family: (meta.family as string) ?? "",
      quantization: (meta.quantization as string) ?? "",
      baseModel:
        (meta.base_model as string) ?? (meta.baseModel as string) ?? "",
      capabilities: Array.isArray(meta.capabilities)
        ? (meta.capabilities as string[])
        : [],
      storageSize: getBytes(meta.storage_size ?? meta.storageSize),
      nLayers: (meta.n_layers as number) ?? (meta.nLayers as number) ?? 0,
      supportsTensor:
        (meta.supports_tensor as boolean) ??
        (meta.supportsTensor as boolean) ??
        false,
    };

    return { prettyName, card };
  }

  let modelRows = $state<ModelRow[]>([]);
  let nodeColumns = $state<NodeColumn[]>([]);
  let infoRow = $state<ModelRow | null>(null);

  let storageConfigNode = $state<NodeColumn | null>(null);
  let configMaxGb = $state<number | null>(null);
  let configNoLimit = $state(true);
  let configPolicy = $state<"manual" | "auto-evict">("manual");
  let configSaving = $state(false);
  let configApplyAll = $state(false);
  let configDiskTotalGb = $derived(
    storageConfigNode
      ? Math.round((storageConfigNode.diskTotal ?? 0) / 1024 ** 3)
      : 0,
  );

  function openStorageConfig(col: NodeColumn) {
    storageConfigNode = col;
    if (col.storageLimit != null) {
      configNoLimit = false;
      configMaxGb = Math.round(col.storageLimit / 1024 ** 3);
    } else {
      configNoLimit = true;
      configMaxGb = null;
    }
    configPolicy = col.storagePolicy ?? "manual";
    configApplyAll = false;
  }

  async function saveStorageConfig() {
    if (!storageConfigNode) return;
    configSaving = true;
    try {
      const maxGb = configNoLimit ? null : configMaxGb;
      const nodeIds = configApplyAll ? null : [storageConfigNode.nodeId];
      await setStorageConfig(nodeIds, maxGb, configPolicy);
      storageConfigNode = null;
      refreshState();
    } catch (error) {
      addToast({
        type: "error",
        message: `Failed to save storage config: ${error instanceof Error ? error.message : String(error)}`,
      });
    } finally {
      configSaving = false;
    }
  }

  $effect(() => {
    try {
      if (!downloadsData || Object.keys(downloadsData).length === 0) {
        modelRows = [];
        nodeColumns = [];
        return;
      }

      const allNodeIds = Object.keys(downloadsData);
      const columns: NodeColumn[] = allNodeIds.map((nodeId) => {
        const diskInfo = nodeDiskData?.[nodeId];
        const storageConfig = storageConfigData?.[nodeId];
        return {
          nodeId,
          label: getNodeLabel(nodeId),
          diskAvailable: diskInfo?.available?.inBytes,
          diskTotal: diskInfo?.total?.inBytes,
          storageLimit: storageConfig?.maxStorage?.inBytes ?? undefined,
          storagePolicy: storageConfig?.storagePolicy,
        };
      });

      const rowMap = new Map<string, ModelRow>();

      for (const [nodeId, nodeDownloads] of Object.entries(downloadsData)) {
        const entries = Array.isArray(nodeDownloads)
          ? nodeDownloads
          : nodeDownloads && typeof nodeDownloads === "object"
            ? Object.values(nodeDownloads as Record<string, unknown>)
            : [];

        for (const entry of entries) {
          const tagged = getDownloadTag(entry);
          if (!tagged) continue;
          const [tag, payload] = tagged;

          const modelId =
            extractModelIdFromDownload(payload) ?? "unknown-model";
          const { prettyName, card } = extractModelCard(payload);

          if (!rowMap.has(modelId)) {
            rowMap.set(modelId, {
              modelId,
              prettyName,
              cells: {},
              shardMetadata: extractShardMetadata(payload),
              modelCard: card,
            });
          }
          const row = rowMap.get(modelId)!;
          if (prettyName && !row.prettyName) row.prettyName = prettyName;
          if (!row.shardMetadata)
            row.shardMetadata = extractShardMetadata(payload);
          if (!row.modelCard && card) row.modelCard = card;

          const modelDirectory =
            ((payload.model_directory ?? payload.modelDirectory) as string) ||
            undefined;
          let cell: CellStatus;
          if (tag === "ModelReady") {
            const totalBytes = getBytes(payload.total);
            cell = {
              kind: "completed",
              totalBytes,
              modelDirectory,
            };
          } else if (tag === "ModelDownloading") {
            const rawProgress =
              payload.download_progress ?? payload.downloadProgress ?? {};
            const prog = rawProgress as Record<string, unknown>;
            const totalBytes = getBytes(prog.total ?? payload.total);
            const downloadedBytes = getBytes(prog.downloaded);
            const speed = (prog.speed as number) ?? 0;
            const etaMs =
              (prog.eta_ms as number) ?? (prog.etaMs as number) ?? 0;
            const percentage =
              totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0;
            cell = {
              kind: "downloading",
              percentage: clampPercent(percentage),
              downloadedBytes,
              totalBytes,
              speed,
              etaMs,
              modelDirectory,
            };
          } else if (tag === "ModelRejected") {
            cell = {
              kind: "rejected",
              reason: (payload.reason as string) ?? "Storage limit exceeded",
              requiredBytes: getBytes(payload.required),
              availableBytes: getBytes(payload.available),
              limitBytes: getBytes(payload.limit),
              modelDirectory,
            };
          } else if (tag === "ModelDownloadFailed") {
            cell = { kind: "failed", modelDirectory };
          } else {
            const downloaded = getBytes(
              payload.downloaded ??
                payload.downloaded_bytes ??
                payload.downloadedBytes,
            );
            const total = getBytes(
              payload.total ?? payload.total_bytes ?? payload.totalBytes,
            );
            cell = {
              kind: "pending",
              downloaded,
              total,
              modelDirectory,
            };
          }

          const existing = row.cells[nodeId];
          if (!existing || shouldUpgradeCell(existing, cell)) {
            row.cells[nodeId] = cell;
          }
        }
      }

      function rowSortKey(row: ModelRow): number {
        // in progress (4) -> completed (3) -> paused (2) -> not started (1) -> not present (0)
        let best = 0;
        for (const cell of Object.values(row.cells)) {
          let score = 0;
          if (cell.kind === "downloading") score = 4;
          else if (cell.kind === "completed") score = 3;
          else if (cell.kind === "pending" && cell.downloaded > 0)
            score = 2; // paused
          else if (cell.kind === "pending" || cell.kind === "failed") score = 1; // not started
          if (score > best) best = score;
        }
        return best;
      }

      function totalCompletedBytes(row: ModelRow): number {
        let total = 0;
        for (const cell of Object.values(row.cells)) {
          if (cell.kind === "completed") total += cell.totalBytes;
        }
        return total;
      }

      const rows = Array.from(rowMap.values()).sort((a, b) => {
        const aPriority = rowSortKey(a);
        const bPriority = rowSortKey(b);
        if (aPriority !== bPriority) return bPriority - aPriority;
        // Within completed or paused, sort by biggest size first
        if (aPriority === 3 && bPriority === 3) {
          const sizeDiff = totalCompletedBytes(b) - totalCompletedBytes(a);
          if (sizeDiff !== 0) return sizeDiff;
        }
        if (aPriority === 2 && bPriority === 2) {
          const aSize = Math.max(
            ...Object.values(a.cells).map((c) =>
              c.kind === "pending" ? c.total : 0,
            ),
          );
          const bSize = Math.max(
            ...Object.values(b.cells).map((c) =>
              c.kind === "pending" ? c.total : 0,
            ),
          );
          if (aSize !== bSize) return bSize - aSize;
        }
        return a.modelId.localeCompare(b.modelId);
      });

      modelRows = rows;
      nodeColumns = columns;
    } catch (err) {
      console.error("Parse downloads error", err);
      modelRows = [];
      nodeColumns = [];
    }
  });

  const hasDownloads = $derived(modelRows.length > 0);
  const lastUpdateTs = $derived(lastUpdateStore());
  const downloadKeys = $derived(Object.keys(downloadsData || {}));

  onMount(() => {
    refreshState();
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
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 overflow-x-auto"
      >
        <table class="w-full text-left font-mono text-xs">
          <thead>
            <tr class="border-b border-exo-medium-gray/30">
              <th
                class="sticky left-0 z-10 bg-exo-black px-4 py-3 text-[11px] uppercase tracking-wider text-exo-yellow font-medium whitespace-nowrap border-r border-exo-medium-gray/20"
              >
                Model
              </th>
              {#each nodeColumns as col}
                {@const usedStorage = getNodeUsedStorage(col.nodeId)}
                {@const storageMax = col.storageLimit ?? col.diskTotal ?? 0}
                {@const storagePercent =
                  storageMax > 0
                    ? Math.min(100, (usedStorage / storageMax) * 100)
                    : 0}
                <th
                  class="px-4 py-3 text-[11px] uppercase tracking-wider text-exo-light-gray font-medium text-center whitespace-nowrap min-w-[120px]"
                >
                  <div class="flex items-center justify-center gap-1">
                    <span>{col.label}</span>
                    <button
                      type="button"
                      class="p-0.5 rounded hover:bg-white/10 transition-colors"
                      onclick={() => openStorageConfig(col)}
                      title="Storage settings"
                      aria-label="Storage settings for {col.label}"
                    >
                      <svg
                        class="w-3.5 h-3.5 text-white/40 hover:text-exo-yellow transition-colors"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                      >
                        <path
                          fill-rule="evenodd"
                          d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z"
                          clip-rule="evenodd"
                        />
                      </svg>
                    </button>
                  </div>
                  {#if storageMax > 0}
                    <div class="text-[9px] normal-case tracking-normal mt-1">
                      <div
                        class="w-full h-1.5 bg-white/10 rounded-full overflow-hidden"
                      >
                        <div
                          class="h-full rounded-full transition-all duration-300 {storageBarColor(
                            storagePercent,
                          )}"
                          style="width: {storagePercent.toFixed(1)}%"
                        ></div>
                      </div>
                      <div class="text-white/60 mt-0.5">
                        {formatBytes(usedStorage)} / {formatBytes(storageMax)}
                      </div>
                    </div>
                  {/if}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each modelRows as row}
              <tr
                class="group border-b border-exo-medium-gray/20 hover:bg-exo-medium-gray/10 transition-colors"
              >
                <td
                  class="sticky left-0 z-10 bg-exo-dark-gray group-hover:bg-[oklch(0.18_0_0)] transition-colors px-4 py-3 whitespace-nowrap border-r border-exo-medium-gray/20"
                >
                  <div class="flex items-center gap-2">
                    <div class="min-w-0">
                      <div class="text-white text-xs" title={row.modelId}>
                        {row.prettyName ?? row.modelId}
                      </div>
                      {#if row.prettyName}
                        <div
                          class="text-[10px] text-white/60"
                          title={row.modelId}
                        >
                          {row.modelId}
                        </div>
                      {/if}
                    </div>
                    <button
                      type="button"
                      class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0 opacity-60 group-hover:opacity-100"
                      onclick={() => (infoRow = row)}
                      title="View model details"
                    >
                      <svg
                        class="w-4 h-4 text-white/60 hover:text-white/80"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path
                          d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"
                        />
                      </svg>
                    </button>
                  </div>
                </td>

                {#each nodeColumns as col}
                  {@const cell = row.cells[col.nodeId] ?? {
                    kind: "not_present" as const,
                  }}
                  <td class="px-4 py-3 text-center align-middle">
                    {#if cell.kind === "completed"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title="Completed ({formatBytes(cell.totalBytes)})"
                      >
                        <svg
                          class="w-7 h-7 text-green-400"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fill-rule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clip-rule="evenodd"
                          ></path>
                        </svg>
                        <span class="text-xs text-white/70"
                          >{formatBytes(cell.totalBytes)}</span
                        >
                        <button
                          type="button"
                          class="text-white/50 hover:text-red-400 transition-colors mt-0.5 cursor-pointer"
                          onclick={() =>
                            deleteDownload(col.nodeId, row.modelId)}
                          title="Delete from this node"
                        >
                          <svg
                            class="w-5 h-5"
                            viewBox="0 0 20 20"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                          >
                            <path
                              d="M4 6h12M8 6V4h4v2m1 0v10a1 1 0 01-1 1H8a1 1 0 01-1-1V6h6"
                              stroke-linecap="round"
                              stroke-linejoin="round"
                            ></path>
                          </svg>
                        </button>
                      </div>
                    {:else if cell.kind === "downloading"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title="{formatBytes(
                          cell.downloadedBytes,
                        )} / {formatBytes(cell.totalBytes)} - {formatSpeed(
                          cell.speed,
                        )} - ETA {formatEta(cell.etaMs)}"
                      >
                        <span class="text-exo-yellow text-sm font-medium"
                          >{clampPercent(cell.percentage).toFixed(1)}%</span
                        >
                        <div
                          class="w-16 h-2 bg-exo-black/60 rounded-sm overflow-hidden"
                        >
                          <div
                            class="h-full bg-gradient-to-r from-exo-yellow to-exo-yellow/70 transition-all duration-300"
                            style="width: {clampPercent(
                              cell.percentage,
                            ).toFixed(1)}%"
                          ></div>
                        </div>
                        <span class="text-[10px] text-white/70"
                          >{formatSpeed(cell.speed)}</span
                        >
                      </div>
                    {:else if cell.kind === "pending"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title={cell.downloaded > 0
                          ? `${formatBytes(cell.downloaded)} / ${formatBytes(cell.total)} downloaded (paused)`
                          : "Download pending"}
                      >
                        {#if cell.downloaded > 0 && cell.total > 0}
                          <span class="text-white/70 text-xs"
                            >{formatBytes(cell.downloaded)} / {formatBytes(
                              cell.total,
                            )}</span
                          >
                          <div
                            class="w-full h-1.5 bg-white/10 rounded-full overflow-hidden"
                          >
                            <div
                              class="h-full bg-exo-light-gray/40 rounded-full"
                              style="width: {(
                                (cell.downloaded / cell.total) *
                                100
                              ).toFixed(1)}%"
                            ></div>
                          </div>
                          {#if row.shardMetadata}
                            <button
                              type="button"
                              class="text-white/50 hover:text-exo-yellow transition-colors cursor-pointer"
                              onclick={() =>
                                startDownload(col.nodeId, row.shardMetadata!)}
                              title="Resume download on this node"
                            >
                              <svg
                                class="w-5 h-5"
                                viewBox="0 0 20 20"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                              >
                                <path
                                  d="M10 3v10m0 0l-3-3m3 3l3-3M3 17h14"
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                ></path>
                              </svg>
                            </button>
                          {:else}
                            <span class="text-white/50 text-[10px]">paused</span
                            >
                          {/if}
                        {:else if row.shardMetadata}
                          <button
                            type="button"
                            class="text-white/50 hover:text-exo-yellow transition-colors cursor-pointer"
                            onclick={() =>
                              startDownload(col.nodeId, row.shardMetadata!)}
                            title="Start download on this node"
                          >
                            <svg
                              class="w-6 h-6"
                              viewBox="0 0 20 20"
                              fill="none"
                              stroke="currentColor"
                              stroke-width="2"
                            >
                              <path
                                d="M10 3v10m0 0l-3-3m3 3l3-3M3 17h14"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                              ></path>
                            </svg>
                          </button>
                        {:else}
                          <span class="text-white/40 text-sm">...</span>
                        {/if}
                      </div>
                    {:else if cell.kind === "rejected"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title={cell.reason}
                      >
                        <svg
                          class="w-7 h-7 text-orange-400"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fill-rule="evenodd"
                            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                            clip-rule="evenodd"
                          ></path>
                        </svg>
                        <span class="text-[10px] text-orange-400/80"
                          >Need {formatBytes(cell.requiredBytes)}</span
                        >
                        <span class="text-[10px] text-white/50"
                          >{formatBytes(cell.availableBytes)} avail</span
                        >
                        {#if row.shardMetadata}
                          <button
                            type="button"
                            class="text-white/50 hover:text-exo-yellow transition-colors cursor-pointer"
                            onclick={() =>
                              startDownload(col.nodeId, row.shardMetadata!)}
                            title="Retry download on this node"
                          >
                            <svg
                              class="w-5 h-5"
                              viewBox="0 0 20 20"
                              fill="none"
                              stroke="currentColor"
                              stroke-width="2"
                            >
                              <path
                                d="M10 3v10m0 0l-3-3m3 3l3-3M3 17h14"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                              ></path>
                            </svg>
                          </button>
                        {/if}
                      </div>
                    {:else if cell.kind === "failed"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title="Download failed"
                      >
                        <svg
                          class="w-7 h-7 text-red-400"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fill-rule="evenodd"
                            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                            clip-rule="evenodd"
                          ></path>
                        </svg>
                        {#if row.shardMetadata}
                          <button
                            type="button"
                            class="text-white/50 hover:text-exo-yellow transition-colors cursor-pointer"
                            onclick={() =>
                              startDownload(col.nodeId, row.shardMetadata!)}
                            title="Retry download on this node"
                          >
                            <svg
                              class="w-5 h-5"
                              viewBox="0 0 20 20"
                              fill="none"
                              stroke="currentColor"
                              stroke-width="2"
                            >
                              <path
                                d="M10 3v10m0 0l-3-3m3 3l3-3M3 17h14"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                              ></path>
                            </svg>
                          </button>
                        {/if}
                      </div>
                    {:else}
                      <div
                        class="flex flex-col items-center"
                        title="Not on this node"
                      >
                        <span class="text-exo-medium-gray text-lg leading-none"
                          >--</span
                        >
                        {#if row.shardMetadata}
                          <button
                            type="button"
                            class="text-white/50 hover:text-exo-yellow transition-colors mt-0.5 opacity-0 group-hover:opacity-100 cursor-pointer"
                            onclick={() =>
                              startDownload(col.nodeId, row.shardMetadata!)}
                            title="Download to this node"
                          >
                            <svg
                              class="w-5 h-5"
                              viewBox="0 0 20 20"
                              fill="none"
                              stroke="currentColor"
                              stroke-width="2"
                            >
                              <path
                                d="M10 3v10m0 0l-3-3m3 3l3-3M3 17h14"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                              ></path>
                            </svg>
                          </button>
                        {/if}
                      </div>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</div>

<!-- Info modal -->
{#if infoRow}
  <div
    class="fixed inset-0 z-[60] bg-black/60"
    transition:fade={{ duration: 150 }}
    onclick={() => (infoRow = null)}
    role="presentation"
  ></div>
  <div
    class="fixed z-[60] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(80vw,400px)] bg-exo-dark-gray border border-exo-yellow/10 rounded-lg shadow-2xl p-4"
    transition:fly={{ y: 10, duration: 200, easing: cubicOut }}
    role="dialog"
    aria-modal="true"
  >
    <div class="flex items-start justify-between mb-3">
      <h3 class="font-mono text-lg text-white">
        {infoRow.prettyName ?? infoRow.modelId}
      </h3>
      <button
        type="button"
        class="p-1 rounded hover:bg-white/10 transition-colors text-white/50"
        onclick={() => (infoRow = null)}
        title="Close model details"
        aria-label="Close info dialog"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
          />
        </svg>
      </button>
    </div>
    <div class="space-y-2 text-xs font-mono">
      <div class="flex items-center gap-2">
        <span class="text-white/40">Model ID:</span>
        <span class="text-white/70">{infoRow.modelId}</span>
      </div>
      {#if infoRow.modelCard}
        {#if infoRow.modelCard.family}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Family:</span>
            <span class="text-white/70">{infoRow.modelCard.family}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.baseModel}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Base model:</span>
            <span class="text-white/70">{infoRow.modelCard.baseModel}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.quantization}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Quantization:</span>
            <span class="text-white/70">{infoRow.modelCard.quantization}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.storageSize > 0}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Size:</span>
            <span class="text-white/70"
              >{formatBytes(infoRow.modelCard.storageSize)}</span
            >
          </div>
        {/if}
        {#if infoRow.modelCard.nLayers > 0}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Layers:</span>
            <span class="text-white/70">{infoRow.modelCard.nLayers}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.capabilities.length > 0}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Capabilities:</span>
            <span class="text-white/70"
              >{infoRow.modelCard.capabilities.join(", ")}</span
            >
          </div>
        {/if}
        <div class="flex items-center gap-2">
          <span class="text-white/40">Tensor parallelism:</span>
          <span class="text-white/70"
            >{infoRow.modelCard.supportsTensor ? "Yes" : "No"}</span
          >
        </div>
      {/if}

      <!-- Per-node download status -->
      {#if nodeColumns.filter((col) => (infoRow?.cells[col.nodeId]?.kind ?? "not_present") !== "not_present").length > 0}
        <div class="mt-3 pt-3 border-t border-exo-yellow/10">
          <div class="flex items-center gap-2 mb-1">
            <svg
              class="w-3.5 h-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path
                class="text-white/40"
                d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
              />
              <path class="text-green-400" d="m9 13 2 2 4-4" />
            </svg>
            <span class="text-white/40">On nodes:</span>
          </div>
          <div class="flex flex-col gap-1.5 mt-1">
            {#each nodeColumns as col}
              {@const cellStatus = infoRow?.cells[col.nodeId]}
              {#if cellStatus && cellStatus.kind !== "not_present"}
                <div class="flex flex-col gap-0.5">
                  <span
                    class="inline-block w-fit px-1.5 py-0.5 rounded text-[10px] {cellStatus.kind ===
                    'completed'
                      ? 'bg-green-500/10 text-green-400/80 border border-green-500/20'
                      : cellStatus.kind === 'downloading'
                        ? 'bg-exo-yellow/10 text-exo-yellow/80 border border-exo-yellow/20'
                        : cellStatus.kind === 'rejected'
                          ? 'bg-orange-500/10 text-orange-400/80 border border-orange-500/20'
                          : cellStatus.kind === 'failed'
                            ? 'bg-red-500/10 text-red-400/80 border border-red-500/20'
                            : 'bg-white/5 text-white/50 border border-white/10'}"
                  >
                    {col.label}
                    {#if cellStatus.kind === "downloading" && "percentage" in cellStatus}
                      ({clampPercent(cellStatus.percentage).toFixed(0)}%)
                    {/if}
                  </span>
                  {#if "modelDirectory" in cellStatus && cellStatus.modelDirectory}
                    <span
                      class="text-[9px] text-white/30 break-all pl-1"
                      title={cellStatus.modelDirectory}
                    >
                      {cellStatus.modelDirectory}
                    </span>
                  {/if}
                </div>
              {/if}
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

<!-- Storage config modal -->
{#if storageConfigNode}
  <div
    class="fixed inset-0 z-[60] bg-black/60"
    transition:fade={{ duration: 150 }}
    onclick={() => (storageConfigNode = null)}
    role="presentation"
  ></div>
  <div
    class="fixed z-[60] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(80vw,360px)] bg-exo-dark-gray border border-exo-yellow/10 rounded-lg shadow-2xl p-4"
    transition:fly={{ y: 10, duration: 200, easing: cubicOut }}
    role="dialog"
    aria-modal="true"
    onkeydown={(e) => {
      if (e.key === "Escape") storageConfigNode = null;
    }}
  >
    <div class="flex items-start justify-between mb-4">
      <h3 class="font-mono text-sm text-white">
        Storage — {configApplyAll ? "All nodes" : storageConfigNode.label}
      </h3>
      <button
        type="button"
        class="p-1 rounded hover:bg-white/10 transition-colors text-white/50"
        onclick={() => (storageConfigNode = null)}
        aria-label="Close storage settings"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
          />
        </svg>
      </button>
    </div>

    <div class="space-y-4">
      <!-- Apply to all nodes -->
      {#if nodeColumns.length > 1}
        <label class="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            bind:checked={configApplyAll}
            class="accent-exo-yellow w-4 h-4"
          />
          <span class="text-xs font-mono text-white/80">Apply to all nodes</span
          >
        </label>
      {/if}

      <!-- No limit checkbox -->
      <label class="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          bind:checked={configNoLimit}
          onchange={() => {
            if (!configNoLimit && configMaxGb == null) {
              configMaxGb = configDiskTotalGb || 50;
            }
          }}
          class="accent-exo-yellow w-4 h-4"
        />
        <span class="text-xs font-mono text-white/80">Unlimited storage</span>
      </label>

      <!-- Max storage slider -->
      <div class="space-y-1.5">
        <div class="flex items-baseline justify-between">
          <label
            class="text-[11px] font-mono text-white/50 uppercase tracking-wider"
            for="storage-max-gb"
          >
            Max storage
          </label>
          <span
            class="text-xs font-mono tabular-nums transition-opacity {configNoLimit
              ? 'opacity-30'
              : 'text-white'}"
          >
            {configMaxGb ?? 0} GB
          </span>
        </div>
        <input
          id="storage-max-gb"
          type="range"
          min="1"
          max={Math.max(configDiskTotalGb, configMaxGb ?? 1)}
          step="1"
          bind:value={configMaxGb}
          disabled={configNoLimit}
          class="slider w-full h-1.5 rounded-full appearance-none cursor-pointer
            disabled:opacity-30 disabled:cursor-not-allowed"
        />
        <div
          class="flex justify-between text-[10px] font-mono text-white/30 transition-opacity {configNoLimit
            ? 'opacity-30'
            : ''}"
        >
          <span>1 GB</span>
          <span>{Math.max(configDiskTotalGb, configMaxGb ?? 1)} GB</span>
        </div>
      </div>

      <!-- Policy selector -->
      <div class="space-y-1.5">
        <div
          class="text-[11px] font-mono text-white/50 uppercase tracking-wider"
        >
          Eviction policy
        </div>
        <div class="flex gap-1">
          <button
            type="button"
            class="flex-1 px-3 py-1.5 rounded text-xs font-mono transition-colors
              {configPolicy === 'manual'
              ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/40'
              : 'bg-exo-black/40 text-white/50 border border-exo-medium-gray/30 hover:text-white/70'}"
            onclick={() => (configPolicy = "manual")}
          >
            Manual
          </button>
          <button
            type="button"
            class="flex-1 px-3 py-1.5 rounded text-xs font-mono transition-colors
              {configPolicy === 'auto-evict'
              ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/40'
              : 'bg-exo-black/40 text-white/50 border border-exo-medium-gray/30 hover:text-white/70'}"
            onclick={() => (configPolicy = "auto-evict")}
          >
            Auto-evict
          </button>
        </div>
        <p class="text-[10px] text-white/40 font-mono">
          {#if configPolicy === "manual"}
            Downloads that exceed the limit are rejected. Delete models
            manually.
          {:else}
            Oldest unused models are automatically removed to make room.
          {/if}
        </p>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex justify-end gap-2 mt-5">
      <button
        type="button"
        class="px-3 py-1.5 rounded text-xs font-mono text-white/50 hover:text-white/70 transition-colors"
        onclick={() => (storageConfigNode = null)}
      >
        Cancel
      </button>
      <button
        type="button"
        class="px-3 py-1.5 rounded text-xs font-mono bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/40 hover:bg-exo-yellow/30 transition-colors disabled:opacity-50"
        onclick={saveStorageConfig}
        disabled={configSaving ||
          (!configNoLimit && (configMaxGb == null || configMaxGb <= 0))}
      >
        {configSaving ? "Saving..." : "Save"}
      </button>
    </div>
  </div>
{/if}

<style>
  table {
    min-width: max-content;
  }

  .slider {
    background: rgba(255, 255, 255, 0.1);
  }
  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #f5c518;
    cursor: pointer;
  }
  .slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: none;
    background: #f5c518;
    cursor: pointer;
  }
  .slider:disabled::-webkit-slider-thumb {
    cursor: not-allowed;
  }
  .slider:disabled::-moz-range-thumb {
    cursor: not-allowed;
  }
</style>

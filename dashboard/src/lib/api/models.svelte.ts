import { browser } from "$app/environment";
import {
  extractModelIdFromDownload,
  getDownloadTag,
  getShardMetadataForModel,
} from "$lib/utils/downloads";

export interface ModelEntry {
  id: string;
  name: string;
  family: string | null;
  quantization: string | null;
  baseModel: string | null;
  storageMb: number;
  contextLength: number | null;
  capabilities: string[];
  isCustom: boolean;
  supportsTensor: boolean;
}

export type NodeDownloadStatus = "completed" | "ongoing" | "pending" | "failed";

export interface NodeDownloadProgress {
  status: NodeDownloadStatus;
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  speedBps: number;
  etaMs: number;
  filesDone: number;
  filesTotal: number;
}

export interface NodeInfo {
  nodeId: string;
  shortId: string;
  friendlyName: string;
  isLocal: boolean;
  diskAvailableBytes: number;
  diskTotalBytes: number;
  ramAvailableBytes: number;
  ramTotalBytes: number;
}

export interface ModelsSnapshot {
  all: ModelEntry[];
  downloadedIds: Set<string>;
  runningModelIds: Set<string>;
  /** Per-model per-node download/launch status. */
  perModelNodes: Map<string, Map<string, NodeDownloadProgress>>;
  /** Set of node IDs where each running model has an active instance. */
  runningOnNodes: Map<string, Set<string>>;
  nodes: NodeInfo[];
  localNodeId: string | null;
  loaded: boolean;
}

interface RawModel {
  id: string;
  name?: string;
  family?: string | null;
  quantization?: string | null;
  base_model?: string | null;
  storage_size_megabytes?: number;
  context_length?: number | null;
  capabilities?: string[];
  is_custom?: boolean;
  supports_tensor?: boolean;
}

interface RawModelList {
  data?: RawModel[];
}

interface RawInstanceInner {
  shardAssignments?: {
    modelId?: string;
    nodeToRunner?: Record<string, string>;
  };
}

interface RawState {
  /** Tagged-union envelope: { MlxRingInstance: {...} }. */
  instances?: Record<string, Record<string, RawInstanceInner>>;
  downloads?: Record<string, unknown[]>;
  nodeDisk?: Record<
    string,
    {
      total?: { inBytes?: number };
      available?: { inBytes?: number };
    }
  >;
  nodeMemory?: Record<
    string,
    {
      ramTotal?: { inBytes?: number };
      ramAvailable?: { inBytes?: number };
    }
  >;
  nodeIdentities?: Record<
    string,
    {
      friendlyName?: string;
    }
  >;
}

const POLL_MS = 2000;

function adapt(m: RawModel): ModelEntry {
  return {
    id: m.id,
    name: m.name ?? m.id.split("/").pop() ?? m.id,
    family: m.family ?? null,
    quantization: m.quantization ?? null,
    baseModel: m.base_model ?? null,
    storageMb: m.storage_size_megabytes ?? 0,
    contextLength: m.context_length ?? null,
    capabilities: m.capabilities ?? [],
    isCustom: m.is_custom ?? false,
    supportsTensor: m.supports_tensor ?? false,
  };
}

function bytes(v: { inBytes?: number } | undefined): number {
  return v?.inBytes ?? 0;
}

function tagToStatus(tag: string): NodeDownloadStatus | null {
  if (tag === "DownloadCompleted") return "completed";
  if (tag === "DownloadOngoing") return "ongoing";
  if (tag === "DownloadFailed") return "failed";
  // DownloadPending in exo means "exo knows this model exists on this node but
  // no bytes have moved yet" — it's seeded for every catalog model, not a real
  // user-visible "queued" state. Treat as missing so the UI isn't drowned in
  // false-positive download badges.
  return null;
}

function progressFromPayload(
  tag: string,
  payload: Record<string, unknown>,
): NodeDownloadProgress {
  const status = tagToStatus(tag) ?? "pending";

  // DownloadCompleted has total directly.
  if (tag === "DownloadCompleted") {
    const total = bytes(
      (payload.total as { inBytes?: number } | undefined) ??
        (payload.totalBytes as { inBytes?: number } | undefined),
    );
    return {
      status,
      downloadedBytes: total,
      totalBytes: total,
      percent: 100,
      speedBps: 0,
      etaMs: 0,
      filesDone: 0,
      filesTotal: 0,
    };
  }

  // DownloadOngoing nests progress.
  const prog =
    (payload.downloadProgress as Record<string, unknown> | undefined) ??
    (payload.download_progress as Record<string, unknown> | undefined);
  if (prog) {
    const downloaded = bytes(prog.downloaded as { inBytes?: number } | undefined);
    const total = bytes(prog.total as { inBytes?: number } | undefined);
    return {
      status,
      downloadedBytes: downloaded,
      totalBytes: total,
      percent: total > 0 ? (downloaded / total) * 100 : 0,
      speedBps: (prog.speed as number) ?? 0,
      etaMs:
        (prog.etaMs as number) ?? (prog.eta_ms as number) ?? 0,
      filesDone:
        (prog.completedFiles as number) ??
        (prog.completed_files as number) ??
        0,
      filesTotal:
        (prog.totalFiles as number) ??
        (prog.total_files as number) ??
        0,
    };
  }

  // DownloadPending — has downloaded/total at top level.
  const downloaded = bytes(
    payload.downloaded as { inBytes?: number } | undefined,
  );
  const total = bytes(payload.total as { inBytes?: number } | undefined);
  return {
    status,
    downloadedBytes: downloaded,
    totalBytes: total,
    percent: total > 0 ? (downloaded / total) * 100 : 0,
    speedBps: 0,
    etaMs: 0,
    filesDone: 0,
    filesTotal: 0,
  };
}

function statusRank(s: NodeDownloadStatus): number {
  switch (s) {
    case "completed":
      return 4;
    case "ongoing":
      return 3;
    case "pending":
      return 2;
    case "failed":
      return 1;
  }
}

function createModelsStore() {
  let snapshot = $state<ModelsSnapshot>({
    all: [],
    downloadedIds: new Set(),
    runningModelIds: new Set(),
    perModelNodes: new Map(),
    runningOnNodes: new Map(),
    nodes: [],
    localNodeId: null,
    loaded: false,
  });
  let started = false;

  async function fetchOnce() {
    try {
      const [allRes, stateRes, idRes] = await Promise.all([
        fetch("/v1/models"),
        fetch("/state"),
        fetch("/node_id"),
      ]);
      if (!allRes.ok) throw new Error(`models ${allRes.status}`);
      const allList = (await allRes.json()) as RawModelList;
      const all = (allList.data ?? []).map(adapt);

      let localNodeId: string | null = null;
      if (idRes.ok) {
        const raw = await idRes.text();
        try {
          localNodeId = JSON.parse(raw) as string;
        } catch {
          localNodeId = raw.replace(/^"|"$/g, "");
        }
      }

      const runningModelIds = new Set<string>();
      const runningOnNodes = new Map<string, Set<string>>();
      const perModelNodes = new Map<
        string,
        Map<string, NodeDownloadProgress>
      >();
      const downloadedIds = new Set<string>();
      const nodes: NodeInfo[] = [];

      if (stateRes.ok) {
        const state = (await stateRes.json()) as RawState;

        // Running instances → running model + which nodes hold them.
        // Each instance is a discriminated-union envelope: { MlxRingInstance: {...} }.
        for (const env of Object.values(state.instances ?? {})) {
          const keys = Object.keys(env);
          if (keys.length !== 1) continue;
          const inner = env[keys[0]!];
          const mid = inner?.shardAssignments?.modelId;
          if (!mid) continue;
          runningModelIds.add(mid);
          const set = runningOnNodes.get(mid) ?? new Set<string>();
          for (const nodeId of Object.keys(
            inner.shardAssignments?.nodeToRunner ?? {},
          )) {
            set.add(nodeId);
          }
          runningOnNodes.set(mid, set);
        }

        // Per-node downloads — extract status per (model, node).
        const downloads = state.downloads ?? {};
        for (const [nodeId, list] of Object.entries(downloads)) {
          if (!Array.isArray(list)) continue;
          for (const entry of list) {
            const tagged = getDownloadTag(entry);
            if (!tagged) continue;
            const [tag, payload] = tagged;
            const modelId = extractModelIdFromDownload(payload);
            if (!modelId) continue;
            const status = tagToStatus(tag);
            if (!status) continue;

            const inner =
              perModelNodes.get(modelId) ??
              new Map<string, NodeDownloadProgress>();
            const next = progressFromPayload(tag, payload);
            const existing = inner.get(nodeId);
            // Prefer the highest-rank status when multiple shards report.
            if (!existing || statusRank(next.status) > statusRank(existing.status)) {
              inner.set(nodeId, next);
            }
            perModelNodes.set(modelId, inner);

            if (status === "completed") downloadedIds.add(modelId);
          }
        }

        // Per-node info (disk, ram, friendly name).
        const ids = new Set<string>([
          ...Object.keys(state.nodeDisk ?? {}),
          ...Object.keys(state.nodeMemory ?? {}),
          ...Object.keys(state.nodeIdentities ?? {}),
        ]);
        for (const nodeId of ids) {
          const disk = state.nodeDisk?.[nodeId];
          const mem = state.nodeMemory?.[nodeId];
          const ident = state.nodeIdentities?.[nodeId];
          nodes.push({
            nodeId,
            shortId: nodeId.slice(-6),
            friendlyName: ident?.friendlyName ?? nodeId.slice(-8),
            isLocal: nodeId === localNodeId,
            diskAvailableBytes: bytes(disk?.available),
            diskTotalBytes: bytes(disk?.total),
            ramAvailableBytes: bytes(mem?.ramAvailable),
            ramTotalBytes: bytes(mem?.ramTotal),
          });
        }
        nodes.sort((a, b) => {
          if (a.isLocal !== b.isLocal) return a.isLocal ? -1 : 1;
          return a.friendlyName.localeCompare(b.friendlyName);
        });
      }

      snapshot = {
        all,
        downloadedIds,
        runningModelIds,
        perModelNodes,
        runningOnNodes,
        nodes,
        localNodeId,
        loaded: true,
      };
    } catch (_err) {
      // Keep last good snapshot.
    }
  }

  /** Look up the shard_metadata blob for a given model from any download record. */
  async function getShardMetadata(
    modelId: string,
  ): Promise<Record<string, unknown> | null> {
    try {
      const res = await fetch("/state");
      if (!res.ok) return null;
      const state = (await res.json()) as RawState;
      return getShardMetadataForModel(state.downloads ?? {}, modelId);
    } catch {
      return null;
    }
  }

  function start() {
    if (started || !browser) return;
    started = true;
    fetchOnce();
    setInterval(fetchOnce, POLL_MS);
  }

  if (browser) start();

  return {
    get value() {
      return snapshot;
    },
    refresh: fetchOnce,
    getShardMetadata,
  };
}

export const models = createModelsStore();

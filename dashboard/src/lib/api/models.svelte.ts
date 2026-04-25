import { browser } from "$app/environment";

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

export interface ModelsSnapshot {
  all: ModelEntry[];
  downloadedIds: Set<string>;
  runningModelIds: Set<string>;
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

const POLL_MS = 4000;

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

function createModelsStore() {
  let snapshot = $state<ModelsSnapshot>({
    all: [],
    downloadedIds: new Set(),
    runningModelIds: new Set(),
    loaded: false,
  });
  let started = false;

  async function fetchOnce() {
    try {
      const [allRes, dlRes, stateRes] = await Promise.all([
        fetch("/v1/models"),
        fetch("/v1/models?status=downloaded"),
        fetch("/state"),
      ]);
      if (!allRes.ok) throw new Error(`models ${allRes.status}`);
      const allList = (await allRes.json()) as RawModelList;
      const dlList = dlRes.ok ? ((await dlRes.json()) as RawModelList) : { data: [] };
      const downloadedIds = new Set((dlList.data ?? []).map((m) => m.id));
      const all = (allList.data ?? []).map(adapt);

      const runningModelIds = new Set<string>();
      if (stateRes.ok) {
        const state = (await stateRes.json()) as {
          instances?: Record<string, {
            shardAssignments?: { modelId?: string };
          }>;
        };
        for (const inst of Object.values(state.instances ?? {})) {
          const mid = inst.shardAssignments?.modelId;
          if (mid) runningModelIds.add(mid);
        }
      }
      snapshot = { all, downloadedIds, runningModelIds, loaded: true };
    } catch (_err) {
      // Keep last good snapshot.
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
  };
}

export const models = createModelsStore();

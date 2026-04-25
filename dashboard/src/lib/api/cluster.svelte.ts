import { browser } from "$app/environment";

export interface ClusterNode {
  id: string;
  shortId: string;
  friendlyName: string;
  chip: string;
  totalMemoryGB: number;
  usedMemoryGB: number;
  memoryFraction: number; // 0..1
  tempC?: number;
  isLocal: boolean;
  isLeader: boolean;
}

export interface LoadedInstance {
  id: string;
  modelId: string;
  shortId: string;
  nodes: string[]; // node ids
  shardCount: number;
}

export interface ClusterSnapshot {
  nodes: ClusterNode[];
  instances: LoadedInstance[];
  localNodeId: string | null;
  leaderNodeId: string | null;
}

const POLL_MS = 2000;

interface RawByteValue {
  inBytes?: number;
}

interface RawIdentity {
  friendlyName?: string;
  chipId?: string;
  modelId?: string;
}

interface RawMemory {
  ramTotal?: RawByteValue;
  ramAvailable?: RawByteValue;
}

interface RawSystem {
  temp?: number;
  gpuTempAvg?: number;
}

interface RawShardAssignments {
  modelId?: string;
  nodeToRunner?: Record<string, string>;
  runnerToShard?: Record<string, unknown>;
}

interface RawInstanceInner {
  shardAssignments?: RawShardAssignments;
}

/** Instances arrive as a tagged-union envelope: { MlxRingInstance: {...} }. */
type RawInstance = Record<string, RawInstanceInner>;

interface RawState {
  nodeIdentities?: Record<string, RawIdentity>;
  nodeMemory?: Record<string, RawMemory>;
  nodeSystem?: Record<string, RawSystem>;
  instances?: Record<string, RawInstance>;
}

function unwrapInstance(env: RawInstance): RawInstanceInner | null {
  const keys = Object.keys(env);
  if (keys.length !== 1) return null;
  const inner = env[keys[0]!];
  return (inner as RawInstanceInner) ?? null;
}

const GB = 1024 * 1024 * 1024;

function shortenNodeId(id: string): string {
  if (!id) return "—";
  return id.length <= 8 ? id : id.slice(0, 6);
}

function transform(raw: RawState, localNodeId: string | null): ClusterSnapshot {
  const identities = raw.nodeIdentities ?? {};
  const memory = raw.nodeMemory ?? {};
  const sys = raw.nodeSystem ?? {};
  const rawInstances = raw.instances ?? {};

  const nodeIds = Object.keys(identities);
  const leaderNodeId = localNodeId; // placeholder until election state is wired

  const nodes: ClusterNode[] = nodeIds.map((id, idx) => {
    const ident = identities[id] ?? {};
    const mem = memory[id] ?? {};
    const sysInfo = sys[id] ?? {};
    const totalBytes = mem.ramTotal?.inBytes ?? 0;
    const availableBytes = mem.ramAvailable?.inBytes ?? 0;
    const usedBytes = Math.max(0, totalBytes - availableBytes);
    const totalGB = totalBytes / GB;
    const usedGB = usedBytes / GB;
    const tempC = sysInfo.temp ?? sysInfo.gpuTempAvg;
    return {
      id,
      shortId: shortenNodeId(id),
      friendlyName: ident.friendlyName ?? `node-${idx}`,
      chip: ident.chipId ?? ident.modelId ?? "Apple Silicon",
      totalMemoryGB: totalGB,
      usedMemoryGB: usedGB,
      memoryFraction: totalGB > 0 ? Math.min(1, usedGB / totalGB) : 0,
      tempC,
      isLocal: id === localNodeId,
      isLeader: id === leaderNodeId,
    };
  });

  const instances: LoadedInstance[] = Object.entries(rawInstances).map(([id, env]) => {
    const inner = unwrapInstance(env);
    const assignments = inner?.shardAssignments ?? {};
    const nodes = Object.keys(assignments.nodeToRunner ?? {});
    const shardCount = Object.keys(assignments.runnerToShard ?? {}).length || nodes.length;
    return {
      id,
      modelId: assignments.modelId ?? "(unknown)",
      shortId: id.slice(0, 8),
      nodes,
      shardCount,
    };
  });

  return { nodes, instances, localNodeId, leaderNodeId };
}

function createClusterStore() {
  let snapshot = $state<ClusterSnapshot>({ nodes: [], instances: [], localNodeId: null, leaderNodeId: null });
  let lastError = $state<string | null>(null);
  let started = false;
  let localNodeId: string | null = null;

  async function fetchLocalNodeId() {
    try {
      const res = await fetch("/node_id");
      if (!res.ok) return;
      const text = await res.text();
      try {
        localNodeId = JSON.parse(text);
      } catch {
        localNodeId = text;
      }
    } catch {
      // ignore
    }
  }

  async function fetchOnce() {
    try {
      const res = await fetch("/state");
      if (!res.ok) throw new Error(`status ${res.status}`);
      const raw = (await res.json()) as RawState;
      snapshot = transform(raw, localNodeId);
      lastError = null;
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    }
  }

  async function start() {
    if (started || !browser) return;
    started = true;
    await fetchLocalNodeId();
    fetchOnce();
    setInterval(fetchOnce, POLL_MS);
  }

  if (browser) start();

  return {
    get value() {
      return snapshot;
    },
    get lastError() {
      return lastError;
    },
  };
}

export const cluster = createClusterStore();

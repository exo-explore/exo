/**
 * Shared utilities for parsing and querying download state.
 *
 * The download state from `/state` is shaped as:
 *   Record<NodeId, Array<TaggedDownloadEntry>>
 *
 * Each entry is a tagged union object like:
 *   { "DownloadCompleted": { shard_metadata: { "PipelineShardMetadata": { model_card: { model_id: "..." }, ... } }, ... } }
 */

/** Unwrap one level of tagged-union envelope, returning [tag, payload]. */
function unwrapTagged(
  obj: Record<string, unknown>,
): [string, Record<string, unknown>] | null {
  const keys = Object.keys(obj);
  if (keys.length !== 1) return null;
  const tag = keys[0];
  const payload = obj[tag];
  if (!payload || typeof payload !== "object") return null;
  return [tag, payload as Record<string, unknown>];
}

/** Extract the model ID string from a download entry's nested shard_metadata. */
export function extractModelIdFromDownload(
  downloadPayload: Record<string, unknown>,
): string | null {
  const shardMetadata =
    downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
  if (!shardMetadata || typeof shardMetadata !== "object") return null;

  const unwrapped = unwrapTagged(shardMetadata as Record<string, unknown>);
  if (!unwrapped) return null;
  const [, shardData] = unwrapped;

  const modelMeta = shardData.model_card ?? shardData.modelCard;
  if (!modelMeta || typeof modelMeta !== "object") return null;

  const meta = modelMeta as Record<string, unknown>;
  return (meta.model_id as string) ?? (meta.modelId as string) ?? null;
}

/** Extract the shard_metadata object from a download entry payload. */
export function extractShardMetadata(
  downloadPayload: Record<string, unknown>,
): Record<string, unknown> | null {
  const shardMetadata =
    downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
  if (!shardMetadata || typeof shardMetadata !== "object") return null;
  return shardMetadata as Record<string, unknown>;
}

/** Get the download tag (DownloadCompleted, DownloadOngoing, etc.) from a wrapped entry. */
export function getDownloadTag(
  entry: unknown,
): [string, Record<string, unknown>] | null {
  if (!entry || typeof entry !== "object") return null;
  return unwrapTagged(entry as Record<string, unknown>);
}

/**
 * Iterate over all download entries for a given node, yielding [tag, payload, modelId].
 */
function* iterNodeDownloads(
  nodeDownloads: unknown[],
): Generator<[string, Record<string, unknown>, string]> {
  for (const entry of nodeDownloads) {
    const tagged = getDownloadTag(entry);
    if (!tagged) continue;
    const [tag, payload] = tagged;
    const modelId = extractModelIdFromDownload(payload);
    if (!modelId) continue;
    yield [tag, payload, modelId];
  }
}

/** Check if a specific model is fully downloaded (DownloadCompleted) on a specific node. */
export function isModelDownloadedOnNode(
  downloadsData: Record<string, unknown[]>,
  nodeId: string,
  modelId: string,
): boolean {
  const nodeDownloads = downloadsData[nodeId];
  if (!Array.isArray(nodeDownloads)) return false;

  for (const [tag, , entryModelId] of iterNodeDownloads(nodeDownloads)) {
    if (tag === "DownloadCompleted" && entryModelId === modelId) return true;
  }
  return false;
}

/** Get all node IDs where a model is fully downloaded (DownloadCompleted). */
export function getNodesWithModelDownloaded(
  downloadsData: Record<string, unknown[]>,
  modelId: string,
): string[] {
  const result: string[] = [];
  for (const nodeId of Object.keys(downloadsData)) {
    if (isModelDownloadedOnNode(downloadsData, nodeId, modelId)) {
      result.push(nodeId);
    }
  }
  return result;
}

/**
 * Find shard metadata for a model from any download entry across all nodes.
 * Returns the first match found (completed entries are preferred).
 */
export function getShardMetadataForModel(
  downloadsData: Record<string, unknown[]>,
  modelId: string,
): Record<string, unknown> | null {
  let fallback: Record<string, unknown> | null = null;

  for (const nodeDownloads of Object.values(downloadsData)) {
    if (!Array.isArray(nodeDownloads)) continue;

    for (const [tag, payload, entryModelId] of iterNodeDownloads(
      nodeDownloads,
    )) {
      if (entryModelId !== modelId) continue;
      const shard = extractShardMetadata(payload);
      if (!shard) continue;

      if (tag === "DownloadCompleted") return shard;
      if (!fallback) fallback = shard;
    }
  }
  return fallback;
}

/**
 * Get the download status tag for a specific model on a specific node.
 * Returns the "best" status: DownloadCompleted > DownloadOngoing > others.
 */
export function getModelDownloadStatus(
  downloadsData: Record<string, unknown[]>,
  nodeId: string,
  modelId: string,
): string | null {
  const nodeDownloads = downloadsData[nodeId];
  if (!Array.isArray(nodeDownloads)) return null;

  let best: string | null = null;
  for (const [tag, , entryModelId] of iterNodeDownloads(nodeDownloads)) {
    if (entryModelId !== modelId) continue;
    if (tag === "DownloadCompleted") return tag;
    if (tag === "DownloadOngoing") best = tag;
    else if (!best) best = tag;
  }
  return best;
}

/** What a device has of a model: all of it, or how many bytes so far. */
export interface DeviceModelBytes {
  completed: boolean;
  downloadedBytes: number;
}

function inBytes(value: unknown): number {
  if (typeof value === "number") return value;
  if (value && typeof value === "object") {
    const bytes = (value as { inBytes?: unknown }).inBytes;
    if (typeof bytes === "number") return bytes;
  }
  return 0;
}

/**
 * Every model's bytes on every device, in one pass over the downloads, for
 * questions asked about many models at once (e.g. which fit on disk).
 */
export function indexModelDownloads(
  downloadsData: Record<string, unknown[]>,
): Map<string, Map<string, DeviceModelBytes>> {
  const index = new Map<string, Map<string, DeviceModelBytes>>();
  for (const [nodeId, nodeDownloads] of Object.entries(downloadsData)) {
    if (!Array.isArray(nodeDownloads)) continue;
    for (const [tag, payload, modelId] of iterNodeDownloads(nodeDownloads)) {
      let bytes: DeviceModelBytes | null = null;
      if (tag === "DownloadCompleted") {
        bytes = { completed: true, downloadedBytes: inBytes(payload.total) };
      } else if (tag === "DownloadOngoing") {
        const progress = (payload.downloadProgress ??
          payload.download_progress) as { downloaded?: unknown } | undefined;
        bytes = {
          completed: false,
          downloadedBytes: inBytes(progress?.downloaded),
        };
      } else if (tag === "DownloadPending") {
        bytes = {
          completed: false,
          downloadedBytes: inBytes(payload.downloaded),
        };
      }
      if (!bytes) continue;
      let devices = index.get(modelId);
      if (!devices) index.set(modelId, (devices = new Map()));
      // A device can list a model twice (e.g. two shard layouts): keep the most
      const known = devices.get(nodeId);
      if (
        !known ||
        (bytes.completed && !known.completed) ||
        (!known.completed && bytes.downloadedBytes > known.downloadedBytes)
      ) {
        devices.set(nodeId, bytes);
      }
    }
  }
  return index;
}

// Runner statuses with the model's weights in memory
const IN_MEMORY = new Set([
  "RunnerLoaded",
  "RunnerWarmingUp",
  "RunnerReady",
  "RunnerRunning",
]);
// Runner statuses after which the model no longer runs there
const STOPPED = new Set([
  "RunnerFailed",
  "RunnerShuttingDown",
  "RunnerShutdown",
]);

/**
 * Whether a model runs on a node (its weights are in memory there), is
 * starting there (an instance has a runner there, still downloading,
 * connecting or loading), or neither. Instances and runner statuses are
 * tagged: { MlxRingInstance: { shardAssignments } }, { RunnerReady: {...} }.
 */
export function modelStateOnNode(
  instances: Record<string, unknown>,
  runners: Record<string, unknown>,
  nodeId: string,
  modelId: string,
): "running" | "starting" | null {
  for (const wrapped of Object.values(instances)) {
    if (!wrapped || typeof wrapped !== "object") continue;
    const unwrapped = unwrapTagged(wrapped as Record<string, unknown>);
    if (!unwrapped) continue;
    const assignments = unwrapped[1].shardAssignments as
      | { modelId?: string; nodeToRunner?: Record<string, string> }
      | undefined;
    if (assignments?.modelId !== modelId) continue;
    const runnerId = assignments.nodeToRunner?.[nodeId];
    if (!runnerId) continue;
    const status = runners[runnerId];
    const tag =
      status && typeof status === "object"
        ? unwrapTagged(status as Record<string, unknown>)?.[0]
        : undefined;
    if (tag && IN_MEMORY.has(tag)) return "running";
    if (tag && STOPPED.has(tag)) continue;
    return "starting";
  }
  return null;
}

/** What to ask before deleting a model's files from a device. */
export function deleteDownloadQuestion(options: {
  modelName: string;
  deviceName: string;
  /** e.g. "16 GB", or "6.8 GB downloaded" */
  size: string | null;
  state: "running" | "starting" | null;
}): string {
  const { modelName, deviceName, size, state } = options;
  const question = `Delete ${modelName}${size ? ` (${size})` : ""} from ${deviceName}?`;
  if (state === "running") {
    return (
      `${question}\n\n` +
      `It's running on ${deviceName}. It keeps running from memory for now, ` +
      `and ${deviceName} will download it again the next time it starts.`
    );
  }
  if (state === "starting") {
    return (
      `${question}\n\n` +
      `It's starting on ${deviceName}, which needs these files: ` +
      `${deviceName} will download them again.`
    );
  }
  return `${question}\n\nYou'll need to download it again to run it on this device.`;
}

/** What the downloads page shows about a model, from its card. */
export type ModelCardInfo = {
  family: string;
  quantization: string;
  baseModel: string;
  capabilities: string[];
  storageSize: number;
  nLayers: number;
  supportsTensor: boolean;
};

/** A model as `GET /models` lists it: the fields the downloads page uses. */
export type CatalogueModel = {
  id: string;
  family?: string;
  quantization?: string;
  base_model?: string;
  capabilities?: string[];
  storage_size_megabytes?: number;
  supports_tensor?: boolean;
};

/** A model's display name, "Llama 3.2 1B · 4bit", or null without a base model. */
export function modelDisplayName(
  baseModel: string,
  quantization: string,
): string | null {
  if (!baseModel) return null;
  return quantization ? `${baseModel} · ${quantization}` : baseModel;
}

/**
 * The display name and card details of a model in the catalogue. exo lists a
 * model on a device only when the device has some of it, so the downloads page
 * takes every other model from the catalogue.
 */
export function catalogueModelInfo(model: CatalogueModel): {
  prettyName: string | null;
  card: ModelCardInfo;
} {
  const baseModel = model.base_model ?? "";
  const quantization = model.quantization ?? "";
  return {
    prettyName: modelDisplayName(baseModel, quantization),
    card: {
      family: model.family ?? "",
      quantization,
      baseModel,
      capabilities: model.capabilities ?? [],
      // The catalogue gives whole megabytes
      storageSize: (model.storage_size_megabytes ?? 0) * 1024 * 1024,
      // Not in the catalogue; the details leave it out when 0
      nLayers: 0,
      supportsTensor: model.supports_tensor ?? false,
    },
  };
}

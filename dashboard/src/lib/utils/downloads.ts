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

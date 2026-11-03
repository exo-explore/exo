/**
 * One progress figure for a model downloading to several devices.
 *
 * Every device of the model counts, including those that have finished and
 * those that haven't started yet, so the total only moves forward while bytes
 * are being added. Leaving finished devices out made the total jump back
 * each time one finished (73.8% -> 44.4% as two of three finished), and a
 * device starting late made it drop too (62.8% -> 48.1%). Every device
 * downloads the whole repo (download_utils.resolve_allow_patterns), so every
 * device counts for the same size.
 */

export interface NodeDownload {
  nodeId: string;
  /** completed: all of it is on the device; downloading: in progress;
   * pending: known but not downloading; missing: no progress reported yet */
  state: "completed" | "downloading" | "pending" | "missing";
  downloadedBytes: number;
  /** 0 when the device hasn't reported how much it needs */
  totalBytes: number;
  /** Bytes per second; only a downloading device has one */
  speed: number;
}

export interface DownloadTotal {
  downloadedBytes: number;
  totalBytes: number;
  percentage: number;
  /** Sum of the devices' speeds */
  speed: number;
  /** Until the last device finishes; null when it can't be known (a device
   * hasn't started, or isn't moving) */
  etaMs: number | null;
}

/** A repo this much smaller than the model is one downloaded after it: the
 * vision weights some models keep in their own repo (Kimi K2.x) are reported
 * under the model's id once its main repo is done. */
const ADD_ON_FRACTION = 0.1;

/**
 * @param nodes every device of the model
 * @param estimatedNodeBytes the model's size (its card's storage size). Every
 *   device downloads the whole repo, so a device that hasn't said how much it
 *   needs is assumed to need as much as the most any device reported, or
 *   this, whichever is more.
 */
export function totalDownload(
  nodes: NodeDownload[],
  estimatedNodeBytes: number,
): DownloadTotal {
  const perDevice = Math.max(
    estimatedNodeBytes,
    0,
    ...nodes.map((node) => node.totalBytes),
  );
  let downloadedBytes = 0;
  let totalBytes = 0;
  let speed = 0;
  let etaMs: number | null = 0;
  for (const node of nodes) {
    const known = node.totalBytes > 0;
    // Past its main repo, on a small one after it (or done): counts as the
    // whole model, so the total doesn't drop to that repo's size
    const onAddOn =
      known &&
      node.state !== "completed" &&
      node.totalBytes < perDevice * ADD_ON_FRACTION;
    const total =
      node.state === "completed" || onAddOn || !known
        ? perDevice
        : node.totalBytes;
    const downloaded =
      node.state === "completed" || onAddOn
        ? total
        : Math.min(Math.max(node.downloadedBytes, 0), total);
    totalBytes += total;
    downloadedBytes += downloaded;
    if (node.state === "downloading") speed += Math.max(node.speed, 0);

    if (node.state === "completed") continue;
    // What it still has to download, add-on repo included
    const remaining = known
      ? node.totalBytes - Math.max(node.downloadedBytes, 0)
      : total;
    if (known && remaining <= 0) continue;
    // Devices download at the same time, so the slowest one decides
    const nodeEta = node.speed > 0 ? (remaining / node.speed) * 1000 : null;
    if (etaMs === null || !known || nodeEta === null) {
      etaMs = null;
    } else {
      etaMs = Math.max(etaMs, nodeEta);
    }
  }
  return {
    downloadedBytes,
    totalBytes,
    percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
    speed,
    etaMs,
  };
}

/**
 * One progress figure for a model downloading to several devices.
 *
 * Every device of the model counts, including those that have finished and
 * those that haven't started yet, so the total only moves forward while bytes
 * are being added. Leaving finished devices out made the total jump back
 * each time one finished (73.8% -> 44.4% as two of three finished), and a
 * device starting late made it drop too (62.8% -> 48.1%).
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

/**
 * @param nodes every device of the model
 * @param estimatedNodeBytes what a device that hasn't said how much it needs
 *   is assumed to need: the whole model, which no device needs more than, so
 *   the total can only go up when the device reports its real size
 */
export function totalDownload(
  nodes: NodeDownload[],
  estimatedNodeBytes: number,
): DownloadTotal {
  let downloadedBytes = 0;
  let totalBytes = 0;
  let speed = 0;
  let etaMs: number | null = 0;
  for (const node of nodes) {
    const known = node.totalBytes > 0;
    const total = known ? node.totalBytes : estimatedNodeBytes;
    const downloaded =
      node.state === "completed"
        ? total
        : Math.min(Math.max(node.downloadedBytes, 0), total);
    totalBytes += total;
    downloadedBytes += downloaded;
    if (node.state === "downloading") speed += Math.max(node.speed, 0);

    const remaining = total - downloaded;
    if (node.state === "completed" || (known && remaining <= 0)) continue;
    // Devices download at the same time, so the slowest one decides
    if (etaMs === null || !known || node.speed <= 0) {
      etaMs = null;
    } else {
      etaMs = Math.max(etaMs, (remaining / node.speed) * 1000);
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

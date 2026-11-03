/**
 * Whether each download is moving, and how fast, from the bytes /state
 * reports over time.
 *
 * The speed a device reports is its average since the download (or its
 * retry) started. It isn't a measure of now: it stays above zero while a
 * download is stuck, and right after a start (or a long wait before it) it
 * is so low that the ETA reads "17143h". So the dashboard times the bytes
 * itself:
 * - a download whose bytes haven't gone up for DOWNLOAD_STALL_MS is stalled
 *   (the downloader is waiting to retry);
 * - its rate is the bytes added over the last RATE_WINDOW_MS, and there is
 *   no rate (so no ETA) until MIN_RATE_SPAN_MS of it has been seen.
 */

export const DOWNLOAD_STALL_MS = 15_000;
const RATE_WINDOW_MS = 10_000;
const MIN_RATE_SPAN_MS = 5_000;

export interface DownloadActivity {
  /** When its bytes last went up (browser clock, ms) */
  progressAt: number;
  /** Bytes per second lately; null until enough of it has been seen */
  rate: number | null;
}

export function downloadKey(nodeId: string, modelId: string): string {
  return `${nodeId}\n${modelId}`;
}

export class DownloadActivityTracker {
  private samples = new Map<string, Array<{ at: number; bytes: number }>>();
  private progressAt = new Map<string, number>();

  /**
   * Record a /state poll.
   * @param downloading the downloaded bytes of every download in progress,
   *   by downloadKey
   */
  observe(
    downloading: Record<string, number>,
    now: number,
  ): Record<string, DownloadActivity> {
    const result: Record<string, DownloadActivity> = {};
    for (const [key, bytes] of Object.entries(downloading)) {
      let samples = this.samples.get(key) ?? [];
      const last = samples[samples.length - 1];
      if (!last || bytes < last.bytes) {
        // New, or started over
        samples = [];
        this.progressAt.set(key, now);
      } else if (bytes > last.bytes) {
        this.progressAt.set(key, now);
      }
      if (samples.length === 0 || now > samples[samples.length - 1].at) {
        samples.push({ at: now, bytes });
      }
      // Keep the window, and the sample just before it to measure from
      while (samples.length > 2 && samples[1].at <= now - RATE_WINDOW_MS) {
        samples.shift();
      }
      this.samples.set(key, samples);
      const first = samples[0];
      const span = now - first.at;
      result[key] = {
        progressAt: this.progressAt.get(key) ?? now,
        rate:
          span >= MIN_RATE_SPAN_MS
            ? ((bytes - first.bytes) / span) * 1000
            : null,
      };
    }
    for (const key of [...this.samples.keys()]) {
      if (!(key in downloading)) {
        this.samples.delete(key);
        this.progressAt.delete(key);
      }
    }
    return result;
  }
}

/** Whether a download has added no bytes for DOWNLOAD_STALL_MS. */
export function isStalled(
  activity: DownloadActivity | undefined,
  now: number,
): boolean {
  return (
    activity !== undefined && now - activity.progressAt >= DOWNLOAD_STALL_MS
  );
}

/** How long a download will take at its recent rate; null when unknown
 * (stalled, not moving, or not watched long enough). */
export function estimateRemainingMs(
  remainingBytes: number,
  activity: DownloadActivity | undefined,
  now: number,
): number | null {
  if (remainingBytes <= 0) return 0;
  if (!activity || isStalled(activity, now)) return null;
  if (activity.rate === null || activity.rate <= 0) return null;
  return (remainingBytes / activity.rate) * 1000;
}

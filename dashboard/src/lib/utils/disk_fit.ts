/**
 * Whether a model can be downloaded where it would run.
 *
 * Fitting in memory isn't enough: each device the model runs on downloads
 * it first, into its models directory, and each one downloads all of it
 * (download_utils.resolve_allow_patterns). So the model fits on disk when the
 * devices that have room for what they still need of it have enough free
 * memory between them to run it. A device that has part of it already needs
 * only the rest, as the backend's own check does. /state reports each
 * device's free space in its models directory (nodeDisk); a device that
 * hasn't reported it yet isn't held against the model.
 */

export interface DeviceSpace {
  name: string;
  /** Free bytes in its models directory; null when not reported */
  freeDiskBytes: number | null;
  /** Memory it has free for a model */
  freeMemoryBytes: number;
  /** The model is downloaded there already */
  hasModel: boolean;
  /** How much of it is there already (a paused or running download) */
  downloadedBytes: number;
}

export type DiskFit =
  | { fits: true }
  | {
      fits: false;
      /** The device that would help most, and what it lacks */
      device: string;
      neededBytes: number;
      freeBytes: number;
      /** It has part of the model already: neededBytes is the rest */
      partly: boolean;
    };

export function checkDiskFit(
  modelBytes: number,
  devices: DeviceSpace[],
): DiskFit {
  if (modelBytes <= 0 || devices.length === 0) return { fits: true };
  const needs = (device: DeviceSpace) =>
    device.hasModel
      ? 0
      : Math.max(0, modelBytes - Math.max(device.downloadedBytes, 0));
  const hasRoom = (device: DeviceSpace) =>
    device.freeDiskBytes === null || device.freeDiskBytes >= needs(device);
  const memoryWithRoom = devices
    .filter(hasRoom)
    .reduce((sum, device) => sum + Math.max(device.freeMemoryBytes, 0), 0);
  if (memoryWithRoom >= modelBytes) return { fits: true };
  // Name the device short on disk that has the most memory to offer
  const short = devices
    .filter((device) => !hasRoom(device))
    .sort((a, b) => b.freeMemoryBytes - a.freeMemoryBytes)[0];
  if (!short || short.freeDiskBytes === null) return { fits: true };
  return {
    fits: false,
    device: short.name,
    neededBytes: needs(short),
    freeBytes: short.freeDiskBytes,
    partly: short.downloadedBytes > 0,
  };
}

function gb(bytes: number): string {
  const value = bytes / 1024 ** 3;
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} GB`;
}

/** e.g. "Not enough disk space: needs 155 GB, 108 GB free on studio-a" */
export function describeDiskFit(fit: DiskFit): string | null {
  if (fit.fits) return null;
  const more = fit.partly ? " more" : "";
  return `Not enough disk space: needs ${gb(fit.neededBytes)}${more}, ${gb(fit.freeBytes)} free on ${fit.device}`;
}

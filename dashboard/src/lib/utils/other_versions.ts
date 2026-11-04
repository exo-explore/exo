/**
 * Devices on the network under this cluster's name that can't join it,
 * because they run another version of exo (GET /v1/devices-on-other-versions).
 */

export type OtherVersionReason = "older_exo" | "wire_version" | "build";

export type DeviceOnOtherVersion = {
  /** Every address it was heard on: one per network interface */
  addresses: string[];
  reason: OtherVersionReason;
  /** Its wire version, when it runs another one */
  wireVersion: number | null;
};

/**
 * How often to ask. exo lists a device until 30 s after the last Hello it
 * heard from it (one a second), so asking every 10 s shows a device within
 * 10 s of its first Hello, and drops one within 40 s of its last.
 */
export const OTHER_VERSIONS_POLL_MS = 10_000;

/**
 * How long one ask may take. Well under the time between asks: the answer is
 * a few hundred bytes from this node itself, so a slower one means it is busy,
 * and the next ask will do.
 */
export const OTHER_VERSIONS_TIMEOUT_MS = 5_000;

const REASONS: readonly OtherVersionReason[] = [
  "older_exo",
  "wire_version",
  "build",
];

/** The devices in the endpoint's answer; anything unexpected reads as none. */
export function parseDevicesOnOtherVersions(
  body: unknown,
): DeviceOnOtherVersion[] {
  if (!body || typeof body !== "object") return [];
  const devices = (body as { devices?: unknown }).devices;
  if (!Array.isArray(devices)) return [];
  return devices.flatMap((device: unknown) => {
    if (!device || typeof device !== "object") return [];
    const { addresses, reason, wireVersion } = device as Record<
      string,
      unknown
    >;
    if (
      !Array.isArray(addresses) ||
      addresses.length === 0 ||
      !addresses.every((address) => typeof address === "string") ||
      !REASONS.includes(reason as OtherVersionReason)
    )
      return [];
    return [
      {
        addresses: addresses as string[],
        reason: reason as OtherVersionReason,
        wireVersion: typeof wireVersion === "number" ? wireVersion : null,
      },
    ];
  });
}

/** "fe80::1 (and 2 more addresses) runs exo wire version 2" */
export function describeDevice(device: DeviceOnOtherVersion): string {
  const [first, ...rest] = device.addresses;
  const where =
    rest.length === 0
      ? first
      : `${first} (and ${rest.length} more address${rest.length === 1 ? "" : "es"})`;
  switch (device.reason) {
    case "older_exo":
      return `${where} runs an older exo`;
    case "wire_version":
      return `${where} runs exo wire version ${device.wireVersion ?? "?"}`;
    case "build":
      return `${where} runs another build of exo`;
  }
}

/** The devices exo lists, or null when it didn't answer in time. */
export async function fetchDevicesOnOtherVersions(
  timeoutMs: number = OTHER_VERSIONS_TIMEOUT_MS,
  fetchImpl: typeof fetch = fetch,
): Promise<DeviceOnOtherVersion[] | null> {
  try {
    const response = await fetchImpl("/v1/devices-on-other-versions", {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) return null;
    return parseDevicesOnOtherVersions(await response.json());
  } catch {
    return null;
  }
}

/**
 * How many asks in a row may fail before the list is dropped. A failed ask
 * keeps the last list, so one slow answer doesn't make the notice flicker; 3
 * asks (30 s) is about as long as exo itself keeps listing a device after its
 * last Hello, so a device that left isn't shown much longer than that.
 */
export const OTHER_VERSIONS_FAILED_ASKS_BEFORE_DROPPING = 3;

/**
 * Asks every `intervalMs`, never twice at once, so an old answer can't arrive
 * after a newer one. A failed ask keeps the last list, until
 * `OTHER_VERSIONS_FAILED_ASKS_BEFORE_DROPPING` in a row drop it. Returns a stop
 * function.
 */
export function pollDevicesOnOtherVersions(
  onDevices: (devices: DeviceOnOtherVersion[]) => void,
  intervalMs: number = OTHER_VERSIONS_POLL_MS,
  fetchDevices: () => Promise<DeviceOnOtherVersion[] | null> = () =>
    fetchDevicesOnOtherVersions(),
): () => void {
  let asking = false;
  let stopped = false;
  let failedInARow = 0;
  const ask = async () => {
    if (asking || stopped) return;
    asking = true;
    try {
      const devices = await fetchDevices();
      if (stopped) return;
      if (devices !== null) {
        failedInARow = 0;
        onDevices(devices);
      } else if (
        ++failedInARow === OTHER_VERSIONS_FAILED_ASKS_BEFORE_DROPPING
      ) {
        onDevices([]);
      }
    } finally {
      asking = false;
    }
  };
  void ask();
  const timer = setInterval(() => void ask(), intervalMs);
  return () => {
    stopped = true;
    clearInterval(timer);
  };
}

/**
 * Notices for models that stopped on their own (a device left, or exo
 * restarted on a device). Nothing restarts such a model, so the notice, with
 * its "Start again", stays until it is dismissed or the model runs again,
 * across reloads and in every tab: it is kept in localStorage.
 */

export const STOPPED_MODELS_KEY = "exo-stopped-models";
/** The ids this browser's own device has had (exo picks a new one each
 * start), so a restart of this device can be told from a device leaving */
export const THIS_DEVICE_IDS_KEY = "exo-this-device-ids";
const MAX_THIS_DEVICE_IDS = 20;

export interface StoppedModelNotice {
  modelId: string;
  message: string;
  /** When it stopped (ms) */
  at: number;
}

export type StopReason =
  /** exo restarted on the device serving this dashboard */
  | { kind: "this-device-restarted" }
  /** exo restarted on another device: it is back under a new identity */
  | { kind: "device-restarted"; device: string }
  | { kind: "device-left"; device: string | null }
  /** Its devices are all still here */
  | { kind: "stopped" };

export interface StopContext {
  /** The model's devices */
  nodeIds: string[];
  /** Devices in the cluster now */
  connectedNodeIds: Set<string>;
  /** A device's name when the model was last seen running */
  nameOf: (nodeId: string) => string | undefined;
  /** Names of the devices in the cluster now */
  connectedNames: Set<string>;
  /** Ids this dashboard's own device has had */
  thisDeviceIds: Set<string>;
}

/** Why a model that wasn't stopped from this dashboard went away. */
export function whyModelStopped(context: StopContext): StopReason {
  const gone = context.nodeIds.filter(
    (nodeId) => !context.connectedNodeIds.has(nodeId),
  );
  if (gone.length === 0 || context.connectedNodeIds.size === 0) {
    return { kind: "stopped" };
  }
  // This dashboard is still being served, so its device is up again under
  // a new identity: exo restarted on it
  if (gone.some((nodeId) => context.thisDeviceIds.has(nodeId))) {
    return { kind: "this-device-restarted" };
  }
  const names = gone.map(context.nameOf);
  if (names.every((name) => name && context.connectedNames.has(name))) {
    return { kind: "device-restarted", device: names.join(", ") };
  }
  const missing = names.filter(
    (name): name is string => !!name && !context.connectedNames.has(name),
  );
  return {
    kind: "device-left",
    device: missing.length > 0 ? missing.join(", ") : null,
  };
}

/** The notice's text. */
export function stoppedModelMessage(
  modelName: string,
  reason: StopReason,
): string {
  switch (reason.kind) {
    case "this-device-restarted":
      return `${modelName} stopped — exo restarted on this device`;
    case "device-restarted":
      return `${modelName} stopped — exo restarted on ${reason.device}`;
    case "device-left":
      return reason.device
        ? `${modelName} stopped — ${reason.device} left the cluster`
        : `${modelName} stopped — a device left the cluster`;
    case "stopped":
      return `${modelName} stopped`;
  }
}

/** Notices from storage; anything unreadable is dropped. */
export function parseNotices(stored: string | null): StoppedModelNotice[] {
  if (!stored) return [];
  try {
    const parsed: unknown = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (notice): notice is StoppedModelNotice =>
        !!notice &&
        typeof notice.modelId === "string" &&
        typeof notice.message === "string" &&
        typeof notice.at === "number",
    );
  } catch {
    return [];
  }
}

/** One notice per model: a newer one replaces the older. */
export function withNotice(
  notices: StoppedModelNotice[],
  notice: StoppedModelNotice,
): StoppedModelNotice[] {
  return [...notices.filter((n) => n.modelId !== notice.modelId), notice];
}

export function withoutNotice(
  notices: StoppedModelNotice[],
  modelId: string,
): StoppedModelNotice[] {
  return notices.filter((n) => n.modelId !== modelId);
}

/** The ids this device has had, with a new one added (newest last). */
export function withThisDeviceId(ids: string[], id: string): string[] {
  return [...ids.filter((known) => known !== id), id].slice(
    -MAX_THIS_DEVICE_IDS,
  );
}

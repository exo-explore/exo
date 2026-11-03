/**
 * Whether a running model can serve right now, from what /state says about
 * its devices and runners.
 *
 * A runner reports its status when it changes, so a model whose device has
 * frozen, quit or crashed keeps its last status (often Ready) until the
 * master notices the silence and removes the model, 15 s later
 * (NODE_SILENCE_TIMEOUT in master/main.py). Meanwhile two things in /state
 * already tell the truth:
 * - lastSeen: every device reports about once a second, so a device whose
 *   lastSeen stops moving has gone quiet;
 * - runners: a runner that shuts down is removed from the state, so a model
 *   with some runners Ready and others gone can't serve.
 */

/** Silence after which a device counts as not responding. Healthy devices
 * report about every second (at most 6.3 s apart on four busy devices); the
 * master removes a device after 15 s. */
export const DEVICE_SILENCE_MS = 8_000;

/**
 * When each device was last heard from, on this browser's clock.
 *
 * lastSeen is stamped by each device's own clock, which may not match the
 * browser's, so silence is measured by when a device's lastSeen last changed.
 * A device seen for the first time is placed by how far its lastSeen lags the
 * newest one, so that a device that went quiet before the page loaded is
 * known at once.
 */
export class DeviceSilenceTracker {
  private heard = new Map<string, { lastSeen: string; heardAt: number }>();

  /** Record a /state poll; returns when each device was last heard from. */
  observe(
    lastSeen: Record<string, string>,
    now: number,
  ): Record<string, number> {
    const stamps = Object.values(lastSeen)
      .map((value) => Date.parse(value))
      .filter((ms) => !Number.isNaN(ms));
    const newest = stamps.length > 0 ? Math.max(...stamps) : NaN;
    const next = new Map<string, { lastSeen: string; heardAt: number }>();
    for (const [nodeId, value] of Object.entries(lastSeen)) {
      const known = this.heard.get(nodeId);
      if (known && known.lastSeen === value) {
        next.set(nodeId, known);
        continue;
      }
      let heardAt = now;
      if (!known) {
        const lag = newest - Date.parse(value);
        if (Number.isFinite(lag) && lag > 0) heardAt = now - lag;
      }
      next.set(nodeId, { lastSeen: value, heardAt });
    }
    this.heard = next;
    return Object.fromEntries(
      [...next].map(([nodeId, { heardAt }]) => [nodeId, heardAt]),
    );
  }
}

export type InstanceHealth =
  | { kind: "ok" }
  /** This dashboard hears from no device: the cluster's state is frozen,
   * e.g. while a new master is elected */
  | { kind: "cluster-silent" }
  /** A device of the model is no longer in the cluster */
  | { kind: "device-left"; nodeIds: string[] }
  /** A device of the model has been silent for DEVICE_SILENCE_MS */
  | { kind: "not-responding"; nodeIds: string[] }
  /** Some runners of the model are Ready while others have shut down */
  | { kind: "runner-stopped"; nodeIds: string[] };

// Runner statuses reached only after every runner of the model has connected
const STARTED_RUNNER = new Set([
  "RunnerLoading",
  "RunnerLoaded",
  "RunnerWarmingUp",
  "RunnerReady",
  "RunnerRunning",
]);

export interface InstanceHealthInput {
  /** The model's devices and their runners (shardAssignments.nodeToRunner) */
  nodeToRunner: Record<string, string>;
  /** Each runner's status tag (e.g. "RunnerReady"), undefined when the state
   * has no status for it */
  runnerStatus: (runnerId: string) => string | undefined;
  /** The devices in the cluster now */
  clusterNodeIds: Set<string>;
  /** When each device was last heard from (DeviceSilenceTracker.observe) */
  heardAt: Record<string, number>;
  now: number;
  silenceMs?: number;
}

export function checkInstanceHealth(
  input: InstanceHealthInput,
): InstanceHealth {
  const { nodeToRunner, runnerStatus, clusterNodeIds, heardAt, now } = input;
  const silenceMs = input.silenceMs ?? DEVICE_SILENCE_MS;
  const nodeIds = Object.keys(nodeToRunner);
  if (nodeIds.length === 0) return { kind: "ok" };

  const silent = (nodeId: string) =>
    heardAt[nodeId] !== undefined && now - heardAt[nodeId] >= silenceMs;
  const known = Object.keys(heardAt);
  if (known.length > 0 && known.every(silent))
    return { kind: "cluster-silent" };

  const left = nodeIds.filter((nodeId) => !clusterNodeIds.has(nodeId));
  if (left.length > 0) return { kind: "device-left", nodeIds: left };

  const quiet = nodeIds.filter(silent);
  if (quiet.length > 0) return { kind: "not-responding", nodeIds: quiet };

  // Runners start together; once one has got past connecting, every runner
  // of the model existed, so a missing one has shut down
  const statuses = nodeIds.map((nodeId) => runnerStatus(nodeToRunner[nodeId]));
  if (statuses.some((status) => status && STARTED_RUNNER.has(status))) {
    const stopped = nodeIds.filter((_, i) => statuses[i] === undefined);
    if (stopped.length > 0) return { kind: "runner-stopped", nodeIds: stopped };
  }
  return { kind: "ok" };
}

/** One line saying why the model can't serve, naming its devices. */
export function describeInstanceHealth(
  health: InstanceHealth,
  deviceName: (nodeId: string) => string,
): string | null {
  const names = (nodeIds: string[]) => nodeIds.map(deviceName).join(", ");
  switch (health.kind) {
    case "ok":
      return null;
    case "cluster-silent":
      return "Waiting for the cluster to respond…";
    case "device-left":
      return `${names(health.nodeIds)} left the cluster`;
    case "not-responding":
      return health.nodeIds.length === 1
        ? `${names(health.nodeIds)} isn't responding. Waiting for it…`
        : `${names(health.nodeIds)} aren't responding. Waiting for them…`;
    case "runner-stopped":
      return `The model stopped on ${names(health.nodeIds)}`;
  }
}

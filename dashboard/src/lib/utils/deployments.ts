/**
 * Deployments: the models users launched, which exo keeps running. `/state`
 * has them as `deployments`, keyed by id; each instance names its deployment
 * (`deploymentId`). A deployment without an instance is starting again,
 * waiting for something, or held until the user resumes it. The backend says
 * which, and why: the dashboard only shows it.
 *
 * Stop: DELETE /instance/{deploymentId}. Resume (or start now):
 * POST /deployments/{deploymentId}/resume.
 */

export type HoldKind =
  | "CrashLoop"
  | "StartFailed"
  | "WaitedTooLong"
  | "Restored";
export type WaitKind =
  | "DeviceAway"
  | "NoDevicesWithFiles"
  | "NotEnoughDevices"
  | "DoesNotFit";

export interface RawDeployment {
  deploymentId?: string;
  spec?: { modelId?: string; minNodes?: number; pinnedNodes?: string[] };
  hold?: { kind?: string; message?: string | null } | null;
  waiting?: {
    kind?: string;
    needed?: number | null;
    have?: number | null;
    until?: string | null;
  } | null;
  note?: string | null;
  lastNodes?: string[];
}

export type DeploymentPhase =
  /** Placed: its instance's card shows how it runs */
  | { kind: "running" }
  /** No instance, nothing to wait for: it is being placed again */
  | { kind: "restarting" }
  | { kind: "waiting"; reason: string; canStartNow: boolean }
  | { kind: "paused"; reason: string };

export interface Deployment {
  id: string;
  modelId: string;
  minNodes: number;
  phase: DeploymentPhase;
  /** How it runs, e.g. that it fell back from RDMA to TCP */
  note: string | null;
}

function waitingReason(
  waiting: NonNullable<RawDeployment["waiting"]>,
  now: number,
): string {
  switch (waiting.kind as WaitKind) {
    case "DeviceAway": {
      const until = waiting.until ? Date.parse(waiting.until) : NaN;
      if (Number.isNaN(until) || until <= now) {
        return "Waiting for the device it ran on to come back";
      }
      const minutes = Math.ceil((until - now) / 60_000);
      return `Waiting for the device it ran on to come back (up to ${minutes} min)`;
    }
    case "NoDevicesWithFiles":
      return "No connected device has its files";
    case "NotEnoughDevices":
      return `Needs ${waiting.needed ?? "more"} devices that have its files; ${waiting.have ?? 0} connected`;
    case "DoesNotFit":
      return "Doesn't fit in the free memory of the devices that have its files";
    default:
      return "Waiting to start again";
  }
}

/** "It lost a device…" reads "Paused: it lost a device…" */
function asClause(message: string): string {
  return message.charAt(0).toLowerCase() + message.slice(1);
}

function pausedReason(hold: NonNullable<RawDeployment["hold"]>): string {
  switch (hold.kind as HoldKind) {
    case "CrashLoop":
      return hold.message
        ? `Paused: ${asClause(hold.message)}`
        : "Paused: it kept losing a device";
    case "StartFailed":
      return hold.message
        ? `Paused: it failed to start (${hold.message})`
        : "Paused: it failed to start";
    case "WaitedTooLong":
      return hold.message
        ? `Paused: ${asClause(hold.message)}`
        : "Paused: it couldn't start again for a long time";
    case "Restored":
      return "Paused: exo restarted on every device. Resume to load it again";
    default:
      return hold.message ? `Paused: ${asClause(hold.message)}` : "Paused";
  }
}

/**
 * The deployments in `/state`, with what each is doing.
 * @param placed the deployment ids that have an instance
 * @param now the time (ms), for how long a wait has left
 */
export function parseDeployments(
  raw: unknown,
  placed: Set<string>,
  now: number,
): Deployment[] {
  if (!raw || typeof raw !== "object") return [];
  const out: Deployment[] = [];
  for (const [id, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!value || typeof value !== "object") continue;
    const deployment = value as RawDeployment;
    const modelId = deployment.spec?.modelId;
    if (typeof modelId !== "string") continue;
    let phase: DeploymentPhase;
    // An instance runs it: that is what the user sees, whatever else the
    // backend says (a hold is only ever of a deployment without one)
    if (placed.has(id)) {
      phase = { kind: "running" };
    } else if (deployment.hold) {
      phase = { kind: "paused", reason: pausedReason(deployment.hold) };
    } else if (deployment.waiting) {
      phase = {
        kind: "waiting",
        reason: waitingReason(deployment.waiting, now),
        // Waiting for a device that may come back: the user can start it here now
        canStartNow: deployment.waiting.kind === "DeviceAway",
      };
    } else {
      phase = { kind: "restarting" };
    }
    out.push({
      id,
      modelId,
      minNodes: deployment.spec?.minNodes ?? 1,
      phase,
      note: typeof deployment.note === "string" ? deployment.note : null,
    });
  }
  return out;
}

/**
 * The notes on how running models run (e.g. that one fell back from RDMA to
 * TCP), by deployment id, for their instances' cards.
 */
export function runningNotes(list: Deployment[]): Map<string, string> {
  return new Map(
    list.flatMap((deployment) =>
      deployment.phase.kind === "running" && deployment.note !== null
        ? [[deployment.id, deployment.note] as [string, string]]
        : [],
    ),
  );
}

/**
 * What a dashboard whose onboarding isn't done does on its first cluster state:
 * - "announce": a model runs, so the cluster is set up; say so, offer the chat;
 * - "skip": a model was launched but doesn't run now (paused after every device
 *   restarted, or waiting), so the cluster is set up too, and its card says
 *   what to do; onboarding would hide it;
 * - "onboard": nothing was launched.
 */
export function onboardingDecision(
  runningModelId: string | undefined,
  notRunning: number,
): "announce" | "skip" | "onboard" {
  if (runningModelId !== undefined) return "announce";
  return notRunning > 0 ? "skip" : "onboard";
}

/**
 * What a launch tells exo to keep when it starts the model again: the number
 * of devices and the devices the user chose. A preview the user picked fixes
 * its number of devices; otherwise the minimum chosen (1, the default, lets it
 * come back on fewer devices).
 */
export function launchConstraints(
  chosenMinNodes: number,
  pickedPreviewNodeCount: number | null,
  filteredNodes: Iterable<string>,
): { min_nodes: number; pinned_nodes: string[] } {
  return {
    min_nodes: Math.max(1, pickedPreviewNodeCount ?? chosenMinNodes),
    pinned_nodes: [...filteredNodes],
  };
}

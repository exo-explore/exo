import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEVICE_SILENCE_MS,
  DeviceSilenceTracker,
  checkInstanceHealth,
  describeInstanceHealth,
  type InstanceHealthInput,
} from "../src/lib/utils/instance_health.ts";

const NODES = ["studio-a", "max-c", "max-d"];
const nodeToRunner = { "studio-a": "r0", "max-c": "r1", "max-d": "r2" };
const allReady: Record<string, string> = {
  r0: "RunnerReady",
  r1: "RunnerReady",
  r2: "RunnerReady",
};
const T0 = 1_000_000;

function input(
  overrides: Partial<InstanceHealthInput> & {
    statuses?: Record<string, string>;
  } = {},
): InstanceHealthInput {
  const statuses = overrides.statuses ?? allReady;
  return {
    nodeToRunner,
    runnerStatus: (runnerId) => statuses[runnerId],
    clusterNodeIds: new Set(NODES),
    heardAt: Object.fromEntries(NODES.map((node) => [node, T0])),
    now: T0 + 1_000,
    ...overrides,
  };
}

const iso = (ms: number) => new Date(ms).toISOString();

test("a model on devices that all report is ok", () => {
  assert.deepEqual(checkInstanceHealth(input()), { kind: "ok" });
});

test("a device that stops reporting makes the model unavailable", () => {
  const heardAt = { "studio-a": T0 + 9_000, "max-c": T0, "max-d": T0 + 9_000 };
  assert.deepEqual(
    checkInstanceHealth(input({ heardAt, now: T0 + DEVICE_SILENCE_MS + 1 })),
    { kind: "not-responding", nodeIds: ["max-c"] },
  );
  // Not yet: reports can be a few seconds apart on a busy cluster
  assert.deepEqual(
    checkInstanceHealth(input({ heardAt, now: T0 + DEVICE_SILENCE_MS - 1 })),
    { kind: "ok" },
  );
});

test("a device that left the cluster", () => {
  assert.deepEqual(
    checkInstanceHealth(
      input({ clusterNodeIds: new Set(["studio-a", "max-d"]) }),
    ),
    { kind: "device-left", nodeIds: ["max-c"] },
  );
});

test("runners that shut down while the others still say Ready", () => {
  // After a device quits, the other runners fail, shut down and are removed
  // from the state; the quitting device's last status is still Ready
  assert.deepEqual(
    checkInstanceHealth(input({ statuses: { r1: "RunnerReady" } })),
    { kind: "runner-stopped", nodeIds: ["studio-a", "max-d"] },
  );
});

test("runners that haven't reported yet while starting are fine", () => {
  assert.deepEqual(
    checkInstanceHealth(input({ statuses: { r0: "RunnerConnecting" } })),
    { kind: "ok" },
  );
  assert.deepEqual(checkInstanceHealth(input({ statuses: {} })), {
    kind: "ok",
  });
});

test("when no device is heard from, the cluster is silent", () => {
  assert.deepEqual(
    checkInstanceHealth(input({ now: T0 + DEVICE_SILENCE_MS })),
    { kind: "cluster-silent" },
  );
});

test("silence is measured on the browser's clock, not the devices'", () => {
  const tracker = new DeviceSilenceTracker();
  // Device clocks a minute behind the browser
  const skew = -60_000;
  let heard = tracker.observe({ a: iso(T0 + skew), b: iso(T0 + skew) }, T0);
  assert.deepEqual(heard, { a: T0, b: T0 });
  // b keeps reporting, a goes quiet
  heard = tracker.observe(
    { a: iso(T0 + skew), b: iso(T0 + 5_000 + skew) },
    T0 + 5_000,
  );
  heard = tracker.observe(
    { a: iso(T0 + skew), b: iso(T0 + 9_000 + skew) },
    T0 + 9_000,
  );
  assert.equal(heard.a, T0);
  assert.equal(heard.b, T0 + 9_000);
});

test("a device already quiet when the page loads is known at once", () => {
  // The old identity of a restarted device: its lastSeen stopped 10 s ago
  const tracker = new DeviceSilenceTracker();
  const heard = tracker.observe(
    { old: iso(T0 - 10_000), current: iso(T0), other: iso(T0 - 500) },
    T0,
  );
  assert.equal(heard.old, T0 - 10_000);
  assert.equal(heard.current, T0);
  assert.equal(heard.other, T0 - 500);
});

test("devices that leave are forgotten", () => {
  const tracker = new DeviceSilenceTracker();
  tracker.observe({ a: iso(T0), b: iso(T0) }, T0);
  assert.deepEqual(Object.keys(tracker.observe({ a: iso(T0 + 1) }, T0 + 1)), [
    "a",
  ]);
});

test("each reason reads as one line naming the devices", () => {
  const name = (nodeId: string) => nodeId;
  assert.equal(describeInstanceHealth({ kind: "ok" }, name), null);
  assert.equal(
    describeInstanceHealth(
      { kind: "not-responding", nodeIds: ["max-c"] },
      name,
    ),
    "max-c isn't responding. Waiting for it…",
  );
  assert.equal(
    describeInstanceHealth(
      { kind: "not-responding", nodeIds: ["max-c", "max-d"] },
      name,
    ),
    "max-c, max-d aren't responding. Waiting for them…",
  );
  assert.equal(
    describeInstanceHealth({ kind: "device-left", nodeIds: ["max-c"] }, name),
    "max-c left the cluster",
  );
  assert.equal(
    describeInstanceHealth(
      { kind: "runner-stopped", nodeIds: ["max-d"] },
      name,
    ),
    "The model stopped on max-d",
  );
  assert.equal(
    describeInstanceHealth({ kind: "cluster-silent" }, name),
    "Waiting for the cluster to respond…",
  );
});

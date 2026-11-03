import { test } from "node:test";
import assert from "node:assert/strict";
import {
  launchConstraints,
  onboardingDecision,
  parseDeployments,
  runningNotes,
} from "../src/lib/utils/deployments.ts";

const NOW = Date.parse("2026-11-03T12:00:00Z");

function deployment(extra: Record<string, unknown> = {}) {
  return {
    deploymentId: "d",
    spec: { modelId: "org/model", minNodes: 2, pinnedNodes: [] },
    hold: null,
    waiting: null,
    note: null,
    ...extra,
  };
}

test("onboarding is skipped for a cluster with a model launched, running or not", () => {
  assert.equal(onboardingDecision("org/model", 0), "announce");
  // Paused after every device restarted: its card says to resume it, and
  // onboarding over it would hide that
  assert.equal(onboardingDecision(undefined, 1), "skip");
  assert.equal(onboardingDecision(undefined, 0), "onboard");
});

test("a deployment with an instance is running", () => {
  const [d] = parseDeployments({ d: deployment() }, new Set(["d"]), NOW);
  assert.deepEqual(d.phase, { kind: "running" });
  assert.equal(d.minNodes, 2);
});

test("one without an instance and nothing to wait for is restarting", () => {
  const [d] = parseDeployments({ d: deployment() }, new Set(), NOW);
  assert.deepEqual(d.phase, { kind: "restarting" });
});

test("a wait for a device says how long, and can be started now", () => {
  const [d] = parseDeployments(
    {
      d: deployment({
        waiting: { kind: "DeviceAway", until: "2026-11-03T12:04:30Z" },
      }),
    },
    new Set(),
    NOW,
  );
  assert.deepEqual(d.phase, {
    kind: "waiting",
    reason: "Waiting for the device it ran on to come back (up to 5 min)",
    canStartNow: true,
  });
});

test("other waits say why, and can't be started now", () => {
  const [d] = parseDeployments(
    {
      d: deployment({
        waiting: { kind: "NotEnoughDevices", needed: 2, have: 1 },
      }),
    },
    new Set(),
    NOW,
  );
  assert.deepEqual(d.phase, {
    kind: "waiting",
    reason: "Needs 2 devices that have its files; 1 connected",
    canStartNow: false,
  });
});

test("a held deployment is paused with its reason", () => {
  const cases: Array<[Record<string, unknown>, string]> = [
    [
      { kind: "CrashLoop", message: "It lost a device 3 times in 10 minutes" },
      "Paused: it lost a device 3 times in 10 minutes",
    ],
    [
      { kind: "StartFailed", message: "out of memory" },
      "Paused: it failed to start (out of memory)",
    ],
    [
      { kind: "WaitedTooLong", message: "It couldn't start again within 1 h" },
      "Paused: it couldn't start again within 1 h",
    ],
    [
      { kind: "Restored", message: null },
      "Paused: exo restarted on every device. Resume to load it again",
    ],
  ];
  for (const [hold, reason] of cases) {
    const [d] = parseDeployments({ d: deployment({ hold }) }, new Set(), NOW);
    assert.deepEqual(d.phase, { kind: "paused", reason });
  }
});

test("a deployment with a running instance is running, whatever else it says", () => {
  // Never a Paused card next to the model's running one
  const [d] = parseDeployments(
    { d: deployment({ hold: { kind: "WaitedTooLong", message: "x" } }) },
    new Set(["d"]),
    NOW,
  );
  assert.deepEqual(d.phase, { kind: "running" });
});

test("a note says how a running model runs, for its instance's card", () => {
  const list = parseDeployments(
    {
      d: deployment({ note: "Running over TCP" }),
      e: deployment({ deploymentId: "e", note: "Waiting one" }),
    },
    new Set(["d"]),
    NOW,
  );
  assert.equal(list[0].note, "Running over TCP");
  // Only the running one: a waiting card shows its own note
  assert.deepEqual([...runningNotes(list)], [["d", "Running over TCP"]]);
});

test("anything unreadable is skipped", () => {
  assert.deepEqual(parseDeployments(undefined, new Set(), NOW), []);
  assert.deepEqual(
    parseDeployments({ x: null, y: { spec: {} } }, new Set(), NOW),
    [],
  );
});

test("a launch keeps what the user chose", () => {
  assert.deepEqual(launchConstraints(1, null, []), {
    min_nodes: 1,
    pinned_nodes: [],
  });
  assert.deepEqual(launchConstraints(2, null, ["a"]), {
    min_nodes: 2,
    pinned_nodes: ["a"],
  });
  // A preview the user picked fixes its number of devices
  assert.deepEqual(launchConstraints(1, 3, new Set(["a", "b"])), {
    min_nodes: 3,
    pinned_nodes: ["a", "b"],
  });
});

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseNotices,
  stoppedModelMessage,
  whyModelStopped,
  withNotice,
  withThisDeviceId,
  withoutNotice,
  type StopContext,
} from "../src/lib/utils/stopped_models.ts";

const names: Record<string, string> = {
  "old-a": "studio-a",
  "new-a": "studio-a",
  c: "max-c",
  d: "max-d",
};

function context(overrides: Partial<StopContext>): StopContext {
  const connected = overrides.connectedNodeIds ?? new Set(["old-a", "c", "d"]);
  return {
    nodeIds: ["old-a", "c", "d"],
    connectedNodeIds: connected,
    nameOf: (nodeId) => names[nodeId],
    thisDeviceIds: new Set(),
    ...overrides,
  };
}

test("a device that left, by name", () => {
  const reason = whyModelStopped(
    context({ connectedNodeIds: new Set(["old-a", "d"]) }),
  );
  assert.deepEqual(reason, { kind: "device-left", device: "max-c" });
  assert.equal(
    stoppedModelMessage("Qwen3 30B", reason),
    "Qwen3 30B stopped — max-c left the cluster",
  );
});

test("exo restarted on the dashboard's own device is not a device leaving", () => {
  const reason = whyModelStopped(
    context({
      connectedNodeIds: new Set(["new-a", "c", "d"]),
      thisDeviceIds: new Set(["old-a", "new-a"]),
    }),
  );
  assert.deepEqual(reason, { kind: "this-device-restarted" });
  assert.equal(
    stoppedModelMessage("Qwen3 30B", reason),
    "Qwen3 30B stopped — exo restarted on this device",
  );
});

test("a device back under a new identity counts as having left", () => {
  // Its new identity can't be linked to the old one: names aren't unique
  const reason = whyModelStopped(
    context({ connectedNodeIds: new Set(["new-a", "c", "d"]) }),
  );
  assert.deepEqual(reason, { kind: "device-left", device: "studio-a" });
});

test("all devices still here: just stopped", () => {
  assert.deepEqual(whyModelStopped(context({})), { kind: "stopped" });
});

test("no cluster in view says nothing about devices", () => {
  assert.deepEqual(whyModelStopped(context({ connectedNodeIds: new Set() })), {
    kind: "stopped",
  });
});

test("an unnamed device that left", () => {
  const reason = whyModelStopped(
    context({ nodeIds: ["x"], connectedNodeIds: new Set(["c"]) }),
  );
  assert.equal(
    stoppedModelMessage("Llama", reason),
    "Llama stopped — a device left the cluster",
  );
});

test("notices: one per model, removable, robust to bad storage", () => {
  let notices = withNotice([], { modelId: "m", message: "old", at: 1 });
  notices = withNotice(notices, { modelId: "m", message: "new", at: 2 });
  notices = withNotice(notices, { modelId: "n", message: "other", at: 3 });
  assert.deepEqual(
    notices.map((n) => n.message),
    ["new", "other"],
  );
  assert.deepEqual(
    withoutNotice(notices, "m").map((n) => n.modelId),
    ["n"],
  );
  assert.deepEqual(parseNotices(JSON.stringify(notices)), notices);
  assert.deepEqual(parseNotices("not json"), []);
  assert.deepEqual(parseNotices('[{"modelId": 1}]'), []);
  assert.deepEqual(parseNotices(null), []);
});

test("this device's ids: newest last, bounded", () => {
  let ids: string[] = [];
  for (let i = 0; i < 25; i++) ids = withThisDeviceId(ids, `id-${i}`);
  ids = withThisDeviceId(ids, "id-10");
  assert.equal(ids.length, 20);
  assert.equal(ids[ids.length - 1], "id-10");
});

// Device names are not identities. macOS gives every Mac a user sets up
// the ComputerName "<Name>'s Mac Studio"; two identical Macs in one cluster can
// share it. When one of them leaves, the other's name is still in the cluster,
// which must not make the notice say exo restarted on it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { whyModelStopped } from "../src/lib/utils/stopped_models.ts";

test("a device that left is not called restarted because another device shares its name", () => {
  const names: Record<string, string> = {
    a: "Sam's Mac Studio",
    b: "Sam's Mac Studio",
  };
  const reason = whyModelStopped({
    nodeIds: ["a", "b"],
    connectedNodeIds: new Set(["b"]), // a left; nothing new joined
    nameOf: (id) => names[id],
    thisDeviceIds: new Set(),
  });
  assert.equal(reason.kind, "device-left");
});

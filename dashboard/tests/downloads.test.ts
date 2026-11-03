import { test } from "node:test";
import assert from "node:assert/strict";
import {
  deleteDownloadQuestion,
  isModelRunningOnNode,
} from "../src/lib/utils/downloads.ts";

const instances = {
  "inst-1": {
    MlxRingInstance: {
      shardAssignments: {
        modelId: "org/qwen",
        nodeToRunner: { "node-a": "runner-0", "node-c": "runner-1" },
        runnerToShard: {},
      },
    },
  },
};

test("a model runs on the nodes its instance has runners on", () => {
  assert.equal(isModelRunningOnNode(instances, "node-a", "org/qwen"), true);
  assert.equal(isModelRunningOnNode(instances, "node-d", "org/qwen"), false);
  assert.equal(isModelRunningOnNode(instances, "node-a", "org/llama"), false);
  assert.equal(isModelRunningOnNode({}, "node-a", "org/qwen"), false);
});

test("deleting a running model's files says it is running", () => {
  const question = deleteDownloadQuestion({
    modelName: "Qwen3 30B",
    deviceName: "studio-a",
    size: "16 GB",
    running: true,
  });
  assert.match(question, /^Delete Qwen3 30B \(16 GB\) from studio-a\?/);
  assert.match(question, /running on studio-a/);
  assert.match(question, /download it again the next time it starts/);
});

test("deleting files of a model that isn't running", () => {
  assert.equal(
    deleteDownloadQuestion({
      modelName: "Qwen3 30B",
      deviceName: "studio-a",
      size: null,
      running: false,
    }),
    "Delete Qwen3 30B from studio-a?\n\nYou'll need to download it again to run it on this device.",
  );
});

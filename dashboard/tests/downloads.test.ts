import { test } from "node:test";
import assert from "node:assert/strict";
import {
  deleteDownloadQuestion,
  modelStateOnNode,
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

test("a model runs where its runner has it in memory, starts where it doesn't yet", () => {
  const runners = {
    "runner-0": { RunnerReady: {} },
    "runner-1": { RunnerLoading: { layersLoaded: 3, totalLayers: 48 } },
  };
  assert.equal(
    modelStateOnNode(instances, runners, "node-a", "org/qwen"),
    "running",
  );
  assert.equal(
    modelStateOnNode(instances, runners, "node-c", "org/qwen"),
    "starting",
  );
  assert.equal(
    modelStateOnNode(instances, runners, "node-d", "org/qwen"),
    null,
  );
  assert.equal(
    modelStateOnNode(instances, runners, "node-a", "org/llama"),
    null,
  );
  // No runner status yet (still downloading): starting
  assert.equal(
    modelStateOnNode(instances, {}, "node-a", "org/qwen"),
    "starting",
  );
  // A runner that failed or is stopping: neither
  assert.equal(
    modelStateOnNode(
      instances,
      { "runner-0": { RunnerFailed: {} } },
      "node-a",
      "org/qwen",
    ),
    null,
  );
});

test("deleting a running model's files says it keeps running from memory", () => {
  const question = deleteDownloadQuestion({
    modelName: "Qwen3 30B",
    deviceName: "studio-a",
    size: "16 GB",
    state: "running",
  });
  assert.match(question, /^Delete Qwen3 30B \(16 GB\) from studio-a\?/);
  assert.match(question, /running on studio-a/);
  assert.match(question, /keeps running from memory/);
});

test("deleting a starting model's files doesn't say it keeps running", () => {
  const question = deleteDownloadQuestion({
    modelName: "Qwen3 30B",
    deviceName: "studio-a",
    size: "6.8 GB downloaded",
    state: "starting",
  });
  assert.match(question, /starting on studio-a, which needs these files/);
  assert.doesNotMatch(question, /keeps running/);
});

test("deleting files of a model that isn't running", () => {
  assert.equal(
    deleteDownloadQuestion({
      modelName: "Qwen3 30B",
      deviceName: "studio-a",
      size: null,
      state: null,
    }),
    "Delete Qwen3 30B from studio-a?\n\nYou'll need to download it again to run it on this device.",
  );
});

// Cases where a model's download total used to go back.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  totalDownload,
  type NodeDownload,
} from "../src/lib/utils/download_progress.ts";

const GB = 1024 ** 3;
const ongoing = (
  nodeId: string,
  dlGB: number,
  totGB: number,
): NodeDownload => ({
  nodeId,
  state: "downloading",
  downloadedBytes: dlGB * GB,
  totalBytes: totGB * GB,
  speed: 100 * 1024 * 1024,
});
const completed = (nodeId: string, totGB: number): NodeDownload => ({
  nodeId,
  state: "completed",
  downloadedBytes: 0,
  totalBytes: totGB * GB,
  speed: 0,
});

// A model whose vision weights live in another repo (the Kimi K2.x cards):
// the worker downloads the main repo, then reports the vision repo's own
// bytes under the same model id, and its DownloadCompleted carries the vision
// repo's total. (impl_shard_downloader.py ensure_shard: vision_progress is
// sent as the main shard's progress.)
test("a device moving from the main repo to the vision repo doesn't take the total back", () => {
  const W = 600; // the model's main repo per device
  const before = totalDownload(
    [
      ongoing("a", 600, W),
      ongoing("b", 500, W),
      ongoing("c", 500, W),
      ongoing("d", 400, W),
    ],
    W * GB,
  );
  // a finished the main repo and now downloads the 1 GB vision repo
  const after = totalDownload(
    [
      ongoing("a", 0, 1),
      ongoing("b", 500, W),
      ongoing("c", 500, W),
      ongoing("d", 400, W),
    ],
    W * GB,
  );
  assert.ok(
    after.percentage >= before.percentage,
    `${before.percentage.toFixed(1)}% -> ${after.percentage.toFixed(1)}%`,
  );
});

test("a device whose real size is above the card's estimate doesn't take the total back", () => {
  // c hasn't reported yet: estimated as the card's storage size (16 GB);
  // it then reports the repo's real size, 20 GB (extra files in the repo)
  const before = totalDownload(
    [
      ongoing("a", 10, 20),
      ongoing("b", 10, 20),
      {
        nodeId: "c",
        state: "missing",
        downloadedBytes: 0,
        totalBytes: 0,
        speed: 0,
      },
    ],
    16 * GB,
  );
  const after = totalDownload(
    [ongoing("a", 10, 20), ongoing("b", 10, 20), ongoing("c", 0.1, 20)],
    16 * GB,
  );
  assert.ok(
    after.percentage >= before.percentage,
    `${before.percentage.toFixed(1)}% -> ${after.percentage.toFixed(1)}%`,
  );
});

test("a completed device counts as the whole model, not its last repo", () => {
  const total = totalDownload(
    [completed("a", 1), ongoing("b", 300, 600)],
    600 * GB,
  );
  // a has all 600 GB (+1 GB vision); b is halfway: about 75%, not 50%
  assert.ok(total.percentage > 70, `${total.percentage.toFixed(1)}%`);
});

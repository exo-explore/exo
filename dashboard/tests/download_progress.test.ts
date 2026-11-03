import { test } from "node:test";
import assert from "node:assert/strict";
import {
  totalDownload,
  type NodeDownload,
} from "../src/lib/utils/download_progress.ts";

const GB = 1024 ** 3;
const MODEL = 16 * GB;

const node = (
  nodeId: string,
  state: NodeDownload["state"],
  downloadedGB = 0,
  totalGB = 0,
  speedMBs = 0,
): NodeDownload => ({
  nodeId,
  state,
  downloadedBytes: downloadedGB * GB,
  totalBytes: totalGB * GB,
  speed: speedMBs * 1024 * 1024,
});

test("finished devices still count", () => {
  const total = totalDownload(
    [node("a", "completed", 0, 16), node("b", "downloading", 8, 16, 50)],
    MODEL,
  );
  assert.equal(total.percentage, 75);
  assert.equal(total.totalBytes, 32 * GB);
});

test("a device that hasn't started counts as the whole model", () => {
  const total = totalDownload(
    [node("a", "downloading", 8, 16, 50), node("b", "missing")],
    MODEL,
  );
  assert.equal(total.percentage, 25);
  assert.equal(total.etaMs, null);
});

// The 3-device download where the total went 62.8 -> 48.1 -> 73.8 -> 67.5 -> 44.4
test("the total never goes back while bytes are added", () => {
  const snapshots: NodeDownload[][] = [
    // c hasn't started (it waited on rate limiting)
    [
      node("a", "downloading", 9, 16, 60),
      node("b", "downloading", 11, 16, 60),
      node("c", "missing"),
    ],
    // c starts late
    [
      node("a", "downloading", 10, 16, 60),
      node("b", "downloading", 12, 16, 60),
      node("c", "downloading", 0.1, 16, 60),
    ],
    [
      node("a", "downloading", 15, 16, 60),
      node("b", "downloading", 15.5, 16, 60),
      node("c", "downloading", 4, 16, 60),
    ],
    // a finishes
    [
      node("a", "completed", 0, 16),
      node("b", "downloading", 15.9, 16, 60),
      node("c", "downloading", 5, 16, 60),
    ],
    // b finishes
    [
      node("a", "completed", 0, 16),
      node("b", "completed", 0, 16),
      node("c", "downloading", 6, 16, 60),
    ],
    [
      node("a", "completed", 0, 16),
      node("b", "completed", 0, 16),
      node("c", "downloading", 16, 16, 60),
    ],
  ];
  let previous = -1;
  for (const snapshot of snapshots) {
    const { percentage } = totalDownload(snapshot, MODEL);
    assert.ok(
      percentage >= previous,
      `went back from ${previous.toFixed(1)}% to ${percentage.toFixed(1)}%`,
    );
    previous = percentage;
  }
  assert.equal(previous, 100);
});

test("a device reporting less than the card's size only moves the total forward", () => {
  // The card overstates the repo a little: 16 GB on the card, 15 GB to fetch
  const before = totalDownload(
    [node("a", "downloading", 2, 15, 50), node("b", "missing")],
    MODEL,
  );
  const after = totalDownload(
    [node("a", "downloading", 2, 15, 50), node("b", "downloading", 0, 15, 50)],
    MODEL,
  );
  assert.ok(after.percentage >= before.percentage);
});

test("a small repo fetched after the model's counts as the model done", () => {
  // Vision weights in their own repo, reported under the model's id
  const total = totalDownload(
    [node("a", "downloading", 0.2, 1, 50), node("b", "downloading", 8, 16, 50)],
    MODEL,
  );
  assert.equal(total.percentage, 75);
  // Its remaining 0.8 GB still counts for the ETA
  assert.ok(total.etaMs! > 0);
});

test("the ETA is the slowest device's, not the sum over all", () => {
  const total = totalDownload(
    [
      node("a", "completed", 0, 16),
      node("b", "downloading", 8, 16, 100), // 8 GB left at 100 MB/s
      node("c", "downloading", 15.5, 16, 10), // 0.5 GB left at 10 MB/s
    ],
    MODEL,
  );
  assert.equal(Math.round(total.etaMs! / 1000), Math.round((8 * 1024) / 100));
  assert.equal(total.speed, 110 * 1024 * 1024);
});

test("a device that isn't moving makes the ETA unknown", () => {
  const total = totalDownload(
    [node("a", "downloading", 8, 16, 100), node("b", "downloading", 2, 16, 0)],
    MODEL,
  );
  assert.equal(total.etaMs, null);
});

test("all finished", () => {
  const total = totalDownload(
    [node("a", "completed", 0, 16), node("b", "completed", 0, 16)],
    MODEL,
  );
  assert.equal(total.percentage, 100);
  assert.equal(total.etaMs, 0);
});

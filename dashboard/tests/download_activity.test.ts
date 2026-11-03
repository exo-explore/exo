import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DOWNLOAD_STALL_MS,
  DownloadActivityTracker,
  downloadKey,
  estimateRemainingMs,
  isStalled,
} from "../src/lib/utils/download_activity.ts";

const MB = 1024 * 1024;
const KEY = downloadKey("max-c", "org/model");

// Poll once a second, with the bytes the device reports at each second
function run(bytesAt: (second: number) => number, seconds: number) {
  const tracker = new DownloadActivityTracker();
  let activity;
  for (let s = 0; s <= seconds; s++) {
    activity = tracker.observe({ [KEY]: bytesAt(s) }, s * 1000)[KEY];
  }
  return activity!;
}

test("a moving download has a rate and an ETA", () => {
  const activity = run((s) => s * 50 * MB, 12);
  assert.equal(Math.round(activity.rate! / MB), 50);
  assert.equal(isStalled(activity, 12_000), false);
  // 500 MB left at 50 MB/s
  assert.equal(estimateRemainingMs(500 * MB, activity, 12_000), 10_000);
});

test("no ETA in the first seconds of a download", () => {
  // The device's own average is wild here ("ETA 17143h")
  const activity = run((s) => s * 50 * MB, 3);
  assert.equal(activity.rate, null);
  assert.equal(estimateRemainingMs(500 * MB, activity, 3_000), null);
});

test("a download whose bytes stop going up is stalled after 15 s", () => {
  const bytes = (s: number) => Math.min(s, 10) * 50 * MB; // stops at 10 s
  let activity = run(bytes, 10 + 14);
  assert.equal(isStalled(activity, 24_000), false);
  activity = run(bytes, 10 + 15);
  assert.equal(isStalled(activity, 25_000), true);
  assert.equal(estimateRemainingMs(500 * MB, activity, 25_000), null);
  assert.equal(activity.rate, 0);
});

test("a download stuck from the start (nothing reported) is stalled too", () => {
  const activity = run(() => 0, 16);
  assert.equal(isStalled(activity, 16_000), true);
});

test("moving again ends the stall", () => {
  const tracker = new DownloadActivityTracker();
  for (let s = 0; s <= 20; s++) tracker.observe({ [KEY]: 0 }, s * 1000);
  const activity = tracker.observe({ [KEY]: 10 * MB }, 21_000)[KEY];
  assert.equal(isStalled(activity, 21_000), false);
});

test("a download that starts over is timed afresh", () => {
  const tracker = new DownloadActivityTracker();
  for (let s = 0; s <= 10; s++)
    tracker.observe({ [KEY]: s * 50 * MB }, s * 1000);
  const activity = tracker.observe({ [KEY]: 0 }, 11_000)[KEY];
  assert.equal(activity.rate, null);
  assert.equal(activity.progressAt, 11_000);
});

test("finished downloads are forgotten", () => {
  const tracker = new DownloadActivityTracker();
  tracker.observe({ [KEY]: 1 }, 0);
  assert.deepEqual(tracker.observe({}, 1000), {});
  // Back again later: a new download
  const activity = tracker.observe({ [KEY]: 1 }, DOWNLOAD_STALL_MS * 2)[KEY];
  assert.equal(isStalled(activity, DOWNLOAD_STALL_MS * 2), false);
});

test("a stall says what is known about it, and no more", async () => {
  const { describeStall } =
    await import("../src/lib/utils/download_activity.ts");
  const activity = { progressAt: 0, rate: 0 };
  assert.equal(
    describeStall(activity, 16 * 1024 * MB, 45_000),
    "Stalled: no data for 45s",
  );
  assert.equal(
    describeStall(activity, 16 * 1024 * MB, 80_000),
    "Stalled: no data for 1m 20s",
  );
  assert.equal(
    describeStall(activity, 0, 45_000),
    "Stalled: waiting for the model's file list",
  );
  assert.doesNotMatch(describeStall(activity, 1, 1), /retry/i);
});

test("several stalled devices: one line for the model", async () => {
  const { describeStalls } =
    await import("../src/lib/utils/download_activity.ts");
  // All still at 0 of 0 bytes (Hugging Face unreachable)
  assert.equal(
    describeStalls(
      [
        { activity: { progressAt: 0, rate: null }, totalBytes: 0 },
        { activity: { progressAt: 0, rate: null }, totalBytes: 0 },
      ],
      30_000,
    ),
    "Stalled: waiting for the model's file list",
  );
  // The most recent data from any device counts
  assert.equal(
    describeStalls(
      [
        { activity: { progressAt: 10_000, rate: 0 }, totalBytes: 16 * MB },
        { activity: { progressAt: 40_000, rate: 0 }, totalBytes: 16 * MB },
      ],
      60_000,
    ),
    "Stalled: no data for 20s",
  );
});

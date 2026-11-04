import { test, mock } from "node:test";
import assert from "node:assert/strict";
import {
  describeDevice,
  fetchDevicesOnOtherVersions,
  parseDevicesOnOtherVersions,
  pollDevicesOnOtherVersions,
  type DeviceOnOtherVersion,
} from "../src/lib/utils/other_versions.ts";

const device: DeviceOnOtherVersion = {
  addresses: ["fe80::1", "fe80::2", "fe80::3"],
  reason: "wire_version",
  wireVersion: 2,
};

test("reads one entry per device", () => {
  assert.deepEqual(
    parseDevicesOnOtherVersions({
      wireVersion: 1,
      buildId: "1.0.72 (4f1c2a9)",
      devices: [
        {
          addresses: ["fe80::1", "fe80::2", "fe80::3"],
          reason: "wire_version",
          wireVersion: 2,
        },
        { addresses: ["fe80::9"], reason: "build", wireVersion: null },
      ],
    }),
    [device, { addresses: ["fe80::9"], reason: "build", wireVersion: null }],
  );
});

test("an unexpected answer reads as no devices", () => {
  assert.deepEqual(parseDevicesOnOtherVersions(null), []);
  assert.deepEqual(parseDevicesOnOtherVersions({ detail: "Not Found" }), []);
  assert.deepEqual(
    parseDevicesOnOtherVersions({ devices: [{ address: "fe80::1" }] }),
    [],
  );
  assert.deepEqual(
    parseDevicesOnOtherVersions({
      devices: [{ addresses: ["fe80::1"], reason: "other" }],
    }),
    [],
  );
});

test("says what each device runs, once per device", () => {
  assert.equal(
    describeDevice(device),
    "fe80::1 (and 2 more addresses) runs exo wire version 2",
  );
  assert.equal(
    describeDevice({
      addresses: ["fe80::9"],
      reason: "build",
      wireVersion: null,
    }),
    "fe80::9 runs another build of exo",
  );
  assert.equal(
    describeDevice({
      addresses: ["fe80::4", "fe80::5"],
      reason: "older_exo",
      wireVersion: null,
    }),
    "fe80::4 (and 1 more address) runs an older exo",
  );
});

test("an ask that takes too long is given up", async () => {
  const neverAnswers = ((_url: string, init?: RequestInit) =>
    new Promise((_, reject) =>
      init?.signal?.addEventListener("abort", () =>
        reject(init.signal?.reason),
      ),
    )) as typeof fetch;
  const started = Date.now();
  assert.equal(await fetchDevicesOnOtherVersions(20, neverAnswers), null);
  assert.ok(Date.now() - started < 1_000);
});

test("never asks twice at once, and stops when told", async () => {
  mock.timers.enable({ apis: ["setInterval"] });
  try {
    const answers: Array<(devices: DeviceOnOtherVersion[] | null) => void> = [];
    const seen: DeviceOnOtherVersion[][] = [];
    const stop = pollDevicesOnOtherVersions(
      (devices) => seen.push(devices),
      10,
      () => new Promise((resolve) => answers.push(resolve)),
    );
    // The first ask is still out: three ticks don't start another
    mock.timers.tick(30);
    assert.equal(answers.length, 1);
    answers[0]([device]);
    await Promise.resolve();
    await Promise.resolve();
    assert.deepEqual(seen, [[device]]);
    mock.timers.tick(10);
    assert.equal(answers.length, 2);
    // A failed ask keeps the last list
    answers[1](null);
    await Promise.resolve();
    await Promise.resolve();
    assert.deepEqual(seen, [[device]]);
    // An answer that arrives after stopping is dropped
    mock.timers.tick(10);
    stop();
    answers[2]([]);
    await Promise.resolve();
    await Promise.resolve();
    assert.deepEqual(seen, [[device]]);
    mock.timers.tick(100);
    assert.equal(answers.length, 3);
  } finally {
    mock.timers.reset();
  }
});

test("three failed asks in a row drop the list", async () => {
  mock.timers.enable({ apis: ["setInterval"] });
  try {
    const answers: Array<(devices: DeviceOnOtherVersion[] | null) => void> = [];
    const seen: DeviceOnOtherVersion[][] = [];
    const stop = pollDevicesOnOtherVersions(
      (devices) => seen.push(devices),
      10,
      () => new Promise((resolve) => answers.push(resolve)),
    );
    const settle = async () => {
      await Promise.resolve();
      await Promise.resolve();
    };
    answers[0]([device]);
    await settle();
    // Two failures keep the list; the third drops it
    for (const ask of [1, 2, 3]) {
      mock.timers.tick(10);
      answers[ask](null);
      await settle();
    }
    assert.deepEqual(seen, [[device], []]);
    // A success counts again from zero
    mock.timers.tick(10);
    answers[4]([device]);
    await settle();
    mock.timers.tick(10);
    answers[5](null);
    await settle();
    assert.deepEqual(seen, [[device], [], [device]]);
    stop();
  } finally {
    mock.timers.reset();
  }
});

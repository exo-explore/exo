import { test } from "node:test";
import assert from "node:assert/strict";
import {
  checkDiskFit,
  describeDiskFit,
  type DeviceSpace,
} from "../src/lib/utils/disk_fit.ts";

const GB = 1024 ** 3;
const device = (
  name: string,
  freeDiskGB: number | null,
  freeMemoryGB: number,
  hasModel = false,
  downloadedGB = 0,
): DeviceSpace => ({
  name,
  freeDiskBytes: freeDiskGB === null ? null : freeDiskGB * GB,
  freeMemoryBytes: freeMemoryGB * GB,
  hasModel,
  downloadedBytes: downloadedGB * GB,
});

test("one Mac with 108 GB free disk: a 155 GB model doesn't fit", () => {
  const fit = checkDiskFit(155 * GB, [device("studio-a", 108, 200)]);
  assert.equal(fit.fits, false);
  assert.equal(
    describeDiskFit(fit),
    "Not enough disk space: needs 155 GB, 108 GB free on studio-a",
  );
});

test("a model that fits on disk and in memory", () => {
  assert.deepEqual(checkDiskFit(16 * GB, [device("studio-a", 108, 200)]), {
    fits: true,
  });
});

test("a cluster where one device is short on disk", () => {
  // 330 GB model: needs all three devices' memory, but max-c has 88 GB free
  const devices = [
    device("studio-a", 337, 200),
    device("max-c", 88, 50),
    device("max-d", 400, 100),
  ];
  const fit = checkDiskFit(330 * GB, devices);
  assert.equal(fit.fits, false);
  assert.match(describeDiskFit(fit)!, /free on max-c/);
});

test("a smaller model can leave the device short on disk out", () => {
  const devices = [
    device("studio-a", 337, 200),
    device("max-c", 88, 50),
    device("max-d", 255, 100),
  ];
  assert.equal(checkDiskFit(150 * GB, devices).fits, true);
});

test("a device that has the model already needs no space for it", () => {
  assert.equal(
    checkDiskFit(155 * GB, [device("studio-a", 1, 200, true)]).fits,
    true,
  );
});

test("devices that haven't reported their disk aren't held against a model", () => {
  assert.equal(
    checkDiskFit(155 * GB, [device("studio-a", null, 200)]).fits,
    true,
  );
});

test("a model partly downloaded needs only the rest", () => {
  // Hy3, 155 GB: 80 GB in, so 75 GB more; 108 GB free
  assert.equal(
    checkDiskFit(155 * GB, [device("studio-a", 108, 200, false, 80)]).fits,
    true,
  );
  // 30 GB in, 125 GB more: doesn't fit, and says how much more
  const fit = checkDiskFit(155 * GB, [device("studio-a", 108, 200, false, 30)]);
  assert.equal(
    describeDiskFit(fit),
    "Not enough disk space: needs 125 GB more, 108 GB free on studio-a",
  );
});

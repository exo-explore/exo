import { test } from "node:test";
import assert from "node:assert/strict";
import { formatBytes } from "../src/lib/utils/format_bytes.ts";

test("formats with binary units", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(1024 ** 3), "1 GB");
  assert.equal(formatBytes(1536, 2, ""), "1.5KB");
});

test("moves to the next unit when the value rounds up to 1024", () => {
  assert.equal(formatBytes(1024 * 1024 - 1), "1 MB");
  assert.equal(formatBytes(1024 ** 3 - 1024 ** 2 / 50), "1 GB");
  assert.equal(formatBytes(1024 * 1024 - 1, 2), "1 MB");
  assert.equal(formatBytes(1024 * 1024 - 600, 2), "1023.41 KB");
});

test("stays in the largest unit past it", () => {
  assert.equal(formatBytes(2048 * 1024 ** 4), "2048 TB");
});

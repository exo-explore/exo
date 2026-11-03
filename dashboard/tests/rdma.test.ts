import { test } from "node:test";
import assert from "node:assert/strict";
import {
  devicesMissingRdma,
  isThunderbolt5Link,
} from "../src/lib/utils/rdma.ts";

// What Mac Studios report (from a real cluster's /state): three ports cabled
// to other Macs, three idle
const cabled = {
  interfaces: [
    { linkSpeed: "Up to 120 Gb/s" },
    { linkSpeed: "Up to 120 Gb/s" },
    { linkSpeed: "80 Gb/s" },
    { linkSpeed: "80 Gb/s" },
    { linkSpeed: "80 Gb/s" },
    { linkSpeed: "Up to 120 Gb/s" },
  ],
};
// The same Macs on Ethernet: every port idle
const idle = {
  interfaces: [
    { linkSpeed: "Up to 120 Gb/s" },
    { linkSpeed: "Up to 120 Gb/s" },
    { linkSpeed: "Up to 120 Gb/s" },
  ],
};

test("a live Thunderbolt 5 link, an idle port, Thunderbolt 4", () => {
  assert.equal(isThunderbolt5Link({ linkSpeed: "80 Gb/s" }), true);
  assert.equal(isThunderbolt5Link({ linkSpeed: "120 Gb/s" }), true);
  assert.equal(isThunderbolt5Link({ linkSpeed: "Up to 120 Gb/s" }), false);
  assert.equal(isThunderbolt5Link({ linkSpeed: "40 Gb/s" }), false);
  assert.equal(isThunderbolt5Link({ linkSpeed: "" }), false);
  assert.equal(isThunderbolt5Link({}), false);
});

test("a cluster on Ethernet has nothing to enable", () => {
  assert.deepEqual(
    devicesMissingRdma(
      { a: idle, b: idle, c: idle },
      { a: { enabled: false }, b: { enabled: false }, c: { enabled: false } },
    ),
    [],
  );
});

test("Macs cabled with Thunderbolt 5 without RDMA", () => {
  assert.deepEqual(
    devicesMissingRdma(
      { a: cabled, b: cabled, c: idle },
      { a: { enabled: true }, b: { enabled: false } },
    ),
    ["b"],
  );
});

test("RDMA enabled on all cabled Macs", () => {
  assert.deepEqual(
    devicesMissingRdma(
      { a: cabled, b: cabled },
      { a: { enabled: true }, b: { enabled: true } },
    ),
    [],
  );
});

test("one cabled Mac alone (e.g. to a display) isn't a cluster link", () => {
  assert.deepEqual(
    devicesMissingRdma({ a: cabled, b: idle }, { a: { enabled: false } }),
    [],
  );
});

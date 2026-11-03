import { test } from "node:test";
import assert from "node:assert/strict";
import { describeStreamError } from "../src/lib/utils/stream_errors.ts";

// What a 3-device chat streamed when one device quit mid-reply
const RING_TRACEBACK = [
  "Runner shutdown before completing command (exitcode=1",
  "Runner error: RuntimeError: [ring] connection to a peer was lost",
  "Traceback (most recent call last):",
  '  File "exo/worker/runner/bootstrap.py", line 113, in entrypoint',
  '  File "exo/worker/runner/runner.py", line 221, in main',
  "RuntimeError: [ring] connection to a peer was lost",
  ")",
].join("\n");

test("a broken ring connection says a device left", () => {
  assert.equal(
    describeStreamError(RING_TRACEBACK),
    "A device left the cluster before the reply finished.",
  );
});

test("never shows a traceback", () => {
  const message = describeStreamError(
    [
      "Runner shutdown before completing command (exitcode=1",
      "Runner error: ValueError: tokenizer failed",
      "Traceback (most recent call last):",
      '  File "exo/worker/runner/runner.py", line 221, in main',
      "ValueError: tokenizer failed",
      ")",
    ].join("\n"),
  );
  assert.equal(
    message,
    "The model stopped before the reply finished: tokenizer failed.",
  );
  assert.ok(!message.includes("Traceback"));
  assert.ok(!message.includes("File "));
});

test("a runner stopped by its supervisor gives the reason without the parenthesis", () => {
  assert.equal(
    describeStreamError(
      "Runner shutdown before completing command (exitcode=1\nRunner error: Runner stopped responding)",
    ),
    "The model stopped before the reply finished: Runner stopped responding.",
  );
});

test("a runner killed by a signal", () => {
  assert.equal(
    describeStreamError(
      "Runner shutdown before completing command (signal=9 (Killed))",
    ),
    "The model stopped unexpectedly before the reply finished.",
  );
});

test("a short server message is kept as it is", () => {
  assert.equal(
    describeStreamError(
      "The model instance serving this request stopped before it finished",
    ),
    "The model instance serving this request stopped before it finished.",
  );
  assert.equal(
    describeStreamError("No instance found for model foo/bar."),
    "No instance found for model foo/bar.",
  );
});

test("the browser losing the connection", () => {
  for (const message of [
    "Failed to fetch",
    "Load failed",
    "network error",
    "NetworkError when attempting to fetch resource.",
  ]) {
    assert.equal(
      describeStreamError(message),
      "Lost the connection to exo before the reply finished.",
    );
  }
});

test("images are worded as images", () => {
  assert.equal(
    describeStreamError(RING_TRACEBACK, "image"),
    "A device left the cluster before the image finished.",
  );
});

test("long messages are cut to one short line", () => {
  const message = describeStreamError("x".repeat(1000));
  assert.ok(message.length <= 201, `${message.length} characters`);
  assert.ok(message.endsWith("…"));
});

test("an empty message still says something", () => {
  assert.equal(
    describeStreamError(""),
    "Something went wrong before the reply finished.",
  );
});

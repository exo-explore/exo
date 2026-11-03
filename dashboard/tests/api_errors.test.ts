import { test } from "node:test";
import assert from "node:assert/strict";
import { extractApiErrorMessage } from "../src/lib/utils/api_errors.ts";

test("reads an OpenAI-style error body", () => {
  assert.equal(
    extractApiErrorMessage({ error: { message: "No instance found" } }),
    "No instance found",
  );
});

test("reads a FastAPI detail body", () => {
  assert.equal(extractApiErrorMessage({ detail: "Not found" }), "Not found");
});

test("returns null for a body without an error", () => {
  assert.equal(extractApiErrorMessage({ choices: [] }), null);
  assert.equal(extractApiErrorMessage("text"), null);
});

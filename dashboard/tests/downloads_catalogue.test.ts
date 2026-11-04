import { test } from "node:test";
import assert from "node:assert/strict";
import {
  catalogueModelInfo,
  modelDisplayName,
} from "../src/lib/utils/downloads.ts";

test("names a catalogue model as a download entry would", () => {
  const { prettyName, card } = catalogueModelInfo({
    id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
    family: "llama",
    base_model: "Llama 3.2 1B",
    quantization: "4bit",
    capabilities: ["text"],
    storage_size_megabytes: 695,
    supports_tensor: true,
  });
  assert.equal(prettyName, "Llama 3.2 1B · 4bit");
  assert.deepEqual(card, {
    family: "llama",
    quantization: "4bit",
    baseModel: "Llama 3.2 1B",
    capabilities: ["text"],
    storageSize: 695 * 1024 * 1024,
    nLayers: 0,
    supportsTensor: true,
  });
});

test("a catalogue model without details still gets a row", () => {
  const { prettyName, card } = catalogueModelInfo({ id: "org/custom" });
  assert.equal(prettyName, null);
  assert.equal(card.storageSize, 0);
  assert.deepEqual(card.capabilities, []);
});

test("display names", () => {
  assert.equal(modelDisplayName("Qwen3 8B", ""), "Qwen3 8B");
  assert.equal(modelDisplayName("", "4bit"), null);
});

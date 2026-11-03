import { test } from "node:test";
import assert from "node:assert/strict";
import { indexModelDownloads } from "../src/lib/utils/downloads.ts";

const GB = 1024 ** 3;
const shard = (modelId: string) => ({
  PipelineShardMetadata: { modelCard: { modelId } },
});

test("each device's bytes of each model, in one pass", () => {
  const index = indexModelDownloads({
    a: [
      {
        DownloadCompleted: {
          shardMetadata: shard("org/m"),
          total: { inBytes: 16 * GB },
        },
      },
      {
        DownloadPending: {
          shardMetadata: shard("org/other"),
          downloaded: { inBytes: 0 },
          total: { inBytes: 5 * GB },
        },
      },
    ],
    b: [
      {
        DownloadOngoing: {
          shardMetadata: shard("org/m"),
          downloadProgress: { downloaded: { inBytes: 6 * GB } },
        },
      },
      // The same model listed twice: the most it has counts
      {
        DownloadPending: {
          shardMetadata: shard("org/m"),
          downloaded: { inBytes: 2 * GB },
        },
      },
    ],
  });
  assert.deepEqual(index.get("org/m")?.get("a"), {
    completed: true,
    downloadedBytes: 16 * GB,
  });
  assert.deepEqual(index.get("org/m")?.get("b"), {
    completed: false,
    downloadedBytes: 6 * GB,
  });
  assert.equal(index.get("org/other")?.get("a")?.downloadedBytes, 0);
  assert.equal(index.get("org/none"), undefined);
});

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  TOMBSTONE_TTL_MS,
  mergeConversations,
  mergeTombstones,
  sameConversations,
} from "../src/lib/utils/conversation_sync.ts";

interface Conversation {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
}

const conv = (id: string, createdAt: number, updatedAt = createdAt) => ({
  id,
  name: id,
  createdAt,
  updatedAt,
});

test("a conversation another tab created is kept", () => {
  // Tab B created "b" and saved; tab A, which never saw it, now saves
  const tabA = [conv("a", 1, 50)];
  const stored = [conv("b", 2, 10), conv("a", 1, 5)];
  const merged = mergeConversations<Conversation>(tabA, stored, {});
  assert.deepEqual(
    merged.map((c) => c.id),
    ["b", "a"],
  );
});

test("the version updated last wins", () => {
  const mine = [conv("a", 1, 50)];
  const theirs = [{ ...conv("a", 1, 80), name: "renamed elsewhere" }];
  assert.equal(
    mergeConversations<Conversation>(mine, theirs, {})[0].name,
    "renamed elsewhere",
  );
  const older = [{ ...conv("a", 1, 20), name: "stale" }];
  assert.equal(mergeConversations<Conversation>(mine, older, {})[0].name, "a");
});

test("this tab's objects are kept when they are as new", () => {
  const mine = [conv("a", 1, 50)];
  const merged = mergeConversations<Conversation>(mine, [conv("a", 1, 50)], {});
  assert.equal(merged[0], mine[0]);
  assert.ok(sameConversations(mine, merged));
});

test("a deleted conversation doesn't come back from another copy", () => {
  const stored = [conv("a", 1, 5), conv("b", 2, 10)];
  const merged = mergeConversations<Conversation>([conv("b", 2, 10)], stored, {
    a: 100,
  });
  assert.deepEqual(
    merged.map((c) => c.id),
    ["b"],
  );
});

test("a deletion wins over a tab still adding to the conversation", () => {
  // Deleted at 100 in one tab; another tab streamed into it until 200
  const merged = mergeConversations<Conversation>([conv("a", 1, 200)], [], {
    a: 100,
  });
  assert.deepEqual(merged, []);
});

test("newest created first", () => {
  const merged = mergeConversations<Conversation>(
    [conv("old", 1), conv("new", 3)],
    [conv("middle", 2)],
    {},
  );
  assert.deepEqual(
    merged.map((c) => c.id),
    ["new", "middle", "old"],
  );
});

test("tombstones merge and expire", () => {
  const now = TOMBSTONE_TTL_MS * 2;
  assert.deepEqual(
    mergeTombstones({ a: now - 10, old: 1 }, { a: now - 5, b: now - 1 }, now),
    { a: now - 5, b: now - 1 },
  );
});

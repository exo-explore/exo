/**
 * Merging the conversations of several dashboard tabs.
 *
 * Every tab keeps the conversations in memory and saves the whole list to
 * localStorage. Saving it as it was would be last-writer-wins: a tab that
 * hadn't seen a conversation another tab created would overwrite it. So a
 * tab merges before it saves, and when another tab saves (the "storage"
 * event): by conversation id, the version updated last wins, and a
 * conversation only goes away when a tab deleted it. Deletions are kept as
 * tombstones (id -> when), so a merge doesn't bring a deleted conversation
 * back from another tab's copy. A deletion always wins, also over a tab still
 * streaming a reply into the conversation: that tab stops.
 */

export interface SyncedConversation {
  id: string;
  createdAt: number;
  updatedAt: number;
}

/** Deleted conversation ids, with when they were deleted (ms) */
export type Tombstones = Record<string, number>;

/** How long a deletion is remembered: longer than any tab stays stale */
export const TOMBSTONE_TTL_MS = 30 * 24 * 60 * 60 * 1000;

// Ids are random UUIDs, never reused: a deleted id stays deleted
function isDeleted(
  conversation: SyncedConversation,
  deleted: Tombstones,
): boolean {
  return deleted[conversation.id] !== undefined;
}

/**
 * This tab's conversations merged with another copy (localStorage).
 * Keeps this tab's objects when they are at least as new, so code holding
 * them (a reply streaming in) keeps working. Newest created first.
 */
export function mergeConversations<T extends SyncedConversation>(
  mine: T[],
  theirs: T[],
  deleted: Tombstones,
): T[] {
  const byId = new Map<string, T>();
  for (const conversation of mine) byId.set(conversation.id, conversation);
  for (const conversation of theirs) {
    const own = byId.get(conversation.id);
    if (!own || conversation.updatedAt > own.updatedAt) {
      byId.set(conversation.id, conversation);
    }
  }
  return [...byId.values()]
    .filter((conversation) => !isDeleted(conversation, deleted))
    .sort((a, b) => b.createdAt - a.createdAt);
}

/** Both tombstone sets, without those older than TOMBSTONE_TTL_MS. */
export function mergeTombstones(
  a: Tombstones,
  b: Tombstones,
  now: number,
): Tombstones {
  const merged: Tombstones = {};
  for (const source of [a, b]) {
    for (const [id, deletedAt] of Object.entries(source)) {
      if (now - deletedAt > TOMBSTONE_TTL_MS) continue;
      merged[id] = Math.max(merged[id] ?? 0, deletedAt);
    }
  }
  return merged;
}

/** Whether a merge changed anything for this tab (to skip needless updates). */
export function sameConversations<T extends SyncedConversation>(
  before: T[],
  after: T[],
): boolean {
  return (
    before.length === after.length &&
    before.every((conversation, i) => conversation === after[i])
  );
}

<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import FamilyLogos from "$lib/components/FamilyLogos.svelte";
  import {
    instances,
    instanceLinks,
    nodeIdentities,
    refreshState,
    createInstanceLink,
    updateInstanceLink,
    deleteInstanceLink,
    type Instance,
  } from "$lib/stores/app.svelte";
  import { deriveBaseModel, deriveFamily } from "$lib/utils/model_family";

  type InstanceWrapper = {
    MlxRingInstance?: Instance;
    MlxJacclInstance?: Instance;
    VllmInstance?: Instance;
  };

  let interval: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    refreshState();
    interval = setInterval(refreshState, 3000);
  });
  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  type InstanceRow = {
    id: string;
    modelId: string;
    family: string;
    baseModel: string;
    nodeNames: string[];
  };

  const instanceRows = $derived.by<InstanceRow[]>(() => {
    const rows: InstanceRow[] = [];
    const ids = nodeIdentities();
    for (const [id, raw] of Object.entries(instances())) {
      const wrapper = raw as InstanceWrapper;
      const inst =
        wrapper.MlxRingInstance ??
        wrapper.MlxJacclInstance ??
        wrapper.VllmInstance;
      const modelId = inst?.shardAssignments?.modelId ?? "";
      const nodeToRunner = inst?.shardAssignments?.nodeToRunner ?? {};
      const nodeNames = Object.keys(nodeToRunner)
        .map((nodeId) => ids[nodeId]?.friendlyName ?? nodeId.slice(0, 6))
        .filter((name) => !!name);
      rows.push({
        id,
        modelId,
        family: deriveFamily(modelId),
        baseModel: deriveBaseModel(modelId),
        nodeNames,
      });
    }
    rows.sort((a, b) => a.modelId.localeCompare(b.modelId));
    return rows;
  });

  const instanceById = $derived(
    Object.fromEntries(instanceRows.map((r) => [r.id, r])),
  );

  type LinkRow = {
    linkId: string;
    prefill: string[];
    decode: string[];
    families: string[];
  };

  const linkRows = $derived.by<LinkRow[]>(() => {
    const rows: LinkRow[] = [];
    for (const [, link] of Object.entries(instanceLinks())) {
      const fams = new Set<string>();
      for (const id of [...link.prefillInstances, ...link.decodeInstances]) {
        const r = instanceById[id];
        if (r && r.baseModel) fams.add(r.baseModel.toLowerCase());
      }
      rows.push({
        linkId: link.linkId,
        prefill: link.prefillInstances,
        decode: link.decodeInstances,
        families: Array.from(fams),
      });
    }
    return rows;
  });

  let editingLinkId = $state<string | null>(null);
  let editingPrefill = $state<Set<string>>(new Set());
  let editingDecode = $state<Set<string>>(new Set());
  let saving = $state(false);
  let errorMessage = $state<string | null>(null);

  function startCreate() {
    editingLinkId = "new";
    editingPrefill = new Set();
    editingDecode = new Set();
    errorMessage = null;
  }

  function startEdit(row: LinkRow) {
    editingLinkId = row.linkId;
    editingPrefill = new Set(row.prefill);
    editingDecode = new Set(row.decode);
    errorMessage = null;
  }

  function cancelEdit() {
    editingLinkId = null;
    editingPrefill = new Set();
    editingDecode = new Set();
    errorMessage = null;
  }

  type Role = "prefill" | "decode" | "none";

  function roleOf(id: string): Role {
    if (editingPrefill.has(id)) return "prefill";
    if (editingDecode.has(id)) return "decode";
    return "none";
  }

  function setRole(id: string, role: Role) {
    const p = new Set(editingPrefill);
    const d = new Set(editingDecode);
    p.delete(id);
    d.delete(id);
    if (role === "prefill") p.add(id);
    if (role === "decode") d.add(id);
    editingPrefill = p;
    editingDecode = d;
  }

  const editingFamilies = $derived.by<string[]>(() => {
    const fams = new Set<string>();
    for (const id of [...editingPrefill, ...editingDecode]) {
      const r = instanceById[id];
      if (r && r.baseModel) fams.add(r.baseModel.toLowerCase());
    }
    return Array.from(fams);
  });

  const editingMismatch = $derived(editingFamilies.length > 1);
  const canSave = $derived(
    editingLinkId !== null &&
      editingPrefill.size > 0 &&
      editingDecode.size > 0 &&
      !saving,
  );

  async function save() {
    if (editingLinkId === null) return;
    saving = true;
    errorMessage = null;
    try {
      const prefill = Array.from(editingPrefill);
      const decode = Array.from(editingDecode);
      if (editingLinkId === "new") {
        await createInstanceLink(prefill, decode);
      } else {
        await updateInstanceLink(editingLinkId, prefill, decode);
      }
      cancelEdit();
      await refreshState();
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  async function remove(linkId: string) {
    if (!confirm("Remove this routing?")) return;
    try {
      await deleteInstanceLink(linkId);
      if (editingLinkId === linkId) cancelEdit();
      await refreshState();
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }
</script>

<div class="tab-content">
  <p class="tab-intro">
    Route prefill from one set of instances to another. The decode worker
    decides per-request whether to use a remote prefill (gated on prefix-cache
    miss size). Linked instances should be running the same model family.
  </p>

  {#if errorMessage}
    <div class="error">{errorMessage}</div>
  {/if}

  <section>
    <div class="section-head">
      <h2>Existing routes</h2>
      <button
        class="primary"
        onclick={startCreate}
        disabled={editingLinkId !== null}
      >
        + New route
      </button>
    </div>

    {#if linkRows.length === 0}
      <p class="empty">No routes yet. Create one to enable remote prefill.</p>
    {:else}
      <div class="link-grid">
        {#each linkRows as row (row.linkId)}
          <article class="link-card">
            {#if row.families.length > 1}
              <div class="warn-banner">
                ⚠ Mixed model families: {row.families.join(", ")}
              </div>
            {/if}
            <div class="route">
              <div class="route-side">
                <span class="role-tag prefill">PREFILL</span>
                <ul class="instance-list">
                  {#each row.prefill as id (id)}
                    {@const r = instanceById[id]}
                    {#if r}
                      <li class="chip">
                        <FamilyLogos family={r.family} />
                        <div>
                          <div class="model-name">
                            {r.baseModel || r.modelId}
                          </div>
                          <div class="node-name">
                            {r.nodeNames.join(", ") || "?"}
                          </div>
                        </div>
                      </li>
                    {/if}
                  {/each}
                </ul>
              </div>
              <div class="arrow" aria-hidden="true">→</div>
              <div class="route-side">
                <span class="role-tag decode">DECODE</span>
                <ul class="instance-list">
                  {#each row.decode as id (id)}
                    {@const r = instanceById[id]}
                    {#if r}
                      <li class="chip">
                        <FamilyLogos family={r.family} />
                        <div>
                          <div class="model-name">
                            {r.baseModel || r.modelId}
                          </div>
                          <div class="node-name">
                            {r.nodeNames.join(", ") || "?"}
                          </div>
                        </div>
                      </li>
                    {/if}
                  {/each}
                </ul>
              </div>
            </div>
            <div class="link-actions">
              <button
                onclick={() => startEdit(row)}
                disabled={editingLinkId !== null}
              >
                Edit
              </button>
              <button class="danger" onclick={() => remove(row.linkId)}
                >Remove</button
              >
            </div>
          </article>
        {/each}
      </div>
    {/if}
  </section>

  {#if editingLinkId !== null}
    <section class="editor">
      <h2>{editingLinkId === "new" ? "New route" : "Edit route"}</h2>

      {#if editingMismatch}
        <div class="warn-banner">
          ⚠ Selected instances span multiple model families:
          <strong>{editingFamilies.join(", ")}</strong>. Linking across families
          produces a corrupt KV cache.
        </div>
      {/if}

      <p class="hint">
        Pick a role for each instance: <strong>Prefill</strong> serves KV cache,
        <strong>Decode</strong> consumes it.
      </p>

      {#if instanceRows.length === 0}
        <p class="empty">No instances available.</p>
      {:else}
        <div class="instance-picker">
          {#each instanceRows as row (row.id)}
            {@const role = roleOf(row.id)}
            <div
              class="picker-card"
              class:picker-prefill={role === "prefill"}
              class:picker-decode={role === "decode"}
            >
              <div class="picker-meta">
                <FamilyLogos family={row.family} />
                <div>
                  <div class="model-name">{row.baseModel || row.modelId}</div>
                  <div class="node-name">{row.nodeNames.join(", ") || "?"}</div>
                </div>
              </div>
              <div class="role-toggle">
                <button
                  class="role-btn"
                  class:active={role === "none"}
                  onclick={() => setRole(row.id, "none")}
                  title="Don't include in this route">–</button
                >
                <button
                  class="role-btn prefill"
                  class:active={role === "prefill"}
                  onclick={() => setRole(row.id, "prefill")}>Prefill</button
                >
                <button
                  class="role-btn decode"
                  class:active={role === "decode"}
                  onclick={() => setRole(row.id, "decode")}>Decode</button
                >
              </div>
            </div>
          {/each}
        </div>
      {/if}

      <div class="editor-actions">
        <button class="primary" onclick={save} disabled={!canSave}>
          {saving ? "Saving..." : "Save route"}
        </button>
        <button onclick={cancelEdit} disabled={saving}>Cancel</button>
      </div>
    </section>
  {/if}
</div>

<style>
  .tab-content {
    color: var(--text-primary, #eee);
  }
  .tab-intro {
    margin: 0 0 1.5rem 0;
    color: var(--text-secondary, #aaa);
    max-width: 70ch;
  }
  .page-header h1 {
    margin: 0 0 0.5rem 0;
  }
  .page-header p {
    margin: 0;
    color: var(--text-secondary, #aaa);
    max-width: 70ch;
  }
  .error {
    margin: 1rem 0;
    padding: 0.75rem 1rem;
    background: rgba(255, 80, 80, 0.15);
    border: 1px solid rgba(255, 80, 80, 0.5);
    border-radius: 6px;
  }
  .section-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 2rem 0 1rem 0;
  }
  .section-head h2,
  .editor h2 {
    margin: 0;
  }
  button {
    background: rgba(255, 255, 255, 0.08);
    color: inherit;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    padding: 0.45rem 0.9rem;
    cursor: pointer;
    font: inherit;
  }
  button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.14);
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  button.primary {
    background: rgba(80, 180, 255, 0.2);
    border-color: rgba(80, 180, 255, 0.5);
  }
  button.danger {
    background: rgba(255, 80, 80, 0.18);
    border-color: rgba(255, 80, 80, 0.4);
  }
  .empty {
    color: var(--text-secondary, #888);
    font-style: italic;
  }
  .link-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .link-card {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    background: rgba(255, 255, 255, 0.02);
  }
  .route {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1rem;
    align-items: center;
  }
  .route-side {
    min-width: 0;
  }
  .arrow {
    font-size: 1.5rem;
    color: var(--text-secondary, #888);
  }
  .role-tag {
    display: inline-block;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-bottom: 0.5rem;
  }
  .role-tag.prefill {
    background: rgba(80, 180, 255, 0.2);
    color: #6cb6ff;
  }
  .role-tag.decode {
    background: rgba(150, 220, 100, 0.2);
    color: #b3e07d;
  }
  .instance-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .chip {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 0.45rem 0.6rem;
  }
  .chip :global(svg) {
    flex-shrink: 0;
    color: var(--text-secondary, #ccc);
  }
  .model-name {
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .node-name {
    font-size: 0.8em;
    color: var(--text-secondary, #999);
  }
  .link-actions {
    margin-top: 0.85rem;
    display: flex;
    gap: 0.5rem;
    justify-content: flex-end;
  }
  .editor {
    margin-top: 2rem;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
  }
  .editor h2 {
    margin: 0 0 0.5rem 0;
  }
  .editor .hint {
    color: var(--text-secondary, #aaa);
    margin: 0 0 1rem 0;
    font-size: 0.95em;
  }
  .editor-actions {
    margin-top: 1.25rem;
    display: flex;
    gap: 0.5rem;
  }
  .warn-banner {
    margin-bottom: 0.85rem;
    padding: 0.6rem 0.85rem;
    background: rgba(255, 183, 77, 0.1);
    border: 1px solid rgba(255, 183, 77, 0.4);
    border-radius: 6px;
    color: #ffb74d;
    font-size: 0.9em;
  }
  .instance-picker {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 0.75rem;
  }
  .picker-card {
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    background: rgba(255, 255, 255, 0.02);
    transition:
      border-color 0.15s,
      background 0.15s;
  }
  .picker-card.picker-prefill {
    border-color: rgba(80, 180, 255, 0.5);
    background: rgba(80, 180, 255, 0.06);
  }
  .picker-card.picker-decode {
    border-color: rgba(150, 220, 100, 0.5);
    background: rgba(150, 220, 100, 0.06);
  }
  .picker-meta {
    display: flex;
    gap: 0.7rem;
    align-items: center;
  }
  .picker-meta :global(svg) {
    flex-shrink: 0;
    color: var(--text-secondary, #ddd);
  }
  .role-toggle {
    display: flex;
    gap: 0.35rem;
  }
  .role-btn {
    flex: 1;
    padding: 0.35rem 0.5rem;
    font-size: 0.85em;
  }
  .role-btn.active {
    background: rgba(255, 255, 255, 0.18);
    border-color: rgba(255, 255, 255, 0.35);
  }
  .role-btn.prefill.active {
    background: rgba(80, 180, 255, 0.3);
    border-color: rgba(80, 180, 255, 0.7);
    color: #fff;
  }
  .role-btn.decode.active {
    background: rgba(150, 220, 100, 0.3);
    border-color: rgba(150, 220, 100, 0.7);
    color: #fff;
  }
</style>

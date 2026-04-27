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
    nodeCount: number;
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
      const nodeIds = Object.keys(nodeToRunner);
      const nodeNames = nodeIds
        .map((nodeId) => ids[nodeId]?.friendlyName ?? nodeId.slice(0, 6))
        .filter((name) => !!name);
      rows.push({
        id,
        modelId,
        family: deriveFamily(modelId),
        baseModel: deriveBaseModel(modelId),
        nodeNames,
        nodeCount: nodeIds.length,
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
    multiNode: boolean;
  };

  const linkRows = $derived.by<LinkRow[]>(() => {
    const rows: LinkRow[] = [];
    for (const [, link] of Object.entries(instanceLinks())) {
      const fams = new Set<string>();
      let multiNode = false;
      for (const id of [...link.prefillInstances, ...link.decodeInstances]) {
        const r = instanceById[id];
        if (r && r.baseModel) fams.add(r.baseModel.toLowerCase());
        if (r && r.nodeCount > 1) multiNode = true;
      }
      rows.push({
        linkId: link.linkId,
        prefill: link.prefillInstances,
        decode: link.decodeInstances,
        families: Array.from(fams),
        multiNode,
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

  const editingMultiNode = $derived.by<string[]>(() => {
    const names: string[] = [];
    for (const id of [...editingPrefill, ...editingDecode]) {
      const r = instanceById[id];
      if (r && r.nodeCount > 1) {
        names.push(r.baseModel || r.modelId);
      }
    }
    return names;
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

<div class="font-mono text-foreground">
  <div class="mb-6 space-y-4">
    <details open class="group [&_summary::-webkit-details-marker]:hidden">
      <summary
        class="cursor-pointer list-none text-exo-yellow text-xs font-mono tracking-widest uppercase flex items-center gap-2 hover:opacity-80 transition-opacity"
      >
        <span
          class="inline-block transition-transform group-open:rotate-90 text-exo-light-gray"
          >▶</span
        >
        Prefill vs Decode
      </summary>
      <div class="mt-2 text-white/80 text-sm leading-relaxed">
        Prefill is the compute-heavy pass that consumes the entire prompt and
        builds a KV cache. Decode is the memory-bandwidth-bound loop that emits
        tokens sequentially from that cache. The two phases have very different
        bottlenecks, so running them on different hardware can be substantially
        faster than doing both on one node.
      </div>
    </details>
    <details class="group [&_summary::-webkit-details-marker]:hidden">
      <summary
        class="cursor-pointer list-none text-exo-yellow text-xs font-mono tracking-widest uppercase flex items-center gap-2 hover:opacity-80 transition-opacity"
      >
        <span
          class="inline-block transition-transform group-open:rotate-90 text-exo-light-gray"
          >▶</span
        >
        Linking Instances
      </summary>
      <div class="mt-2 text-white/80 text-sm leading-relaxed space-y-2">
        <p>
          A linked route here tells the cluster: when a request is sent to a
          model in that cluster, the decode node (or the least active one if
          there are multiple) will handle it. If it decides it must do a lot of
          prefill not already cached in the prefix cache, it routes the request
          to the prefill node over TCP IP. The prefill node streams the KV cache
          back to the decode node which picks up from there.
        </p>
        <p>
          Linked instances must be running the same model family — KV layouts
          differ across architectures. More on the <a
            class="text-exo-yellow underline underline-offset-2 hover:text-exo-yellow-darker transition-colors"
            href="https://blog.exolabs.net/nvidia-dgx-spark/"
            target="_blank"
            rel="noreferrer noopener">blog</a
          >.
        </p>
      </div>
    </details>
  </div>

  {#if errorMessage}
    <div
      class="mb-4 px-4 py-3 bg-red-500/10 border border-red-500/40 text-red-300 text-sm"
    >
      {errorMessage}
    </div>
  {/if}

  <section class="mt-12">
    <h2
      class="text-exo-yellow text-xs font-mono tracking-widest uppercase m-0 mb-3"
    >
      Existing routes
    </h2>

    {#if linkRows.length === 0}
      {#if editingLinkId === null}
        <div class="flex items-center justify-between">
          <p class="text-exo-light-gray italic text-sm m-0">
            No routes yet. Create one to enable remote prefill.
          </p>
          <button
            class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-yellow/15 border border-exo-yellow/50 text-exo-yellow hover:bg-exo-yellow/25 hover:border-exo-yellow/80 transition-colors"
            onclick={startCreate}
          >
            + New route
          </button>
        </div>
      {/if}
    {:else}
      {#if editingLinkId === null}
        <div class="flex justify-end mb-3">
          <button
            class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-yellow/15 border border-exo-yellow/50 text-exo-yellow hover:bg-exo-yellow/25 hover:border-exo-yellow/80 transition-colors"
            onclick={startCreate}
          >
            + New route
          </button>
        </div>
      {/if}
      <div
        class="bg-exo-dark-gray/60 border border-exo-medium-gray/40 flex flex-col"
      >
        {#each linkRows as row (row.linkId)}
          {#if editingLinkId !== row.linkId}
            <article
              class="p-4 border-b border-exo-light-gray/25 last:border-b-0"
            >
              {#if row.multiNode}
                <div
                  class="mb-3 px-3 py-2 bg-red-500/10 border border-red-500/40 text-red-300 text-xs tracking-wide"
                >
                  ⚠ Multi-node instance detected. Remote prefill currently only
                  works on single-node (rank-0) instances. This route will not
                  function until that's supported.
                </div>
              {/if}
              {#if row.families.length > 1}
                <div
                  class="mb-3 px-3 py-2 bg-amber-500/10 border border-amber-500/40 text-amber-300 text-xs tracking-wide"
                >
                  ⚠ Mixed model families: {row.families.join(", ")}
                </div>
              {/if}
              <div
                class="grid grid-cols-[1fr_auto_1fr_auto] items-center gap-x-3 gap-y-2"
              >
                <span
                  class="inline-block justify-self-start text-[10px] font-mono tracking-widest uppercase px-2 py-0.5 bg-exo-yellow/15 border border-exo-yellow/40 text-exo-yellow"
                  >Prefill</span
                >
                <span></span>
                <span
                  class="inline-block justify-self-start text-[10px] font-mono tracking-widest uppercase px-2 py-0.5 bg-exo-medium-gray/40 border border-exo-medium-gray/60 text-foreground"
                  >Decode</span
                >
                <span></span>
                <div class="min-w-0">
                  <ul class="list-none p-0 m-0 flex flex-col gap-2">
                    {#each row.prefill as id (id)}
                      {@const r = instanceById[id]}
                      {#if r}
                        <li
                          class="flex items-center gap-2 px-2.5 py-2 bg-exo-medium-gray/20 border border-exo-medium-gray/40"
                        >
                          <FamilyLogos family={r.family} />
                          <div class="min-w-0 flex-1">
                            <div
                              class="text-exo-yellow text-xs font-mono truncate"
                            >
                              {r.baseModel || r.modelId}
                            </div>
                            <div
                              class="text-exo-light-gray text-[11px] truncate"
                            >
                              {r.nodeNames.join(", ") || "?"}{r.nodeCount > 1
                                ? ` (${r.nodeCount} nodes)`
                                : ""}
                            </div>
                            <div
                              class="text-exo-light-gray/40 text-[10px] font-mono truncate"
                              title={r.id}
                            >
                              {r.id.slice(0, 8)}
                            </div>
                          </div>
                        </li>
                      {/if}
                    {/each}
                  </ul>
                </div>
                <div class="text-exo-yellow/60 text-xl px-2" aria-hidden="true">
                  →
                </div>
                <div class="min-w-0">
                  <ul class="list-none p-0 m-0 flex flex-col gap-2">
                    {#each row.decode as id (id)}
                      {@const r = instanceById[id]}
                      {#if r}
                        <li
                          class="flex items-center gap-2 px-2.5 py-2 bg-exo-medium-gray/20 border border-exo-medium-gray/40"
                        >
                          <FamilyLogos family={r.family} />
                          <div class="min-w-0 flex-1">
                            <div
                              class="text-exo-yellow text-xs font-mono truncate"
                            >
                              {r.baseModel || r.modelId}
                            </div>
                            <div
                              class="text-exo-light-gray text-[11px] truncate"
                            >
                              {r.nodeNames.join(", ") || "?"}{r.nodeCount > 1
                                ? ` (${r.nodeCount} nodes)`
                                : ""}
                            </div>
                            <div
                              class="text-exo-light-gray/40 text-[10px] font-mono truncate"
                              title={r.id}
                            >
                              {r.id.slice(0, 8)}
                            </div>
                          </div>
                        </li>
                      {/if}
                    {/each}
                  </ul>
                </div>
                <div class="flex gap-2 pl-3">
                  <button
                    class="px-2 py-0.5 text-[11px] font-mono tracking-wider uppercase bg-exo-medium-gray/30 border border-exo-medium-gray/60 rounded text-foreground hover:border-exo-yellow/60 hover:text-exo-yellow disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    onclick={() => startEdit(row)}
                    disabled={editingLinkId !== null}
                  >
                    Edit
                  </button>
                  <button
                    class="px-2 py-0.5 text-[11px] font-mono tracking-wider uppercase bg-red-500/15 border border-red-500/40 rounded text-red-300 hover:bg-red-500/25 transition-colors"
                    onclick={() => remove(row.linkId)}
                  >
                    Remove
                  </button>
                </div>
              </div>
            </article>
          {/if}
        {/each}
      </div>
    {/if}
  </section>

  {#if editingLinkId !== null && instanceRows.length === 0}
    <section
      class="mt-6 bg-exo-dark-gray/60 border border-exo-yellow/30 px-4 py-2.5 flex items-center justify-between gap-3"
    >
      <span class="text-exo-light-gray italic text-sm font-mono"
        >No instances available.</span
      >
      <button
        class="px-3 py-1 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/30 border border-exo-medium-gray/60 rounded text-foreground hover:border-exo-yellow/60 transition-colors"
        onclick={cancelEdit}
      >
        Cancel
      </button>
    </section>
  {:else if editingLinkId !== null}
    <section class="mt-6 bg-exo-dark-gray/60 border border-exo-yellow/30 p-5">
      <h2
        class="text-exo-yellow text-xs font-mono tracking-widest uppercase m-0 mb-3"
      >
        {editingLinkId === "new" ? "New route" : "Edit route"}
      </h2>

      {#if editingMismatch}
        <div
          class="mb-3 px-3 py-2 bg-amber-500/10 border border-amber-500/40 text-amber-300 text-xs tracking-wide"
        >
          ⚠ Selected instances span multiple model families: <strong
            >{editingFamilies.join(", ")}</strong
          >. Linking across families produces a corrupt KV cache.
        </div>
      {/if}

      {#if editingMultiNode.length > 0}
        <div
          class="mb-3 px-3 py-2 bg-red-500/10 border border-red-500/40 text-red-300 text-xs tracking-wide"
        >
          ⚠ Multi-node instance(s) selected: <strong
            >{editingMultiNode.join(", ")}</strong
          >. Remote prefill currently only works on single-node instances. This
          route will not function until multi-node support lands.
        </div>
      {/if}

      <p class="text-exo-light-gray text-xs mb-4">
        Pick a role for each instance:
        <span class="text-exo-yellow">Prefill</span>
        serves KV cache,
        <span class="text-foreground">Decode</span> consumes it.
      </p>
      <div
        class="grid gap-2.5"
        style="grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));"
      >
        {#each instanceRows as row (row.id)}
          {@const role = roleOf(row.id)}
          <div
            class="border p-3 flex flex-col gap-2.5 transition-colors {role ===
            'prefill'
              ? 'border-exo-yellow/60 bg-exo-dark-gray/60'
              : role === 'decode'
                ? 'border-exo-light-gray/60 bg-exo-dark-gray/60'
                : 'border-exo-medium-gray/40 bg-exo-dark-gray/40'}"
          >
            <div class="flex items-center gap-2">
              <FamilyLogos family={row.family} />
              <div class="min-w-0 flex-1">
                <div class="text-exo-yellow text-xs font-mono truncate">
                  {row.baseModel || row.modelId}
                </div>
                <div class="text-exo-light-gray text-[11px] truncate">
                  {row.nodeNames.join(", ") || "?"}{row.nodeCount > 1
                    ? ` (${row.nodeCount} nodes)`
                    : ""}
                </div>
                <div
                  class="text-exo-light-gray/40 text-[10px] font-mono truncate"
                  title={row.id}
                >
                  {row.id.slice(0, 8)}
                </div>
              </div>
              {#if row.nodeCount > 1}
                <span
                  class="text-[9px] font-mono tracking-widest uppercase px-1.5 py-0.5 bg-red-500/15 border border-red-500/40 text-red-300"
                  title="Multi-node instances are not supported by remote prefill yet."
                  >Unsupported</span
                >
              {/if}
            </div>
            <div
              class="flex rounded-md overflow-hidden border border-exo-light-gray/40 divide-x divide-exo-light-gray/40"
            >
              <button
                class="flex-1 px-2 py-1 text-[11px] font-mono tracking-wider uppercase transition-colors {role ===
                'prefill'
                  ? 'bg-exo-yellow/20 text-exo-yellow'
                  : 'bg-transparent text-white/80 hover:text-exo-yellow'}"
                onclick={() =>
                  setRole(row.id, role === "prefill" ? "none" : "prefill")}
                >Prefill</button
              >
              <button
                class="flex-1 px-2 py-1 text-[11px] font-mono tracking-wider uppercase transition-colors {role ===
                'decode'
                  ? 'bg-exo-medium-gray/50 text-foreground'
                  : 'bg-transparent text-white/80 hover:text-foreground'}"
                onclick={() =>
                  setRole(row.id, role === "decode" ? "none" : "decode")}
                >Decode</button
              >
            </div>
          </div>
        {/each}
      </div>

      <div class="flex gap-2 mt-5 justify-end">
        <button
          class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-yellow/15 border border-exo-yellow/50 text-exo-yellow hover:bg-exo-yellow/25 hover:border-exo-yellow/80 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          onclick={save}
          disabled={!canSave}
        >
          {saving ? "Saving..." : "Save route"}
        </button>
        <button
          class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/30 border border-exo-medium-gray/60 text-foreground hover:border-exo-yellow/60 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          onclick={cancelEdit}
          disabled={saving}
        >
          Cancel
        </button>
      </div>
    </section>
  {/if}
</div>

<script lang="ts">
  import { onMount } from "svelte";
  import {
    listTrajectories,
    deleteTrajectories,
    getTrajectoryRawUrl,
    type TrajectoryListItem,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  let trajectories = $state<TrajectoryListItem[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let selectedIds = $state<Set<string>>(new Set());
  let deleting = $state(false);
  let compareA = $state<string | null>(null);
  let compareB = $state<string | null>(null);

  let allSelected = $derived(
    trajectories.length > 0 && selectedIds.size === trajectories.length,
  );

  function toggleSelect(sessionId: string) {
    const next = new Set(selectedIds);
    if (next.has(sessionId)) {
      next.delete(sessionId);
    } else {
      next.add(sessionId);
    }
    selectedIds = next;
  }

  function toggleSelectAll() {
    if (allSelected) {
      selectedIds = new Set();
    } else {
      selectedIds = new Set(trajectories.map((t) => t.sessionId));
    }
  }

  async function handleDelete() {
    if (selectedIds.size === 0) return;
    const count = selectedIds.size;
    if (
      !confirm(
        `Delete ${count} trajector${count === 1 ? "y" : "ies"}? This cannot be undone.`,
      )
    )
      return;
    deleting = true;
    try {
      await deleteTrajectories([...selectedIds]);
      selectedIds = new Set();
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to delete trajectories";
    } finally {
      deleting = false;
    }
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleString();
  }

  async function downloadTrajectory(sessionId: string) {
    const response = await fetch(getTrajectoryRawUrl(sessionId));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function pickForCompare(sessionId: string) {
    if (compareA === null) {
      compareA = sessionId;
    } else if (compareB === null && sessionId !== compareA) {
      compareB = sessionId;
    } else {
      compareA = sessionId;
      compareB = null;
    }
  }

  function clearCompare() {
    compareA = null;
    compareB = null;
  }

  async function refresh() {
    loading = true;
    error = null;
    try {
      const response = await listTrajectories();
      trajectories = response.trajectories;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load trajectories";
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    refresh();
  });
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <h1 class="text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow">
        Trajectories
      </h1>
      <div class="flex items-center gap-3">
        {#if compareA && compareB}
          <a
            href="#/trajectories/compare?a={encodeURIComponent(
              compareA,
            )}&b={encodeURIComponent(compareB)}"
            class="text-xs font-mono text-exo-dark-gray bg-exo-yellow hover:bg-exo-yellow/90 transition-colors uppercase px-2 py-1 rounded font-semibold"
          >
            Compare
          </a>
          <button
            type="button"
            class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
            onclick={clearCompare}
          >
            Clear
          </button>
        {:else if compareA}
          <span class="text-xs font-mono text-exo-light-gray uppercase">
            Pick another to compare
          </span>
          <button
            type="button"
            class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
            onclick={clearCompare}
          >
            Clear
          </button>
        {/if}
        {#if selectedIds.size > 0}
          <button
            type="button"
            class="text-xs font-mono text-red-400 hover:text-red-300 transition-colors uppercase border border-red-500/40 px-2 py-1 rounded"
            onclick={handleDelete}
            disabled={deleting}
          >
            {deleting ? "Deleting..." : `Delete (${selectedIds.size})`}
          </button>
        {/if}
        <button
          type="button"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
          onclick={refresh}
          disabled={loading}
        >
          Refresh
        </button>
      </div>
    </div>

    {#if loading}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">Loading trajectories...</div>
      </div>
    {:else if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-6 text-center text-red-400"
      >
        <div class="text-sm">{error}</div>
      </div>
    {:else if trajectories.length === 0}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray space-y-2"
      >
        <div class="text-sm">No trajectories found.</div>
        <div class="text-xs text-exo-light-gray/70">
          Run exo with --trajectories or EXO_TRAJECTORIES=1 to record live chat
          completions as ATIF-v1.4 JSON.
        </div>
      </div>
    {:else}
      <div class="space-y-3">
        <div class="flex items-center gap-2 px-1">
          <button
            type="button"
            class="text-xs font-mono uppercase transition-colors {allSelected
              ? 'text-exo-yellow'
              : 'text-exo-light-gray hover:text-exo-yellow'}"
            onclick={toggleSelectAll}
          >
            {allSelected ? "Deselect all" : "Select all"}
          </button>
        </div>
        {#each trajectories as trajectory}
          {@const isSelected = selectedIds.has(trajectory.sessionId)}
          {@const isCompareA = compareA === trajectory.sessionId}
          {@const isCompareB = compareB === trajectory.sessionId}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            role="button"
            tabindex="0"
            class="w-full text-left rounded border-l-2 border-r border-t border-b transition-all p-4 flex items-center justify-between gap-4 cursor-pointer {isSelected
              ? 'bg-exo-yellow/10 border-l-exo-yellow border-r-exo-medium-gray/30 border-t-exo-medium-gray/30 border-b-exo-medium-gray/30'
              : 'bg-exo-black/30 border-l-transparent border-r-exo-medium-gray/30 border-t-exo-medium-gray/30 border-b-exo-medium-gray/30 hover:bg-white/[0.03]'}"
            onclick={() => toggleSelect(trajectory.sessionId)}
            onkeydown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                toggleSelect(trajectory.sessionId);
              }
            }}
          >
            <div class="min-w-0 flex-1">
              <a
                href="#/trajectories/{trajectory.sessionId}"
                class="text-sm font-mono transition-colors truncate block {isSelected
                  ? 'text-exo-yellow'
                  : 'text-white hover:text-exo-yellow'}"
                onclick={(e) => e.stopPropagation()}
              >
                {trajectory.sessionId}
                {#if isCompareA}
                  <span class="ml-2 text-xs text-exo-yellow">[A]</span>
                {:else if isCompareB}
                  <span class="ml-2 text-xs text-exo-yellow">[B]</span>
                {/if}
              </a>
              <div class="text-xs text-exo-light-gray font-mono mt-1">
                {formatDate(trajectory.updatedAt)} &bull; {trajectory.totalSteps}
                steps &bull; {trajectory.model}
              </div>
            </div>
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <div
              class="flex items-center gap-2 shrink-0"
              onclick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
                onclick={() => pickForCompare(trajectory.sessionId)}
              >
                {isCompareA ? "Set B" : "Compare"}
              </button>
              <a
                href="#/trajectories/{trajectory.sessionId}"
                class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
              >
                View
              </a>
              <button
                type="button"
                class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
                onclick={() => downloadTrajectory(trajectory.sessionId)}
              >
                Download
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

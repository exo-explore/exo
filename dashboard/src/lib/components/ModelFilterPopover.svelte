<script lang="ts">
  import { fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";

  interface FilterState {
    capabilities: string[];
    sizeRange: { min: number; max: number } | null;
    downloadedOnly: boolean;
    readyOnly: boolean;
  }

  type ModelFilterPopoverProps = {
    filters: FilterState;
    onChange: (filters: FilterState) => void;
    onClear: () => void;
    onClose: () => void;
  };

  let { filters, onChange, onClear, onClose }: ModelFilterPopoverProps =
    $props();

  // Available capabilities
  const availableCapabilities = [
    { id: "text", label: "Text" },
    { id: "thinking", label: "Thinking" },
    { id: "code", label: "Code" },
    { id: "vision", label: "Vision" },
    { id: "image_gen", label: "Image Gen" },
    { id: "image_edit", label: "Image Edit" },
  ];

  // Size ranges
  const sizeRanges = [
    { label: "< 10GB", min: 0, max: 10 },
    { label: "10-50GB", min: 10, max: 50 },
    { label: "50-200GB", min: 50, max: 200 },
    { label: "> 200GB", min: 200, max: 10000 },
  ];

  function toggleCapability(cap: string) {
    const next = filters.capabilities.includes(cap)
      ? filters.capabilities.filter((c) => c !== cap)
      : [...filters.capabilities, cap];
    onChange({ ...filters, capabilities: next });
  }

  function selectSizeRange(range: { min: number; max: number } | null) {
    // Toggle off if same range is clicked
    if (
      filters.sizeRange &&
      range &&
      filters.sizeRange.min === range.min &&
      filters.sizeRange.max === range.max
    ) {
      onChange({ ...filters, sizeRange: null });
    } else {
      onChange({ ...filters, sizeRange: range });
    }
  }

  function handleClickOutside(e: MouseEvent) {
    const target = e.target as HTMLElement;
    if (
      !target.closest(".filter-popover") &&
      !target.closest(".filter-toggle")
    ) {
      onClose();
    }
  }
</script>

<svelte:window onclick={handleClickOutside} />

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="filter-popover absolute right-0 top-full mt-2 w-64 bg-exo-dark-gray border border-exo-yellow/10 rounded-lg shadow-xl z-10"
  transition:fly={{ y: -10, duration: 200, easing: cubicOut }}
  onclick={(e) => e.stopPropagation()}
  role="dialog"
  aria-label="Filter options"
>
  <div class="p-3 space-y-4">
    <!-- Capabilities -->
    <div>
      <h4 class="text-xs font-mono text-white/50 mb-2">Capabilities</h4>
      <div class="flex flex-wrap gap-1.5">
        {#each availableCapabilities as cap}
          {@const isSelected = filters.capabilities.includes(cap.id)}
          <button
            type="button"
            class="px-2 py-1 text-xs font-mono rounded transition-colors {isSelected
              ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/30'
              : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'}"
            onclick={() => toggleCapability(cap.id)}
          >
            {#if cap.id === "text"}
              <svg
                class="w-3.5 h-3.5 inline-block"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                ><path
                  d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /></svg
              >
            {:else if cap.id === "thinking"}
              <svg
                class="w-3.5 h-3.5 inline-block"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                ><path
                  d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7zM9 20h6M10 22h4"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /></svg
              >
            {:else if cap.id === "code"}
              <svg
                class="w-3.5 h-3.5 inline-block"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                ><path
                  d="M16 18l6-6-6-6M8 6l-6 6 6 6"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /></svg
              >
            {:else if cap.id === "vision"}
              <svg
                class="w-3.5 h-3.5 inline-block"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                ><path
                  d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /><circle cx="12" cy="12" r="3" /></svg
              >
            {:else if cap.id === "image_gen"}
              <svg
                class="w-3.5 h-3.5 inline-block"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                ><rect
                  x="3"
                  y="3"
                  width="18"
                  height="18"
                  rx="2"
                  ry="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /><circle cx="8.5" cy="8.5" r="1.5" /><path
                  d="M21 15l-5-5L5 21"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /></svg
              >
            {:else if cap.id === "image_edit"}
              <svg
                class="w-3.5 h-3.5 inline-block"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                ><path
                  d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /><path
                  d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                /></svg
              >
            {/if}
            <span class="ml-1">{cap.label}</span>
          </button>
        {/each}
      </div>
    </div>

    <!-- Availability filters -->
    <div>
      <h4 class="text-xs font-mono text-white/50 mb-2">Availability</h4>
      <div class="flex flex-wrap gap-1.5">
        <button
          type="button"
          class="px-2 py-1 text-xs font-mono rounded transition-colors {filters.downloadedOnly
            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
            : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'}"
          onclick={() =>
            onChange({ ...filters, downloadedOnly: !filters.downloadedOnly })}
        >
          <svg
            class="w-3.5 h-3.5 inline-block"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path
              class="text-white/40"
              d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
            />
            <path class="text-green-400" d="m9 13 2 2 4-4" />
          </svg>
          <span class="ml-1">Downloaded</span>
        </button>
        <button
          type="button"
          class="px-2 py-1 text-xs font-mono rounded transition-colors {filters.readyOnly
            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
            : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'}"
          onclick={() =>
            onChange({ ...filters, readyOnly: !filters.readyOnly })}
        >
          <svg
            class="w-3.5 h-3.5 inline-block"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          <span class="ml-1">Ready</span>
        </button>
      </div>
    </div>

    <!-- Size range -->
    <div>
      <h4 class="text-xs font-mono text-white/50 mb-2">Model Size</h4>
      <div class="flex flex-wrap gap-1.5">
        {#each sizeRanges as range}
          {@const isSelected =
            filters.sizeRange &&
            filters.sizeRange.min === range.min &&
            filters.sizeRange.max === range.max}
          <button
            type="button"
            class="px-2 py-1 text-xs font-mono rounded transition-colors {isSelected
              ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/30'
              : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'}"
            onclick={() => selectSizeRange(range)}
          >
            {range.label}
          </button>
        {/each}
      </div>
    </div>

    <!-- Clear button -->
    <button
      type="button"
      class="w-full py-1.5 text-xs font-mono text-white/50 hover:text-white/70 hover:bg-white/5 rounded transition-colors"
      onclick={onClear}
    >
      Clear all filters
    </button>
  </div>
</div>

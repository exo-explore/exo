<script lang="ts">
  interface ModelInfo {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
    base_model?: string;
    quantization?: string;
    supports_tensor?: boolean;
    capabilities?: string[];
    family?: string;
    is_custom?: boolean;
  }

  interface ModelGroup {
    id: string;
    name: string;
    capabilities: string[];
    family: string;
    variants: ModelInfo[];
    smallestVariant: ModelInfo;
    hasMultipleVariants: boolean;
  }

  type DownloadAvailability = {
    available: boolean;
    nodeNames: string[];
    nodeIds: string[];
  };
  type ModelFitStatus = "fits_now" | "fits_cluster_capacity" | "too_large";

  type ModelPickerGroupProps = {
    group: ModelGroup;
    isExpanded: boolean;
    isFavorite: boolean;
    selectedModelId: string | null;
    canModelFit: (id: string) => boolean;
    getModelFitStatus: (id: string) => ModelFitStatus;
    onToggleExpand: () => void;
    onSelectModel: (modelId: string) => void;
    onToggleFavorite: (baseModelId: string) => void;
    onShowInfo: (group: ModelGroup) => void;
    downloadStatusMap?: Map<string, DownloadAvailability>;
    launchedAt?: number;
    instanceStatuses?: Record<string, { status: string; statusClass: string }>;
  };

  let {
    group,
    isExpanded,
    isFavorite,
    selectedModelId,
    canModelFit,
    getModelFitStatus,
    onToggleExpand,
    onSelectModel,
    onToggleFavorite,
    onShowInfo,
    downloadStatusMap,
    launchedAt,
    instanceStatuses = {},
  }: ModelPickerGroupProps = $props();

  // Group-level download status: show if any variant is downloaded
  const groupDownloadStatus = $derived.by(() => {
    if (!downloadStatusMap || downloadStatusMap.size === 0) return undefined;
    // Return the first available entry (prefer "available" ones)
    for (const avail of downloadStatusMap.values()) {
      if (avail.available) return avail;
    }
    return downloadStatusMap.values().next().value;
  });

  // Format storage size
  function formatSize(mb: number | undefined): string {
    if (!mb) return "";
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(0)}GB`;
    }
    return `${mb}MB`;
  }

  function timeAgo(ts: number): string {
    const seconds = Math.floor((Date.now() - ts) / 1000);
    if (seconds < 60) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  // Check if any variant can fit
  const anyVariantFits = $derived(
    group.variants.some((v) => canModelFit(v.id)),
  );
  // Check if any variant has an active instance (ready, loading, downloading)
  const anyVariantHasInstance = $derived(
    instanceStatuses
      ? group.variants.some((v) => instanceStatuses[v.id] != null)
      : false,
  );
  const groupFitStatus = $derived.by((): ModelFitStatus => {
    let hasClusterCapacityOnly = false;
    for (const variant of group.variants) {
      const fitStatus = getModelFitStatus(variant.id);
      if (fitStatus === "fits_now") {
        return "fits_now";
      }
      if (fitStatus === "fits_cluster_capacity") {
        hasClusterCapacityOnly = true;
      }
    }
    return hasClusterCapacityOnly ? "fits_cluster_capacity" : "too_large";
  });

  function getSizeClassForFitStatus(fitStatus: ModelFitStatus): string {
    switch (fitStatus) {
      case "fits_now":
        return "text-white/40";
      case "fits_cluster_capacity":
        return "text-orange-400/80";
      case "too_large":
        return "text-red-400/70";
    }
  }

  // Check if this group's model is currently selected (for single-variant groups)
  const isMainSelected = $derived(
    !group.hasMultipleVariants &&
      group.variants.some((v) => v.id === selectedModelId),
  );

  // Group-level instance status: show the "best" status across all variants
  const groupInstanceStatus = $derived.by(() => {
    if (!instanceStatuses) return null;
    const readyStatuses = ["READY", "LOADED", "RUNNING"];
    const loadingStatuses = ["LOADING", "WARMING UP"];
    let bestStatus: { status: string; statusClass: string } | null = null;
    for (const variant of group.variants) {
      const s = instanceStatuses[variant.id];
      if (!s) continue;
      if (readyStatuses.includes(s.status)) return s; // Ready is best
      if (loadingStatuses.includes(s.status) || s.status === "DOWNLOADING") {
        bestStatus = s;
      }
    }
    return bestStatus;
  });
</script>

<div
  class="border-b border-white/5 last:border-b-0 {!anyVariantFits &&
  !anyVariantHasInstance
    ? 'opacity-50'
    : ''}"
>
  <!-- Main row -->
  <div
    class="flex items-center gap-2 px-3 py-2.5 transition-colors {anyVariantFits ||
    anyVariantHasInstance
      ? 'hover:bg-white/5 cursor-pointer'
      : 'cursor-not-allowed'} {isMainSelected
      ? 'bg-exo-yellow/10 border-l-2 border-exo-yellow'
      : 'border-l-2 border-transparent'}"
    onclick={() => {
      if (group.hasMultipleVariants) {
        onToggleExpand();
      } else {
        const modelId = group.variants[0]?.id;
        if (modelId && (canModelFit(modelId) || instanceStatuses[modelId])) {
          onSelectModel(modelId);
        }
      }
    }}
    role="button"
    tabindex="0"
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (group.hasMultipleVariants) {
          onToggleExpand();
        } else {
          const modelId = group.variants[0]?.id;
          if (modelId && (canModelFit(modelId) || instanceStatuses[modelId])) {
            onSelectModel(modelId);
          }
        }
      }
    }}
  >
    <!-- Expand/collapse chevron (for groups with variants) -->
    {#if group.hasMultipleVariants}
      <svg
        class="w-4 h-4 text-white/40 transition-transform duration-200 flex-shrink-0 {isExpanded
          ? 'rotate-90'
          : ''}"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z" />
      </svg>
    {:else}
      <div class="w-4 flex-shrink-0"></div>
    {/if}

    <!-- Model name -->
    <div class="flex-1 min-w-0">
      <div class="flex items-center gap-2">
        <span class="font-mono text-sm text-white truncate">
          {group.name}
        </span>
        <!-- Capability icons -->
        {#each group.capabilities.filter((c) => c !== "text") as cap}
          {#if cap === "thinking"}
            <svg
              class="w-3.5 h-3.5 text-white/40 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              title="Supports Thinking"
            >
              <path
                d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7zM9 20h6M10 22h4"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          {:else if cap === "code"}
            <svg
              class="w-3.5 h-3.5 text-white/40 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              title="Supports code generation"
            >
              <path
                d="M16 18l6-6-6-6M8 6l-6 6 6 6"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          {:else if cap === "vision"}
            <svg
              class="w-3.5 h-3.5 text-white/40 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              title="Supports image input"
            >
              <path
                d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <circle cx="12" cy="12" r="3" />
            </svg>
          {:else if cap === "image_gen"}
            <svg
              class="w-3.5 h-3.5 text-white/40 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              title="Supports image generation"
            >
              <rect
                x="3"
                y="3"
                width="18"
                height="18"
                rx="2"
                ry="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <path
                d="M21 15l-5-5L5 21"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          {:else if cap === "image_edit"}
            <svg
              class="w-3.5 h-3.5 text-white/40 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              title="Supports image editing"
            >
              <path
                d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          {/if}
        {/each}
      </div>
    </div>

    <!-- Size indicator (smallest variant) -->
    {#if !group.hasMultipleVariants && group.smallestVariant?.storage_size_megabytes}
      {@const singleVariantFitStatus = getModelFitStatus(
        group.smallestVariant.id,
      )}
      <span
        class="text-xs font-mono flex-shrink-0 {getSizeClassForFitStatus(
          singleVariantFitStatus,
        )}"
      >
        {formatSize(group.smallestVariant.storage_size_megabytes)}
      </span>
    {/if}

    <!-- Variant count with size range -->
    {#if group.hasMultipleVariants}
      {@const sizes = group.variants
        .map((v) => v.storage_size_megabytes || 0)
        .filter((s) => s > 0)
        .sort((a, b) => a - b)}
      <span
        class="text-xs font-mono flex-shrink-0 {getSizeClassForFitStatus(
          groupFitStatus,
        )}"
      >
        {group.variants.length} variants{#if sizes.length >= 2}{" "}({formatSize(
            sizes[0],
          )}-{formatSize(sizes[sizes.length - 1])}){/if}
      </span>
    {/if}

    <!-- Time ago (for recent models) -->
    {#if launchedAt}
      <span class="text-xs font-mono text-white/20 flex-shrink-0">
        {timeAgo(launchedAt)}
      </span>
    {/if}

    <!-- Download availability indicator -->
    {#if groupDownloadStatus && groupDownloadStatus.nodeIds.length > 0}
      <span
        class="flex-shrink-0"
        title={groupDownloadStatus.available
          ? `Ready — downloaded on ${groupDownloadStatus.nodeNames.join(", ")}`
          : `Downloaded on ${groupDownloadStatus.nodeNames.join(", ")} (may need more nodes)`}
      >
        <svg
          class="w-4 h-4"
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
      </span>
    {/if}

    <!-- Instance status badge -->
    {#if groupInstanceStatus}
      {#if groupInstanceStatus.status === "READY" || groupInstanceStatus.status === "LOADED" || groupInstanceStatus.status === "RUNNING"}
        <span class="flex-shrink-0" title="Running">
          <svg
            class="w-3 h-3 text-green-400"
            viewBox="0 0 12 12"
            fill="currentColor"
          >
            <circle cx="6" cy="6" r="5" />
          </svg>
        </span>
      {:else if groupInstanceStatus.status === "DOWNLOADING"}
        <span class="flex-shrink-0 animate-pulse" title="Downloading">
          <svg
            class="w-3.5 h-3.5 text-blue-400"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </span>
      {:else if groupInstanceStatus.status === "LOADING" || groupInstanceStatus.status === "WARMING UP"}
        <span class="flex-shrink-0 animate-pulse" title="Loading">
          <svg
            class="w-3 h-3 text-yellow-400"
            viewBox="0 0 12 12"
            fill="currentColor"
          >
            <circle cx="6" cy="6" r="5" />
          </svg>
        </span>
      {/if}
    {/if}

    <!-- Check mark if selected (single-variant) -->
    {#if isMainSelected}
      <svg
        class="w-4 h-4 text-exo-yellow flex-shrink-0"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
      </svg>
    {/if}

    <!-- Favorite star -->
    <button
      type="button"
      class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0"
      onclick={(e) => {
        e.stopPropagation();
        onToggleFavorite(group.id);
      }}
      title={isFavorite ? "Remove from favorites" : "Add to favorites"}
    >
      {#if isFavorite}
        <svg
          class="w-4 h-4 text-amber-400"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path
            d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
          />
        </svg>
      {:else}
        <svg
          class="w-4 h-4 text-white/30 hover:text-white/50"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <path
            d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
          />
        </svg>
      {/if}
    </button>

    <!-- Info button -->
    <button
      type="button"
      class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0"
      onclick={(e) => {
        e.stopPropagation();
        onShowInfo(group);
      }}
      title="View model details"
    >
      <svg
        class="w-4 h-4 text-white/30 hover:text-white/50"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <path
          d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"
        />
      </svg>
    </button>
  </div>

  <!-- Expanded variants -->
  {#if isExpanded && group.hasMultipleVariants}
    <div class="bg-black/20 border-t border-white/5">
      {#each group.variants as variant}
        {@const fitStatus = getModelFitStatus(variant.id)}
        {@const modelCanFit = canModelFit(variant.id)}
        {@const variantHasInstance = instanceStatuses[variant.id] != null}
        {@const isSelected = selectedModelId === variant.id}
        <div
          class="w-full flex items-center gap-3 px-3 py-2 pl-10 hover:bg-white/5 transition-colors text-left {!modelCanFit &&
          !variantHasInstance
            ? 'opacity-50 cursor-not-allowed'
            : 'cursor-pointer'} {isSelected
            ? 'bg-exo-yellow/10 border-l-2 border-exo-yellow'
            : 'border-l-2 border-transparent'}"
          role="button"
          tabindex="0"
          onclick={() => {
            if (modelCanFit || variantHasInstance) {
              onSelectModel(variant.id);
            }
          }}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              if (modelCanFit) {
                onSelectModel(variant.id);
              }
            }
          }}
        >
          <!-- Quantization badge -->
          <span
            class="text-xs font-mono px-1.5 py-0.5 rounded bg-white/10 text-white/70 flex-shrink-0"
          >
            {variant.quantization || "default"}
          </span>

          <!-- Size -->
          <span
            class="text-xs font-mono flex-1 {getSizeClassForFitStatus(
              fitStatus,
            )}"
          >
            {formatSize(variant.storage_size_megabytes)}
          </span>

          <!-- Download indicator for this variant -->
          {#if downloadStatusMap?.get(variant.id)}
            {@const variantDl = downloadStatusMap.get(variant.id)}
            {#if variantDl}
              <span
                class="flex-shrink-0"
                title={`Downloaded on ${variantDl.nodeNames.join(", ")}`}
              >
                <svg
                  class="w-3.5 h-3.5"
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
              </span>
            {/if}
          {/if}

          <!-- Instance status badge -->
          {#if instanceStatuses[variant.id]}
            {@const instStatus = instanceStatuses[variant.id]}
            {#if instStatus.status === "READY" || instStatus.status === "LOADED" || instStatus.status === "RUNNING"}
              <span class="flex-shrink-0" title="Running">
                <svg
                  class="w-3 h-3 text-green-400"
                  viewBox="0 0 12 12"
                  fill="currentColor"
                >
                  <circle cx="6" cy="6" r="5" />
                </svg>
              </span>
            {:else if instStatus.status === "DOWNLOADING"}
              <span class="flex-shrink-0 animate-pulse" title="Downloading">
                <svg
                  class="w-3.5 h-3.5 text-blue-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
              </span>
            {:else if instStatus.status === "LOADING" || instStatus.status === "WARMING UP"}
              <span class="flex-shrink-0 animate-pulse" title="Loading">
                <svg
                  class="w-3 h-3 text-yellow-400"
                  viewBox="0 0 12 12"
                  fill="currentColor"
                >
                  <circle cx="6" cy="6" r="5" />
                </svg>
              </span>
            {/if}
          {/if}

          <!-- Check mark if selected -->
          {#if isSelected}
            <svg
              class="w-4 h-4 text-exo-yellow"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path
                d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"
              />
            </svg>
          {/if}

          <!-- Info button -->
          <button
            type="button"
            class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0"
            onclick={(e) => {
              e.stopPropagation();
              onShowInfo({
                id: variant.id,
                name: variant.name || variant.id,
                capabilities: group.capabilities,
                family: group.family,
                variants: [variant],
                smallestVariant: variant,
                hasMultipleVariants: false,
              });
            }}
            title="View variant details"
          >
            <svg
              class="w-4 h-4 text-white/30 hover:text-white/50"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"
              />
            </svg>
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>

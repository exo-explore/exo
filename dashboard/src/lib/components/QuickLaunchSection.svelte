<script lang="ts">
  interface ModelInfo {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
    base_model?: string;
  }

  type ModelFitStatus = "fits_now" | "fits_cluster_capacity" | "too_large";

  type QuickLaunchProps = {
    models: ModelInfo[];
    getModelFitStatus: (model: ModelInfo) => ModelFitStatus;
    getModelSizeGB: (model: ModelInfo) => number;
    favoriteIds: Set<string>;
    recentModelIds: string[];
    downloadedModelIds: Set<string>;
    hasRunningInstance: boolean;
    onLaunch: (modelId: string) => void;
    onSelect: (modelId: string) => void;
  };

  let {
    models,
    getModelFitStatus,
    getModelSizeGB,
    favoriteIds,
    recentModelIds,
    downloadedModelIds,
    hasRunningInstance,
    onLaunch,
    onSelect,
  }: QuickLaunchProps = $props();

  // Filter to fits_now models and score/sort them
  const recommendedModels = $derived.by(() => {
    const fitting = models.filter(
      (m) => getModelFitStatus(m) === "fits_now",
    );

    // Score each model for ranking: recent > downloaded > favorite > largest
    const recentSet = new Set(recentModelIds);
    const scored = fitting.map((m) => {
      const baseId = m.base_model ?? m.id;
      let score = 0;
      const recentIdx = recentModelIds.indexOf(m.id);
      if (recentIdx >= 0) score += 1000 - recentIdx; // recent, higher = more recent
      if (downloadedModelIds.has(m.id)) score += 500;
      if (favoriteIds.has(baseId)) score += 250;
      score += getModelSizeGB(m); // bigger = better
      return { model: m, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 3).map((s) => s.model);
  });

  function formatSize(gb: number): string {
    if (gb >= 1) return `${gb.toFixed(0)}GB`;
    return `${(gb * 1024).toFixed(0)}MB`;
  }
</script>

{#if !hasRunningInstance && recommendedModels.length > 0}
  <div class="mb-3">
    <div class="flex items-center gap-2 mb-2">
      <span class="text-[10px] text-white/40 font-mono tracking-wider uppercase">Recommended</span>
      <div class="flex-1 h-px bg-white/10"></div>
    </div>
    <div class="space-y-1.5">
      {#each recommendedModels as model}
        {@const sizeGB = getModelSizeGB(model)}
        <div class="flex items-center gap-2 group">
          <button
            type="button"
            class="flex-1 min-w-0 flex items-center gap-2 px-2.5 py-1.5 bg-white/5 border border-white/10 rounded text-left hover:border-exo-yellow/30 hover:bg-white/8 transition-all cursor-pointer"
            onclick={() => onSelect(model.id)}
          >
            <span class="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0"></span>
            <span class="text-xs font-mono text-white/80 truncate flex-1">
              {model.name || model.id}
            </span>
            <span class="text-[10px] font-mono text-white/30 flex-shrink-0">
              {formatSize(sizeGB)}
            </span>
          </button>
          <button
            type="button"
            class="flex-shrink-0 px-2.5 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-yellow/10 text-exo-yellow border border-exo-yellow/30 rounded hover:bg-exo-yellow/20 transition-all cursor-pointer"
            onclick={() => onLaunch(model.id)}
          >
            &#9656; LAUNCH
          </button>
        </div>
      {/each}
    </div>
  </div>
{/if}

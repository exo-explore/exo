<script lang="ts" module>
  export interface ChatModelInfo {
    id: string;
    name: string;
    base_model: string;
    storage_size_megabytes: number;
    capabilities: string[];
    family: string;
    quantization: string;
  }

  // Auto mode tier list (for when user just starts typing)
  export const AUTO_TIERS: string[][] = [
    // Tier 1 (frontier)
    ["DeepSeek V3.1", "GLM-5", "Kimi K2.5", "Qwen3 Coder Next"],
    // Tier 2 (excellent)
    [
      "Kimi K2",
      "Qwen3 235B",
      "MiniMax M2.5",
      "Step 3.5 Flash",
      "Qwen3 Next 80B",
    ],
    // Tier 3 (great)
    [
      "GLM 4.7",
      "MiniMax M2.1",
      "Qwen3 Coder 480B",
      "GLM 4.5 Air",
      "Llama 3.3 70B",
    ],
    // Tier 4 (good)
    ["GPT-OSS 120B", "Qwen3 30B", "Llama 3.1 70B", "GLM 4.7 Flash"],
    // Tier 5 (small/fast)
    [
      "Llama 3.1 8B",
      "GPT-OSS 20B",
      "Llama 3.2 3B",
      "Qwen3 0.6B",
      "Llama 3.2 1B",
    ],
  ];

  /** Return the tier index (0 = best) for a base_model name. */
  export function getAutoTierIndex(baseModel: string): number {
    for (let i = 0; i < AUTO_TIERS.length; i++) {
      if (AUTO_TIERS[i].includes(baseModel)) return i;
    }
    return AUTO_TIERS.length; // not in any tier → lowest priority
  }

  /** Auto mode: walk tiers top-down, pick biggest fitting variant from highest tier. */
  export function pickAutoModel(
    modelList: ChatModelInfo[],
    memoryGB: number,
  ): ChatModelInfo | null {
    for (const tier of AUTO_TIERS) {
      const candidates: ChatModelInfo[] = [];
      for (const baseModel of tier) {
        const variants = modelList
          .filter(
            (m) =>
              m.base_model === baseModel &&
              (m.storage_size_megabytes || 0) / 1024 <= memoryGB &&
              (m.storage_size_megabytes || 0) > 0,
          )
          .sort(
            (a, b) =>
              (b.storage_size_megabytes || 0) - (a.storage_size_megabytes || 0),
          );
        if (variants[0]) candidates.push(variants[0]);
      }
      if (candidates.length > 0) {
        candidates.sort(
          (a, b) =>
            (b.storage_size_megabytes || 0) - (a.storage_size_megabytes || 0),
        );
        return candidates[0];
      }
    }
    return null;
  }
</script>

<script lang="ts">
  interface CategoryRecommendation {
    category: "coding" | "writing" | "agentic" | "biggest";
    label: string;
    model: ChatModelInfo | null;
    tooltip: string;
  }

  interface Props {
    models: ChatModelInfo[];
    clusterLabel: string;
    totalMemoryGB: number;
    onSelect: (modelId: string, category: string) => void;
    onAddModel: () => void;
    class?: string;
  }

  let {
    models,
    clusterLabel,
    totalMemoryGB,
    onSelect,
    onAddModel,
    class: className = "",
  }: Props = $props();

  // --- Hardcoded Rankings ---
  const CODING_RANKING = [
    "Qwen3 Coder Next",
    "Qwen3 Coder 480B",
    "Qwen3 30B",
    "GPT-OSS 20B",
    "Llama 3.1 8B",
    "Llama 3.2 3B",
    "Qwen3 0.6B",
  ];

  const WRITING_RANKING = [
    "Kimi K2.5",
    "Kimi K2",
    "Qwen3 Next 80B",
    "Llama 3.3 70B",
    "MiniMax M2.5",
    "GLM 4.5 Air",
    "GLM 4.7 Flash",
    "GPT-OSS 20B",
    "Llama 3.1 8B",
    "Llama 3.2 3B",
    "Qwen3 0.6B",
  ];

  const AGENTIC_RANKING = [
    "DeepSeek V3.1",
    "GLM-5",
    "Qwen3 235B",
    "Step 3.5 Flash",
    "GLM 4.7",
    "MiniMax M2.1",
    "GPT-OSS 120B",
    "Llama 3.3 70B",
    "Llama 3.1 70B",
    "GLM 4.7 Flash",
    "GPT-OSS 20B",
    "Qwen3 30B",
    "Llama 3.1 8B",
    "Llama 3.2 3B",
    "Qwen3 0.6B",
  ];

  function getModelSizeGB(m: ChatModelInfo): number {
    return (m.storage_size_megabytes || 0) / 1024;
  }

  function fitsInMemory(m: ChatModelInfo): boolean {
    return getModelSizeGB(m) <= totalMemoryGB && getModelSizeGB(m) > 0;
  }

  /** For a given base_model name, find the biggest quant variant that fits in memory. */
  function pickBestVariant(baseModel: string): ChatModelInfo | null {
    const variants = models
      .filter((m) => m.base_model === baseModel && fitsInMemory(m))
      .sort((a, b) => getModelSizeGB(b) - getModelSizeGB(a));
    return variants[0] ?? null;
  }

  /** Walk a ranked list of base_model names, return the first that has a fitting variant. */
  function pickFromRanking(ranking: string[]): ChatModelInfo | null {
    for (const baseModel of ranking) {
      const pick = pickBestVariant(baseModel);
      if (pick) return pick;
    }
    return null;
  }

  /** Pick the single biggest model that fits. */
  function pickBiggest(): ChatModelInfo | null {
    const fitting = models
      .filter((m) => fitsInMemory(m))
      .sort((a, b) => getModelSizeGB(b) - getModelSizeGB(a));
    return fitting[0] ?? null;
  }

  const recommendations = $derived.by((): CategoryRecommendation[] => {
    return [
      {
        category: "coding",
        label: "Best for Coding",
        model: pickFromRanking(CODING_RANKING),
        tooltip:
          "Ranked by coding benchmark performance (LiveCodeBench, SWE-bench)",
      },
      {
        category: "writing",
        label: "Best for Writing",
        model: pickFromRanking(WRITING_RANKING),
        tooltip: "Ranked by creative writing quality and instruction following",
      },
      {
        category: "agentic",
        label: "Best Agentic",
        model: pickFromRanking(AGENTIC_RANKING),
        tooltip: "Ranked by reasoning, planning, and tool-use capability",
      },
      {
        category: "biggest",
        label: "Biggest",
        model: pickBiggest(),
        tooltip: "Largest model that fits in your available memory",
      },
    ];
  });

  function formatSize(mb: number): string {
    const gb = mb / 1024;
    if (gb >= 100) return `${Math.round(gb)} GB`;
    return `${gb.toFixed(1)} GB`;
  }

  // Category icons (SVG paths)
  const categoryIcons: Record<string, string> = {
    coding:
      "M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z",
    writing:
      "M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z",
    agentic:
      "M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z",
    biggest:
      "M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10",
  };

  let hoveredTooltip = $state<string | null>(null);
  let tooltipAnchor = $state<{ x: number; y: number } | null>(null);

  function showTooltip(category: string, e: MouseEvent | FocusEvent) {
    hoveredTooltip = category;
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    tooltipAnchor = { x: rect.left + rect.width / 2, y: rect.top };
  }

  function hideTooltip() {
    hoveredTooltip = null;
    tooltipAnchor = null;
  }
</script>

<div class="flex flex-col items-center justify-center gap-6 {className}">
  <!-- Header -->
  <div class="text-center">
    <p class="text-xs text-exo-light-gray uppercase tracking-[0.2em] mb-1">
      Recommended for your
    </p>
    <p class="text-sm text-white font-mono tracking-wide">{clusterLabel}</p>
  </div>

  <!-- Category Cards Grid -->
  <div class="grid grid-cols-2 gap-3 w-full max-w-md">
    {#each recommendations as rec}
      {#if rec.model}
        <button
          type="button"
          onclick={() => rec.model && onSelect(rec.model.id, rec.category)}
          class="group relative flex flex-col items-start gap-2 p-4 rounded-lg border border-exo-medium-gray/50 bg-exo-dark-gray/50 hover:border-exo-yellow/40 hover:bg-exo-dark-gray transition-all duration-200 cursor-pointer text-left"
        >
          <!-- Category icon + label -->
          <div class="flex items-center gap-2 w-full">
            <svg
              class="w-4 h-4 text-exo-yellow/70 group-hover:text-exo-yellow transition-colors flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="1.5"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d={categoryIcons[rec.category]}
              />
            </svg>
            <span
              class="text-xs font-mono uppercase tracking-wider text-exo-light-gray group-hover:text-white transition-colors"
            >
              {rec.label}
            </span>
            <!-- Info tooltip -->
            <div class="ml-auto flex-shrink-0">
              <span
                role="button"
                tabindex="-1"
                class="text-exo-light-gray/40 hover:text-exo-light-gray transition-colors cursor-help inline-flex"
                onmouseenter={(e: MouseEvent) => showTooltip(rec.category, e)}
                onmouseleave={() => hideTooltip()}
                onclick={(e: MouseEvent) => {
                  e.stopPropagation();
                  if (hoveredTooltip === rec.category) {
                    hideTooltip();
                  } else {
                    showTooltip(rec.category, e);
                  }
                }}
              >
                <svg
                  class="w-3.5 h-3.5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  stroke-width="2"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 16v-4m0-4h.01" />
                </svg>
              </span>
            </div>
          </div>

          <!-- Model name + size -->
          <div class="w-full">
            <p class="text-sm text-white font-mono truncate">
              {rec.model.base_model}
            </p>
            <p class="text-xs text-exo-light-gray/60 font-mono mt-0.5">
              {formatSize(rec.model.storage_size_megabytes)}
              {#if rec.model.quantization}
                <span class="text-exo-light-gray/40"
                  >&middot; {rec.model.quantization}</span
                >
              {/if}
            </p>
          </div>
        </button>
      {:else}
        <!-- No model fits for this category -->
        <div
          class="flex flex-col items-start gap-2 p-4 rounded-lg border border-exo-medium-gray/30 bg-exo-dark-gray/30 opacity-50"
        >
          <div class="flex items-center gap-2">
            <svg
              class="w-4 h-4 text-exo-light-gray/40 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="1.5"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d={categoryIcons[rec.category]}
              />
            </svg>
            <span
              class="text-xs font-mono uppercase tracking-wider text-exo-light-gray/50"
              >{rec.label}</span
            >
          </div>
          <p class="text-xs text-exo-light-gray/40 font-mono">No model fits</p>
        </div>
      {/if}
    {/each}
  </div>

  <!-- Add Model Button -->
  <button
    type="button"
    onclick={onAddModel}
    class="flex items-center gap-2 px-4 py-2 text-xs font-mono uppercase tracking-wider text-exo-light-gray hover:text-exo-yellow border border-exo-medium-gray/30 hover:border-exo-yellow/30 rounded-lg transition-all duration-200 cursor-pointer"
  >
    <svg
      class="w-3.5 h-3.5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      stroke-width="2"
    >
      <path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4" />
    </svg>
    Add Model
  </button>

  <!-- Auto hint -->
  <p class="text-xs text-exo-light-gray/40 font-mono tracking-wide text-center">
    Or just start typing &mdash; we'll pick the best model automatically
  </p>
</div>

<!-- Fixed-position tooltip (escapes overflow-hidden ancestors) -->
{#if hoveredTooltip && tooltipAnchor}
  {@const rec = recommendations.find((r) => r.category === hoveredTooltip)}
  {#if rec}
    <div
      class="fixed z-[9999] px-3 py-2 bg-exo-black border border-exo-medium-gray/50 rounded text-xs text-exo-light-gray whitespace-nowrap shadow-lg pointer-events-none"
      style="left: {tooltipAnchor.x}px; top: {tooltipAnchor.y -
        8}px; transform: translate(-50%, -100%);"
    >
      {rec.tooltip}
    </div>
  {/if}
{/if}

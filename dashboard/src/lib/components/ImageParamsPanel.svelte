<script lang="ts">
  import {
    imageGenerationParams,
    setImageGenerationParams,
    resetImageGenerationParams,
    type ImageGenerationParams,
  } from "$lib/stores/app.svelte";

  interface Props {
    isEditMode?: boolean;
  }

  let { isEditMode = false }: Props = $props();

  let showAdvanced = $state(false);

  // Custom dropdown state
  let isSizeDropdownOpen = $state(false);
  let isQualityDropdownOpen = $state(false);
  let sizeButtonRef: HTMLButtonElement | undefined = $state();
  let qualityButtonRef: HTMLButtonElement | undefined = $state();

  const sizeDropdownPosition = $derived(() => {
    if (!sizeButtonRef || !isSizeDropdownOpen)
      return { top: 0, left: 0, width: 0 };
    const rect = sizeButtonRef.getBoundingClientRect();
    return { top: rect.top, left: rect.left, width: rect.width };
  });

  const qualityDropdownPosition = $derived(() => {
    if (!qualityButtonRef || !isQualityDropdownOpen)
      return { top: 0, left: 0, width: 0 };
    const rect = qualityButtonRef.getBoundingClientRect();
    return { top: rect.top, left: rect.left, width: rect.width };
  });

  const params = $derived(imageGenerationParams());

  const inputFidelityOptions: ImageGenerationParams["inputFidelity"][] = [
    "low",
    "high",
  ];

  const outputFormatOptions: ImageGenerationParams["outputFormat"][] = [
    "png",
    "jpeg",
  ];

  function handleInputFidelityChange(
    value: ImageGenerationParams["inputFidelity"],
  ) {
    setImageGenerationParams({ inputFidelity: value });
  }

  function handleOutputFormatChange(
    value: ImageGenerationParams["outputFormat"],
  ) {
    setImageGenerationParams({ outputFormat: value });
  }

  const sizeOptions: ImageGenerationParams["size"][] = [
    "512x512",
    "768x768",
    "1024x1024",
    "1024x768",
    "768x1024",
  ];

  const qualityOptions: ImageGenerationParams["quality"][] = [
    "low",
    "medium",
    "high",
  ];

  function selectSize(value: ImageGenerationParams["size"]) {
    setImageGenerationParams({ size: value });
    isSizeDropdownOpen = false;
  }

  function selectQuality(value: ImageGenerationParams["quality"]) {
    setImageGenerationParams({ quality: value });
    isQualityDropdownOpen = false;
  }

  function handleSeedChange(event: Event) {
    const input = event.target as HTMLInputElement;
    const value = input.value.trim();
    if (value === "") {
      setImageGenerationParams({ seed: null });
    } else {
      const num = parseInt(value, 10);
      if (!isNaN(num) && num >= 0) {
        setImageGenerationParams({ seed: num });
      }
    }
  }

  function handleStepsChange(event: Event) {
    const value = parseInt((event.target as HTMLInputElement).value, 10);
    setImageGenerationParams({ numInferenceSteps: value });
  }

  function handleGuidanceChange(event: Event) {
    const value = parseFloat((event.target as HTMLInputElement).value);
    setImageGenerationParams({ guidance: value });
  }

  function handleNegativePromptChange(event: Event) {
    const value = (event.target as HTMLTextAreaElement).value;
    setImageGenerationParams({ negativePrompt: value || null });
  }

  function clearSteps() {
    setImageGenerationParams({ numInferenceSteps: null });
  }

  function clearGuidance() {
    setImageGenerationParams({ guidance: null });
  }

  function handleReset() {
    resetImageGenerationParams();
    showAdvanced = false;
  }

  const hasAdvancedParams = $derived(
    params.seed !== null ||
      params.numInferenceSteps !== null ||
      params.guidance !== null ||
      (params.negativePrompt !== null && params.negativePrompt.trim() !== ""),
  );
</script>

<div class="border-b border-exo-medium-gray/30 px-3 py-2">
  <!-- Basic params row -->
  <div class="flex items-center gap-3 flex-wrap">
    <!-- Size -->
    <div class="flex items-center gap-1.5">
      <span class="text-xs text-exo-light-gray uppercase tracking-wider"
        >SIZE:</span
      >
      <div class="relative">
        <button
          bind:this={sizeButtonRef}
          type="button"
          onclick={() => (isSizeDropdownOpen = !isSizeDropdownOpen)}
          class="bg-exo-medium-gray/50 border border-exo-yellow/30 rounded pl-2 pr-6 py-1 text-xs font-mono text-exo-yellow cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70 {isSizeDropdownOpen
            ? 'border-exo-yellow/70'
            : ''}"
        >
          {params.size}
        </button>
        <div
          class="absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none transition-transform duration-200 {isSizeDropdownOpen
            ? 'rotate-180'
            : ''}"
        >
          <svg
            class="w-3 h-3 text-exo-yellow/60"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>

      {#if isSizeDropdownOpen}
        <!-- Backdrop to close dropdown -->
        <button
          type="button"
          class="fixed inset-0 z-[9998] cursor-default"
          onclick={() => (isSizeDropdownOpen = false)}
          aria-label="Close dropdown"
        ></button>

        <!-- Dropdown Panel - fixed positioning to escape overflow:hidden -->
        <div
          class="fixed bg-exo-dark-gray border border-exo-yellow/30 rounded shadow-lg shadow-black/50 z-[9999] max-h-48 overflow-y-auto min-w-max"
          style="bottom: calc(100vh - {sizeDropdownPosition()
            .top}px + 4px); left: {sizeDropdownPosition().left}px;"
        >
          <div class="py-1">
            {#each sizeOptions as size}
              <button
                type="button"
                onclick={() => selectSize(size)}
                class="w-full px-3 py-1.5 text-left text-xs font-mono tracking-wide transition-colors duration-100 flex items-center gap-2 {params.size ===
                size
                  ? 'bg-transparent text-exo-yellow'
                  : 'text-exo-light-gray hover:text-exo-yellow'}"
              >
                {#if params.size === size}
                  <svg
                    class="w-3 h-3 flex-shrink-0"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fill-rule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clip-rule="evenodd"
                    />
                  </svg>
                {:else}
                  <span class="w-3"></span>
                {/if}
                <span>{size}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Quality -->
    <div class="flex items-center gap-1.5">
      <span class="text-xs text-exo-light-gray uppercase tracking-wider"
        >QUALITY:</span
      >
      <div class="relative">
        <button
          bind:this={qualityButtonRef}
          type="button"
          onclick={() => (isQualityDropdownOpen = !isQualityDropdownOpen)}
          class="bg-exo-medium-gray/50 border border-exo-yellow/30 rounded pl-2 pr-6 py-1 text-xs font-mono text-exo-yellow cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70 {isQualityDropdownOpen
            ? 'border-exo-yellow/70'
            : ''}"
        >
          {params.quality.toUpperCase()}
        </button>
        <div
          class="absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none transition-transform duration-200 {isQualityDropdownOpen
            ? 'rotate-180'
            : ''}"
        >
          <svg
            class="w-3 h-3 text-exo-yellow/60"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>

      {#if isQualityDropdownOpen}
        <!-- Backdrop to close dropdown -->
        <button
          type="button"
          class="fixed inset-0 z-[9998] cursor-default"
          onclick={() => (isQualityDropdownOpen = false)}
          aria-label="Close dropdown"
        ></button>

        <!-- Dropdown Panel - fixed positioning to escape overflow:hidden -->
        <div
          class="fixed bg-exo-dark-gray border border-exo-yellow/30 rounded shadow-lg shadow-black/50 z-[9999] max-h-48 overflow-y-auto min-w-max"
          style="bottom: calc(100vh - {qualityDropdownPosition()
            .top}px + 4px); left: {qualityDropdownPosition().left}px;"
        >
          <div class="py-1">
            {#each qualityOptions as quality}
              <button
                type="button"
                onclick={() => selectQuality(quality)}
                class="w-full px-3 py-1.5 text-left text-xs font-mono tracking-wide transition-colors duration-100 flex items-center gap-2 {params.quality ===
                quality
                  ? 'bg-transparent text-exo-yellow'
                  : 'text-exo-light-gray hover:text-exo-yellow'}"
              >
                {#if params.quality === quality}
                  <svg
                    class="w-3 h-3 flex-shrink-0"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fill-rule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clip-rule="evenodd"
                    />
                  </svg>
                {:else}
                  <span class="w-3"></span>
                {/if}
                <span>{quality.toUpperCase()}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Format -->
    <div class="flex items-center gap-1.5">
      <span class="text-xs text-exo-light-gray uppercase tracking-wider"
        >FORMAT:</span
      >
      <div class="flex rounded overflow-hidden border border-exo-yellow/30">
        {#each outputFormatOptions as format}
          <button
            type="button"
            onclick={() => handleOutputFormatChange(format)}
            class="px-2 py-1 text-xs font-mono uppercase transition-all duration-200 cursor-pointer {params.outputFormat ===
            format
              ? 'bg-exo-yellow text-exo-black'
              : 'bg-exo-medium-gray/50 text-exo-light-gray hover:text-exo-yellow'}"
          >
            {format}
          </button>
        {/each}
      </div>
    </div>

    <!-- Input Fidelity (edit mode only) -->
    {#if isEditMode}
      <div class="flex items-center gap-1.5">
        <span class="text-xs text-exo-light-gray uppercase tracking-wider"
          >FIDELITY:</span
        >
        <div class="flex rounded overflow-hidden border border-exo-yellow/30">
          {#each inputFidelityOptions as fidelity}
            <button
              type="button"
              onclick={() => handleInputFidelityChange(fidelity)}
              class="px-2 py-1 text-xs font-mono uppercase transition-all duration-200 cursor-pointer {params.inputFidelity ===
              fidelity
                ? 'bg-exo-yellow text-exo-black'
                : 'bg-exo-medium-gray/50 text-exo-light-gray hover:text-exo-yellow'}"
              title={fidelity === "low"
                ? "More creative variation"
                : "Closer to original"}
            >
              {fidelity}
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Spacer -->
    <div class="flex-1"></div>

    <!-- Advanced toggle -->
    <button
      type="button"
      onclick={() => (showAdvanced = !showAdvanced)}
      class="flex items-center gap-1 text-xs font-mono tracking-wider uppercase transition-colors duration-200 {showAdvanced ||
      hasAdvancedParams
        ? 'text-exo-yellow'
        : 'text-exo-light-gray hover:text-exo-yellow'}"
    >
      <span>ADVANCED</span>
      <svg
        class="w-3 h-3 transition-transform duration-200 {showAdvanced
          ? 'rotate-180'
          : ''}"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M19 9l-7 7-7-7"
        />
      </svg>
      {#if hasAdvancedParams && !showAdvanced}
        <span class="w-1.5 h-1.5 rounded-full bg-exo-yellow"></span>
      {/if}
    </button>
  </div>

  <!-- Advanced params section -->
  {#if showAdvanced}
    <div class="mt-3 pt-3 border-t border-exo-medium-gray/20 space-y-3">
      <!-- Row 1: Seed and Steps -->
      <div class="flex items-center gap-4 flex-wrap">
        <!-- Seed -->
        <div class="flex items-center gap-1.5">
          <span class="text-xs text-exo-light-gray uppercase tracking-wider"
            >SEED:</span
          >
          <input
            type="number"
            min="0"
            value={params.seed ?? ""}
            oninput={handleSeedChange}
            placeholder="Random"
            class="w-24 bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1 text-xs font-mono text-exo-yellow placeholder:text-exo-light-gray/50 transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
          />
        </div>

        <!-- Steps Slider -->
        <div class="flex items-center gap-1.5 flex-1 min-w-[200px]">
          <span
            class="text-xs text-exo-light-gray uppercase tracking-wider whitespace-nowrap"
            >STEPS:</span
          >
          <div class="flex items-center gap-2 flex-1">
            <input
              type="range"
              min="1"
              max="100"
              value={params.numInferenceSteps ?? 50}
              oninput={handleStepsChange}
              class="flex-1 h-1 bg-exo-medium-gray/50 rounded appearance-none cursor-pointer accent-exo-yellow"
            />
            <span class="text-xs font-mono text-exo-yellow w-8 text-right">
              {params.numInferenceSteps ?? "--"}
            </span>
            {#if params.numInferenceSteps !== null}
              <button
                type="button"
                onclick={clearSteps}
                class="text-exo-light-gray hover:text-exo-yellow transition-colors"
                title="Clear"
              >
                <svg
                  class="w-3 h-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            {/if}
          </div>
        </div>
      </div>

      <!-- Row 2: Guidance -->
      <div class="flex items-center gap-1.5">
        <span
          class="text-xs text-exo-light-gray uppercase tracking-wider whitespace-nowrap"
          >GUIDANCE:</span
        >
        <div class="flex items-center gap-2 flex-1 max-w-xs">
          <input
            type="range"
            min="1"
            max="20"
            step="0.5"
            value={params.guidance ?? 7.5}
            oninput={handleGuidanceChange}
            class="flex-1 h-1 bg-exo-medium-gray/50 rounded appearance-none cursor-pointer accent-exo-yellow"
          />
          <span class="text-xs font-mono text-exo-yellow w-8 text-right">
            {params.guidance !== null ? params.guidance.toFixed(1) : "--"}
          </span>
          {#if params.guidance !== null}
            <button
              type="button"
              onclick={clearGuidance}
              class="text-exo-light-gray hover:text-exo-yellow transition-colors"
              title="Clear"
            >
              <svg
                class="w-3 h-3"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          {/if}
        </div>
      </div>

      <!-- Row 3: Negative Prompt -->
      <div class="flex flex-col gap-1.5">
        <span class="text-xs text-exo-light-gray uppercase tracking-wider"
          >NEGATIVE PROMPT:</span
        >
        <textarea
          value={params.negativePrompt ?? ""}
          oninput={handleNegativePromptChange}
          placeholder="Things to avoid in the image..."
          rows={2}
          class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1.5 text-xs font-mono text-exo-yellow placeholder:text-exo-light-gray/50 resize-none transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
        ></textarea>
      </div>

      <!-- Reset Button -->
      <div class="flex justify-end pt-1">
        <button
          type="button"
          onclick={handleReset}
          class="text-xs font-mono tracking-wider uppercase text-exo-light-gray hover:text-exo-yellow transition-colors duration-200"
        >
          RESET TO DEFAULTS
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  /* Custom range slider styling */
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #ffd700;
    cursor: pointer;
    border: none;
  }

  input[type="range"]::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #ffd700;
    cursor: pointer;
    border: none;
  }

  /* Hide number input spinners */
  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }

  input[type="number"] {
    -moz-appearance: textfield;
  }
</style>

<script lang="ts">
  import {
    RUNTIME_DESCRIPTIONS,
    RUNTIME_LABELS,
    RUNTIMES,
    type InstanceMeta,
  } from "$lib/utils/runtimes";

  let {
    value,
    onchange,
  }: {
    value: InstanceMeta;
    onchange: (runtime: InstanceMeta) => void;
  } = $props();
</script>

<fieldset>
  <legend class="text-xs text-white/50 font-mono mb-2">Runtime</legend>
  <div class="flex flex-wrap gap-2">
    {#each RUNTIMES as runtime (runtime)}
      <button
        type="button"
        aria-pressed={value === runtime}
        title={RUNTIME_DESCRIPTIONS[runtime]}
        onclick={() => onchange(runtime)}
        class="py-1.5 px-3 text-xs font-mono border rounded transition-colors cursor-pointer {value ===
        runtime
          ? 'text-exo-yellow border-exo-yellow'
          : 'text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
      >
        {RUNTIME_LABELS[runtime]}
      </button>
    {/each}
  </div>
  <p class="text-xs text-white/50 mt-2">{RUNTIME_DESCRIPTIONS[value]}</p>
</fieldset>

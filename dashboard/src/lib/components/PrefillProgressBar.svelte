<script lang="ts">
  import type { PrefillProgress } from "$lib/stores/app.svelte";

  interface Props {
    progress: PrefillProgress;
    class?: string;
  }

  let { progress, class: className = "" }: Props = $props();

  const percentage = $derived(
    progress.total > 0
      ? Math.round((progress.processed / progress.total) * 100)
      : 0,
  );

  function formatTokenCount(count: number): string {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`;
    }
    return count.toString();
  }
</script>

<div class="prefill-progress {className}">
  <div
    class="flex items-center justify-between text-xs text-exo-light-gray mb-1"
  >
    <span>Processing prompt</span>
    <span class="font-mono">
      {formatTokenCount(progress.processed)} / {formatTokenCount(
        progress.total,
      )} tokens
    </span>
  </div>
  <div class="h-1.5 bg-exo-black/60 rounded-full overflow-hidden">
    <div
      class="h-full bg-exo-yellow rounded-full transition-all duration-150 ease-out"
      style="width: {percentage}%"
    ></div>
  </div>
  <div class="text-right text-xs text-exo-light-gray/70 mt-0.5 font-mono">
    {percentage}%
  </div>
</div>

<style>
  .prefill-progress {
    width: 100%;
  }
</style>

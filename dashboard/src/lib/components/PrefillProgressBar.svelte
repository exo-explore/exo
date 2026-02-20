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

  const etaText = $derived.by(() => {
    if (progress.processed <= 0 || progress.total <= 0) return null;
    const elapsedMs = performance.now() - progress.startedAt;
    if (elapsedMs < 200) return null; // need a minimum sample window
    const tokensPerMs = progress.processed / elapsedMs;
    const remainingTokens = progress.total - progress.processed;
    const remainingMs = remainingTokens / tokensPerMs;
    const remainingSec = Math.ceil(remainingMs / 1000);
    if (remainingSec <= 0) return null;
    if (remainingSec < 60) return `~${remainingSec}s remaining`;
    const mins = Math.floor(remainingSec / 60);
    const secs = remainingSec % 60;
    return `~${mins}m ${secs}s remaining`;
  });

  function formatTokenCount(count: number | undefined): string {
    if (count == null) return "0";
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
  <div
    class="flex items-center justify-between text-xs text-exo-light-gray/70 mt-0.5 font-mono"
  >
    <span>{etaText ?? ""}</span>
    <span>{percentage}%</span>
  </div>
</div>

<style>
  .prefill-progress {
    width: 100%;
  }
</style>

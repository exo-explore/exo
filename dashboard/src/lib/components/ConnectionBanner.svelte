<script lang="ts">
  import { isConnected } from "$lib/stores/app.svelte";
  import { slide } from "svelte/transition";

  const connected = $derived(isConnected());
</script>

{#if !connected}
  <div
    transition:slide={{ duration: 200 }}
    class="relative z-50 flex items-center justify-center gap-2 px-4 py-2 bg-red-950/80 border-b border-red-500/30"
    role="alert"
    aria-live="assertive"
  >
    <div class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
    <span class="text-xs font-mono text-red-300 tracking-wider uppercase">
      Connection lost &mdash; Reconnecting to backend&hellip;
    </span>
  </div>
{/if}

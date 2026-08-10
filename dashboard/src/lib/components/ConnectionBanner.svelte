<script lang="ts">
  import { isConnected } from "$lib/stores/app.svelte";
  import { slide } from "svelte/transition";

  const connected = $derived(isConnected());
</script>

{#if !connected}
  <div
    transition:slide={{ duration: 200 }}
    class="relative z-50 flex min-h-9 items-center justify-center gap-2.5 border-b border-amber-400/15 bg-amber-400/[0.055] px-4 py-2"
    role="status"
    aria-live="polite"
  >
    <span class="relative flex h-2 w-2" aria-hidden="true">
      <span
        class="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-300 opacity-50"
      ></span>
      <span class="relative inline-flex h-2 w-2 rounded-full bg-amber-300"
      ></span>
    </span>
    <span
      class="text-[11px] font-mono font-medium tracking-[0.08em] text-amber-100/80"
    >
      Backend offline
    </span>
    <span class="hidden h-3 w-px bg-white/10 sm:block" aria-hidden="true"
    ></span>
    <span class="hidden text-[11px] text-white/40 sm:inline"
      >Reconnecting automatically&hellip;</span
    >
  </div>
{/if}

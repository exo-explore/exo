<script lang="ts">
  import { browser } from "$app/environment";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import PrefillDecodeDisaggregation from "$lib/components/PrefillDecodeDisaggregation.svelte";
  import { featureFlags, refreshState } from "$lib/stores/app.svelte";
  import { onMount } from "svelte";

  type TabId = "prefill-decode";

  const tabs: { id: TabId; label: string }[] = [
    { id: "prefill-decode", label: "Prefill / Decode" },
  ];

  let activeTab = $state<TabId>(tabs[0].id);
  let flagsLoaded = $state(false);

  onMount(() => {
    refreshState().finally(() => {
      flagsLoaded = true;
    });
  });

  const flags = $derived(featureFlags());
  const enabled = $derived(flags["disaggregation"] === true);

  $effect(() => {
    if (browser && flagsLoaded && !enabled) {
      // No advanced features enabled — bounce home.
      window.location.hash = "/";
    }
  });
</script>

<div class="min-h-screen bg-exo-dark-gray flex flex-col">
  <HeaderNav />

  <main class="flex-1 max-w-[1100px] mx-auto w-full px-4 md:px-6 py-8">
    {#if !flagsLoaded}
      <div class="text-exo-light-gray/60 text-sm">Loading…</div>
    {:else if !enabled}
      <div class="text-exo-light-gray/60 text-sm">
        No advanced features enabled. Set <code
          class="text-exo-yellow font-mono">ENABLE_DISAGGREGATION=true</code
        > on the cluster to access prefill/decode disaggregation.
      </div>
    {:else}
      <div class="mb-4">
        <h1
          class="text-white text-xl md:text-2xl font-semibold tracking-wide mb-2"
        >
          Advanced
        </h1>
        <p class="text-exo-light-gray/60 text-sm">
          Cluster-level configuration. Most users don't need anything here.
        </p>
      </div>

      <div
        class="flex flex-wrap gap-2 mb-6 border-b border-exo-light-gray/10 pb-3"
      >
        {#each tabs as tab (tab.id)}
          <button
            onclick={() => (activeTab = tab.id)}
            class="px-3 py-1.5 text-xs rounded-md transition-all cursor-pointer
              {activeTab === tab.id
              ? 'bg-exo-yellow/15 text-exo-yellow border border-exo-yellow/30'
              : 'text-exo-light-gray/60 hover:text-white/80 border border-transparent hover:border-exo-light-gray/20'}"
          >
            {tab.label}
          </button>
        {/each}
      </div>

      <div class="space-y-4">
        {#if activeTab === "prefill-decode"}
          <PrefillDecodeDisaggregation />
        {/if}
      </div>
    {/if}
  </main>
</div>

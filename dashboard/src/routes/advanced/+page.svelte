<script lang="ts">
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import PrefillDecodeDisaggregation from "$lib/components/PrefillDecodeDisaggregation.svelte";

  type TabId = "prefill-decode";

  type Tab = {
    id: TabId;
    label: string;
  };

  const tabs: Tab[] = [
    { id: "prefill-decode", label: "Prefill / Decode disaggregation" },
  ];

  let activeTab = $state<TabId>(tabs[0].id);
</script>

<HeaderNav />

<main class="page">
  <header class="page-header">
    <h1>Advanced</h1>
    <p>Cluster-level configuration. Most users don't need anything here.</p>
  </header>

  <div class="layout">
    <nav class="tab-list" aria-label="Advanced settings sections">
      {#each tabs as tab (tab.id)}
        <button
          class="tab"
          class:active={activeTab === tab.id}
          onclick={() => (activeTab = tab.id)}
        >
          {tab.label}
        </button>
      {/each}
    </nav>

    <section class="tab-panel">
      {#if activeTab === "prefill-decode"}
        <PrefillDecodeDisaggregation />
      {/if}
    </section>
  </div>
</main>

<style>
  .page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.5rem;
    color: var(--text-primary, #eee);
  }
  .page-header h1 {
    margin: 0 0 0.5rem 0;
  }
  .page-header p {
    margin: 0 0 1.5rem 0;
    color: var(--text-secondary, #aaa);
  }
  .layout {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: 1.5rem;
    align-items: start;
  }
  @media (max-width: 720px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }
  .tab-list {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    padding-right: 1rem;
  }
  @media (max-width: 720px) {
    .tab-list {
      flex-direction: row;
      flex-wrap: wrap;
      border-right: none;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-right: 0;
      padding-bottom: 0.5rem;
    }
  }
  .tab {
    background: transparent;
    color: inherit;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    text-align: left;
    cursor: pointer;
    font: inherit;
  }
  .tab:hover:not(.active) {
    background: rgba(255, 255, 255, 0.05);
  }
  .tab.active {
    background: rgba(80, 180, 255, 0.15);
    border-color: rgba(80, 180, 255, 0.5);
  }
  .tab-panel {
    min-width: 0;
  }
</style>

<script lang="ts">
  import { onMount } from "svelte";

  let origin = $state("http://localhost:52415");
  onMount(() => {
    if (typeof window !== "undefined") origin = window.location.origin;
  });

  const endpoints = $derived([
    { name: "OpenAI", url: `${origin}/v1` },
    { name: "Claude", url: `${origin}` },
    { name: "Ollama", url: `${origin}/ollama/api` },
    { name: "Embeddings", url: `${origin}/v1/embeddings` },
  ]);

  let copied = $state<string | null>(null);
  async function copy(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      copied = url;
      setTimeout(() => {
        if (copied === url) copied = null;
      }, 1600);
    } catch {
      // ignore
    }
  }
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">API endpoints</span>
    <span class="card-sublabel">{endpoints.length} ACTIVE</span>
  </div>
  <div class="list">
    {#each endpoints as ep}
      <div class="row">
        <span class="name">{ep.name}</span>
        <span class="url">{ep.url}</span>
        <button class="copy" onclick={() => copy(ep.url)} title="Copy URL">
          {#if copied === ep.url}✓{:else}⧉{/if}
        </button>
      </div>
    {/each}
  </div>
</div>

<style>
  .card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border-bottom: 1px solid var(--ux-border);
  }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--ux-text);
  }
  .card-sublabel {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .list {
    padding: 4px 0;
  }
  .row {
    display: grid;
    grid-template-columns: 100px 1fr auto;
    align-items: center;
    gap: 12px;
    padding: 11px 18px;
    border-bottom: 1px solid var(--ux-border);
  }
  .row:last-child {
    border-bottom: none;
  }
  .name {
    font-size: 12px;
    color: var(--ux-text);
    font-weight: 500;
  }
  .url {
    font-family: var(--ux-mono);
    font-size: 11.5px;
    color: var(--ux-text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .copy {
    background: transparent;
    border: none;
    color: var(--ux-text-faint);
    cursor: pointer;
    font-size: 14px;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    transition: background 120ms, color 120ms;
  }
  .copy:hover {
    background: var(--ux-bg-hover);
    color: var(--ux-text);
  }
</style>

<script lang="ts">
  import {
    registerCustomModel,
    startDownload,
    refreshState,
    topologyData,
  } from "$lib/stores/app.svelte";

  const data = $derived(topologyData());

  let customModelId = $state("");
  let isRegistering = $state(false);
  let registerError = $state<string | null>(null);

  async function handleDownloadCustom() {
    if (!customModelId.trim()) return;
    isRegistering = true;
    registerError = null;
    try {
      const { modelCard } = await registerCustomModel(customModelId.trim());
      // Construct shard metadata for the custom model
      const shardMetadata = {
        model_card: modelCard,
        device_rank: 0,
        world_size: 1,
        start_layer: 0,
        end_layer: modelCard.nLayers,
        n_layers: modelCard.nLayers,
      };

      // Start download on the first available node (or maybe let user pick?)
      // For now, let's pick the first node in the list
      const nodeIds = Object.keys(data?.nodes || {});
      if (nodeIds.length > 0) {
        await startDownload(nodeIds[0], shardMetadata);
        customModelId = ""; // Clear input on success
        refreshState();
      } else {
        registerError = "No nodes available to download to.";
      }
    } catch (err: any) {
      registerError = err.message || "Failed to register/download model";
    } finally {
      isRegistering = false;
    }
  }
</script>

<div
  class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-3 flex flex-col"
>
  <div>
    <h2 class="text-lg font-mono uppercase text-exo-yellow">Custom Models</h2>
    <p class="text-xs text-exo-light-gray">
      Download custom models from HuggingFace (e.g.
      mlx-community/Qwen2.5-0.5B-Instruct-4bit)
    </p>
  </div>
  <div class="flex gap-2">
    <input
      type="text"
      bind:value={customModelId}
      placeholder="Enter HuggingFace Model ID"
      class="flex-1 bg-exo-dark-gray border border-exo-medium-gray/50 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-exo-yellow font-mono placeholder:text-exo-light-gray/50"
      onkeydown={(e) => e.key === "Enter" && handleDownloadCustom()}
      disabled={isRegistering}
    />
    <button
      class="bg-exo-yellow text-exo-black px-4 py-2 rounded text-sm font-bold font-mono uppercase hover:bg-exo-yellow/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      onclick={handleDownloadCustom}
      disabled={isRegistering || !customModelId.trim()}
    >
      {isRegistering ? "Processing..." : "Download Model"}
    </button>
  </div>
  {#if registerError}
    <div class="text-red-400 text-xs font-mono">{registerError}</div>
  {/if}
</div>

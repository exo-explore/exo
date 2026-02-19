<script lang="ts">
  import type {
    MetaInstance,
    MetaInstanceStatus,
    NodeInfo,
  } from "$lib/stores/app.svelte";
  import {
    getMetaInstanceStatus,
    getMetaInstanceBackingNodes,
    topologyData,
  } from "$lib/stores/app.svelte";

  interface Props {
    metaInstance: MetaInstance;
    onDelete?: (metaInstanceId: string) => void;
  }

  let { metaInstance, onDelete }: Props = $props();

  const status: MetaInstanceStatus = $derived(
    getMetaInstanceStatus(metaInstance),
  );
  const backingNodeIds: string[] = $derived(
    getMetaInstanceBackingNodes(metaInstance),
  );

  const statusConfig = $derived.by(() => {
    switch (status) {
      case "active":
        return {
          label: "ACTIVE",
          dotClass: "bg-green-400",
          borderClass:
            "border-green-500/30 border-l-green-400",
          cornerClass: "border-green-500/50",
          glowClass: "shadow-[0_0_6px_rgba(74,222,128,0.4)]",
          animate: false,
        };
      case "provisioning":
        return {
          label: "PROVISIONING",
          dotClass: "bg-yellow-400",
          borderClass:
            "border-exo-yellow/30 border-l-yellow-400",
          cornerClass: "border-yellow-500/50",
          glowClass: "shadow-[0_0_6px_rgba(250,204,21,0.4)]",
          animate: true,
        };
      case "error":
        return {
          label: "ERROR",
          dotClass: "bg-red-400",
          borderClass: "border-red-500/30 border-l-red-400",
          cornerClass: "border-red-500/50",
          glowClass: "shadow-[0_0_6px_rgba(248,113,113,0.4)]",
          animate: false,
        };
    }
  });

  function getNodeName(nodeId: string): string {
    const topo = topologyData();
    if (!topo?.nodes) return nodeId.slice(0, 8);
    const node = topo.nodes[nodeId];
    return node?.friendly_name || node?.system_info?.model_id || nodeId.slice(0, 8);
  }

  function formatModelId(modelId: string): string {
    // Show just the model name part after the org prefix
    const parts = modelId.split("/");
    return parts.length > 1 ? parts[parts.length - 1] : modelId;
  }

  function handleDelete() {
    if (
      onDelete &&
      confirm(
        `Delete meta-instance for ${formatModelId(metaInstance.modelId)}?`,
      )
    ) {
      onDelete(metaInstance.metaInstanceId);
    }
  }
</script>

<div class="relative group">
  <!-- Corner accents -->
  <div
    class="absolute -top-px -left-px w-2 h-2 border-l border-t {statusConfig.cornerClass}"
  ></div>
  <div
    class="absolute -top-px -right-px w-2 h-2 border-r border-t {statusConfig.cornerClass}"
  ></div>
  <div
    class="absolute -bottom-px -left-px w-2 h-2 border-l border-b {statusConfig.cornerClass}"
  ></div>
  <div
    class="absolute -bottom-px -right-px w-2 h-2 border-r border-b {statusConfig.cornerClass}"
  ></div>

  <div
    class="bg-exo-dark-gray/60 border border-l-2 {statusConfig.borderClass} p-3"
  >
    <!-- Header: Status + Delete -->
    <div class="flex justify-between items-start mb-2 pl-2">
      <div class="flex items-center gap-2">
        <div
          class="w-1.5 h-1.5 {statusConfig.dotClass} rounded-full {statusConfig.glowClass} {statusConfig.animate
            ? 'animate-pulse'
            : ''}"
        ></div>
        <span
          class="text-xs font-mono tracking-[0.15em] uppercase {status === 'active'
            ? 'text-green-400'
            : status === 'error'
              ? 'text-red-400'
              : 'text-yellow-400'}"
        >
          {statusConfig.label}
        </span>
      </div>
      <button
        onclick={handleDelete}
        class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
      >
        DELETE
      </button>
    </div>

    <!-- Model Info -->
    <div class="pl-2 space-y-1">
      <div class="text-exo-yellow text-xs font-mono tracking-wide truncate">
        {metaInstance.modelId}
      </div>

      <!-- Sharding + Runtime badges -->
      <div class="flex items-center gap-2">
        <span
          class="inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono tracking-wider uppercase border border-white/10 text-white/50"
        >
          {metaInstance.sharding}
        </span>
        <span
          class="inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono tracking-wider uppercase border border-white/10 text-white/50"
        >
          {metaInstance.instanceMeta}
        </span>
        {#if metaInstance.minNodes > 1}
          <span
            class="inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono tracking-wider uppercase border border-white/10 text-white/50"
          >
            {metaInstance.minNodes}+ nodes
          </span>
        {/if}
      </div>

      <!-- Node Assignments (when active) -->
      {#if backingNodeIds.length > 0}
        <div class="flex items-center gap-1.5 mt-1">
          <svg
            class="w-3 h-3 text-green-400/70 flex-shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              d="M22 12h-4l-3 9L9 3l-3 9H2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
          <span class="text-white/60 text-xs font-mono truncate">
            {backingNodeIds.map((id) => getNodeName(id)).join(", ")}
          </span>
        </div>
      {/if}

      <!-- Pinned nodes constraint -->
      {#if metaInstance.nodeIds && metaInstance.nodeIds.length > 0}
        <div class="flex items-center gap-1.5">
          <svg
            class="w-3 h-3 text-white/40 flex-shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
            <path d="M7 11V7a5 5 0 0 1 10 0v4" />
          </svg>
          <span class="text-white/40 text-[11px] font-mono">
            Pinned: {metaInstance.nodeIds
              .map((id) => getNodeName(id))
              .join(", ")}
          </span>
        </div>
      {/if}

      <!-- Error details -->
      {#if metaInstance.placementError}
        <div
          class="mt-1.5 p-2 bg-red-500/5 border border-red-500/15 rounded-sm"
        >
          <div class="text-red-400 text-[11px] font-mono leading-relaxed">
            {metaInstance.placementError}
          </div>
        </div>
      {/if}

      <!-- Retry counter -->
      {#if metaInstance.consecutiveFailures > 0}
        <div class="flex items-center gap-1.5 mt-1">
          <svg
            class="w-3 h-3 text-yellow-500/60 flex-shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          <span class="text-yellow-500/60 text-[11px] font-mono">
            {metaInstance.consecutiveFailures} consecutive
            failure{metaInstance.consecutiveFailures !== 1 ? "s" : ""}
          </span>
        </div>
      {/if}
    </div>
  </div>
</div>

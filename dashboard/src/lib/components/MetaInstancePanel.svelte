<script lang="ts">
  import type { MetaInstance } from "$lib/stores/app.svelte";

  interface Props {
    metaInstances: Record<string, MetaInstance>;
    instances: Record<string, unknown>;
    onDelete?: (metaInstanceId: string) => void;
    onHoverNodes?: (nodeIds: Set<string>) => void;
    onHoverEnd?: () => void;
  }

  let { metaInstances, instances, onDelete, onHoverNodes, onHoverEnd }: Props =
    $props();

  function getTagged(obj: unknown): [string | null, unknown] {
    if (!obj || typeof obj !== "object") return [null, null];
    const keys = Object.keys(obj as Record<string, unknown>);
    if (keys.length === 1) {
      return [keys[0], (obj as Record<string, unknown>)[keys[0]]];
    }
    return [null, null];
  }

  interface LinkedInstance {
    instanceId: string;
    modelId: string;
    nodeIds: string[];
  }

  function findLinkedInstance(metaInstanceId: string): LinkedInstance | null {
    for (const [instanceId, instanceWrapped] of Object.entries(instances)) {
      const [, instance] = getTagged(instanceWrapped);
      if (!instance || typeof instance !== "object") continue;
      const inst = instance as {
        metaInstanceId?: string;
        shardAssignments?: {
          modelId?: string;
          nodeToRunner?: Record<string, string>;
        };
      };
      if (inst.metaInstanceId === metaInstanceId) {
        return {
          instanceId,
          modelId: inst.shardAssignments?.modelId || "Unknown",
          nodeIds: Object.keys(inst.shardAssignments?.nodeToRunner ?? {}),
        };
      }
    }
    return null;
  }

  type MetaStatus = "active" | "provisioning" | "error" | "retrying";

  function getStatus(
    meta: MetaInstance,
    linked: LinkedInstance | null,
  ): MetaStatus {
    if (meta.placementError || meta.lastFailureError) {
      if (meta.consecutiveFailures > 0 && meta.consecutiveFailures < 3)
        return "retrying";
      return "error";
    }
    if (linked) return "active";
    return "provisioning";
  }

  function statusLabel(status: MetaStatus): string {
    switch (status) {
      case "active":
        return "ACTIVE";
      case "provisioning":
        return "PROVISIONING";
      case "error":
        return "ERROR";
      case "retrying":
        return "RETRYING";
    }
  }

  function statusDotClass(status: MetaStatus): string {
    switch (status) {
      case "active":
        return "bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.6)]";
      case "provisioning":
        return "bg-yellow-400 animate-pulse shadow-[0_0_6px_rgba(250,204,21,0.6)]";
      case "error":
        return "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.6)]";
      case "retrying":
        return "bg-orange-400 animate-pulse shadow-[0_0_6px_rgba(251,146,60,0.6)]";
    }
  }

  function statusTextClass(status: MetaStatus): string {
    switch (status) {
      case "active":
        return "text-green-400";
      case "provisioning":
        return "text-yellow-400";
      case "error":
        return "text-red-400";
      case "retrying":
        return "text-orange-400";
    }
  }

  function borderClass(status: MetaStatus): string {
    switch (status) {
      case "active":
        return "border-green-500/30 border-l-green-400";
      case "provisioning":
        return "border-purple-500/30 border-l-purple-400";
      case "error":
        return "border-red-500/30 border-l-red-400";
      case "retrying":
        return "border-orange-500/30 border-l-orange-400";
    }
  }

  function cornerClass(status: MetaStatus): string {
    switch (status) {
      case "active":
        return "border-green-500/50";
      case "provisioning":
        return "border-purple-500/50";
      case "error":
        return "border-red-500/50";
      case "retrying":
        return "border-orange-500/50";
    }
  }

  function handleHover(meta: MetaInstance, linked: LinkedInstance | null) {
    if (!onHoverNodes) return;
    if (linked && linked.nodeIds.length > 0) {
      onHoverNodes(new Set(linked.nodeIds));
    } else if (meta.nodeIds && meta.nodeIds.length > 0) {
      onHoverNodes(new Set(meta.nodeIds));
    }
  }

  function formatModelId(modelId: string): string {
    return modelId.split("/").pop() || modelId;
  }
</script>

<!-- Panel Header -->
<div class="flex items-center gap-2 mb-4">
  <div
    class="w-2 h-2 bg-purple-400 rounded-full shadow-[0_0_8px_rgba(168,85,247,0.6)] animate-pulse"
  ></div>
  <h3 class="text-xs text-purple-400 font-mono tracking-[0.2em] uppercase">
    Meta-Instances
  </h3>
  <div
    class="flex-1 h-px bg-gradient-to-r from-purple-400/30 to-transparent"
  ></div>
</div>

<div
  class="space-y-3 max-h-72 xl:max-h-96 overflow-y-auto overflow-x-hidden py-px"
>
  {#each Object.entries(metaInstances) as [id, meta]}
    {@const linked = findLinkedInstance(meta.metaInstanceId)}
    {@const status = getStatus(meta, linked)}
    {@const corners = cornerClass(status)}
    <div
      class="relative group cursor-default"
      role="group"
      onmouseenter={() => handleHover(meta, linked)}
      onmouseleave={() => onHoverEnd?.()}
    >
      <!-- Corner accents -->
      <div
        class="absolute -top-px -left-px w-2 h-2 border-l border-t {corners}"
      ></div>
      <div
        class="absolute -top-px -right-px w-2 h-2 border-r border-t {corners}"
      ></div>
      <div
        class="absolute -bottom-px -left-px w-2 h-2 border-l border-b {corners}"
      ></div>
      <div
        class="absolute -bottom-px -right-px w-2 h-2 border-r border-b {corners}"
      ></div>

      <div
        class="bg-exo-dark-gray/60 border border-l-2 {borderClass(status)} p-3"
      >
        <div class="flex justify-between items-start mb-2 pl-2">
          <div class="flex items-center gap-2">
            <div
              class="w-1.5 h-1.5 {statusDotClass(status)} rounded-full"
            ></div>
            <span class="text-exo-light-gray font-mono text-sm tracking-wider">
              {meta.metaInstanceId.slice(0, 8).toUpperCase()}
            </span>
            <span
              class="{statusTextClass(
                status,
              )} text-[10px] font-mono tracking-wider"
            >
              {statusLabel(status)}
            </span>
          </div>
          <button
            onclick={() => onDelete?.(meta.metaInstanceId)}
            class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
          >
            DELETE
          </button>
        </div>
        <div class="pl-2">
          <div class="text-exo-yellow text-xs font-mono tracking-wide truncate">
            {formatModelId(meta.modelId)}
          </div>
          <div class="text-white/60 text-xs font-mono">
            {meta.sharding} &middot; {meta.instanceMeta} &middot; min {meta.minNodes}
            node{meta.minNodes !== 1 ? "s" : ""}
          </div>
          {#if meta.nodeIds && meta.nodeIds.length > 0}
            <div class="text-white/50 text-[10px] font-mono mt-0.5">
              Pinned: {meta.nodeIds.map((n) => n.slice(0, 8)).join(", ")}
            </div>
          {/if}
          {#if meta.placementError}
            <div
              class="text-red-400/80 text-[10px] font-mono mt-1 truncate"
              title={meta.placementError}
            >
              {meta.placementError}
            </div>
          {/if}
          {#if meta.lastFailureError}
            <div
              class="text-orange-400/80 text-[10px] font-mono mt-0.5 truncate"
              title={meta.lastFailureError}
            >
              Failure: {meta.lastFailureError}
            </div>
          {/if}
          {#if meta.consecutiveFailures > 0}
            <div class="text-orange-400/60 text-[10px] font-mono mt-0.5">
              Retries: {meta.consecutiveFailures}/3
            </div>
          {/if}
          {#if linked}
            <div class="text-purple-400/60 text-[10px] font-mono mt-1">
              Instance: {linked.instanceId.slice(0, 8)} &middot; {linked.nodeIds
                .length} node{linked.nodeIds.length !== 1 ? "s" : ""}
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/each}
</div>

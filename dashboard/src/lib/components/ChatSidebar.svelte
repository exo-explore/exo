<script lang="ts">
  import {
    conversations,
    activeConversationId,
    createConversation,
    loadConversation,
    deleteConversation,
    deleteAllConversations,
    renameConversation,
    clearChat,
    instances,
    debugMode,
    toggleDebugMode,
    topologyOnlyMode,
    toggleTopologyOnlyMode,
  } from "$lib/stores/app.svelte";

  interface Props {
    class?: string;
  }

  let { class: className = "" }: Props = $props();

  const conversationList = $derived(conversations());
  const activeId = $derived(activeConversationId());
  const instanceData = $derived(instances());
  const debugEnabled = $derived(debugMode());
  const topologyOnlyEnabled = $derived(topologyOnlyMode());

  let searchQuery = $state("");
  let editingId = $state<string | null>(null);
  let editingName = $state("");
  let deleteConfirmId = $state<string | null>(null);
  let showDeleteAllConfirm = $state(false);

  const filteredConversations = $derived(
    searchQuery.trim()
      ? conversationList.filter((c) =>
          c.name.toLowerCase().includes(searchQuery.toLowerCase()),
        )
      : conversationList,
  );

  function handleNewChat() {
    createConversation();
  }

  function handleSelectConversation(id: string) {
    loadConversation(id);
  }

  function handleStartEdit(id: string, name: string, event: MouseEvent) {
    event.stopPropagation();
    editingId = id;
    editingName = name;
  }

  function handleSaveEdit() {
    if (editingId && editingName.trim()) {
      renameConversation(editingId, editingName.trim());
    }
    editingId = null;
    editingName = "";
  }

  function handleCancelEdit() {
    editingId = null;
    editingName = "";
  }

  function handleEditKeydown(event: KeyboardEvent) {
    if (event.key === "Enter") {
      handleSaveEdit();
    } else if (event.key === "Escape") {
      handleCancelEdit();
    }
  }

  function handleDeleteClick(id: string, event: MouseEvent) {
    event.stopPropagation();
    deleteConfirmId = id;
  }

  function handleConfirmDelete() {
    if (deleteConfirmId) {
      deleteConversation(deleteConfirmId);
      deleteConfirmId = null;
    }
  }

  function handleCancelDelete() {
    deleteConfirmId = null;
  }

  function handleDeleteAllClick() {
    showDeleteAllConfirm = true;
  }

  function handleConfirmDeleteAll() {
    deleteAllConversations();
    showDeleteAllConfirm = false;
  }

  function handleCancelDeleteAll() {
    showDeleteAllConfirm = false;
  }

  function formatDate(timestamp: number): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diffDays = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24),
    );

    if (diffDays === 0) {
      return date.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      });
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return date.toLocaleDateString("en-US", { weekday: "short" });
    } else {
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      });
    }
  }

  function getLastAssistantStats(
    conversation: (typeof conversationList)[0],
  ): { ttftMs?: number; tps?: number } | null {
    // Find the last assistant message with stats
    for (let i = conversation.messages.length - 1; i >= 0; i--) {
      const msg = conversation.messages[i];
      if (msg.role === "assistant" && (msg.ttftMs || msg.tps)) {
        return { ttftMs: msg.ttftMs, tps: msg.tps };
      }
    }
    return null;
  }

  function formatModelName(modelId: string | null | undefined): string {
    if (!modelId) return "Unknown Model";
    const parts = modelId.split("/");
    const tail = parts[parts.length - 1] || modelId;
    return tail || modelId;
  }

  function formatStrategy(
    sharding: string | null | undefined,
    instanceType: string | null | undefined,
  ): string {
    const shardLabel = sharding ?? "Unknown";
    const typeLabel = instanceType ?? null;
    return typeLabel ? `${shardLabel} (${typeLabel})` : shardLabel;
  }

  function getTaggedValue(obj: unknown): [string | null, unknown] {
    if (!obj || typeof obj !== "object") return [null, null];
    const keys = Object.keys(obj as Record<string, unknown>);
    if (keys.length === 1) {
      return [keys[0], (obj as Record<string, unknown>)[keys[0]]];
    }
    return [null, null];
  }

  function extractInstanceModelId(instanceWrapped: unknown): string | null {
    const [, instance] = getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") return null;
    const inst = instance as { shardAssignments?: { modelId?: string } };
    return inst.shardAssignments?.modelId ?? null;
  }

  function describeInstance(instanceWrapped: unknown): {
    sharding: string | null;
    instanceType: string | null;
  } {
    const [instanceTag, instance] = getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return { sharding: null, instanceType: null };
    }

    let instanceType: string | null = null;
    if (instanceTag === "MlxRingInstance") instanceType = "MLX Ring";
    else if (
      instanceTag === "MlxIbvInstance" ||
      instanceTag === "MlxJacclInstance"
    )
      instanceType = "MLX RDMA";

    let sharding: string | null = null;
    const inst = instance as {
      shardAssignments?: { runnerToShard?: Record<string, unknown> };
    };
    const runnerToShard = inst.shardAssignments?.runnerToShard || {};
    const firstShardWrapped = Object.values(runnerToShard)[0];
    if (firstShardWrapped) {
      const [shardTag] = getTaggedValue(firstShardWrapped);
      if (shardTag === "PipelineShardMetadata") sharding = "Pipeline";
      else if (shardTag === "TensorShardMetadata") sharding = "Tensor";
      else if (shardTag === "PrefillDecodeShardMetadata")
        sharding = "Prefill/Decode";
    }

    return { sharding, instanceType };
  }

  function resolveConversationInfo(
    conversation: (typeof conversationList)[0],
  ): { modelLabel: string; strategyLabel: string } {
    // Attempt to match conversation model to an instance
    let matchedInstance: unknown = null;
    let modelId = conversation.modelId ?? null;

    if (modelId) {
      for (const [, instanceWrapper] of Object.entries(instanceData)) {
        const candidate = extractInstanceModelId(instanceWrapper);
        if (candidate === modelId) {
          matchedInstance = instanceWrapper;
          break;
        }
      }
    }

    // Fallback: use the first available instance if no explicit match
    if (!matchedInstance) {
      const firstInstance = Object.values(instanceData)[0];
      if (firstInstance) {
        matchedInstance = firstInstance;
        modelId = modelId ?? extractInstanceModelId(firstInstance);
      }
    }

    const instanceDetails = matchedInstance
      ? describeInstance(matchedInstance)
      : { sharding: null, instanceType: null };
    const displayModel = modelId ?? conversation.modelId ?? null;
    const sharding =
      conversation.sharding ?? instanceDetails.sharding ?? "Unknown";
    const instanceType =
      conversation.instanceType ?? instanceDetails.instanceType;

    return {
      modelLabel: formatModelName(displayModel),
      strategyLabel: formatStrategy(sharding, instanceType),
    };
  }
</script>

<aside
  class="flex flex-col h-full bg-exo-dark-gray border-r border-exo-yellow/10 {className}"
>
  <!-- Header -->
  <div class="p-4">
    <button
      onclick={handleNewChat}
      class="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-transparent border border-exo-yellow/30 text-exo-yellow text-xs font-mono tracking-wider uppercase hover:border-exo-yellow/50 transition-all cursor-pointer"
    >
      <svg
        class="w-4 h-4"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M12 4v16m8-8H4"
        />
      </svg>
      NEW CHAT
    </button>
  </div>

  <!-- Search -->
  <div class="px-4 py-3">
    <div class="relative">
      <svg
        class="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-white/50"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search conversations..."
        class="w-full bg-exo-black/40 border border-exo-medium-gray/30 rounded px-3 py-2 pl-9 text-xs text-white/90 placeholder:text-white/40 focus:outline-none focus:border-exo-yellow/30"
      />
    </div>
  </div>

  <!-- Conversation List -->
  <div class="flex-1 overflow-y-auto">
    {#if filteredConversations.length > 0}
      <div class="py-2">
        <div class="px-4 py-2">
          <span
            class="text-sm text-white/70 font-mono tracking-wider uppercase"
          >
            {searchQuery ? "SEARCH RESULTS" : "CONVERSATIONS"}
          </span>
        </div>

        {#each filteredConversations as conversation (conversation.id)}
          {@const info = resolveConversationInfo(conversation)}
          <div class="px-2">
            {#if editingId === conversation.id}
              <!-- Edit mode -->
              <div
                class="p-2 bg-transparent border border-exo-yellow/20 rounded mb-1"
              >
                <input
                  type="text"
                  bind:value={editingName}
                  onkeydown={handleEditKeydown}
                  class="w-full bg-exo-black/60 border border-exo-yellow/30 rounded px-2 py-1.5 text-xs text-exo-light-gray focus:outline-none focus:border-exo-yellow/50 mb-2"
                  autofocus
                />
                <div class="flex gap-2">
                  <button
                    onclick={handleSaveEdit}
                    class="flex-1 py-1.5 text-xs font-mono tracking-wider uppercase bg-transparent text-exo-yellow border border-exo-yellow/30 rounded hover:border-exo-yellow/50 cursor-pointer"
                  >
                    SAVE
                  </button>
                  <button
                    onclick={handleCancelEdit}
                    class="flex-1 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/20 text-exo-light-gray border border-exo-medium-gray/30 rounded hover:bg-exo-medium-gray/30 cursor-pointer"
                  >
                    CANCEL
                  </button>
                </div>
              </div>
            {:else if deleteConfirmId === conversation.id}
              <!-- Delete confirmation -->
              <div
                class="p-2 bg-red-500/10 border border-red-500/30 rounded mb-1"
              >
                <p class="text-xs text-red-400 mb-2">
                  Delete "{conversation.name}"?
                </p>
                <div class="flex gap-2">
                  <button
                    onclick={handleConfirmDelete}
                    class="flex-1 py-1.5 text-xs font-mono tracking-wider uppercase bg-red-500/20 text-red-400 border border-red-500/30 rounded hover:bg-red-500/30 cursor-pointer"
                  >
                    DELETE
                  </button>
                  <button
                    onclick={handleCancelDelete}
                    class="flex-1 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/20 text-exo-light-gray border border-exo-medium-gray/30 rounded hover:bg-exo-medium-gray/30 cursor-pointer"
                  >
                    CANCEL
                  </button>
                </div>
              </div>
            {:else}
              <!-- Normal view -->
              {@const stats = getLastAssistantStats(conversation)}
              <div
                role="button"
                tabindex="0"
                onclick={() => handleSelectConversation(conversation.id)}
                onkeydown={(e) =>
                  e.key === "Enter" &&
                  handleSelectConversation(conversation.id)}
                class="group w-full flex items-center justify-between p-2 rounded mb-1 transition-all text-left cursor-pointer
									{activeId === conversation.id
                  ? 'bg-transparent border border-exo-yellow/30'
                  : 'hover:border-exo-yellow/20 border border-transparent'}"
              >
                <div class="flex-1 min-w-0 pr-2">
                  <div
                    class="text-sm truncate {activeId === conversation.id
                      ? 'text-exo-yellow'
                      : 'text-white/90'}"
                  >
                    {conversation.name}
                  </div>
                  <div class="text-sm text-white/50 mt-0.5">
                    {formatDate(conversation.updatedAt)}
                  </div>
                  <div class="text-sm text-white/70 truncate">
                    {info.modelLabel}
                  </div>
                  <div class="text-xs text-white/60 font-mono">
                    Strategy: <span class="text-white/80"
                      >{info.strategyLabel}</span
                    >
                  </div>
                  {#if stats}
                    <div class="text-xs text-white/60 font-mono mt-1">
                      {#if stats.ttftMs}<span class="text-white/40">TTFT</span>
                        {stats.ttftMs.toFixed(
                          0,
                        )}ms{/if}{#if stats.ttftMs && stats.tps}<span
                          class="text-white/30 mx-1.5">•</span
                        >{/if}{#if stats.tps}{stats.tps.toFixed(1)}
                        <span class="text-white/40">tok/s</span>{/if}
                    </div>
                  {/if}
                </div>

                <div
                  class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <button
                    type="button"
                    onclick={(e) =>
                      handleStartEdit(conversation.id, conversation.name, e)}
                    class="p-1 text-exo-light-gray hover:text-exo-yellow transition-colors cursor-pointer"
                    title="Rename"
                  >
                    <svg
                      class="w-3 h-3"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                      />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onclick={(e) => handleDeleteClick(conversation.id, e)}
                    class="p-1 text-exo-light-gray hover:text-red-400 transition-colors cursor-pointer"
                    title="Delete"
                  >
                    <svg
                      class="w-3 h-3"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <div
        class="flex flex-col items-center justify-center h-full p-4 text-center"
      >
        <div
          class="w-12 h-12 border border-exo-yellow/20 rounded-full flex items-center justify-center mb-3"
        >
          <svg
            class="w-6 h-6 text-exo-yellow/40"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="1.5"
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
        </div>
        <p
          class="text-xs text-white/70 font-mono tracking-wider uppercase mb-1"
        >
          {searchQuery ? "NO RESULTS" : "NO CONVERSATIONS"}
        </p>
        <p class="text-sm text-white/50">
          {searchQuery ? "Try a different search" : "Start a new chat to begin"}
        </p>
      </div>
    {/if}
  </div>

  <!-- Footer -->
  <div class="p-3 border-t border-exo-yellow/10">
    {#if showDeleteAllConfirm}
      <div class="bg-red-500/10 border border-red-500/30 rounded p-2 mb-2">
        <p class="text-xs text-red-400 text-center mb-2">
          Delete all {conversationList.length} conversations?
        </p>
        <div class="flex gap-2">
          <button
            onclick={handleConfirmDeleteAll}
            class="flex-1 py-1.5 text-xs font-mono tracking-wider uppercase bg-red-500/20 text-red-400 border border-red-500/30 rounded hover:bg-red-500/30 transition-colors cursor-pointer"
          >
            DELETE ALL
          </button>
          <button
            onclick={handleCancelDeleteAll}
            class="flex-1 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/20 text-exo-light-gray border border-exo-medium-gray/30 rounded hover:bg-exo-medium-gray/30 transition-colors cursor-pointer"
          >
            CANCEL
          </button>
        </div>
      </div>
    {:else if conversationList.length > 0}
      <button
        onclick={handleDeleteAllClick}
        class="w-full flex items-center justify-center gap-2 py-1.5 text-sm font-mono tracking-wider uppercase text-white/70 hover:text-red-400 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 rounded transition-all cursor-pointer"
      >
        <svg
          class="w-3.5 h-3.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
          />
        </svg>
        DELETE ALL CHATS
      </button>
    {/if}
    <div
      class="flex items-center justify-center gap-3 {conversationList.length >
        0 && !showDeleteAllConfirm
        ? 'mt-2'
        : ''}"
    >
      <button
        type="button"
        onclick={toggleDebugMode}
        class="p-1.5 rounded border border-exo-medium-gray/40 hover:border-exo-yellow/50 transition-colors cursor-pointer"
        title="Toggle debug mode"
      >
        <svg
          class="w-4 h-4 {debugEnabled
            ? 'text-exo-yellow'
            : 'text-exo-medium-gray'}"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            d="M19 8h-1.81A6.002 6.002 0 0 0 12 2a6.002 6.002 0 0 0-5.19 3H5a1 1 0 0 0 0 2h1v2H5a1 1 0 0 0 0 2h1v2H5a1 1 0 0 0 0 2h1.81A6.002 6.002 0 0 0 12 22a6.002 6.002 0 0 0 5.19-3H19a1 1 0 0 0 0-2h-1v-2h1a1 1 0 0 0 0-2h-1v-2h1a1 1 0 1 0 0-2Zm-5 10.32V19a1 1 0 1 1-2 0v-.68a3.999 3.999 0 0 1-3-3.83V9.32a3.999 3.999 0 0 1 3-3.83V5a1 1 0 0 1 2 0v.49a3.999 3.999 0 0 1 3 3.83v5.17a3.999 3.999 0 0 1-3 3.83Z"
          />
        </svg>
      </button>
      <div class="text-xs text-white/60 font-mono tracking-wider text-center">
        {conversationList.length} CONVERSATION{conversationList.length !== 1
          ? "S"
          : ""}
      </div>
      <button
        type="button"
        onclick={toggleTopologyOnlyMode}
        class="p-1.5 rounded border border-exo-medium-gray/40 hover:border-exo-yellow/50 transition-colors cursor-pointer"
        title="Toggle topology only mode"
      >
        <svg
          class="w-4 h-4 {topologyOnlyEnabled
            ? 'text-exo-yellow'
            : 'text-exo-medium-gray'}"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          <circle cx="12" cy="5" r="2" fill="currentColor" />
          <circle cx="5" cy="19" r="2" fill="currentColor" />
          <circle cx="19" cy="19" r="2" fill="currentColor" />
          <path stroke-linecap="round" d="M12 7v5m0 0l-5 5m5-5l5 5" />
        </svg>
      </button>
    </div>
  </div>
</aside>

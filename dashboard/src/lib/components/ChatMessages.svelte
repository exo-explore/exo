<script lang="ts">
  import {
    messages,
    currentResponse,
    isLoading,
    deleteMessage,
    editAndRegenerate,
    regenerateLastResponse,
    setEditingImage,
  } from "$lib/stores/app.svelte";
  import type { Message } from "$lib/stores/app.svelte";
  import type { MessageAttachment } from "$lib/stores/app.svelte";
  import MarkdownContent from "./MarkdownContent.svelte";

  interface Props {
    class?: string;
    scrollParent?: HTMLElement | null;
  }

  let { class: className = "", scrollParent = null }: Props = $props();

  const messageList = $derived(messages());
  const response = $derived(currentResponse());
  const loading = $derived(isLoading());

  // Scroll management - user controls scroll, show button when not at bottom
  const SCROLL_THRESHOLD = 100;
  let showScrollButton = $state(false);
  let lastMessageCount = 0;
  let containerRef: HTMLDivElement | undefined = $state();

  function getScrollContainer(): HTMLElement | null {
    if (scrollParent) return scrollParent;
    return containerRef?.parentElement ?? null;
  }

  function isNearBottom(el: HTMLElement): boolean {
    return el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD;
  }

  function scrollToBottom() {
    const el = getScrollContainer();
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }

  function updateScrollButtonVisibility() {
    const el = getScrollContainer();
    if (!el) return;
    showScrollButton = !isNearBottom(el);
  }

  // Attach scroll listener
  $effect(() => {
    const el = scrollParent ?? containerRef?.parentElement;
    if (!el) return;

    el.addEventListener("scroll", updateScrollButtonVisibility, {
      passive: true,
    });
    // Initial check
    updateScrollButtonVisibility();
    return () => el.removeEventListener("scroll", updateScrollButtonVisibility);
  });

  // Auto-scroll when user sends a new message
  $effect(() => {
    const count = messageList.length;
    if (count > lastMessageCount) {
      const el = getScrollContainer();
      if (el) {
        requestAnimationFrame(() => {
          el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
        });
      }
    }
    lastMessageCount = count;
  });

  // Update scroll button visibility when content changes
  $effect(() => {
    // Track response to trigger re-check during streaming
    const _ = response;

    // Small delay to let DOM update
    requestAnimationFrame(() => updateScrollButtonVisibility());
  });

  // Edit state
  let editingMessageId = $state<string | null>(null);
  let editContent = $state("");
  let editTextareaRef: HTMLTextAreaElement | undefined = $state();

  // Delete confirmation state
  let deleteConfirmId = $state<string | null>(null);

  // Copied state for feedback
  let copiedMessageId = $state<string | null>(null);
  let expandedThinkingMessageIds = $state<Set<string>>(new Set());

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function getAttachmentIcon(attachment: MessageAttachment): string {
    switch (attachment.type) {
      case "image":
        return "🖼";
      case "text":
        return "📄";
      default:
        return "📎";
    }
  }

  function truncateName(name: string, maxLen: number = 25): string {
    if (name.length <= maxLen) return name;
    const ext = name.slice(name.lastIndexOf("."));
    const base = name.slice(0, name.lastIndexOf("."));
    const available = maxLen - ext.length - 3;
    return base.slice(0, available) + "..." + ext;
  }

  async function handleCopy(content: string, messageId: string) {
    try {
      await navigator.clipboard.writeText(content);
      copiedMessageId = messageId;
      setTimeout(() => {
        copiedMessageId = null;
      }, 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  }

  function toggleThinkingVisibility(messageId: string) {
    const next = new Set(expandedThinkingMessageIds);
    if (next.has(messageId)) {
      next.delete(messageId);
    } else {
      next.add(messageId);
    }
    expandedThinkingMessageIds = next;
  }

  function isThinkingExpanded(messageId: string): boolean {
    return expandedThinkingMessageIds.has(messageId);
  }

  function handleStartEdit(messageId: string, content: string) {
    editingMessageId = messageId;
    editContent = content;
    setTimeout(() => {
      if (editTextareaRef) {
        editTextareaRef.focus();
        editTextareaRef.setSelectionRange(
          editTextareaRef.value.length,
          editTextareaRef.value.length,
        );
        // Auto-resize
        editTextareaRef.style.height = "auto";
        editTextareaRef.style.height =
          Math.min(editTextareaRef.scrollHeight, 200) + "px";
      }
    }, 10);
  }

  function handleCancelEdit() {
    editingMessageId = null;
    editContent = "";
  }

  function handleSaveEdit() {
    if (editingMessageId && editContent.trim()) {
      editAndRegenerate(editingMessageId, editContent.trim());
    }
    editingMessageId = null;
    editContent = "";
  }

  function handleEditKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSaveEdit();
    } else if (event.key === "Escape") {
      handleCancelEdit();
    }
  }

  function handleEditInput() {
    if (editTextareaRef) {
      editTextareaRef.style.height = "auto";
      editTextareaRef.style.height =
        Math.min(editTextareaRef.scrollHeight, 200) + "px";
    }
  }

  function handleDeleteClick(messageId: string) {
    deleteConfirmId = messageId;
  }

  function handleConfirmDelete() {
    if (deleteConfirmId) {
      deleteMessage(deleteConfirmId);
      deleteConfirmId = null;
    }
  }

  function handleCancelDelete() {
    deleteConfirmId = null;
  }

  function handleRegenerate() {
    regenerateLastResponse();
  }

  // Check if a message is the last assistant message
  function isLastAssistantMessage(messageId: string): boolean {
    for (let i = messageList.length - 1; i >= 0; i--) {
      if (messageList[i].role === "assistant") {
        return messageList[i].id === messageId;
      }
    }
    return false;
  }
</script>

<div class="flex flex-col gap-4 sm:gap-6 {className}">
  {#each messageList as message (message.id)}
    <div
      class="group flex {message.role === 'user'
        ? 'justify-end'
        : 'justify-start'}"
    >
      <div
        class={message.role === "user"
          ? "max-w-[85%] sm:max-w-[70%] flex flex-col items-end"
          : "w-full max-w-[98%] sm:max-w-[95%]"}
      >
        {#if message.role === "assistant"}
          <!-- Assistant message header -->
          <div class="flex items-center gap-1.5 sm:gap-2 mb-1.5 sm:mb-2">
            <div
              class="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-exo-yellow rounded-full shadow-[0_0_10px_rgba(255,215,0,0.5)]"
            ></div>
            <span
              class="text-sm sm:text-xs text-exo-yellow tracking-[0.15em] sm:tracking-[0.2em] uppercase font-medium"
              >EXO</span
            >
            <span
              class="text-xs sm:text-sm text-exo-light-gray tracking-wider tabular-nums"
              >{formatTimestamp(message.timestamp)}</span
            >
            {#if message.ttftMs || message.tps}
              <span class="text-xs text-exo-light-gray/80 font-mono ml-2">
                {#if message.ttftMs}<span class="text-exo-light-gray/50"
                    >TTFT</span
                  >
                  {message.ttftMs.toFixed(
                    0,
                  )}ms{/if}{#if message.ttftMs && message.tps}<span
                    class="text-exo-light-gray/30 mx-1">•</span
                  >{/if}{#if message.tps}{message.tps.toFixed(1)}
                  <span class="text-exo-light-gray/50">tok/s</span>{/if}
              </span>
            {/if}
          </div>
        {:else}
          <!-- User message header -->
          <div
            class="flex items-center justify-end gap-1.5 sm:gap-2 mb-1.5 sm:mb-2"
          >
            <span
              class="text-xs sm:text-sm text-exo-light-gray tracking-wider tabular-nums"
              >{formatTimestamp(message.timestamp)}</span
            >
            <span
              class="text-sm sm:text-xs text-exo-light-gray tracking-[0.1em] sm:tracking-[0.15em] uppercase"
              >QUERY</span
            >
            <div
              class="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-exo-light-gray/50 rounded-full"
            ></div>
          </div>
        {/if}

        {#if deleteConfirmId === message.id}
          <!-- Delete confirmation -->
          <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p class="text-xs text-red-400 mb-3">
              Delete this message{message.role === "user"
                ? " and all responses after it"
                : ""}?
            </p>
            <div class="flex gap-2 justify-end">
              <button
                onclick={handleCancelDelete}
                class="px-3 py-1.5 text-sm font-mono tracking-wider uppercase bg-exo-medium-gray/20 text-exo-light-gray border border-exo-medium-gray/30 rounded hover:bg-exo-medium-gray/30 transition-colors cursor-pointer"
              >
                CANCEL
              </button>
              <button
                onclick={handleConfirmDelete}
                class="px-3 py-1.5 text-sm font-mono tracking-wider uppercase bg-red-500/20 text-red-400 border border-red-500/30 rounded hover:bg-red-500/30 transition-colors cursor-pointer"
              >
                DELETE
              </button>
            </div>
          </div>
        {:else if editingMessageId === message.id}
          <!-- Edit mode -->
          <div class="command-panel rounded-lg p-3">
            <textarea
              bind:this={editTextareaRef}
              bind:value={editContent}
              onkeydown={handleEditKeydown}
              oninput={handleEditInput}
              class="w-full bg-exo-black/60 border border-exo-yellow/30 rounded px-3 py-2 text-sm text-foreground font-mono focus:outline-none focus:border-exo-yellow/50 resize-none"
              style="min-height: 60px; max-height: 200px;"
            ></textarea>
            <div class="flex gap-2 justify-end mt-2">
              <button
                onclick={handleCancelEdit}
                class="px-3 py-1.5 text-sm font-mono tracking-wider uppercase bg-exo-medium-gray/20 text-exo-light-gray border border-exo-medium-gray/30 rounded hover:bg-exo-medium-gray/30 transition-colors cursor-pointer"
              >
                CANCEL
              </button>
              <button
                onclick={handleSaveEdit}
                disabled={!editContent.trim()}
                class="px-3 py-1.5 text-sm font-mono tracking-wider uppercase bg-transparent text-exo-yellow border border-exo-yellow/30 rounded hover:border-exo-yellow/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 cursor-pointer"
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
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
                SEND
              </button>
            </div>
          </div>
        {:else}
          <div
            class={message.role === "user"
              ? "command-panel rounded-lg rounded-tr-sm inline-block"
              : "command-panel rounded-lg rounded-tl-sm border-l-2 border-l-exo-yellow/50 block w-full"}
          >
            {#if message.role === "user"}
              <!-- User message styling -->
              <div class="px-4 py-3">
                <!-- Attachments -->
                {#if message.attachments && message.attachments.length > 0}
                  <div class="flex flex-wrap gap-2 mb-3">
                    {#each message.attachments as attachment}
                      <div
                        class="flex items-center gap-2 bg-exo-dark-gray/60 border border-exo-yellow/20 rounded px-2 py-1 text-xs font-mono"
                      >
                        {#if attachment.type === "image" && attachment.preview}
                          <img
                            src={attachment.preview}
                            alt={attachment.name}
                            class="w-12 h-12 object-cover rounded border border-exo-yellow/20"
                          />
                        {:else}
                          <span>{getAttachmentIcon(attachment)}</span>
                        {/if}
                        <span class="text-exo-yellow" title={attachment.name}
                          >{truncateName(attachment.name)}</span
                        >
                      </div>
                    {/each}
                  </div>
                {/if}

                {#if message.content}
                  <div
                    class="text-xs text-foreground font-mono tracking-wide whitespace-pre-wrap break-words leading-relaxed"
                  >
                    {message.content}
                  </div>
                {/if}
              </div>
            {:else}
              <!-- Assistant message styling -->
              <div class="p-3 sm:p-4">
                {#if message.thinking && message.thinking.trim().length > 0}
                  <div
                    class="mb-3 rounded border border-exo-yellow/20 bg-exo-black/40"
                  >
                    <button
                      type="button"
                      class="w-full flex items-center justify-between px-3 py-2 text-xs font-mono uppercase tracking-[0.2em] text-exo-light-gray/80 hover:text-exo-yellow transition-colors cursor-pointer"
                      onclick={() => toggleThinkingVisibility(message.id)}
                      aria-expanded={isThinkingExpanded(message.id)}
                      aria-controls={`thinking-panel-${message.id}`}
                    >
                      <span class="flex items-center gap-2 tracking-[0.25em]">
                        <svg
                          class={`w-3.5 h-3.5 text-current transition-transform duration-200 ${isThinkingExpanded(message.id) ? "rotate-90" : ""}`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M9 5l7 7-7 7"
                          />
                        </svg>
                        <span>Thinking...</span>
                      </span>
                      <span
                        class="text-[10px] tracking-[0.2em] text-exo-light-gray/60 ml-4"
                      >
                        {isThinkingExpanded(message.id) ? "HIDE" : "SHOW"}
                      </span>
                    </button>
                    {#if isThinkingExpanded(message.id)}
                      <div
                        id={`thinking-panel-${message.id}`}
                        class="px-3 pb-3 text-xs text-exo-light-gray/90 font-mono whitespace-pre-wrap break-words leading-relaxed"
                      >
                        {message.thinking.trim()}
                      </div>
                    {/if}
                  </div>
                {/if}

                <!-- Generated Images -->
                {#if message.attachments?.some((a) => a.type === "generated-image")}
                  <div class="mb-3">
                    {#each message.attachments.filter((a) => a.type === "generated-image") as attachment}
                      <div class="relative group/img inline-block">
                        <img
                          src={attachment.preview}
                          alt=""
                          class="max-w-full max-h-[512px] rounded-lg border border-exo-yellow/20 shadow-lg shadow-black/20"
                        />
                        <!-- Button overlay -->
                        <div
                          class="absolute top-2 right-2 flex gap-1 opacity-0 group-hover/img:opacity-100 transition-opacity"
                        >
                          <!-- Edit button -->
                          <button
                            type="button"
                            class="p-2 rounded-lg bg-exo-dark-gray/80 border border-exo-yellow/30 text-exo-yellow hover:bg-exo-dark-gray hover:border-exo-yellow/50 cursor-pointer"
                            onclick={() => {
                              if (attachment.preview) {
                                setEditingImage(attachment.preview, message);
                              }
                            }}
                            title="Edit image"
                          >
                            <svg
                              class="w-4 h-4"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              stroke-width="2"
                            >
                              <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                              />
                            </svg>
                          </button>
                          <!-- Download button -->
                          <button
                            type="button"
                            class="p-2 rounded-lg bg-exo-dark-gray/80 border border-exo-yellow/30 text-exo-yellow hover:bg-exo-dark-gray hover:border-exo-yellow/50 cursor-pointer"
                            onclick={() => {
                              if (attachment.preview) {
                                const link = document.createElement("a");
                                link.href = attachment.preview;
                                const ext =
                                  attachment.name?.split(".").pop() || "png";
                                link.download = `generated-image-${Date.now()}.${ext}`;
                                link.click();
                              }
                            }}
                            title="Download image"
                          >
                            <svg
                              class="w-4 h-4"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              stroke-width="2"
                            >
                              <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                              />
                            </svg>
                          </button>
                        </div>
                      </div>
                    {/each}
                  </div>
                {/if}

                <div class="text-xs text-foreground">
                  {#if message.content === "Generating image..." || message.content === "Editing image..." || message.content?.startsWith("Generating...") || message.content?.startsWith("Editing...")}
                    <div class="flex items-center gap-3 text-exo-yellow">
                      <div class="relative">
                        <div
                          class="w-8 h-8 border-2 border-exo-yellow/30 border-t-exo-yellow rounded-full animate-spin"
                        ></div>
                        <svg
                          class="absolute inset-0 w-8 h-8 p-1.5 text-exo-yellow/60"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          stroke-width="2"
                        >
                          <rect
                            x="3"
                            y="3"
                            width="18"
                            height="18"
                            rx="2"
                            ry="2"
                          />
                          <circle cx="8.5" cy="8.5" r="1.5" />
                          <polyline points="21 15 16 10 5 21" />
                        </svg>
                      </div>
                      <span class="font-mono tracking-wider uppercase text-sm"
                        >{message.content}</span
                      >
                    </div>
                  {:else if message.content || (loading && !message.attachments?.some((a) => a.type === "generated-image"))}
                    <MarkdownContent
                      content={message.content || (loading ? response : "")}
                    />
                    {#if loading && !message.content}
                      <span
                        class="inline-block w-2 h-4 bg-exo-yellow/70 ml-1 cursor-blink"
                      ></span>
                    {/if}
                  {/if}
                </div>
              </div>
            {/if}
          </div>

          <!-- Action buttons -->
          <div
            class="flex items-center gap-1 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity {message.role ===
            'user'
              ? 'justify-end'
              : 'justify-start'}"
          >
            <!-- Copy button -->
            <button
              onclick={() => handleCopy(message.content, message.id)}
              class="p-1.5 text-exo-light-gray hover:text-exo-yellow transition-colors rounded cursor-pointer"
              title="Copy message"
            >
              {#if copiedMessageId === message.id}
                <svg
                  class="w-3.5 h-3.5 text-green-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              {:else}
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
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              {/if}
            </button>

            <!-- Edit button (user messages only) -->
            {#if message.role === "user"}
              <button
                onclick={() => handleStartEdit(message.id, message.content)}
                class="p-1.5 text-exo-light-gray hover:text-exo-yellow transition-colors rounded cursor-pointer"
                title="Edit message"
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
                    d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                  />
                </svg>
              </button>
            {/if}

            <!-- Regenerate button (last assistant message only) -->
            {#if message.role === "assistant" && isLastAssistantMessage(message.id) && !loading}
              <button
                onclick={handleRegenerate}
                class="p-1.5 text-exo-light-gray hover:text-exo-yellow transition-colors rounded cursor-pointer"
                title="Regenerate response"
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
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
              </button>
            {/if}

            <!-- Delete button -->
            <button
              onclick={() => handleDeleteClick(message.id)}
              class="p-1.5 text-exo-light-gray hover:text-red-400 transition-colors rounded hover:bg-red-500/10 cursor-pointer"
              title="Delete message"
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
            </button>
          </div>
        {/if}
      </div>
    </div>
  {/each}

  {#if messageList.length === 0}
    <div
      class="flex-1 flex flex-col items-center justify-center text-center pt-[20vh]"
    >
      <div
        class="w-12 h-12 sm:w-16 sm:h-16 border border-exo-yellow/20 rounded-full flex items-center justify-center mb-3 sm:mb-4"
      >
        <div
          class="w-6 h-6 sm:w-8 sm:h-8 border border-exo-yellow/40 rounded-full flex items-center justify-center"
        >
          <div
            class="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-exo-yellow/60 rounded-full"
          ></div>
        </div>
      </div>
      <p
        class="text-xs sm:text-sm text-exo-light-gray tracking-[0.15em] sm:tracking-[0.2em] uppercase"
      >
        AWAITING INPUT
      </p>
      <p class="text-sm sm:text-xs text-exo-light-gray tracking-wider mt-1">
        ENTER A QUERY TO BEGIN
      </p>
    </div>
  {/if}

  <!-- Invisible element for container reference -->
  <div bind:this={containerRef}></div>

  <!-- Scroll to bottom button -->
  {#if showScrollButton}
    <button
      type="button"
      onclick={scrollToBottom}
      class="sticky bottom-4 left-1/2 -translate-x-1/2 w-10 h-10 rounded-full bg-exo-dark-gray/90 border border-exo-medium-gray/50 flex items-center justify-center text-exo-light-gray hover:text-exo-yellow hover:border-exo-yellow/50 transition-all shadow-lg cursor-pointer z-10"
      title="Scroll to bottom"
    >
      <svg
        class="w-5 h-5"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M19 14l-7 7m0 0l-7-7m7 7V3"
        />
      </svg>
    </button>
  {/if}
</div>

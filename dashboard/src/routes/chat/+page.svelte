<script lang="ts">
  import ChatMessages from "$lib/components/ChatMessages.svelte";
  import ChatForm from "$lib/components/ChatForm.svelte";
  import {
    sendMessage,
    messages,
    clearChat,
    selectedChatModel,
    setSelectedChatModel,
    thinkingEnabled,
    instances,
  } from "$lib/stores/app.svelte";

  let scrollParent = $state<HTMLDivElement | null>(null);
  let pickerOpen = $state(false);

  let messageList = $derived(messages());
  let selectedModel = $derived(selectedChatModel());
  let isEmpty = $derived(messageList.length === 0);
  let runningInstances = $derived(instances());
  let runningModelIds = $derived.by(() => {
    const ids = new Set<string>();
    // Each instance arrives wrapped in a tagged-union envelope:
    // { MlxRingInstance: { shardAssignments: {...} } } (or MlxJacclInstance, etc.).
    for (const env of Object.values(runningInstances ?? {})) {
      if (!env || typeof env !== "object") continue;
      const keys = Object.keys(env);
      if (keys.length !== 1) continue;
      const inner = (env as Record<string, unknown>)[keys[0]!] as
        | { shardAssignments?: { modelId?: string } }
        | undefined;
      const id = inner?.shardAssignments?.modelId;
      if (id) ids.add(id);
    }
    return [...ids];
  });
  // Display model: explicit selection wins; otherwise, if exactly one model is
  // running, use that (matches the API's fallback-to-first-running behavior).
  let model = $derived(
    selectedModel ||
      (runningModelIds.length === 1 ? runningModelIds[0]! : null),
  );
  let hasModelLoaded = $derived(runningModelIds.length > 0 || !!selectedModel);

  function shortName(id: string): string {
    return id.split("/").pop() ?? id;
  }

  function pick(id: string) {
    setSelectedChatModel(id);
    pickerOpen = false;
  }

  function togglePicker() {
    pickerOpen = !pickerOpen;
  }

  function closePicker() {
    pickerOpen = false;
  }

  const suggestions = [
    "Explain this codebase from a high level.",
    "Write a Python script that downloads a file with progress.",
    "Plan a 3-day trip to Tokyo for a first-time visitor.",
    "Summarize the latest research on Apple Silicon performance.",
  ];

  function handleSend(
    content: string,
    files?: {
      id: string;
      name: string;
      type: string;
      textContent?: string;
      preview?: string;
    }[],
  ) {
    sendMessage(content, files, thinkingEnabled());
  }

  function fillSuggestion(text: string) {
    sendMessage(text, undefined, thinkingEnabled());
  }
</script>

<div class="chat-fullbleed">
  <header class="chat-header" class:minimal={isEmpty}>
    <div class="header-left">
      <div class="eyebrow">CHAT</div>
      <div class="model-picker">
        <button
          type="button"
          class="model-button"
          onclick={togglePicker}
          aria-haspopup="listbox"
          aria-expanded={pickerOpen}
        >
          <span class="model-label">
            {model ? shortName(model) : "Select model"}
          </span>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
        {#if pickerOpen}
          <button
            class="picker-backdrop"
            type="button"
            onclick={closePicker}
            aria-label="Close model picker"
          ></button>
          <div class="picker-menu" role="listbox">
            {#if runningModelIds.length === 0}
              <div class="picker-empty">
                No models running yet.
                <a href="#/models" onclick={closePicker}>Open Models →</a>
              </div>
            {:else}
              {#each runningModelIds as id}
                <button
                  type="button"
                  class="picker-item"
                  class:active={id === model}
                  role="option"
                  aria-selected={id === model}
                  onclick={() => pick(id)}
                >
                  <span class="picker-dot"></span>
                  <span class="picker-name">{shortName(id)}</span>
                  <span class="picker-id">{id}</span>
                </button>
              {/each}
            {/if}
            <a href="#/models" class="picker-footer" onclick={closePicker}>
              Manage models →
            </a>
          </div>
        {/if}
      </div>
    </div>
    <div class="actions">
      {#if !isEmpty}
        <button class="btn ghost" onclick={clearChat}>+ New chat</button>
      {/if}
    </div>
  </header>

  <div class="chat-pane" bind:this={scrollParent}>
    {#if isEmpty}
      <div class="empty-wrap">
        <div class="empty">
          <div class="empty-eyebrow">CHAT</div>
          <h1 class="empty-title">Ready when you are.</h1>
          <p class="empty-sub">
            {#if hasModelLoaded}
              Talking to <span class="empty-model">{model || "an active model"}</span>.
              Streaming, prefix-cache hits, and per-request timings all land here automatically.
            {:else}
              No model is loaded yet — <a href="#/models">open Models</a> to launch one,
              or pick one below and start chatting.
            {/if}
          </p>
          <div class="suggestions">
            {#each suggestions as s}
              <button class="chip" onclick={() => fillSuggestion(s)}>
                <span class="chip-arrow">→</span>
                <span class="chip-text">{s}</span>
              </button>
            {/each}
          </div>
        </div>
      </div>
    {:else}
      <ChatMessages {scrollParent} />
    {/if}
  </div>

  <div class="chat-form-wrap" class:floating={isEmpty}>
    <ChatForm
      onAutoSend={handleSend}
      placeholder="Send a message…"
      autofocus={true}
      showHelperText={isEmpty}
      showModelSelector={false}
    />
  </div>
</div>

<style>
  .chat-fullbleed {
    margin: -48px -28px -96px;
    padding: 0;
    height: calc(100vh - 80px);
    display: grid;
    grid-template-rows: auto 1fr auto;
    background: var(--ux-bg);
    /* Re-bind legacy tokens used by ChatForm/ChatMessages so they adapt
       to the active theme instead of being permanently dark. The chat
       components use bg-exo-*, text-exo-*, border-exo-* utilities which
       resolve to these vars; overriding them here remaps the whole tree. */
    --exo-black: var(--ux-bg);
    --exo-dark-gray: var(--ux-card);
    --exo-medium-gray: var(--ux-bg-raised);
    --exo-light-gray: var(--ux-text-dim);
    /* oMLX-style restraint: the legacy "yellow" was used for the ▶ prompt
       symbol, the form's top accent line, the SEND button hover, and a few
       other functional bits. Remap to the regular text color so all of
       those become refined dark-on-light (or light-on-dark) instead of
       amber. Status/badge amber elsewhere on the page is unaffected. */
    --exo-yellow: var(--ux-text);
    --exo-yellow-darker: var(--ux-text);
    --exo-yellow-glow: var(--ux-text);
    --background: var(--ux-bg);
    --foreground: var(--ux-text);
    --card: var(--ux-card);
    --border: var(--ux-border);
    --primary: var(--ux-text);
    --muted: var(--ux-bg-raised);
    --muted-foreground: var(--ux-text-dim);
  }
  /* Override the legacy .command-panel global (defined with hardcoded oklch
     in app.css) so the chat input + message bubbles match the active theme. */
  .chat-fullbleed :global(.command-panel) {
    background: var(--ux-card) !important;
    border-color: var(--ux-border) !important;
    box-shadow: none !important;
  }
  .chat-fullbleed :global(.command-panel .text-foreground),
  .chat-fullbleed :global(.command-panel) :global(textarea) {
    color: var(--ux-text);
  }
  .chat-fullbleed :global(textarea::placeholder) {
    color: var(--ux-text-faint) !important;
  }
  @media (max-width: 720px) {
    .chat-fullbleed {
      margin: -28px -16px -64px;
      height: calc(100vh - 80px);
    }
  }
  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 28px 12px;
    border-bottom: 1px solid var(--ux-border);
    flex-wrap: wrap;
    gap: 12px;
  }
  .chat-header.minimal {
    border-bottom-color: transparent;
    padding: 12px 28px 8px;
  }
  .header-left {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ux-text-faint);
    font-size: 10px;
    font-weight: 600;
    font-family: var(--ux-mono);
  }
  .actions {
    display: flex;
    gap: 8px;
  }
  .model-picker {
    position: relative;
  }
  .model-button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px 6px 12px;
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius-sm);
    color: var(--ux-text);
    font-family: var(--ux-mono);
    font-size: 12.5px;
    cursor: pointer;
    transition: border-color 120ms, background 120ms;
  }
  .model-button:hover {
    border-color: var(--ux-border-strong);
    background: var(--ux-bg-hover);
  }
  .model-button svg {
    color: var(--ux-text-faint);
  }
  .model-label {
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .picker-backdrop {
    position: fixed;
    inset: 0;
    background: transparent;
    border: none;
    padding: 0;
    z-index: 9;
    cursor: default;
  }
  .picker-menu {
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    min-width: 320px;
    max-width: 480px;
    background: var(--ux-card);
    border: 1px solid var(--ux-border-strong);
    border-radius: var(--ux-radius);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    z-index: 10;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .picker-item {
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-rows: auto auto;
    gap: 0 10px;
    padding: 8px 10px;
    background: transparent;
    border: none;
    border-radius: var(--ux-radius-sm);
    text-align: left;
    cursor: pointer;
    color: var(--ux-text);
    font-family: var(--ux-sans);
    transition: background 120ms;
  }
  .picker-item:hover {
    background: var(--ux-bg-hover);
  }
  .picker-item.active {
    background: var(--ux-bg-hover);
    box-shadow: inset 2px 0 0 var(--ux-text);
  }
  .picker-dot {
    grid-row: 1 / span 2;
    align-self: center;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ux-green);
    box-shadow: 0 0 0 2px var(--ux-green-bg);
  }
  .picker-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--ux-text);
  }
  .picker-id {
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
    grid-column: 2;
  }
  .picker-empty {
    padding: 12px;
    color: var(--ux-text-dim);
    font-size: 12px;
    text-align: center;
  }
  .picker-empty a {
    display: block;
    margin-top: 6px;
    color: var(--ux-text);
    text-decoration: underline;
    text-decoration-color: var(--ux-border-strong);
  }
  .picker-footer {
    margin-top: 4px;
    padding: 8px 10px;
    border-top: 1px solid var(--ux-border);
    font-family: var(--ux-mono);
    font-size: 11px;
    color: var(--ux-text-faint);
    text-decoration: none;
    transition: color 120ms;
  }
  .picker-footer:hover {
    color: var(--ux-text);
  }
  .btn {
    font-family: var(--ux-sans);
    font-size: 12px;
    font-weight: 500;
    padding: 7px 12px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: var(--ux-card);
    color: var(--ux-text);
    cursor: pointer;
    text-decoration: none;
    transition: background 120ms;
  }
  .btn:hover {
    background: var(--ux-bg-hover);
  }
  .btn.ghost {
    background: transparent;
  }
  .chat-pane {
    overflow-y: auto;
    padding: 0;
    position: relative;
  }
  .empty-wrap {
    min-height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 24px;
  }
  .empty {
    text-align: center;
    max-width: 620px;
    width: 100%;
  }
  .empty-eyebrow {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.16em;
    margin-bottom: 16px;
  }
  .empty-title {
    font-size: 32px;
    font-weight: 600;
    color: var(--ux-text);
    margin: 0 0 12px;
    letter-spacing: -0.02em;
    line-height: 1.1;
  }
  .empty-sub {
    font-size: 13.5px;
    color: var(--ux-text-dim);
    line-height: 1.55;
    margin: 0 auto 28px;
    max-width: 460px;
    font-family: var(--ux-sans);
  }
  .empty-sub a {
    color: var(--ux-text);
    text-decoration: none;
    border-bottom: 1px dashed var(--ux-border-strong);
  }
  .empty-sub a:hover {
    border-bottom-style: solid;
    border-bottom-color: var(--ux-text);
  }
  .empty-model {
    font-family: var(--ux-mono);
    color: var(--ux-text);
    font-size: 12.5px;
    background: var(--ux-bg-raised);
    padding: 1px 6px;
    border-radius: 3px;
    border: 1px solid var(--ux-border);
  }
  .suggestions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
    text-align: left;
  }
  @media (max-width: 600px) {
    .suggestions {
      grid-template-columns: 1fr;
    }
  }
  .chip {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 14px;
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius-sm);
    color: var(--ux-text-dim);
    font-family: var(--ux-sans);
    font-size: 13px;
    line-height: 1.4;
    cursor: pointer;
    transition: border-color 120ms, color 120ms, background 120ms, transform 120ms;
    text-align: left;
  }
  .chip:hover {
    border-color: var(--ux-border-strong);
    color: var(--ux-text);
    background: var(--ux-bg-hover);
    transform: translateY(-1px);
  }
  .chip-arrow {
    font-family: var(--ux-mono);
    color: var(--ux-text-faint);
    flex-shrink: 0;
    font-size: 13px;
    line-height: 1.4;
  }
  .chip-text {
    flex: 1;
  }
  .chat-form-wrap {
    padding: 14px 22px 20px;
    border-top: 1px solid var(--ux-border);
    background: var(--ux-bg);
  }
  .chat-form-wrap.floating {
    border-top-color: transparent;
  }
</style>

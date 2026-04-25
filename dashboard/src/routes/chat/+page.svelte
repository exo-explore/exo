<script lang="ts">
  import ChatMessages from "$lib/components/ChatMessages.svelte";
  import ChatForm from "$lib/components/ChatForm.svelte";
  import {
    sendMessage,
    messages,
    clearChat,
    selectedChatModel,
    thinkingEnabled,
    instances,
  } from "$lib/stores/app.svelte";

  let scrollParent = $state<HTMLDivElement | null>(null);

  let messageList = $derived(messages());
  let model = $derived(selectedChatModel());
  let isEmpty = $derived(messageList.length === 0);
  let runningInstances = $derived(instances());
  let hasModelLoaded = $derived(
    Object.keys(runningInstances ?? {}).length > 0 || !!model,
  );

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
    <div>
      <div class="eyebrow">CHAT</div>
      <div class="title">
        {#if isEmpty}
          {model ? `Talking to ${model}` : "No model selected"}
        {:else}
          {model ? `Talking to ${model}` : "Conversation"}
        {/if}
      </div>
    </div>
    <div class="actions">
      {#if !isEmpty}
        <button class="btn ghost" onclick={clearChat}>+ New chat</button>
      {/if}
      <a class="btn ghost" href="#/models">Switch model</a>
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
      showModelSelector={true}
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
    padding: 18px 28px 14px;
    border-bottom: 1px solid var(--ux-border);
    flex-wrap: wrap;
    gap: 12px;
  }
  .chat-header.minimal {
    border-bottom-color: transparent;
    padding: 14px 28px 10px;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ux-text-faint);
    font-size: 10px;
    font-weight: 600;
    font-family: var(--ux-mono);
    margin-bottom: 4px;
  }
  .title {
    font-size: 14px;
    font-weight: 500;
    color: var(--ux-text);
    font-family: var(--ux-mono);
  }
  .actions {
    display: flex;
    gap: 8px;
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
    color: var(--ux-accent);
    text-decoration: none;
    border-bottom: 1px dashed var(--ux-accent);
  }
  .empty-sub a:hover {
    border-bottom-style: solid;
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
    border-color: var(--ux-accent);
    color: var(--ux-text);
    background: var(--ux-bg-hover);
    transform: translateY(-1px);
  }
  .chip-arrow {
    font-family: var(--ux-mono);
    color: var(--ux-accent);
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

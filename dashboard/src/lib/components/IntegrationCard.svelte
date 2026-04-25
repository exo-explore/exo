<script lang="ts">
  import { copyText } from "$lib/utils/clipboard";

  interface Props {
    title: string;
    subtitle: string;
    config: string;
    description?: string;
    language?: "json" | "bash";
  }

  let {
    title,
    subtitle,
    config,
    description = "",
    language = "json",
  }: Props = $props();

  let copied = $state(false);
  let failed = $state(false);

  async function copyToClipboard() {
    const ok = await copyText(config);
    if (ok) {
      copied = true;
      setTimeout(() => (copied = false), 2000);
    } else {
      failed = true;
      setTimeout(() => (failed = false), 2000);
    }
  }
</script>

<div class="ic">
  <div class="ic-head">
    <div>
      <h3 class="ic-title">{title}</h3>
      <p class="ic-sub">{subtitle}</p>
    </div>
    <button
      onclick={copyToClipboard}
      class="ic-copy"
      class:copied
      class:failed
    >
      {copied ? "Copied" : failed ? "Failed" : "Copy"}
    </button>
  </div>
  {#if description}
    <p class="ic-desc">{description}</p>
  {/if}
  <div class="ic-code">
    <pre>{config}</pre>
  </div>
</div>

<style>
  .ic {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    overflow: hidden;
  }
  .ic-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    padding: 14px 18px 10px;
  }
  .ic-title {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--ux-text);
    letter-spacing: -0.005em;
  }
  .ic-sub {
    margin: 2px 0 0;
    font-family: var(--ux-mono);
    font-size: 11px;
    color: var(--ux-text-faint);
  }
  .ic-copy {
    flex-shrink: 0;
    font-family: var(--ux-sans);
    font-size: 11px;
    padding: 5px 11px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: var(--ux-bg-raised);
    color: var(--ux-text-dim);
    cursor: pointer;
    transition: border-color 120ms, color 120ms, background 120ms;
  }
  .ic-copy:hover {
    color: var(--ux-text);
    border-color: var(--ux-accent);
  }
  .ic-copy.copied {
    color: var(--ux-green);
    border-color: var(--ux-green-border);
    background: var(--ux-green-bg);
  }
  .ic-copy.failed {
    color: var(--ux-red);
    border-color: var(--ux-red-border);
    background: var(--ux-red-bg);
  }
  .ic-desc {
    margin: 0;
    padding: 0 18px 12px;
    font-size: 12px;
    color: var(--ux-text-dim);
    line-height: 1.45;
  }
  .ic-code {
    background: var(--ux-bg-raised);
    border-top: 1px solid var(--ux-border);
  }
  .ic-code pre {
    margin: 0;
    padding: 14px 18px;
    font-family: var(--ux-mono);
    font-size: 12px;
    color: var(--ux-text);
    overflow-x: auto;
    white-space: pre;
    line-height: 1.5;
  }
</style>

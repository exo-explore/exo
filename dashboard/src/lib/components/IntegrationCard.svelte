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

<div
  class="border border-exo-light-gray/20 rounded-lg bg-exo-medium-gray/20 overflow-hidden"
>
  <div class="flex items-center justify-between px-5 py-4">
    <div>
      <h3 class="text-white text-sm font-semibold tracking-wide">{title}</h3>
      <p class="text-exo-light-gray/60 text-xs mt-0.5 font-mono">{subtitle}</p>
    </div>
    <button
      onclick={copyToClipboard}
      class="px-3 py-1.5 text-xs rounded border transition-all duration-200 cursor-pointer
        {copied
        ? 'border-green-500/50 text-green-400 bg-green-500/10'
        : failed
          ? 'border-red-500/50 text-red-400 bg-red-500/10'
          : 'border-exo-light-gray/30 text-exo-light-gray hover:border-exo-yellow/50 hover:text-exo-yellow'}"
    >
      {copied ? "Copied!" : failed ? "Copy failed" : "Copy"}
    </button>
  </div>
  {#if description}
    <p class="text-exo-light-gray/70 text-xs px-5 pb-3">{description}</p>
  {/if}
  <div class="bg-black/30 border-t border-exo-light-gray/10">
    <pre
      class="text-xs text-exo-light-gray/90 font-mono p-4 overflow-x-auto whitespace-pre">{config}</pre>
  </div>
</div>

<script lang="ts">
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";

  interface Props {
    src: string | null;
    onclose: () => void;
  }

  let { src, onclose }: Props = $props();

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      onclose();
    }
  }

  function extensionFromSrc(dataSrc: string): string {
    const match = dataSrc.match(/^data:image\/(\w+)/);
    if (match) return match[1] === "jpeg" ? "jpg" : match[1];
    const urlMatch = dataSrc.match(/\.(\w+)(?:\?|$)/);
    if (urlMatch) return urlMatch[1];
    return "png";
  }

  function handleDownload(e: MouseEvent) {
    e.stopPropagation();
    if (!src) return;
    const link = document.createElement("a");
    link.href = src;
    link.download = `image-${Date.now()}.${extensionFromSrc(src)}`;
    link.click();
  }

  function handleClose(e: MouseEvent) {
    e.stopPropagation();
    onclose();
  }
</script>

<svelte:window onkeydown={src ? handleKeydown : undefined} />

{#if src}
  <div
    class="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center"
    transition:fade={{ duration: 200 }}
    onclick={onclose}
    role="presentation"
    onintrostart={() => (document.body.style.overflow = "hidden")}
    onoutroend={() => (document.body.style.overflow = "")}
  >
    <div class="absolute top-4 right-4 flex gap-2 z-10">
      <button
        type="button"
        class="p-2 rounded-lg bg-exo-dark-gray/80 border border-exo-yellow/30 text-exo-yellow hover:bg-exo-dark-gray hover:border-exo-yellow/50 cursor-pointer transition-colors"
        onclick={handleDownload}
        title="Download image"
      >
        <svg
          class="w-5 h-5"
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
      <button
        type="button"
        class="p-2 rounded-lg bg-exo-dark-gray/80 border border-exo-yellow/30 text-exo-yellow hover:bg-exo-dark-gray hover:border-exo-yellow/50 cursor-pointer transition-colors"
        onclick={handleClose}
        title="Close"
      >
        <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
          />
        </svg>
      </button>
    </div>

    <!-- svelte-ignore a11y_no_noninteractive_element_interactions, a11y_click_events_have_key_events -->
    <img
      {src}
      alt=""
      class="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
      transition:fly={{ y: 20, duration: 300, easing: cubicOut }}
      onclick={(e) => e.stopPropagation()}
    />
  </div>
{/if}

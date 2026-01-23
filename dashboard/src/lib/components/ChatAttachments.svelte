<script lang="ts">
  import type { ChatUploadedFile } from "$lib/types/files";
  import { formatFileSize, getFileCategory } from "$lib/types/files";

  interface Props {
    files: ChatUploadedFile[];
    readonly?: boolean;
    onRemove?: (fileId: string) => void;
  }

  let { files, readonly = false, onRemove }: Props = $props();

  function getFileIcon(file: ChatUploadedFile): string {
    const category = getFileCategory(file.type, file.name);
    switch (category) {
      case "image":
        return "🖼";
      case "text":
        return "📄";
      case "pdf":
        return "📑";
      case "audio":
        return "🎵";
      default:
        return "📎";
    }
  }

  function truncateName(name: string, maxLen: number = 20): string {
    if (name.length <= maxLen) return name;
    const ext = name.slice(name.lastIndexOf("."));
    const base = name.slice(0, name.lastIndexOf("."));
    const available = maxLen - ext.length - 3;
    return base.slice(0, available) + "..." + ext;
  }
</script>

{#if files.length > 0}
  <div class="flex flex-wrap gap-2 mb-3 px-1">
    {#each files as file (file.id)}
      <div
        class="group relative flex items-center gap-2 bg-exo-dark-gray/80 border border-exo-yellow/30 rounded px-2.5 py-1.5 text-xs font-mono transition-all hover:border-exo-yellow/50 hover:shadow-[0_0_10px_rgba(255,215,0,0.1)]"
      >
        <!-- File preview or icon -->
        {#if file.preview && getFileCategory(file.type, file.name) === "image"}
          <img
            src={file.preview}
            alt={file.name}
            class="w-8 h-8 object-cover rounded border border-exo-yellow/20"
          />
        {:else}
          <span class="text-base">{getFileIcon(file)}</span>
        {/if}

        <!-- File info -->
        <div class="flex flex-col min-w-0">
          <span
            class="text-exo-yellow truncate max-w-[120px]"
            title={file.name}
          >
            {truncateName(file.name)}
          </span>
          <span class="text-exo-light-gray text-xs">
            {formatFileSize(file.size)}
          </span>
        </div>

        <!-- Remove button -->
        {#if !readonly && onRemove}
          <button
            type="button"
            onclick={() => onRemove?.(file.id)}
            class="ml-1 w-4 h-4 flex items-center justify-center text-exo-light-gray hover:text-red-400 transition-colors cursor-pointer"
            title="Remove file"
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
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        {/if}
      </div>
    {/each}
  </div>
{/if}

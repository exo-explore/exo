<script lang="ts">
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";

  interface Props {
    isOpen: boolean;
    onClose: () => void;
  }

  let { isOpen, onClose }: Props = $props();

  let bugReportId = $state<string | null>(null);
  let githubIssueUrl = $state<string | null>(null);
  let isLoading = $state(false);
  let error = $state<string | null>(null);

  async function generateBugReport() {
    isLoading = true;
    error = null;
    try {
      const response = await fetch("/bug-report", { method: "POST" });
      if (!response.ok) {
        error = "Failed to generate bug report. Please try again.";
        return;
      }
      const data = await response.json();
      bugReportId = data.bugReportId;
      githubIssueUrl = data.githubIssueUrl;
    } catch {
      error = "Failed to connect to the server. Please try again.";
    } finally {
      isLoading = false;
    }
  }

  function handleClose() {
    bugReportId = null;
    githubIssueUrl = null;
    error = null;
    isLoading = false;
    onClose();
  }

  // Generate bug report when modal opens
  $effect(() => {
    if (isOpen && !bugReportId && !isLoading) {
      generateBugReport();
    }
  });
</script>

{#if isOpen}
  <!-- Backdrop -->
  <div
    class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm"
    transition:fade={{ duration: 200 }}
    onclick={handleClose}
    role="presentation"
  ></div>

  <!-- Modal -->
  <div
    class="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(90vw,480px)] bg-exo-dark-gray border border-exo-yellow/10 rounded-lg shadow-2xl overflow-hidden flex flex-col"
    transition:fly={{ y: 20, duration: 300, easing: cubicOut }}
    role="dialog"
    aria-modal="true"
    aria-label="Bug Report"
  >
    <!-- Header -->
    <div
      class="flex items-center justify-between px-5 py-4 border-b border-exo-medium-gray/30"
    >
      <div class="flex items-center gap-2">
        <svg
          class="w-5 h-5 text-exo-yellow"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <h2 class="text-sm font-mono text-exo-yellow tracking-wider uppercase">
          Report a Bug
        </h2>
      </div>
      <button
        onclick={handleClose}
        class="text-exo-light-gray hover:text-white transition-colors cursor-pointer"
        aria-label="Close"
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
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    </div>

    <!-- Body -->
    <div class="px-5 py-5 space-y-4">
      {#if isLoading}
        <div class="flex items-center justify-center py-6">
          <div
            class="w-5 h-5 border-2 border-exo-yellow/30 border-t-exo-yellow rounded-full animate-spin"
          ></div>
          <span class="ml-3 text-sm text-exo-light-gray font-mono"
            >Generating bug report...</span
          >
        </div>
      {:else if error}
        <div
          class="text-sm text-red-400 font-mono bg-red-400/10 border border-red-400/20 rounded px-4 py-3"
        >
          {error}
        </div>
        <button
          onclick={generateBugReport}
          class="w-full px-4 py-2.5 bg-exo-medium-gray/50 border border-exo-yellow/30 rounded text-sm font-mono text-exo-yellow hover:border-exo-yellow/60 transition-colors cursor-pointer"
        >
          Try Again
        </button>
      {:else if bugReportId && githubIssueUrl}
        <p class="text-sm text-exo-light-gray leading-relaxed">
          Would you like to create a GitHub issue? This would help us track and
          fix the issue for you.
        </p>

        <!-- Bug Report ID -->
        <div
          class="bg-exo-black/50 border border-exo-medium-gray/30 rounded px-4 py-3"
        >
          <div
            class="text-[11px] text-exo-light-gray/60 font-mono tracking-wider uppercase mb-1"
          >
            Bug Report ID
          </div>
          <div class="text-sm text-exo-yellow font-mono tracking-wide">
            {bugReportId}
          </div>
          <div class="text-[11px] text-exo-light-gray/50 font-mono mt-1">
            Include this ID when communicating with the team.
          </div>
        </div>

        <p class="text-xs text-exo-light-gray/60 leading-relaxed">
          No diagnostic data is attached. The issue template contains
          placeholder fields for you to fill in.
        </p>

        <!-- Actions -->
        <div class="flex gap-3 pt-1">
          <a
            href={githubIssueUrl}
            target="_blank"
            rel="noopener noreferrer"
            class="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-exo-yellow/10 border border-exo-yellow/40 rounded text-sm font-mono text-exo-yellow hover:bg-exo-yellow/20 hover:border-exo-yellow/60 transition-colors"
          >
            <svg class="w-4 h-4" viewBox="0 0 16 16" fill="currentColor">
              <path
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
              />
            </svg>
            Create GitHub Issue
          </a>
          <button
            onclick={handleClose}
            class="px-4 py-2.5 border border-exo-medium-gray/40 rounded text-sm font-mono text-exo-light-gray hover:border-exo-medium-gray/60 transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      {/if}
    </div>
  </div>
{/if}

<script lang="ts">
  import { browser } from "$app/environment";

  export let showHome = true;
  export let onHome: (() => void) | null = null;
  export let showSidebarToggle = false;
  export let sidebarVisible = true;
  export let onToggleSidebar: (() => void) | null = null;
  export let downloadProgress: {
    count: number;
    percentage: number;
  } | null = null;

  function handleHome(): void {
    if (onHome) {
      onHome();
      return;
    }
    if (browser) {
      // Hash router: send to root
      window.location.hash = "/";
    }
  }

  function handleToggleSidebar(): void {
    if (onToggleSidebar) {
      onToggleSidebar();
    }
  }
</script>

<header
  class="relative z-20 flex items-center justify-center px-6 pt-8 pb-4 bg-exo-dark-gray"
>
  <!-- Left: Sidebar Toggle -->
  {#if showSidebarToggle}
    <div class="absolute left-6 top-1/2 -translate-y-1/2">
      <button
        onclick={handleToggleSidebar}
        class="p-2 rounded border border-exo-light-gray/30 hover:border-exo-yellow/50 hover:bg-exo-medium-gray/30 transition-colors cursor-pointer"
        title={sidebarVisible ? "Hide sidebar" : "Show sidebar"}
        aria-label={sidebarVisible
          ? "Hide conversation sidebar"
          : "Show conversation sidebar"}
        aria-pressed={sidebarVisible}
      >
        <svg
          class="w-5 h-5 {sidebarVisible
            ? 'text-exo-yellow'
            : 'text-exo-light-gray'}"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          {#if sidebarVisible}
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
            />
          {:else}
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M13 5l7 7-7 7M5 5l7 7-7 7"
            />
          {/if}
        </svg>
      </button>
    </div>
  {/if}

  <!-- Center: Logo (clickable to go home) -->
  <button
    onclick={handleHome}
    class="bg-transparent border-none outline-none focus:outline-none transition-opacity duration-200 hover:opacity-90 {showHome
      ? 'cursor-pointer'
      : 'cursor-default'}"
    title={showHome ? "Go to home" : ""}
    disabled={!showHome}
  >
    <img
      src="/exo-logo.png"
      alt="EXO"
      class="h-18 drop-shadow-[0_0_4px_rgba(255,215,0,0.3)]"
    />
  </button>

  <!-- Right: Home + Downloads -->
  <nav
    class="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-4"
    aria-label="Main navigation"
  >
    {#if showHome}
      <button
        onclick={handleHome}
        class="text-sm text-white/70 hover:text-exo-yellow transition-colors tracking-wider uppercase flex items-center gap-2 cursor-pointer"
        title="Back to topology view"
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
            d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
          />
        </svg>
        Home
      </button>
    {/if}
    <a
      href="/#/downloads"
      class="text-sm text-white/70 hover:text-exo-yellow transition-colors tracking-wider uppercase flex items-center gap-2 cursor-pointer"
      title="View downloads overview"
    >
      {#if downloadProgress}
        <!-- Compact download progress indicator -->
        <div class="relative w-4 h-4 flex-shrink-0">
          <svg class="w-4 h-4 -rotate-90" viewBox="0 0 20 20">
            <circle
              cx="10"
              cy="10"
              r="8"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              opacity="0.2"
            />
            <circle
              cx="10"
              cy="10"
              r="8"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-dasharray={2 * Math.PI * 8}
              stroke-dashoffset={2 *
                Math.PI *
                8 *
                (1 - downloadProgress.percentage / 100)}
              class="text-blue-400 transition-all duration-300"
            />
          </svg>
          <div
            class="absolute inset-0 flex items-center justify-center text-[6px] font-mono text-blue-400"
          >
            {downloadProgress.count}
          </div>
        </div>
      {:else}
        <svg
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M12 3v12" />
          <path d="M7 12l5 5 5-5" />
          <path d="M5 21h14" />
        </svg>
      {/if}
      Downloads
    </a>
  </nav>
</header>

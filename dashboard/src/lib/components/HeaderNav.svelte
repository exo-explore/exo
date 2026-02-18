<script lang="ts">
  import { browser } from "$app/environment";
  import { toggleMode, mode } from "mode-watcher";

  let {
    showHome = true,
    onHome = null,
    showSidebarToggle = false,
    sidebarVisible = true,
    onToggleSidebar = null,
  }: {
    showHome?: boolean;
    onHome?: (() => void) | null;
    showSidebarToggle?: boolean;
    sidebarVisible?: boolean;
    onToggleSidebar?: (() => void) | null;
  } = $props();

  const currentMode = $derived(mode.current);

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
        class="p-2 rounded border border-exo-medium-gray/40 hover:border-exo-yellow/50 transition-colors cursor-pointer"
        title={sidebarVisible ? "Hide sidebar" : "Show sidebar"}
      >
        <svg
          class="w-5 h-5 {sidebarVisible
            ? 'text-exo-yellow'
            : 'text-exo-medium-gray'}"
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
      class="h-18 drop-shadow-[0_0_20px_color-mix(in_oklch,var(--exo-yellow)_50%,transparent)]"
    />
  </button>

  <!-- Right: Theme Toggle + Home + Downloads -->
  <div
    class="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-4"
  >
    <!-- Theme Toggle Button -->
    <button
      onclick={() => toggleMode()}
      class="p-2 rounded border border-exo-medium-gray/40 hover:border-exo-yellow/50 transition-colors cursor-pointer"
      title={currentMode === "dark"
        ? "Switch to light mode"
        : "Switch to dark mode"}
      aria-label={currentMode === "dark"
        ? "Switch to light mode"
        : "Switch to dark mode"}
    >
      {#if currentMode === "dark"}
        <!-- Sun icon - click to go light -->
        <svg
          class="w-4 h-4 text-exo-light-gray hover:text-exo-yellow transition-colors"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          <circle cx="12" cy="12" r="5" />
          <path
            stroke-linecap="round"
            d="M12 1v2m0 18v2m-9-11H1m22 0h-2m-2.636-6.364l-1.414 1.414M6.05 6.05L4.636 4.636m0 14.728l1.414-1.414m11.314 1.414l1.414 1.414"
          />
        </svg>
      {:else}
        <!-- Moon icon - click to go dark -->
        <svg
          class="w-4 h-4 text-exo-light-gray hover:text-exo-yellow transition-colors"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z"
          />
        </svg>
      {/if}
    </button>

    {#if showHome}
      <button
        onclick={handleHome}
        class="text-sm text-exo-light-gray hover:text-exo-yellow transition-colors tracking-wider uppercase flex items-center gap-2 cursor-pointer"
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
      class="text-sm text-exo-light-gray hover:text-exo-yellow transition-colors tracking-wider uppercase flex items-center gap-2 cursor-pointer"
      title="View downloads overview"
    >
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
      Downloads
    </a>
  </div>
</header>

<script lang="ts">
  import { browser } from "$app/environment";

  interface Props {
    showHome?: boolean;
    onHome?: (() => void) | null;
    showSidebarToggle?: boolean;
    sidebarVisible?: boolean;
    onToggleSidebar?: (() => void) | null;
    showMobileMenuToggle?: boolean;
    mobileMenuOpen?: boolean;
    onToggleMobileMenu?: (() => void) | null;
    showMobileRightToggle?: boolean;
    mobileRightOpen?: boolean;
    onToggleMobileRight?: (() => void) | null;
    downloadProgress?: {
      count: number;
      percentage: number;
    } | null;
  }

  let {
    showHome = true,
    onHome = null,
    showSidebarToggle = false,
    sidebarVisible = true,
    onToggleSidebar = null,
    showMobileMenuToggle = false,
    mobileMenuOpen = false,
    onToggleMobileMenu = null,
    showMobileRightToggle = false,
    mobileRightOpen = false,
    onToggleMobileRight = null,
    downloadProgress = null,
  }: Props = $props();

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

  function handleToggleMobileMenu(): void {
    if (onToggleMobileMenu) {
      onToggleMobileMenu();
    }
  }

  function handleToggleMobileRight(): void {
    if (onToggleMobileRight) {
      onToggleMobileRight();
    }
  }
</script>

<header
  class="relative z-20 flex items-center justify-center px-4 md:px-6 pt-4 md:pt-8 pb-3 md:pb-4 bg-exo-dark-gray"
>
  <!-- Left: Sidebar Toggle (desktop) or Mobile Sidebar Toggle (mobile) -->
  <div
    class="absolute left-4 md:left-6 top-1/2 -translate-y-1/2 flex items-center gap-2"
  >
    <!-- Mobile sidebar toggle -->
    <button
      onclick={handleToggleMobileMenu}
      class="p-2 rounded border border-exo-light-gray/30 hover:border-exo-yellow/50 hover:bg-exo-medium-gray/30 transition-colors cursor-pointer md:hidden"
      title={mobileMenuOpen ? "Hide sidebar" : "Show sidebar"}
      aria-label={mobileMenuOpen
        ? "Hide conversation sidebar"
        : "Show conversation sidebar"}
      aria-pressed={mobileMenuOpen}
    >
      <svg
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        stroke-width="2"
        class="w-5 h-5 {mobileMenuOpen
          ? 'text-exo-yellow'
          : 'text-exo-light-gray'}"
      >
        {#if mobileMenuOpen}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
          ></path>
        {:else}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M13 5l7 7-7 7M5 5l7 7-7 7"
          ></path>
        {/if}
      </svg>
    </button>
    <!-- Desktop sidebar toggle -->
    <button
      onclick={handleToggleSidebar}
      class="p-2 rounded border border-exo-light-gray/30 hover:border-exo-yellow/50 hover:bg-exo-medium-gray/30 transition-colors cursor-pointer hidden md:block"
      title={sidebarVisible ? "Hide sidebar" : "Show sidebar"}
      aria-label={sidebarVisible
        ? "Hide conversation sidebar"
        : "Show conversation sidebar"}
      aria-pressed={sidebarVisible}
    >
      <svg
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        stroke-width="2"
        class="w-5 h-5 {sidebarVisible
          ? 'text-exo-yellow'
          : 'text-exo-light-gray'}"
      >
        {#if sidebarVisible}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
          ></path>
        {:else}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M13 5l7 7-7 7M5 5l7 7-7 7"
          ></path>
        {/if}
      </svg>
    </button>
  </div>

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
      class="h-12 md:h-18 drop-shadow-[0_0_4px_rgba(255,215,0,0.3)]"
    />
  </button>

  <!-- Right: Home + Downloads + Mobile Right Toggle -->
  <nav
    class="absolute right-4 md:right-6 top-1/2 -translate-y-1/2 flex items-center gap-2 md:gap-4"
    aria-label="Main navigation"
  >
    <!-- Mobile right sidebar toggle (instances/models) - only show when not in chat mode -->
    {#if showMobileRightToggle}
      <button
        onclick={handleToggleMobileRight}
        class="p-2 rounded border border-exo-light-gray/30 hover:border-exo-yellow/50 hover:bg-exo-medium-gray/30 transition-colors cursor-pointer md:hidden"
        title={mobileRightOpen ? "Hide instances" : "Show instances"}
        aria-label={mobileRightOpen
          ? "Hide instances panel"
          : "Show instances panel"}
        aria-pressed={mobileRightOpen}
      >
        <svg
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
          class="w-5 h-5 {mobileRightOpen
            ? 'text-exo-yellow'
            : 'text-exo-light-gray'}"
        >
          {#if mobileRightOpen}
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M13 5l7 7-7 7M5 5l7 7-7 7"
            ></path>
          {:else}
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
            ></path>
          {/if}
        </svg>
      </button>
    {/if}
    {#if showHome}
      <button
        onclick={handleHome}
        class="flex text-sm text-white/70 hover:text-exo-yellow transition-colors tracking-wider uppercase items-center gap-2 cursor-pointer"
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
        <span class="hidden sm:inline">Home</span>
      </button>
    {/if}
    <a
      href="/#/downloads"
      class="text-xs md:text-sm text-white/70 hover:text-exo-yellow transition-colors tracking-wider uppercase flex items-center gap-1.5 md:gap-2 cursor-pointer"
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
      <span class="hidden sm:inline">Downloads</span>
    </a>
    <a
      href="/#/integrations"
      class="text-xs md:text-sm text-white/70 hover:text-exo-yellow transition-colors tracking-wider uppercase flex items-center gap-1.5 md:gap-2 cursor-pointer"
      title="Integration configs for external tools"
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
        <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
        <path
          d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"
        />
      </svg>
      <span class="hidden sm:inline">Integrations</span>
    </a>
  </nav>
</header>

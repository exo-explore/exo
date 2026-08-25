<script lang="ts">
  import { browser } from "$app/environment";
  import { featureFlags } from "$lib/stores/app.svelte";

  const showAdvanced = $derived(featureFlags()["disaggregation"] === true);

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

  let mobileNavOpen = $state(false);

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
  class="relative z-20 flex h-[72px] items-center justify-center border-b border-white/[0.07] bg-exo-black/75 px-4 backdrop-blur-xl md:px-6"
>
  <!-- Left: Sidebar Toggle (desktop) or Mobile Sidebar Toggle (mobile) -->
  <div
    class="absolute left-4 md:left-6 top-1/2 -translate-y-1/2 flex items-center gap-2"
  >
    <!-- Mobile sidebar toggle -->
    <button
      onclick={handleToggleMobileMenu}
      class="grid h-10 w-10 place-items-center rounded-xl border border-white/10 bg-white/[0.025] transition-all hover:border-exo-yellow/35 hover:bg-exo-yellow/[0.06] cursor-pointer md:hidden"
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
      class="h-10 w-10 rounded-xl border border-white/10 bg-white/[0.025] transition-all hover:border-exo-yellow/35 hover:bg-exo-yellow/[0.06] cursor-pointer hidden md:grid place-items-center"
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
    class="group flex items-center gap-3 rounded-xl bg-transparent border-none px-2 py-1 transition-opacity duration-200 hover:opacity-90 {showHome
      ? 'cursor-pointer'
      : 'cursor-default'}"
    title={showHome ? "Go to home" : ""}
    disabled={!showHome}
  >
    <img
      src="/exo-logo.png"
      alt="EXO"
      class="h-9 md:h-11 drop-shadow-[0_0_12px_rgba(255,215,0,0.22)]"
    />
    <span class="hidden lg:block border-l border-white/10 pl-3 text-left">
      <span
        class="block text-[10px] font-mono font-semibold tracking-[0.18em] text-white/55 uppercase"
        >Distributed AI</span
      >
      <span class="mt-0.5 block text-[10px] text-white/30">Cluster control</span
      >
    </span>
  </button>

  <!-- Right: Home + Downloads + Mobile Right Toggle -->
  <nav
    class="absolute right-4 md:right-6 top-1/2 -translate-y-1/2 flex items-center gap-1 rounded-xl border border-white/[0.06] bg-white/[0.025] p-1"
    aria-label="Main navigation"
  >
    <!-- Mobile right sidebar toggle (instances/models) - only show when not in chat mode -->
    {#if showMobileRightToggle}
      <button
        onclick={handleToggleMobileRight}
        class="grid h-9 w-9 place-items-center rounded-lg text-white/65 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow cursor-pointer md:hidden"
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
        class="hidden h-9 items-center gap-2 rounded-lg px-2.5 text-xs font-medium text-white/65 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow cursor-pointer sm:flex"
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
      class="hidden h-9 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium text-white/65 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow cursor-pointer sm:flex md:gap-2"
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
      class="hidden h-9 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium text-white/65 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow cursor-pointer sm:flex md:gap-2"
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
    {#if showAdvanced}
      <a
        href="/#/advanced"
        class="hidden h-9 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium text-white/65 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow cursor-pointer sm:flex md:gap-2"
        title="Advanced cluster settings"
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
          <circle cx="12" cy="12" r="3" />
          <path
            d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
          />
        </svg>
        <span class="hidden sm:inline">Advanced</span>
      </a>
    {/if}

    <div class="relative sm:hidden">
      <button
        type="button"
        class="grid h-9 w-9 place-items-center rounded-lg text-white/65 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow cursor-pointer"
        onclick={() => (mobileNavOpen = !mobileNavOpen)}
        aria-label="Open navigation menu"
        aria-expanded={mobileNavOpen}
      >
        <svg
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        >
          <circle cx="12" cy="5" r="1" fill="currentColor" />
          <circle cx="12" cy="12" r="1" fill="currentColor" />
          <circle cx="12" cy="19" r="1" fill="currentColor" />
        </svg>
      </button>

      {#if mobileNavOpen}
        <div
          class="absolute right-0 top-12 z-50 w-48 overflow-hidden rounded-xl border border-white/10 bg-exo-dark-gray/95 p-1.5 shadow-2xl backdrop-blur-xl"
        >
          <a
            href="/#/downloads"
            onclick={() => (mobileNavOpen = false)}
            class="flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs text-white/70 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow"
            >Downloads</a
          >
          <a
            href="/#/integrations"
            onclick={() => (mobileNavOpen = false)}
            class="flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs text-white/70 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow"
            >Integrations</a
          >
          {#if showAdvanced}
            <a
              href="/#/advanced"
              onclick={() => (mobileNavOpen = false)}
              class="flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs text-white/70 transition-colors hover:bg-white/[0.06] hover:text-exo-yellow"
              >Advanced</a
            >
          {/if}
        </div>
      {/if}
    </div>
  </nav>
</header>

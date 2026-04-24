<script lang="ts">
  import TopBar from "./TopBar.svelte";
  import ActivityTicker from "./ActivityTicker.svelte";

  let { children } = $props();
</script>

<div class="ux-shell">
  <TopBar />
  <ActivityTicker />
  <main class="ux-main">
    {@render children?.()}
  </main>
</div>

<style>
  .ux-shell {
    min-height: 100vh;
    background: var(--ux-bg);
    color: var(--ux-text);
    font-family: var(--ux-sans);
  }
  .ux-main {
    padding: 48px 28px 96px;
    max-width: 1280px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
  }
  @media (max-width: 720px) {
    .ux-main {
      padding: 28px 16px 64px;
    }
  }
</style>

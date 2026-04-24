<script lang="ts">
  import "../app.css";
  import ToastContainer from "$lib/components/ToastContainer.svelte";
  import { page } from "$app/state";
  import { onMount } from "svelte";
  import AppShell from "$lib/components/shell/AppShell.svelte";

  let { children } = $props();

  // The /legacy/* tree keeps the existing command-center aesthetic.
  // Everything else gets the new oMLX-feel shell.
  let onLegacy = $derived(page.url.hash.startsWith("#/legacy"));

  onMount(() => {
    const apply = () => {
      const legacy = location.hash.startsWith("#/legacy");
      document.body.classList.toggle("ux-new-shell", !legacy);
    };
    apply();
    window.addEventListener("hashchange", apply);
    return () => window.removeEventListener("hashchange", apply);
  });
</script>

<svelte:head>
  <title>EXO</title>
  <meta name="description" content="EXO — distributed AI cluster" />
</svelte:head>

{#if onLegacy}
  <div class="min-h-screen bg-background text-foreground">
    {@render children?.()}
    <ToastContainer />
  </div>
{:else}
  <AppShell>
    {@render children?.()}
  </AppShell>
  <ToastContainer />
{/if}

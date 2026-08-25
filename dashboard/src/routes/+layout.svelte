<script lang="ts">
  import "../app.css";
  import ToastContainer from "$lib/components/ToastContainer.svelte";
  import ConnectionBanner from "$lib/components/ConnectionBanner.svelte";

  let { children } = $props();
</script>

<svelte:head>
  <title>EXO — Distributed AI Cluster</title>
  <meta name="description" content="EXO - Distributed AI Cluster Dashboard" />
  <meta name="theme-color" content="#0b0d11" />
  <meta name="color-scheme" content="dark" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
  <a class="exo-skip-link" href="#main-content">Skip to dashboard</a>
  <ConnectionBanner />
  {@render children?.()}
  <ToastContainer />
</div>

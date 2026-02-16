<script lang="ts">
  import "../app.css";
  import ToastContainer from "$lib/components/ToastContainer.svelte";
  import ConnectionBanner from "$lib/components/ConnectionBanner.svelte";

  let { children } = $props();
</script>

<svelte:head>
  <title>EXO</title>
  <meta name="description" content="EXO - Distributed AI Cluster Dashboard" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
  <ConnectionBanner />
  {@render children?.()}
  <ToastContainer />
</div>

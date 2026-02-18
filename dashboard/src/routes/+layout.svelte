<script lang="ts">
  import "../app.css";
  import { onMount } from "svelte";
  import { theme } from "$lib/stores/theme.svelte";

  let { children } = $props();

  onMount(() => {
    theme.init();
  });
</script>

<svelte:head>
  <title>EXO</title>
  <meta name="description" content="EXO - Distributed AI Cluster Dashboard" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
  {@render children?.()}
</div>

<script lang="ts">
  import "../app.css";
  import { ModeWatcher } from "mode-watcher";

  let { children } = $props();
</script>

<ModeWatcher defaultMode="dark" />

<svelte:head>
  <title>EXO</title>
  <meta name="description" content="EXO - Distributed AI Cluster Dashboard" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
  {@render children?.()}
</div>

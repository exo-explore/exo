<script lang="ts">
	import '../app.css';
	
	let { children } = $props();
</script>

<svelte:head>
	<title>CellHasher - Distributed AI</title>
	<meta name="description" content="CellHasher - Distributed AI Cluster Dashboard" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
	{@render children?.()}
</div>


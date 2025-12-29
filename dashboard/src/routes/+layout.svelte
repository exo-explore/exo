<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	
	let { children } = $props();
	let isPageHidden = $state(false);
	
	onMount(() => {
		if (!browser) return;
		
		// Listen for visibility changes to pause animations when hidden
		const handleVisibilityChange = () => {
			isPageHidden = document.visibilityState === 'hidden';
		};
		
		document.addEventListener('visibilitychange', handleVisibilityChange);
		
		return () => {
			document.removeEventListener('visibilitychange', handleVisibilityChange);
		};
	});
</script>

<svelte:head>
	<title>EXO</title>
	<meta name="description" content="EXO - Distributed AI Cluster Dashboard" />
</svelte:head>

<div class="min-h-screen bg-background text-foreground" data-page-hidden={isPageHidden}>
	{@render children?.()}
</div>


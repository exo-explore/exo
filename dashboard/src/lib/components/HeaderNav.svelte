<script lang="ts">
	import { browser } from '$app/environment';

	export let showHome = true;
	export let onHome: (() => void) | null = null;

	function handleHome(): void {
		if (onHome) {
			onHome();
			return;
		}
		if (browser) {
			window.location.hash = '/';
		}
	}
</script>

<header class="relative z-20 flex items-center justify-center px-6 pt-6 pb-4 bg-[#0f1116]/80 backdrop-blur-md border-b border-white/[0.06]">
	<!-- Center: CellHasher Logo + Branding -->
	<button
		onclick={handleHome}
		class="hover:opacity-80 transition-opacity {showHome ? 'cursor-pointer' : 'cursor-default'} flex items-center gap-3"
		title={showHome ? 'Go to home' : ''}
		disabled={!showHome}
	>
		<img src="/Cellhasher-logo.svg" alt="CellHasher" class="h-10 w-auto" />
		<div class="flex flex-col items-start">
			<span class="text-lg font-bold text-white tracking-tight">CellHasher</span>
			<span class="text-xs text-[#60a5fa] font-medium tracking-wider uppercase">Distributed AI</span>
		</div>
	</button>

	<!-- Right: Home + Downloads -->
	<div class="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-4">
		{#if showHome}
			<button
				onclick={handleHome}
				class="text-sm text-[#9ca3af] hover:text-[#60a5fa] transition-colors tracking-wider uppercase flex items-center gap-2 cursor-pointer"
				title="Back to topology view"
			>
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
				</svg>
				Home
			</button>
		{/if}
		<a
			href="/#/downloads"
			class="text-sm text-[#9ca3af] hover:text-[#60a5fa] transition-colors tracking-wider uppercase flex items-center gap-2 cursor-pointer"
			title="View downloads overview"
		>
			<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d="M12 3v12" />
				<path d="M7 12l5 5 5-5" />
				<path d="M5 21h14" />
			</svg>
			Downloads
		</a>
	</div>
</header>

<script lang="ts">
	import type { PrefillProgress } from '$lib/stores/app.svelte';

	interface Props {
		progress: PrefillProgress;
		class?: string;
	}

	let { progress, class: className = '' }: Props = $props();

	const percentage = $derived(
		progress.total > 0 ? Math.round((progress.processed / progress.total) * 100) : 0
	);

	function formatTokenCount(count: number): string {
		if (count >= 1000) {
			return `${(count / 1000).toFixed(1)}k`;
		}
		return count.toString();
	}
</script>

<div class="prefill-progress {className}">
	<div class="flex items-center justify-between text-xs text-gray-400 mb-1">
		<span class="flex items-center gap-1.5">
			<svg
				class="w-3.5 h-3.5 animate-spin"
				fill="none"
				viewBox="0 0 24 24"
				xmlns="http://www.w3.org/2000/svg"
			>
				<circle
					class="opacity-25"
					cx="12"
					cy="12"
					r="10"
					stroke="currentColor"
					stroke-width="4"
				></circle>
				<path
					class="opacity-75"
					fill="currentColor"
					d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
				></path>
			</svg>
			<span>Processing prompt</span>
		</span>
		<span class="font-mono">
			{formatTokenCount(progress.processed)} / {formatTokenCount(progress.total)} tokens
		</span>
	</div>
	<div class="h-1.5 bg-gray-700 rounded-full overflow-hidden">
		<div
			class="h-full bg-blue-500 rounded-full transition-all duration-150 ease-out"
			style="width: {percentage}%"
		></div>
	</div>
	<div class="text-right text-xs text-gray-500 mt-0.5">
		{percentage}%
	</div>
</div>

<style>
	.prefill-progress {
		width: 100%;
	}
</style>

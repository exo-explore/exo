<script lang="ts">
	import type { TokenData } from '$lib/stores/app.svelte';

	interface Props {
		tokens: TokenData[];
		class?: string;
	}

	let { tokens, class: className = '' }: Props = $props();

	// Tooltip state
	let hoveredToken = $state<{ token: TokenData; x: number; y: number } | null>(null);

	/**
	 * Get confidence level based on probability
	 * High: >0.8 (logprob > -0.22)
	 * Medium: 0.5-0.8 (logprob -0.69 to -0.22)
	 * Low: 0.2-0.5 (logprob -1.61 to -0.69)
	 * Very Low: <0.2 (logprob < -1.61)
	 */
	function getConfidenceClass(probability: number): string {
		if (probability > 0.8) return 'bg-green-500/30 text-green-100';
		if (probability > 0.5) return 'bg-yellow-500/30 text-yellow-100';
		if (probability > 0.2) return 'bg-orange-500/30 text-orange-100';
		return 'bg-red-500/40 text-red-100';
	}

	/**
	 * Get border color for token based on probability
	 */
	function getBorderClass(probability: number): string {
		if (probability > 0.8) return 'border-green-500/50';
		if (probability > 0.5) return 'border-yellow-500/50';
		if (probability > 0.2) return 'border-orange-500/50';
		return 'border-red-500/50';
	}

	function handleMouseEnter(event: MouseEvent, token: TokenData) {
		const rect = (event.target as HTMLElement).getBoundingClientRect();
		hoveredToken = {
			token,
			x: rect.left + rect.width / 2,
			y: rect.top - 10
		};
	}

	function handleMouseLeave() {
		hoveredToken = null;
	}

	function formatProbability(prob: number): string {
		return (prob * 100).toFixed(1) + '%';
	}

	function formatLogprob(logprob: number): string {
		return logprob.toFixed(3);
	}
</script>

<div class="token-heatmap leading-relaxed {className}">
	{#each tokens as tokenData, i (i)}
		<span
			role="button"
			tabindex="0"
			class="token-span inline rounded px-0.5 py-0.5 cursor-pointer transition-all duration-150 border {getConfidenceClass(tokenData.probability)} {getBorderClass(tokenData.probability)} hover:opacity-80"
			onmouseenter={(e) => handleMouseEnter(e, tokenData)}
			onmouseleave={handleMouseLeave}
		>{tokenData.token}</span>
	{/each}
</div>

<!-- Tooltip -->
{#if hoveredToken}
	<div
		class="fixed z-50 pointer-events-none"
		style="left: {hoveredToken.x}px; top: {hoveredToken.y}px; transform: translate(-50%, -100%);"
	>
		<div class="bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-3 text-sm min-w-48">
			<!-- Token info -->
			<div class="mb-2">
				<span class="text-gray-400 text-xs">Token:</span>
				<span class="text-white font-mono ml-1">"{hoveredToken.token.token}"</span>
				<span class="text-green-400 ml-2">{formatProbability(hoveredToken.token.probability)}</span>
			</div>

			<div class="text-gray-400 text-xs mb-1">
				logprob: <span class="text-gray-300 font-mono">{formatLogprob(hoveredToken.token.logprob)}</span>
			</div>

			<!-- Top alternatives -->
			{#if hoveredToken.token.topLogprobs.length > 0}
				<div class="border-t border-gray-700 mt-2 pt-2">
					<div class="text-gray-400 text-xs mb-1">Alternatives:</div>
					{#each hoveredToken.token.topLogprobs.slice(0, 5) as alt, idx (idx)}
						{@const altProb = Math.exp(alt.logprob)}
						<div class="flex justify-between items-center text-xs py-0.5">
							<span class="text-gray-300 font-mono truncate max-w-24">"{alt.token}"</span>
							<span class="text-gray-400 ml-2">{formatProbability(altProb)}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>
		<!-- Arrow -->
		<div class="absolute left-1/2 -translate-x-1/2 top-full">
			<div class="border-8 border-transparent border-t-gray-900"></div>
		</div>
	</div>
{/if}

<style>
	.token-heatmap {
		word-wrap: break-word;
		white-space: pre-wrap;
	}

	.token-span {
		margin: 0;
		border-width: 1px;
	}
</style>

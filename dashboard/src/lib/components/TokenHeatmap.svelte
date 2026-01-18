<script lang="ts">
	import type { TokenData } from '$lib/stores/app.svelte';

	interface Props {
		tokens: TokenData[];
		class?: string;
		isGenerating?: boolean;
		onRegenerateFrom?: (tokenIndex: number) => void;
	}

	let { tokens, class: className = '', isGenerating = false, onRegenerateFrom }: Props = $props();

	// Tooltip state - track both token data and index
	let hoveredTokenIndex = $state<number | null>(null);
	let hoveredPosition = $state<{ x: number; y: number } | null>(null);
	let isTooltipHovered = $state(false);
	let hideTimeoutId: ReturnType<typeof setTimeout> | null = null;

	// Derive the hovered token from the index (stable across re-renders)
	const hoveredToken = $derived(
		hoveredTokenIndex !== null && hoveredPosition && tokens[hoveredTokenIndex]
			? { token: tokens[hoveredTokenIndex], index: hoveredTokenIndex, ...hoveredPosition }
			: null
	);

	/**
	 * Get confidence styling based on probability.
	 * Following Apple design principles: high confidence tokens blend in,
	 * only uncertainty draws attention.
	 */
	function getConfidenceClass(probability: number): string {
		if (probability > 0.8) return 'text-inherit'; // Expected tokens - blend in
		if (probability > 0.5) return 'bg-gray-500/10 text-inherit'; // Slight hint
		if (probability > 0.2) return 'bg-amber-500/15 text-amber-200/90'; // Subtle warmth
		return 'bg-red-500/20 text-red-200/90'; // Draws attention
	}

	/**
	 * Get border/underline styling for uncertain tokens
	 */
	function getBorderClass(probability: number): string {
		if (probability > 0.8) return 'border-transparent'; // No border for expected
		if (probability > 0.5) return 'border-gray-500/20';
		if (probability > 0.2) return 'border-amber-500/30';
		return 'border-red-500/40';
	}

	function clearHideTimeout() {
		if (hideTimeoutId) {
			clearTimeout(hideTimeoutId);
			hideTimeoutId = null;
		}
	}

	function handleMouseEnter(event: MouseEvent, token: TokenData, index: number) {
		clearHideTimeout();
		const rect = (event.target as HTMLElement).getBoundingClientRect();
		hoveredTokenIndex = index;
		hoveredPosition = {
			x: rect.left + rect.width / 2,
			y: rect.top - 10
		};
	}

	function handleMouseLeave() {
		clearHideTimeout();
		// Use longer delay during generation to account for re-renders
		const delay = isGenerating ? 300 : 100;
		hideTimeoutId = setTimeout(() => {
			if (!isTooltipHovered) {
				hoveredTokenIndex = null;
				hoveredPosition = null;
			}
		}, delay);
	}

	function handleTooltipEnter() {
		clearHideTimeout();
		isTooltipHovered = true;
	}

	function handleTooltipLeave() {
		isTooltipHovered = false;
		hoveredTokenIndex = null;
		hoveredPosition = null;
	}

	function handleRegenerate() {
		if (hoveredToken && onRegenerateFrom) {
			const indexToRegenerate = hoveredToken.index;
			// Clear hover state immediately
			hoveredTokenIndex = null;
			hoveredPosition = null;
			isTooltipHovered = false;
			// Call regenerate
			onRegenerateFrom(indexToRegenerate);
		}
	}

	function formatProbability(prob: number): string {
		return (prob * 100).toFixed(1) + '%';
	}

	function formatLogprob(logprob: number): string {
		return logprob.toFixed(3);
	}

	function getProbabilityColor(probability: number): string {
		if (probability > 0.8) return 'text-gray-300';
		if (probability > 0.5) return 'text-gray-400';
		if (probability > 0.2) return 'text-amber-400';
		return 'text-red-400';
	}
</script>

<div class="token-heatmap leading-relaxed {className}">
	{#each tokens as tokenData, i (i)}
		<span
			role="button"
			tabindex="0"
			class="token-span inline rounded px-0.5 py-0.5 cursor-pointer transition-all duration-150 border {getConfidenceClass(tokenData.probability)} {getBorderClass(tokenData.probability)} hover:opacity-80"
			onmouseenter={(e) => handleMouseEnter(e, tokenData, i)}
			onmouseleave={handleMouseLeave}
		>{tokenData.token}</span>
	{/each}
</div>

<!-- Tooltip -->
{#if hoveredToken}
	<div
		class="fixed z-50"
		style="left: {hoveredToken.x}px; top: {hoveredToken.y}px; transform: translate(-50%, -100%);"
		onmouseenter={handleTooltipEnter}
		onmouseleave={handleTooltipLeave}
	>
		<div class="bg-gray-900/95 backdrop-blur-sm border border-gray-700/50 rounded-xl shadow-xl p-3 text-sm min-w-48">
			<!-- Token info -->
			<div class="mb-2">
				<span class="text-gray-500 text-xs">Token:</span>
				<span class="text-white font-mono ml-1">"{hoveredToken.token.token}"</span>
				<span class="{getProbabilityColor(hoveredToken.token.probability)} ml-2">{formatProbability(hoveredToken.token.probability)}</span>
			</div>

			<div class="text-gray-400 text-xs mb-1">
				logprob: <span class="text-gray-300 font-mono">{formatLogprob(hoveredToken.token.logprob)}</span>
			</div>

			<!-- Top alternatives -->
			{#if hoveredToken.token.topLogprobs.length > 0}
				<div class="border-t border-gray-700/50 mt-2 pt-2">
					<div class="text-gray-500 text-xs mb-1">Alternatives:</div>
					{#each hoveredToken.token.topLogprobs.slice(0, 5) as alt, idx (idx)}
						{@const altProb = Math.exp(alt.logprob)}
						<div class="flex justify-between items-center text-xs py-0.5">
							<span class="text-gray-300 font-mono truncate max-w-24">"{alt.token}"</span>
							<span class="text-gray-400 ml-2">{formatProbability(altProb)}</span>
						</div>
					{/each}
				</div>
			{/if}

			<!-- Regenerate button -->
			{#if onRegenerateFrom}
				<button
					onclick={handleRegenerate}
					class="w-full mt-2 pt-2 border-t border-gray-700/50 flex items-center justify-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors cursor-pointer"
				>
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
					</svg>
					Regenerate from here
				</button>
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

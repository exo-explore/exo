<script lang="ts">
	interface HuggingFaceModel {
		id: string;
		author: string;
		downloads: number;
		likes: number;
		last_modified: string;
		tags: string[];
	}

	type HuggingFaceResultItemProps = {
		model: HuggingFaceModel;
		isAdded: boolean;
		isAdding: boolean;
		onAdd: () => void;
		onSelect: () => void;
	};

	let { model, isAdded, isAdding, onAdd, onSelect }: HuggingFaceResultItemProps = $props();

	function formatNumber(num: number): string {
		if (num >= 1000000) {
			return `${(num / 1000000).toFixed(1)}M`;
		} else if (num >= 1000) {
			return `${(num / 1000).toFixed(1)}k`;
		}
		return num.toString();
	}

	// Extract model name from full ID (e.g., "mlx-community/Llama-3.2-1B" -> "Llama-3.2-1B")
	const modelName = $derived(model.id.split('/').pop() || model.id);
</script>

<div class="flex items-center justify-between gap-3 px-3 py-2.5 hover:bg-white/5 transition-colors border-b border-white/5 last:border-b-0">
	<div class="flex-1 min-w-0">
		<div class="flex items-center gap-2">
			<span class="text-sm font-mono text-white truncate" title={model.id}>{modelName}</span>
			{#if isAdded}
				<span class="px-1.5 py-0.5 text-[10px] font-mono bg-green-500/20 text-green-400 rounded">Added</span>
			{/if}
		</div>
		<div class="flex items-center gap-3 mt-0.5 text-xs text-white/40">
			<span class="truncate">{model.author}</span>
			<span class="flex items-center gap-1 shrink-0" title="Downloads">
				<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
				</svg>
				{formatNumber(model.downloads)}
			</span>
			<span class="flex items-center gap-1 shrink-0" title="Likes">
				<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
				</svg>
				{formatNumber(model.likes)}
			</span>
		</div>
	</div>

	<div class="flex items-center gap-2 shrink-0">
		{#if isAdded}
			<button
				type="button"
				onclick={onSelect}
				class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-exo-yellow/10 text-exo-yellow border border-exo-yellow/30 hover:bg-exo-yellow/20 transition-colors rounded"
			>
				Select
			</button>
		{:else}
			<button
				type="button"
				onclick={onAdd}
				disabled={isAdding}
				class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-orange-500/10 text-orange-400 border border-orange-400/30 hover:bg-orange-500/20 transition-colors rounded disabled:opacity-50 disabled:cursor-not-allowed"
			>
				{#if isAdding}
					<span class="flex items-center gap-1.5">
						<span class="w-3 h-3 border-2 border-orange-400 border-t-transparent rounded-full animate-spin"></span>
						Adding...
					</span>
				{:else}
					+ Add
				{/if}
			</button>
		{/if}
	</div>
</div>

<script lang="ts">
	interface ModelInfo {
		id: string;
		name?: string;
		storage_size_megabytes?: number;
		base_model?: string;
		base_model_name?: string;
		quantization?: string;
		architecture?: string;
		supports_tensor?: boolean;
		tagline?: string;
		capabilities?: string[];
		family?: string;
	}

	interface ModelGroup {
		id: string;
		name: string;
		tagline: string;
		capabilities: string[];
		family: string;
		variants: ModelInfo[];
		smallestVariant: ModelInfo;
		hasMultipleVariants: boolean;
	}

	type ModelPickerGroupProps = {
		group: ModelGroup;
		isExpanded: boolean;
		isFavorite: boolean;
		selectedModelId: string | null;
		canModelFit: (id: string) => boolean;
		onToggleExpand: () => void;
		onSelectModel: (modelId: string) => void;
		onToggleFavorite: (baseModelId: string) => void;
		onShowInfo: (group: ModelGroup) => void;
	};

	let {
		group,
		isExpanded,
		isFavorite,
		selectedModelId,
		canModelFit,
		onToggleExpand,
		onSelectModel,
		onToggleFavorite,
		onShowInfo,
	}: ModelPickerGroupProps = $props();

	// Format storage size
	function formatSize(mb: number | undefined): string {
		if (!mb) return '';
		if (mb >= 1024) {
			return `${(mb / 1024).toFixed(0)}GB`;
		}
		return `${mb}MB`;
	}

	// Get capability icon
	function getCapabilityIcon(cap: string): string {
		switch (cap) {
			case 'thinking':
				return '&#129504;'; // brain
			case 'code':
				return '&#128187;'; // laptop
			case 'vision':
				return '&#128248;'; // camera
			case 'image_gen':
				return '&#127912;'; // art palette
			default:
				return '';
		}
	}

	// Check if model name includes "Thinking"
	const isThinkingModel = $derived(group.name.toLowerCase().includes('thinking') || group.capabilities.includes('thinking'));

	// Check if the smallest variant can fit (used for single-variant groups or to indicate group fitness)
	const smallestCanFit = $derived(canModelFit(group.smallestVariant?.id || group.variants[0]?.id || ''));
	// Check if any variant can fit
	const anyVariantFits = $derived(group.variants.some(v => canModelFit(v.id)));
</script>

<div class="border-b border-white/5 last:border-b-0 {!anyVariantFits ? 'opacity-50' : ''}">
	<!-- Main row -->
	<div
		class="flex items-center gap-2 px-3 py-2.5 transition-colors {anyVariantFits ? 'hover:bg-white/5 cursor-pointer' : 'cursor-not-allowed'}"
		onclick={() => {
			if (group.hasMultipleVariants) {
				onToggleExpand();
			} else {
				const modelId = group.variants[0]?.id;
				if (modelId && canModelFit(modelId)) {
					onSelectModel(modelId);
				}
			}
		}}
		role="button"
		tabindex="0"
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				if (group.hasMultipleVariants) {
					onToggleExpand();
				} else {
					const modelId = group.variants[0]?.id;
					if (modelId && canModelFit(modelId)) {
						onSelectModel(modelId);
					}
				}
			}
		}}
	>
		<!-- Expand/collapse chevron (for groups with variants) -->
		{#if group.hasMultipleVariants}
			<svg
				class="w-4 h-4 text-white/40 transition-transform duration-200 flex-shrink-0 {isExpanded ? 'rotate-90' : ''}"
				viewBox="0 0 24 24"
				fill="currentColor"
			>
				<path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z" />
			</svg>
		{:else}
			<div class="w-4 flex-shrink-0"></div>
		{/if}

		<!-- Model name and tagline -->
		<div class="flex-1 min-w-0">
			<div class="flex items-center gap-2">
				<span class="font-mono text-sm text-white truncate">
					{group.name}
				</span>
				{#if isThinkingModel}
					<span class="text-xs text-purple-400 font-mono">(Thinking)</span>
				{/if}
				<!-- Capability badges -->
				{#each group.capabilities.filter(c => c !== 'text') as cap}
					<span
						class="text-xs text-white/40"
						title={cap}
					>
						{@html getCapabilityIcon(cap)}
					</span>
				{/each}
			</div>
			{#if group.tagline}
				<p class="font-mono text-xs text-white/40 truncate mt-0.5">{group.tagline}</p>
			{/if}
		</div>

		<!-- Size indicator (smallest variant) -->
		{#if !group.hasMultipleVariants && group.smallestVariant?.storage_size_megabytes}
			<span class="text-xs font-mono text-white/30 flex-shrink-0">
				{formatSize(group.smallestVariant.storage_size_megabytes)}
			</span>
		{/if}

		<!-- Variant count -->
		{#if group.hasMultipleVariants}
			<span class="text-xs font-mono text-white/30 flex-shrink-0">
				{group.variants.length} variants
			</span>
		{/if}

		<!-- Too large indicator -->
		{#if !anyVariantFits}
			<span class="text-xs font-mono text-red-400/70 flex-shrink-0">Too large</span>
		{/if}

		<!-- Favorite star -->
		<button
			type="button"
			class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0"
			onclick={(e) => {
				e.stopPropagation();
				onToggleFavorite(group.id);
			}}
			title={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
		>
			{#if isFavorite}
				<svg class="w-4 h-4 text-amber-400" viewBox="0 0 24 24" fill="currentColor">
					<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
				</svg>
			{:else}
				<svg class="w-4 h-4 text-white/30 hover:text-white/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
				</svg>
			{/if}
		</button>

		<!-- Info button -->
		<button
			type="button"
			class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0"
			onclick={(e) => {
				e.stopPropagation();
				onShowInfo(group);
			}}
			title="Model info"
		>
			<svg class="w-4 h-4 text-white/30 hover:text-white/50" viewBox="0 0 24 24" fill="currentColor">
				<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
			</svg>
		</button>
	</div>

	<!-- Expanded variants -->
	{#if isExpanded && group.hasMultipleVariants}
		<div class="bg-black/20 border-t border-white/5">
			{#each group.variants as variant}
				{@const modelCanFit = canModelFit(variant.id)}
				{@const isSelected = selectedModelId === variant.id}
				<button
					type="button"
					class="w-full flex items-center gap-3 px-3 py-2 pl-10 hover:bg-white/5 transition-colors text-left {!modelCanFit ? 'opacity-50 cursor-not-allowed' : ''} {isSelected ? 'bg-exo-yellow/10 border-l-2 border-exo-yellow' : 'border-l-2 border-transparent'}"
					disabled={!modelCanFit}
					onclick={() => {
						if (modelCanFit) {
							onSelectModel(variant.id);
						}
					}}
				>
					<!-- Quantization badge -->
					<span class="text-xs font-mono px-1.5 py-0.5 rounded bg-white/10 text-white/70 flex-shrink-0">
						{variant.quantization || 'default'}
					</span>

					<!-- Size -->
					<span class="text-xs font-mono text-white/40 flex-1">
						{formatSize(variant.storage_size_megabytes)}
					</span>

					<!-- Check mark if selected -->
					{#if isSelected}
						<svg class="w-4 h-4 text-exo-yellow" viewBox="0 0 24 24" fill="currentColor">
							<path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
						</svg>
					{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>

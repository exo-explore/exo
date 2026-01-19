<script lang="ts">
	import { fade, fly } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';
	import FamilySidebar from './FamilySidebar.svelte';
	import ModelPickerGroup from './ModelPickerGroup.svelte';
	import ModelFilterPopover from './ModelFilterPopover.svelte';
	import HuggingFaceResultItem from './HuggingFaceResultItem.svelte';

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

	interface FilterState {
		capabilities: string[];
		sizeRange: { min: number; max: number } | null;
	}

	interface HuggingFaceModel {
		id: string;
		author: string;
		downloads: number;
		likes: number;
		last_modified: string;
		tags: string[];
	}

	type ModelPickerModalProps = {
		isOpen: boolean;
		models: ModelInfo[];
		selectedModelId: string | null;
		favorites: Set<string>;
		existingModelIds: Set<string>;
		canModelFit: (modelId: string) => boolean;
		onSelect: (modelId: string) => void;
		onClose: () => void;
		onToggleFavorite: (baseModelId: string) => void;
		onAddModel: (modelId: string) => Promise<void>;
	};

	let {
		isOpen,
		models,
		selectedModelId,
		favorites,
		existingModelIds,
		canModelFit,
		onSelect,
		onClose,
		onToggleFavorite,
		onAddModel,
	}: ModelPickerModalProps = $props();

	// Local state
	let searchQuery = $state('');
	let selectedFamily = $state<string | null>(null);
	let expandedGroups = $state<Set<string>>(new Set());
	let showFilters = $state(false);
	let filters = $state<FilterState>({ capabilities: [], sizeRange: null });
	let infoGroup = $state<ModelGroup | null>(null);

	// HuggingFace Hub state
	let hfSearchQuery = $state('');
	let hfSearchResults = $state<HuggingFaceModel[]>([]);
	let hfTrendingModels = $state<HuggingFaceModel[]>([]);
	let hfIsSearching = $state(false);
	let hfIsLoadingTrending = $state(false);
	let addingModelId = $state<string | null>(null);
	let hfSearchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
	let manualModelId = $state('');
	let addModelError = $state<string | null>(null);

	// Reset state when modal opens
	$effect(() => {
		if (isOpen) {
			searchQuery = '';
			selectedFamily = null;
			expandedGroups = new Set();
			showFilters = false;
			// Reset HuggingFace state
			hfSearchQuery = '';
			hfSearchResults = [];
			manualModelId = '';
			addModelError = null;
		}
	});

	// Fetch trending models when HuggingFace is selected
	$effect(() => {
		if (selectedFamily === 'huggingface' && hfTrendingModels.length === 0 && !hfIsLoadingTrending) {
			fetchTrendingModels();
		}
	});

	async function fetchTrendingModels() {
		hfIsLoadingTrending = true;
		try {
			const response = await fetch('/models/search?query=&limit=20');
			if (response.ok) {
				hfTrendingModels = await response.json();
			}
		} catch (error) {
			console.error('Failed to fetch trending models:', error);
		} finally {
			hfIsLoadingTrending = false;
		}
	}

	async function searchHuggingFace(query: string) {
		if (query.length < 2) {
			hfSearchResults = [];
			return;
		}

		hfIsSearching = true;
		try {
			const response = await fetch(`/models/search?query=${encodeURIComponent(query)}&limit=20`);
			if (response.ok) {
				hfSearchResults = await response.json();
			} else {
				hfSearchResults = [];
			}
		} catch (error) {
			console.error('Failed to search models:', error);
			hfSearchResults = [];
		} finally {
			hfIsSearching = false;
		}
	}

	function handleHfSearchInput(query: string) {
		hfSearchQuery = query;
		addModelError = null;

		if (hfSearchDebounceTimer) {
			clearTimeout(hfSearchDebounceTimer);
		}

		if (query.length >= 2) {
			hfSearchDebounceTimer = setTimeout(() => {
				searchHuggingFace(query);
			}, 300);
		} else {
			hfSearchResults = [];
		}
	}

	async function handleAddModel(modelId: string) {
		addingModelId = modelId;
		addModelError = null;
		try {
			await onAddModel(modelId);
		} catch (error) {
			addModelError = error instanceof Error ? error.message : 'Failed to add model';
		} finally {
			addingModelId = null;
		}
	}

	async function handleAddManualModel() {
		if (!manualModelId.trim()) return;
		await handleAddModel(manualModelId.trim());
		if (!addModelError) {
			manualModelId = '';
		}
	}

	function handleSelectHfModel(modelId: string) {
		onSelect(modelId);
		onClose();
	}

	// Models to display in HuggingFace view
	const hfDisplayModels = $derived.by((): HuggingFaceModel[] => {
		if (hfSearchQuery.length >= 2) {
			return hfSearchResults;
		}
		return hfTrendingModels;
	});

	// Group models by base_model
	const groupedModels = $derived.by((): ModelGroup[] => {
		const groups = new Map<string, ModelGroup>();

		for (const model of models) {
			const groupId = model.base_model || model.id;
			const groupName = model.base_model_name || model.name || model.id;

			if (!groups.has(groupId)) {
				groups.set(groupId, {
					id: groupId,
					name: groupName,
					tagline: model.tagline || '',
					capabilities: model.capabilities || ['text'],
					family: model.family || '',
					variants: [],
					smallestVariant: model,
					hasMultipleVariants: false,
				});
			}

			const group = groups.get(groupId)!;
			group.variants.push(model);

			// Track smallest variant
			if ((model.storage_size_megabytes || 0) < (group.smallestVariant.storage_size_megabytes || Infinity)) {
				group.smallestVariant = model;
			}

			// Update tagline/capabilities if not set
			if (!group.tagline && model.tagline) {
				group.tagline = model.tagline;
			}
			if (group.capabilities.length <= 1 && model.capabilities && model.capabilities.length > 1) {
				group.capabilities = model.capabilities;
			}
			if (!group.family && model.family) {
				group.family = model.family;
			}
		}

		// Sort variants within each group by size
		for (const group of groups.values()) {
			group.variants.sort((a, b) => (a.storage_size_megabytes || 0) - (b.storage_size_megabytes || 0));
			group.hasMultipleVariants = group.variants.length > 1;
		}

		// Convert to array and sort by smallest variant size (biggest first)
		return Array.from(groups.values()).sort((a, b) => {
			return (b.smallestVariant.storage_size_megabytes || 0) - (a.smallestVariant.storage_size_megabytes || 0);
		});
	});

	// Get unique families
	const uniqueFamilies = $derived.by((): string[] => {
		const families = new Set<string>();
		for (const group of groupedModels) {
			if (group.family) {
				families.add(group.family);
			}
		}
		// Sort families with common ones first
		const familyOrder = ['llama', 'qwen', 'deepseek', 'gpt-oss', 'glm', 'minimax', 'kimi'];
		return Array.from(families).sort((a, b) => {
			const aIdx = familyOrder.indexOf(a);
			const bIdx = familyOrder.indexOf(b);
			if (aIdx === -1 && bIdx === -1) return a.localeCompare(b);
			if (aIdx === -1) return 1;
			if (bIdx === -1) return -1;
			return aIdx - bIdx;
		});
	});

	// Filter models based on search, family, and filters
	const filteredGroups = $derived.by((): ModelGroup[] => {
		let result: ModelGroup[] = [...groupedModels];

		// Filter by family
		if (selectedFamily === 'favorites') {
			result = result.filter((g) => favorites.has(g.id));
		} else if (selectedFamily) {
			result = result.filter((g) => g.family === selectedFamily);
		}

		// Filter by search query
		if (searchQuery.trim()) {
			const query = searchQuery.toLowerCase().trim();
			result = result.filter(
				(g) =>
					g.name.toLowerCase().includes(query) ||
					g.tagline.toLowerCase().includes(query) ||
					g.variants.some((v) => v.id.toLowerCase().includes(query))
			);
		}

		// Filter by capabilities
		if (filters.capabilities.length > 0) {
			result = result.filter((g) =>
				filters.capabilities.every((cap) => g.capabilities.includes(cap))
			);
		}

		// Filter by size range
		if (filters.sizeRange) {
			const { min, max } = filters.sizeRange;
			result = result.filter((g) => {
				const sizeGB = (g.smallestVariant.storage_size_megabytes || 0) / 1024;
				return sizeGB >= min && sizeGB <= max;
			});
		}

		// Sort: models that fit first, then by size (largest first)
		result.sort((a, b) => {
			const aFits = a.variants.some((v) => canModelFit(v.id));
			const bFits = b.variants.some((v) => canModelFit(v.id));

			// Models that fit come first
			if (aFits && !bFits) return -1;
			if (!aFits && bFits) return 1;

			// Within each group, sort by size (largest first)
			return (b.smallestVariant.storage_size_megabytes || 0) - (a.smallestVariant.storage_size_megabytes || 0);
		});

		return result;
	});

	// Check if any favorites exist
	const hasFavorites = $derived(favorites.size > 0);

	function toggleGroupExpanded(groupId: string) {
		const next = new Set(expandedGroups);
		if (next.has(groupId)) {
			next.delete(groupId);
		} else {
			next.add(groupId);
		}
		expandedGroups = next;
	}

	function handleSelect(modelId: string) {
		onSelect(modelId);
		onClose();
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			onClose();
		}
	}

	function handleFiltersChange(newFilters: FilterState) {
		filters = newFilters;
	}

	function clearFilters() {
		filters = { capabilities: [], sizeRange: null };
	}

	const hasActiveFilters = $derived(filters.capabilities.length > 0 || filters.sizeRange !== null);
</script>

<svelte:window onkeydown={handleKeydown} />

{#if isOpen}
	<!-- Backdrop -->
	<div
		class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm"
		transition:fade={{ duration: 200 }}
		onclick={onClose}
		role="presentation"
	></div>

	<!-- Modal -->
	<div
		class="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(90vw,600px)] h-[min(80vh,700px)] bg-exo-dark-gray border border-white/10 rounded-lg shadow-2xl overflow-hidden flex flex-col"
		transition:fly={{ y: 20, duration: 300, easing: cubicOut }}
		role="dialog"
		aria-modal="true"
		aria-label="Select a model"
	>
		<!-- Header with search -->
		<div class="flex items-center gap-2 p-3 border-b border-white/10 bg-exo-medium-gray/30">
			{#if selectedFamily === 'huggingface'}
				<!-- HuggingFace search -->
				<svg class="w-5 h-5 text-orange-400/60 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<circle cx="11" cy="11" r="8" />
					<path d="M21 21l-4.35-4.35" />
				</svg>
				<input
					type="search"
					class="flex-1 bg-transparent border-none outline-none text-sm font-mono text-white placeholder-white/40"
					placeholder="Search mlx-community models..."
					value={hfSearchQuery}
					oninput={(e) => handleHfSearchInput(e.currentTarget.value)}
				/>
				{#if hfIsSearching}
					<div class="flex-shrink-0">
						<span class="w-4 h-4 border-2 border-orange-400 border-t-transparent rounded-full animate-spin block"></span>
					</div>
				{/if}
			{:else}
				<!-- Normal model search -->
				<svg class="w-5 h-5 text-white/40 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<circle cx="11" cy="11" r="8" />
					<path d="M21 21l-4.35-4.35" />
				</svg>
				<input
					type="search"
					class="flex-1 bg-transparent border-none outline-none text-sm font-mono text-white placeholder-white/40"
					placeholder="Search models..."
					bind:value={searchQuery}
				/>
				<!-- Filter button -->
				<div class="relative">
					<button
						type="button"
						class="p-1.5 rounded hover:bg-white/10 transition-colors {hasActiveFilters ? 'text-exo-yellow' : 'text-white/50'}"
						onclick={() => showFilters = !showFilters}
						title="Filters"
					>
						<svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
							<path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z" />
						</svg>
					</button>
					{#if showFilters}
						<ModelFilterPopover
							{filters}
							onChange={handleFiltersChange}
							onClear={clearFilters}
							onClose={() => showFilters = false}
						/>
					{/if}
				</div>
			{/if}
			<!-- Close button -->
			<button
				type="button"
				class="p-1.5 rounded hover:bg-white/10 transition-colors text-white/50 hover:text-white/70"
				onclick={onClose}
				title="Close"
			>
				<svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
					<path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z" />
				</svg>
			</button>
		</div>

		<!-- Body -->
		<div class="flex flex-1 overflow-hidden">
			<!-- Family sidebar -->
			<FamilySidebar
				families={uniqueFamilies}
				{selectedFamily}
				{hasFavorites}
				onSelect={(family) => selectedFamily = family}
			/>

			<!-- Model list -->
			<div class="flex-1 overflow-y-auto flex flex-col">
				{#if selectedFamily === 'huggingface'}
					<!-- HuggingFace Hub view -->
					<div class="flex-1 flex flex-col min-h-0">
						<!-- Section header -->
						<div class="sticky top-0 z-10 px-3 py-2 bg-exo-dark-gray/95 border-b border-white/10">
							<span class="text-xs font-mono text-white/40">
								{#if hfSearchQuery.length >= 2}
									Search results for "{hfSearchQuery}"
								{:else}
									Trending on mlx-community
								{/if}
							</span>
						</div>

						<!-- Results list -->
						<div class="flex-1 overflow-y-auto">
							{#if hfIsLoadingTrending && hfTrendingModels.length === 0}
								<div class="flex items-center justify-center py-12 text-white/40">
									<span class="w-5 h-5 border-2 border-orange-400 border-t-transparent rounded-full animate-spin mr-2"></span>
									<span class="font-mono text-sm">Loading trending models...</span>
								</div>
							{:else if hfDisplayModels.length === 0}
								<div class="flex flex-col items-center justify-center py-12 text-white/40">
									<svg class="w-10 h-10 mb-2" viewBox="0 0 24 24" fill="currentColor">
										<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 13.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm4 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm2-4.5H8c0-2.21 1.79-4 4-4s4 1.79 4 4z" />
									</svg>
									<p class="font-mono text-sm">No models found</p>
									{#if hfSearchQuery}
										<p class="font-mono text-xs mt-1">Try a different search term</p>
									{/if}
								</div>
							{:else}
								{#each hfDisplayModels as model}
									<HuggingFaceResultItem
										{model}
										isAdded={existingModelIds.has(model.id)}
										isAdding={addingModelId === model.id}
										onAdd={() => handleAddModel(model.id)}
										onSelect={() => handleSelectHfModel(model.id)}
									/>
								{/each}
							{/if}
						</div>

						<!-- Manual input footer -->
						<div class="sticky bottom-0 border-t border-white/10 bg-exo-dark-gray p-3">
							{#if addModelError}
								<div class="bg-red-500/10 border border-red-500/30 rounded px-3 py-2 mb-2">
									<p class="text-red-400 text-xs font-mono break-words">{addModelError}</p>
								</div>
							{/if}
							<div class="flex gap-2">
								<input
									type="text"
									class="flex-1 bg-exo-black/60 border border-white/10 rounded px-3 py-1.5 text-xs font-mono text-white placeholder-white/30 focus:outline-none focus:border-orange-400/50"
									placeholder="Or paste model ID directly..."
									bind:value={manualModelId}
									onkeydown={(e) => { if (e.key === 'Enter') handleAddManualModel(); }}
								/>
								<button
									type="button"
									onclick={handleAddManualModel}
									disabled={!manualModelId.trim() || addingModelId !== null}
									class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-orange-500/10 text-orange-400 border border-orange-400/30 hover:bg-orange-500/20 transition-colors rounded disabled:opacity-50 disabled:cursor-not-allowed"
								>
									Add
								</button>
							</div>
						</div>
					</div>
				{:else if filteredGroups.length === 0}
					<div class="flex flex-col items-center justify-center h-full text-white/40 p-8">
						<svg class="w-12 h-12 mb-3" viewBox="0 0 24 24" fill="currentColor">
							<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
						</svg>
						<p class="font-mono text-sm">No models found</p>
						{#if hasActiveFilters || searchQuery}
							<button
								type="button"
								class="mt-2 text-xs text-exo-yellow hover:underline"
								onclick={() => {
									searchQuery = '';
									clearFilters();
								}}
							>
								Clear filters
							</button>
						{/if}
					</div>
				{:else}
					{#each filteredGroups as group}
						<ModelPickerGroup
							{group}
							isExpanded={expandedGroups.has(group.id)}
							isFavorite={favorites.has(group.id)}
							{selectedModelId}
							{canModelFit}
							onToggleExpand={() => toggleGroupExpanded(group.id)}
							onSelectModel={handleSelect}
							{onToggleFavorite}
							onShowInfo={(g) => infoGroup = g}
						/>
					{/each}
				{/if}
			</div>
		</div>

		<!-- Footer with active filters indicator -->
		{#if hasActiveFilters}
			<div class="flex items-center gap-2 px-3 py-2 border-t border-white/10 bg-exo-medium-gray/20 text-xs font-mono text-white/50">
				<span>Filters:</span>
				{#each filters.capabilities as cap}
					<span class="px-1.5 py-0.5 bg-exo-yellow/20 text-exo-yellow rounded">{cap}</span>
				{/each}
				{#if filters.sizeRange}
					<span class="px-1.5 py-0.5 bg-exo-yellow/20 text-exo-yellow rounded">
						{filters.sizeRange.min}GB - {filters.sizeRange.max}GB
					</span>
				{/if}
				<button
					type="button"
					class="ml-auto text-white/40 hover:text-white/60"
					onclick={clearFilters}
				>
					Clear all
				</button>
			</div>
		{/if}
	</div>

	<!-- Info modal -->
	{#if infoGroup}
		<div
			class="fixed inset-0 z-60 bg-black/60"
			transition:fade={{ duration: 150 }}
			onclick={() => infoGroup = null}
			role="presentation"
		></div>
		<div
			class="fixed z-60 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(80vw,400px)] bg-exo-dark-gray border border-white/10 rounded-lg shadow-2xl p-4"
			transition:fly={{ y: 10, duration: 200, easing: cubicOut }}
			role="dialog"
			aria-modal="true"
		>
			<div class="flex items-start justify-between mb-3">
				<h3 class="font-mono text-lg text-white">{infoGroup.name}</h3>
				<button
					type="button"
					class="p-1 rounded hover:bg-white/10 transition-colors text-white/50"
					onclick={() => infoGroup = null}
					title="Close"
					aria-label="Close info dialog"
				>
					<svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
						<path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z" />
					</svg>
				</button>
			</div>
			{#if infoGroup.tagline}
				<p class="font-mono text-sm text-white/60 mb-3">{infoGroup.tagline}</p>
			{/if}
			<div class="space-y-2 text-xs font-mono">
				<div class="flex items-center gap-2">
					<span class="text-white/40">Family:</span>
					<span class="text-white/70">{infoGroup.family || 'Unknown'}</span>
				</div>
				<div class="flex items-center gap-2">
					<span class="text-white/40">Capabilities:</span>
					<span class="text-white/70">{infoGroup.capabilities.join(', ')}</span>
				</div>
				<div class="flex items-center gap-2">
					<span class="text-white/40">Variants:</span>
					<span class="text-white/70">{infoGroup.variants.length}</span>
				</div>
				{#if infoGroup.variants.length > 0}
					<div class="mt-3 pt-3 border-t border-white/10">
						<span class="text-white/40">Available quantizations:</span>
						<div class="flex flex-wrap gap-1 mt-1">
							{#each infoGroup.variants as variant}
								<span class="px-1.5 py-0.5 bg-white/10 text-white/60 rounded text-[10px]">
									{variant.quantization || 'default'} ({Math.round((variant.storage_size_megabytes || 0) / 1024)}GB)
								</span>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	{/if}
{/if}

<script lang="ts">
	import { fly } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';

	interface FilterState {
		capabilities: string[];
		sizeRange: { min: number; max: number } | null;
	}

	type ModelFilterPopoverProps = {
		filters: FilterState;
		onChange: (filters: FilterState) => void;
		onClear: () => void;
		onClose: () => void;
	};

	let { filters, onChange, onClear, onClose }: ModelFilterPopoverProps = $props();

	// Available capabilities
	const availableCapabilities = [
		{ id: 'text', label: 'Text', icon: '&#128172;' }, // speech bubble
		{ id: 'thinking', label: 'Thinking', icon: '&#129504;' }, // brain
		{ id: 'code', label: 'Code', icon: '&#128187;' }, // laptop
		{ id: 'vision', label: 'Vision', icon: '&#128248;' }, // camera
	];

	// Size ranges
	const sizeRanges = [
		{ label: '< 10GB', min: 0, max: 10 },
		{ label: '10-50GB', min: 10, max: 50 },
		{ label: '50-200GB', min: 50, max: 200 },
		{ label: '> 200GB', min: 200, max: 10000 },
	];

	function toggleCapability(cap: string) {
		const next = filters.capabilities.includes(cap)
			? filters.capabilities.filter((c) => c !== cap)
			: [...filters.capabilities, cap];
		onChange({ ...filters, capabilities: next });
	}

	function selectSizeRange(range: { min: number; max: number } | null) {
		// Toggle off if same range is clicked
		if (filters.sizeRange && range && filters.sizeRange.min === range.min && filters.sizeRange.max === range.max) {
			onChange({ ...filters, sizeRange: null });
		} else {
			onChange({ ...filters, sizeRange: range });
		}
	}

	function handleClickOutside(e: MouseEvent) {
		const target = e.target as HTMLElement;
		if (!target.closest('.filter-popover')) {
			onClose();
		}
	}
</script>

<svelte:window onclick={handleClickOutside} />

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="filter-popover absolute right-0 top-full mt-2 w-64 bg-exo-dark-gray border border-white/10 rounded-lg shadow-xl z-10"
	transition:fly={{ y: -10, duration: 200, easing: cubicOut }}
	onclick={(e) => e.stopPropagation()}
	role="dialog"
	aria-label="Filter options"
>
	<div class="p-3 space-y-4">
		<!-- Capabilities -->
		<div>
			<h4 class="text-xs font-mono text-white/50 mb-2">Capabilities</h4>
			<div class="flex flex-wrap gap-1.5">
				{#each availableCapabilities as cap}
					{@const isSelected = filters.capabilities.includes(cap.id)}
					<button
						type="button"
						class="px-2 py-1 text-xs font-mono rounded transition-colors {isSelected ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/30' : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'}"
						onclick={() => toggleCapability(cap.id)}
					>
						<span>{@html cap.icon}</span>
						<span class="ml-1">{cap.label}</span>
					</button>
				{/each}
			</div>
		</div>

		<!-- Size range -->
		<div>
			<h4 class="text-xs font-mono text-white/50 mb-2">Model Size</h4>
			<div class="flex flex-wrap gap-1.5">
				{#each sizeRanges as range}
					{@const isSelected = filters.sizeRange && filters.sizeRange.min === range.min && filters.sizeRange.max === range.max}
					<button
						type="button"
						class="px-2 py-1 text-xs font-mono rounded transition-colors {isSelected ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/30' : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'}"
						onclick={() => selectSizeRange(range)}
					>
						{range.label}
					</button>
				{/each}
			</div>
		</div>

		<!-- Clear button -->
		<button
			type="button"
			class="w-full py-1.5 text-xs font-mono text-white/50 hover:text-white/70 hover:bg-white/5 rounded transition-colors"
			onclick={onClear}
		>
			Clear all filters
		</button>
	</div>
</div>

<script lang="ts">
	import {
		imageGenerationParams,
		setImageGenerationParams,
		resetImageGenerationParams,
		type ImageGenerationParams,
	} from "$lib/stores/app.svelte";

	let showAdvanced = $state(false);

	const params = $derived(imageGenerationParams());

	const sizeOptions: ImageGenerationParams["size"][] = [
		"512x512",
		"768x768",
		"1024x1024",
		"1024x768",
		"768x1024",
	];

	const qualityOptions: ImageGenerationParams["quality"][] = [
		"low",
		"medium",
		"high",
	];

	function handleSizeChange(event: Event) {
		const value = (event.target as HTMLSelectElement)
			.value as ImageGenerationParams["size"];
		setImageGenerationParams({ size: value });
	}

	function handleQualityChange(event: Event) {
		const value = (event.target as HTMLSelectElement)
			.value as ImageGenerationParams["quality"];
		setImageGenerationParams({ quality: value });
	}

	function handleSeedChange(event: Event) {
		const input = event.target as HTMLInputElement;
		const value = input.value.trim();
		if (value === "") {
			setImageGenerationParams({ seed: null });
		} else {
			const num = parseInt(value, 10);
			if (!isNaN(num) && num >= 0) {
				setImageGenerationParams({ seed: num });
			}
		}
	}

	function handleStepsChange(event: Event) {
		const value = parseInt((event.target as HTMLInputElement).value, 10);
		setImageGenerationParams({ numInferenceSteps: value });
	}

	function handleGuidanceChange(event: Event) {
		const value = parseFloat((event.target as HTMLInputElement).value);
		setImageGenerationParams({ guidance: value });
	}

	function handleNegativePromptChange(event: Event) {
		const value = (event.target as HTMLTextAreaElement).value;
		setImageGenerationParams({ negativePrompt: value || null });
	}

	function clearSteps() {
		setImageGenerationParams({ numInferenceSteps: null });
	}

	function clearGuidance() {
		setImageGenerationParams({ guidance: null });
	}

	function handleReset() {
		resetImageGenerationParams();
		showAdvanced = false;
	}

	const hasAdvancedParams = $derived(
		params.seed !== null ||
			params.numInferenceSteps !== null ||
			params.guidance !== null ||
			(params.negativePrompt !== null && params.negativePrompt.trim() !== ""),
	);
</script>

<div class="border-b border-exo-medium-gray/30 px-3 py-2">
	<!-- Basic params row -->
	<div class="flex items-center gap-3 flex-wrap">
		<!-- Size -->
		<div class="flex items-center gap-1.5">
			<span class="text-xs text-exo-light-gray uppercase tracking-wider"
				>SIZE:</span
			>
			<select
				value={params.size}
				onchange={handleSizeChange}
				class="bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1 text-xs font-mono text-exo-yellow cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
			>
				{#each sizeOptions as size}
					<option value={size}>{size}</option>
				{/each}
			</select>
		</div>

		<!-- Quality -->
		<div class="flex items-center gap-1.5">
			<span class="text-xs text-exo-light-gray uppercase tracking-wider"
				>QUALITY:</span
			>
			<select
				value={params.quality}
				onchange={handleQualityChange}
				class="bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1 text-xs font-mono text-exo-yellow cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
			>
				{#each qualityOptions as quality}
					<option value={quality}>{quality.toUpperCase()}</option>
				{/each}
			</select>
		</div>

		<!-- Spacer -->
		<div class="flex-1"></div>

		<!-- Advanced toggle -->
		<button
			type="button"
			onclick={() => (showAdvanced = !showAdvanced)}
			class="flex items-center gap-1 text-xs font-mono tracking-wider uppercase transition-colors duration-200 {showAdvanced ||
			hasAdvancedParams
				? 'text-exo-yellow'
				: 'text-exo-light-gray hover:text-exo-yellow'}"
		>
			<span>ADVANCED</span>
			<svg
				class="w-3 h-3 transition-transform duration-200 {showAdvanced
					? 'rotate-180'
					: ''}"
				fill="none"
				viewBox="0 0 24 24"
				stroke="currentColor"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M19 9l-7 7-7-7"
				/>
			</svg>
			{#if hasAdvancedParams && !showAdvanced}
				<span class="w-1.5 h-1.5 rounded-full bg-exo-yellow"></span>
			{/if}
		</button>
	</div>

	<!-- Advanced params section -->
	{#if showAdvanced}
		<div class="mt-3 pt-3 border-t border-exo-medium-gray/20 space-y-3">
			<!-- Row 1: Seed and Steps -->
			<div class="flex items-center gap-4 flex-wrap">
				<!-- Seed -->
				<div class="flex items-center gap-1.5">
					<span class="text-xs text-exo-light-gray uppercase tracking-wider"
						>SEED:</span
					>
					<input
						type="number"
						min="0"
						value={params.seed ?? ""}
						oninput={handleSeedChange}
						placeholder="Random"
						class="w-24 bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1 text-xs font-mono text-exo-yellow placeholder:text-exo-light-gray/50 transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
					/>
				</div>

				<!-- Steps Slider -->
				<div class="flex items-center gap-1.5 flex-1 min-w-[200px]">
					<span
						class="text-xs text-exo-light-gray uppercase tracking-wider whitespace-nowrap"
						>STEPS:</span
					>
					<div class="flex items-center gap-2 flex-1">
						<input
							type="range"
							min="1"
							max="100"
							value={params.numInferenceSteps ?? 50}
							oninput={handleStepsChange}
							class="flex-1 h-1 bg-exo-medium-gray/50 rounded appearance-none cursor-pointer accent-exo-yellow"
						/>
						<span class="text-xs font-mono text-exo-yellow w-8 text-right">
							{params.numInferenceSteps ?? "--"}
						</span>
						{#if params.numInferenceSteps !== null}
							<button
								type="button"
								onclick={clearSteps}
								class="text-exo-light-gray hover:text-exo-yellow transition-colors"
								title="Clear"
							>
								<svg
									class="w-3 h-3"
									fill="none"
									viewBox="0 0 24 24"
									stroke="currentColor"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="2"
										d="M6 18L18 6M6 6l12 12"
									/>
								</svg>
							</button>
						{/if}
					</div>
				</div>
			</div>

			<!-- Row 2: Guidance -->
			<div class="flex items-center gap-1.5">
				<span
					class="text-xs text-exo-light-gray uppercase tracking-wider whitespace-nowrap"
					>GUIDANCE:</span
				>
				<div class="flex items-center gap-2 flex-1 max-w-xs">
					<input
						type="range"
						min="1"
						max="20"
						step="0.5"
						value={params.guidance ?? 7.5}
						oninput={handleGuidanceChange}
						class="flex-1 h-1 bg-exo-medium-gray/50 rounded appearance-none cursor-pointer accent-exo-yellow"
					/>
					<span class="text-xs font-mono text-exo-yellow w-8 text-right">
						{params.guidance !== null ? params.guidance.toFixed(1) : "--"}
					</span>
					{#if params.guidance !== null}
						<button
							type="button"
							onclick={clearGuidance}
							class="text-exo-light-gray hover:text-exo-yellow transition-colors"
							title="Clear"
						>
							<svg
								class="w-3 h-3"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
									d="M6 18L18 6M6 6l12 12"
								/>
							</svg>
						</button>
					{/if}
				</div>
			</div>

			<!-- Row 3: Negative Prompt -->
			<div class="flex flex-col gap-1.5">
				<span class="text-xs text-exo-light-gray uppercase tracking-wider"
					>NEGATIVE PROMPT:</span
				>
				<textarea
					value={params.negativePrompt ?? ""}
					oninput={handleNegativePromptChange}
					placeholder="Things to avoid in the image..."
					rows={2}
					class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1.5 text-xs font-mono text-exo-yellow placeholder:text-exo-light-gray/50 resize-none transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
				></textarea>
			</div>

			<!-- Reset Button -->
			<div class="flex justify-end pt-1">
				<button
					type="button"
					onclick={handleReset}
					class="text-xs font-mono tracking-wider uppercase text-exo-light-gray hover:text-exo-yellow transition-colors duration-200"
				>
					RESET TO DEFAULTS
				</button>
			</div>
		</div>
	{/if}
</div>

<style>
	/* Custom range slider styling */
	input[type="range"]::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: #ffd700;
		cursor: pointer;
		border: none;
	}

	input[type="range"]::-moz-range-thumb {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: #ffd700;
		cursor: pointer;
		border: none;
	}

	/* Hide number input spinners */
	input[type="number"]::-webkit-inner-spin-button,
	input[type="number"]::-webkit-outer-spin-button {
		-webkit-appearance: none;
		margin: 0;
	}

	input[type="number"] {
		-moz-appearance: textfield;
	}
</style>

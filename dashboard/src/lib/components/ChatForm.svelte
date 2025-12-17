<script lang="ts">
	import { isLoading, sendMessage, selectedChatModel, setSelectedChatModel, instances, ttftMs, tps, totalTokens } from '$lib/stores/app.svelte';
	import ChatAttachments from './ChatAttachments.svelte';
	import type { ChatUploadedFile } from '$lib/types/files';
	import { processUploadedFiles, getAcceptString } from '$lib/types/files';

	interface Props {
		class?: string;
		placeholder?: string;
		showHelperText?: boolean;
		autofocus?: boolean;
		showModelSelector?: boolean;
	}

	let { 
		class: className = '', 
		placeholder = 'Ask anything',
		showHelperText = false,
		autofocus = true,
		showModelSelector = false
	}: Props = $props();

	let message = $state('');
	let textareaRef: HTMLTextAreaElement | undefined = $state();
	let fileInputRef: HTMLInputElement | undefined = $state();
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let isDragOver = $state(false);
	let loading = $derived(isLoading());
	const currentModel = $derived(selectedChatModel());
	const instanceData = $derived(instances());
	const currentTtft = $derived(ttftMs());
	const currentTps = $derived(tps());
	const currentTokens = $derived(totalTokens());
	
	// Custom dropdown state
	let isModelDropdownOpen = $state(false);
	let dropdownButtonRef: HTMLButtonElement | undefined = $state();
	let dropdownPosition = $derived(() => {
		if (!dropdownButtonRef || !isModelDropdownOpen) return { top: 0, left: 0, width: 0 };
		const rect = dropdownButtonRef.getBoundingClientRect();
		return {
			top: rect.top,
			left: rect.left,
			width: rect.width
		};
	});

	// Accept all supported file types
	const acceptString = getAcceptString(['image', 'text', 'pdf']);

	// Extract available models from running instances
	const availableModels = $derived(() => {
		const models: Array<{id: string, label: string}> = [];
		for (const [, instance] of Object.entries(instanceData)) {
			const modelId = getInstanceModelId(instance);
			if (modelId && modelId !== 'Unknown' && !models.some(m => m.id === modelId)) {
				models.push({ id: modelId, label: modelId.split('/').pop() || modelId });
			}
		}
		return models;
	});

	// Auto-select the first available model if none is selected
	$effect(() => {
		const models = availableModels();
		if (models.length > 0 && !currentModel) {
			setSelectedChatModel(models[0].id);
		}
	});

	function getInstanceModelId(instanceWrapped: unknown): string {
		if (!instanceWrapped || typeof instanceWrapped !== 'object') return '';
		const keys = Object.keys(instanceWrapped as Record<string, unknown>);
		if (keys.length === 1) {
			const instance = (instanceWrapped as Record<string, unknown>)[keys[0]] as { shardAssignments?: { modelId?: string } };
			return instance?.shardAssignments?.modelId || '';
		}
		return '';
	}

	async function handleFiles(files: File[]) {
		if (files.length === 0) return;
		const processed = await processUploadedFiles(files);
		uploadedFiles = [...uploadedFiles, ...processed];
	}

	function handleFileInputChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			handleFiles(Array.from(input.files));
			input.value = ''; // Reset for next selection
		}
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter(f => f.id !== fileId);
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;
		
		const files = Array.from(event.clipboardData.items)
			.filter(item => item.kind === 'file')
			.map(item => item.getAsFile())
			.filter((file): file is File => file !== null);
		
		if (files.length > 0) {
			event.preventDefault();
			handleFiles(files);
			return;
		}
		
		// Handle long text paste as file
		const text = event.clipboardData.getData('text/plain');
		if (text.length > 2500) {
			event.preventDefault();
			const textFile = new File([text], 'pasted-text.txt', { type: 'text/plain' });
			handleFiles([textFile]);
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		isDragOver = true;
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		
		if (event.dataTransfer?.files) {
			handleFiles(Array.from(event.dataTransfer.files));
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSubmit();
		}
	}

	function handleSubmit() {
		if ((!message.trim() && uploadedFiles.length === 0) || loading) return;
		
		const content = message.trim();
		const files = [...uploadedFiles];
		
		message = '';
		uploadedFiles = [];
		resetTextareaHeight();
		
		sendMessage(content, files);
		
		// Refocus the textarea after sending
		setTimeout(() => textareaRef?.focus(), 10);
	}

	function handleInput() {
		if (!textareaRef) return;
		textareaRef.style.height = 'auto';
		textareaRef.style.height = Math.min(textareaRef.scrollHeight, 150) + 'px';
	}

	function resetTextareaHeight() {
		if (textareaRef) {
			textareaRef.style.height = 'auto';
		}
	}

	function openFilePicker() {
		fileInputRef?.click();
	}

	// Track previous loading state to detect when loading completes
	let wasLoading = $state(false);
	
	$effect(() => {
		if (autofocus && textareaRef) {
			setTimeout(() => textareaRef?.focus(), 10);
		}
	});
	
	// Refocus after loading completes (AI response finished)
	$effect(() => {
		if (wasLoading && !loading && textareaRef) {
			setTimeout(() => textareaRef?.focus(), 50);
		}
		wasLoading = loading;
	});

	const canSend = $derived(message.trim().length > 0 || uploadedFiles.length > 0);
</script>

<!-- Hidden file input -->
<input
	bind:this={fileInputRef}
	type="file"
	accept={acceptString}
	multiple
	class="hidden"
	onchange={handleFileInputChange}
/>

<form 
	onsubmit={(e) => { e.preventDefault(); handleSubmit(); }} 
	class="w-full {className}"
	ondragover={handleDragOver}
	ondragleave={handleDragLeave}
	ondrop={handleDrop}
>
	<div 
		class="relative command-panel rounded overflow-hidden transition-all duration-200 {isDragOver ? 'ring-2 ring-exo-yellow ring-opacity-50' : ''}"
	>
		<!-- Top accent line -->
		<div class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-exo-yellow/50 to-transparent"></div>
		
		<!-- Drag overlay -->
		{#if isDragOver}
			<div class="absolute inset-0 bg-exo-dark-gray/80 z-10 flex items-center justify-center">
				<div class="text-exo-yellow text-sm font-mono tracking-wider uppercase">
					DROP FILES HERE
				</div>
			</div>
		{/if}
		
		<!-- Model selector (when enabled) -->
		{#if showModelSelector && availableModels().length > 0}
			<div class="flex items-center justify-between gap-2 px-3 py-2 border-b border-exo-medium-gray/30">
				<div class="flex items-center gap-2 flex-1">
					<span class="text-xs text-exo-light-gray uppercase tracking-wider flex-shrink-0">MODEL:</span>
					<!-- Custom dropdown -->
					<div class="relative flex-1 max-w-xs">
						<button
							bind:this={dropdownButtonRef}
							type="button"
							onclick={() => isModelDropdownOpen = !isModelDropdownOpen}
							class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded pl-3 pr-8 py-1.5 text-xs font-mono text-left tracking-wide cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70 {isModelDropdownOpen ? 'border-exo-yellow/70' : ''}"
						>
							{#if availableModels().find(m => m.id === currentModel)}
								<span class="text-exo-yellow truncate">{availableModels().find(m => m.id === currentModel)?.label}</span>
							{:else if availableModels().length > 0}
								<span class="text-exo-yellow truncate">{availableModels()[0].label}</span>
							{:else}
								<span class="text-exo-light-gray/50">— SELECT MODEL —</span>
							{/if}
						</button>
						<div class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none transition-transform duration-200 {isModelDropdownOpen ? 'rotate-180' : ''}">
							<svg class="w-3 h-3 text-exo-yellow/60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
							</svg>
						</div>
					</div>
					
					{#if isModelDropdownOpen}
						<!-- Backdrop to close dropdown -->
						<button 
							type="button"
							class="fixed inset-0 z-[9998] cursor-default" 
							onclick={() => isModelDropdownOpen = false}
							aria-label="Close dropdown"
						></button>
						
						<!-- Dropdown Panel - fixed positioning to escape overflow:hidden -->
						<div 
							class="fixed bg-exo-dark-gray border border-exo-yellow/30 rounded shadow-lg shadow-black/50 z-[9999] max-h-48 overflow-y-auto"
							style="bottom: calc(100vh - {dropdownPosition().top}px + 4px); left: {dropdownPosition().left}px; width: {dropdownPosition().width}px;"
						>
							<div class="py-1">
								{#each availableModels() as model}
									<button
										type="button"
										onclick={() => {
											setSelectedChatModel(model.id);
											isModelDropdownOpen = false;
										}}
										class="w-full px-3 py-2 text-left text-xs font-mono tracking-wide transition-colors duration-100 flex items-center gap-2 {
											currentModel === model.id 
											? 'bg-transparent text-exo-yellow' 
											: 'text-exo-light-gray hover:text-exo-yellow'
										}"
									>
										{#if currentModel === model.id}
											<svg class="w-3 h-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
												<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
											</svg>
										{:else}
											<span class="w-3"></span>
										{/if}
										<span class="truncate">{model.label}</span>
									</button>
								{/each}
							</div>
						</div>
					{/if}
				</div>
				<!-- Performance stats -->
				{#if currentTtft !== null || currentTps !== null}
					<div class="flex items-center gap-4 text-xs font-mono flex-shrink-0">
						{#if currentTtft !== null}
							<span class="text-exo-light-gray">
								<span class="text-white/70">TTFT</span> <span class="text-exo-yellow">{currentTtft.toFixed(1)}ms</span>
							</span>
						{/if}
						{#if currentTps !== null}
							<span class="text-exo-light-gray">
								<span class="text-white/70">TPS</span> <span class="text-exo-yellow">{currentTps.toFixed(1)}</span> <span class="text-white/60">tok/s</span>
								<span class="text-white/50">({(1000 / currentTps).toFixed(1)} ms/tok)</span>
							</span>
						{/if}
					</div>
				{/if}
			</div>
		{/if}
		
		<!-- Attached files preview -->
		{#if uploadedFiles.length > 0}
			<div class="px-3 pt-3">
				<ChatAttachments 
					files={uploadedFiles} 
					onRemove={handleFileRemove}
				/>
			</div>
		{/if}
		
		<!-- Input area -->
		<div class="flex items-start gap-2 sm:gap-3 py-3 px-3 sm:px-4">
			<!-- Attach file button -->
			<button
				type="button"
				onclick={openFilePicker}
				disabled={loading}
				class="flex items-center justify-center w-7 h-7 rounded text-exo-light-gray hover:text-exo-yellow transition-all disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 cursor-pointer"
				title="Attach file"
			>
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
				</svg>
			</button>
			
			<!-- Terminal prompt -->
			<span class="text-exo-yellow text-sm font-bold flex-shrink-0 leading-7">▶</span>
			
			<textarea
				bind:this={textareaRef}
				bind:value={message}
				onkeydown={handleKeydown}
				oninput={handleInput}
				onpaste={handlePaste}
				{placeholder}
				disabled={loading}
				rows={1}
				class="flex-1 resize-none bg-transparent text-foreground placeholder:text-exo-light-gray/60 placeholder:text-sm placeholder:tracking-[0.15em] placeholder:leading-7 focus:outline-none focus:ring-0 focus:border-none disabled:opacity-50 text-sm leading-7 font-mono"
				style="min-height: 28px; max-height: 150px;"
			></textarea>
			
			<button
				type="submit"
				disabled={!canSend || loading}
				class="px-2.5 sm:px-4 py-1.5 sm:py-2 rounded text-xs sm:text-xs tracking-[0.1em] sm:tracking-[0.15em] uppercase font-medium transition-all duration-200 whitespace-nowrap
					{!canSend || loading 
						? 'bg-exo-medium-gray/50 text-exo-light-gray cursor-not-allowed' 
						: 'bg-exo-yellow text-exo-black hover:bg-exo-yellow-darker hover:shadow-[0_0_20px_rgba(255,215,0,0.3)]'}"
				aria-label="Send message"
			>
				{#if loading}
					<span class="inline-flex items-center gap-1 sm:gap-2">
						<span class="w-2.5 h-2.5 sm:w-3 sm:h-3 border-2 border-current border-t-transparent rounded-full animate-spin"></span>
						<span class="hidden sm:inline">PROCESSING</span>
						<span class="sm:hidden">...</span>
					</span>
				{:else}
					SEND
				{/if}
			</button>
		</div>
		
		<!-- Bottom accent line -->
		<div class="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-exo-yellow/30 to-transparent"></div>
	</div>
	
	{#if showHelperText}
		<p class="mt-2 sm:mt-3 text-center text-xs sm:text-xs text-exo-light-gray tracking-[0.1em] sm:tracking-[0.15em] uppercase">
			<kbd class="px-1 sm:px-1.5 py-0.5 rounded bg-exo-medium-gray/30 text-exo-light-gray border border-exo-medium-gray/50">ENTER</kbd>
			<span class="mx-0.5 sm:mx-1">TO SEND</span>
			<span class="text-exo-medium-gray mx-1 sm:mx-2">|</span>
			<kbd class="px-1 sm:px-1.5 py-0.5 rounded bg-exo-medium-gray/30 text-exo-light-gray border border-exo-medium-gray/50">SHIFT+ENTER</kbd>
			<span class="mx-0.5 sm:mx-1">NEW LINE</span>
			<span class="text-exo-medium-gray mx-1 sm:mx-2">|</span>
			<span class="text-exo-light-gray">DRAG & DROP OR PASTE FILES</span>
		</p>
	{/if}
</form>

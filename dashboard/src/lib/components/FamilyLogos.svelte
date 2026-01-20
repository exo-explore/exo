<script lang="ts">
	/**
	 * SVG logos for model families.
	 * Size: 24x24px, monochrome (accepts currentColor).
	 */
	type FamilyLogoProps = {
		family: string;
		class?: string;
	};

	let { family, class: className = '' }: FamilyLogoProps = $props();
</script>

{#if family === 'favorites'}
	<!-- Star icon for favorites -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
	</svg>
{:else if family === 'llama' || family === 'meta'}
	<!-- Meta infinity logo -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 5.5c-1.7 0-3.2.7-4.2 1.8L4 12l3.8 4.7c1 1.1 2.5 1.8 4.2 1.8s3.2-.7 4.2-1.8L20 12l-3.8-4.7C15.2 6.2 13.7 5.5 12 5.5zm0 2c1.1 0 2.1.5 2.8 1.2l2.6 3.3-2.6 3.3c-.7.7-1.7 1.2-2.8 1.2s-2.1-.5-2.8-1.2L6.6 12l2.6-3.3c.7-.7 1.7-1.2 2.8-1.2z" />
	</svg>
{:else if family === 'qwen'}
	<!-- Qwen logo (stylized Q) -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 4C7.58 4 4 7.58 4 12s3.58 8 8 8c1.85 0 3.55-.63 4.9-1.69l1.41 1.41 1.41-1.41-1.41-1.41C19.37 15.55 20 13.85 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6z" />
	</svg>
{:else if family === 'deepseek'}
	<!-- DeepSeek whale/dolphin logo -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 3c-4.97 0-9 3.13-9 7 0 2.38 1.19 4.47 3 5.74V19c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1.12c.93.17 1.92.27 2.95.27s2.07-.09 3.05-.27V19c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-3.26c1.81-1.27 3-3.36 3-5.74 0-3.87-4.03-7-9-7zm-4 8c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1z" />
	</svg>
{:else if family === 'openai' || family === 'gpt-oss'}
	<!-- OpenAI spiral logo -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M22.2 8.3c-.2-1.5-.8-2.9-1.8-4.1-1.5-1.7-3.6-2.7-5.9-2.6-1.4.1-2.7.5-3.9 1.3-.8-.5-1.7-.8-2.6-.9C5.6 1.7 3.3 2.7 2 4.7c-.8 1.2-1.1 2.7-.9 4.1.1.5.2 1 .4 1.4-.9 1.1-1.4 2.4-1.5 3.8-.2 2.4.8 4.7 2.6 6.1 1.2.9 2.6 1.4 4 1.5.5 0 1 0 1.5-.1.8.8 1.8 1.4 2.9 1.7 1.7.4 3.6.1 5-1 1.2-.9 2-2.2 2.3-3.6.2-.5.2-1.1.2-1.6 1.1-.8 1.9-2 2.3-3.3.6-1.8.5-3.7-.6-5.4zm-4.3 9.9c-.2.8-.7 1.5-1.3 2-.9.7-2.1.9-3.2.6-.7-.2-1.3-.6-1.8-1.1l.1-.1 4.1-2.4c.3-.2.5-.5.5-.8V11l1.7 1v5c0 .4 0 .9-.1 1.2zm-11.9-1c-.5-.3-.9-.8-1.2-1.3-.4-.9-.5-1.9-.3-2.8.1-.5.4-1 .7-1.4l.1.1 4.1 2.4c.3.2.6.2.9 0l5-2.9v2l-4.2 2.4c-1.5.8-3.3.9-4.9.2l-.2-.7zm-1.4-8.4c.3-.6.7-1.1 1.2-1.4-.1.4-.1.8-.1 1.2v4.7l-1.7-1V7.8c0-.4.2-.7.6-1zm13 6l-4.1 2.4-.1.1-1.7-1V11c0-.3.2-.6.5-.8l4.1-2.4.1-.1.1.1 1.7 1v4.9c0 .3-.2.6-.6.8z" />
	</svg>
{:else if family === 'anthropic'}
	<!-- Anthropic 'A' logo -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 2L3 20h3.5l1.5-3h8l1.5 3H21L12 2zm0 5.5L15.5 15h-7L12 7.5z" />
	</svg>
{:else if family === 'mistral'}
	<!-- Mistral 'M' logo -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M4 4v16h3v-8l5 8 5-8v8h3V4h-3l-5 8-5-8H4z" />
	</svg>
{:else if family === 'google' || family === 'gemini'}
	<!-- Google Gemini sparkle -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8L12 2z" />
	</svg>
{:else if family === 'glm'}
	<!-- Zhipu AI GLM logo (stylized Z) -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M4 5h16v3H8.5L20 17v3H4v-3h11.5L4 8V5z" />
	</svg>
{:else if family === 'minimax'}
	<!-- MiniMax logo (stylized MM) -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M2 5v14h3v-9l3 6 3-6v9h3V5h-3l-3 6-3-6H2zm11 0v14h3v-9l3 6 3-6v9h3V5h-3l-3 6-3-6h-3z" />
	</svg>
{:else if family === 'kimi'}
	<!-- Moonshot/Kimi crescent moon logo -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 3a9 9 0 109 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 01-4.4 2.26 5.403 5.403 0 01-3.14-9.8c-.44-.06-.9-.1-1.36-.1z" />
	</svg>
{:else if family === 'huggingface'}
	<!-- HuggingFace logo (stylized face) -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 13.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm4 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm2-4.5H8c0-2.21 1.79-4 4-4s4 1.79 4 4z" />
	</svg>
{:else}
	<!-- Default generic AI/model icon -->
	<svg class="w-6 h-6 {className}" viewBox="0 0 24 24" fill="currentColor">
		<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
	</svg>
{/if}

<script lang="ts">
	import { marked } from 'marked';
	import hljs from 'highlight.js';
	import katex from 'katex';
	import 'katex/dist/katex.min.css';
	import { browser } from '$app/environment';

	interface Props {
		content: string;
		class?: string;
	}

	let { content, class: className = '' }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let processedHtml = $state('');

	// Configure marked with syntax highlighting
	marked.setOptions({
		gfm: true,
		breaks: true
	});

	// Custom renderer for code blocks
	const renderer = new marked.Renderer();

	renderer.code = function ({ text, lang }: { text: string; lang?: string }) {
		const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext';
		const highlighted = hljs.highlight(text, { language }).value;
		const codeId = `code-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

		return `
			<div class="code-block-wrapper">
				<div class="code-block-header">
					<span class="code-language">${language}</span>
					<button type="button" class="copy-code-btn" data-code="${encodeURIComponent(text)}" title="Copy code">
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
							<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
						</svg>
					</button>
				</div>
				<pre><code class="hljs language-${language}" data-code-id="${codeId}">${highlighted}</code></pre>
			</div>
		`;
	};

	// Inline code
	renderer.codespan = function ({ text }: { text: string }) {
		return `<code class="inline-code">${text}</code>`;
	};

	marked.use({ renderer });

	/**
	 * Preprocess LaTeX: convert \(...\) to $...$ and \[...\] to $$...$$
	 * Also protect code blocks from LaTeX processing
	 */
	function preprocessLaTeX(text: string): string {
		// Protect code blocks
		const codeBlocks: string[] = [];
		let processed = text.replace(/```[\s\S]*?```|`[^`]+`/g, (match) => {
			codeBlocks.push(match);
			return `<<CODE_${codeBlocks.length - 1}>>`;
		});

		// Convert \(...\) to $...$
		processed = processed.replace(/\\\((.+?)\\\)/g, '$$$1$');
		
		// Convert \[...\] to $$...$$
		processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$');

		// Restore code blocks
		processed = processed.replace(/<<CODE_(\d+)>>/g, (_, index) => codeBlocks[parseInt(index)]);

		return processed;
	}

	/**
	 * Render math expressions with KaTeX after HTML is generated
	 */
	function renderMath(html: string): string {
		// Render display math ($$...$$)
		html = html.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
			try {
				return katex.renderToString(math.trim(), {
					displayMode: true,
					throwOnError: false,
					output: 'html'
				});
			} catch {
				return `<span class="math-error">$$${math}$$</span>`;
			}
		});

		// Render inline math ($...$) but avoid matching currency like $5
		html = html.replace(/\$([^\$\n]+?)\$/g, (match, math) => {
			// Skip if it looks like currency ($ followed by number)
			if (/^\d/.test(math.trim())) {
				return match;
			}
			try {
				return katex.renderToString(math.trim(), {
					displayMode: false,
					throwOnError: false,
					output: 'html'
				});
			} catch {
				return `<span class="math-error">$${math}$</span>`;
			}
		});

		return html;
	}

	function processMarkdown(text: string): string {
		try {
			// Preprocess LaTeX notation
			const preprocessed = preprocessLaTeX(text);
			// Parse markdown
			let html = marked.parse(preprocessed) as string;
			// Render math expressions
			html = renderMath(html);
			return html;
		} catch (error) {
			console.error('Markdown processing error:', error);
			return text.replace(/\n/g, '<br>');
		}
	}

	async function handleCopyClick(event: Event) {
		const target = event.currentTarget as HTMLButtonElement;
		const encodedCode = target.getAttribute('data-code');
		if (!encodedCode) return;

		const code = decodeURIComponent(encodedCode);

		try {
			await navigator.clipboard.writeText(code);
			// Show copied feedback
			const originalHtml = target.innerHTML;
			target.innerHTML = `
				<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
					<path d="M20 6L9 17l-5-5"/>
				</svg>
			`;
			target.classList.add('copied');
			setTimeout(() => {
				target.innerHTML = originalHtml;
				target.classList.remove('copied');
			}, 2000);
		} catch (error) {
			console.error('Failed to copy:', error);
		}
	}

	function setupCopyButtons() {
		if (!containerRef || !browser) return;

		const buttons = containerRef.querySelectorAll<HTMLButtonElement>('.copy-code-btn');
		for (const button of buttons) {
			if (button.dataset.listenerBound !== 'true') {
				button.dataset.listenerBound = 'true';
				button.addEventListener('click', handleCopyClick);
			}
		}
	}

	$effect(() => {
		if (content) {
			processedHtml = processMarkdown(content);
		} else {
			processedHtml = '';
		}
	});

	$effect(() => {
		if (containerRef && processedHtml) {
			setupCopyButtons();
		}
	});
</script>

<div bind:this={containerRef} class="markdown-content {className}">
	{@html processedHtml}
</div>

<style>
	.markdown-content {
		line-height: 1.6;
	}

	/* Paragraphs */
	.markdown-content :global(p) {
		margin-bottom: 1rem;
	}

	.markdown-content :global(p:last-child) {
		margin-bottom: 0;
	}

	/* Headers */
	.markdown-content :global(h1) {
		font-size: 1.5rem;
		font-weight: 700;
		margin: 1.5rem 0 0.75rem 0;
		color: var(--exo-yellow, #ffd700);
	}

	.markdown-content :global(h2) {
		font-size: 1.25rem;
		font-weight: 600;
		margin: 1.25rem 0 0.5rem 0;
		color: var(--exo-yellow, #ffd700);
	}

	.markdown-content :global(h3) {
		font-size: 1.125rem;
		font-weight: 600;
		margin: 1rem 0 0.5rem 0;
	}

	.markdown-content :global(h4),
	.markdown-content :global(h5),
	.markdown-content :global(h6) {
		font-size: 1rem;
		font-weight: 600;
		margin: 0.75rem 0 0.25rem 0;
	}

	/* Bold and italic */
	.markdown-content :global(strong) {
		font-weight: 600;
	}

	.markdown-content :global(em) {
		font-style: italic;
	}

	/* Inline code */
	.markdown-content :global(.inline-code) {
		background: rgba(255, 215, 0, 0.1);
		color: var(--exo-yellow, #ffd700);
		padding: 0.125rem 0.375rem;
		border-radius: 0.25rem;
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, Consolas, monospace;
		font-size: 0.875em;
	}

	/* Links */
	.markdown-content :global(a) {
		color: var(--exo-yellow, #ffd700);
		text-decoration: underline;
		text-underline-offset: 2px;
	}

	.markdown-content :global(a:hover) {
		opacity: 0.8;
	}

	/* Lists */
	.markdown-content :global(ul) {
		list-style-type: disc;
		margin-left: 1.5rem;
		margin-bottom: 1rem;
	}

	.markdown-content :global(ol) {
		list-style-type: decimal;
		margin-left: 1.5rem;
		margin-bottom: 1rem;
	}

	.markdown-content :global(li) {
		margin-bottom: 0.25rem;
	}

	.markdown-content :global(li::marker) {
		color: var(--exo-light-gray, #9ca3af);
	}

	/* Blockquotes */
	.markdown-content :global(blockquote) {
		border-left: 3px solid var(--exo-yellow, #ffd700);
		padding: 0.5rem 1rem;
		margin: 1rem 0;
		background: rgba(255, 215, 0, 0.05);
		border-radius: 0 0.25rem 0.25rem 0;
	}

	/* Tables */
	.markdown-content :global(table) {
		width: 100%;
		margin: 1rem 0;
		border-collapse: collapse;
		font-size: 0.875rem;
	}

	.markdown-content :global(th) {
		background: rgba(255, 215, 0, 0.1);
		border: 1px solid rgba(255, 215, 0, 0.2);
		padding: 0.5rem;
		text-align: left;
		font-weight: 600;
	}

	.markdown-content :global(td) {
		border: 1px solid rgba(255, 255, 255, 0.1);
		padding: 0.5rem;
	}

	/* Horizontal rule */
	.markdown-content :global(hr) {
		border: none;
		border-top: 1px solid rgba(255, 255, 255, 0.1);
		margin: 1.5rem 0;
	}

	/* Code block wrapper */
	.markdown-content :global(.code-block-wrapper) {
		margin: 1rem 0;
		border-radius: 0.5rem;
		overflow: hidden;
		border: 1px solid rgba(255, 215, 0, 0.2);
		background: rgba(0, 0, 0, 0.4);
	}

	.markdown-content :global(.code-block-header) {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 0.75rem;
		background: rgba(255, 215, 0, 0.05);
		border-bottom: 1px solid rgba(255, 215, 0, 0.1);
	}

	.markdown-content :global(.code-language) {
		color: var(--exo-yellow, #ffd700);
		font-size: 0.7rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, Consolas, monospace;
	}

	.markdown-content :global(.copy-code-btn) {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0.25rem;
		background: transparent;
		border: none;
		color: var(--exo-light-gray, #9ca3af);
		cursor: pointer;
		transition: color 0.2s;
		border-radius: 0.25rem;
	}

	.markdown-content :global(.copy-code-btn:hover) {
		color: var(--exo-yellow, #ffd700);
	}

	.markdown-content :global(.copy-code-btn.copied) {
		color: #22c55e;
	}

	.markdown-content :global(.code-block-wrapper pre) {
		margin: 0;
		padding: 1rem;
		overflow-x: auto;
		background: transparent;
	}

	.markdown-content :global(.code-block-wrapper code) {
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, Consolas, monospace;
		font-size: 0.8125rem;
		line-height: 1.5;
		background: transparent;
	}

	/* Syntax highlighting - dark theme matching EXO style */
	.markdown-content :global(.hljs) {
		color: #e5e7eb;
	}

	.markdown-content :global(.hljs-keyword),
	.markdown-content :global(.hljs-selector-tag),
	.markdown-content :global(.hljs-literal),
	.markdown-content :global(.hljs-section),
	.markdown-content :global(.hljs-link) {
		color: #c084fc;
	}

	.markdown-content :global(.hljs-string),
	.markdown-content :global(.hljs-title),
	.markdown-content :global(.hljs-name),
	.markdown-content :global(.hljs-type),
	.markdown-content :global(.hljs-attribute),
	.markdown-content :global(.hljs-symbol),
	.markdown-content :global(.hljs-bullet),
	.markdown-content :global(.hljs-addition),
	.markdown-content :global(.hljs-variable),
	.markdown-content :global(.hljs-template-tag),
	.markdown-content :global(.hljs-template-variable) {
		color: #fbbf24;
	}

	.markdown-content :global(.hljs-comment),
	.markdown-content :global(.hljs-quote),
	.markdown-content :global(.hljs-deletion),
	.markdown-content :global(.hljs-meta) {
		color: #6b7280;
	}

	.markdown-content :global(.hljs-number),
	.markdown-content :global(.hljs-regexp),
	.markdown-content :global(.hljs-literal),
	.markdown-content :global(.hljs-built_in) {
		color: #34d399;
	}

	.markdown-content :global(.hljs-function),
	.markdown-content :global(.hljs-class .hljs-title) {
		color: #60a5fa;
	}

	/* KaTeX math styling */
	.markdown-content :global(.katex) {
		font-size: 1.1em;
	}

	.markdown-content :global(.katex-display) {
		margin: 1rem 0;
		overflow-x: auto;
		overflow-y: hidden;
		padding: 0.5rem 0;
	}

	.markdown-content :global(.katex-display > .katex) {
		text-align: center;
	}

	.markdown-content :global(.math-error) {
		color: #f87171;
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, Consolas, monospace;
		font-size: 0.875em;
		background: rgba(248, 113, 113, 0.1);
		padding: 0.125rem 0.25rem;
		border-radius: 0.25rem;
	}
</style>

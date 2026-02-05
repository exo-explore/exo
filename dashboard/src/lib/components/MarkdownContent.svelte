<script lang="ts">
  import { marked } from "marked";
  import hljs from "highlight.js";
  import katex from "katex";
  import "katex/dist/katex.min.css";
  import { browser } from "$app/environment";

  interface Props {
    content: string;
    class?: string;
  }

  let { content, class: className = "" }: Props = $props();

  let containerRef = $state<HTMLDivElement>();
  let processedHtml = $state("");

  // Configure marked with syntax highlighting
  marked.setOptions({
    gfm: true,
    breaks: true,
  });

  // Custom renderer for code blocks
  const renderer = new marked.Renderer();

  renderer.code = function ({ text, lang }: { text: string; lang?: string }) {
    const language = lang && hljs.getLanguage(lang) ? lang : "plaintext";
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
   * Unescape HTML entities that marked may have escaped
   */
  function unescapeHtmlEntities(text: string): string {
    return text
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
  }

  // Storage for math expressions extracted before markdown processing
  const mathExpressions: Map<
    string,
    { content: string; displayMode: boolean }
  > = new Map();
  let mathCounter = 0;

  // Storage for HTML snippets that need protection from markdown
  const htmlSnippets: Map<string, string> = new Map();
  let htmlCounter = 0;

  // Use alphanumeric placeholders that won't be interpreted as HTML tags
  const MATH_PLACEHOLDER_PREFIX = "MATHPLACEHOLDER";
  const CODE_PLACEHOLDER_PREFIX = "CODEPLACEHOLDER";
  const HTML_PLACEHOLDER_PREFIX = "HTMLPLACEHOLDER";

  /**
   * Preprocess LaTeX: extract math, handle LaTeX document commands, and protect content
   */
  function preprocessLaTeX(text: string): string {
    // Reset storage
    mathExpressions.clear();
    mathCounter = 0;
    htmlSnippets.clear();
    htmlCounter = 0;

    // Protect code blocks first
    const codeBlocks: string[] = [];
    let processed = text.replace(/```[\s\S]*?```|`[^`]+`/g, (match) => {
      codeBlocks.push(match);
      return `${CODE_PLACEHOLDER_PREFIX}${codeBlocks.length - 1}END`;
    });

    // Remove LaTeX document commands
    processed = processed.replace(/\\documentclass(\[[^\]]*\])?\{[^}]*\}/g, "");
    processed = processed.replace(/\\usepackage(\[[^\]]*\])?\{[^}]*\}/g, "");
    processed = processed.replace(/\\begin\{document\}/g, "");
    processed = processed.replace(/\\end\{document\}/g, "");
    processed = processed.replace(/\\maketitle/g, "");
    processed = processed.replace(/\\title\{[^}]*\}/g, "");
    processed = processed.replace(/\\author\{[^}]*\}/g, "");
    processed = processed.replace(/\\date\{[^}]*\}/g, "");

    // Remove \require{...} commands (MathJax-specific, not supported by KaTeX)
    processed = processed.replace(/\$\\require\{[^}]*\}\$/g, "");
    processed = processed.replace(/\\require\{[^}]*\}/g, "");

    // Remove unsupported LaTeX commands/environments (tikzpicture, figure, center, etc.)
    processed = processed.replace(
      /\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}/g,
      () => {
        const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
        htmlSnippets.set(
          placeholder,
          '<div class="latex-diagram-placeholder"><span class="latex-diagram-icon">📐</span><span class="latex-diagram-text">Diagram</span></div>',
        );
        htmlCounter++;
        return placeholder;
      },
    );
    processed = processed.replace(
      /\\begin\{figure\}[\s\S]*?\\end\{figure\}/g,
      () => {
        const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
        htmlSnippets.set(
          placeholder,
          '<div class="latex-diagram-placeholder"><span class="latex-diagram-icon">🖼️</span><span class="latex-diagram-text">Figure</span></div>',
        );
        htmlCounter++;
        return placeholder;
      },
    );
    // Strip center environment (layout only, no content change)
    processed = processed.replace(/\\begin\{center\}/g, "");
    processed = processed.replace(/\\end\{center\}/g, "");
    // Strip other layout environments
    processed = processed.replace(/\\begin\{flushleft\}/g, "");
    processed = processed.replace(/\\end\{flushleft\}/g, "");
    processed = processed.replace(/\\begin\{flushright\}/g, "");
    processed = processed.replace(/\\end\{flushright\}/g, "");
    processed = processed.replace(/\\label\{[^}]*\}/g, "");
    processed = processed.replace(/\\caption\{[^}]*\}/g, "");

    // Protect escaped dollar signs (e.g., \$50 should become $50, not LaTeX)
    processed = processed.replace(/\\\$/g, "ESCAPEDDOLLARPLACEHOLDER");

    // Convert LaTeX math environments to display math (both bare and wrapped in $...$)
    const mathEnvs = [
      "align",
      "align\\*",
      "equation",
      "equation\\*",
      "gather",
      "gather\\*",
      "multline",
      "multline\\*",
      "eqnarray",
      "eqnarray\\*",
      "array",
      "matrix",
      "pmatrix",
      "bmatrix",
      "vmatrix",
      "cases",
    ];
    for (const env of mathEnvs) {
      // Handle $\begin{env}...\end{env}$ (with dollar signs, possibly multiline)
      const wrappedRegex = new RegExp(
        `\\$\\\\begin\\{${env}\\}(\\{[^}]*\\})?([\\s\\S]*?)\\\\end\\{${env}\\}\\$`,
        "g",
      );
      processed = processed.replace(wrappedRegex, (_, args, content) => {
        const cleanEnv = env.replace("\\*", "*");
        const mathContent = `\\begin{${cleanEnv}}${args || ""}${content}\\end{${cleanEnv}}`;
        const placeholder = `${MATH_PLACEHOLDER_PREFIX}DISPLAY${mathCounter}END`;
        mathExpressions.set(placeholder, {
          content: mathContent,
          displayMode: true,
        });
        mathCounter++;
        return placeholder;
      });

      // Handle bare \begin{env}...\end{env} (without dollar signs)
      const bareRegex = new RegExp(
        `\\\\begin\\{${env}\\}(\\{[^}]*\\})?([\\s\\S]*?)\\\\end\\{${env}\\}`,
        "g",
      );
      processed = processed.replace(bareRegex, (_, args, content) => {
        const cleanEnv = env.replace("\\*", "*");
        const mathContent = `\\begin{${cleanEnv}}${args || ""}${content}\\end{${cleanEnv}}`;
        const placeholder = `${MATH_PLACEHOLDER_PREFIX}DISPLAY${mathCounter}END`;
        mathExpressions.set(placeholder, {
          content: mathContent,
          displayMode: true,
        });
        mathCounter++;
        return placeholder;
      });
    }

    // Convert LaTeX proof environments to styled blocks (use placeholders for HTML)
    processed = processed.replace(
      /\\begin\{proof\}([\s\S]*?)\\end\{proof\}/g,
      (_, content) => {
        const html = `<div class="latex-proof"><div class="latex-proof-header">Proof</div><div class="latex-proof-content">${content}</div></div>`;
        const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
        htmlSnippets.set(placeholder, html);
        htmlCounter++;
        return placeholder;
      },
    );

    // Convert LaTeX theorem-like environments
    const theoremEnvs = [
      "theorem",
      "lemma",
      "corollary",
      "proposition",
      "definition",
      "remark",
      "example",
    ];
    for (const env of theoremEnvs) {
      const envRegex = new RegExp(
        `\\\\begin\\{${env}\\}([\\s\\S]*?)\\\\end\\{${env}\\}`,
        "gi",
      );
      const envName = env.charAt(0).toUpperCase() + env.slice(1);
      processed = processed.replace(envRegex, (_, content) => {
        const html = `<div class="latex-theorem"><div class="latex-theorem-header">${envName}</div><div class="latex-theorem-content">${content}</div></div>`;
        const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
        htmlSnippets.set(placeholder, html);
        htmlCounter++;
        return placeholder;
      });
    }

    // Convert LaTeX text formatting commands (use placeholders to protect from markdown)
    processed = processed.replace(/\\emph\{([^}]*)\}/g, (_, content) => {
      const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
      htmlSnippets.set(placeholder, `<em>${content}</em>`);
      htmlCounter++;
      return placeholder;
    });
    processed = processed.replace(/\\textit\{([^}]*)\}/g, (_, content) => {
      const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
      htmlSnippets.set(placeholder, `<em>${content}</em>`);
      htmlCounter++;
      return placeholder;
    });
    processed = processed.replace(/\\textbf\{([^}]*)\}/g, (_, content) => {
      const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
      htmlSnippets.set(placeholder, `<strong>${content}</strong>`);
      htmlCounter++;
      return placeholder;
    });
    processed = processed.replace(/\\texttt\{([^}]*)\}/g, (_, content) => {
      const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
      htmlSnippets.set(
        placeholder,
        `<code class="inline-code">${content}</code>`,
      );
      htmlCounter++;
      return placeholder;
    });
    processed = processed.replace(/\\underline\{([^}]*)\}/g, (_, content) => {
      const placeholder = `${HTML_PLACEHOLDER_PREFIX}${htmlCounter}END`;
      htmlSnippets.set(placeholder, `<u>${content}</u>`);
      htmlCounter++;
      return placeholder;
    });

    // Handle LaTeX line breaks and spacing
    processed = processed.replace(/\\\\(?:\s*\n)?/g, "\n"); // \\ -> newline
    processed = processed.replace(/\\newline/g, "\n");
    processed = processed.replace(/\\par\b/g, "\n\n");
    processed = processed.replace(/\\quad/g, " ");
    processed = processed.replace(/\\qquad/g, "  ");
    processed = processed.replace(/~~/g, " "); // non-breaking space

    // Remove other common LaTeX commands that don't render
    processed = processed.replace(/\\centering/g, "");
    processed = processed.replace(/\\noindent/g, "");
    processed = processed.replace(/\\hfill/g, "");
    processed = processed.replace(/\\vspace\{[^}]*\}/g, "");
    processed = processed.replace(/\\hspace\{[^}]*\}/g, " ");

    // Convert \(...\) to placeholder (display: false)
    processed = processed.replace(/\\\(([\s\S]+?)\\\)/g, (_, content) => {
      const placeholder = `${MATH_PLACEHOLDER_PREFIX}INLINE${mathCounter}END`;
      mathExpressions.set(placeholder, { content, displayMode: false });
      mathCounter++;
      return placeholder;
    });

    // Convert \[...\] to placeholder (display: true)
    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, content) => {
      const placeholder = `${MATH_PLACEHOLDER_PREFIX}DISPLAY${mathCounter}END`;
      mathExpressions.set(placeholder, { content, displayMode: true });
      mathCounter++;
      return placeholder;
    });

    // Extract display math ($$...$$) BEFORE markdown processing
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, content) => {
      const placeholder = `${MATH_PLACEHOLDER_PREFIX}DISPLAY${mathCounter}END`;
      mathExpressions.set(placeholder, {
        content: content.trim(),
        displayMode: true,
      });
      mathCounter++;
      return placeholder;
    });

    // Extract inline math ($...$) BEFORE markdown processing
    // Allow single-line only, skip currency patterns like $5 or $50
    processed = processed.replace(/\$([^\$\n]+?)\$/g, (match, content) => {
      if (/^\d/.test(content.trim())) {
        return match; // Keep as-is for currency
      }
      const placeholder = `${MATH_PLACEHOLDER_PREFIX}INLINE${mathCounter}END`;
      mathExpressions.set(placeholder, {
        content: content.trim(),
        displayMode: false,
      });
      mathCounter++;
      return placeholder;
    });

    // Restore escaped dollar signs
    processed = processed.replace(/ESCAPEDDOLLARPLACEHOLDER/g, "$");

    // Restore code blocks
    processed = processed.replace(
      new RegExp(`${CODE_PLACEHOLDER_PREFIX}(\\d+)END`, "g"),
      (_, index) => codeBlocks[parseInt(index)],
    );

    // Clean up any remaining stray backslashes from unrecognized commands
    processed = processed.replace(/\\(?=[a-zA-Z])/g, ""); // Remove \ before letters (unrecognized commands)

    return processed;
  }

  /**
   * Render math expressions with KaTeX and restore HTML placeholders
   */
  function renderMath(html: string): string {
    // Replace all math placeholders with rendered KaTeX
    for (const [placeholder, { content, displayMode }] of mathExpressions) {
      const escapedPlaceholder = placeholder.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&",
      );
      const regex = new RegExp(escapedPlaceholder, "g");

      html = html.replace(regex, () => {
        try {
          const rendered = katex.renderToString(content, {
            displayMode,
            throwOnError: false,
            output: "html",
          });

          if (displayMode) {
            return `
							<div class="math-display-wrapper">
								<div class="math-display-header">
									<span class="math-label">LaTeX</span>
									<button type="button" class="copy-math-btn" data-math-source="${encodeURIComponent(content)}" title="Copy LaTeX source">
										<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
											<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
											<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
										</svg>
									</button>
								</div>
								<div class="math-display-content">
									${rendered}
								</div>
							</div>
						`;
          } else {
            return `<span class="math-inline">${rendered}</span>`;
          }
        } catch {
          const display = displayMode ? `$$${content}$$` : `$${content}$`;
          return `<span class="math-error"><span class="math-error-icon">⚠</span> ${display}</span>`;
        }
      });
    }

    // Restore HTML placeholders (for \textbf, \emph, etc.)
    for (const [placeholder, htmlContent] of htmlSnippets) {
      const escapedPlaceholder = placeholder.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&",
      );
      const regex = new RegExp(escapedPlaceholder, "g");
      html = html.replace(regex, htmlContent);
    }

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
      console.error("Markdown processing error:", error);
      return text.replace(/\n/g, "<br>");
    }
  }

  async function handleCopyClick(event: Event) {
    const target = event.currentTarget as HTMLButtonElement;
    const encodedCode = target.getAttribute("data-code");
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
      target.classList.add("copied");
      setTimeout(() => {
        target.innerHTML = originalHtml;
        target.classList.remove("copied");
      }, 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  }

  async function handleMathCopyClick(event: Event) {
    const target = event.currentTarget as HTMLButtonElement;
    const encodedSource = target.getAttribute("data-math-source");
    if (!encodedSource) return;

    const source = decodeURIComponent(encodedSource);

    try {
      await navigator.clipboard.writeText(source);
      // Show copied feedback
      const originalHtml = target.innerHTML;
      target.innerHTML = `
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
					<path d="M20 6L9 17l-5-5"/>
				</svg>
			`;
      target.classList.add("copied");
      setTimeout(() => {
        target.innerHTML = originalHtml;
        target.classList.remove("copied");
      }, 2000);
    } catch (error) {
      console.error("Failed to copy math:", error);
    }
  }

  function setupCopyButtons() {
    if (!containerRef || !browser) return;

    const codeButtons =
      containerRef.querySelectorAll<HTMLButtonElement>(".copy-code-btn");
    for (const button of codeButtons) {
      if (button.dataset.listenerBound !== "true") {
        button.dataset.listenerBound = "true";
        button.addEventListener("click", handleCopyClick);
      }
    }

    const mathButtons =
      containerRef.querySelectorAll<HTMLButtonElement>(".copy-math-btn");
    for (const button of mathButtons) {
      if (button.dataset.listenerBound !== "true") {
        button.dataset.listenerBound = "true";
        button.addEventListener("click", handleMathCopyClick);
      }
    }
  }

  $effect(() => {
    if (content) {
      processedHtml = processMarkdown(content);
    } else {
      processedHtml = "";
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
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace;
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
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace;
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
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace;
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

  /* KaTeX math styling - Base */
  .markdown-content :global(.katex) {
    font-size: 1.1em;
    color: oklch(0.9 0 0);
  }

  /* Display math container wrapper */
  .markdown-content :global(.math-display-wrapper) {
    margin: 1rem 0;
    border-radius: 0.5rem;
    overflow: hidden;
    border: 1px solid rgba(255, 215, 0, 0.15);
    background: rgba(0, 0, 0, 0.3);
    transition:
      border-color 0.2s ease,
      box-shadow 0.2s ease;
  }

  .markdown-content :global(.math-display-wrapper:hover) {
    border-color: rgba(255, 215, 0, 0.25);
    box-shadow: 0 0 12px rgba(255, 215, 0, 0.08);
  }

  /* Display math header - hidden by default, slides in on hover */
  .markdown-content :global(.math-display-header) {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.375rem 0.75rem;
    background: rgba(255, 215, 0, 0.03);
    border-bottom: 1px solid rgba(255, 215, 0, 0.08);
    opacity: 0;
    max-height: 0;
    padding-top: 0;
    padding-bottom: 0;
    overflow: hidden;
    transition:
      opacity 0.2s ease,
      max-height 0.2s ease,
      padding 0.2s ease;
  }

  .markdown-content :global(.math-display-wrapper:hover .math-display-header) {
    opacity: 1;
    max-height: 2.5rem;
    padding: 0.375rem 0.75rem;
  }

  .markdown-content :global(.math-label) {
    color: rgba(255, 215, 0, 0.7);
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace;
  }

  .markdown-content :global(.copy-math-btn) {
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
    opacity: 0;
    transition:
      color 0.2s,
      opacity 0.15s ease;
  }

  .markdown-content :global(.math-display-wrapper:hover .copy-math-btn) {
    opacity: 1;
  }

  .markdown-content :global(.copy-math-btn:hover) {
    color: var(--exo-yellow, #ffd700);
  }

  .markdown-content :global(.copy-math-btn.copied) {
    color: #22c55e;
  }

  /* Display math content area */
  .markdown-content :global(.math-display-content) {
    padding: 1rem 1.25rem;
    overflow-x: auto;
    overflow-y: hidden;
  }

  /* Custom scrollbar for math overflow */
  .markdown-content :global(.math-display-content::-webkit-scrollbar) {
    height: 6px;
  }

  .markdown-content :global(.math-display-content::-webkit-scrollbar-track) {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 3px;
  }

  .markdown-content :global(.math-display-content::-webkit-scrollbar-thumb) {
    background: rgba(255, 215, 0, 0.2);
    border-radius: 3px;
  }

  .markdown-content
    :global(.math-display-content::-webkit-scrollbar-thumb:hover) {
    background: rgba(255, 215, 0, 0.35);
  }

  .markdown-content :global(.math-display-content .katex-display) {
    margin: 0;
    padding: 0;
  }

  .markdown-content :global(.math-display-content .katex-display > .katex) {
    text-align: center;
  }

  /* Inline math wrapper */
  .markdown-content :global(.math-inline) {
    display: inline;
    padding: 0 0.125rem;
    border-radius: 0.25rem;
    transition: background-color 0.15s ease;
  }

  .markdown-content :global(.math-inline:hover) {
    background: rgba(255, 215, 0, 0.05);
  }

  /* Dark theme KaTeX overrides */
  .markdown-content :global(.katex .mord),
  .markdown-content :global(.katex .minner),
  .markdown-content :global(.katex .mop),
  .markdown-content :global(.katex .mbin),
  .markdown-content :global(.katex .mrel),
  .markdown-content :global(.katex .mpunct) {
    color: oklch(0.9 0 0);
  }

  /* Fraction lines and rules */
  .markdown-content :global(.katex .frac-line),
  .markdown-content :global(.katex .overline-line),
  .markdown-content :global(.katex .underline-line),
  .markdown-content :global(.katex .hline),
  .markdown-content :global(.katex .rule) {
    border-color: oklch(0.85 0 0) !important;
    background: oklch(0.85 0 0);
  }

  /* Square roots and SVG elements */
  .markdown-content :global(.katex .sqrt-line) {
    border-color: oklch(0.85 0 0) !important;
  }

  .markdown-content :global(.katex svg) {
    fill: oklch(0.85 0 0);
    stroke: oklch(0.85 0 0);
  }

  .markdown-content :global(.katex svg path) {
    stroke: oklch(0.85 0 0);
  }

  /* Delimiters (parentheses, brackets, braces) */
  .markdown-content :global(.katex .delimsizing),
  .markdown-content :global(.katex .delim-size1),
  .markdown-content :global(.katex .delim-size2),
  .markdown-content :global(.katex .delim-size3),
  .markdown-content :global(.katex .delim-size4),
  .markdown-content :global(.katex .mopen),
  .markdown-content :global(.katex .mclose) {
    color: oklch(0.75 0 0);
  }

  /* Math error styling */
  .markdown-content :global(.math-error) {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    color: #f87171;
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace;
    font-size: 0.875em;
    background: rgba(248, 113, 113, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid rgba(248, 113, 113, 0.2);
  }

  .markdown-content :global(.math-error-icon) {
    font-size: 0.875em;
    opacity: 0.9;
  }

  /* LaTeX proof environment */
  .markdown-content :global(.latex-proof) {
    margin: 1rem 0;
    padding: 1rem 1.25rem;
    background: rgba(255, 255, 255, 0.02);
    border-left: 3px solid rgba(255, 215, 0, 0.4);
    border-radius: 0 0.375rem 0.375rem 0;
  }

  .markdown-content :global(.latex-proof-header) {
    font-weight: 600;
    font-style: italic;
    color: oklch(0.85 0 0);
    margin-bottom: 0.5rem;
  }

  .markdown-content :global(.latex-proof-header::after) {
    content: ".";
  }

  .markdown-content :global(.latex-proof-content) {
    color: oklch(0.9 0 0);
  }

  .markdown-content :global(.latex-proof-content p:last-child) {
    margin-bottom: 0;
  }

  /* QED symbol at end of proof */
  .markdown-content :global(.latex-proof-content::after) {
    content: "∎";
    display: block;
    text-align: right;
    color: oklch(0.7 0 0);
    margin-top: 0.5rem;
  }

  /* LaTeX theorem-like environments */
  .markdown-content :global(.latex-theorem) {
    margin: 1rem 0;
    padding: 1rem 1.25rem;
    background: rgba(255, 215, 0, 0.03);
    border: 1px solid rgba(255, 215, 0, 0.15);
    border-radius: 0.375rem;
  }

  .markdown-content :global(.latex-theorem-header) {
    font-weight: 700;
    color: var(--exo-yellow, #ffd700);
    margin-bottom: 0.5rem;
  }

  .markdown-content :global(.latex-theorem-header::after) {
    content: ".";
  }

  .markdown-content :global(.latex-theorem-content) {
    color: oklch(0.9 0 0);
    font-style: italic;
  }

  .markdown-content :global(.latex-theorem-content p:last-child) {
    margin-bottom: 0;
  }

  /* LaTeX diagram/figure placeholder */
  .markdown-content :global(.latex-diagram-placeholder) {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 1rem 0;
    padding: 1.5rem 2rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px dashed rgba(255, 215, 0, 0.25);
    border-radius: 0.5rem;
    color: rgba(255, 215, 0, 0.6);
    font-size: 0.875rem;
  }

  .markdown-content :global(.latex-diagram-icon) {
    font-size: 1.25rem;
    opacity: 0.8;
  }

  .markdown-content :global(.latex-diagram-text) {
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
</style>

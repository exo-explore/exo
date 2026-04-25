<script lang="ts">
  import { models } from "$lib/api/models.svelte";

  interface BenchResult {
    runIndex: number;
    promptTps: number;
    generationTps: number;
    promptTokens: number;
    generationTokens: number;
    prefixCacheHit: "none" | "partial" | "exact";
    peakMemoryGb: number | null;
    avgPowerWatts: number | null;
    energyJoules: number | null;
    elapsedSeconds: number | null;
    durationMs: number;
  }

  const DEFAULT_PROMPT =
    "Explain how transformers handle long context windows in 4 short paragraphs.";

  let snap = $derived(models.value);
  let runningIds = $derived([...snap.runningModelIds]);
  let model = $state<string | null>(null);
  let prompt = $state(DEFAULT_PROMPT);
  let maxTokens = $state(256);
  let runs = $state(3);
  let usePrefixCache = $state(true);

  let inProgress = $state(false);
  let currentRun = $state(0);
  let results = $state<BenchResult[]>([]);
  let lastError = $state<string | null>(null);

  $effect(() => {
    if (model === null && runningIds.length > 0) {
      model = runningIds[0]!;
    }
    if (model !== null && !runningIds.includes(model)) {
      model = runningIds[0] ?? null;
    }
  });

  async function runOne(runIndex: number): Promise<BenchResult> {
    const t0 = performance.now();
    const res = await fetch("/bench/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens,
        stream: false,
        use_prefix_cache: usePrefixCache,
      }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`run ${runIndex + 1}: ${text || res.status}`);
    }
    const body = await res.json();
    const stats = body.generation_stats ?? {};
    const power = body.power_usage ?? null;
    const peakBytes: number | undefined =
      stats.peak_memory_usage?.in_bytes ?? stats.peak_memory_usage?.inBytes;
    return {
      runIndex,
      promptTps: stats.prompt_tps ?? 0,
      generationTps: stats.generation_tps ?? 0,
      promptTokens: stats.prompt_tokens ?? 0,
      generationTokens: stats.generation_tokens ?? 0,
      prefixCacheHit: stats.prefix_cache_hit ?? "none",
      peakMemoryGb: peakBytes ? peakBytes / 1024 ** 3 : null,
      avgPowerWatts: power?.total_avg_sys_power_watts ?? null,
      energyJoules: power?.total_energy_joules ?? null,
      elapsedSeconds: power?.elapsed_seconds ?? null,
      durationMs: performance.now() - t0,
    };
  }

  async function runBench() {
    if (!model) return;
    inProgress = true;
    lastError = null;
    results = [];
    currentRun = 0;
    try {
      for (let i = 0; i < runs; i++) {
        currentRun = i + 1;
        const r = await runOne(i);
        results = [...results, r];
      }
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      inProgress = false;
      currentRun = 0;
    }
  }

  function avg(get: (r: BenchResult) => number | null): number | null {
    const vals = results
      .map(get)
      .filter((v): v is number => v !== null && Number.isFinite(v));
    if (vals.length === 0) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  let avgPromptTps = $derived(avg((r) => r.promptTps));
  let avgGenTps = $derived(avg((r) => r.generationTps));
  let avgPower = $derived(avg((r) => r.avgPowerWatts));
  let avgPeakMem = $derived(avg((r) => r.peakMemoryGb));

  function fmt(v: number | null, digits = 1): string {
    if (v === null || !Number.isFinite(v)) return "—";
    return v.toFixed(digits);
  }
</script>

<div class="page-header">
  <div>
    <div class="eyebrow">BENCHMARK</div>
    <h1>One-click cluster benchmark.</h1>
    <div class="subtitle">
      Prompt-processing and token-generation tok/s, with optional prefix-cache testing.
    </div>
  </div>
</div>

{#if runningIds.length === 0}
  <div class="empty-block">
    <div class="empty-eyebrow">NO MODEL RUNNING</div>
    <div class="empty-title">Load a model to benchmark it.</div>
    <div class="empty-sub">
      Open <a href="#/models">Models</a> and launch one — the benchmark calls the same inference path your real requests use.
    </div>
  </div>
{:else}
  <div class="form-card">
    <div class="form-grid">
      <label class="field">
        <span class="field-label">MODEL</span>
        <select bind:value={model} disabled={inProgress}>
          {#each runningIds as id}
            <option value={id}>{id}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        <span class="field-label">RUNS</span>
        <input
          type="number"
          min="1"
          max="20"
          bind:value={runs}
          disabled={inProgress}
        />
      </label>
      <label class="field">
        <span class="field-label">MAX TOKENS</span>
        <input
          type="number"
          min="1"
          max="4096"
          bind:value={maxTokens}
          disabled={inProgress}
        />
      </label>
      <label class="field checkbox">
        <input
          type="checkbox"
          bind:checked={usePrefixCache}
          disabled={inProgress}
        />
        <span class="field-label">Use prefix cache</span>
      </label>
    </div>
    <label class="field full">
      <span class="field-label">PROMPT</span>
      <textarea bind:value={prompt} rows="3" disabled={inProgress}></textarea>
    </label>
    <div class="form-actions">
      <button
        class="btn primary"
        disabled={inProgress || !model || prompt.trim().length === 0}
        onclick={runBench}
      >
        {#if inProgress}
          Running run {currentRun} / {runs}…
        {:else}
          Run benchmark
        {/if}
      </button>
      {#if results.length > 0 && !inProgress}
        <span class="muted">{results.length} runs complete</span>
      {/if}
    </div>
  </div>

  {#if lastError}
    <div class="error-banner">
      <span class="error-tag">ERROR</span>
      {lastError}
      <button class="dismiss" onclick={() => (lastError = null)}>×</button>
    </div>
  {/if}

  {#if results.length > 0}
    <div class="summary">
      <div class="summary-card">
        <div class="summary-label">PROMPT TOK/S</div>
        <div class="summary-value">{fmt(avgPromptTps)}</div>
        <div class="summary-foot">avg of {results.length} runs</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">GEN TOK/S</div>
        <div class="summary-value accent">{fmt(avgGenTps)}</div>
        <div class="summary-foot">avg of {results.length} runs</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">PEAK MEMORY</div>
        <div class="summary-value">
          {fmt(avgPeakMem)}
          <span class="unit">GB</span>
        </div>
        <div class="summary-foot">avg of {results.length} runs</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">POWER</div>
        <div class="summary-value">
          {avgPower !== null ? fmt(avgPower, 1) : "—"}
          <span class="unit">W</span>
        </div>
        <div class="summary-foot">total cluster avg</div>
      </div>
    </div>

    <div class="runs-card">
      <div class="runs-header">
        <span class="runs-title">Per-run breakdown</span>
        <span class="runs-hint">Newest at top.</span>
      </div>
      <table class="runs-table">
        <thead>
          <tr>
            <th>RUN</th>
            <th>PROMPT TOK/S</th>
            <th>GEN TOK/S</th>
            <th>TOKENS</th>
            <th>PREFIX</th>
            <th>PEAK MEM</th>
            <th>POWER</th>
            <th>WALL</th>
          </tr>
        </thead>
        <tbody>
          {#each [...results].reverse() as r}
            <tr>
              <td>#{r.runIndex + 1}</td>
              <td>{fmt(r.promptTps)}</td>
              <td class="accent">{fmt(r.generationTps)}</td>
              <td class="muted">
                {r.promptTokens} → {r.generationTokens}
              </td>
              <td>
                <span class="prefix-tag" data-state={r.prefixCacheHit}>
                  {r.prefixCacheHit}
                </span>
              </td>
              <td>{fmt(r.peakMemoryGb)} GB</td>
              <td>
                {r.avgPowerWatts !== null ? `${fmt(r.avgPowerWatts)} W` : "—"}
              </td>
              <td class="muted">{(r.durationMs / 1000).toFixed(1)} s</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
{/if}

<style>
  .page-header {
    margin-bottom: 28px;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ux-text-faint);
    font-size: 10px;
    font-weight: 600;
    font-family: var(--ux-mono);
    margin-bottom: 6px;
  }
  h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ux-text);
  }
  .subtitle {
    color: var(--ux-text-dim);
    font-size: 13px;
    margin-top: 6px;
  }

  .empty-block {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 60px 24px;
    text-align: center;
  }
  .empty-eyebrow {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.14em;
    margin-bottom: 12px;
  }
  .empty-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--ux-text);
    margin-bottom: 8px;
  }
  .empty-sub {
    font-size: 13px;
    color: var(--ux-text-dim);
  }
  .empty-sub a {
    color: var(--ux-text);
    text-decoration: none;
    border-bottom: 1px dashed var(--ux-border-strong);
  }
  .empty-sub a:hover {
    border-bottom-color: var(--ux-text);
  }

  .form-card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 18px 20px;
    margin-bottom: 16px;
  }
  .form-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr auto;
    gap: 14px;
    margin-bottom: 14px;
  }
  @media (max-width: 720px) {
    .form-grid {
      grid-template-columns: 1fr 1fr;
    }
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .field.checkbox {
    flex-direction: row;
    align-items: center;
    gap: 8px;
    margin-top: 22px;
  }
  .field.full {
    margin-bottom: 14px;
  }
  .field-label {
    font-family: var(--ux-mono);
    font-size: 9.5px;
    color: var(--ux-text-faint);
    letter-spacing: 0.12em;
  }
  .field input[type="number"],
  .field select,
  .field textarea {
    background: var(--ux-bg-raised);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius-sm);
    color: var(--ux-text);
    font: inherit;
    font-family: var(--ux-mono);
    font-size: 12.5px;
    padding: 8px 10px;
    outline: none;
    width: 100%;
  }
  .field textarea {
    resize: vertical;
    min-height: 70px;
    line-height: 1.45;
  }
  .field select {
    appearance: none;
    cursor: pointer;
  }
  .field input:disabled,
  .field select:disabled,
  .field textarea:disabled {
    opacity: 0.55;
  }
  .field input[type="checkbox"] {
    accent-color: var(--ux-text);
    width: 14px;
    height: 14px;
    margin: 0;
  }

  .form-actions {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .btn {
    font-family: var(--ux-sans);
    font-size: 13px;
    font-weight: 500;
    padding: 9px 18px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: var(--ux-card);
    color: var(--ux-text);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--ux-bg-hover);
  }
  .btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--ux-text);
    color: var(--ux-text-invert);
    border-color: var(--ux-text);
    font-weight: 600;
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--ux-primary-hover);
    border-color: var(--ux-primary-hover);
  }
  .muted {
    font-family: var(--ux-mono);
    font-size: 11px;
    color: var(--ux-text-faint);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: var(--ux-red-bg);
    border: 1px solid var(--ux-red-border);
    border-radius: var(--ux-radius-sm);
    margin-bottom: 16px;
    font-family: var(--ux-mono);
    font-size: 12px;
    color: var(--ux-red-text);
  }
  .error-tag {
    background: var(--ux-red-bg);
    color: var(--ux-red);
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.08em;
  }
  .dismiss {
    margin-left: auto;
    background: transparent;
    border: none;
    color: var(--ux-red-text);
    cursor: pointer;
    font-size: 18px;
    line-height: 1;
    padding: 0 4px;
  }

  .summary {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }
  @media (max-width: 720px) {
    .summary {
      grid-template-columns: 1fr 1fr;
    }
  }
  .summary-card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 16px 18px;
  }
  .summary-label {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.12em;
    margin-bottom: 8px;
  }
  .summary-value {
    font-family: var(--ux-mono);
    font-size: 26px;
    font-weight: 500;
    color: var(--ux-text);
    line-height: 1;
  }
  .summary-value.accent {
    color: var(--ux-text);
  }
  .summary-value .unit {
    font-size: 13px;
    color: var(--ux-text-faint);
    margin-left: 2px;
    font-weight: 400;
  }
  .summary-foot {
    margin-top: 8px;
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
  }

  .runs-card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 16px 0 8px;
  }
  .runs-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0 18px 12px;
    border-bottom: 1px solid var(--ux-border);
  }
  .runs-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--ux-text);
  }
  .runs-hint {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.1em;
  }
  .runs-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--ux-mono);
    font-size: 12px;
  }
  .runs-table th {
    text-align: left;
    padding: 10px 18px;
    color: var(--ux-text-faint);
    font-size: 9.5px;
    letter-spacing: 0.12em;
    font-weight: 600;
    border-bottom: 1px solid var(--ux-border);
  }
  .runs-table td {
    padding: 10px 18px;
    color: var(--ux-text);
    border-bottom: 1px solid var(--ux-border);
  }
  .runs-table tr:last-child td {
    border-bottom: none;
  }
  .runs-table .accent {
    color: var(--ux-text);
    font-weight: 500;
  }
  .runs-table .muted {
    color: var(--ux-text-dim);
  }
  .prefix-tag {
    font-family: var(--ux-mono);
    font-size: 9.5px;
    padding: 2px 6px;
    border-radius: 3px;
    background: var(--ux-bg-raised);
    color: var(--ux-text-faint);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .prefix-tag[data-state="exact"] {
    background: var(--ux-green-bg);
    color: var(--ux-green);
  }
  .prefix-tag[data-state="partial"] {
    background: var(--ux-accent-bg);
    color: var(--ux-accent);
  }
</style>

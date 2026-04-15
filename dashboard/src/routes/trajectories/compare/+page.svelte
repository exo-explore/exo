<script lang="ts">
  import { page } from "$app/stores";
  import { onMount } from "svelte";
  import {
    getTrajectory,
    type AtifTrajectory,
    type AtifStep,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  const idA = $derived($page.url.searchParams.get("a") ?? "");
  const idB = $derived($page.url.searchParams.get("b") ?? "");

  let trajectoryA = $state<AtifTrajectory | null>(null);
  let trajectoryB = $state<AtifTrajectory | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  async function load() {
    loading = true;
    error = null;
    try {
      const [a, b] = await Promise.all([getTrajectory(idA), getTrajectory(idB)]);
      trajectoryA = a;
      trajectoryB = b;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load trajectories";
    } finally {
      loading = false;
    }
  }

  onMount(load);

  function delta(a: number, b: number): { value: string; cls: string } {
    const d = b - a;
    const sign = d > 0 ? "+" : "";
    const cls =
      d === 0
        ? "text-exo-light-gray"
        : d > 0
          ? "text-red-400"
          : "text-green-400";
    return { value: `${sign}${d}`, cls };
  }

  type AlignedRow = { a: AtifStep | null; b: AtifStep | null };

  function alignByStepId(
    a: AtifTrajectory | null,
    b: AtifTrajectory | null,
  ): AlignedRow[] {
    if (!a && !b) return [];
    const maxLen = Math.max(a?.steps.length ?? 0, b?.steps.length ?? 0);
    const rows: AlignedRow[] = [];
    for (let i = 0; i < maxLen; i++) {
      rows.push({
        a: a?.steps[i] ?? null,
        b: b?.steps[i] ?? null,
      });
    }
    return rows;
  }

  const rows = $derived(alignByStepId(trajectoryA, trajectoryB));
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div>
      <a
        href="#/trajectories"
        class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow uppercase"
      >
        &larr; Trajectories
      </a>
      <h1
        class="mt-2 text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow"
      >
        Compare
      </h1>
    </div>

    {#if loading}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        Loading...
      </div>
    {:else if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-6 text-center text-red-400"
      >
        {error}
      </div>
    {:else if trajectoryA && trajectoryB}
      <div class="grid grid-cols-2 gap-6">
        <div
          class="rounded border border-exo-yellow/40 bg-exo-black/30 p-4 text-xs font-mono space-y-1"
        >
          <div class="text-exo-yellow uppercase">A</div>
          <div class="truncate text-white">{trajectoryA.session_id}</div>
          <div class="text-exo-light-gray">
            model: {trajectoryA.agent.model}
          </div>
          <div class="text-exo-light-gray">
            steps: {trajectoryA.final_metrics.total_steps}
            &bull; prompt_tokens: {trajectoryA.final_metrics.total_prompt_tokens}
            &bull; completion_tokens: {trajectoryA.final_metrics.total_completion_tokens}
          </div>
        </div>
        <div
          class="rounded border border-exo-yellow/40 bg-exo-black/30 p-4 text-xs font-mono space-y-1"
        >
          <div class="text-exo-yellow uppercase">B</div>
          <div class="truncate text-white">{trajectoryB.session_id}</div>
          <div class="text-exo-light-gray">
            model: {trajectoryB.agent.model}
          </div>
          <div class="text-exo-light-gray">
            steps: {trajectoryB.final_metrics.total_steps}
            &bull; prompt_tokens: {trajectoryB.final_metrics.total_prompt_tokens}
            &bull; completion_tokens: {trajectoryB.final_metrics.total_completion_tokens}
          </div>
        </div>
      </div>

      {@const promptDelta = delta(
        trajectoryA.final_metrics.total_prompt_tokens,
        trajectoryB.final_metrics.total_prompt_tokens,
      )}
      {@const completionDelta = delta(
        trajectoryA.final_metrics.total_completion_tokens,
        trajectoryB.final_metrics.total_completion_tokens,
      )}
      {@const stepsDelta = delta(
        trajectoryA.final_metrics.total_steps,
        trajectoryB.final_metrics.total_steps,
      )}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/20 p-4 text-xs font-mono flex flex-wrap gap-4"
      >
        <span>steps Δ <span class={stepsDelta.cls}>{stepsDelta.value}</span></span>
        <span
          >prompt_tokens Δ <span class={promptDelta.cls}>{promptDelta.value}</span></span
        >
        <span
          >completion_tokens Δ <span class={completionDelta.cls}
            >{completionDelta.value}</span
          ></span
        >
      </div>

      <div class="space-y-3">
        {#each rows as row, i}
          <div class="grid grid-cols-2 gap-6">
            {#each [row.a, row.b] as step, sideIdx}
              <div
                class="rounded border border-exo-medium-gray/30 bg-exo-black/20 p-3 text-xs font-mono space-y-1"
              >
                <div class="flex items-center gap-2 uppercase">
                  <span class="text-exo-yellow">{sideIdx === 0 ? "A" : "B"}</span>
                  <span class="text-exo-light-gray">#{i + 1}</span>
                  {#if step}
                    <span class="text-exo-light-gray">{step.source}</span>
                  {:else}
                    <span class="text-exo-light-gray/50">—</span>
                  {/if}
                </div>
                {#if step}
                  <pre class="whitespace-pre-wrap text-white/90 text-sm">{step.message}</pre>
                  {#if step.metrics}
                    <div class="flex flex-wrap gap-1 text-[10px]">
                      <span
                        class="px-1 py-0.5 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                        >p {step.metrics.prompt_tokens}</span
                      >
                      <span
                        class="px-1 py-0.5 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                        >c {step.metrics.completion_tokens}</span
                      >
                      {#if step.metrics._exo_extensions?.generation_tps !== undefined}
                        <span
                          class="px-1 py-0.5 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                          >tps {step.metrics._exo_extensions.generation_tps.toFixed(
                            1,
                          )}</span
                        >
                      {/if}
                    </div>
                  {/if}
                {/if}
              </div>
            {/each}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

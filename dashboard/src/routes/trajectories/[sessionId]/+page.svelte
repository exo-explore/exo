<script lang="ts">
  import { page } from "$app/stores";
  import { onMount } from "svelte";
  import {
    getTrajectory,
    type AtifTrajectory,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  const sessionId = $derived($page.params.sessionId);

  let trajectory = $state<AtifTrajectory | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  async function load() {
    loading = true;
    error = null;
    try {
      trajectory = await getTrajectory(sessionId);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load trajectory";
    } finally {
      loading = false;
    }
  }

  onMount(load);

  function formatTimestamp(iso: string): string {
    return new Date(iso).toLocaleString();
  }
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-5xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div>
      <a
        href="#/trajectories"
        class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow uppercase"
      >
        &larr; Trajectories
      </a>
      <h1
        class="mt-2 text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow truncate"
      >
        {sessionId}
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
    {:else if trajectory}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 text-xs font-mono text-exo-light-gray space-y-1"
      >
        <div>schema: {trajectory.schema_version}</div>
        <div>agent: {trajectory.agent.name} / {trajectory.agent.model}</div>
        <div>
          totals: steps={trajectory.final_metrics.total_steps}
          &bull; prompt_tokens={trajectory.final_metrics.total_prompt_tokens}
          &bull; completion_tokens={trajectory.final_metrics.total_completion_tokens}
          &bull; cost=${trajectory.final_metrics.total_cost.toFixed(4)}
        </div>
      </div>

      <div class="space-y-3">
        {#each trajectory.steps as step}
          <div
            class="rounded border border-exo-medium-gray/30 bg-exo-black/20 p-4 space-y-2"
          >
            <div class="flex items-center gap-3 text-xs font-mono uppercase">
              <span class="text-exo-yellow">#{step.step_id}</span>
              <span class="text-exo-light-gray">{step.source}</span>
              {#if step.model_name}
                <span class="text-exo-light-gray/70">{step.model_name}</span>
              {/if}
              <span class="text-exo-light-gray/60 ml-auto">
                {formatTimestamp(step.timestamp)}
              </span>
            </div>
            {#if step.message}
              <pre
                class="whitespace-pre-wrap text-sm font-mono text-white/90">{step.message}</pre>
            {/if}
            {#if step.reasoning_content}
              <details
                class="text-xs font-mono text-exo-light-gray/80 border border-exo-medium-gray/30 rounded p-2"
              >
                <summary class="cursor-pointer uppercase"
                  >reasoning_content</summary
                >
                <pre class="whitespace-pre-wrap mt-2">{step.reasoning_content}</pre>
              </details>
            {/if}
            {#if step.tool_calls && step.tool_calls.length > 0}
              <div class="space-y-1">
                {#each step.tool_calls as tc}
                  <div
                    class="rounded border border-exo-yellow/40 bg-exo-yellow/5 p-2 text-xs font-mono"
                  >
                    <div class="text-exo-yellow uppercase">
                      tool_call: {tc.function_name}
                    </div>
                    <pre
                      class="whitespace-pre-wrap mt-1">{JSON.stringify(tc.arguments, null, 2)}</pre>
                  </div>
                {/each}
              </div>
            {/if}
            {#if step.observation && step.observation.results.length > 0}
              <div class="space-y-1">
                {#each step.observation.results as r}
                  <div
                    class="rounded border border-green-500/40 bg-green-500/5 p-2 text-xs font-mono"
                  >
                    <div class="text-green-400 uppercase">
                      observation: {r.source_call_id}
                    </div>
                    <pre class="whitespace-pre-wrap mt-1">{r.content}</pre>
                  </div>
                {/each}
              </div>
            {/if}
            {#if step.metrics}
              <div class="flex flex-wrap gap-2 text-[10px] font-mono uppercase">
                <span
                  class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                  >prompt {step.metrics.prompt_tokens}</span
                >
                <span
                  class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                  >completion {step.metrics.completion_tokens}</span
                >
                <span
                  class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                  >cached {step.metrics.cached_tokens}</span
                >
                {#if step.metrics._exo_extensions?.prompt_tps !== undefined}
                  <span
                    class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                    >prompt_tps {step.metrics._exo_extensions.prompt_tps.toFixed(
                      1,
                    )}</span
                  >
                {/if}
                {#if step.metrics._exo_extensions?.generation_tps !== undefined}
                  <span
                    class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                    >gen_tps {step.metrics._exo_extensions.generation_tps.toFixed(
                      1,
                    )}</span
                  >
                {/if}
                {#if step.metrics._exo_extensions?.prefix_cache_hit}
                  <span
                    class="px-2 py-1 rounded bg-exo-medium-gray/20 text-exo-light-gray"
                    >prefix_cache {step.metrics._exo_extensions
                      .prefix_cache_hit}</span
                  >
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

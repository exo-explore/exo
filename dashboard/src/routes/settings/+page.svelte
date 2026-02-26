<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import { settingsStore, type ExoSettings } from "$lib/stores/settings.svelte";
  import { addToast } from "$lib/stores/toast.svelte";

  let draft = $state<ExoSettings | null>(null);
  const loading = $derived(settingsStore.loading);

  onMount(async () => {
    await settingsStore.load();
    draft = structuredClone(settingsStore.settings);
  });

  async function handleSave() {
    if (!draft) return;
    const ok = await settingsStore.save(draft);
    if (ok) {
      addToast({ type: "success", message: "Settings saved" });
    } else {
      addToast({ type: "error", message: settingsStore.error ?? "Failed to save settings" });
    }
  }

  function handleReset() {
    draft = settingsStore.resetToDefaults();
  }

  const KV_OPTIONS: { label: string; value: 4 | 8 | null }[] = [
    { label: "None (full precision)", value: null },
    { label: "4-bit", value: 4 },
    { label: "8-bit", value: 8 },
  ];
</script>

<HeaderNav showHome={true} />

{#if draft}
  <div class="min-h-screen bg-background text-foreground" in:fade={{ duration: 200 }}>
    <div class="max-w-2xl mx-auto px-6 py-8">
      <h1 class="text-2xl font-bold text-exo-yellow tracking-wider uppercase mb-8">Settings</h1>

      <!-- Memory / Safety -->
      <section class="mb-10">
        <h2 class="text-sm font-semibold text-white/50 tracking-widest uppercase mb-4">Memory / Safety</h2>
        <div class="space-y-5">
          <!-- OOM Prevention Toggle -->
          <div class="flex items-center justify-between">
            <div>
              <div class="text-sm text-white/90">OOM Prevention</div>
              <div class="text-xs text-white/40 mt-0.5">Stop generation when memory is low</div>
            </div>
            <button
              onclick={() => { if (draft) draft.memory.oom_prevention = !draft.memory.oom_prevention; }}
              class="relative w-11 h-6 rounded-full transition-colors duration-200 cursor-pointer {draft.memory.oom_prevention ? 'bg-exo-yellow' : 'bg-exo-medium-gray'}"
              role="switch"
              aria-checked={draft.memory.oom_prevention}
            >
              <span
                class="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200 {draft.memory.oom_prevention ? 'translate-x-5' : 'translate-x-0'}"
              ></span>
            </button>
          </div>

          <!-- Memory Threshold Slider -->
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <div>
                <div class="text-sm text-white/90">Memory Threshold</div>
                <div class="text-xs text-white/40 mt-0.5">KV cache eviction triggers above this level</div>
              </div>
              <span class="text-sm font-mono text-exo-yellow">{(draft.memory.memory_threshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.99"
              step="0.01"
              bind:value={draft.memory.memory_threshold}
              class="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-exo-medium-gray accent-exo-yellow"
            />
          </div>

          <!-- Memory Floor -->
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <div>
                <div class="text-sm text-white/90">Memory Floor</div>
                <div class="text-xs text-white/40 mt-0.5">Minimum free memory to reserve (GB)</div>
              </div>
              <span class="text-sm font-mono text-exo-yellow">{draft.memory.memory_floor_gb.toFixed(1)} GB</span>
            </div>
            <input
              type="number"
              min="0"
              max="64"
              step="0.5"
              bind:value={draft.memory.memory_floor_gb}
              class="w-full bg-exo-medium-gray border border-exo-light-gray/20 rounded px-3 py-1.5 text-sm text-white/90 font-mono focus:outline-none focus:border-exo-yellow/50"
            />
          </div>
        </div>
      </section>

      <!-- Generation / Performance -->
      <section class="mb-10">
        <h2 class="text-sm font-semibold text-white/50 tracking-widest uppercase mb-4">Generation / Performance</h2>
        <div class="space-y-5">
          <!-- Prefill Step Size -->
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <div>
                <div class="text-sm text-white/90">Prefill Step Size</div>
                <div class="text-xs text-white/40 mt-0.5">Token chunk size during prompt processing</div>
              </div>
              <span class="text-sm font-mono text-exo-yellow">{draft.generation.prefill_step_size.toLocaleString()}</span>
            </div>
            <input
              type="number"
              min="128"
              max="32768"
              step="128"
              bind:value={draft.generation.prefill_step_size}
              class="w-full bg-exo-medium-gray border border-exo-light-gray/20 rounded px-3 py-1.5 text-sm text-white/90 font-mono focus:outline-none focus:border-exo-yellow/50"
            />
          </div>

          <!-- Max Tokens -->
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <div>
                <div class="text-sm text-white/90">Max Tokens</div>
                <div class="text-xs text-white/40 mt-0.5">Maximum generation length per response</div>
              </div>
              <span class="text-sm font-mono text-exo-yellow">{draft.generation.max_tokens.toLocaleString()}</span>
            </div>
            <input
              type="number"
              min="1"
              max="131072"
              step="1024"
              bind:value={draft.generation.max_tokens}
              class="w-full bg-exo-medium-gray border border-exo-light-gray/20 rounded px-3 py-1.5 text-sm text-white/90 font-mono focus:outline-none focus:border-exo-yellow/50"
            />
          </div>

          <!-- KV Cache Bits -->
          <div>
            <div class="mb-1.5">
              <div class="text-sm text-white/90">KV Cache Quantization</div>
              <div class="text-xs text-white/40 mt-0.5">Lower bits save memory at slight quality cost</div>
            </div>
            <select
              bind:value={draft.generation.kv_cache_bits}
              class="w-full bg-exo-medium-gray border border-exo-light-gray/20 rounded px-3 py-1.5 text-sm text-white/90 font-mono focus:outline-none focus:border-exo-yellow/50 cursor-pointer"
            >
              {#each KV_OPTIONS as opt}
                <option value={opt.value}>{opt.label}</option>
              {/each}
            </select>
          </div>
        </div>
      </section>

      <!-- Action Buttons -->
      <div class="flex items-center gap-3">
        <button
          onclick={handleSave}
          disabled={loading}
          class="px-5 py-2 rounded text-sm font-semibold tracking-wider uppercase transition-colors cursor-pointer
            bg-exo-yellow text-exo-black hover:bg-exo-yellow-darker
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Saving..." : "Save"}
        </button>
        <button
          onclick={handleReset}
          disabled={loading}
          class="px-5 py-2 rounded text-sm font-semibold tracking-wider uppercase transition-colors cursor-pointer
            border border-exo-light-gray/30 text-white/70 hover:border-exo-yellow/50 hover:text-exo-yellow
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reset to Defaults
        </button>
      </div>
    </div>
  </div>
{:else}
  <div class="min-h-screen bg-background flex items-center justify-center">
    <div class="text-white/40 text-sm">Loading settings...</div>
  </div>
{/if}

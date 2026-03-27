<script lang="ts">
  import { browser } from "$app/environment";
  import { fade } from "svelte/transition";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import IntegrationCard from "$lib/components/IntegrationCard.svelte";
  import { instances, refreshState } from "$lib/stores/app.svelte";
  import { onMount } from "svelte";

  const apiUrl = browser ? window.location.origin : "http://localhost:52415";

  const instancesData = $derived(instances());

  let modelCapabilities = $state<Record<string, string[]>>({});
  let modelContextLengths = $state<Record<string, number>>({});

  const runningModels = $derived.by(() => {
    const models: string[] = [];
    for (const [, wrapper] of Object.entries(instancesData)) {
      if (wrapper && typeof wrapper === "object") {
        const values = Object.values(wrapper as Record<string, unknown>);
        if (values.length > 0) {
          const instance = values[0];
          if (instance && typeof instance === "object") {
            const inst = instance as {
              shardAssignments?: { modelId?: string };
            };
            const modelId = inst.shardAssignments?.modelId;
            if (modelId && !models.includes(modelId)) {
              models.push(modelId);
            }
          }
        }
      }
    }
    return models;
  });

  function estimateParamSize(modelId: string): number {
    const match = modelId.match(/(\d+(?:\.\d+)?)[Bb]/);
    return match ? parseFloat(match[1]) : 0;
  }

  const modelsBySize = $derived(
    [...runningModels].sort(
      (a, b) => estimateParamSize(b) - estimateParamSize(a),
    ),
  );

  const defaultTiers = $derived.by(() => {
    const n = modelsBySize.length;
    if (n === 0)
      return {
        opus: "your-model-id",
        sonnet: "your-model-id",
        haiku: "your-model-id",
      };
    if (n === 1)
      return {
        opus: modelsBySize[0],
        sonnet: modelsBySize[0],
        haiku: modelsBySize[0],
      };
    if (n === 2)
      return {
        opus: modelsBySize[0],
        sonnet: modelsBySize[1],
        haiku: modelsBySize[1],
      };
    return {
      opus: modelsBySize[0],
      sonnet: modelsBySize[Math.floor(n / 2)],
      haiku: modelsBySize[n - 1],
    };
  });

  let opusModel = $state("");
  let sonnetModel = $state("");
  let haikuModel = $state("");

  $effect(() => {
    opusModel = defaultTiers.opus;
    sonnetModel = defaultTiers.sonnet;
    haikuModel = defaultTiers.haiku;
  });

  let codexModel = $state("");
  let openClawModel = $state("");
  let openClawToolsProfile = $state("coding");
  $effect(() => {
    const def = modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id";
    codexModel = def;
    openClawModel = def;
  });

  const claudeShellCommand = $derived(
    [
      `ANTHROPIC_BASE_URL=${apiUrl} \\`,
      `ANTHROPIC_API_KEY=x \\`,
      `ANTHROPIC_DEFAULT_OPUS_MODEL=${opusModel} \\`,
      `ANTHROPIC_DEFAULT_SONNET_MODEL=${sonnetModel} \\`,
      `ANTHROPIC_DEFAULT_HAIKU_MODEL=${haikuModel} \\`,
      `API_TIMEOUT_MS=3000000 \\`,
      `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \\`,
      `claude`,
    ].join("\n"),
  );

  const claudeSettingsJson = $derived(
    JSON.stringify(
      {
        env: {
          ANTHROPIC_BASE_URL: apiUrl,
          ANTHROPIC_API_KEY: "x",
          ANTHROPIC_DEFAULT_OPUS_MODEL: opusModel,
          ANTHROPIC_DEFAULT_SONNET_MODEL: sonnetModel,
          ANTHROPIC_DEFAULT_HAIKU_MODEL: haikuModel,
          API_TIMEOUT_MS: "3000000",
          CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
        },
      },
      null,
      2,
    ),
  );

  const openCodeConfig = $derived.by(() => {
    const models: Record<string, Record<string, unknown>> = {};
    for (const modelId of runningModels) {
      const caps = modelCapabilities[modelId] || [];
      const ctxLen = modelContextLengths[modelId] || 0;
      const entry: Record<string, unknown> = { name: modelId };
      if (ctxLen > 0) {
        entry.limit = { context: ctxLen, output: Math.min(ctxLen, 16384) };
      }
      if (caps.includes("vision")) {
        entry.modalities = { input: ["text", "image"], output: ["text"] };
      }
      models[modelId] = entry;
    }
    if (Object.keys(models).length === 0) {
      models["your-model-id"] = { name: "your-model-name" };
    }
    const firstModel =
      runningModels.length > 0 ? runningModels[0] : "your-model-id";
    return JSON.stringify(
      {
        $schema: "https://opencode.ai/config.json",
        provider: {
          exo: {
            npm: "@ai-sdk/openai-compatible",
            name: "exo",
            options: {
              baseURL: `${apiUrl}/v1`,
              apiKey: "x",
            },
            models,
          },
        },
        model: `exo/${firstModel}`,
      },
      null,
      2,
    );
  });

  const codexShellCommand = $derived(`EXO_API_KEY=x npx @openai/codex`);

  const codexConfig = $derived(
    [
      `model = "${codexModel}"`,
      `model_provider = "exo"`,
      ``,
      `[model_providers.exo]`,
      `name = "exo"`,
      `base_url = "${apiUrl}/v1"`,
      `env_key = "EXO_API_KEY"`,
    ].join("\n"),
  );

  const openClawConfig = $derived(
    JSON.stringify(
      {
        model: openClawModel,
        modelProvider: {
          name: "exo",
          baseURL: `${apiUrl}/v1`,
          apiKey: "x",
        },
        toolsProfile: openClawToolsProfile,
      },
      null,
      2,
    ),
  );

  const ollamaCommand = $derived(
    `OLLAMA_HOST=${apiUrl}/ollama ollama run ${modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id"}`,
  );

  const n8nConfig = $derived.by(() => {
    const steps = [
      "1. In n8n, go to Credentials → New Credential → OpenAI API",
      `2. Set API Key to: x`,
      `3. Set Base URL to: ${apiUrl}/v1`,
      "4. Save the credential",
      `5. In your AI Agent or LLM Chain node, use the OpenAI Chat Model sub-node`,
      `6. Enter model name: ${modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id"}`,
    ];
    return steps.join("\n");
  });

  const tabs = [
    "Claude Code",
    "OpenCode",
    "Codex",
    "OpenClaw",
    "Ollama",
    "n8n",
  ] as const;
  type Tab = (typeof tabs)[number];
  const stored = browser ? localStorage.getItem("exo-integrations-tab") : null;
  let activeTab = $state<Tab>(
    stored && tabs.includes(stored as Tab) ? (stored as Tab) : "Claude Code",
  );
  $effect(() => {
    if (browser) localStorage.setItem("exo-integrations-tab", activeTab);
  });

  const selectClass =
    "bg-black/30 border border-exo-light-gray/20 rounded px-2 py-1.5 text-white font-mono text-xs focus:border-exo-yellow/50 focus:outline-none appearance-none cursor-pointer";

  onMount(async () => {
    refreshState();
    try {
      const resp = await fetch("/v1/models");
      const data = (await resp.json()) as {
        data: { id: string; capabilities: string[]; context_length: number }[];
      };
      const caps: Record<string, string[]> = {};
      const ctxs: Record<string, number> = {};
      for (const model of data.data) {
        caps[model.id] = model.capabilities || [];
        if (model.context_length > 0) ctxs[model.id] = model.context_length;
      }
      modelCapabilities = caps;
      modelContextLengths = ctxs;
    } catch {
      /* ignore */
    }
  });
</script>

<div class="min-h-screen bg-exo-dark-gray flex flex-col">
  <HeaderNav showHome={true} />

  <main
    class="flex-1 max-w-3xl mx-auto w-full px-4 md:px-6 py-8"
    in:fade={{ duration: 200 }}
  >
    <div class="mb-8">
      <h1
        class="text-white text-xl md:text-2xl font-semibold tracking-wide mb-2"
      >
        Integrations
      </h1>
      <p class="text-exo-light-gray/60 text-sm">
        Connect external tools to your exo cluster.
      </p>
    </div>

    <!-- Status -->
    <div class="mb-8">
      <span class="text-exo-light-gray/70 text-xs uppercase tracking-wider"
        >API Endpoint</span
      >
      <span class="text-white font-mono text-sm ml-2">{apiUrl}</span>
      {#if runningModels.length > 0}
        <div class="text-exo-light-gray/50 text-xs mt-2">
          Running model{runningModels.length > 1 ? "s" : ""}:
          <ul class="mt-1 space-y-0.5 list-none">
            {#each runningModels as model}
              <li class="text-exo-yellow font-mono">{model}</li>
            {/each}
          </ul>
        </div>
      {:else}
        <p class="text-exo-light-gray/40 text-xs mt-2 italic">
          No models currently running
        </p>
      {/if}
    </div>

    <!-- API Endpoints -->
    <div class="mb-8">
      <div
        class="flex flex-col sm:flex-row gap-3 text-xs font-mono text-exo-light-gray/70"
      >
        <div
          class="flex-1 bg-black/20 border border-exo-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-exo-light-gray/40 text-[10px] uppercase block mb-1"
            >OpenAI-compatible</span
          >
          <span class="text-white/80">{apiUrl}/v1</span>
        </div>
        <div
          class="flex-1 bg-black/20 border border-exo-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-exo-light-gray/40 text-[10px] uppercase block mb-1"
            >Claude-compatible</span
          >
          <span class="text-white/80">{apiUrl}</span>
        </div>
        <div
          class="flex-1 bg-black/20 border border-exo-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-exo-light-gray/40 text-[10px] uppercase block mb-1"
            >Ollama-compatible</span
          >
          <span class="text-white/80">{apiUrl}/ollama</span>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <div
      class="flex flex-wrap gap-2 mb-6 border-b border-exo-light-gray/10 pb-3"
    >
      {#each tabs as tab}
        <button
          onclick={() => (activeTab = tab)}
          class="px-3 py-1.5 text-xs rounded-md transition-all cursor-pointer
            {activeTab === tab
            ? 'bg-exo-yellow/15 text-exo-yellow border border-exo-yellow/30'
            : 'text-exo-light-gray/60 hover:text-white/80 border border-transparent hover:border-exo-light-gray/20'}"
        >
          {tab}
        </button>
      {/each}
    </div>

    <!-- Tab Content -->
    <div class="space-y-4">
      {#if activeTab === "Claude Code"}
        {#if runningModels.length > 1}
          <div class="grid grid-cols-3 gap-3 text-xs">
            {#each [{ label: "Opus", bind: () => opusModel, set: (v: string) => (opusModel = v) }, { label: "Sonnet", bind: () => sonnetModel, set: (v: string) => (sonnetModel = v) }, { label: "Haiku", bind: () => haikuModel, set: (v: string) => (haikuModel = v) }] as tier}
              <div>
                <span
                  class="text-exo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
                  >{tier.label}</span
                >
                <select
                  value={tier.bind()}
                  onchange={(e) =>
                    tier.set((e.target as HTMLSelectElement).value)}
                  class="w-full {selectClass}"
                >
                  {#each runningModels as model}
                    <option value={model}>{model.split("/").pop()}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>
        {/if}
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Launch Claude Code with exo as the backend. Paste this into your terminal."
          config={claudeShellCommand}
          language="bash"
        />
        <IntegrationCard
          title="Settings File"
          subtitle="~/.claude/settings.json"
          description="Or add this to your Claude Code settings for persistent configuration."
          config={claudeSettingsJson}
        />
      {:else if activeTab === "OpenCode"}
        <IntegrationCard
          title="Config File"
          subtitle="opencode.json"
          description="Add this to your project root or ~/.config/opencode/opencode.json for global config. Vision models automatically get image input modality."
          config={openCodeConfig}
        />
      {:else if activeTab === "Codex"}
        {#if runningModels.length > 1}
          <div class="text-xs">
            <span
              class="text-exo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >Model</span
            >
            <select bind:value={codexModel} class={selectClass}>
              {#each runningModels as model}
                <option value={model}>{model.split("/").pop()}</option>
              {/each}
            </select>
          </div>
        {/if}
        <IntegrationCard
          title="Config File"
          subtitle="~/.codex/config.toml"
          description="Add this to your Codex CLI config so the model and provider persist."
          config={codexConfig}
        />
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Launch Codex with exo as the backend."
          config={codexShellCommand}
          language="bash"
        />
      {:else if activeTab === "OpenClaw"}
        <div class="flex gap-3 text-xs">
          {#if runningModels.length > 1}
            <div>
              <span
                class="text-exo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
                >Model</span
              >
              <select bind:value={openClawModel} class={selectClass}>
                {#each runningModels as model}
                  <option value={model}>{model.split("/").pop()}</option>
                {/each}
              </select>
            </div>
          {/if}
          <div>
            <span
              class="text-exo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >Tools Profile</span
            >
            <select bind:value={openClawToolsProfile} class={selectClass}>
              {#each ["minimal", "coding", "messaging", "full"] as profile}
                <option value={profile}>{profile}</option>
              {/each}
            </select>
          </div>
        </div>
        <IntegrationCard
          title="Config File"
          subtitle="~/.openclaw/openclaw.json"
          description="Add this to your OpenClaw config."
          config={openClawConfig}
        />
      {:else if activeTab === "Ollama"}
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Set OLLAMA_HOST to point the Ollama CLI at your exo cluster."
          config={ollamaCommand}
          language="bash"
        />
      {:else if activeTab === "n8n"}
        <IntegrationCard
          title="Credential Setup"
          subtitle="n8n UI"
          description="Configure an OpenAI credential in n8n to use your exo cluster."
          config={n8nConfig}
        />
      {/if}
    </div>
  </main>
</div>

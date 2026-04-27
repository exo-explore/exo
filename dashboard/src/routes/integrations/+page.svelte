<script lang="ts">
  import { browser } from "$app/environment";
  import { fade } from "svelte/transition";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import IntegrationCard from "$lib/components/IntegrationCard.svelte";
  import { instances, refreshState } from "$lib/stores/app.svelte";
  import { onMount } from "svelte";

  const apiUrl = browser
    ? window.location.origin.replace("localhost", "127.0.0.1")
    : "http://127.0.0.1:52415";

  const instancesData = $derived(instances());

  let modelCapabilities = $state<Record<string, string[]>>({});
  let modelContextLengths = $state<Record<string, number>>({});
  let modelReasoningDialects = $state<Record<string, string>>({});

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
  let codexMcpPath = $state("/Users/username");
  let openClawModel = $state("");
  let piModel = $state("");
  $effect(() => {
    const def = modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id";
    codexModel = def;
    openClawModel = def;
    piModel = def;
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
      const dialect = modelReasoningDialects[modelId];
      const entry: Record<string, unknown> = { name: modelId };
      if (ctxLen > 0) {
        entry.limit = { context: ctxLen, output: Math.min(ctxLen, 16384) };
      }
      if (caps.includes("vision")) {
        entry.modalities = { input: ["text", "image"], output: ["text"] };
      }
      // Reasoning round-trip: opencode's `interleaved` field tells the
      // openai-compatible adapter to send the assistant's prior
      // reasoning_content back in subsequent turns. Emit it for dialects
      // whose chat templates use prior reasoning:
      //   - `tool_conditional` (DeepSeek V3.2 / V4): wrapper preserves all
      //     reasoning when tools are present.
      //   - `post_last_user` (Qwen3-Thinking, GLM 4.5+, MiniMax M2.x):
      //     Jinja template reads reasoning_content for assistant turns since
      //     the last user message — exactly the tool-chain window.
      //   - `channel` (gpt-oss / Harmony): the model's Jinja template reads
      //     `message.thinking` rather than `message.reasoning_content`, but
      //     the server bridges `reasoning_content` → `thinking` before
      //     rendering, so the round-trip works through the standard field.
      // `suffix` (Kimi): reasoning lives in content; no separate field path.
      if (
        dialect === "tool_conditional" ||
        dialect === "post_last_user" ||
        dialect === "channel"
      ) {
        entry.interleaved = { field: "reasoning_content" };
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
      ``,
      `[mcp_servers.filesystem]`,
      `command = "npx"`,
      `args = ["-y", "@modelcontextprotocol/server-filesystem", "${codexMcpPath}"]`,
    ].join("\n"),
  );

  const openClawConfig = $derived(
    JSON.stringify(
      {
        gateway: { mode: "local" },
        models: {
          providers: {
            exo: {
              baseUrl: `${apiUrl}/v1`,
              apiKey: "x",
              api: "openai-completions",
              models: [
                {
                  id: openClawModel,
                  name: "exo local",
                  input: (modelCapabilities[openClawModel] || []).includes(
                    "vision",
                  )
                    ? ["text", "image"]
                    : ["text"],
                },
              ],
            },
          },
        },
        agents: {
          defaults: {
            model: `exo/${openClawModel}`,
          },
        },
      },
      null,
      2,
    ),
  );

  const piModelsJson = $derived.by(() => {
    const models: Record<string, unknown>[] = [];
    for (const modelId of runningModels) {
      const caps = modelCapabilities[modelId] || [];
      const ctxLen = modelContextLengths[modelId] || 0;
      const entry: Record<string, unknown> = { id: modelId };
      if (caps.includes("vision")) {
        entry.input = ["text", "image"];
      }
      // Mark thinking-capable models so pi surfaces its thinking-level selector
      // for them. exo capability strings: "thinking" (model emits reasoning
      // content) and "thinking_toggle" (user can turn it on/off).
      if (caps.includes("thinking") || caps.includes("thinking_toggle")) {
        entry.reasoning = true;
      }
      if (ctxLen > 0) {
        entry.contextWindow = ctxLen;
      }
      models.push(entry);
    }
    if (models.length === 0) {
      models.push({ id: "your-model-id" });
    }
    return JSON.stringify(
      {
        providers: {
          exo: {
            baseUrl: `${apiUrl}/v1`,
            api: "openai-completions",
            apiKey: "exo",
            compat: {
              supportsDeveloperRole: false,
              // exo's OpenAI surface takes a boolean `enable_thinking` toggle,
              // not graded effort levels, so disable pi's `reasoning_effort`
              // parameter and use the matching top-level-boolean format.
              supportsReasoningEffort: false,
              thinkingFormat: "qwen",
            },
            models,
          },
        },
      },
      null,
      2,
    );
  });

  const piShellCommand = $derived(`pi --provider exo --model ${piModel}`);

  const ollamaCommand = $derived(
    `OLLAMA_HOST=${apiUrl}/ollama ollama run ${modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id"}`,
  );

  const openWebUiCommand = $derived(
    [
      `docker run -d -p 3000:8080 \\`,
      `  -e OLLAMA_BASE_URL=${apiUrl.replace("localhost", "host.docker.internal")}/ollama \\`,
      `  -v open-webui:/app/backend/data \\`,
      `  --name open-webui \\`,
      `  ghcr.io/open-webui/open-webui:main`,
    ].join("\n"),
  );

  const n8nDockerCommand = $derived(
    [
      `docker run -d -p 5678:5678 \\`,
      `  -v n8n_data:/home/node/.n8n \\`,
      `  --name n8n \\`,
      `  docker.n8n.io/n8nio/n8n`,
    ].join("\n"),
  );

  const n8nCredentialSteps = $derived(
    [
      `1. Go to Credentials → Add Credential → search "OpenAI API"`,
      `2. Set API Key to: x`,
      `3. Set Base URL to: ${apiUrl.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")}/v1`,
      `4. Save the credential`,
    ].join("\n"),
  );

  const n8nWorkflowSteps = $derived(
    [
      `1. Create a new workflow → "Start from Scratch"`,
      `2. Add an "AI Agent" or "Basic LLM Chain" node`,
      `3. Inside it, add an "OpenAI Chat Model" sub-node`,
      `4. Select the OpenAI credential you just created`,
      `5. Set Model to "From list" and pick your model (e.g. ${modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id"})`,
      `6. Optionally toggle "Use Responses API", add Built-in Tools, or click "Add Option" for sampling settings`,
      `7. Connect a "Chat Trigger" node for interactive chat`,
      `8. On the Chat Trigger, enable "Allow File Uploads" for vision`,
    ].join("\n"),
  );

  const firefoxConfig = $derived(
    [
      `1. Open about:config in Firefox`,
      `2. Set browser.ml.chat.enabled to true`,
      `3. Set browser.ml.chat.hideLocalhost to false`,
      `4. Set browser.ml.chat.provider to: ${apiUrl}/`,
    ].join("\n"),
  );

  const tabs = [
    "Claude Code",
    "OpenCode",
    "Codex",
    "OpenClaw",
    "Pi",
    "Open WebUI",
    "n8n",
    "Firefox",
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
        data: {
          id: string;
          capabilities: string[];
          context_length: number;
          reasoning_dialect?: string;
        }[];
      };
      const caps: Record<string, string[]> = {};
      const ctxs: Record<string, number> = {};
      const dialects: Record<string, string> = {};
      for (const model of data.data) {
        caps[model.id] = model.capabilities || [];
        if (model.context_length > 0) ctxs[model.id] = model.context_length;
        if (model.reasoning_dialect)
          dialects[model.id] = model.reasoning_dialect;
      }
      modelCapabilities = caps;
      modelContextLengths = ctxs;
      modelReasoningDialects = dialects;
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
        <div class="flex gap-3 text-xs">
          {#if runningModels.length > 1}
            <div>
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
          <div class="flex-1">
            <span
              class="text-exo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >MCP Filesystem Path</span
            >
            <input
              type="text"
              bind:value={codexMcpPath}
              class="w-full bg-black/30 border border-exo-light-gray/20 rounded px-2 py-1.5 text-white font-mono text-xs focus:border-exo-yellow/50 focus:outline-none"
            />
          </div>
        </div>
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
        {#if runningModels.length > 1}
          <div class="text-xs">
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
        <IntegrationCard
          title="Config File"
          subtitle="~/.openclaw/openclaw.json"
          description="Add this to your OpenClaw config. If you haven't installed OpenClaw yet, run: npm install -g openclaw@latest"
          config={openClawConfig}
        />
        <IntegrationCard
          title="Setup Commands"
          subtitle="Run in terminal"
          description="After saving the config, run these commands to fix metadata and start the gateway."
          config={`openclaw doctor --fix${(modelCapabilities[openClawModel] || []).includes("vision") ? `\nopenclaw models set-image exo/${openClawModel}` : ""}\nopenclaw gateway &\nopenclaw dashboard`}
          language="bash"
        />
      {:else if activeTab === "Pi"}
        {#if runningModels.length > 1}
          <div class="text-xs">
            <span
              class="text-exo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >Model</span
            >
            <select bind:value={piModel} class={selectClass}>
              {#each runningModels as model}
                <option value={model}>{model.split("/").pop()}</option>
              {/each}
            </select>
          </div>
        {/if}
        <IntegrationCard
          title="Models Config"
          subtitle="~/.pi/agent/models.json"
          description="Register exo as a custom provider in pi. Create or edit this file, then run pi and pick an exo model via /model. Install pi with: npm install -g @mariozechner/pi-coding-agent"
          config={piModelsJson}
        />
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Launch pi directly with the exo provider and model selected."
          config={piShellCommand}
          language="bash"
        />
      {:else if activeTab === "Open WebUI"}
        <IntegrationCard
          title="1. Start Open WebUI"
          subtitle="Run in terminal"
          description="Run this to start Open WebUI."
          config={openWebUiCommand}
          language="bash"
        />
        <IntegrationCard
          title="2. Open & Select Model"
          subtitle="http://localhost:3000"
          description={`Open http://localhost:3000 in your browser. Select the running model from the dropdown at the top: ${runningModels.length > 0 ? runningModels.join(", ") : "no models running"}`}
          config={"open http://localhost:3000"}
          language="bash"
        />
        <IntegrationCard
          title="Ollama CLI"
          subtitle="Run in terminal"
          description="Or use the Ollama CLI directly."
          config={ollamaCommand}
          language="bash"
        />
      {:else if activeTab === "n8n"}
        <IntegrationCard
          title="1. Start n8n"
          subtitle="Run in terminal"
          description="Start n8n with Docker. If you already have n8n running, skip this step."
          config={n8nDockerCommand}
          language="bash"
        />
        <IntegrationCard
          title="2. Open n8n"
          subtitle="http://localhost:5678"
          description="Open n8n in your browser. If this is your first time, complete the setup and select 'Start from Scratch' when prompted."
          config={"open http://localhost:5678"}
          language="bash"
        />
        <IntegrationCard
          title="3. Add OpenAI Credential"
          subtitle="n8n UI → Credentials"
          description="Create an OpenAI credential pointing at your exo cluster."
          config={n8nCredentialSteps}
        />
        <IntegrationCard
          title="4. Build a Workflow"
          subtitle="n8n UI → Workflows"
          description="Create a workflow that uses your exo-powered model."
          config={n8nWorkflowSteps}
        />
      {:else if activeTab === "Firefox"}
        <IntegrationCard
          title="Firefox AI Chatbot"
          subtitle="about:config"
          description="Use the exo dashboard as Firefox's built-in AI chatbot. Requires Firefox 130+."
          config={firefoxConfig}
        />
      {/if}
    </div>
  </main>
</div>

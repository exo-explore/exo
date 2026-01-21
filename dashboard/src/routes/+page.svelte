<script lang="ts">
  import {
    TopologyGraph,
    ChatForm,
    ChatMessages,
    ChatSidebar,
    ModelCard,
  } from "$lib/components";
  import {
    hasStartedChat,
    isTopologyMinimized,
    topologyData,
    lastUpdate,
    clearChat,
    instances,
    runners,
    downloads,
    placementPreviews,
    selectedPreviewModelId,
    isLoadingPreviews,
    selectPreviewModel,
    createConversation,
    setSelectedChatModel,
    selectedChatModel,
    debugMode,
    toggleDebugMode,
    topologyOnlyMode,
    toggleTopologyOnlyMode,
    chatSidebarVisible,
    toggleChatSidebarVisible,
    type DownloadProgress,
    type PlacementPreview,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import { fade, fly } from "svelte/transition";
  import { cubicInOut } from "svelte/easing";
  import { onMount } from "svelte";

  const chatStarted = $derived(hasStartedChat());
  const minimized = $derived(isTopologyMinimized());
  const data = $derived(topologyData());
  const update = $derived(lastUpdate());
  const instanceData = $derived(instances());
  const runnersData = $derived(runners());
  const downloadsData = $derived(downloads());
  const previewsData = $derived(placementPreviews());
  const selectedModelId = $derived(selectedPreviewModelId());
  const loadingPreviews = $derived(isLoadingPreviews());
  const debugEnabled = $derived(debugMode());
  const topologyOnlyEnabled = $derived(topologyOnlyMode());
  const sidebarVisible = $derived(chatSidebarVisible());

  let mounted = $state(false);

  // Instance launch state
  let models = $state<
    Array<{
      id: string;
      name?: string;
      storage_size_megabytes?: number;
      tasks?: string[];
      hugging_face_id?: string;
    }>
  >([]);

  // Model tasks lookup for ChatForm - maps both short IDs and full HuggingFace IDs
  const modelTasks = $derived(() => {
    const tasks: Record<string, string[]> = {};
    for (const model of models) {
      if (model.tasks && model.tasks.length > 0) {
        // Map by short ID
        tasks[model.id] = model.tasks;
        // Also map by hugging_face_id from the API response
        if (model.hugging_face_id) {
          tasks[model.hugging_face_id] = model.tasks;
        }
      }
    }
    return tasks;
  });

  // Helper to check if a model supports image generation
  function modelSupportsImageGeneration(modelId: string): boolean {
    const model = models.find(
      (m) => m.id === modelId || m.hugging_face_id === modelId,
    );
    if (!model?.tasks) return false;
    return (
      model.tasks.includes("TextToImage") ||
      model.tasks.includes("ImageToImage")
    );
  }
  let selectedSharding = $state<"Pipeline" | "Tensor">("Pipeline");
  type InstanceMeta = "MlxRing" | "MlxIbv" | "MlxJaccl";

  // Launch defaults persistence
  const LAUNCH_DEFAULTS_KEY = "exo-launch-defaults";
  interface LaunchDefaults {
    modelId: string | null;
    sharding: "Pipeline" | "Tensor";
    instanceType: InstanceMeta;
    minNodes: number;
  }

  function saveLaunchDefaults(): void {
    const defaults: LaunchDefaults = {
      modelId: selectedPreviewModelId(),
      sharding: selectedSharding,
      instanceType: selectedInstanceType,
      minNodes: selectedMinNodes,
    };
    try {
      localStorage.setItem(LAUNCH_DEFAULTS_KEY, JSON.stringify(defaults));
    } catch (e) {
      console.warn("Failed to save launch defaults:", e);
    }
  }

  function loadLaunchDefaults(): LaunchDefaults | null {
    try {
      const stored = localStorage.getItem(LAUNCH_DEFAULTS_KEY);
      if (!stored) return null;
      return JSON.parse(stored) as LaunchDefaults;
    } catch (e) {
      console.warn("Failed to load launch defaults:", e);
      return null;
    }
  }

  function applyLaunchDefaults(
    availableModels: Array<{ id: string }>,
    maxNodes: number,
  ): void {
    const defaults = loadLaunchDefaults();
    if (!defaults) return;

    // Apply sharding and instance type unconditionally
    selectedSharding = defaults.sharding;
    selectedInstanceType = defaults.instanceType;

    // Apply minNodes if valid (between 1 and maxNodes)
    if (
      defaults.minNodes &&
      defaults.minNodes >= 1 &&
      defaults.minNodes <= maxNodes
    ) {
      selectedMinNodes = defaults.minNodes;
    }

    // Only apply model if it exists in the available models
    if (
      defaults.modelId &&
      availableModels.some((m) => m.id === defaults.modelId)
    ) {
      selectPreviewModel(defaults.modelId);
    }
  }

  let selectedInstanceType = $state<InstanceMeta>("MlxRing");
  let selectedMinNodes = $state<number>(1);
  let minNodesInitialized = $state(false);
  let launchingModelId = $state<string | null>(null);
  let instanceDownloadExpandedNodes = $state<Set<string>>(new Set());

  // Custom dropdown state
  let isModelDropdownOpen = $state(false);
  let modelDropdownSearch = $state("");

  // Slider dragging state
  let isDraggingSlider = $state(false);
  let sliderTrackElement: HTMLDivElement | null = $state(null);

  // Instances container ref for scrolling
  let instancesContainerRef: HTMLDivElement | null = $state(null);
  // Chat scroll container ref for precise scroll behavior
  let chatScrollRef: HTMLDivElement | null = $state(null);

  // Instance hover state for highlighting nodes in topology
  let hoveredInstanceId = $state<string | null>(null);

  // Preview card hover state for highlighting nodes in topology
  let hoveredPreviewNodes = $state<Set<string>>(new Set());

  // Helper to unwrap tagged instance for hover highlighting
  function unwrapInstanceNodes(instanceWrapped: unknown): Set<string> {
    if (!instanceWrapped || typeof instanceWrapped !== "object")
      return new Set();
    const keys = Object.keys(instanceWrapped as Record<string, unknown>);
    if (keys.length !== 1) return new Set();
    const instance = (instanceWrapped as Record<string, unknown>)[keys[0]];
    if (!instance || typeof instance !== "object") return new Set();
    const inst = instance as {
      shardAssignments?: { nodeToRunner?: Record<string, string> };
    };
    if (!inst.shardAssignments?.nodeToRunner) return new Set();
    return new Set(Object.keys(inst.shardAssignments.nodeToRunner));
  }

  function toggleInstanceDownloadDetails(nodeId: string): void {
    const next = new Set(instanceDownloadExpandedNodes);
    if (next.has(nodeId)) {
      next.delete(nodeId);
    } else {
      next.add(nodeId);
    }
    instanceDownloadExpandedNodes = next;
  }

  // Compute highlighted nodes from hovered instance or hovered preview
  const highlightedNodes = $derived(() => {
    // First check instance hover
    if (hoveredInstanceId) {
      const instanceWrapped = instanceData[hoveredInstanceId];
      return unwrapInstanceNodes(instanceWrapped);
    }
    // Then check preview hover
    if (hoveredPreviewNodes.size > 0) {
      return hoveredPreviewNodes;
    }
    return new Set<string>();
  });

  // Helper to estimate memory from model ID (mirrors ModelCard logic)
  // Uses regex with word boundaries to avoid false matches like '4bit' matching '4b'
  function estimateMemoryGB(modelId: string, modelName?: string): number {
    // Check both ID and name for quantization info
    const combined = `${modelId} ${modelName || ""}`.toLowerCase();

    // Detect quantization level - affects memory by roughly 2x between levels
    const is4bit =
      combined.includes("4bit") ||
      combined.includes("4-bit") ||
      combined.includes(":4bit");
    const is8bit =
      combined.includes("8bit") ||
      combined.includes("8-bit") ||
      combined.includes(":8bit");
    // 4-bit = 0.5 bytes/param, 8-bit = 1 byte/param, fp16 = 2 bytes/param
    const quantMultiplier = is4bit ? 0.5 : is8bit ? 1 : 2;
    const id = modelId.toLowerCase();

    // Known large models that don't follow the standard naming pattern
    // DeepSeek V3 has 685B parameters
    if (id.includes("deepseek-v3")) {
      return Math.round(685 * quantMultiplier);
    }
    // DeepSeek V2 has 236B parameters
    if (id.includes("deepseek-v2")) {
      return Math.round(236 * quantMultiplier);
    }
    // Llama 4 Scout/Maverick are large models
    if (id.includes("llama-4")) {
      return Math.round(400 * quantMultiplier);
    }

    // Match parameter counts with word boundaries (e.g., "70b" but not "4bit")
    const paramMatch = id.match(/(\d+(?:\.\d+)?)\s*b(?![a-z])/i);
    if (paramMatch) {
      const params = parseFloat(paramMatch[1]);
      return Math.max(4, Math.round(params * quantMultiplier));
    }

    // Fallback patterns for explicit size markers (assume fp16 baseline, adjust for quant)
    if (id.includes("405b") || id.includes("400b"))
      return Math.round(405 * quantMultiplier);
    if (id.includes("180b")) return Math.round(180 * quantMultiplier);
    if (id.includes("141b") || id.includes("140b"))
      return Math.round(140 * quantMultiplier);
    if (id.includes("123b") || id.includes("120b"))
      return Math.round(123 * quantMultiplier);
    if (id.includes("72b") || id.includes("70b"))
      return Math.round(70 * quantMultiplier);
    if (id.includes("67b") || id.includes("65b"))
      return Math.round(65 * quantMultiplier);
    if (
      id.includes("35b") ||
      id.includes("34b") ||
      id.includes("32b") ||
      id.includes("30b")
    )
      return Math.round(32 * quantMultiplier);
    if (id.includes("27b") || id.includes("26b") || id.includes("22b"))
      return Math.round(24 * quantMultiplier);
    if (id.includes("14b") || id.includes("13b") || id.includes("15b"))
      return Math.round(14 * quantMultiplier);
    if (id.includes("8b") || id.includes("9b") || id.includes("7b"))
      return Math.round(8 * quantMultiplier);
    if (id.includes("3b") || id.includes("3.8b"))
      return Math.round(4 * quantMultiplier);
    if (
      id.includes("2b") ||
      id.includes("1b") ||
      id.includes("1.5b") ||
      id.includes("0.5b")
    )
      return Math.round(2 * quantMultiplier);
    return 16; // Default fallback
  }

  // Helper to estimate performance from model ID
  function estimatePerformance(modelId: string): { ttft: number; tps: number } {
    const id = modelId.toLowerCase();
    if (id.includes("405b") || id.includes("400b"))
      return { ttft: 8000, tps: 3 };
    if (id.includes("180b")) return { ttft: 4000, tps: 5 };
    if (id.includes("141b") || id.includes("140b"))
      return { ttft: 3500, tps: 6 };
    if (id.includes("123b") || id.includes("120b"))
      return { ttft: 3000, tps: 7 };
    if (id.includes("72b") || id.includes("70b"))
      return { ttft: 1800, tps: 12 };
    if (id.includes("67b") || id.includes("65b"))
      return { ttft: 1600, tps: 14 };
    if (
      id.includes("35b") ||
      id.includes("34b") ||
      id.includes("32b") ||
      id.includes("30b")
    )
      return { ttft: 900, tps: 22 };
    if (id.includes("27b") || id.includes("26b") || id.includes("22b"))
      return { ttft: 700, tps: 28 };
    if (id.includes("14b") || id.includes("13b") || id.includes("15b"))
      return { ttft: 400, tps: 45 };
    if (id.includes("8b") || id.includes("9b") || id.includes("7b"))
      return { ttft: 200, tps: 65 };
    if (id.includes("4b") || id.includes("3b") || id.includes("3.8b"))
      return { ttft: 100, tps: 95 };
    if (
      id.includes("2b") ||
      id.includes("1b") ||
      id.includes("1.5b") ||
      id.includes("0.5b")
    )
      return { ttft: 50, tps: 150 };
    return { ttft: 300, tps: 50 };
  }

  const matchesSelectedRuntime = (runtime: InstanceMeta): boolean =>
    selectedInstanceType === "MlxRing"
      ? runtime === "MlxRing"
      : runtime === "MlxIbv" || runtime === "MlxJaccl";

  // Helper to check if a model can be launched (has valid placement with >= minNodes)
  function canModelFit(modelId: string): boolean {
    // Find previews matching the model, sharding, and instance type
    const matchingPreviews = previewsData.filter(
      (p: PlacementPreview) =>
        p.model_id === modelId &&
        p.sharding === selectedSharding &&
        matchesSelectedRuntime(p.instance_meta) &&
        p.error === null &&
        p.memory_delta_by_node !== null,
    );

    // Check if any preview has node count >= selectedMinNodes
    return matchingPreviews.some(
      (p: PlacementPreview) => getPreviewNodeCount(p) >= selectedMinNodes,
    );
  }

  // Helper to get model size in GB (from megabytes)
  function getModelSizeGB(model: {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
  }): number {
    if (model.storage_size_megabytes) {
      return model.storage_size_megabytes / 1024;
    }
    return estimateMemoryGB(model.id, model.name);
  }

  // Calculate available memory in the cluster (in GB)
  const availableMemoryGB = $derived(() => {
    if (!data) return 0;
    return (
      Object.values(data.nodes).reduce((acc, n) => {
        const total =
          n.macmon_info?.memory?.ram_total ?? n.system_info?.memory ?? 0;
        const used = n.macmon_info?.memory?.ram_usage ?? 0;
        return acc + (total - used);
      }, 0) /
      (1024 * 1024 * 1024)
    );
  });

  // Check if a model has enough memory to run
  function hasEnoughMemory(model: {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
  }): boolean {
    const modelSizeGB = getModelSizeGB(model);
    return modelSizeGB <= availableMemoryGB();
  }

  // Sorted models for dropdown - biggest first, unrunnable at the end
  const sortedModels = $derived(() => {
    return [...models].sort((a, b) => {
      // First: models that have enough memory come before those that don't
      const aCanFit = hasEnoughMemory(a);
      const bCanFit = hasEnoughMemory(b);
      if (aCanFit && !bCanFit) return -1;
      if (!aCanFit && bCanFit) return 1;

      // Then: sort by size (biggest first)
      const aSize = getModelSizeGB(a);
      const bSize = getModelSizeGB(b);
      return bSize - aSize;
    });
  });

  // Compute model tags (FASTEST, BIGGEST)
  const modelTags = $derived(() => {
    const tags: Record<string, string[]> = {};
    if (models.length === 0) return tags;

    // Find the fastest model (highest TPS)
    let fastestId = "";
    let highestTps = 0;

    // Find the biggest model (most memory)
    let biggestId = "";
    let highestMemory = 0;

    for (const model of models) {
      const perf = estimatePerformance(model.id);
      const mem = getModelSizeGB(model);

      if (perf.tps > highestTps) {
        highestTps = perf.tps;
        fastestId = model.id;
      }

      if (mem > highestMemory) {
        highestMemory = mem;
        biggestId = model.id;
      }
    }

    if (fastestId) {
      tags[fastestId] = tags[fastestId] || [];
      tags[fastestId].push("FASTEST");
    }

    if (biggestId && biggestId !== fastestId) {
      tags[biggestId] = tags[biggestId] || [];
      tags[biggestId].push("BIGGEST");
    } else if (biggestId === fastestId && biggestId) {
      // Same model is both - unlikely but handle it
      tags[biggestId].push("BIGGEST");
    }

    return tags;
  });

  onMount(() => {
    mounted = true;
    fetchModels();
  });

  async function fetchModels() {
    try {
      const response = await fetch("/models");
      if (response.ok) {
        const data = await response.json();
        // API returns { data: [{ id, name }] } format
        models = data.data || [];
        // Restore last launch defaults if available
        const currentNodeCount = topologyData()
          ? Object.keys(topologyData()!.nodes).length
          : 1;
        applyLaunchDefaults(models, currentNodeCount);
      }
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  }

  async function launchInstance(
    modelId: string,
    specificPreview?: PlacementPreview | null,
  ) {
    if (!modelId || launchingModelId) return;

    launchingModelId = modelId;

    try {
      // Use the specific preview if provided, otherwise fall back to filtered preview
      const preview = specificPreview ?? filteredPreview();

      let instanceData: unknown;

      if (preview?.instance) {
        // Use the instance from the preview
        instanceData = preview.instance;
      } else {
        // Fallback: GET placement from API
        const placementResponse = await fetch(
          `/instance/placement?model_id=${encodeURIComponent(modelId)}&sharding=${selectedSharding}&instance_meta=${selectedInstanceType}&min_nodes=${selectedMinNodes}`,
        );

        if (!placementResponse.ok) {
          const errorText = await placementResponse.text();
          console.error("Failed to get placement:", errorText);
          return;
        }

        instanceData = await placementResponse.json();
      }

      // POST the instance to create it
      const response = await fetch("/instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance: instanceData }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Failed to launch instance:", errorText);
      } else {
        // Always auto-select the newly launched model so the user chats to what they just launched
        setSelectedChatModel(modelId);

        // Scroll to the bottom of instances container to show the new instance
        // Use multiple attempts to ensure DOM has updated with the new instance
        const scrollToBottom = () => {
          if (instancesContainerRef) {
            instancesContainerRef.scrollTo({
              top: instancesContainerRef.scrollHeight,
              behavior: "smooth",
            });
          }
        };
        setTimeout(scrollToBottom, 200);
        setTimeout(scrollToBottom, 500);
        setTimeout(scrollToBottom, 1000);
      }
    } catch (error) {
      console.error("Error launching instance:", error);
    } finally {
      launchingModelId = null;
    }
  }

  // Helper to extract model ID from download shard metadata
  function extractModelIdFromDownload(
    downloadPayload: Record<string, unknown>,
  ): string | null {
    const shardMetadata =
      downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
    if (!shardMetadata || typeof shardMetadata !== "object") return null;

    // Shard metadata is a tagged union: { PipelineShardMetadata: {...} } or { TensorShardMetadata: {...} }
    const shardObj = shardMetadata as Record<string, unknown>;
    const shardKeys = Object.keys(shardObj);
    if (shardKeys.length !== 1) return null;

    const shardData = shardObj[shardKeys[0]] as Record<string, unknown>;
    if (!shardData) return null;

    // Model meta is nested: shard.model_card.model_id
    const modelMeta = shardData.model_card ?? shardData.modelCard;
    if (!modelMeta || typeof modelMeta !== "object") return null;

    const meta = modelMeta as Record<string, unknown>;
    return (meta.model_id as string) ?? (meta.modelId as string) ?? null;
  }

  // Helper to parse download progress from payload
  function parseDownloadProgress(
    payload: Record<string, unknown>,
  ): DownloadProgress | null {
    const progress = payload.download_progress ?? payload.downloadProgress;
    if (!progress || typeof progress !== "object") return null;

    const prog = progress as Record<string, unknown>;
    const totalBytes = getBytes(prog.total_bytes ?? prog.totalBytes);
    const downloadedBytes = getBytes(
      prog.downloaded_bytes ?? prog.downloadedBytes,
    );
    const speed = (prog.speed as number) ?? 0;
    const completedFiles =
      (prog.completed_files as number) ?? (prog.completedFiles as number) ?? 0;
    const totalFiles =
      (prog.total_files as number) ?? (prog.totalFiles as number) ?? 0;
    const etaMs = (prog.eta_ms as number) ?? (prog.etaMs as number) ?? 0;

    const files: DownloadProgress["files"] = [];
    const filesObj = (prog.files ?? {}) as Record<string, unknown>;
    for (const [fileName, fileData] of Object.entries(filesObj)) {
      if (!fileData || typeof fileData !== "object") continue;
      const fd = fileData as Record<string, unknown>;
      const fTotal = getBytes(fd.total_bytes ?? fd.totalBytes);
      const fDownloaded = getBytes(fd.downloaded_bytes ?? fd.downloadedBytes);
      files.push({
        name: fileName,
        totalBytes: fTotal,
        downloadedBytes: fDownloaded,
        speed: (fd.speed as number) ?? 0,
        etaMs: (fd.eta_ms as number) ?? (fd.etaMs as number) ?? 0,
        percentage: fTotal > 0 ? (fDownloaded / fTotal) * 100 : 0,
      });
    }

    return {
      totalBytes,
      downloadedBytes,
      speed,
      etaMs:
        etaMs ||
        (speed > 0 ? ((totalBytes - downloadedBytes) / speed) * 1000 : 0),
      percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
      completedFiles,
      totalFiles,
      files,
    };
  }

  // Helper to get download status for a model (checks all downloads for matching model ID)
  function getModelDownloadStatus(modelId: string): {
    isDownloading: boolean;
    progress: DownloadProgress | null;
    perNode: Array<{
      nodeId: string;
      nodeName: string;
      progress: DownloadProgress;
    }>;
  } {
    if (!downloadsData || Object.keys(downloadsData).length === 0) {
      return { isDownloading: false, progress: null, perNode: [] };
    }

    let totalBytes = 0;
    let downloadedBytes = 0;
    let totalSpeed = 0;
    let completedFiles = 0;
    let totalFiles = 0;
    let isDownloading = false;
    const allFiles: DownloadProgress["files"] = [];
    const perNode: Array<{
      nodeId: string;
      nodeName: string;
      progress: DownloadProgress;
    }> = [];

    // Check all nodes for downloads matching this model
    for (const [nodeId, nodeDownloads] of Object.entries(downloadsData)) {
      if (!Array.isArray(nodeDownloads)) continue;

      for (const downloadWrapped of nodeDownloads) {
        if (!downloadWrapped || typeof downloadWrapped !== "object") continue;

        const keys = Object.keys(downloadWrapped as Record<string, unknown>);
        if (keys.length !== 1) continue;

        const downloadKind = keys[0];
        const downloadPayload = (downloadWrapped as Record<string, unknown>)[
          downloadKind
        ] as Record<string, unknown>;

        if (downloadKind !== "DownloadOngoing") continue;
        if (!downloadPayload) continue;

        const downloadModelId = extractModelIdFromDownload(downloadPayload);

        // Match if the model ID contains or equals the requested model
        // (handles cases like "mlx-community/Meta-Llama..." matching)
        if (
          !downloadModelId ||
          !downloadModelId.includes(modelId.split("/").pop() || modelId)
        ) {
          // Try exact match or partial match
          if (downloadModelId !== modelId) continue;
        }

        isDownloading = true;

        const progress = parseDownloadProgress(downloadPayload);
        if (progress) {
          // Sum all values across nodes - each node downloads independently
          totalBytes += progress.totalBytes;
          downloadedBytes += progress.downloadedBytes;
          totalSpeed += progress.speed;
          completedFiles += progress.completedFiles;
          totalFiles += progress.totalFiles;
          allFiles.push(...progress.files);

          const nodeName =
            data?.nodes?.[nodeId]?.friendly_name ?? nodeId.slice(0, 8);
          perNode.push({ nodeId, nodeName, progress });
        }
      }
    }

    if (!isDownloading) {
      return { isDownloading: false, progress: null, perNode: [] };
    }

    // ETA = total remaining bytes / total speed across all nodes
    const remainingBytes = totalBytes - downloadedBytes;
    const etaMs = totalSpeed > 0 ? (remainingBytes / totalSpeed) * 1000 : 0;

    return {
      isDownloading: true,
      progress: {
        totalBytes,
        downloadedBytes,
        speed: totalSpeed,
        etaMs,
        percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
        completedFiles,
        totalFiles,
        files: allFiles,
      },
      perNode,
    };
  }

  // Debug: Log downloads data when it changes
  $effect(() => {
    if (downloadsData && Object.keys(downloadsData).length > 0) {
      console.log("[Download Debug] Current downloads:", downloadsData);
    }
  });

  // Helper to get download status for an instance
  function getInstanceDownloadStatus(
    instanceId: string,
    instanceWrapped: unknown,
  ): {
    isDownloading: boolean;
    progress: DownloadProgress | null;
    statusText: string;
    perNode: Array<{
      nodeId: string;
      nodeName: string;
      progress: DownloadProgress;
    }>;
  } {
    if (!downloadsData || Object.keys(downloadsData).length === 0) {
      return {
        isDownloading: false,
        progress: null,
        statusText: "RUNNING",
        perNode: [],
      };
    }

    // Unwrap the instance
    const [instanceTag, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return {
        isDownloading: false,
        progress: null,
        statusText: "PREPARING",
        perNode: [],
      };
    }

    const inst = instance as {
      shardAssignments?: {
        nodeToRunner?: Record<string, string>;
        runnerToShard?: Record<string, unknown>;
        modelId?: string;
      };
    };
    const nodeToRunner = inst.shardAssignments?.nodeToRunner || {};
    const runnerToShard = inst.shardAssignments?.runnerToShard || {};
    const instanceModelId = inst.shardAssignments?.modelId;

    // Build reverse mapping: runnerId -> nodeId
    const runnerToNode: Record<string, string> = {};
    for (const [nodeId, runnerId] of Object.entries(nodeToRunner)) {
      runnerToNode[runnerId] = nodeId;
    }

    let totalBytes = 0;
    let downloadedBytes = 0;
    let totalSpeed = 0;
    let completedFiles = 0;
    let totalFiles = 0;
    let isDownloading = false;
    const allFiles: DownloadProgress["files"] = [];
    const perNode: Array<{
      nodeId: string;
      nodeName: string;
      progress: DownloadProgress;
    }> = [];

    // Check downloads for nodes that are part of this instance
    for (const runnerId of Object.keys(runnerToShard)) {
      const nodeId = runnerToNode[runnerId];
      if (!nodeId) continue;

      const nodeDownloads = downloadsData[nodeId];
      if (!Array.isArray(nodeDownloads)) continue;

      for (const downloadWrapped of nodeDownloads) {
        if (!downloadWrapped || typeof downloadWrapped !== "object") continue;

        const keys = Object.keys(downloadWrapped as Record<string, unknown>);
        if (keys.length !== 1) continue;

        const downloadKind = keys[0];
        const downloadPayload = (downloadWrapped as Record<string, unknown>)[
          downloadKind
        ] as Record<string, unknown>;

        if (downloadKind !== "DownloadOngoing") continue;
        if (!downloadPayload) continue;

        // Check if this download is for this instance's model
        const downloadModelId = extractModelIdFromDownload(downloadPayload);
        if (
          instanceModelId &&
          downloadModelId &&
          downloadModelId === instanceModelId
        ) {
          isDownloading = true;

          const progress = parseDownloadProgress(downloadPayload);
          if (progress) {
            // Sum all values across nodes - each node downloads independently
            totalBytes += progress.totalBytes;
            downloadedBytes += progress.downloadedBytes;
            totalSpeed += progress.speed;
            completedFiles += progress.completedFiles;
            totalFiles += progress.totalFiles;
            allFiles.push(...progress.files);

            const nodeName =
              data?.nodes?.[nodeId]?.friendly_name ?? nodeId.slice(0, 8);
            perNode.push({ nodeId, nodeName, progress });
          }
        }
      }
    }

    if (!isDownloading) {
      // Check runner status for other states
      const statusInfo = deriveInstanceStatus(instanceWrapped);
      return {
        isDownloading: false,
        progress: null,
        statusText: statusInfo.statusText,
        perNode: [],
      };
    }

    // ETA = total remaining bytes / total speed across all nodes
    const remainingBytes = totalBytes - downloadedBytes;
    const etaMs = totalSpeed > 0 ? (remainingBytes / totalSpeed) * 1000 : 0;

    return {
      isDownloading: true,
      progress: {
        totalBytes,
        downloadedBytes,
        speed: totalSpeed,
        etaMs,
        percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
        completedFiles,
        totalFiles,
        files: allFiles,
      },
      statusText: "DOWNLOADING",
      perNode,
    };
  }

  // Derive instance status from runners
  // Get color class for a status
  function getStatusColor(statusText: string): string {
    switch (statusText) {
      case "FAILED":
        return "text-red-400";
      case "SHUTDOWN":
        return "text-gray-400";
      case "DOWNLOADING":
        return "text-blue-400";
      case "LOADING":
      case "WARMING UP":
      case "WAITING":
      case "INITIALIZING":
        return "text-yellow-400";
      case "RUNNING":
        return "text-teal-400";
      case "READY":
      case "LOADED":
        return "text-green-400";
      default:
        return "text-exo-light-gray";
    }
  }

  function deriveInstanceStatus(instanceWrapped: unknown): {
    statusText: string;
    statusClass: string;
  } {
    const [, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return { statusText: "PREPARING", statusClass: "inactive" };
    }

    const inst = instance as {
      shardAssignments?: { runnerToShard?: Record<string, unknown> };
    };
    const runnerIds = Object.keys(inst.shardAssignments?.runnerToShard || {});

    const statuses = runnerIds
      .map((rid) => {
        const r = runnersData[rid];
        if (!r) return null;
        const [kind] = getTagged(r);
        const statusMap: Record<string, string> = {
          RunnerWaitingForInitialization: "WaitingForInitialization",
          RunnerInitializingBackend: "InitializingBackend",
          RunnerWaitingForModel: "WaitingForModel",
          RunnerLoading: "Loading",
          RunnerLoaded: "Loaded",
          RunnerWarmingUp: "WarmingUp",
          RunnerReady: "Ready",
          RunnerRunning: "Running",
          RunnerShutdown: "Shutdown",
          RunnerFailed: "Failed",
        };
        return kind ? statusMap[kind] || null : null;
      })
      .filter((s): s is string => s !== null);

    const has = (s: string) => statuses.includes(s);

    if (statuses.length === 0)
      return { statusText: "PREPARING", statusClass: "inactive" };
    if (has("Failed")) return { statusText: "FAILED", statusClass: "failed" };
    if (has("Shutdown"))
      return { statusText: "SHUTDOWN", statusClass: "inactive" };
    if (has("Loading"))
      return { statusText: "LOADING", statusClass: "starting" };
    if (has("WarmingUp"))
      return { statusText: "WARMING UP", statusClass: "starting" };
    if (has("Running"))
      return { statusText: "RUNNING", statusClass: "running" };
    if (has("Ready")) return { statusText: "READY", statusClass: "loaded" };
    if (has("Loaded")) return { statusText: "LOADED", statusClass: "loaded" };
    if (has("WaitingForModel"))
      return { statusText: "WAITING", statusClass: "starting" };
    if (has("InitializingBackend"))
      return { statusText: "INITIALIZING", statusClass: "starting" };
    if (has("WaitingForInitialization"))
      return { statusText: "INITIALIZING", statusClass: "starting" };

    return { statusText: "RUNNING", statusClass: "active" };
  }

  function getBytes(value: unknown): number {
    if (typeof value === "number") return value;
    if (value && typeof value === "object") {
      const v = value as Record<string, unknown>;
      if (typeof v.in_bytes === "number") return v.in_bytes;
      if (typeof v.inBytes === "number") return v.inBytes;
    }
    return 0;
  }

  async function deleteInstance(instanceId: string) {
    if (!confirm(`Delete instance ${instanceId.slice(0, 8)}...?`)) return;

    // Get the model ID of the instance being deleted before we delete it
    const deletedInstanceModelId = getInstanceModelId(instanceData[instanceId]);
    const wasSelected = selectedChatModel() === deletedInstanceModelId;

    try {
      const response = await fetch(`/instance/${instanceId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        console.error("Failed to delete instance:", response.status);
      } else if (wasSelected) {
        // If we deleted the currently selected model, switch to another available model
        // Find another instance that isn't the one we just deleted
        const remainingInstances = Object.entries(instanceData).filter(
          ([id]) => id !== instanceId,
        );
        if (remainingInstances.length > 0) {
          // Select the last instance (most recently added, since objects preserve insertion order)
          const [, lastInstance] =
            remainingInstances[remainingInstances.length - 1];
          const newModelId = getInstanceModelId(lastInstance);
          if (
            newModelId &&
            newModelId !== "Unknown" &&
            newModelId !== "Unknown Model"
          ) {
            setSelectedChatModel(newModelId);
          } else {
            // Clear selection if no valid model found
            setSelectedChatModel("");
          }
        } else {
          // No more instances, clear the selection
          setSelectedChatModel("");
        }
      }
    } catch (error) {
      console.error("Error deleting instance:", error);
    }
  }

  // Helper to unwrap tagged unions like { MlxRingInstance: {...} }
  function getTagged(obj: unknown): [string | null, unknown] {
    if (!obj || typeof obj !== "object") return [null, null];
    const keys = Object.keys(obj as Record<string, unknown>);
    if (keys.length === 1) {
      return [keys[0], (obj as Record<string, unknown>)[keys[0]]];
    }
    return [null, null];
  }

  // Get model ID from an instance
  function getInstanceModelId(instanceWrapped: unknown): string {
    const [, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") return "Unknown";
    const inst = instance as { shardAssignments?: { modelId?: string } };
    return inst.shardAssignments?.modelId || "Unknown Model";
  }

  // Get instance details: type (MLX Ring/IBV), sharding (Pipeline/Tensor), and node names
  function getInstanceInfo(instanceWrapped: unknown): {
    instanceType: string;
    sharding: string;
    nodeNames: string[];
    nodeIds: string[];
    nodeCount: number;
  } {
    const [instanceTag, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return {
        instanceType: "Unknown",
        sharding: "Unknown",
        nodeNames: [],
        nodeIds: [],
        nodeCount: 0,
      };
    }

    // Instance type from tag
    let instanceType = "Unknown";
    if (instanceTag === "MlxRingInstance") instanceType = "MLX Ring";
    else if (
      instanceTag === "MlxIbvInstance" ||
      instanceTag === "MlxJacclInstance"
    )
      instanceType = "MLX RDMA";

    const inst = instance as {
      shardAssignments?: {
        nodeToRunner?: Record<string, string>;
        runnerToShard?: Record<string, unknown>;
      };
    };

    // Sharding strategy from first shard
    let sharding = "Unknown";
    const runnerToShard = inst.shardAssignments?.runnerToShard || {};
    const firstShardWrapped = Object.values(runnerToShard)[0];
    if (firstShardWrapped) {
      const [shardTag] = getTagged(firstShardWrapped);
      if (shardTag === "PipelineShardMetadata") sharding = "Pipeline";
      else if (shardTag === "TensorShardMetadata") sharding = "Tensor";
      else if (shardTag === "PrefillDecodeShardMetadata")
        sharding = "Prefill/Decode";
    }

    // Node names from topology
    const nodeToRunner = inst.shardAssignments?.nodeToRunner || {};
    const nodeIds = Object.keys(nodeToRunner);
    const nodeNames = nodeIds.map((nodeId) => {
      const node = data?.nodes?.[nodeId];
      return node?.friendly_name || nodeId.slice(0, 8);
    });

    return {
      instanceType,
      sharding,
      nodeNames,
      nodeIds,
      nodeCount: nodeIds.length,
    };
  }

  function formatLastUpdate(): string {
    if (!update) return "ACQUIRING...";
    const seconds = Math.floor((Date.now() - update) / 1000);
    if (seconds < 5) return "LIVE";
    return `${seconds}s AGO`;
  }

  function formatBytes(bytes: number, decimals = 2): string {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + " " + sizes[i]
    );
  }

  function formatSpeed(bps: number): string {
    if (!bps || bps <= 0) return "0 B/s";
    return formatBytes(bps, 1) + "/s";
  }

  function getNodeLabel(nodeId: string): string {
    const node = data?.nodes?.[nodeId];
    return node?.friendly_name || nodeId.slice(0, 8);
  }

  function getInterfaceLabel(
    nodeId: string,
    ip?: string,
  ): { label: string; missing: boolean } {
    if (!ip) return { label: "?", missing: true };
    const node = data?.nodes?.[nodeId];
    if (!node) return { label: "?", missing: true };

    // Prefer explicit network_interfaces from NodePerformanceProfile
    const matchFromInterfaces = node.network_interfaces?.find((iface) =>
      (iface.addresses || []).some((addr) => addr === ip),
    );
    if (matchFromInterfaces?.name) {
      return {
        label: `${matchFromInterfaces.name} on ${getNodeLabel(nodeId)}`,
        missing: false,
      };
    }

    // Fallback to derived ip_to_interface map
    const mapped = node.ip_to_interface?.[ip];
    if (mapped && mapped.trim().length > 0) {
      return { label: `${mapped} on ${getNodeLabel(nodeId)}`, missing: false };
    }

    return { label: "?", missing: true };
  }

  function getOrderedRunnerNodes(
    instance: Record<string, unknown>,
    shardType: "Pipeline" | "Tensor",
  ) {
    const runnerToShard =
      (
        instance.shardAssignments as
          | { runnerToShard?: Record<string, unknown> }
          | undefined
      )?.runnerToShard || {};
    const nodeToRunner =
      (
        instance.shardAssignments as
          | { nodeToRunner?: Record<string, string> }
          | undefined
      )?.nodeToRunner || {};
    const runnerEntries = Object.entries(runnerToShard).map(
      ([runnerId, shardWrapped]) => {
        const [tag, shard] = getTagged(shardWrapped);
        const meta = shard as
          | {
              modelMeta?: {
                worldSize?: number;
                nLayers?: number;
                deviceRank?: number;
              };
            }
          | undefined;
        const deviceRank = meta?.modelMeta?.deviceRank ?? 0;
        return { runnerId, tag, deviceRank };
      },
    );

    const ordered = runnerEntries
      .filter((r) =>
        shardType === "Pipeline"
          ? r.tag === "PipelineShardMetadata"
          : r.tag === "TensorShardMetadata",
      )
      .sort((a, b) => a.deviceRank - b.deviceRank)
      .map((r, idx) => {
        const nodeId = Object.entries(nodeToRunner).find(
          ([, rid]) => rid === r.runnerId,
        )?.[0];
        return { nodeId, runnerId: r.runnerId, order: idx };
      })
      .filter((item) => item.nodeId);

    return ordered as Array<{
      nodeId: string;
      runnerId: string;
      order: number;
    }>;
  }

  function pickHost(
    hosts?: Array<{ ip: string; port: number }>,
  ): { ip: string; port: number } | null {
    if (!hosts || hosts.length === 0) return null;
    const scored = hosts
      .filter((h) => h.ip && h.ip !== "0.0.0.0" && h.port && h.port > 0)
      .map((h) => {
        const ip = h.ip;
        const score =
          ip.startsWith("10.") ||
          ip.startsWith("172.") ||
          ip.startsWith("192.168")
            ? 3
            : ip.startsWith("169.254")
              ? 2
              : 1;
        return { host: h, score };
      });
    if (scored.length === 0) return null;
    scored.sort((a, b) => b.score - a.score);
    return scored[0].host;
  }

  function getInstanceConnections(instanceWrapped: unknown): Array<{
    from: string;
    to: string;
    ip: string;
    ifaceLabel: string;
    missingIface: boolean;
  }> {
    const [instanceTag, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") return [];

    // Jaccl (RDMA) – show RDMA interfaces from ibvDevices
    if (instanceTag === "MlxJacclInstance") {
      const ordered = getOrderedRunnerNodes(
        instance as Record<string, unknown>,
        "Tensor",
      );
      const ibvDevices =
        (instance as { ibvDevices?: Array<Array<string | null>> }).ibvDevices ||
        [];
      const rows: Array<{
        from: string;
        to: string;
        ip: string;
        ifaceLabel: string;
        missingIface: boolean;
      }> = [];

      for (let i = 0; i < ordered.length; i++) {
        for (let j = i + 1; j < ordered.length; j++) {
          const iface = ibvDevices[i]?.[j] ?? ibvDevices[j]?.[i] ?? null;
          if (!iface) continue;
          const fromId = ordered[i].nodeId;
          const toId = ordered[j].nodeId;
          rows.push({
            from: getNodeLabel(fromId),
            to: getNodeLabel(toId),
            ip: iface,
            ifaceLabel: `RDMA ${iface}`,
            missingIface: false,
          });
        }
      }
      return rows;
    }

    // Ring – derive ring order from pipeline shard ranks and pick host IPs from hostsByNode
    if (instanceTag === "MlxRingInstance") {
      const ordered = getOrderedRunnerNodes(
        instance as Record<string, unknown>,
        "Pipeline",
      );
      const hostsByNode =
        (
          instance as {
            hostsByNode?: Record<string, Array<{ ip: string; port: number }>>;
          }
        ).hostsByNode || {};
      const rows: Array<{
        from: string;
        to: string;
        ip: string;
        ifaceLabel: string;
        missingIface: boolean;
      }> = [];
      if (ordered.length === 0) return rows;

      for (let idx = 0; idx < ordered.length; idx++) {
        const current = ordered[idx];
        const next = ordered[(idx + 1) % ordered.length];
        const host = pickHost(hostsByNode[next.nodeId]);
        const ip = host ? `${host.ip}:${host.port}` : "?";
        const ifaceInfo = host
          ? getInterfaceLabel(next.nodeId, host.ip)
          : { label: "?", missing: true };
        rows.push({
          from: getNodeLabel(current.nodeId),
          to: getNodeLabel(next.nodeId),
          ip,
          ifaceLabel: ifaceInfo.label,
          missingIface: ifaceInfo.missing,
        });
      }
      return rows;
    }

    return [];
  }

  function formatEta(ms: number): string {
    if (!ms || ms <= 0) return "--";
    const totalSeconds = Math.round(ms / 1000);
    const s = totalSeconds % 60;
    const m = Math.floor(totalSeconds / 60) % 60;
    const h = Math.floor(totalSeconds / 3600);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function handleNewChat() {
    createConversation();
  }

  function handleGoHome() {
    clearChat();
  }

  // Slider drag handlers
  function handleSliderDrag(clientX: number) {
    if (!sliderTrackElement || availableMinNodes <= 1) return;

    const rect = sliderTrackElement.getBoundingClientRect();
    const percentage = Math.max(
      0,
      Math.min(1, (clientX - rect.left) / rect.width),
    );
    const rawValue = Math.round(percentage * (availableMinNodes - 1)) + 1;
    const clampedValue = Math.max(1, Math.min(availableMinNodes, rawValue));

    // Find nearest valid value
    const validCounts = validMinNodeCounts();
    if (validCounts.has(clampedValue)) {
      selectedMinNodes = clampedValue;
    } else {
      // Find nearest valid value
      let nearest = clampedValue;
      let minDist = Infinity;
      for (const v of validCounts) {
        const dist = Math.abs(v - clampedValue);
        if (dist < minDist) {
          minDist = dist;
          nearest = v;
        }
      }
      if (validCounts.size > 0) {
        selectedMinNodes = nearest;
      }
    }
  }

  function handleSliderMouseDown(event: MouseEvent) {
    isDraggingSlider = true;
    handleSliderDrag(event.clientX);
  }

  function handleSliderMouseMove(event: MouseEvent) {
    if (isDraggingSlider) {
      handleSliderDrag(event.clientX);
    }
  }

  function handleSliderMouseUp() {
    isDraggingSlider = false;
    saveLaunchDefaults();
  }

  // Handle touch events for mobile
  function handleSliderTouchStart(event: TouchEvent) {
    isDraggingSlider = true;
    if (event.touches.length > 0) {
      handleSliderDrag(event.touches[0].clientX);
    }
  }

  function handleSliderTouchMove(event: TouchEvent) {
    if (isDraggingSlider && event.touches.length > 0) {
      event.preventDefault();
      handleSliderDrag(event.touches[0].clientX);
    }
  }

  function handleSliderTouchEnd() {
    isDraggingSlider = false;
    saveLaunchDefaults();
  }

  const nodeCount = $derived(data ? Object.keys(data.nodes).length : 0);
  const instanceCount = $derived(Object.keys(instanceData).length);

  // Helper to get the number of nodes in a placement preview
  function getPreviewNodeCount(preview: PlacementPreview): number {
    if (!preview.memory_delta_by_node) return 0;
    // Count nodes that have non-zero memory delta (i.e. nodes actually used)
    return Object.entries(preview.memory_delta_by_node).filter(
      ([_, delta]) => delta > 0,
    ).length;
  }

  // Available min nodes options based on topology (like old dashboard)
  const availableMinNodes = $derived(Math.max(1, nodeCount));

  // Compute which min node values have valid previews for the current model/sharding/instance type
  // A minNodes value N is valid if there exists a placement with nodeCount >= N
  // Note: previewsData already contains previews for the selected model (fetched via API)
  const validMinNodeCounts = $derived(() => {
    if (!selectedModelId || previewsData.length === 0) {
      // If no model selected or no previews, allow all node counts (UI shows all as clickable)
      return new Set(
        Array.from({ length: availableMinNodes }, (_, i) => i + 1),
      );
    }

    // Find the max node count among valid placements for this model/sharding/instance
    // (model_id filter not needed since previewsData is already for selected model)
    let maxValidNodes = 0;
    for (const preview of previewsData) {
      if (preview.sharding !== selectedSharding) continue;
      if (!matchesSelectedRuntime(preview.instance_meta)) continue;
      if (preview.error !== null) continue;
      if (!preview.memory_delta_by_node) continue;

      const previewNodes = getPreviewNodeCount(preview);
      if (previewNodes > maxValidNodes) {
        maxValidNodes = previewNodes;
      }
    }

    // All values from 1 to maxValidNodes are valid (since there's a placement with >= that many nodes)
    if (maxValidNodes === 0) return new Set<number>();
    return new Set(Array.from({ length: maxValidNodes }, (_, i) => i + 1));
  });

  // Get ALL filtered previews based on current settings (matching minimum nodes)
  // Note: previewsData already contains previews for the selected model (fetched via API)
  // We filter by sharding/instance type and min nodes, returning ALL eligible previews
  const filteredPreviews = $derived(() => {
    if (!selectedModelId || previewsData.length === 0) return [];

    // Find previews matching sharding/instance type (model_id filter not needed since previewsData is already for selected model)
    const matchingPreviews = previewsData.filter(
      (p: PlacementPreview) =>
        p.sharding === selectedSharding &&
        matchesSelectedRuntime(p.instance_meta) &&
        p.error === null &&
        p.memory_delta_by_node !== null,
    );

    // Filter to previews with node count >= selectedMinNodes, sorted by node count (ascending)
    return matchingPreviews
      .filter(
        (p: PlacementPreview) => getPreviewNodeCount(p) >= selectedMinNodes,
      )
      .sort(
        (a: PlacementPreview, b: PlacementPreview) =>
          getPreviewNodeCount(a) - getPreviewNodeCount(b),
      );
  });

  // Get the first filtered preview (for launch function compatibility)
  const filteredPreview = $derived(() => filteredPreviews()[0] ?? null);

  // Auto-update selectedMinNodes when node count changes (default to 1 = show all placements)
  $effect(() => {
    const maxNodes = availableMinNodes;
    if (!minNodesInitialized && maxNodes > 0) {
      // On initial load, default to 1 (minimum) to show all valid placements
      selectedMinNodes = 1;
      minNodesInitialized = true;
    } else if (selectedMinNodes > maxNodes) {
      // If current selection exceeds available nodes, cap it
      selectedMinNodes = maxNodes;
    }
  });

  // Auto-adjust selectedMinNodes to a valid value when it becomes invalid
  $effect(() => {
    const valid = validMinNodeCounts();
    if (valid.size > 0 && !valid.has(selectedMinNodes)) {
      // Find the smallest valid count >= current selection, or the largest valid count
      const validArray = Array.from(valid).sort((a, b) => a - b);
      const nextValid =
        validArray.find((n) => n >= selectedMinNodes) ??
        validArray[validArray.length - 1];
      if (nextValid !== undefined) {
        selectedMinNodes = nextValid;
      }
    }
  });

  // Calculate total memory usage across all nodes
  const clusterMemory = $derived(() => {
    if (!data) return { used: 0, total: 0 };
    return Object.values(data.nodes).reduce(
      (acc, n) => {
        const total =
          n.macmon_info?.memory?.ram_total ?? n.system_info?.memory ?? 0;
        const used = n.macmon_info?.memory?.ram_usage ?? 0;
        return { used: acc.used + used, total: acc.total + total };
      },
      { used: 0, total: 0 },
    );
  });
</script>

<!-- Global event listeners for slider dragging -->
<svelte:window
  onmousemove={handleSliderMouseMove}
  onmouseup={handleSliderMouseUp}
  ontouchmove={handleSliderTouchMove}
  ontouchend={handleSliderTouchEnd}
/>

<div
  class="relative h-screen w-full flex flex-col bg-exo-dark-gray overflow-hidden"
>
  <!-- Scanline overlay -->
  <div
    class="fixed inset-0 pointer-events-none z-50 scanlines opacity-20"
  ></div>

  <!-- Shooting Stars Background - one every ~15s -->
  <div class="shooting-stars">
    <div
      class="shooting-star"
      style="top: 10%; left: 20%; --duration: 45s; --delay: 0s;"
    ></div>
    <div
      class="shooting-star"
      style="top: 30%; left: 65%; --duration: 45s; --delay: 15s;"
    ></div>
    <div
      class="shooting-star"
      style="top: 50%; left: 40%; --duration: 45s; --delay: 30s;"
    ></div>
  </div>

  {#if !topologyOnlyEnabled}
    <HeaderNav
      showHome={chatStarted}
      onHome={handleGoHome}
      showSidebarToggle={true}
      {sidebarVisible}
      onToggleSidebar={toggleChatSidebarVisible}
    />
  {/if}

  <!-- Main Content -->
  <main class="flex-1 flex overflow-hidden relative">
    <!-- Left: Conversation History Sidebar (hidden in topology-only mode or when toggled off) -->
    {#if !topologyOnlyEnabled && sidebarVisible}
      <div class="w-80 flex-shrink-0 border-r border-exo-yellow/10">
        <ChatSidebar class="h-full" />
      </div>
    {/if}

    {#if topologyOnlyEnabled}
      <!-- TOPOLOGY ONLY MODE: Full-screen topology -->
      <div
        class="flex-1 flex flex-col min-h-0 min-w-0 p-4"
        in:fade={{ duration: 300 }}
      >
        <div
          class="flex-1 relative bg-exo-dark-gray/40 rounded-lg overflow-hidden"
        >
          <TopologyGraph
            class="w-full h-full"
            highlightedNodes={highlightedNodes()}
          />
          <!-- Exit topology-only mode button -->
          <button
            type="button"
            onclick={toggleTopologyOnlyMode}
            class="absolute bottom-4 right-4 p-2 rounded border border-exo-yellow/30 bg-exo-dark-gray/80 hover:border-exo-yellow/50 hover:bg-exo-dark-gray transition-colors cursor-pointer backdrop-blur-sm"
            title="Exit topology only mode"
          >
            <svg
              class="w-5 h-5 text-exo-yellow"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <circle cx="12" cy="5" r="2" fill="currentColor" />
              <circle cx="5" cy="19" r="2" fill="currentColor" />
              <circle cx="19" cy="19" r="2" fill="currentColor" />
              <path stroke-linecap="round" d="M12 7v5m0 0l-5 5m5-5l5 5" />
            </svg>
          </button>
        </div>
      </div>
    {:else if !chatStarted}
      <!-- WELCOME STATE: Topology + Instance Controls (no left sidebar for cleaner look) -->
      <div
        class="flex-1 flex overflow-visible relative"
        in:fade={{ duration: 300 }}
        out:fade={{ duration: 200 }}
      >
        <!-- Center: MAIN TOPOLOGY DISPLAY -->
        <div class="flex-1 flex flex-col min-h-0 min-w-0 py-4">
          <!-- Topology Container - Takes most of the space -->
          <div
            class="flex-1 relative bg-exo-dark-gray/40 mx-4 mb-4 rounded-lg overflow-hidden"
          >
            <!-- The main topology graph - full container -->
            <TopologyGraph
              class="w-full h-full"
              highlightedNodes={highlightedNodes()}
            />
          </div>

          <!-- Chat Input - Below topology -->
          <div class="px-4 pt-6 pb-8">
            <div class="max-w-3xl mx-auto">
              <ChatForm
                placeholder="Ask anything"
                showHelperText={false}
                showModelSelector={true}
                modelTasks={modelTasks()}
              />
            </div>
          </div>
        </div>

        <!-- Right Sidebar: Instance Controls (wider on welcome page for better visibility) -->
        <aside
          class="w-80 border-l border-exo-yellow/10 bg-exo-dark-gray flex flex-col flex-shrink-0"
        >
          <!-- Running Instances Panel (only shown when instances exist) - Scrollable -->
          {#if instanceCount > 0}
            <div class="p-4 flex-shrink-0">
              <!-- Panel Header -->
              <div class="flex items-center gap-2 mb-4">
                <div
                  class="w-2 h-2 bg-exo-yellow rounded-full shadow-[0_0_8px_rgba(255,215,0,0.6)] animate-pulse"
                ></div>
                <h3
                  class="text-xs text-exo-yellow font-mono tracking-[0.2em] uppercase"
                >
                  Instances
                </h3>
                <div
                  class="flex-1 h-px bg-gradient-to-r from-exo-yellow/30 to-transparent"
                ></div>
              </div>

              <div
                bind:this={instancesContainerRef}
                class="max-h-72 xl:max-h-96 space-y-3 overflow-y-auto overflow-x-hidden py-px"
              >
                {#each Object.entries(instanceData) as [id, instance]}
                  {@const downloadInfo = getInstanceDownloadStatus(
                    id,
                    instance,
                  )}
                  {@const statusText = downloadInfo.statusText}
                  {@const isDownloading = downloadInfo.isDownloading}
                  {@const isFailed = statusText === "FAILED"}
                  {@const isLoading =
                    statusText === "LOADING" ||
                    statusText === "WARMING UP" ||
                    statusText === "WAITING"}
                  {@const isReady =
                    statusText === "READY" || statusText === "LOADED"}
                  {@const isRunning = statusText === "RUNNING"}
                  <!-- Instance Card -->
                  {@const instanceModelId = getInstanceModelId(instance)}
                  {@const instanceInfo = getInstanceInfo(instance)}
                  {@const instanceConnections =
                    getInstanceConnections(instance)}
                  <div
                    class="relative group cursor-pointer"
                    role="button"
                    tabindex="0"
                    onmouseenter={() => (hoveredInstanceId = id)}
                    onmouseleave={() => (hoveredInstanceId = null)}
                    onclick={() => {
                      if (
                        instanceModelId &&
                        instanceModelId !== "Unknown" &&
                        instanceModelId !== "Unknown Model"
                      ) {
                        setSelectedChatModel(instanceModelId);
                      }
                    }}
                    onkeydown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        if (
                          instanceModelId &&
                          instanceModelId !== "Unknown" &&
                          instanceModelId !== "Unknown Model"
                        ) {
                          setSelectedChatModel(instanceModelId);
                        }
                      }
                    }}
                  >
                    <!-- Corner accents -->
                    <div
                      class="absolute -top-px -left-px w-2 h-2 border-l border-t {isDownloading
                        ? 'border-blue-500/50'
                        : isFailed
                          ? 'border-red-500/50'
                          : isLoading
                            ? 'border-yellow-500/50'
                            : isReady
                              ? 'border-green-500/50'
                              : 'border-teal-500/50'}"
                    ></div>
                    <div
                      class="absolute -top-px -right-px w-2 h-2 border-r border-t {isDownloading
                        ? 'border-blue-500/50'
                        : isFailed
                          ? 'border-red-500/50'
                          : isLoading
                            ? 'border-yellow-500/50'
                            : isReady
                              ? 'border-green-500/50'
                              : 'border-teal-500/50'}"
                    ></div>
                    <div
                      class="absolute -bottom-px -left-px w-2 h-2 border-l border-b {isDownloading
                        ? 'border-blue-500/50'
                        : isFailed
                          ? 'border-red-500/50'
                          : isLoading
                            ? 'border-yellow-500/50'
                            : isReady
                              ? 'border-green-500/50'
                              : 'border-teal-500/50'}"
                    ></div>
                    <div
                      class="absolute -bottom-px -right-px w-2 h-2 border-r border-b {isDownloading
                        ? 'border-blue-500/50'
                        : isFailed
                          ? 'border-red-500/50'
                          : isLoading
                            ? 'border-yellow-500/50'
                            : isReady
                              ? 'border-green-500/50'
                              : 'border-teal-500/50'}"
                    ></div>

                    <div
                      class="bg-exo-dark-gray/60 border border-l-2 {isDownloading
                        ? 'border-blue-500/30 border-l-blue-400'
                        : isFailed
                          ? 'border-red-500/30 border-l-red-400'
                          : isLoading
                            ? 'border-exo-yellow/30 border-l-yellow-400'
                            : isReady
                              ? 'border-green-500/30 border-l-green-400'
                              : 'border-teal-500/30 border-l-teal-400'} p-3"
                    >
                      <div class="flex justify-between items-start mb-2 pl-2">
                        <div class="flex items-center gap-2">
                          <div
                            class="w-1.5 h-1.5 {isDownloading
                              ? 'bg-blue-400 animate-pulse'
                              : isFailed
                                ? 'bg-red-400'
                                : isLoading
                                  ? 'bg-yellow-400 animate-pulse'
                                  : isReady
                                    ? 'bg-green-400'
                                    : 'bg-teal-400'} rounded-full shadow-[0_0_6px_currentColor]"
                          ></div>
                          <span
                            class="text-exo-light-gray font-mono text-sm tracking-wider"
                            >{id.slice(0, 8).toUpperCase()}</span
                          >
                        </div>
                        <button
                          onclick={() => deleteInstance(id)}
                          class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
                        >
                          DELETE
                        </button>
                      </div>
                      <div class="pl-2">
                        <div
                          class="text-exo-yellow text-xs font-mono tracking-wide truncate"
                        >
                          {getInstanceModelId(instance)}
                        </div>
                        <div class="text-white/60 text-xs font-mono">
                          Strategy: <span class="text-white/80"
                            >{instanceInfo.sharding} ({instanceInfo.instanceType})</span
                          >
                        </div>
                        {#if instanceModelId && instanceModelId !== "Unknown" && instanceModelId !== "Unknown Model"}
                          <a
                            class="inline-flex items-center gap-1 text-[11px] text-white/60 hover:text-exo-yellow transition-colors mt-1"
                            href={`https://huggingface.co/${instanceModelId}`}
                            target="_blank"
                            rel="noreferrer noopener"
                            aria-label="View model on Hugging Face"
                          >
                            <span>Hugging Face</span>
                            <svg
                              class="w-3.5 h-3.5"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              stroke-width="2"
                              stroke-linecap="round"
                              stroke-linejoin="round"
                            >
                              <path d="M14 3h7v7" />
                              <path d="M10 14l11-11" />
                              <path
                                d="M21 14v6a1 1 0 0 1-1 1h-16a1 1 0 0 1-1-1v-16a1 1 0 0 1 1-1h6"
                              />
                            </svg>
                          </a>
                        {/if}
                        {#if instanceInfo.nodeNames.length > 0}
                          <div class="text-white/60 text-xs font-mono">
                            {instanceInfo.nodeNames.join(", ")}
                          </div>
                        {/if}
                        {#if debugEnabled && instanceConnections.length > 0}
                          <div class="mt-2 space-y-1">
                            {#each instanceConnections as conn}
                              <div
                                class="text-[11px] leading-snug font-mono text-white/70"
                              >
                                <span>{conn.from} -> {conn.to}: {conn.ip}</span>
                                <span
                                  class={conn.missingIface
                                    ? "text-red-400"
                                    : "text-white/60"}
                                >
                                  ({conn.ifaceLabel})</span
                                >
                              </div>
                            {/each}
                          </div>
                        {/if}

                        <!-- Download Progress -->
                        {#if downloadInfo.isDownloading && downloadInfo.progress}
                          <div class="mt-2 space-y-1">
                            <div class="flex justify-between text-xs font-mono">
                              <span class="text-blue-400"
                                >{downloadInfo.progress.percentage.toFixed(
                                  1,
                                )}%</span
                              >
                              <span class="text-exo-light-gray"
                                >{formatBytes(
                                  downloadInfo.progress.downloadedBytes,
                                )}/{formatBytes(
                                  downloadInfo.progress.totalBytes,
                                )}</span
                              >
                            </div>
                            <div
                              class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                            >
                              <div
                                class="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
                                style="width: {downloadInfo.progress
                                  .percentage}%"
                              ></div>
                            </div>
                            <div
                              class="flex justify-between text-xs font-mono text-exo-light-gray"
                            >
                              <span
                                >{formatSpeed(
                                  downloadInfo.progress.speed,
                                )}</span
                              >
                              <span
                                >ETA: {formatEta(
                                  downloadInfo.progress.etaMs,
                                )}</span
                              >
                              <span
                                >{downloadInfo.progress
                                  .completedFiles}/{downloadInfo.progress
                                  .totalFiles} files</span
                              >
                            </div>
                          </div>
                          {#if downloadInfo.perNode.length > 0}
                            <div
                              class="mt-2 space-y-2 max-h-48 overflow-y-auto pr-1"
                            >
                              {#each downloadInfo.perNode as nodeProg}
                                {@const nodePercent = Math.min(
                                  100,
                                  Math.max(0, nodeProg.progress.percentage),
                                )}
                                {@const isExpanded =
                                  instanceDownloadExpandedNodes.has(
                                    nodeProg.nodeId,
                                  )}
                                <div
                                  class="rounded border border-exo-medium-gray/40 bg-exo-black/30 p-2"
                                >
                                  <button
                                    type="button"
                                    class="w-full text-left space-y-1.5"
                                    onclick={() =>
                                      toggleInstanceDownloadDetails(
                                        nodeProg.nodeId,
                                      )}
                                  >
                                    <div
                                      class="flex items-center justify-between text-[11px] font-mono text-exo-light-gray"
                                    >
                                      <span class="text-white/80 truncate pr-2"
                                        >{nodeProg.nodeName}</span
                                      >
                                      <span
                                        class="flex items-center gap-1 text-blue-300"
                                      >
                                        {nodePercent.toFixed(1)}%
                                        <svg
                                          class="w-3 h-3 text-exo-light-gray"
                                          viewBox="0 0 20 20"
                                          fill="none"
                                          stroke="currentColor"
                                          stroke-width="2"
                                        >
                                          <path
                                            d="M6 8l4 4 4-4"
                                            class={isExpanded
                                              ? "transform rotate-180 origin-center transition-transform duration-150"
                                              : "transition-transform duration-150"}
                                          ></path>
                                        </svg>
                                      </span>
                                    </div>
                                    <div
                                      class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                                    >
                                      <div
                                        class="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
                                        style="width: {nodePercent.toFixed(1)}%"
                                      ></div>
                                    </div>
                                    <div
                                      class="flex items-center justify-between text-[11px] font-mono text-exo-light-gray"
                                    >
                                      <span
                                        >{formatBytes(
                                          nodeProg.progress.downloadedBytes,
                                        )} / {formatBytes(
                                          nodeProg.progress.totalBytes,
                                        )}</span
                                      >
                                      <span
                                        >{formatSpeed(nodeProg.progress.speed)} •
                                        ETA {formatEta(
                                          nodeProg.progress.etaMs,
                                        )}</span
                                      >
                                    </div>
                                  </button>

                                  {#if isExpanded}
                                    <div class="mt-2 space-y-1.5">
                                      {#if nodeProg.progress.files.length === 0}
                                        <div
                                          class="text-[11px] font-mono text-exo-light-gray/70"
                                        >
                                          No file details reported.
                                        </div>
                                      {:else}
                                        {#each nodeProg.progress.files as f}
                                          {@const filePercent = Math.min(
                                            100,
                                            Math.max(0, f.percentage ?? 0),
                                          )}
                                          {@const isFileComplete =
                                            filePercent >= 100}
                                          <div
                                            class="rounded border border-exo-medium-gray/30 bg-exo-black/40 p-2"
                                          >
                                            <div
                                              class="flex items-center justify-between text-[10px] font-mono text-exo-light-gray/90"
                                            >
                                              <span class="truncate pr-2"
                                                >{f.name}</span
                                              >
                                              <span
                                                class={isFileComplete
                                                  ? "text-green-400"
                                                  : "text-white/80"}
                                                >{filePercent.toFixed(1)}%</span
                                              >
                                            </div>
                                            <div
                                              class="relative h-1 bg-exo-black/60 rounded-sm overflow-hidden mt-1"
                                            >
                                              <div
                                                class="absolute inset-y-0 left-0 bg-gradient-to-r {isFileComplete
                                                  ? 'from-green-500 to-green-400'
                                                  : 'from-exo-yellow to-exo-yellow/70'} transition-all duration-300"
                                                style="width: {filePercent.toFixed(
                                                  1,
                                                )}%"
                                              ></div>
                                            </div>
                                            <div
                                              class="flex items-center justify-between text-[10px] text-exo-light-gray/70 mt-0.5"
                                            >
                                              <span
                                                >{formatBytes(
                                                  f.downloadedBytes,
                                                )} / {formatBytes(
                                                  f.totalBytes,
                                                )}</span
                                              >
                                              <span
                                                >{formatSpeed(f.speed)} • ETA {formatEta(
                                                  f.etaMs,
                                                )}</span
                                              >
                                            </div>
                                          </div>
                                        {/each}
                                      {/if}
                                    </div>
                                  {/if}
                                </div>
                              {/each}
                            </div>
                          {/if}
                          <div
                            class="text-xs text-blue-400 font-mono tracking-wider mt-1"
                          >
                            DOWNLOADING
                          </div>
                        {:else}
                          <div
                            class="text-xs {getStatusColor(
                              downloadInfo.statusText,
                            )} font-mono tracking-wider mt-1"
                          >
                            {downloadInfo.statusText}
                          </div>
                        {/if}
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Models Panel - Scrollable -->
          <div class="p-4 flex-1 overflow-y-auto">
            <!-- Panel Header -->
            <div class="flex items-center gap-2 mb-3 flex-shrink-0">
              <div class="w-2 h-2 border border-exo-yellow/60 rotate-45"></div>
              <h3
                class="text-xs text-exo-yellow font-mono tracking-[0.2em] uppercase"
              >
                Launch Instance
              </h3>
              <div
                class="flex-1 h-px bg-gradient-to-r from-exo-yellow/30 to-transparent"
              ></div>
              <span class="text-sm text-white/70 font-mono"
                >{models.length} models</span
              >
            </div>

            <!-- Model Dropdown (Custom) -->
            <div class="flex-shrink-0 mb-3 relative">
              <button
                type="button"
                onclick={() => (isModelDropdownOpen = !isModelDropdownOpen)}
                class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded pl-3 pr-8 py-2.5 text-sm font-mono text-left tracking-wide cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70 {isModelDropdownOpen
                  ? 'border-exo-yellow/70'
                  : ''}"
              >
                {#if selectedModelId}
                  {@const foundModel = models.find(
                    (m) => m.id === selectedModelId,
                  )}
                  {#if foundModel}
                    {@const sizeGB = getModelSizeGB(foundModel)}
                    {@const isImageModel = modelSupportsImageGeneration(
                      foundModel.id,
                    )}
                    <span
                      class="flex items-center justify-between gap-2 w-full pr-4"
                    >
                      <span
                        class="flex items-center gap-2 text-exo-light-gray truncate"
                      >
                        {#if isImageModel}
                          <svg
                            class="w-4 h-4 flex-shrink-0 text-exo-yellow"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            stroke-width="2"
                          >
                            <rect
                              x="3"
                              y="3"
                              width="18"
                              height="18"
                              rx="2"
                              ry="2"
                            />
                            <circle cx="8.5" cy="8.5" r="1.5" />
                            <polyline points="21 15 16 10 5 21" />
                          </svg>
                        {/if}
                        <span class="truncate"
                          >{foundModel.name || foundModel.id}</span
                        >
                      </span>
                      <span class="text-white/50 text-xs flex-shrink-0"
                        >{sizeGB >= 1
                          ? sizeGB.toFixed(0)
                          : sizeGB.toFixed(1)}GB</span
                      >
                    </span>
                  {:else}
                    <span class="text-exo-light-gray">{selectedModelId}</span>
                  {/if}
                {:else}
                  <span class="text-white/50">— SELECT MODEL —</span>
                {/if}
              </button>
              <div
                class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none transition-transform duration-200 {isModelDropdownOpen
                  ? 'rotate-180'
                  : ''}"
              >
                <svg
                  class="w-4 h-4 text-exo-yellow/60"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>

              {#if isModelDropdownOpen}
                <!-- Backdrop to close dropdown -->
                <button
                  type="button"
                  class="fixed inset-0 z-40 cursor-default"
                  onclick={() => (isModelDropdownOpen = false)}
                  aria-label="Close dropdown"
                ></button>

                <!-- Dropdown Panel -->
                <div
                  class="absolute top-full left-0 right-0 mt-1 bg-exo-dark-gray border border-exo-yellow/30 rounded shadow-lg shadow-black/50 z-50 max-h-64 overflow-y-auto"
                >
                  <!-- Search within dropdown -->
                  <div
                    class="sticky top-0 bg-exo-dark-gray border-b border-exo-medium-gray/30 p-2"
                  >
                    <input
                      type="text"
                      placeholder="Search models..."
                      bind:value={modelDropdownSearch}
                      class="w-full bg-exo-dark-gray/60 border border-exo-medium-gray/30 rounded px-2 py-1.5 text-xs font-mono text-white/80 placeholder:text-white/40 focus:outline-none focus:border-exo-yellow/50"
                    />
                  </div>

                  <!-- Options -->
                  <div class="py-1">
                    {#each sortedModels().filter((m) => !modelDropdownSearch || (m.name || m.id)
                          .toLowerCase()
                          .includes(modelDropdownSearch.toLowerCase())) as model}
                      {@const sizeGB = getModelSizeGB(model)}
                      {@const modelCanFit = hasEnoughMemory(model)}
                      {@const isImageModel = modelSupportsImageGeneration(
                        model.id,
                      )}
                      <button
                        type="button"
                        onclick={() => {
                          if (modelCanFit) {
                            selectPreviewModel(model.id);
                            saveLaunchDefaults();
                            isModelDropdownOpen = false;
                            modelDropdownSearch = "";
                          }
                        }}
                        disabled={!modelCanFit}
                        class="w-full px-3 py-2 text-left text-sm font-mono tracking-wide transition-colors duration-100 flex items-center justify-between gap-2 {selectedModelId ===
                        model.id
                          ? 'bg-transparent text-exo-yellow cursor-pointer'
                          : modelCanFit
                            ? 'text-white/80 hover:text-exo-yellow cursor-pointer'
                            : 'text-white/30 cursor-default'}"
                      >
                        <span class="flex items-center gap-2 truncate flex-1">
                          {#if isImageModel}
                            <svg
                              class="w-4 h-4 flex-shrink-0 text-exo-yellow"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              stroke-width="2"
                              aria-label="Image generation model"
                            >
                              <rect
                                x="3"
                                y="3"
                                width="18"
                                height="18"
                                rx="2"
                                ry="2"
                              />
                              <circle cx="8.5" cy="8.5" r="1.5" />
                              <polyline points="21 15 16 10 5 21" />
                            </svg>
                          {/if}
                          <span class="truncate">{model.name || model.id}</span>
                        </span>
                        <span
                          class="flex-shrink-0 text-xs {modelCanFit
                            ? 'text-white/50'
                            : 'text-red-400/60'}"
                        >
                          {sizeGB >= 1
                            ? sizeGB.toFixed(0)
                            : sizeGB.toFixed(1)}GB
                        </span>
                      </button>
                    {:else}
                      <div class="px-3 py-2 text-xs text-white/50 font-mono">
                        No models found
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>

            <!-- Configuration Options -->
            <div class="flex-shrink-0 mb-4 space-y-3">
              <!-- Sharding -->
              <div>
                <div class="text-xs text-white/70 font-mono mb-2">
                  Sharding:
                </div>
                <div class="flex gap-2">
                  <button
                    onclick={() => {
                      selectedSharding = "Pipeline";
                      saveLaunchDefaults();
                    }}
                    class="flex items-center gap-2 py-2 px-4 text-sm font-mono border rounded transition-all duration-200 cursor-pointer {selectedSharding ===
                    'Pipeline'
                      ? 'bg-transparent text-exo-yellow border-exo-yellow'
                      : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                  >
                    <span
                      class="w-4 h-4 rounded-full border-2 flex items-center justify-center {selectedSharding ===
                      'Pipeline'
                        ? 'border-exo-yellow'
                        : 'border-exo-medium-gray'}"
                    >
                      {#if selectedSharding === "Pipeline"}
                        <span class="w-2 h-2 rounded-full bg-exo-yellow"></span>
                      {/if}
                    </span>
                    Pipeline
                  </button>
                  <button
                    onclick={() => {
                      selectedSharding = "Tensor";
                      saveLaunchDefaults();
                    }}
                    class="flex items-center gap-2 py-2 px-4 text-sm font-mono border rounded transition-all duration-200 cursor-pointer {selectedSharding ===
                    'Tensor'
                      ? 'bg-transparent text-exo-yellow border-exo-yellow'
                      : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                  >
                    <span
                      class="w-4 h-4 rounded-full border-2 flex items-center justify-center {selectedSharding ===
                      'Tensor'
                        ? 'border-exo-yellow'
                        : 'border-exo-medium-gray'}"
                    >
                      {#if selectedSharding === "Tensor"}
                        <span class="w-2 h-2 rounded-full bg-exo-yellow"></span>
                      {/if}
                    </span>
                    Tensor
                  </button>
                </div>
              </div>

              <!-- Instance Type -->
              <div>
                <div class="text-xs text-white/70 font-mono mb-2">
                  Instance Type:
                </div>
                <div class="flex gap-2">
                  <button
                    onclick={() => {
                      selectedInstanceType = "MlxRing";
                      saveLaunchDefaults();
                    }}
                    class="flex items-center gap-2 py-2 px-4 text-sm font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType ===
                    'MlxRing'
                      ? 'bg-transparent text-exo-yellow border-exo-yellow'
                      : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                  >
                    <span
                      class="w-4 h-4 rounded-full border-2 flex items-center justify-center {selectedInstanceType ===
                      'MlxRing'
                        ? 'border-exo-yellow'
                        : 'border-exo-medium-gray'}"
                    >
                      {#if selectedInstanceType === "MlxRing"}
                        <span class="w-2 h-2 rounded-full bg-exo-yellow"></span>
                      {/if}
                    </span>
                    MLX Ring
                  </button>
                  <button
                    onclick={() => {
                      selectedInstanceType = "MlxIbv";
                      saveLaunchDefaults();
                    }}
                    class="flex items-center gap-2 py-2 px-4 text-sm font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType ===
                    'MlxIbv'
                      ? 'bg-transparent text-exo-yellow border-exo-yellow'
                      : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                  >
                    <span
                      class="w-4 h-4 rounded-full border-2 flex items-center justify-center {selectedInstanceType ===
                      'MlxIbv'
                        ? 'border-exo-yellow'
                        : 'border-exo-medium-gray'}"
                    >
                      {#if selectedInstanceType === "MlxIbv"}
                        <span class="w-2 h-2 rounded-full bg-exo-yellow"></span>
                      {/if}
                    </span>
                    MLX RDMA
                  </button>
                </div>
              </div>

              <!-- Minimum Nodes (discrete slider with drag support) -->
              <div>
                <div class="text-xs text-white/70 font-mono mb-2">
                  Minimum Nodes:
                </div>
                <!-- Discrete slider track with drag support -->
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div
                  bind:this={sliderTrackElement}
                  class="relative h-16 cursor-pointer select-none px-2 pr-6"
                  onmousedown={handleSliderMouseDown}
                  ontouchstart={handleSliderTouchStart}
                >
                  <!-- Track background - extends full width to align with edge dots -->
                  <div
                    class="absolute top-6 left-0 right-0 h-2 bg-exo-medium-gray/50 rounded-full"
                  ></div>
                  <!-- Active track (fills up to selected) -->
                  {#if availableMinNodes > 1}
                    <div
                      class="absolute top-6 left-0 h-2 bg-white/30 rounded-full transition-all pointer-events-none"
                      style="width: {((selectedMinNodes - 1) /
                        (availableMinNodes - 1)) *
                        100}%"
                    ></div>
                  {/if}
                  <!-- Dots and labels for each node count -->
                  {#each Array.from({ length: availableMinNodes }, (_, i) => i + 1) as n}
                    {@const isValid = validMinNodeCounts().has(n)}
                    {@const isSelected = selectedMinNodes === n}
                    {@const position =
                      availableMinNodes > 1
                        ? ((n - 1) / (availableMinNodes - 1)) * 100
                        : 50}
                    <div
                      class="absolute flex flex-col items-center pointer-events-none"
                      style="left: {position}%; top: 0; transform: translateX(-50%);"
                    >
                      <!-- Dot -->
                      <span
                        class="rounded-full transition-all {isSelected
                          ? 'w-6 h-6 bg-exo-yellow shadow-[0_0_10px_rgba(255,215,0,0.6)]'
                          : isValid
                            ? 'w-4 h-4 bg-exo-light-gray/70 mt-1'
                            : 'w-3 h-3 bg-exo-medium-gray/50 mt-1.5'}"
                      ></span>
                      <!-- Number label below dot -->
                      <span
                        class="text-sm font-mono mt-1.5 tabular-nums transition-colors {isSelected
                          ? 'text-exo-yellow font-bold'
                          : isValid
                            ? 'text-white/70'
                            : 'text-white/30'}">{n}</span
                      >
                    </div>
                  {/each}
                </div>
              </div>
            </div>

            <!-- Selected Model Preview -->
            <div class="space-y-3">
              {#if models.length === 0}
                <div class="text-center py-8">
                  <div
                    class="text-xs text-white/70 font-mono tracking-wider uppercase"
                  >
                    Loading models...
                  </div>
                </div>
              {:else if loadingPreviews}
                <div class="text-center py-8">
                  <div
                    class="text-xs text-exo-yellow font-mono tracking-wider uppercase animate-pulse"
                  >
                    Loading preview...
                  </div>
                </div>
              {:else}
                {@const selectedModel = models.find(
                  (m) => m.id === selectedModelId,
                )}
                {@const allPreviews = filteredPreviews()}
                {#if selectedModel && allPreviews.length > 0}
                  {@const downloadStatus = getModelDownloadStatus(
                    selectedModel.id,
                  )}
                  {@const tags = modelTags()[selectedModel.id] || []}
                  <div class="space-y-3">
                    {#each allPreviews as apiPreview, i}
                      <div
                        role="group"
                        onmouseenter={() => {
                          if (apiPreview.memory_delta_by_node) {
                            hoveredPreviewNodes = new Set(
                              Object.entries(apiPreview.memory_delta_by_node)
                                .filter(([, delta]) => (delta ?? 0) > 0)
                                .map(([nodeId]) => nodeId),
                            );
                          }
                        }}
                        onmouseleave={() => (hoveredPreviewNodes = new Set())}
                      >
                        <ModelCard
                          model={selectedModel}
                          isLaunching={launchingModelId === selectedModel.id}
                          {downloadStatus}
                          nodes={data?.nodes ?? {}}
                          sharding={apiPreview.sharding}
                          runtime={apiPreview.instance_meta}
                          onLaunch={() =>
                            launchInstance(selectedModel.id, apiPreview)}
                          {tags}
                          {apiPreview}
                          modelIdOverride={apiPreview.model_id}
                        />
                      </div>
                    {/each}
                  </div>
                {:else if selectedModel}
                  <div class="text-center py-4">
                    <div class="text-xs text-white/50 font-mono">
                      No valid configurations for current settings
                    </div>
                  </div>
                {/if}
              {/if}
            </div>
          </div>
        </aside>
      </div>
    {:else}
      <!-- CHAT STATE: Chat + Mini-Map -->
      <div class="flex-1 flex overflow-hidden">
        <!-- Chat Area -->
        <div
          class="flex-1 flex flex-col min-w-0 overflow-hidden"
          in:fade={{ duration: 300, delay: 100 }}
        >
          <div
            class="flex-1 overflow-y-auto px-8 py-6"
            bind:this={chatScrollRef}
          >
            <div class="max-w-7xl mx-auto">
              <ChatMessages scrollParent={chatScrollRef} />
            </div>
          </div>

          <div
            class="flex-shrink-0 px-8 pb-6 pt-4 bg-gradient-to-t from-exo-black via-exo-black to-transparent"
          >
            <div class="max-w-7xl mx-auto">
              <ChatForm
                placeholder="Ask anything"
                showModelSelector={true}
                modelTasks={modelTasks()}
              />
            </div>
          </div>
        </div>

        <!-- Right: Mini-Map Sidebar -->
        {#if minimized}
          <aside
            class="w-80 border-l border-exo-yellow/20 bg-exo-dark-gray flex flex-col flex-shrink-0 overflow-y-auto"
            in:fly={{ x: 100, duration: 400, easing: cubicInOut }}
          >
            <!-- Topology Section - clickable to go back to main view -->
            <button
              class="p-4 border-b border-exo-medium-gray/30 w-full text-left cursor-pointer hover:bg-exo-medium-gray/10 transition-colors"
              onclick={handleGoHome}
              title="Click to return to main topology view"
            >
              <div class="flex items-center justify-between mb-3">
                <div
                  class="text-xs text-exo-yellow tracking-[0.2em] uppercase flex items-center gap-2"
                >
                  <span
                    class="w-1.5 h-1.5 bg-exo-yellow rounded-full status-pulse"
                  ></span>
                  TOPOLOGY
                </div>
                <span class="text-xs text-white/70 tabular-nums"
                  >{nodeCount} {nodeCount === 1 ? "NODE" : "NODES"}</span
                >
              </div>

              <div
                class="relative aspect-square bg-exo-dark-gray rounded-lg overflow-hidden"
              >
                <TopologyGraph highlightedNodes={highlightedNodes()} />
              </div>
            </button>

            <!-- Instances Section (only shown when instances exist) -->
            {#if instanceCount > 0}
              <div class="p-4 flex-1">
                <!-- Panel Header -->
                <div class="flex items-center gap-2 mb-4">
                  <div
                    class="w-2 h-2 bg-exo-yellow rounded-full shadow-[0_0_8px_rgba(255,215,0,0.6)] animate-pulse"
                  ></div>
                  <h3
                    class="text-xs text-exo-yellow font-mono tracking-[0.2em] uppercase"
                  >
                    Instances
                  </h3>
                  <div
                    class="flex-1 h-px bg-gradient-to-r from-exo-yellow/30 to-transparent"
                  ></div>
                </div>
                <div
                  class="space-y-3 max-h-72 xl:max-h-96 overflow-y-auto overflow-x-hidden py-px pr-1"
                >
                  {#each Object.entries(instanceData) as [id, instance]}
                    {@const downloadInfo = getInstanceDownloadStatus(
                      id,
                      instance,
                    )}
                    {@const statusText = downloadInfo.statusText}
                    {@const isDownloading = downloadInfo.isDownloading}
                    {@const isFailed = statusText === "FAILED"}
                    {@const isLoading =
                      statusText === "LOADING" ||
                      statusText === "WARMING UP" ||
                      statusText === "WAITING"}
                    {@const isReady =
                      statusText === "READY" || statusText === "LOADED"}
                    {@const isRunning = statusText === "RUNNING"}
                    <!-- Instance Card -->
                    {@const instanceModelId = getInstanceModelId(instance)}
                    {@const instanceInfo = getInstanceInfo(instance)}
                    {@const instanceConnections =
                      getInstanceConnections(instance)}
                    <div
                      class="relative group cursor-pointer"
                      role="button"
                      tabindex="0"
                      onmouseenter={() => (hoveredInstanceId = id)}
                      onmouseleave={() => (hoveredInstanceId = null)}
                      onclick={() => {
                        if (
                          instanceModelId &&
                          instanceModelId !== "Unknown" &&
                          instanceModelId !== "Unknown Model"
                        ) {
                          setSelectedChatModel(instanceModelId);
                        }
                      }}
                      onkeydown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          if (
                            instanceModelId &&
                            instanceModelId !== "Unknown" &&
                            instanceModelId !== "Unknown Model"
                          ) {
                            setSelectedChatModel(instanceModelId);
                          }
                        }
                      }}
                    >
                      <!-- Corner accents -->
                      <div
                        class="absolute -top-px -left-px w-2 h-2 border-l border-t {isDownloading
                          ? 'border-blue-500/50'
                          : isFailed
                            ? 'border-red-500/50'
                            : isLoading
                              ? 'border-yellow-500/50'
                              : isReady
                                ? 'border-green-500/50'
                                : 'border-teal-500/50'}"
                      ></div>
                      <div
                        class="absolute -top-px -right-px w-2 h-2 border-r border-t {isDownloading
                          ? 'border-blue-500/50'
                          : isFailed
                            ? 'border-red-500/50'
                            : isLoading
                              ? 'border-yellow-500/50'
                              : isReady
                                ? 'border-green-500/50'
                                : 'border-teal-500/50'}"
                      ></div>
                      <div
                        class="absolute -bottom-px -left-px w-2 h-2 border-l border-b {isDownloading
                          ? 'border-blue-500/50'
                          : isFailed
                            ? 'border-red-500/50'
                            : isLoading
                              ? 'border-yellow-500/50'
                              : isReady
                                ? 'border-green-500/50'
                                : 'border-teal-500/50'}"
                      ></div>
                      <div
                        class="absolute -bottom-px -right-px w-2 h-2 border-r border-b {isDownloading
                          ? 'border-blue-500/50'
                          : isFailed
                            ? 'border-red-500/50'
                            : isLoading
                              ? 'border-yellow-500/50'
                              : isReady
                                ? 'border-green-500/50'
                                : 'border-teal-500/50'}"
                      ></div>

                      <div
                        class="bg-exo-dark-gray/60 border border-l-2 {isDownloading
                          ? 'border-blue-500/30 border-l-blue-400'
                          : isFailed
                            ? 'border-red-500/30 border-l-red-400'
                            : isLoading
                              ? 'border-exo-yellow/30 border-l-yellow-400'
                              : isReady
                                ? 'border-green-500/30 border-l-green-400'
                                : 'border-teal-500/30 border-l-teal-400'} p-3"
                      >
                        <div class="flex justify-between items-start mb-2 pl-2">
                          <div class="flex items-center gap-2">
                            <div
                              class="w-1.5 h-1.5 {isDownloading
                                ? 'bg-blue-400 animate-pulse'
                                : isFailed
                                  ? 'bg-red-400'
                                  : isLoading
                                    ? 'bg-yellow-400 animate-pulse'
                                    : isReady
                                      ? 'bg-green-400'
                                      : 'bg-teal-400'} rounded-full shadow-[0_0_6px_currentColor]"
                            ></div>
                            <span
                              class="text-exo-light-gray font-mono text-sm tracking-wider"
                              >{id.slice(0, 8).toUpperCase()}</span
                            >
                          </div>
                          <button
                            onclick={() => deleteInstance(id)}
                            class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
                          >
                            DELETE
                          </button>
                        </div>
                        <div class="pl-2">
                          <div
                            class="text-exo-yellow text-xs font-mono tracking-wide truncate"
                          >
                            {getInstanceModelId(instance)}
                          </div>
                          <div class="text-white/60 text-xs font-mono">
                            Strategy: <span class="text-white/80"
                              >{instanceInfo.sharding} ({instanceInfo.instanceType})</span
                            >
                          </div>
                          {#if instanceModelId && instanceModelId !== "Unknown" && instanceModelId !== "Unknown Model"}
                            <a
                              class="inline-flex items-center gap-1 text-[11px] text-white/60 hover:text-exo-yellow transition-colors mt-1"
                              href={`https://huggingface.co/${instanceModelId}`}
                              target="_blank"
                              rel="noreferrer noopener"
                              aria-label="View model on Hugging Face"
                            >
                              <span>Hugging Face</span>
                              <svg
                                class="w-3.5 h-3.5"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                              >
                                <path d="M14 3h7v7" />
                                <path d="M10 14l11-11" />
                                <path
                                  d="M21 14v6a1 1 0 0 1-1 1h-16a1 1 0 0 1-1-1v-16a1 1 0 0 1 1-1h6"
                                />
                              </svg>
                            </a>
                          {/if}
                          {#if instanceInfo.nodeNames.length > 0}
                            <div class="text-white/60 text-xs font-mono">
                              {instanceInfo.nodeNames.join(", ")}
                            </div>
                          {/if}
                          {#if debugEnabled && instanceConnections.length > 0}
                            <div class="mt-2 space-y-1">
                              {#each instanceConnections as conn}
                                <div
                                  class="text-[11px] leading-snug font-mono text-white/70"
                                >
                                  <span
                                    >{conn.from} -> {conn.to}: {conn.ip}</span
                                  >
                                  <span
                                    class={conn.missingIface
                                      ? "text-red-400"
                                      : "text-white/60"}
                                  >
                                    ({conn.ifaceLabel})</span
                                  >
                                </div>
                              {/each}
                            </div>
                          {/if}

                          <!-- Download Progress -->
                          {#if downloadInfo.isDownloading && downloadInfo.progress}
                            <div class="mt-2 space-y-1">
                              <div
                                class="flex justify-between text-xs font-mono"
                              >
                                <span class="text-blue-400"
                                  >{downloadInfo.progress.percentage.toFixed(
                                    1,
                                  )}%</span
                                >
                                <span class="text-exo-light-gray"
                                  >{formatBytes(
                                    downloadInfo.progress.downloadedBytes,
                                  )}/{formatBytes(
                                    downloadInfo.progress.totalBytes,
                                  )}</span
                                >
                              </div>
                              <div
                                class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                              >
                                <div
                                  class="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
                                  style="width: {downloadInfo.progress
                                    .percentage}%"
                                ></div>
                              </div>
                              <div
                                class="flex justify-between text-xs font-mono text-exo-light-gray"
                              >
                                <span
                                  >{formatSpeed(
                                    downloadInfo.progress.speed,
                                  )}</span
                                >
                                <span
                                  >ETA: {formatEta(
                                    downloadInfo.progress.etaMs,
                                  )}</span
                                >
                                <span
                                  >{downloadInfo.progress
                                    .completedFiles}/{downloadInfo.progress
                                    .totalFiles} files</span
                                >
                              </div>
                            </div>
                            {#if downloadInfo.perNode.length > 0}
                              <div
                                class="mt-2 space-y-2 max-h-48 overflow-y-auto pr-1"
                              >
                                {#each downloadInfo.perNode as nodeProg}
                                  {@const nodePercent = Math.min(
                                    100,
                                    Math.max(0, nodeProg.progress.percentage),
                                  )}
                                  {@const isExpanded =
                                    instanceDownloadExpandedNodes.has(
                                      nodeProg.nodeId,
                                    )}
                                  <div
                                    class="rounded border border-exo-medium-gray/40 bg-exo-black/30 p-2"
                                  >
                                    <button
                                      type="button"
                                      class="w-full text-left space-y-1.5"
                                      onclick={() =>
                                        toggleInstanceDownloadDetails(
                                          nodeProg.nodeId,
                                        )}
                                    >
                                      <div
                                        class="flex items-center justify-between text-[11px] font-mono text-exo-light-gray"
                                      >
                                        <span
                                          class="text-white/80 truncate pr-2"
                                          >{nodeProg.nodeName}</span
                                        >
                                        <span
                                          class="flex items-center gap-1 text-blue-300"
                                        >
                                          {nodePercent.toFixed(1)}%
                                          <svg
                                            class="w-3 h-3 text-exo-light-gray"
                                            viewBox="0 0 20 20"
                                            fill="none"
                                            stroke="currentColor"
                                            stroke-width="2"
                                          >
                                            <path
                                              d="M6 8l4 4 4-4"
                                              class={isExpanded
                                                ? "transform rotate-180 origin-center transition-transform duration-150"
                                                : "transition-transform duration-150"}
                                            ></path>
                                          </svg>
                                        </span>
                                      </div>
                                      <div
                                        class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                                      >
                                        <div
                                          class="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
                                          style="width: {nodePercent.toFixed(
                                            1,
                                          )}%"
                                        ></div>
                                      </div>
                                      <div
                                        class="flex items-center justify-between text-[11px] font-mono text-exo-light-gray"
                                      >
                                        <span
                                          >{formatBytes(
                                            nodeProg.progress.downloadedBytes,
                                          )} / {formatBytes(
                                            nodeProg.progress.totalBytes,
                                          )}</span
                                        >
                                        <span
                                          >{formatSpeed(
                                            nodeProg.progress.speed,
                                          )} • ETA {formatEta(
                                            nodeProg.progress.etaMs,
                                          )}</span
                                        >
                                      </div>
                                    </button>

                                    {#if isExpanded}
                                      <div class="mt-2 space-y-1.5">
                                        {#if nodeProg.progress.files.length === 0}
                                          <div
                                            class="text-[11px] font-mono text-exo-light-gray/70"
                                          >
                                            No file details reported.
                                          </div>
                                        {:else}
                                          {#each nodeProg.progress.files as f}
                                            {@const filePercent = Math.min(
                                              100,
                                              Math.max(0, f.percentage ?? 0),
                                            )}
                                            {@const isFileComplete =
                                              filePercent >= 100}
                                            <div
                                              class="rounded border border-exo-medium-gray/30 bg-exo-black/40 p-2"
                                            >
                                              <div
                                                class="flex items-center justify-between text-[10px] font-mono text-exo-light-gray/90"
                                              >
                                                <span class="truncate pr-2"
                                                  >{f.name}</span
                                                >
                                                <span
                                                  class={isFileComplete
                                                    ? "text-green-400"
                                                    : "text-white/80"}
                                                  >{filePercent.toFixed(
                                                    1,
                                                  )}%</span
                                                >
                                              </div>
                                              <div
                                                class="relative h-1 bg-exo-black/60 rounded-sm overflow-hidden mt-1"
                                              >
                                                <div
                                                  class="absolute inset-y-0 left-0 bg-gradient-to-r {isFileComplete
                                                    ? 'from-green-500 to-green-400'
                                                    : 'from-exo-yellow to-exo-yellow/70'} transition-all duration-300"
                                                  style="width: {filePercent.toFixed(
                                                    1,
                                                  )}%"
                                                ></div>
                                              </div>
                                              <div
                                                class="flex items-center justify-between text-[10px] text-exo-light-gray/70 mt-0.5"
                                              >
                                                <span
                                                  >{formatBytes(
                                                    f.downloadedBytes,
                                                  )} / {formatBytes(
                                                    f.totalBytes,
                                                  )}</span
                                                >
                                                <span
                                                  >{formatSpeed(f.speed)} • ETA {formatEta(
                                                    f.etaMs,
                                                  )}</span
                                                >
                                              </div>
                                            </div>
                                          {/each}
                                        {/if}
                                      </div>
                                    {/if}
                                  </div>
                                {/each}
                              </div>
                            {/if}
                            <div
                              class="text-xs text-blue-400 font-mono tracking-wider mt-1"
                            >
                              DOWNLOADING
                            </div>
                          {:else}
                            <div
                              class="text-xs {getStatusColor(
                                downloadInfo.statusText,
                              )} font-mono tracking-wider mt-1"
                            >
                              {downloadInfo.statusText}
                            </div>
                          {/if}
                        </div>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </aside>
        {/if}
      </div>
    {/if}
  </main>
</div>

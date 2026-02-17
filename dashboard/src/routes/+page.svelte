<script lang="ts">
  import {
    TopologyGraph,
    ChatForm,
    ChatMessages,
    ChatSidebar,
    ModelCard,
    ModelPickerModal,
  } from "$lib/components";
  import {
    favorites,
    toggleFavorite,
    getFavoritesSet,
  } from "$lib/stores/favorites.svelte";
  import {
    hasRecents,
    getRecentModelIds,
    recordRecentLaunch,
  } from "$lib/stores/recents.svelte";
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
    togglePreviewNodeFilter,
    clearPreviewNodeFilter,
    previewNodeFilter,
    createConversation,
    setSelectedChatModel,
    selectedChatModel,
    sendMessage,
    debugMode,
    toggleDebugMode,
    topologyOnlyMode,
    toggleTopologyOnlyMode,
    chatSidebarVisible,
    toggleChatSidebarVisible,
    nodeThunderbolt,
    nodeRdmaCtl,
    metaInstances,
    thunderboltBridgeCycles,
    nodeThunderboltBridge,
    nodeIdentities,
    type DownloadProgress,
    type PlacementPreview,
    type MetaInstanceData,
  } from "$lib/stores/app.svelte";
  import { addToast } from "$lib/stores/toast.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import { fade, fly, slide } from "svelte/transition";
  import { tweened } from "svelte/motion";
  import { cubicInOut, cubicOut } from "svelte/easing";
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
  const metaInstancesData = $derived(metaInstances());
  const tbBridgeCycles = $derived(thunderboltBridgeCycles());

  // Get status for a MetaInstance that has no backing instance yet
  function getMetaInstancePlacingStatus(metaInstanceId: string) {
    const meta = metaInstancesData[metaInstanceId];
    const placementError = meta?.placementError;
    const failures = meta?.consecutiveFailures ?? 0;
    const lastError = meta?.lastFailureError;

    if (placementError) {
      return {
        statusText: "PLACEMENT FAILED",
        statusClass: "failed",
        isDownloading: false as const,
        isFailed: true,
        progress: null,
        perNode: [] as Array<{
          nodeId: string;
          nodeName: string;
          progress: DownloadProgress;
        }>,
        perNodeStatus: [] as PerNodeRunnerStatus[],
        errorMessage: placementError,
      };
    }

    if (failures > 0) {
      const retryPosition = ((failures - 1) % 3) + 1;
      const isRecreated = failures % 3 === 0;
      return {
        statusText: isRecreated ? "PLACING" : `RETRYING (${retryPosition}/3)`,
        statusClass: "starting",
        isDownloading: false as const,
        isFailed: false,
        progress: null,
        perNode: [] as Array<{
          nodeId: string;
          nodeName: string;
          progress: DownloadProgress;
        }>,
        perNodeStatus: [] as PerNodeRunnerStatus[],
        errorMessage: isRecreated
          ? `Instance re-created due to failure: ${lastError}`
          : `Previous failure: ${lastError}`,
      };
    }

    return {
      statusText: "PLACING",
      statusClass: "starting",
      isDownloading: false as const,
      isFailed: false,
      progress: null,
      perNode: [] as Array<{
        nodeId: string;
        nodeName: string;
        progress: DownloadProgress;
      }>,
      perNodeStatus: [] as PerNodeRunnerStatus[],
      errorMessage: null,
    };
  }

  const tbBridgeData = $derived(nodeThunderboltBridge());
  const identitiesData = $derived(nodeIdentities());
  const tbIdentifiers = $derived(nodeThunderbolt());
  const rdmaCtlData = $derived(nodeRdmaCtl());
  const nodeFilter = $derived(previewNodeFilter());

  // Aggregate active download progress across all instances for header indicator
  const activeDownloadSummary = $derived.by(() => {
    let totalBytes = 0;
    let downloadedBytes = 0;
    let count = 0;
    for (const [id, inst] of Object.entries(instanceData)) {
      const status = getInstanceDownloadStatus(id, inst);
      if (status.isDownloading && status.progress) {
        count++;
        totalBytes += status.progress.totalBytes || 0;
        downloadedBytes += status.progress.downloadedBytes || 0;
      }
    }
    if (count === 0) return null;
    return {
      count,
      percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
    };
  });

  // Detect macOS version mismatches across cluster nodes
  const macosVersionMismatch = $derived.by(() => {
    if (!identitiesData) return null;
    const entries = Object.entries(identitiesData);
    // Filter to macOS nodes (version starts with a digit, e.g. "15.3")
    const macosNodes = entries.filter(([_, id]) => {
      const v = id.osVersion;
      return v && v !== "Unknown" && /^\d/.test(v);
    });
    if (macosNodes.length < 2) return null;
    // Compare on buildVersion for precise mismatch detection
    const buildVersions = new Set(
      macosNodes.map(([_, id]) => id.osBuildVersion ?? id.osVersion),
    );
    if (buildVersions.size <= 1) return null;
    return macosNodes.map(([nodeId, id]) => ({
      nodeId,
      friendlyName: getNodeName(nodeId),
      version: id.osVersion!,
      buildVersion: id.osBuildVersion ?? "Unknown",
    }));
  });

  // Detect TB5 nodes where RDMA is not enabled
  const tb5WithoutRdma = $derived.by(() => {
    const rdmaCtl = rdmaCtlData;
    if (!rdmaCtl) return false;
    const ids = tbIdentifiers;
    if (!ids) return false;
    // Find nodes with TB5 hardware (any TB interface)
    const tb5NodeIds = Object.entries(ids)
      .filter(([_, node]) => node.interfaces.length > 0)
      .map(([id]) => id);
    if (tb5NodeIds.length < 2) return false;
    // At least one TB5 node has RDMA disabled
    return tb5NodeIds.some((id) => rdmaCtl[id]?.enabled !== true);
  });
  let tb5InfoDismissed = $state(false);

  // Detect [jaccl] RDMA driver errors from MetaInstance failure errors
  const jacclError = $derived.by(() => {
    for (const mi of Object.values(metaInstancesData)) {
      if (mi.lastFailureError?.includes("[jaccl]")) {
        return mi.lastFailureError;
      }
    }
    return null;
  });
  let jacclDismissedError = $state<string | null>(null);

  // Helper to get friendly node name from node ID
  function getNodeName(nodeId: string): string {
    const node = data?.nodes?.[nodeId];
    return node?.friendly_name || nodeId.slice(0, 8) + "...";
  }

  // Helper to get the thunderbolt bridge service name from a cycle
  function getTbBridgeServiceName(cycle: string[]): string {
    // Try to find service name from any node in the cycle
    for (const nodeId of cycle) {
      const nodeData = tbBridgeData?.[nodeId];
      if (nodeData?.serviceName) {
        return nodeData.serviceName;
      }
    }
    return "Thunderbolt Bridge"; // Fallback if no service name found
  }

  // Copy to clipboard state and function
  let copiedCommand = $state(false);
  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copiedCommand = true;
      setTimeout(() => {
        copiedCommand = false;
      }, 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }

  // Warning icon SVG path (reused across warning snippets)
  const warningIconPath =
    "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z";
  const infoIconPath =
    "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z";

  let mounted = $state(false);

  // ── Onboarding wizard state ──
  const ONBOARDING_COMPLETE_KEY = "exo-onboarding-complete";
  let onboardingStep = $state(0); // 0 = not in onboarding, 1-7 = wizard steps
  let onboardingModelId = $state<string | null>(null); // model selected during onboarding
  const showOnboarding = $derived(onboardingStep > 0);

  // ── Step 2 animation state: "Add more devices, run bigger models" ──
  let deviceAnimPhase = $state(0); // 0=waiting, 1=macbook, 2=studio joins, 3=connection+mid unlock, 4=big unlock
  let showContinueStep2 = $state(false);
  const studioX = tweened(540, { duration: 700, easing: cubicOut });
  const studioOpacity = tweened(0, { duration: 700, easing: cubicOut });

  $effect(() => {
    if (onboardingStep === 2) {
      deviceAnimPhase = 0;
      showContinueStep2 = false;
      studioX.set(540, { duration: 0 });
      studioOpacity.set(0, { duration: 0 });

      const t1 = setTimeout(() => {
        deviceAnimPhase = 1;
      }, 100);
      const t2 = setTimeout(() => {
        deviceAnimPhase = 2;
        studioX.set(340);
        studioOpacity.set(1);
      }, 900);
      const t3 = setTimeout(() => {
        deviceAnimPhase = 3;
      }, 1700);
      const t4 = setTimeout(() => {
        deviceAnimPhase = 4;
      }, 2500);
      const t5 = setTimeout(() => {
        showContinueStep2 = true;
      }, 3500);

      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
        clearTimeout(t4);
        clearTimeout(t5);
      };
    }
  });

  // Recommended models for onboarding (sorted by fit, then size desc, limited to 6)
  const onboardingModels = $derived.by(() => {
    if (models.length === 0) return [];
    return [...models]
      .filter((m) => getModelMemoryFitStatus(m) !== "too_large")
      .sort((a, b) => {
        const aFit = hasEnoughMemory(a) ? 0 : 1;
        const bFit = hasEnoughMemory(b) ? 0 : 1;
        if (aFit !== bFit) return aFit - bFit;
        return getModelSizeGB(b) - getModelSizeGB(a);
      })
      .slice(0, 6);
  });

  // Track onboarding instance status for auto-advancing steps.
  // Handles cached models: if no download is needed, skip step 5 entirely.
  $effect(() => {
    if (onboardingStep === 5 && instanceCount > 0) {
      let anyDownloading = false;
      let anyReady = false;
      for (const [id, inst] of Object.entries(instanceData)) {
        const status = getInstanceDownloadStatus(id, inst);
        if (status.isDownloading) {
          anyDownloading = true;
        }
        if (
          status.statusText === "READY" ||
          status.statusText === "LOADED" ||
          status.statusText === "RUNNING"
        ) {
          anyReady = true;
        }
      }
      // Model already cached & ready — skip download AND loading steps
      if (anyReady) {
        onboardingStep = 7;
      } else if (!anyDownloading) {
        // Download finished (or was never needed) but not ready yet
        onboardingStep = 6;
      }
    }
  });

  $effect(() => {
    if (onboardingStep === 6 && instanceCount > 0) {
      for (const [id, inst] of Object.entries(instanceData)) {
        const status = getInstanceDownloadStatus(id, inst);
        if (
          status.statusText === "READY" ||
          status.statusText === "LOADED" ||
          status.statusText === "RUNNING"
        ) {
          onboardingStep = 7;
          break;
        }
      }
    }
  });

  function completeOnboarding() {
    onboardingStep = 0;
    try {
      localStorage.setItem(ONBOARDING_COMPLETE_KEY, "true");
    } catch {
      // ignore
    }
  }

  let onboardingError = $state<string | null>(null);

  async function onboardingLaunchModel(modelId: string) {
    onboardingModelId = modelId;
    onboardingError = null;
    selectPreviewModel(modelId);
    onboardingStep = 5;
    // Launch via API
    try {
      const response = await fetch("/meta_instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelId,
          sharding: selectedSharding,
          instance_meta: selectedInstanceType,
          min_nodes: 1,
        }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        onboardingError = `Failed to launch: ${errorText}`;
        onboardingStep = 4;
        return;
      }
      setSelectedChatModel(modelId);
      recordRecentLaunch(modelId);
    } catch (error) {
      onboardingError = `Network error: ${error}`;
      onboardingStep = 4;
    }
  }

  // Helper to get onboarding download progress
  const onboardingDownloadProgress = $derived.by(() => {
    if (instanceCount === 0) return null;
    for (const [id, inst] of Object.entries(instanceData)) {
      const status = getInstanceDownloadStatus(id, inst);
      if (status.isDownloading && status.progress) {
        return status.progress;
      }
    }
    return null;
  });

  // Instance launch state
  let models = $state<
    Array<{
      id: string;
      name?: string;
      storage_size_megabytes?: number;
      tasks?: string[];
      hugging_face_id?: string;
      is_custom?: boolean;
      family?: string;
      quantization?: string;
      base_model?: string;
      capabilities?: string[];
    }>
  >([]);
  type ModelMemoryFitStatus =
    | "fits_now"
    | "fits_cluster_capacity"
    | "too_large";

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

  const modelCapabilities = $derived(() => {
    const caps: Record<string, string[]> = {};
    for (const model of models) {
      if (model.capabilities && model.capabilities.length > 0) {
        caps[model.id] = model.capabilities;
        if (model.hugging_face_id) {
          caps[model.hugging_face_id] = model.capabilities;
        }
      }
    }
    return caps;
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

  // Helper to check if a model supports image editing
  function modelSupportsImageEditing(modelId: string): boolean {
    const model = models.find(
      (m) => m.id === modelId || m.hugging_face_id === modelId,
    );
    if (!model?.tasks) return false;
    return model.tasks.includes("ImageToImage");
  }
  let selectedSharding = $state<"Pipeline" | "Tensor">("Pipeline");
  type InstanceMeta = "MlxRing" | "MlxJaccl";

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

  // Model picker modal state
  let isModelPickerOpen = $state(false);

  // Advanced options toggle (hides technical jargon for new users)
  let showAdvancedOptions = $state(false);

  // Favorites state (reactive)
  const favoritesSet = $derived(getFavoritesSet());

  // Recent models state (reactive)
  const recentModelIds = $derived(getRecentModelIds());
  const showRecentsTab = $derived(hasRecents());

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

  // Computed: Check if filter is active (from store)
  const isFilterActive = $derived(() => nodeFilter.size > 0);

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
      : runtime === "MlxJaccl" || runtime === "MlxJaccl";

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

  // Calculate total memory in the cluster (in GB)
  const clusterTotalMemoryGB = $derived(() => {
    if (!data) return 0;
    return (
      Object.values(data.nodes).reduce((acc, n) => {
        const total =
          n.macmon_info?.memory?.ram_total ?? n.system_info?.memory ?? 0;
        return acc + total;
      }, 0) /
      (1024 * 1024 * 1024)
    );
  });

  function getModelMemoryFitStatus(model: {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
  }): ModelMemoryFitStatus {
    const modelSizeGB = getModelSizeGB(model);
    if (modelSizeGB <= availableMemoryGB()) {
      return "fits_now";
    }
    if (modelSizeGB <= clusterTotalMemoryGB()) {
      return "fits_cluster_capacity";
    }
    return "too_large";
  }

  // Check if a model has enough memory to run
  function hasEnoughMemory(model: {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
  }): boolean {
    return getModelMemoryFitStatus(model) === "fits_now";
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

    // Handle reset-onboarding query parameter (triggered from native Settings)
    const params = new URLSearchParams(window.location.search);
    if (params.has("reset-onboarding")) {
      localStorage.removeItem(ONBOARDING_COMPLETE_KEY);
      window.history.replaceState({}, "", window.location.pathname);
      onboardingStep = 1;
      return;
    }

    // Show onboarding wizard for first-time users
    if (!localStorage.getItem(ONBOARDING_COMPLETE_KEY)) {
      onboardingStep = 1;
    }
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

  async function addModelFromPicker(modelId: string) {
    const response = await fetch("/models/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });

    if (!response.ok) {
      let message = `Failed to add model (${response.status}: ${response.statusText})`;
      try {
        const err = await response.json();
        if (err.detail) message = err.detail;
      } catch {
        // use default message
      }
      throw new Error(message);
    }

    await fetchModels();
  }

  async function deleteCustomModel(modelId: string) {
    try {
      const response = await fetch(
        `/models/custom/${encodeURIComponent(modelId)}`,
        { method: "DELETE" },
      );
      if (response.ok) {
        await fetchModels();
      }
    } catch {
      console.error("Failed to delete custom model");
    }
  }

  function handleModelPickerSelect(modelId: string) {
    selectPreviewModel(modelId);
    saveLaunchDefaults();
    isModelPickerOpen = false;
  }

  async function launchInstance(
    modelId: string,
    specificPreview?: PlacementPreview | null,
  ) {
    if (!modelId || launchingModelId) return;

    launchingModelId = modelId;

    try {
      const preview = specificPreview ?? filteredPreview();

      // Extract node IDs from the preview the user is seeing
      const previewNodeIds = preview?.memory_delta_by_node
        ? Object.keys(preview.memory_delta_by_node)
        : nodeFilter.size > 0
          ? Array.from(nodeFilter)
          : undefined;

      const response = await fetch("/meta_instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelId,
          sharding: preview?.sharding ?? selectedSharding,
          instance_meta: preview?.instance_meta ?? selectedInstanceType,
          min_nodes: selectedMinNodes,
          node_ids: previewNodeIds,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Failed to launch instance:", errorText);
        addToast({
          type: "error",
          message: `Failed to launch model: ${errorText}`,
        });
      } else {
        addToast({ type: "success", message: `Model launched successfully` });
        // Always auto-select the newly launched model so the user chats to what they just launched
        setSelectedChatModel(modelId);

        // Record the launch in recent models history
        recordRecentLaunch(modelId);

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
      addToast({
        type: "error",
        message: "Failed to launch model. Check console for details.",
      });
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
    isFailed: boolean;
    errorMessage: string | null;
    progress: DownloadProgress | null;
    statusText: string;
    perNode: Array<{
      nodeId: string;
      nodeName: string;
      progress: DownloadProgress;
    }>;
    perNodeStatus: PerNodeRunnerStatus[];
  } {
    if (!downloadsData || Object.keys(downloadsData).length === 0) {
      const statusInfo = deriveInstanceStatus(instanceWrapped);
      return {
        isDownloading: false,
        isFailed: statusInfo.statusText === "FAILED",
        errorMessage: statusInfo.errorMessage,
        progress: null,
        statusText: statusInfo.statusText,
        perNode: [],
        perNodeStatus: statusInfo.perNodeStatus,
      };
    }

    // Unwrap the instance
    const [instanceTag, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return {
        isDownloading: false,
        isFailed: false,
        errorMessage: null,
        progress: null,
        statusText: "PREPARING",
        perNode: [],
        perNodeStatus: [],
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

        // Handle DownloadFailed - return immediately with error info
        if (downloadKind === "DownloadFailed") {
          const downloadModelId = extractModelIdFromDownload(downloadPayload);
          if (
            instanceModelId &&
            downloadModelId &&
            downloadModelId === instanceModelId
          ) {
            return {
              isDownloading: false,
              isFailed: true,
              errorMessage:
                (downloadPayload.errorMessage as string) || "Download failed",
              progress: null,
              statusText: "FAILED",
              perNode: [],
              perNodeStatus: [],
            };
          }
        }

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
        isFailed: statusInfo.statusText === "FAILED",
        errorMessage: statusInfo.errorMessage,
        progress: null,
        statusText: statusInfo.statusText,
        perNode: [],
        perNodeStatus: statusInfo.perNodeStatus,
      };
    }

    // ETA = total remaining bytes / total speed across all nodes
    const remainingBytes = totalBytes - downloadedBytes;
    const etaMs = totalSpeed > 0 ? (remainingBytes / totalSpeed) * 1000 : 0;

    return {
      isDownloading: true,
      isFailed: false,
      errorMessage: null,
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
      perNodeStatus: [],
    };
  }

  // Derive instance status from runners
  // Get color class for a status
  function getStatusColor(statusText: string): string {
    if (statusText === "FAILED" || statusText === "PLACEMENT FAILED")
      return "text-red-400";
    if (statusText.startsWith("RETRYING")) return "text-orange-400";
    if (statusText === "SHUTDOWN") return "text-gray-400";
    if (statusText === "DOWNLOADING") return "text-blue-400";
    if (
      statusText.startsWith("LOADING") ||
      statusText.startsWith("WARMING UP") ||
      statusText === "WAITING" ||
      statusText === "INITIALIZING"
    )
      return "text-yellow-400";
    if (statusText === "RUNNING") return "text-teal-400";
    if (statusText === "READY" || statusText === "LOADED")
      return "text-green-400";
    return "text-exo-light-gray";
  }

  const RUNNER_STATUS_MAP: Record<string, string> = {
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

  // Friendly labels for display
  const RUNNER_STATUS_DISPLAY: Record<string, string> = {
    WaitingForInitialization: "Initializing",
    InitializingBackend: "Initializing",
    WaitingForModel: "Waiting",
    Loading: "Loading",
    Loaded: "Loaded",
    WarmingUp: "Warming Up",
    Ready: "Ready",
    Running: "Running",
    Shutdown: "Shutdown",
    Failed: "Failed",
  };

  interface PerNodeRunnerStatus {
    nodeId: string;
    nodeName: string;
    status: string; // friendly display status
  }

  function deriveInstanceStatus(instanceWrapped: unknown): {
    statusText: string;
    statusClass: string;
    perNodeStatus: PerNodeRunnerStatus[];
    errorMessage: string | null;
  } {
    const [, instance] = getTagged(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return {
        statusText: "PREPARING",
        statusClass: "inactive",
        perNodeStatus: [],
        errorMessage: null,
      };
    }

    const inst = instance as {
      shardAssignments?: {
        runnerToShard?: Record<string, unknown>;
        nodeToRunner?: Record<string, string>;
      };
    };
    const nodeToRunner = inst.shardAssignments?.nodeToRunner || {};
    const runnerIds = Object.keys(inst.shardAssignments?.runnerToShard || {});
    const totalNodes = runnerIds.length;

    // Build per-node status and extract error messages from RunnerFailed
    const perNodeStatus: PerNodeRunnerStatus[] = [];
    const statuses: string[] = [];
    const failedErrors: string[] = [];
    for (const [nodeId, runnerId] of Object.entries(nodeToRunner)) {
      const r = runnersData[runnerId];
      let status: string | null = null;
      if (r) {
        const [kind, runnerData] = getTagged(r);
        status = kind ? RUNNER_STATUS_MAP[kind] || null : null;
        // Extract error message from RunnerFailed
        if (
          kind === "RunnerFailed" &&
          runnerData &&
          typeof runnerData === "object"
        ) {
          const rd = runnerData as { errorMessage?: string };
          if (rd.errorMessage)
            failedErrors.push(`${getNodeName(nodeId)}: ${rd.errorMessage}`);
        }
      }
      if (status) {
        statuses.push(status);
        perNodeStatus.push({
          nodeId,
          nodeName: getNodeName(nodeId),
          status: RUNNER_STATUS_DISPLAY[status] || status,
        });
      }
    }

    const has = (s: string) => statuses.includes(s);
    const count = (s: string) => statuses.filter((v) => v === s).length;

    if (statuses.length === 0)
      return {
        statusText: "PREPARING",
        statusClass: "inactive",
        perNodeStatus,
        errorMessage: null,
      };
    if (has("Failed"))
      return {
        statusText: "FAILED",
        statusClass: "failed",
        perNodeStatus,
        errorMessage: failedErrors.length > 0 ? failedErrors.join("; ") : null,
      };
    if (has("Shutdown"))
      return {
        statusText: "SHUTDOWN",
        statusClass: "inactive",
        perNodeStatus,
        errorMessage: null,
      };

    // For loading/warming states, show node progress when multi-node
    if (has("Loading")) {
      const readyCount = count("Ready") + count("Running") + count("Loaded");
      const statusText =
        totalNodes > 1
          ? `LOADING (${readyCount}/${totalNodes} nodes ready)`
          : "LOADING";
      return {
        statusText,
        statusClass: "starting",
        perNodeStatus,
        errorMessage: null,
      };
    }
    if (has("WarmingUp")) {
      const readyCount = count("Ready") + count("Running");
      const statusText =
        totalNodes > 1
          ? `WARMING UP (${readyCount}/${totalNodes} nodes ready)`
          : "WARMING UP";
      return {
        statusText,
        statusClass: "starting",
        perNodeStatus,
        errorMessage: null,
      };
    }

    if (has("Running"))
      return {
        statusText: "RUNNING",
        statusClass: "running",
        perNodeStatus,
        errorMessage: null,
      };
    if (has("Ready"))
      return {
        statusText: "READY",
        statusClass: "loaded",
        perNodeStatus,
        errorMessage: null,
      };
    if (has("Loaded"))
      return {
        statusText: "LOADED",
        statusClass: "loaded",
        perNodeStatus,
        errorMessage: null,
      };
    if (has("WaitingForModel"))
      return {
        statusText: "WAITING",
        statusClass: "starting",
        perNodeStatus,
        errorMessage: null,
      };
    if (has("InitializingBackend"))
      return {
        statusText: "INITIALIZING",
        statusClass: "starting",
        perNodeStatus,
        errorMessage: null,
      };
    if (has("WaitingForInitialization"))
      return {
        statusText: "INITIALIZING",
        statusClass: "starting",
        perNodeStatus,
        errorMessage: null,
      };

    return {
      statusText: "RUNNING",
      statusClass: "active",
      perNodeStatus,
      errorMessage: null,
    };
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
        addToast({ type: "error", message: "Failed to delete instance" });
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

  async function deleteMetaInstance(metaInstanceId: string) {
    const meta = metaInstancesData[metaInstanceId];
    const modelId = meta?.modelId ?? "unknown";
    if (!confirm(`Delete model ${modelId}?`)) return;

    const wasSelected = selectedChatModel() === modelId;

    try {
      const response = await fetch(`/meta_instance/${metaInstanceId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        console.error("Failed to delete meta instance:", response.status);
      } else if (wasSelected) {
        // Switch to another available model or clear selection
        const remainingInstances = Object.entries(instanceData).filter(
          ([id]) => id !== getBackingInstanceId(metaInstanceId),
        );
        if (remainingInstances.length > 0) {
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
            setSelectedChatModel("");
          }
        } else {
          setSelectedChatModel("");
        }
      }
    } catch (error) {
      console.error("Error deleting meta instance:", error);
    }
  }

  // Find the backing Instance ID for a MetaInstance by scanning instances
  function getBackingInstanceId(metaInstanceId: string): string | null {
    for (const [id, inst] of Object.entries(instanceData)) {
      const [, inner] = getTagged(inst);
      if (
        inner &&
        typeof inner === "object" &&
        (inner as Record<string, unknown>).metaInstanceId === metaInstanceId
      ) {
        return id;
      }
    }
    return null;
  }

  // Get orphan Instance IDs (not backing any MetaInstance)
  function getOrphanInstanceIds(): string[] {
    return Object.keys(instanceData).filter((id) => {
      const [, inner] = getTagged(instanceData[id]);
      return (
        !inner ||
        typeof inner !== "object" ||
        !(inner as Record<string, unknown>).metaInstanceId
      );
    });
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
    else if (instanceTag === "MlxJacclInstance") instanceType = "MLX RDMA";

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
  const metaInstanceCount = $derived(Object.keys(metaInstancesData).length);
  const orphanInstanceIds = $derived(getOrphanInstanceIds());
  const instanceCount = $derived(metaInstanceCount + orphanInstanceIds.length);

  // Unified display items: MetaInstances first, then orphan Instances
  interface DisplayItem {
    id: string; // MetaInstance ID or Instance ID (used as key and displayed)
    modelId: string;
    instance: unknown | null; // The backing/orphan instance (tagged union) or null if placing
    instanceId: string | null; // The actual Instance ID (for topology hover)
    isMetaInstance: boolean;
    sharding: string | null; // From MetaInstance constraints (used when instance is null)
    instanceMeta: string | null; // From MetaInstance constraints (used when instance is null)
  }

  const unifiedDisplayItems = $derived.by((): DisplayItem[] => {
    const items: DisplayItem[] = [];
    // MetaInstances
    for (const [metaId, meta] of Object.entries(metaInstancesData)) {
      const backingId = getBackingInstanceId(metaId);
      items.push({
        id: metaId,
        modelId: meta.modelId,
        instance: backingId ? instanceData[backingId] : null,
        instanceId: backingId,
        isMetaInstance: true,
        sharding: meta.sharding,
        instanceMeta: meta.instanceMeta,
      });
    }
    // Orphan Instances
    for (const orphanId of getOrphanInstanceIds()) {
      const inst = instanceData[orphanId];
      items.push({
        id: orphanId,
        modelId: getInstanceModelId(inst),
        instance: inst,
        instanceId: orphanId,
        isMetaInstance: false,
        sharding: null,
        instanceMeta: null,
      });
    }
    return items;
  });

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
  // Backend handles node_ids filtering, we filter by sharding/instance type and min nodes
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

{#snippet clusterWarnings()}
  {#if tbBridgeCycles.length > 0 || macosVersionMismatch || (tb5WithoutRdma && !tb5InfoDismissed) || (jacclError && jacclError !== jacclDismissedError)}
    <div class="absolute top-4 left-4 flex flex-col gap-2 z-40">
      {#if jacclError && jacclError !== jacclDismissedError}
        <div class="group relative" role="alert">
          <div
            class="flex items-center gap-2 px-3 py-2 rounded border border-red-500/50 bg-red-500/10 backdrop-blur-sm cursor-help"
          >
            <svg
              class="w-5 h-5 text-red-400 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d={warningIconPath}
              />
            </svg>
            <span class="text-sm font-mono text-red-200">
              JACCL RDMA ERROR
            </span>
            <button
              type="button"
              onclick={() => (jacclDismissedError = jacclError)}
              class="ml-1 text-red-300/60 hover:text-red-200 transition-colors cursor-pointer"
              title="Dismiss"
            >
              <svg
                class="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <!-- Tooltip on hover -->
          <div
            class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-red-500/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
          >
            <p class="text-xs text-white/80 mb-2">
              A macOS RDMA driver error was detected. This is a known issue with
              the experimental RDMA driver in macOS.
            </p>
            <p class="text-xs text-white/60 mb-2">
              <span class="text-red-300">Error:</span>
              {jacclError}
            </p>
            <p class="text-xs text-white/60">
              <span class="text-red-300">To fix:</span> Restart the affected machine.
              There is currently no other workaround for this issue.
            </p>
          </div>
        </div>
      {/if}

      {#if tbBridgeCycles.length > 0}
        {@const cycle = tbBridgeCycles[0]}
        {@const serviceName = getTbBridgeServiceName(cycle)}
        {@const disableCmd = `sudo networksetup -setnetworkserviceenabled "${serviceName}" off`}
        <div class="group relative" role="alert">
          <div
            class="flex items-center gap-2 px-3 py-2 rounded border border-yellow-500/50 bg-yellow-500/10 backdrop-blur-sm cursor-help"
          >
            <svg
              class="w-5 h-5 text-yellow-400 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d={warningIconPath}
              />
            </svg>
            <span class="text-sm font-mono text-yellow-200">
              THUNDERBOLT BRIDGE CYCLE DETECTED
            </span>
          </div>

          <!-- Tooltip on hover -->
          <div
            class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-yellow-500/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
          >
            <p class="text-xs text-white/80 mb-2">
              A network routing cycle was detected between nodes connected via
              Thunderbolt Bridge. This can cause connectivity issues.
            </p>
            <p class="text-xs text-white/60 mb-2">
              <span class="text-yellow-300">Affected nodes:</span>
              {cycle.map(getNodeName).join(" → ")}
            </p>
            <p class="text-xs text-white/60 mb-1">
              <span class="text-yellow-300">To fix:</span> Disable the Thunderbolt
              Bridge on one of the affected nodes:
            </p>
            <button
              type="button"
              onclick={() => copyToClipboard(disableCmd)}
              class="w-full flex items-center gap-2 text-[10px] font-mono bg-exo-black/60 px-2 py-1.5 rounded text-exo-yellow break-all text-left hover:bg-exo-black/80 transition-colors cursor-pointer group/copy"
              title="Click to copy"
            >
              <span class="flex-1">{disableCmd}</span>
              <svg
                class="w-3.5 h-3.5 flex-shrink-0 text-white/40 group-hover/copy:text-exo-yellow transition-colors"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2"
              >
                {#if copiedCommand}
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    d="M5 13l4 4L19 7"
                  />
                {:else}
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                {/if}
              </svg>
            </button>
          </div>
        </div>
      {/if}

      {#if macosVersionMismatch}
        <div class="group relative" role="alert">
          <div
            class="flex items-center gap-2 px-3 py-2 rounded border border-yellow-500/50 bg-yellow-500/10 backdrop-blur-sm cursor-help"
          >
            <svg
              class="w-5 h-5 text-yellow-400 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d={warningIconPath}
              />
            </svg>
            <span class="text-sm font-mono text-yellow-200">
              INCOMPATIBLE macOS VERSIONS
            </span>
          </div>

          <!-- Tooltip on hover -->
          <div
            class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-yellow-500/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
          >
            <p class="text-xs text-white/80 mb-2">
              Nodes in this cluster are running different macOS versions. This
              may cause inference compatibility issues.
            </p>
            <div class="text-xs text-white/60 mb-2">
              <span class="text-yellow-300">Node versions:</span>
              {#each macosVersionMismatch as node}
                <div class="ml-2">
                  {node.friendlyName} — macOS {node.version} ({node.buildVersion})
                </div>
              {/each}
            </div>
            <p class="text-xs text-white/60">
              <span class="text-yellow-300">Suggested action:</span> Update all nodes
              to the same macOS version for best compatibility.
            </p>
          </div>
        </div>
      {/if}

      {#if tb5WithoutRdma && !tb5InfoDismissed}
        <div
          class="flex items-center gap-2 px-3 py-2 rounded border border-blue-400/50 bg-blue-400/10 backdrop-blur-sm"
          role="status"
        >
          <svg
            class="w-5 h-5 text-blue-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d={infoIconPath}
            />
          </svg>
          <span class="text-sm font-mono text-blue-200"> RDMA AVAILABLE </span>
          <button
            type="button"
            onclick={() => (tb5InfoDismissed = true)}
            class="ml-1 text-blue-300/60 hover:text-blue-200 transition-colors cursor-pointer"
            title="Dismiss"
          >
            <svg
              class="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      {/if}
    </div>
  {/if}
{/snippet}

{#snippet clusterWarningsCompact()}
  {#if tbBridgeCycles.length > 0 || macosVersionMismatch || (tb5WithoutRdma && !tb5InfoDismissed) || (jacclError && jacclError !== jacclDismissedError)}
    <div class="absolute top-2 left-2 flex flex-col gap-1">
      {#if jacclError && jacclError !== jacclDismissedError}
        <div
          class="flex items-center gap-1.5 px-2 py-1 rounded border border-red-500/50 bg-red-500/10 backdrop-blur-sm"
          title="JACCL RDMA driver error — restart affected machine"
        >
          <svg
            class="w-3.5 h-3.5 text-red-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d={warningIconPath}
            />
          </svg>
          <span class="text-[10px] font-mono text-red-200">JACCL ERROR</span>
        </div>
      {/if}
      {#if tbBridgeCycles.length > 0}
        <div
          class="flex items-center gap-1.5 px-2 py-1 rounded border border-yellow-500/50 bg-yellow-500/10 backdrop-blur-sm"
          title="Thunderbolt Bridge cycle detected"
        >
          <svg
            class="w-3.5 h-3.5 text-yellow-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d={warningIconPath}
            />
          </svg>
          <span class="text-[10px] font-mono text-yellow-200">TB CYCLE</span>
        </div>
      {/if}
      {#if macosVersionMismatch}
        <div
          class="flex items-center gap-1.5 px-2 py-1 rounded border border-yellow-500/50 bg-yellow-500/10 backdrop-blur-sm"
          title="Incompatible macOS versions detected"
        >
          <svg
            class="w-3.5 h-3.5 text-yellow-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d={warningIconPath}
            />
          </svg>
          <span class="text-[10px] font-mono text-yellow-200"
            >macOS MISMATCH</span
          >
        </div>
      {/if}
      {#if tb5WithoutRdma && !tb5InfoDismissed}
        <div
          class="flex items-center gap-1.5 px-2 py-1 rounded border border-blue-400/50 bg-blue-400/10 backdrop-blur-sm"
          title="Thunderbolt 5 detected — RDMA can be enabled for better performance"
        >
          <svg
            class="w-3.5 h-3.5 text-blue-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d={infoIconPath}
            />
          </svg>
          <span class="text-[10px] font-mono text-blue-200">RDMA AVAILABLE</span
          >
        </div>
      {/if}
    </div>
  {/if}
{/snippet}

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
  {#if !showOnboarding}
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
  {/if}

  {#if showOnboarding}
    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- FULL-SCREEN ONBOARDING WIZARD                          -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <div class="flex-1 flex items-center justify-center relative z-10 bg-white">
      {#if onboardingStep === 1}
        <!-- Step 1: Welcome -->
        <div
          class="text-center max-w-lg px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <div class="mb-8">
            <div
              class="text-5xl font-mono font-bold text-exo-yellow tracking-wider mb-4"
            >
              exo
            </div>
            <h1
              class="text-2xl font-sans font-light text-gray-900 mb-3 tracking-wide"
            >
              Welcome to exo
            </h1>
            <p class="text-base font-sans text-gray-500 leading-relaxed">
              Run AI models locally, across all your devices.
            </p>
          </div>
          <button
            type="button"
            onclick={() => (onboardingStep = 2)}
            class="inline-flex items-center gap-2 px-8 py-3 bg-exo-yellow text-exo-black font-sans text-sm font-semibold rounded-full hover:brightness-110 hover:shadow-[0_0_24px_rgba(255,215,0,0.2)] transition-all duration-200 cursor-pointer"
          >
            Get Started
            <svg
              class="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2.5"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M13 7l5 5m0 0l-5 5m5-5H6"
              />
            </svg>
          </button>
        </div>
      {:else if onboardingStep === 2}
        <!-- Step 2: Add more devices, run bigger models -->
        <div
          class="flex flex-col items-center w-full max-w-2xl px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <div class="text-center mb-4">
            <h1
              class="text-2xl font-sans font-light text-gray-900 mb-2 tracking-wide"
            >
              Add more devices, run bigger models
            </h1>
          </div>

          <!-- Animation stage -->
          <div class="relative w-full" style="height: 380px;">
            <svg
              viewBox="0 0 600 380"
              class="w-full h-full"
              xmlns="http://www.w3.org/2000/svg"
            >
              <defs>
                <filter
                  id="onb-gold-glow"
                  x="-50%"
                  y="-50%"
                  width="200%"
                  height="200%"
                >
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feFlood
                    flood-color="#FFD700"
                    flood-opacity="0.6"
                    result="color"
                  />
                  <feComposite
                    in="color"
                    in2="blur"
                    operator="in"
                    result="glow"
                  />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter
                  id="onb-device-glow"
                  x="-50%"
                  y="-50%"
                  width="200%"
                  height="200%"
                >
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feFlood
                    flood-color="#FFD700"
                    flood-opacity="0.3"
                    result="color"
                  />
                  <feComposite
                    in="color"
                    in2="blur"
                    operator="in"
                    result="glow"
                  />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              <!-- MacBook Device (left side) -->
              {#if deviceAnimPhase >= 1}
                <g transform="translate(135, 30)" in:fade={{ duration: 600 }}>
                  <!-- Screen bezel -->
                  <rect
                    x="0"
                    y="0"
                    width="84"
                    height="56"
                    rx="4"
                    fill="none"
                    stroke="#FFD700"
                    stroke-width="1.5"
                    filter="url(#onb-device-glow)"
                  />
                  <!-- Screen interior -->
                  <rect
                    x="4"
                    y="3"
                    width="76"
                    height="48"
                    rx="2"
                    fill="#0a0a0a"
                  />
                  <!-- Memory bar inside screen -->
                  <rect
                    x="10"
                    y="36"
                    width="30"
                    height="8"
                    rx="2"
                    fill="#374151"
                  />
                  <rect
                    x="10"
                    y="36"
                    width="18"
                    height="8"
                    rx="2"
                    fill="rgba(255,215,0,0.4)"
                  />
                  <!-- Keyboard base -->
                  <path
                    d="M -6 60 L 90 60 L 84 70 L 0 70 Z"
                    fill="none"
                    stroke="#FFD700"
                    stroke-width="1.5"
                  />
                  <!-- Label -->
                  <text
                    x="42"
                    y="92"
                    text-anchor="middle"
                    fill="rgba(0,0,0,0.7)"
                    style="font-size: 13px; font-family: system-ui, sans-serif;"
                  >
                    MacBook Pro
                  </text>
                  <text
                    x="42"
                    y="108"
                    text-anchor="middle"
                    fill="rgba(0,0,0,0.35)"
                    style="font-size: 11px; font-family: 'SF Mono', monospace;"
                  >
                    36 GB
                  </text>
                </g>
              {/if}

              <!-- Mac Studio Device (right side) - tweened fly-in -->
              <g transform="translate({$studioX}, 30)" opacity={$studioOpacity}>
                <!-- Studio box -->
                <rect
                  x="0"
                  y="0"
                  width="78"
                  height="62"
                  rx="6"
                  fill="none"
                  stroke="#FFD700"
                  stroke-width="1.5"
                  filter="url(#onb-device-glow)"
                />
                <!-- Inner face -->
                <rect
                  x="5"
                  y="5"
                  width="68"
                  height="52"
                  rx="4"
                  fill="#0a0a0a"
                />
                <!-- Memory bar inside -->
                <rect
                  x="12"
                  y="40"
                  width="54"
                  height="8"
                  rx="2"
                  fill="#374151"
                />
                <rect
                  x="12"
                  y="40"
                  width="42"
                  height="8"
                  rx="2"
                  fill="rgba(255,215,0,0.4)"
                />
                <!-- Front circle detail -->
                <circle
                  cx="39"
                  cy="56"
                  r="3.5"
                  fill="none"
                  stroke="rgba(255,215,0,0.4)"
                  stroke-width="1"
                />
                <!-- Label -->
                <text
                  x="39"
                  y="86"
                  text-anchor="middle"
                  fill="rgba(0,0,0,0.7)"
                  style="font-size: 13px; font-family: system-ui, sans-serif;"
                >
                  Mac Studio
                </text>
                <text
                  x="39"
                  y="102"
                  text-anchor="middle"
                  fill="rgba(0,0,0,0.35)"
                  style="font-size: 11px; font-family: 'SF Mono', monospace;"
                >
                  192 GB
                </text>
              </g>

              <!-- Connection line between devices -->
              {#if deviceAnimPhase >= 3}
                <line
                  x1="223"
                  y1="62"
                  x2="340"
                  y2="62"
                  class="onboarding-connection-line"
                  in:fade={{ duration: 400 }}
                />
              {/if}

              <!-- Combined memory label -->
              {#if deviceAnimPhase >= 3}
                <text
                  x="282"
                  y="50"
                  text-anchor="middle"
                  fill="rgba(255,215,0,0.7)"
                  style="font-size: 11px; font-family: 'SF Mono', monospace;"
                  in:fade={{ duration: 400 }}
                >
                  228 GB combined
                </text>
              {/if}

              <!-- Model cards row -->
              <g transform="translate(0, 200)">
                <!-- Arrows from devices to models -->
                {#if deviceAnimPhase >= 3}
                  <line
                    x1="177"
                    y1="-50"
                    x2="177"
                    y2="-8"
                    stroke="rgba(255,215,0,0.15)"
                    stroke-width="1"
                    stroke-dasharray="4,4"
                    in:fade={{ duration: 300 }}
                  />
                  <line
                    x1="379"
                    y1="-50"
                    x2="379"
                    y2="-8"
                    stroke="rgba(255,215,0,0.15)"
                    stroke-width="1"
                    stroke-dasharray="4,4"
                    in:fade={{ duration: 300 }}
                  />
                {/if}

                <!-- Small model: 8B (visible from phase 1) -->
                {#if deviceAnimPhase >= 1}
                  <g transform="translate(55, 0)" in:fade={{ duration: 500 }}>
                    <rect
                      x="0"
                      y="0"
                      width="110"
                      height="48"
                      rx="8"
                      fill="rgba(0,0,0,0.04)"
                      stroke="rgba(0,0,0,0.1)"
                      stroke-width="1"
                    />
                    <text
                      x="55"
                      y="20"
                      text-anchor="middle"
                      fill="rgba(0,0,0,0.7)"
                      style="font-size: 13px; font-family: system-ui, sans-serif;"
                    >
                      Qwen3 8B
                    </text>
                    <text
                      x="55"
                      y="37"
                      text-anchor="middle"
                      fill="rgba(0,0,0,0.35)"
                      style="font-size: 11px; font-family: 'SF Mono', monospace;"
                    >
                      4 GB
                    </text>
                  </g>
                {/if}

                <!-- Medium model: 30B (locked → unlocks at phase 3) -->
                {#if deviceAnimPhase >= 1}
                  <g
                    transform="translate(185, 0)"
                    in:fade={{ duration: 500, delay: 200 }}
                  >
                    <rect
                      x="0"
                      y="0"
                      width="110"
                      height="48"
                      rx="8"
                      fill={deviceAnimPhase >= 3
                        ? "rgba(255,215,0,0.06)"
                        : "rgba(0,0,0,0.03)"}
                      stroke={deviceAnimPhase >= 3
                        ? "rgba(255,215,0,0.35)"
                        : "rgba(0,0,0,0.08)"}
                      stroke-width="1"
                      filter={deviceAnimPhase >= 3
                        ? "url(#onb-gold-glow)"
                        : "none"}
                      style="transition: fill 500ms, stroke 500ms, filter 500ms;"
                    />
                    {#if deviceAnimPhase < 3}
                      <g transform="translate(48, 12)">
                        <rect
                          x="0"
                          y="9"
                          width="14"
                          height="11"
                          rx="2"
                          fill="none"
                          stroke="rgba(0,0,0,0.2)"
                          stroke-width="1"
                        />
                        <path
                          d="M 2 9 V 6 C 2 3 4.5 1 7 1 C 9.5 1 12 3 12 6 V 9"
                          fill="none"
                          stroke="rgba(0,0,0,0.2)"
                          stroke-width="1"
                        />
                      </g>
                    {:else}
                      <text
                        x="55"
                        y="20"
                        text-anchor="middle"
                        fill="#FFD700"
                        style="font-size: 13px; font-family: system-ui, sans-serif;"
                      >
                        Qwen3 30B
                      </text>
                      <text
                        x="55"
                        y="37"
                        text-anchor="middle"
                        fill="rgba(255,215,0,0.6)"
                        style="font-size: 11px; font-family: 'SF Mono', monospace;"
                      >
                        16 GB
                      </text>
                    {/if}
                  </g>
                {/if}

                <!-- Large model: 72B (locked → unlocks at phase 4) -->
                {#if deviceAnimPhase >= 1}
                  <g
                    transform="translate(315, 0)"
                    in:fade={{ duration: 500, delay: 400 }}
                  >
                    <rect
                      x="0"
                      y="0"
                      width="110"
                      height="48"
                      rx="8"
                      fill={deviceAnimPhase >= 4
                        ? "rgba(255,215,0,0.06)"
                        : "rgba(0,0,0,0.03)"}
                      stroke={deviceAnimPhase >= 4
                        ? "rgba(255,215,0,0.35)"
                        : "rgba(0,0,0,0.08)"}
                      stroke-width="1"
                      filter={deviceAnimPhase >= 4
                        ? "url(#onb-gold-glow)"
                        : "none"}
                      style="transition: fill 500ms, stroke 500ms, filter 500ms;"
                    />
                    {#if deviceAnimPhase < 4}
                      <g transform="translate(48, 12)">
                        <rect
                          x="0"
                          y="9"
                          width="14"
                          height="11"
                          rx="2"
                          fill="none"
                          stroke="rgba(0,0,0,0.2)"
                          stroke-width="1"
                        />
                        <path
                          d="M 2 9 V 6 C 2 3 4.5 1 7 1 C 9.5 1 12 3 12 6 V 9"
                          fill="none"
                          stroke="rgba(0,0,0,0.2)"
                          stroke-width="1"
                        />
                      </g>
                    {:else}
                      <text
                        x="55"
                        y="20"
                        text-anchor="middle"
                        fill="#FFD700"
                        style="font-size: 13px; font-family: system-ui, sans-serif;"
                      >
                        Llama 72B
                      </text>
                      <text
                        x="55"
                        y="37"
                        text-anchor="middle"
                        fill="rgba(255,215,0,0.6)"
                        style="font-size: 11px; font-family: 'SF Mono', monospace;"
                      >
                        36 GB
                      </text>
                    {/if}
                  </g>
                {/if}

                <!-- Extra large model: 405B (locked → unlocks at phase 4 with extra glow) -->
                {#if deviceAnimPhase >= 1}
                  <g
                    transform="translate(445, 0)"
                    in:fade={{ duration: 500, delay: 600 }}
                  >
                    <rect
                      x="0"
                      y="0"
                      width="110"
                      height="48"
                      rx="8"
                      fill={deviceAnimPhase >= 4
                        ? "rgba(255,215,0,0.08)"
                        : "rgba(0,0,0,0.03)"}
                      stroke={deviceAnimPhase >= 4
                        ? "rgba(255,215,0,0.45)"
                        : "rgba(0,0,0,0.08)"}
                      stroke-width={deviceAnimPhase >= 4 ? "1.5" : "1"}
                      filter={deviceAnimPhase >= 4
                        ? "url(#onb-gold-glow)"
                        : "none"}
                      style="transition: fill 700ms, stroke 700ms, filter 700ms, stroke-width 700ms;"
                    />
                    {#if deviceAnimPhase < 4}
                      <g transform="translate(48, 12)">
                        <rect
                          x="0"
                          y="9"
                          width="14"
                          height="11"
                          rx="2"
                          fill="none"
                          stroke="rgba(0,0,0,0.2)"
                          stroke-width="1"
                        />
                        <path
                          d="M 2 9 V 6 C 2 3 4.5 1 7 1 C 9.5 1 12 3 12 6 V 9"
                          fill="none"
                          stroke="rgba(0,0,0,0.2)"
                          stroke-width="1"
                        />
                      </g>
                    {:else}
                      <text
                        x="55"
                        y="20"
                        text-anchor="middle"
                        fill="#FFD700"
                        style="font-size: 13px; font-family: system-ui, sans-serif; font-weight: 600;"
                      >
                        Llama 405B
                      </text>
                      <text
                        x="55"
                        y="37"
                        text-anchor="middle"
                        fill="rgba(255,215,0,0.6)"
                        style="font-size: 11px; font-family: 'SF Mono', monospace;"
                      >
                        203 GB
                      </text>
                    {/if}
                  </g>
                {/if}

                <!-- "Models you can run" label -->
                {#if deviceAnimPhase >= 1}
                  <text
                    x="300"
                    y="72"
                    text-anchor="middle"
                    fill="rgba(0,0,0,0.3)"
                    style="font-size: 11px; font-family: system-ui, sans-serif; text-transform: uppercase; letter-spacing: 0.12em;"
                    in:fade={{ duration: 400 }}
                  >
                    Models you can run
                  </text>
                {/if}
              </g>
            </svg>
          </div>

          <!-- Continue button -->
          {#if showContinueStep2}
            <button
              type="button"
              onclick={() => (onboardingStep = 3)}
              class="inline-flex items-center gap-2 px-8 py-3 mt-2 bg-exo-yellow text-exo-black font-sans text-sm font-semibold rounded-full hover:brightness-110 hover:shadow-[0_0_24px_rgba(255,215,0,0.2)] transition-all duration-200 cursor-pointer"
              in:fade={{ duration: 400 }}
            >
              Continue
              <svg
                class="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2.5"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              </svg>
            </button>
          {/if}
        </div>
      {:else if onboardingStep === 3}
        <!-- Step 3: Your Devices -->
        <div
          class="flex flex-col items-center w-full max-w-4xl px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <div class="text-center mb-6">
            <h1
              class="text-2xl font-sans font-light text-gray-900 mb-2 tracking-wide"
            >
              Your devices
            </h1>
            <p class="text-sm font-sans text-gray-500">
              {nodeCount} device{nodeCount !== 1 ? "s" : ""} connected
              {#if clusterTotalMemoryGB() > 0}
                · {clusterTotalMemoryGB().toFixed(0)} GB total memory
              {/if}
            </p>
          </div>
          <div
            class="w-full h-80 bg-exo-dark-gray rounded-lg overflow-hidden border border-gray-200 mb-8"
          >
            <TopologyGraph
              class="w-full h-full"
              highlightedNodes={highlightedNodes()}
              filteredNodes={nodeFilter}
              onNodeClick={togglePreviewNodeFilter}
            />
          </div>
          <p
            class="text-sm font-sans text-gray-400 mt-2 max-w-md text-center leading-relaxed"
          >
            Install exo on more devices on your network to combine their power —
            they connect automatically.
          </p>
          <button
            type="button"
            onclick={() => (onboardingStep = 4)}
            class="inline-flex items-center gap-2 px-8 py-3 mt-6 bg-exo-yellow text-exo-black font-sans text-sm font-semibold rounded-full hover:brightness-110 hover:shadow-[0_0_24px_rgba(255,215,0,0.2)] transition-all duration-200 cursor-pointer"
          >
            Continue
            <svg
              class="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2.5"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M13 7l5 5m0 0l-5 5m5-5H6"
              />
            </svg>
          </button>
        </div>
      {:else if onboardingStep === 4}
        <!-- Step 4: Choose a Model -->
        <div
          class="flex flex-col items-center w-full max-w-2xl px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <div class="text-center mb-8">
            <h1
              class="text-2xl font-sans font-light text-gray-900 mb-2 tracking-wide"
            >
              Choose a model
            </h1>
            <p class="text-sm font-sans text-gray-500">
              Pick a model to download and run locally.
            </p>
          </div>

          {#if onboardingError}
            <div
              class="w-full mb-6 px-4 py-3 rounded-lg border border-red-200 bg-red-50 text-sm font-mono text-red-600"
              in:fade={{ duration: 200 }}
            >
              {onboardingError}
            </div>
          {/if}

          {#if onboardingModels.length === 0}
            <div class="text-center py-8">
              <div class="text-sm text-gray-400 font-sans animate-pulse">
                Loading models...
              </div>
            </div>
          {:else}
            <div class="w-full space-y-3 mb-8">
              {#each onboardingModels as model}
                {@const sizeGB = getModelSizeGB(model)}
                {@const fitsNow = hasEnoughMemory(model)}
                {@const tags = modelTags()[model.id] || []}
                <button
                  type="button"
                  onclick={() => onboardingLaunchModel(model.id)}
                  class="w-full flex items-center justify-between gap-4 px-5 py-4 rounded-xl border transition-all duration-200 cursor-pointer {fitsNow
                    ? 'border-gray-200 bg-gray-50 hover:border-exo-yellow/50 hover:bg-yellow-50/50'
                    : 'border-gray-200 bg-gray-50/50 hover:border-gray-300 opacity-60'}"
                >
                  <div class="flex flex-col items-start gap-1 min-w-0">
                    <div class="flex items-center gap-2">
                      <span
                        class="text-sm font-sans font-medium text-gray-900 truncate"
                        >{model.name || model.id}</span
                      >
                      {#each tags as tag}
                        <span
                          class="text-[10px] font-sans font-medium px-1.5 py-0.5 rounded-full bg-exo-yellow/10 text-exo-yellow/80"
                          >{tag}</span
                        >
                      {/each}
                    </div>
                    <span class="text-xs font-mono text-gray-400 truncate"
                      >{model.id}</span
                    >
                  </div>
                  <div class="flex items-center gap-3 flex-shrink-0">
                    <span class="text-xs font-mono text-gray-500"
                      >{sizeGB >= 1 ? sizeGB.toFixed(0) : sizeGB.toFixed(1)} GB</span
                    >
                    <svg
                      class="w-4 h-4 text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      stroke-width="2"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </div>
                </button>
              {/each}
            </div>
          {/if}

          <button
            type="button"
            onclick={() => {
              isModelPickerOpen = true;
            }}
            class="text-sm font-sans text-gray-400 hover:text-exo-yellow transition-colors cursor-pointer underline underline-offset-4 decoration-gray-300 hover:decoration-exo-yellow/30"
          >
            Browse all models
          </button>
        </div>
      {:else if onboardingStep === 5}
        <!-- Step 5: Downloading -->
        <div
          class="text-center max-w-lg px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <div class="mb-8">
            <h1
              class="text-2xl font-sans font-light text-gray-900 mb-2 tracking-wide"
            >
              Downloading
            </h1>
            <p class="text-sm text-gray-500">
              {#if onboardingModelId}
                <span class="text-exo-yellow">{onboardingModelId}</span>
              {/if}
            </p>
          </div>

          {#if onboardingDownloadProgress}
            <div class="w-full max-w-md mx-auto space-y-4">
              <div
                class="relative h-2 bg-gray-200 rounded-full overflow-hidden"
              >
                <div
                  class="absolute inset-y-0 left-0 bg-gradient-to-r from-exo-yellow to-amber-400 rounded-full transition-all duration-500"
                  style="width: {onboardingDownloadProgress.percentage}%"
                ></div>
              </div>
              <div class="flex justify-between text-xs font-mono text-gray-500">
                <span>{onboardingDownloadProgress.percentage.toFixed(1)}%</span>
                <span
                  >{formatBytes(onboardingDownloadProgress.downloadedBytes)} /
                  {formatBytes(onboardingDownloadProgress.totalBytes)}</span
                >
              </div>
              <div class="flex justify-between text-xs font-mono text-gray-400">
                <span>{formatSpeed(onboardingDownloadProgress.speed)}</span>
                <span>ETA: {formatEta(onboardingDownloadProgress.etaMs)}</span>
              </div>
            </div>
          {:else}
            <div class="w-full max-w-md mx-auto">
              <div
                class="relative h-2 bg-gray-200 rounded-full overflow-hidden"
              >
                <div
                  class="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-exo-yellow to-amber-400 rounded-full animate-pulse"
                ></div>
              </div>
              <p class="text-xs font-mono text-gray-400 mt-4">
                Preparing download...
              </p>
            </div>
          {/if}

          <p class="text-xs font-sans text-gray-300 mt-8">
            This may take a few minutes depending on your connection.
          </p>
        </div>
      {:else if onboardingStep === 6}
        <!-- Step 6: Loading into memory -->
        <div
          class="text-center max-w-lg px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <div class="mb-8">
            <h1
              class="text-2xl font-sans font-light text-gray-900 mb-2 tracking-wide"
            >
              Loading into memory
            </h1>
            <p class="text-sm text-gray-500">
              {#if onboardingModelId}
                <span class="text-exo-yellow">{onboardingModelId}</span>
              {/if}
            </p>
          </div>

          <div class="flex justify-center mb-6">
            <div
              class="w-12 h-12 border-2 border-exo-yellow/30 border-t-exo-yellow rounded-full animate-spin"
            ></div>
          </div>

          <p class="text-sm text-gray-400 font-sans">Almost ready...</p>
        </div>
      {:else if onboardingStep === 7}
        <!-- Step 7: Chat — centered input auto-appears, first message transitions to dashboard -->
        <div
          class="flex flex-col items-center justify-center w-full max-w-2xl px-8"
          in:fade={{ duration: 400 }}
          out:fade={{ duration: 200 }}
        >
          <!-- Subtle branding -->
          <div
            class="text-2xl font-mono text-gray-200 font-bold tracking-wider mb-8"
          >
            exo
          </div>

          <!-- Model name -->
          {#if onboardingModelId}
            <p class="text-sm text-gray-400 font-sans mb-4">
              {onboardingModelId.split("/").pop() ?? onboardingModelId}
            </p>
          {/if}

          <!-- Centered ChatForm — first message completes onboarding -->
          <div class="w-full bg-exo-dark-gray rounded-xl p-4">
            <ChatForm
              placeholder="Ask anything"
              autofocus={true}
              showHelperText={false}
              showModelSelector={false}
              modelTasks={modelTasks()}
              modelCapabilities={modelCapabilities()}
              onSend={completeOnboarding}
            />
          </div>

          <!-- Suggestion chips -->
          <div class="flex flex-wrap justify-center gap-2.5 mt-8">
            {#each ["Write a poem about the ocean", "Explain quantum computing simply", "Help me debug my code", "Tell me a creative story"] as chip}
              <button
                type="button"
                onclick={() => {
                  sendMessage(chip);
                  completeOnboarding();
                }}
                class="px-4 py-2 rounded-full border border-gray-200 bg-gray-50 text-[13px] font-sans text-gray-500 hover:bg-gray-100 hover:text-gray-700 hover:border-gray-300 transition-all duration-200 cursor-pointer"
              >
                {chip}
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Model Picker Modal (available during onboarding step 4) -->
    {#if onboardingStep === 4}
      <ModelPickerModal
        isOpen={isModelPickerOpen}
        {models}
        {selectedModelId}
        favorites={favoritesSet}
        {recentModelIds}
        hasRecents={showRecentsTab}
        existingModelIds={new Set(models.map((m) => m.id))}
        canModelFit={(modelId) => {
          const model = models.find((m) => m.id === modelId);
          return model ? hasEnoughMemory(model) : false;
        }}
        getModelFitStatus={(modelId): ModelMemoryFitStatus => {
          const model = models.find((m) => m.id === modelId);
          return model ? getModelMemoryFitStatus(model) : "too_large";
        }}
        onSelect={(modelId) => {
          isModelPickerOpen = false;
          onboardingLaunchModel(modelId);
        }}
        onClose={() => (isModelPickerOpen = false)}
        onToggleFavorite={toggleFavorite}
        onAddModel={addModelFromPicker}
        onDeleteModel={deleteCustomModel}
        totalMemoryGB={clusterMemory().total / (1024 * 1024 * 1024)}
        usedMemoryGB={clusterMemory().used / (1024 * 1024 * 1024)}
        {downloadsData}
        topologyNodes={data?.nodes}
      />
    {/if}
  {:else}
    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- MAIN DASHBOARD (shown after onboarding)                -->
    <!-- ═══════════════════════════════════════════════════════ -->
    {#if !topologyOnlyEnabled}
      <HeaderNav
        showHome={chatStarted}
        onHome={handleGoHome}
        showSidebarToggle={true}
        {sidebarVisible}
        onToggleSidebar={toggleChatSidebarVisible}
        downloadProgress={activeDownloadSummary}
      />
    {/if}

    <!-- Main Content -->
    <main class="flex-1 flex overflow-hidden relative">
      <!-- Left: Conversation History Sidebar (hidden in topology-only mode or when toggled off) -->
      {#if !topologyOnlyEnabled && sidebarVisible}
        <div
          class="w-80 flex-shrink-0 border-r border-exo-yellow/10"
          role="complementary"
          aria-label="Conversation history"
        >
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
              filteredNodes={nodeFilter}
              onNodeClick={togglePreviewNodeFilter}
            />

            {@render clusterWarnings()}

            <!-- TB5 RDMA Available Info -->
            {#if tb5WithoutRdma && !tb5InfoDismissed}
              <div
                class="absolute left-4 flex items-center gap-2 px-3 py-2 rounded border border-blue-400/50 bg-blue-400/10 backdrop-blur-sm"
                class:top-16={tbBridgeCycles.length > 0}
                class:top-4={tbBridgeCycles.length === 0}
                role="status"
              >
                <svg
                  class="w-5 h-5 text-blue-400 flex-shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  stroke-width="2"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span class="text-sm font-mono text-blue-200">
                  RDMA AVAILABLE
                </span>
                <button
                  type="button"
                  onclick={() => (tb5InfoDismissed = true)}
                  class="ml-1 text-blue-300/60 hover:text-blue-200 transition-colors cursor-pointer"
                  title="Dismiss"
                  aria-label="Dismiss RDMA available notification"
                >
                  <svg
                    class="w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    stroke-width="2"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            {/if}

            <!-- Exit topology-only mode button -->
            <button
              type="button"
              onclick={toggleTopologyOnlyMode}
              class="absolute bottom-4 right-4 p-2 rounded border border-exo-yellow/30 bg-exo-dark-gray/80 hover:border-exo-yellow/50 hover:bg-exo-dark-gray transition-colors cursor-pointer backdrop-blur-sm"
              title="Exit topology only mode"
              aria-label="Exit topology only mode"
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
                filteredNodes={nodeFilter}
                onNodeClick={togglePreviewNodeFilter}
              />

              <!-- Initial loading state before first data fetch -->
              {#if !update}
                <div
                  class="absolute inset-0 flex items-center justify-center bg-exo-dark-gray/80"
                  in:fade={{ duration: 200 }}
                  out:fade={{ duration: 300 }}
                >
                  <div class="text-center">
                    <div
                      class="w-8 h-8 border-2 border-exo-yellow/30 border-t-exo-yellow rounded-full animate-spin mx-auto mb-4"
                    ></div>
                    <p
                      class="text-xs font-mono text-white/40 tracking-wider uppercase"
                    >
                      Connecting to cluster&hellip;
                    </p>
                  </div>
                </div>
              {/if}

              <!-- Welcome overlay - shown when no instances are running -->
              {#if instanceCount === 0 && update}
                <div
                  class="absolute inset-0 flex items-center justify-center pointer-events-none"
                  in:fade={{ duration: 400, delay: 200 }}
                >
                  <div class="text-center pointer-events-auto max-w-lg px-6">
                    <div class="mb-6">
                      <div
                        class="text-2xl font-mono text-exo-yellow font-bold tracking-wide mb-3 glow-text"
                      >
                        exo
                      </div>
                      <p
                        class="text-sm font-sans text-white/50 leading-relaxed mb-1"
                      >
                        {#if data && Object.keys(data.nodes).length > 1}
                          {Object.keys(data.nodes).length} devices connected. Choose
                          a model to start running AI across your cluster.
                        {:else if data && Object.keys(data.nodes).length === 1}
                          Your device is ready. Choose a model to start running
                          AI locally.
                        {:else}
                          Waiting for devices to connect&hellip;
                        {/if}
                      </p>
                    </div>

                    <button
                      type="button"
                      onclick={() => (isModelPickerOpen = true)}
                      class="inline-flex items-center gap-2 px-6 py-3 bg-exo-yellow text-exo-black font-sans text-sm font-semibold rounded-full hover:brightness-110 hover:shadow-[0_0_24px_rgba(255,215,0,0.2)] transition-all duration-200 cursor-pointer mb-4"
                    >
                      <svg
                        class="w-5 h-5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <path
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          d="M12 4v16m8-8H4"
                        />
                      </svg>
                      Choose a Model
                    </button>

                    <!-- Quick hints -->
                    <div
                      class="flex items-center justify-center gap-4 text-xs text-white/30 font-mono"
                    >
                      <span>models download automatically</span>
                      <span class="text-white/15">&bull;</span>
                      <a
                        href="/#/downloads"
                        class="hover:text-exo-yellow/60 transition-colors"
                        >view downloads</a
                      >
                    </div>
                  </div>
                </div>
              {/if}

              {@render clusterWarnings()}

              <!-- TB5 RDMA Available Info -->
              {#if tb5WithoutRdma && !tb5InfoDismissed}
                <div
                  class="absolute left-4 group"
                  class:top-16={tbBridgeCycles.length > 0}
                  class:top-4={tbBridgeCycles.length === 0}
                  role="status"
                >
                  <div
                    class="flex items-center gap-2 px-3 py-2 rounded border border-blue-400/50 bg-blue-400/10 backdrop-blur-sm"
                  >
                    <svg
                      class="w-5 h-5 text-blue-400 flex-shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      stroke-width="2"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span class="text-sm font-mono text-blue-200">
                      RDMA AVAILABLE
                    </span>
                    <button
                      type="button"
                      onclick={() => (tb5InfoDismissed = true)}
                      class="ml-1 text-blue-300/60 hover:text-blue-200 transition-colors cursor-pointer"
                      title="Dismiss"
                    >
                      <svg
                        class="w-4 h-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <path
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  </div>

                  <!-- Tooltip on hover -->
                  <div
                    class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-blue-400/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
                  >
                    <p class="text-xs text-white/80 mb-2">
                      Thunderbolt 5 hardware detected on multiple nodes. Enable
                      RDMA for significantly faster inter-node communication.
                    </p>
                    <p class="text-xs text-white/60 mb-1.5">
                      <span class="text-blue-300">To enable:</span>
                    </p>
                    <ol
                      class="text-xs text-white/60 list-decimal list-inside space-y-0.5 mb-1.5"
                    >
                      <li>Connect nodes with TB5 cables</li>
                      <li>Boot to Recovery (hold power 10s → Options)</li>
                      <li>
                        Run
                        <code class="text-blue-300 bg-blue-400/10 px-1 rounded"
                          >rdma_ctl enable</code
                        >
                      </li>
                      <li>Reboot</li>
                    </ol>
                    <p class="text-xs text-white/40">
                      Requires macOS 26.2+, TB5 cables, and matching OS
                      versions.
                    </p>
                  </div>
                </div>
              {/if}

              <!-- Node Filter Indicator (top-right corner) -->
              {#if isFilterActive()}
                <button
                  onclick={clearPreviewNodeFilter}
                  class="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-1 bg-exo-dark-gray/80 border border-exo-yellow/40 rounded text-exo-yellow hover:border-exo-yellow/60 transition-colors cursor-pointer backdrop-blur-sm"
                  title="Clear filter"
                >
                  <span class="text-[10px] font-mono tracking-wider">
                    FILTER: {nodeFilter.size}
                  </span>
                  <svg
                    class="w-3 h-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    stroke-width="2"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              {/if}
            </div>

            <!-- Chat Input - Below topology -->
            <div class="px-4 pt-6 pb-8">
              <div class="max-w-3xl mx-auto">
                {#if instanceCount === 0}
                  <!-- No model loaded prompt -->
                  <div class="text-center mb-6">
                    <p class="text-sm text-white/60 font-mono">
                      No model loaded yet. Select a model to get started.
                    </p>
                  </div>
                {/if}
                <ChatForm
                  placeholder={instanceCount === 0
                    ? "Choose a model above to start chatting"
                    : "Ask anything"}
                  showHelperText={false}
                  showModelSelector={true}
                  modelTasks={modelTasks()}
                  modelCapabilities={modelCapabilities()}
                />
              </div>
            </div>
          </div>

          <!-- Right Sidebar: Instance Controls (wider on welcome page for better visibility) -->
          <aside
            class="w-80 border-l border-exo-yellow/10 bg-exo-dark-gray flex flex-col flex-shrink-0"
            aria-label="Instance controls"
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
                  {#each unifiedDisplayItems as item (item.id)}
                    {@const id = item.id}
                    {@const instance = item.instance}
                    {@const downloadInfo = instance
                      ? getInstanceDownloadStatus(
                          item.instanceId ?? id,
                          instance,
                        )
                      : getMetaInstancePlacingStatus(id)}
                    {@const metaData = item.isMetaInstance
                      ? metaInstancesData[id]
                      : null}
                    {@const retryError =
                      metaData?.lastFailureError && !downloadInfo.isFailed
                        ? metaData.consecutiveFailures > 0
                          ? `(${((metaData.consecutiveFailures - 1) % 3) + 1}/3) ${metaData.lastFailureError}`
                          : metaData.lastFailureError
                        : null}
                    {@const statusText = downloadInfo.statusText}
                    {@const isDownloading = downloadInfo.isDownloading}
                    {@const isFailed =
                      statusText === "FAILED" ||
                      statusText === "PLACEMENT FAILED"}
                    {@const isLoading =
                      statusText.startsWith("LOADING") ||
                      statusText.startsWith("WARMING UP") ||
                      statusText === "WAITING" ||
                      statusText === "PLACING" ||
                      statusText.startsWith("RETRYING")}
                    {@const isReady =
                      statusText === "READY" || statusText === "LOADED"}
                    {@const isRunning = statusText === "RUNNING"}
                    <!-- Instance Card -->
                    {@const instanceModelId = item.modelId}
                    {@const instanceInfo = instance
                      ? getInstanceInfo(instance)
                      : {
                          instanceType:
                            item.instanceMeta === "MlxRing"
                              ? "MLX Ring"
                              : item.instanceMeta === "MlxJaccl"
                                ? "MLX RDMA"
                                : "Unknown",
                          sharding: item.sharding ?? "Unknown",
                          nodeNames: [] as string[],
                          nodeIds: [] as string[],
                          nodeCount: 0,
                        }}
                    {@const instanceConnections = instance
                      ? getInstanceConnections(instance)
                      : []}
                    <div
                      class="relative group cursor-pointer"
                      role="button"
                      tabindex="0"
                      transition:slide={{ duration: 250, easing: cubicOut }}
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
                        class="bg-exo-dark-gray/60 border border-l-2 transition-all duration-200 group-hover:bg-exo-dark-gray/80 {isDownloading
                          ? 'border-blue-500/30 border-l-blue-400 group-hover:border-blue-500/50'
                          : isFailed
                            ? 'border-red-500/30 border-l-red-400 group-hover:border-red-500/50'
                            : isLoading
                              ? 'border-exo-yellow/30 border-l-yellow-400 group-hover:border-exo-yellow/50'
                              : isReady
                                ? 'border-green-500/30 border-l-green-400 group-hover:border-green-500/50'
                                : 'border-teal-500/30 border-l-teal-400 group-hover:border-teal-500/50'} p-3"
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
                            onclick={() =>
                              item.isMetaInstance
                                ? deleteMetaInstance(id)
                                : deleteInstance(id)}
                            class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
                          >
                            DELETE
                          </button>
                        </div>
                        <div class="pl-2">
                          <div
                            class="text-exo-yellow text-xs font-mono tracking-wide truncate"
                          >
                            {instanceModelId}
                          </div>
                          <div
                            class="flex items-center gap-2 text-white/60 text-xs font-mono"
                          >
                            <span
                              >{instanceInfo.sharding} &middot; {instanceInfo.instanceType}</span
                            >
                            <span
                              class="px-1.5 py-0.5 text-[10px] tracking-wider uppercase rounded transition-all duration-300 {isDownloading
                                ? 'bg-blue-500/15 text-blue-400'
                                : isFailed
                                  ? 'bg-red-500/15 text-red-400'
                                  : isLoading
                                    ? 'bg-yellow-500/15 text-yellow-400'
                                    : isReady
                                      ? 'bg-green-500/15 text-green-400'
                                      : 'bg-teal-500/15 text-teal-400'}"
                            >
                              {statusText}
                            </span>
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
                            <div class="mt-2 space-y-1">
                              <div
                                class="text-xs text-blue-400 font-mono tracking-wider"
                              >
                                DOWNLOADING
                              </div>
                              <p
                                class="text-[11px] text-white/50 leading-relaxed"
                              >
                                Downloading model files. This runs locally on
                                your device and needs to finish before you can
                                chat.
                              </p>
                            </div>
                          {:else}
                            <div class="mt-1 space-y-1">
                              <div
                                class="text-xs {getStatusColor(
                                  downloadInfo.statusText,
                                )} font-mono tracking-wider"
                              >
                                {downloadInfo.statusText}
                              </div>
                              {#if isLoading}
                                <p
                                  class="text-[11px] text-white/50 leading-relaxed"
                                >
                                  Loading model into memory for fast
                                  inference...
                                </p>
                              {:else if isReady || isRunning}
                                <p
                                  class="text-[11px] text-green-400/70 leading-relaxed"
                                >
                                  Ready to chat! Type a message below.
                                </p>
                              {/if}
                            </div>
                            {#if downloadInfo.isFailed && downloadInfo.errorMessage}
                              <div
                                class="text-xs text-red-400/80 font-mono mt-1 break-words"
                              >
                                {downloadInfo.errorMessage}
                              </div>
                            {:else if retryError}
                              <div
                                class="text-xs text-orange-400/80 font-mono mt-1 break-words"
                              >
                                Retrying after error: {retryError}
                              </div>
                            {/if}
                            {#if downloadInfo.perNodeStatus.length > 1 && (statusText.startsWith("LOADING") || statusText.startsWith("WARMING UP") || statusText === "WAITING" || statusText === "INITIALIZING")}
                              <div class="mt-1.5 space-y-0.5">
                                {#each downloadInfo.perNodeStatus as node}
                                  <div
                                    class="flex items-center justify-between text-[10px] font-mono"
                                  >
                                    <span class="text-white/60 truncate pr-2"
                                      >{node.nodeName}</span
                                    >
                                    <span
                                      class={getStatusColor(
                                        node.status.toUpperCase(),
                                      )}>{node.status}</span
                                    >
                                  </div>
                                {/each}
                              </div>
                            {/if}
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
                <div
                  class="w-2 h-2 border border-exo-yellow/60 rotate-45"
                ></div>
                <h3
                  class="text-xs text-exo-yellow font-mono tracking-[0.2em] uppercase"
                >
                  Load Model
                </h3>
                <div
                  class="flex-1 h-px bg-gradient-to-r from-exo-yellow/30 to-transparent"
                ></div>
                <span class="text-sm text-white/70 font-mono"
                  >{models.length} models</span
                >
              </div>

              <!-- Model Picker Button -->
              <div class="flex-shrink-0 mb-3">
                <button
                  type="button"
                  onclick={() => (isModelPickerOpen = true)}
                  class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded pl-3 pr-8 py-2.5 text-sm font-mono text-left tracking-wide cursor-pointer transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70 relative"
                >
                  {#if selectedModelId}
                    {@const foundModel = models.find(
                      (m) => m.id === selectedModelId,
                    )}
                    {#if foundModel}
                      {@const sizeGB = getModelSizeGB(foundModel)}
                      <span
                        class="flex items-center justify-between gap-2 w-full pr-4"
                      >
                        <span
                          class="flex items-center gap-2 text-exo-light-gray truncate"
                        >
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
                  <div
                    class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
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
                </button>
              </div>

              <!-- Advanced Options Toggle -->
              <div class="flex-shrink-0 mb-4">
                <button
                  type="button"
                  onclick={() => (showAdvancedOptions = !showAdvancedOptions)}
                  class="flex items-center gap-2 text-xs text-white/50 hover:text-white/70 font-mono tracking-wider uppercase transition-colors cursor-pointer py-1"
                  aria-expanded={showAdvancedOptions}
                >
                  <svg
                    class="w-3 h-3 transition-transform duration-200 {showAdvancedOptions
                      ? 'rotate-90'
                      : ''}"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    stroke-width="2"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                  Advanced Options
                </button>

                {#if showAdvancedOptions}
                  <div class="mt-3 space-y-3 pl-1" in:fade={{ duration: 150 }}>
                    <!-- Sharding Strategy -->
                    <div>
                      <div class="text-xs text-white/50 font-mono mb-2">
                        Sharding Strategy:
                      </div>
                      <div class="flex gap-2">
                        <button
                          onclick={() => {
                            selectedSharding = "Pipeline";
                            saveLaunchDefaults();
                          }}
                          class="flex items-center gap-2 py-1.5 px-3 text-xs font-mono border rounded transition-all duration-200 cursor-pointer {selectedSharding ===
                          'Pipeline'
                            ? 'bg-transparent text-exo-yellow border-exo-yellow'
                            : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                        >
                          <span
                            class="w-3 h-3 rounded-full border-2 flex items-center justify-center {selectedSharding ===
                            'Pipeline'
                              ? 'border-exo-yellow'
                              : 'border-exo-medium-gray'}"
                          >
                            {#if selectedSharding === "Pipeline"}
                              <span
                                class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
                              ></span>
                            {/if}
                          </span>
                          Pipeline
                        </button>
                        <button
                          onclick={() => {
                            selectedSharding = "Tensor";
                            saveLaunchDefaults();
                          }}
                          class="flex items-center gap-2 py-1.5 px-3 text-xs font-mono border rounded transition-all duration-200 cursor-pointer {selectedSharding ===
                          'Tensor'
                            ? 'bg-transparent text-exo-yellow border-exo-yellow'
                            : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                        >
                          <span
                            class="w-3 h-3 rounded-full border-2 flex items-center justify-center {selectedSharding ===
                            'Tensor'
                              ? 'border-exo-yellow'
                              : 'border-exo-medium-gray'}"
                          >
                            {#if selectedSharding === "Tensor"}
                              <span
                                class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
                              ></span>
                            {/if}
                          </span>
                          Tensor
                        </button>
                      </div>
                    </div>

                    <!-- Interconnect -->
                    <div>
                      <div class="text-xs text-white/50 font-mono mb-2">
                        Interconnect:
                      </div>
                      <div class="flex gap-2">
                        <button
                          onclick={() => {
                            selectedInstanceType = "MlxRing";
                            saveLaunchDefaults();
                          }}
                          class="flex items-center gap-2 py-1.5 px-3 text-xs font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType ===
                          'MlxRing'
                            ? 'bg-transparent text-exo-yellow border-exo-yellow'
                            : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                        >
                          <span
                            class="w-3 h-3 rounded-full border-2 flex items-center justify-center {selectedInstanceType ===
                            'MlxRing'
                              ? 'border-exo-yellow'
                              : 'border-exo-medium-gray'}"
                          >
                            {#if selectedInstanceType === "MlxRing"}
                              <span
                                class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
                              ></span>
                            {/if}
                          </span>
                          Standard
                        </button>
                        <button
                          onclick={() => {
                            selectedInstanceType = "MlxIbv";
                            saveLaunchDefaults();
                          }}
                          class="flex items-center gap-2 py-1.5 px-3 text-xs font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType ===
                          'MlxIbv'
                            ? 'bg-transparent text-exo-yellow border-exo-yellow'
                            : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                        >
                          <span
                            class="w-3 h-3 rounded-full border-2 flex items-center justify-center {selectedInstanceType ===
                            'MlxIbv'
                              ? 'border-exo-yellow'
                              : 'border-exo-medium-gray'}"
                          >
                            {#if selectedInstanceType === "MlxIbv"}
                              <span
                                class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
                              ></span>
                            {/if}
                          </span>
                          RDMA (Fast)
                        </button>
                      </div>
                    </div>

                    <!-- Minimum Devices -->
                    <div>
                      <div class="text-xs text-white/50 font-mono mb-2">
                        Minimum Devices:
                      </div>
                      <!-- Discrete slider track with drag support -->
                      <!-- svelte-ignore a11y_no_static_element_interactions -->
                      <div
                        bind:this={sliderTrackElement}
                        class="relative h-16 cursor-pointer select-none px-2 pr-6"
                        onmousedown={handleSliderMouseDown}
                        ontouchstart={handleSliderTouchStart}
                      >
                        <!-- Track background -->
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
                        <!-- Dots and labels for each device count -->
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
                            <span
                              class="rounded-full transition-all {isSelected
                                ? 'w-6 h-6 bg-exo-yellow shadow-[0_0_10px_rgba(255,215,0,0.6)]'
                                : isValid
                                  ? 'w-4 h-4 bg-exo-light-gray/70 mt-1'
                                  : 'w-3 h-3 bg-exo-medium-gray/50 mt-1.5'}"
                            ></span>
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
                {/if}
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
              role="log"
              aria-live="polite"
              aria-label="Chat messages"
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
                  modelCapabilities={modelCapabilities()}
                />
              </div>
            </div>
          </div>

          <!-- Right: Mini-Map Sidebar -->
          {#if minimized}
            <aside
              class="w-80 border-l border-exo-yellow/20 bg-exo-dark-gray flex flex-col flex-shrink-0 overflow-y-auto"
              in:fly={{ x: 100, duration: 400, easing: cubicInOut }}
              aria-label="Cluster topology"
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
                  <TopologyGraph
                    highlightedNodes={highlightedNodes()}
                    filteredNodes={nodeFilter}
                    onNodeClick={togglePreviewNodeFilter}
                  />

                  {@render clusterWarningsCompact()}
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
                          <div
                            class="flex justify-between items-start mb-2 pl-2"
                          >
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
                            <div
                              class="flex items-center gap-2 text-white/60 text-xs font-mono"
                            >
                              <span
                                >{instanceInfo.sharding} &middot; {instanceInfo.instanceType}</span
                              >
                              <span
                                class="px-1.5 py-0.5 text-[10px] tracking-wider uppercase rounded transition-all duration-300 {isDownloading
                                  ? 'bg-blue-500/15 text-blue-400'
                                  : isFailed
                                    ? 'bg-red-500/15 text-red-400'
                                    : isLoading
                                      ? 'bg-yellow-500/15 text-yellow-400'
                                      : isReady
                                        ? 'bg-green-500/15 text-green-400'
                                        : 'bg-teal-500/15 text-teal-400'}"
                              >
                                {statusText}
                              </span>
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
                                                    >{formatSpeed(f.speed)} • ETA
                                                    {formatEta(f.etaMs)}</span
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
                              <div class="mt-2 space-y-1">
                                <div
                                  class="text-xs text-blue-400 font-mono tracking-wider"
                                >
                                  DOWNLOADING
                                </div>
                                <p
                                  class="text-[11px] text-white/50 leading-relaxed"
                                >
                                  Downloading model files. This runs locally on
                                  your device and needs to finish before you can
                                  chat.
                                </p>
                              </div>
                            {:else}
                              <div class="mt-1 space-y-1">
                                <div
                                  class="text-xs {getStatusColor(
                                    downloadInfo.statusText,
                                  )} font-mono tracking-wider"
                                >
                                  {downloadInfo.statusText}
                                </div>
                                {#if isLoading}
                                  <p
                                    class="text-[11px] text-white/50 leading-relaxed"
                                  >
                                    Loading model into memory for fast
                                    inference...
                                  </p>
                                {:else if isReady || isRunning}
                                  <p
                                    class="text-[11px] text-green-400/70 leading-relaxed"
                                  >
                                    Ready to chat! Type a message below.
                                  </p>
                                {/if}
                              </div>
                              {#if downloadInfo.isFailed && downloadInfo.errorMessage}
                                <div
                                  class="text-xs text-red-400/80 font-mono mt-1 break-words"
                                >
                                  {downloadInfo.errorMessage}
                                </div>
                              {/if}
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
  {/if}
</div>

{#if !showOnboarding}
  <ModelPickerModal
    isOpen={isModelPickerOpen}
    {models}
    {selectedModelId}
    favorites={favoritesSet}
    {recentModelIds}
    hasRecents={showRecentsTab}
    existingModelIds={new Set(models.map((m) => m.id))}
    canModelFit={(modelId) => {
      const model = models.find((m) => m.id === modelId);
      return model ? hasEnoughMemory(model) : false;
    }}
    getModelFitStatus={(modelId): ModelMemoryFitStatus => {
      const model = models.find((m) => m.id === modelId);
      return model ? getModelMemoryFitStatus(model) : "too_large";
    }}
    onSelect={handleModelPickerSelect}
    onClose={() => (isModelPickerOpen = false)}
    onToggleFavorite={toggleFavorite}
    onAddModel={addModelFromPicker}
    onDeleteModel={deleteCustomModel}
    totalMemoryGB={clusterMemory().total / (1024 * 1024 * 1024)}
    usedMemoryGB={clusterMemory().used / (1024 * 1024 * 1024)}
    {downloadsData}
    topologyNodes={data?.nodes}
  />
{/if}

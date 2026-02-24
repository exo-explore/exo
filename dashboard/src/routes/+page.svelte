<script lang="ts">
  import {
    TopologyGraph,
    ChatForm,
    ChatMessages,
    ChatSidebar,
    ModelCard,
    ModelPickerModal,
    ChatModelSelector,
  } from "$lib/components";
  import {
    pickAutoModel,
    getAutoTierIndex,
  } from "$lib/components/ChatModelSelector.svelte";
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
    messages,
    debugMode,
    toggleDebugMode,
    topologyOnlyMode,
    toggleTopologyOnlyMode,
    chatSidebarVisible,
    toggleChatSidebarVisible,
    nodeThunderbolt,
    nodeRdmaCtl,
    thunderboltBridgeCycles,
    nodeThunderboltBridge,
    nodeIdentities,
    isConnected,
    type DownloadProgress,
    type PlacementPreview,
  } from "$lib/stores/app.svelte";
  import { addToast, dismissByMessage } from "$lib/stores/toast.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import DeviceIcon from "$lib/components/DeviceIcon.svelte";
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
  const tbBridgeCycles = $derived(thunderboltBridgeCycles());
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

  // Detect Mac Studio nodes using RDMA on en2 (the port next to ethernet — RDMA doesn't work there)
  const macStudioEn2RdmaWarning = $derived.by(() => {
    const edges = data?.edges;
    const ids = tbIdentifiers;
    const rdmaCtl = rdmaCtlData;
    if (!edges || !ids || !rdmaCtl) return null;

    const affectedConnections: Array<{
      nodeId: string;
      nodeName: string;
      peerNodeId: string;
      peerNodeName: string;
      rdmaIface: string;
    }> = [];

    const isMacStudio = (node: (typeof data.nodes)[string] | undefined) =>
      node?.system_info?.model_id === "Mac Studio";

    for (const edge of edges) {
      if (!edge.sourceRdmaIface && !edge.sinkRdmaIface) continue;

      const sourceNode = data?.nodes?.[edge.source];
      if (
        isMacStudio(sourceNode) &&
        edge.sourceRdmaIface === "rdma_en2" &&
        rdmaCtl[edge.source]?.enabled
      ) {
        affectedConnections.push({
          nodeId: edge.source,
          nodeName:
            sourceNode?.friendly_name || edge.source.slice(0, 8) + "...",
          peerNodeId: edge.target,
          peerNodeName:
            data?.nodes?.[edge.target]?.friendly_name ||
            edge.target.slice(0, 8) + "...",
          rdmaIface: "en2",
        });
      }

      const sinkNode = data?.nodes?.[edge.target];
      if (
        isMacStudio(sinkNode) &&
        edge.sinkRdmaIface === "rdma_en2" &&
        rdmaCtl[edge.target]?.enabled
      ) {
        affectedConnections.push({
          nodeId: edge.target,
          nodeName: sinkNode?.friendly_name || edge.target.slice(0, 8) + "...",
          peerNodeId: edge.source,
          peerNodeName:
            sourceNode?.friendly_name || edge.source.slice(0, 8) + "...",
          rdmaIface: "en2",
        });
      }
    }

    // Deduplicate by nodeId
    const seen = new Set<string>();
    const unique = affectedConnections.filter((c) => {
      if (seen.has(c.nodeId)) return false;
      seen.add(c.nodeId);
      return true;
    });

    return unique.length > 0 ? unique : null;
  });
  let macStudioEn2Dismissed = $state(false);

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
  let localNodeId = $state<string | null>(null);

  // ── Onboarding wizard state ──
  const ONBOARDING_COMPLETE_KEY = "exo-onboarding-complete";
  let onboardingStep = $state(0); // 0 = not in onboarding, 1-9 = wizard steps
  let onboardingModelId = $state<string | null>(null); // model selected during onboarding
  let onboardingFadingOut = $state(false); // true during fade-out transition
  const showOnboarding = $derived(onboardingStep > 0);
  const showOnboardingOverlay = $derived(showOnboarding || onboardingFadingOut);

  // ── Steps 1-5 animation state: cinematic SVG story ──
  const SIMULATED_STUDIO_GB = 256; // simulated Mac Studio memory
  const onboardingCombinedGB = $derived(
    userDeviceInfo.memoryGB + SIMULATED_STUDIO_GB,
  );

  // Models unlocked by adding the second device — one per base model, well-known preferred
  const unlockedModels = $derived.by(() => {
    if (models.length === 0) return [];
    const singleGB = userDeviceInfo.memoryGB;
    const combinedGB = onboardingCombinedGB;
    const candidates = models
      .filter((m) => {
        const sizeGB = getModelSizeGB(m);
        return sizeGB > singleGB && sizeGB <= combinedGB && m.family;
      })
      .sort((a, b) => getModelSizeGB(a) - getModelSizeGB(b));
    // Deduplicate by base_model (or family as fallback) — keep smallest quant per base
    const seen = new Set<string>();
    const deduped: typeof candidates = [];
    for (const m of candidates) {
      const key = m.base_model || m.family || m.id;
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push(m);
    }
    return deduped.slice(0, 3);
  });

  // User device info from topology — uses /node_id to find our own node
  const userDeviceInfo = $derived.by(() => {
    if (!data || Object.keys(data.nodes).length === 0) {
      return { name: "MacBook Pro", memoryGB: 36, deviceType: "macbook pro" };
    }
    const ourNode = localNodeId ? data.nodes[localNodeId] : undefined;
    const node = ourNode ?? Object.values(data.nodes)[0];
    const totalMem =
      node.macmon_info?.memory?.ram_total ?? node.system_info?.memory ?? 0;
    const memGB = Math.round(totalMem / (1024 * 1024 * 1024));
    const name = node.friendly_name || "Your Mac";
    const modelId = (node.system_info?.model_id || "macbook pro").toLowerCase();
    return { name, memoryGB: memGB || 36, deviceType: modelId };
  });

  let showContinueButton = $state(false);
  let stepTitle = $state("");
  let stepTransitioning = $state(false);

  // Advance to the next onboarding step
  function advanceStep(target: number) {
    showContinueButton = false;
    if (target <= 5) {
      // Steps 1-5 share a persistent SVG canvas — just set the step directly
      onboardingStep = target;
    } else {
      // Leaving the cinematic sequence — fade out, then switch
      stepTransitioning = true;
      setTimeout(() => {
        onboardingStep = target;
        stepTransitioning = false;
      }, 350);
    }
  }

  // Tweened animation values for the persistent SVG canvas
  const device1X = tweened(350, { duration: 800, easing: cubicInOut });
  const device2X = tweened(550, { duration: 800, easing: cubicInOut });
  const device2Opacity = tweened(0, { duration: 600, easing: cubicOut });
  const connectionOpacity = tweened(0, { duration: 500, easing: cubicOut });
  const connectionIsRed = tweened(0, { duration: 500, easing: cubicOut }); // 0=gold, 1=red
  const combinedLabelOpacity = tweened(0, { duration: 500, easing: cubicOut });
  const modelBlockY = tweened(20, { duration: 700, easing: cubicInOut });
  const modelBlockOpacity = tweened(0, { duration: 500, easing: cubicOut });
  const modelSplitProgress = tweened(0, { duration: 800, easing: cubicInOut }); // 0=unified, 1=fully split
  const disconnectXOpacity = tweened(0, { duration: 400, easing: cubicOut });
  const device1Opacity = tweened(1, { duration: 600, easing: cubicOut });
  const logoOpacity = tweened(1, { duration: 600, easing: cubicOut });
  // Step 2 chip fade: 0→N where each chip fades in at its stagger offset
  const chipPhase = tweened(0, { duration: 800, easing: cubicOut });
  const deviceCountOpacity = tweened(0, { duration: 600, easing: cubicOut });
  const topologyOpacity = tweened(1, { duration: 400, easing: cubicOut });
  const titleOpacity = tweened(0, { duration: 500, easing: cubicOut });
  const subtitleOpacity = tweened(0, { duration: 500, easing: cubicOut });

  // ── Step 1: "Your EXO Network" — show real topology ──
  $effect(() => {
    if (onboardingStep === 1) {
      showContinueButton = false;
      stepTitle = "";
      // Reset all tweens to initial
      device1X.set(350, { duration: 0 });
      device1Opacity.set(0, { duration: 0 });
      device2Opacity.set(0, { duration: 0 });
      connectionOpacity.set(0, { duration: 0 });
      connectionIsRed.set(0, { duration: 0 });
      combinedLabelOpacity.set(0, { duration: 0 });
      modelBlockOpacity.set(0, { duration: 0 });
      modelSplitProgress.set(0, { duration: 0 });
      disconnectXOpacity.set(0, { duration: 0 });
      logoOpacity.set(1, { duration: 0 });
      titleOpacity.set(0, { duration: 0 });
      subtitleOpacity.set(0, { duration: 0 });
      chipPhase.set(0, { duration: 0 });
      deviceCountOpacity.set(0, { duration: 0 });
      topologyOpacity.set(1, { duration: 0 });

      const t1 = setTimeout(() => {
        titleOpacity.set(1);
      }, 300);
      const t2 = setTimeout(() => {
        deviceCountOpacity.set(1);
      }, 800);
      const t3 = setTimeout(() => {
        showContinueButton = true;
      }, 1200);

      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
      };
    }
  });

  // ── Step 2: "Add devices to run larger models" — cross-fade topology out, device pair animates in ──
  $effect(() => {
    if (onboardingStep === 2) {
      showContinueButton = false;

      // Cross-fade: fade out real topology
      topologyOpacity.set(0);

      // Immediately transition out step 1 elements
      logoOpacity.set(0);
      deviceCountOpacity.set(0);
      // Smoothly crossfade the title: fade old out, update text, fade new in
      titleOpacity.set(0, { duration: 300 });
      subtitleOpacity.set(0, { duration: 0 });

      // Delay all step 2 animations by 400ms to let topology fade out
      const DELAY = 400;

      const t0 = setTimeout(() => {
        stepTitle = "Add devices to run larger models";
        titleOpacity.set(1, { duration: 400 });
      }, DELAY + 300);

      const t1 = setTimeout(() => {
        device1Opacity.set(1, { duration: 0 });
        device1X.set(220);
        device2X.set(480, { duration: 0 });
        device2Opacity.set(0, { duration: 0 });
      }, DELAY + 200);
      const t2 = setTimeout(() => {
        device2Opacity.set(1);
        device2X.set(480);
      }, DELAY + 700);
      const t3 = setTimeout(() => {
        connectionOpacity.set(1);
      }, DELAY + 1200);
      const t4 = setTimeout(() => {
        combinedLabelOpacity.set(1);
      }, DELAY + 1600);
      // Staggered chip fade-in (each chip offsets by 0.6 in chipPhase)
      const t5 = setTimeout(() => {
        chipPhase.set(3, { duration: 1800 });
      }, DELAY + 1800);
      const t6 = setTimeout(() => {
        showContinueButton = true;
      }, DELAY + 3200);

      return () => {
        clearTimeout(t0);
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
        clearTimeout(t4);
        clearTimeout(t5);
        clearTimeout(t6);
      };
    }
  });

  // ── Step 3: "exo splits the model" — model block appears, splits ──
  $effect(() => {
    if (onboardingStep === 3) {
      showContinueButton = false;
      // Gently fade out the unlock chips
      chipPhase.set(0, { duration: 600 });

      // Crossfade title
      titleOpacity.set(0, { duration: 250 });
      subtitleOpacity.set(0, { duration: 250 });
      setTimeout(() => {
        stepTitle = "exo splits models across devices";
        titleOpacity.set(1, { duration: 400 });
        subtitleOpacity.set(1, { duration: 400 });
      }, 250);

      // Wait for chips to fade before showing model block
      const t1 = setTimeout(() => {
        modelBlockOpacity.set(1);
        modelBlockY.set(50);
      }, 600);
      const t2 = setTimeout(() => {
        modelSplitProgress.set(1);
      }, 1500);
      const t3 = setTimeout(() => {
        showContinueButton = true;
      }, 2300);

      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
      };
    }
  });

  // ── Step 4: "A device disconnects... exo self-heals" — full disconnect+heal sequence ──
  $effect(() => {
    if (onboardingStep === 4) {
      showContinueButton = false;

      // Crossfade title
      titleOpacity.set(0, { duration: 250 });
      subtitleOpacity.set(0, { duration: 250 });
      setTimeout(() => {
        stepTitle = "When a device disconnects...";
        titleOpacity.set(1, { duration: 400 });
        subtitleOpacity.set(1, { duration: 400 });
      }, 250);

      // Phase 1: Disconnect
      const t1 = setTimeout(() => {
        connectionIsRed.set(1);
      }, 400);
      const t2 = setTimeout(() => {
        disconnectXOpacity.set(1);
      }, 800);
      const t3 = setTimeout(() => {
        device2Opacity.set(0);
        connectionOpacity.set(0);
        disconnectXOpacity.set(0);
        combinedLabelOpacity.set(0);
      }, 1600);

      // Phase 2: Self-heal — crossfade title + subtitle
      const t4 = setTimeout(() => {
        titleOpacity.set(0, { duration: 250 });
        subtitleOpacity.set(0, { duration: 250 });
      }, 2550);
      const t4b = setTimeout(() => {
        stepTitle = "exo self-heals";
        titleOpacity.set(1, { duration: 400 });
        subtitleOpacity.set(1, { duration: 400 });
      }, 2800);
      const t5 = setTimeout(() => {
        device1X.set(350);
        device2X.set(350);
      }, 3100);
      const t6 = setTimeout(() => {
        modelSplitProgress.set(0);
        modelBlockY.set(20); // Lift up while merging
        connectionIsRed.set(0);
      }, 3700);
      const t7 = setTimeout(() => {
        modelBlockY.set(125); // Settle back down just above the device
      }, 4800);
      const t8 = setTimeout(() => {
        advanceStep(6);
      }, 6200);

      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
        clearTimeout(t4);
        clearTimeout(t4b);
        clearTimeout(t5);
        clearTimeout(t6);
        clearTimeout(t7);
        clearTimeout(t8);
      };
    }
  });

  // Recommended models for onboarding: 2 large, 2 medium, 2 small
  // Always includes Llama-3.2-3B-4bit as a fast-loading small option
  const PINNED_ONBOARDING_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit";
  const onboardingModels = $derived.by(() => {
    if (models.length === 0) return [];
    const sorted = [...models]
      .filter((m) => hasEnoughMemory(m) && getModelSizeGB(m) > 0)
      .sort((a, b) => getModelSizeGB(b) - getModelSizeGB(a));
    if (sorted.length <= 6) return sorted;

    // Split into thirds by size: large (top third), medium (middle), small (bottom)
    const third = Math.max(1, Math.floor(sorted.length / 3));
    const large = sorted.slice(0, third);
    const medium = sorted.slice(third, third * 2);
    const small = sorted.slice(third * 2);

    // Pick 2 from each tier, ensuring pinned model counts as a small pick
    const pinned =
      small.find((m) => m.id === PINNED_ONBOARDING_MODEL) ||
      sorted.find((m) => m.id === PINNED_ONBOARDING_MODEL);
    const pickLarge = large.slice(0, 2);
    const pickMedium = medium.slice(0, 2);
    const pickSmall = pinned
      ? [
          small.find((m) => m.id !== PINNED_ONBOARDING_MODEL) || small[0],
          pinned,
        ].filter(Boolean)
      : small.slice(0, 2);

    const result = [...pickLarge, ...pickMedium, ...pickSmall];
    // Deduplicate (in case pinned was already picked)
    const seen = new Set<string>();
    return result.filter((m) => {
      if (seen.has(m.id)) return false;
      seen.add(m.id);
      return true;
    });
  });

  // Track onboarding instance status for auto-advancing steps.
  // Uses runner status as source of truth to avoid false "ready" from missing download data.
  // Only tracks the specific model launched during onboarding (ignores other running instances).
  $effect(() => {
    if (onboardingStep === 7 && instanceCount > 0 && onboardingModelId) {
      let anyDownloading = false;
      let anyReady = false;
      for (const [id, inst] of Object.entries(instanceData)) {
        // Only check instances for the model we launched during onboarding
        if (getInstanceModelId(inst) !== onboardingModelId) continue;
        const runnerStatus = deriveInstanceStatus(inst);
        if (
          runnerStatus.statusText === "READY" ||
          runnerStatus.statusText === "LOADED" ||
          runnerStatus.statusText === "RUNNING"
        ) {
          anyReady = true;
        } else if (runnerStatus.statusText === "DOWNLOADING") {
          anyDownloading = true;
        } else {
          const dlStatus = getInstanceDownloadStatus(id, inst);
          if (dlStatus.isDownloading) anyDownloading = true;
        }
      }
      // Model already cached & ready — skip download AND loading steps
      if (anyReady) {
        onboardingStep = 9;
      } else if (anyDownloading) {
        // Stay on step 7 (downloading)
      } else {
        // Not ready and not downloading — could be loading, initializing, or preparing.
        // Only advance to step 8 if runners are actually in a loading state.
        for (const [, inst] of Object.entries(instanceData)) {
          if (getInstanceModelId(inst) !== onboardingModelId) continue;
          const runnerStatus = deriveInstanceStatus(inst);
          if (
            runnerStatus.statusText === "LOADING" ||
            runnerStatus.statusText === "WARMING UP"
          ) {
            onboardingStep = 8;
            break;
          }
        }
      }
    }
  });

  $effect(() => {
    if (onboardingStep === 8 && instanceCount > 0 && onboardingModelId) {
      for (const [, inst] of Object.entries(instanceData)) {
        if (getInstanceModelId(inst) !== onboardingModelId) continue;
        const runnerStatus = deriveInstanceStatus(inst);
        if (
          runnerStatus.statusText === "READY" ||
          runnerStatus.statusText === "LOADED" ||
          runnerStatus.statusText === "RUNNING"
        ) {
          onboardingStep = 9;
          break;
        }
      }
    }
  });

  function completeOnboarding() {
    // Trigger fade-out, then fully remove overlay
    onboardingFadingOut = true;
    onboardingStep = 0;
    try {
      localStorage.setItem(ONBOARDING_COMPLETE_KEY, "true");
    } catch {
      // ignore
    }
    // Persist to server (~/.exo)
    fetch("/onboarding", { method: "POST" }).catch(() => {});
    // Remove overlay after fade-out transition completes
    setTimeout(() => {
      onboardingFadingOut = false;
    }, 500);
  }

  // Auto-complete onboarding when user sends a message from step 9
  $effect(() => {
    if (onboardingStep === 9 && chatStarted) {
      completeOnboarding();
    }
  });

  let onboardingError = $state<string | null>(null);

  async function onboardingLaunchModel(modelId: string) {
    onboardingModelId = modelId;
    onboardingError = null;
    selectPreviewModel(modelId);
    onboardingStep = 7;
    // Launch via standard placement API (same as main dashboard)
    // Single-node: force Pipeline/Ring regardless of persisted defaults
    const nodeCount = topologyData()
      ? Object.keys(topologyData()!.nodes).length
      : 1;
    const sharding = nodeCount <= 1 ? "Pipeline" : selectedSharding;
    const instanceType = nodeCount <= 1 ? "MlxRing" : selectedInstanceType;
    try {
      const placementResponse = await fetch(
        `/instance/placement?model_id=${encodeURIComponent(modelId)}&sharding=${sharding}&instance_meta=${instanceType}&min_nodes=1`,
      );
      if (!placementResponse.ok) {
        const errorText = await placementResponse.text();
        onboardingError = `Failed to get placement: ${errorText}`;
        onboardingStep = 6;
        return;
      }
      const instanceData = await placementResponse.json();
      const response = await fetch("/instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance: instanceData }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        onboardingError = `Failed to launch: ${errorText}`;
        onboardingStep = 6;
        return;
      }
      setSelectedChatModel(modelId);
      recordRecentLaunch(modelId);
    } catch (error) {
      onboardingError = `Network error: ${error}`;
      onboardingStep = 6;
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

  // Helper to get onboarding model loading progress (layers loaded)
  const onboardingLoadProgress = $derived.by(() => {
    if (instanceCount === 0 || !onboardingModelId) return null;
    let layersLoaded = 0,
      totalLayers = 0;
    for (const [, inst] of Object.entries(instanceData)) {
      if (getInstanceModelId(inst) !== onboardingModelId) continue;
      const status = deriveInstanceStatus(inst);
      if (
        status.statusText === "LOADING" &&
        status.totalLayers &&
        status.totalLayers > 0
      ) {
        layersLoaded += status.layersLoaded ?? 0;
        totalLayers += status.totalLayers;
      }
    }
    if (totalLayers === 0) return null;
    return {
      layersLoaded,
      totalLayers,
      percentage: (layersLoaded / totalLayers) * 100,
    };
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
  const LAUNCH_DEFAULTS_KEY = "exo-launch-defaults-v2";
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
    selectedInstanceType =
      defaults.instanceType === "MlxRing" ? "MlxRing" : "MlxJaccl";

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
      setSelectedChatModel(defaults.modelId);
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
      : runtime === "MlxJaccl";

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

  onMount(async () => {
    mounted = true;
    fetchModels();
    fetch("/node_id")
      .then((r) => (r.ok ? r.json() : null))
      .then((id) => {
        if (id) localNodeId = id;
      })
      .catch(() => {});

    // Handle reset-onboarding query parameter (triggered from native Settings)
    const params = new URLSearchParams(window.location.search);
    if (params.has("reset-onboarding")) {
      localStorage.removeItem(ONBOARDING_COMPLETE_KEY);
      window.history.replaceState({}, "", window.location.pathname);
      onboardingStep = 1;
      return;
    }

    // Check server-side onboarding state (persisted in ~/.exo)
    try {
      const res = await fetch("/onboarding");
      if (res.ok) {
        const data = await res.json();
        if (!data.completed) {
          onboardingStep = 1;
        }
        return;
      }
    } catch {
      // Server unreachable — fall through to localStorage
    }

    // Fallback: check localStorage
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
    setSelectedChatModel(modelId);
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
      // Use the specific preview if provided, otherwise fall back to filtered preview
      const preview = specificPreview ?? filteredPreview();

      let response: Response;
      if (preview?.instance) {
        // Launch with pre-computed placement from preview
        response = await fetch("/instance", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ instance: preview.instance }),
        });
      } else {
        // No preview available — use place_instance to let server decide placement
        response = await fetch("/place_instance", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_id: modelId,
            sharding: selectedSharding,
            instance_meta: selectedInstanceType,
            min_nodes: 1,
          }),
        });
      }

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Failed to launch instance:", errorText);
        addToast({
          type: "error",
          message: `Failed to launch model: ${errorText}`,
        });
      } else {
        addToast({ type: "info", message: `Launching model...` });
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
    const totalBytes = getBytes(prog.total);
    const downloadedBytes = getBytes(prog.downloaded);
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
      const fTotal = getBytes(fd.total);
      const fDownloaded = getBytes(fd.downloaded);
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
  } {
    if (!downloadsData || Object.keys(downloadsData).length === 0) {
      // No download data yet — defer to runner status instead of assuming RUNNING
      const statusInfo = deriveInstanceStatus(instanceWrapped);
      return {
        isDownloading: false,
        isFailed: false,
        errorMessage: null,
        progress: null,
        statusText: statusInfo.statusText,
        perNode: [],
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
        errorMessage: null,
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
    layersLoaded?: number;
    totalLayers?: number;
  } {
    const [instanceTag, instance] = getTagged(instanceWrapped);
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
    if (has("Loading")) {
      // Tensor parallel: each runner loads all layers — use max/min (bottleneck)
      // Pipeline parallel: each runner loads a disjoint slice — use sum
      const isTensor = instanceTag === "MlxJacclInstance";
      let layersLoaded = isTensor ? Infinity : 0;
      let totalLayers = 0;
      for (const rid of runnerIds) {
        const r = runnersData[rid];
        if (!r) continue;
        const [kind, payload] = getTagged(r);
        if (
          kind === "RunnerLoading" &&
          payload &&
          typeof payload === "object"
        ) {
          const p = payload as { layersLoaded?: number; totalLayers?: number };
          if (isTensor) {
            layersLoaded = Math.min(layersLoaded, p.layersLoaded ?? 0);
            totalLayers = Math.max(totalLayers, p.totalLayers ?? 0);
          } else {
            layersLoaded += p.layersLoaded ?? 0;
            totalLayers += p.totalLayers ?? 0;
          }
        }
      }
      if (isTensor && layersLoaded === Infinity) layersLoaded = 0;
      return {
        statusText: "LOADING",
        statusClass: "starting",
        layersLoaded,
        totalLayers,
      };
    }
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

  // Compute instance statuses by modelId for the model picker
  const modelInstanceStatuses = $derived.by(() => {
    const result: Record<string, { status: string; statusClass: string }> = {};
    for (const [id, inst] of Object.entries(instanceData)) {
      const modelId = getInstanceModelId(inst);
      if (!modelId || modelId === "Unknown" || modelId === "Unknown Model")
        continue;
      const dlStatus = getInstanceDownloadStatus(id, inst);
      const statusText = dlStatus.statusText;
      let statusClass = "inactive";
      if (
        statusText === "READY" ||
        statusText === "RUNNING" ||
        statusText === "LOADED"
      ) {
        statusClass = "ready";
      } else if (statusText === "DOWNLOADING") {
        statusClass = "downloading";
      } else if (statusText === "LOADING" || statusText === "WARMING UP") {
        statusClass = "loading";
      }
      // Keep the best status per modelId (ready > loading > downloading > other)
      const existing = result[modelId];
      if (existing) {
        const rank = (c: string) =>
          c === "ready" ? 3 : c === "loading" ? 2 : c === "downloading" ? 1 : 0;
        if (rank(statusClass) <= rank(existing.statusClass)) continue;
      }
      result[modelId] = { status: statusText, statusClass };
    }
    return result;
  });

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
    chatLaunchState = "idle";
    pendingChatModelId = null;
    selectedChatCategory = null;
    pendingAutoMessage = null;
    userForcedIdle = true;
    setSelectedChatModel("");
    createConversation();
  }

  function handleGoHome() {
    chatLaunchState = "idle";
    pendingChatModelId = null;
    selectedChatCategory = null;
    pendingAutoMessage = null;
    userForcedIdle = true;
    // Restore chat model from the sidebar preview selection so both selectors stay in sync
    setSelectedChatModel(selectedModelId ?? "");
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

  // ── Instance status transition toasts ──
  // Track previous statuses so we can detect meaningful transitions and fire toasts.
  let previousInstanceStatuses: Record<string, string> = {};

  $effect(() => {
    const currentStatuses: Record<string, string> = {};
    for (const [id, inst] of Object.entries(instanceData)) {
      const dlStatus = getInstanceDownloadStatus(id, inst);
      currentStatuses[id] = dlStatus.statusText;
    }

    const prev = previousInstanceStatuses;

    // Only fire toasts if we had a previous snapshot (skip the very first poll)
    if (Object.keys(prev).length > 0) {
      for (const [id, currentStatus] of Object.entries(currentStatuses)) {
        const prevStatus = prev[id];
        if (!prevStatus || prevStatus === currentStatus) continue;

        const modelId = getInstanceModelId(instanceData[id]);
        const shortName = modelId
          ? (modelId.split("/").pop() ?? modelId)
          : id.slice(0, 8);

        // Downloading -> non-downloading, non-failure = download complete
        if (
          prevStatus === "DOWNLOADING" &&
          currentStatus !== "DOWNLOADING" &&
          currentStatus !== "FAILED"
        ) {
          addToast({
            type: "success",
            message: `Download complete: ${shortName}`,
          });
        }

        // Loading/Warming Up -> Ready/Loaded/Running = model ready
        if (
          (prevStatus === "LOADING" || prevStatus === "WARMING UP") &&
          (currentStatus === "READY" ||
            currentStatus === "LOADED" ||
            currentStatus === "RUNNING")
        ) {
          addToast({ type: "success", message: `Model ready: ${shortName}` });
        }

        // Any -> Failed
        if (prevStatus !== "FAILED" && currentStatus === "FAILED") {
          addToast({ type: "error", message: `Model failed: ${shortName}` });
        }

        // Any -> Shutdown
        if (prevStatus !== "SHUTDOWN" && currentStatus === "SHUTDOWN") {
          addToast({ type: "info", message: `Model shut down: ${shortName}` });
        }
      }
    }

    previousInstanceStatuses = currentStatuses;
  });

  // ── Connection status toasts ──
  let previousConnectionStatus: boolean | null = null;

  $effect(() => {
    const connected = isConnected();
    if (previousConnectionStatus !== null) {
      if (previousConnectionStatus && !connected) {
        addToast({
          type: "warning",
          message: "Connection to server lost",
          persistent: true,
        });
      } else if (!previousConnectionStatus && connected) {
        dismissByMessage("Connection to server lost");
        addToast({ type: "success", message: "Connection restored" });
      }
    }
    previousConnectionStatus = connected;
  });

  const suggestedPrompts = [
    "Write a poem about the ocean",
    "Explain quantum computing simply",
    "Help me debug my code",
    "Tell me a creative story",
  ];

  // ── Seamless chat: launch models from chat view ──
  type ChatLaunchState =
    | "idle"
    | "launching"
    | "downloading"
    | "loading"
    | "ready";
  let chatLaunchState = $state<ChatLaunchState>("idle");
  let pendingChatModelId = $state<string | null>(null);
  let selectedChatCategory = $state<string | null>(null);
  // Guard: when true, the restore $effect must not override chatLaunchState.
  // Set by handleNewChat/handleGoHome; cleared when the user picks a model.
  let userForcedIdle = $state(false);

  // Restore chat launch state when switching conversations
  $effect(() => {
    const currentModel = selectedChatModel();
    // When the user explicitly requested the model selector (New Chat / Go Home),
    // skip restoring state so the selector stays visible.
    if (userForcedIdle) return;
    if (!currentModel) {
      if (chatStarted && chatLaunchState !== "idle") {
        chatLaunchState = "idle";
        pendingChatModelId = null;
        selectedChatCategory = null;
      }
      return;
    }

    // Model is already running — no progress to show
    if (hasRunningInstance(currentModel)) {
      if (chatLaunchState !== "ready") {
        chatLaunchState = "ready";
      }
      pendingChatModelId = currentModel;
      return;
    }

    // Model is downloading
    const dlStatus = getModelDownloadStatus(currentModel);
    if (dlStatus.isDownloading) {
      chatLaunchState = "downloading";
      pendingChatModelId = currentModel;
      return;
    }

    // Model is loading or in another pre-ready state
    for (const [, inst] of Object.entries(instanceData)) {
      if (getInstanceModelId(inst) !== currentModel) continue;
      const status = deriveInstanceStatus(inst);
      if (status.statusText === "LOADING") {
        chatLaunchState = "loading";
        pendingChatModelId = currentModel;
        return;
      }
      if (
        status.statusText === "WARMING UP" ||
        status.statusText === "WAITING" ||
        status.statusText === "INITIALIZING" ||
        status.statusText === "PREPARING"
      ) {
        chatLaunchState = "launching";
        pendingChatModelId = currentModel;
        return;
      }
    }

    // Fallthrough: model exists but has no active instance/download/loading state
    chatLaunchState = "idle";
    pendingChatModelId = null;
    selectedChatCategory = null;
  });

  // Suggested prompts per category
  const categorySuggestedPrompts: Record<string, string[]> = {
    coding: [
      "Write a Snake game in Python",
      "Build a REST API with FastAPI",
      "Explain how async/await works",
      "Help me write unit tests for my code",
    ],
    writing: [
      "Write a short story about time travel",
      "Draft a professional email to a client",
      "Create a haiku about the ocean",
      "Summarize the key ideas of stoicism",
    ],
    agentic: [
      "Plan a weekend trip to Tokyo",
      "Research and compare React vs Svelte",
      "Create a step-by-step guide to learn ML",
      "Analyze the pros and cons of remote work",
    ],
    biggest: [
      "Explain quantum computing simply",
      "Help me brainstorm startup ideas",
      "What are the key differences between TCP and UDP?",
      "Write a Python script to analyze a CSV file",
    ],
    auto: [
      "Explain quantum computing simply",
      "Help me brainstorm ideas for a side project",
      "Write a Python function to sort a list",
      "What makes a great technical interview?",
    ],
  };

  // Cluster label for ChatModelSelector header
  const chatClusterLabel = $derived.by(() => {
    if (!data) return "your Mac";
    const nodes = Object.values(data.nodes);
    if (nodes.length === 0) return "your Mac";
    if (nodes.length === 1) {
      const node = nodes[0];
      const name = node.system_info?.model_id || "your Mac";
      const totalMem =
        node.macmon_info?.memory?.ram_total ?? node.system_info?.memory ?? 0;
      const memGB = Math.round(totalMem / (1024 * 1024 * 1024));
      return `${name} ${memGB}GB`;
    }
    const totalMemGB = Math.round(clusterTotalMemoryGB());
    return `cluster ${totalMemGB}GB`;
  });

  // Check if a model already has a running instance
  function hasRunningInstance(modelId: string): boolean {
    for (const [, inst] of Object.entries(instanceData)) {
      const id = getInstanceModelId(inst);
      if (id === modelId) {
        const status = deriveInstanceStatus(inst);
        if (
          status.statusText === "READY" ||
          status.statusText === "LOADED" ||
          status.statusText === "RUNNING"
        ) {
          return true;
        }
      }
    }
    return false;
  }

  function hasExistingInstance(modelId: string): boolean {
    for (const [, inst] of Object.entries(instanceData)) {
      if (getInstanceModelId(inst) === modelId) return true;
    }
    return false;
  }

  // Pick optimal placement from previews (frontend logic)
  // Rules: 1-node → Pipeline/Ring, multi-node with RDMA → Tensor/Jaccl (most nodes),
  //         multi-node without RDMA → 1-node Pipeline/Ring
  function pickOptimalPlacement(
    previews: PlacementPreview[],
  ): PlacementPreview | null {
    const valid = previews.filter((p) => p.instance && !p.error);

    // Check if any valid placement uses multiple nodes (indicates multi-node cluster)
    const hasMultiNode = valid.some((p) => getPreviewNodeCount(p) > 1);

    if (hasMultiNode) {
      // Multi-node with RDMA: prefer Jaccl + Tensor with most nodes (fastest TPS)
      const jacclTensor = valid
        .filter(
          (p) => p.instance_meta === "MlxJaccl" && p.sharding === "Tensor",
        )
        .sort((a, b) => getPreviewNodeCount(b) - getPreviewNodeCount(a));
      if (jacclTensor.length > 0) return jacclTensor[0];

      // Multi-node without RDMA: fall back to single-node Pipeline/Ring
      const singlePipeline = valid.filter(
        (p) =>
          p.instance_meta === "MlxRing" &&
          p.sharding === "Pipeline" &&
          getPreviewNodeCount(p) === 1,
      );
      if (singlePipeline.length > 0) return singlePipeline[0];
    }

    // Single node (or final fallback): Pipeline/Ring with fewest nodes
    const ringPipeline = valid
      .filter((p) => p.instance_meta === "MlxRing" && p.sharding === "Pipeline")
      .sort((a, b) => getPreviewNodeCount(a) - getPreviewNodeCount(b));
    if (ringPipeline.length > 0) return ringPipeline[0];

    // Last resort: any valid placement, fewest nodes
    return (
      valid.sort(
        (a, b) => getPreviewNodeCount(a) - getPreviewNodeCount(b),
      )[0] ?? null
    );
  }

  // Launch a model for seamless chat
  async function launchModelForChat(
    modelId: string,
    category: string,
    skipCreate = false,
  ) {
    userForcedIdle = false;
    pendingChatModelId = modelId;
    selectedChatCategory = category;

    // Check if already running — skip straight to chat
    if (hasRunningInstance(modelId)) {
      setSelectedChatModel(modelId);
      if (!skipCreate) createConversation();
      chatLaunchState = "ready";
      return;
    }

    // Already has an instance (downloading/loading) — attach to its progress
    if (hasExistingInstance(modelId)) {
      setSelectedChatModel(modelId);
      pendingChatModelId = modelId;
      if (!skipCreate) createConversation();
      const dlStatus = getModelDownloadStatus(modelId);
      if (dlStatus.isDownloading) {
        chatLaunchState = "downloading";
      } else {
        chatLaunchState = "launching";
      }
      return;
    }

    chatLaunchState = "launching";

    try {
      // Fetch placement previews
      const res = await fetch(
        `/instance/previews?model_id=${encodeURIComponent(modelId)}`,
      );
      if (!res.ok) {
        addToast({
          type: "error",
          message: `Failed to get placements: ${await res.text()}`,
        });
        chatLaunchState = "idle";
        return;
      }
      const data: { previews: PlacementPreview[] } = await res.json();
      const placement = pickOptimalPlacement(data.previews);
      if (!placement) {
        addToast({
          type: "error",
          message: "No valid placement found for this model",
        });
        chatLaunchState = "idle";
        return;
      }

      // Launch the instance
      const launchRes = await fetch("/instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance: placement.instance }),
      });
      if (!launchRes.ok) {
        addToast({
          type: "error",
          message: `Failed to launch: ${await launchRes.text()}`,
        });
        chatLaunchState = "idle";
        return;
      }

      setSelectedChatModel(modelId);
      recordRecentLaunch(modelId);
      if (!skipCreate) createConversation();
      chatLaunchState = "downloading";
    } catch (error) {
      addToast({ type: "error", message: `Network error: ${error}` });
      chatLaunchState = "idle";
    }
  }

  // Handle auto-send: user typed without selecting a model
  async function handleAutoSend(
    content: string,
    files?: {
      id: string;
      name: string;
      type: string;
      textContent?: string;
      preview?: string;
    }[],
  ) {
    // Clear forced-idle so restore effect resumes normal operation
    userForcedIdle = false;

    // Find the best already-running model by tier
    let bestRunning: { id: string; tierIndex: number } | null = null;
    for (const [, inst] of Object.entries(instanceData)) {
      const modelId = getInstanceModelId(inst);
      if (modelId === "Unknown" || modelId === "Unknown Model") continue;
      if (!hasRunningInstance(modelId)) continue;
      const info = models.find((m) => m.id === modelId);
      if (!info) continue;
      const tierIndex = getAutoTierIndex(info.base_model ?? "");
      if (!bestRunning || tierIndex < bestRunning.tierIndex) {
        bestRunning = { id: modelId, tierIndex };
      }
    }

    // Find the best auto model that fits in available memory
    const totalMem = availableMemoryGB();
    const modelInfos = models.map((m) => ({
      id: m.id,
      name: m.name ?? "",
      base_model: m.base_model ?? "",
      storage_size_megabytes: m.storage_size_megabytes ?? 0,
      capabilities: m.capabilities ?? [],
      family: m.family ?? "",
      quantization: m.quantization ?? "",
    }));
    const autoModel = pickAutoModel(modelInfos, totalMem);

    // Prefer running model unless auto-pick is a strictly better tier
    if (bestRunning) {
      const autoTier = autoModel
        ? getAutoTierIndex(autoModel.base_model)
        : Infinity;
      if (autoTier >= bestRunning.tierIndex) {
        // Running model is same or better tier — use it directly
        setSelectedChatModel(bestRunning.id);
        if (!chatStarted) createConversation();
        sendMessage(content, files);
        return;
      }
    }

    if (!autoModel) {
      addToast({
        type: "error",
        message: "No model fits in your available memory",
      });
      return;
    }

    // Check if the chosen auto model is already running
    if (hasRunningInstance(autoModel.id)) {
      setSelectedChatModel(autoModel.id);
      if (!chatStarted) createConversation();
      sendMessage(content, files);
      return;
    }

    // Already has an instance (downloading/loading) — attach to its progress
    if (hasExistingInstance(autoModel.id)) {
      selectedChatCategory = "auto";
      setSelectedChatModel(autoModel.id);
      pendingChatModelId = autoModel.id;
      if (!chatStarted) createConversation();
      pendingAutoMessage = { content, files };
      const dlStatus = getModelDownloadStatus(autoModel.id);
      if (dlStatus.isDownloading) {
        chatLaunchState = "downloading";
      } else {
        chatLaunchState = "launching";
      }
      return;
    }

    // Need to launch first, then send
    selectedChatCategory = "auto";
    pendingChatModelId = autoModel.id;
    chatLaunchState = "launching";

    try {
      const res = await fetch(
        `/instance/previews?model_id=${encodeURIComponent(autoModel.id)}`,
      );
      if (!res.ok) {
        addToast({
          type: "error",
          message: `Failed to get placements: ${await res.text()}`,
        });
        chatLaunchState = "idle";
        return;
      }
      const data: { previews: PlacementPreview[] } = await res.json();
      const placement = pickOptimalPlacement(data.previews);
      if (!placement) {
        addToast({ type: "error", message: "No valid placement found" });
        chatLaunchState = "idle";
        return;
      }

      const launchRes = await fetch("/instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance: placement.instance }),
      });
      if (!launchRes.ok) {
        addToast({
          type: "error",
          message: `Failed to launch: ${await launchRes.text()}`,
        });
        chatLaunchState = "idle";
        return;
      }

      setSelectedChatModel(autoModel.id);
      recordRecentLaunch(autoModel.id);
      if (!chatStarted) createConversation();
      chatLaunchState = "downloading";

      // Queue the message to send once model is ready
      pendingAutoMessage = { content, files };
    } catch (error) {
      addToast({ type: "error", message: `Network error: ${error}` });
      chatLaunchState = "idle";
    }
  }

  // Pending message to send after auto-launch completes
  let pendingAutoMessage = $state<{
    content: string;
    files?: {
      id: string;
      name: string;
      type: string;
      textContent?: string;
      preview?: string;
    }[];
  } | null>(null);

  // Best running model by tier (for auto-pick display)
  const bestRunningModelId = $derived.by(() => {
    let best: { id: string; tierIndex: number } | null = null;
    for (const [, inst] of Object.entries(instanceData)) {
      const modelId = getInstanceModelId(inst);
      if (modelId === "Unknown" || modelId === "Unknown Model") continue;
      if (!hasRunningInstance(modelId)) continue;
      const info = models.find((m) => m.id === modelId);
      if (!info) continue;
      const tierIndex = getAutoTierIndex(info.base_model ?? "");
      if (!best || tierIndex < best.tierIndex) {
        best = { id: modelId, tierIndex };
      }
    }
    return best?.id ?? null;
  });

  // Track chat launch progress (download + loading)
  const chatLaunchDownload = $derived.by(() => {
    if (
      !pendingChatModelId ||
      (chatLaunchState !== "downloading" && chatLaunchState !== "launching")
    )
      return null;
    const status = getModelDownloadStatus(pendingChatModelId);
    if (status.isDownloading) return status.progress;
    return null;
  });

  const chatLaunchLoadProgress = $derived.by(() => {
    if (
      !pendingChatModelId ||
      chatLaunchState === "idle" ||
      chatLaunchState === "ready"
    )
      return null;
    let layersLoaded = 0,
      totalLayers = 0;
    for (const [, inst] of Object.entries(instanceData)) {
      if (getInstanceModelId(inst) !== pendingChatModelId) continue;
      const status = deriveInstanceStatus(inst);
      if (
        status.statusText === "LOADING" &&
        status.totalLayers &&
        status.totalLayers > 0
      ) {
        layersLoaded += status.layersLoaded ?? 0;
        totalLayers += status.totalLayers;
      }
    }
    if (totalLayers === 0) return null;
    return {
      layersLoaded,
      totalLayers,
      percentage: (layersLoaded / totalLayers) * 100,
    };
  });

  // Auto-advance chat launch state based on instance status
  $effect(() => {
    if (!pendingChatModelId || chatLaunchState === "idle") return;

    // Check if model is now ready
    if (hasRunningInstance(pendingChatModelId)) {
      chatLaunchState = "ready";
      // Send pending auto message if any
      if (pendingAutoMessage) {
        const msg = pendingAutoMessage;
        pendingAutoMessage = null;
        sendMessage(msg.content, msg.files);
      }
      return;
    }

    // If already ready (set by restore effect), don't downgrade state
    if (chatLaunchState === "ready") return;

    // Check if currently loading
    if (chatLaunchLoadProgress) {
      chatLaunchState = "loading";
      return;
    }

    // Check if currently downloading
    if (chatLaunchDownload) {
      chatLaunchState = "downloading";
    }
  });

  // Check if any instance is running (for showing model selector vs chat)
  const hasAnyRunningInstance = $derived(() => {
    for (const [, inst] of Object.entries(instanceData)) {
      const status = deriveInstanceStatus(inst);
      if (
        status.statusText === "READY" ||
        status.statusText === "LOADED" ||
        status.statusText === "RUNNING"
      ) {
        return true;
      }
    }
    return false;
  });

  // Handle model selection from ChatModelSelector
  function handleChatModelSelect(modelId: string, category: string) {
    launchModelForChat(modelId, category);
  }

  // Handle "+ Add Model" from ChatModelSelector
  function handleChatAddModel() {
    modelPickerContext = "chat";
    isModelPickerOpen = true;
  }

  // Track which context opened the model picker (dashboard launch vs chat selection)
  let modelPickerContext = $state<"dashboard" | "chat">("dashboard");

  // Open the model picker from a chat context (e.g. clicking the model button in ChatForm)
  function openChatModelPicker() {
    modelPickerContext = "chat";
    isModelPickerOpen = true;
  }

  // Handle model selection from the picker when opened from chat context
  function handleChatPickerSelect(modelId: string) {
    setSelectedChatModel(modelId);
    selectPreviewModel(modelId);
    userForcedIdle = false;
    isModelPickerOpen = false;
  }

  // Unified send handler: sends if model running, auto-launches if not
  function handleChatSend(
    content: string,
    files?: {
      id: string;
      name: string;
      type: string;
      textContent?: string;
      preview?: string;
    }[],
  ) {
    const model = selectedChatModel();

    // Model is selected and running — send directly
    if (model && hasRunningInstance(model)) {
      chatLaunchState = "ready";
      sendMessage(content, files, null);
      return;
    }

    // Model is selected but NOT running — launch it, queue the message
    if (model) {
      pendingAutoMessage = { content, files };
      userForcedIdle = false;
      launchModelForChat(model, "picker", messages().length > 0);
      return;
    }

    // No model selected — fall through to auto-pick
    handleAutoSend(content, files);
  }

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
  {#if tbBridgeCycles.length > 0 || macosVersionMismatch || (tb5WithoutRdma && !tb5InfoDismissed) || (macStudioEn2RdmaWarning && !macStudioEn2Dismissed)}
    <div class="absolute top-4 left-4 flex flex-col gap-2 z-40">
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
        <div class="group relative" role="status">
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
              RDMA NOT ENABLED
            </span>
            <button
              type="button"
              onclick={() => (tb5InfoDismissed = true)}
              class="ml-1 text-yellow-300/60 hover:text-yellow-200 transition-colors cursor-pointer"
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
            class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-yellow-500/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
          >
            <p class="text-xs text-white/80 mb-2">
              Thunderbolt 5 hardware detected on multiple nodes. Enable RDMA for
              significantly faster inter-node communication.
            </p>
            <p class="text-xs text-white/60 mb-1.5">
              <span class="text-yellow-300">To enable:</span>
            </p>
            <ol
              class="text-xs text-white/60 list-decimal list-inside space-y-0.5 mb-1.5"
            >
              <li>Connect nodes with TB5 cables</li>
              <li>Boot to Recovery (hold power 10s → Options)</li>
              <li>
                Run
                <code class="text-yellow-300 bg-yellow-400/10 px-1 rounded"
                  >rdma_ctl enable</code
                >
              </li>
              <li>Reboot</li>
            </ol>
            <p class="text-xs text-white/40">
              Requires macOS 26.2+, TB5 cables, and matching OS versions.
            </p>
          </div>
        </div>
      {/if}

      {#if macStudioEn2RdmaWarning && !macStudioEn2Dismissed}
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
              RDMA INCOMPATIBLE PORT
            </span>
            <button
              type="button"
              onclick={() => (macStudioEn2Dismissed = true)}
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

          <!-- Expanded tooltip on hover -->
          <div
            class="absolute top-full left-0 mt-2 w-96 p-4 rounded border border-red-500/30 bg-[#1a1a1a]/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
          >
            <p class="text-xs text-white/80 mb-3">
              The Thunderbolt 5 port next to the Ethernet port on Mac Studio
              does
              <span class="text-red-400 font-semibold">not support RDMA</span>.
              Move the cable to one of the other three TB5 ports.
            </p>

            <div class="text-xs text-white/60 mb-3">
              <span class="text-red-300">Affected:</span>
              {#each macStudioEn2RdmaWarning as conn}
                <div class="ml-2 mt-0.5">
                  <span class="text-white/80">{conn.nodeName}</span>
                  <span class="text-white/30">&rarr;</span>
                  <span class="text-white/60">{conn.peerNodeName}</span>
                  <span class="text-white/30 ml-1">(en2)</span>
                </div>
              {/each}
            </div>

            <!-- Mac Studio back panel illustration -->
            <div class="bg-black/40 rounded p-3 mb-3">
              <p
                class="text-[10px] font-mono text-white/30 uppercase tracking-wider mb-2"
              >
                Mac Studio — Rear Panel
              </p>
              <svg
                viewBox="0 0 320 72"
                class="w-full"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect
                  x="1"
                  y="1"
                  width="318"
                  height="70"
                  rx="6"
                  ry="6"
                  fill="none"
                  stroke="rgba(255,255,255,0.12)"
                  stroke-width="1"
                />
                <!-- TB5 port 1 -->
                <rect
                  x="24"
                  y="22"
                  width="28"
                  height="14"
                  rx="4"
                  fill="none"
                  stroke="rgba(255,255,255,0.3)"
                  stroke-width="1"
                />
                <text
                  x="38"
                  y="52"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.25)"
                  style="font-size:7px;font-family:ui-monospace,monospace;"
                  >TB5</text
                >
                <!-- TB5 port 2 -->
                <rect
                  x="62"
                  y="22"
                  width="28"
                  height="14"
                  rx="4"
                  fill="none"
                  stroke="rgba(255,255,255,0.3)"
                  stroke-width="1"
                />
                <text
                  x="76"
                  y="52"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.25)"
                  style="font-size:7px;font-family:ui-monospace,monospace;"
                  >TB5</text
                >
                <!-- TB5 port 3 -->
                <rect
                  x="100"
                  y="22"
                  width="28"
                  height="14"
                  rx="4"
                  fill="none"
                  stroke="rgba(255,255,255,0.3)"
                  stroke-width="1"
                />
                <text
                  x="114"
                  y="52"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.25)"
                  style="font-size:7px;font-family:ui-monospace,monospace;"
                  >TB5</text
                >
                <!-- TB5 port 4: INCOMPATIBLE (en2) — equally spaced with ports 1-3 -->
                <rect
                  x="138"
                  y="22"
                  width="28"
                  height="14"
                  rx="4"
                  fill="rgba(239,68,68,0.1)"
                  stroke="rgba(239,68,68,0.7)"
                  stroke-width="1.5"
                />
                <line
                  x1="142"
                  y1="25"
                  x2="162"
                  y2="33"
                  stroke="rgba(239,68,68,0.8)"
                  stroke-width="1.5"
                  stroke-linecap="round"
                />
                <line
                  x1="162"
                  y1="25"
                  x2="142"
                  y2="33"
                  stroke="rgba(239,68,68,0.8)"
                  stroke-width="1.5"
                  stroke-linecap="round"
                />
                <text
                  x="152"
                  y="52"
                  text-anchor="middle"
                  fill="rgba(239,68,68,0.6)"
                  style="font-size:7px;font-family:ui-monospace,monospace;font-weight:600;"
                  >en2</text
                >
                <!-- Ethernet port -->
                <rect
                  x="196"
                  y="19"
                  width="24"
                  height="20"
                  rx="2"
                  fill="none"
                  stroke="rgba(255,255,255,0.2)"
                  stroke-width="1"
                />
                <rect
                  x="200"
                  y="23"
                  width="16"
                  height="12"
                  rx="1"
                  fill="none"
                  stroke="rgba(255,255,255,0.12)"
                  stroke-width="0.75"
                />
                <text
                  x="208"
                  y="52"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.25)"
                  style="font-size:7px;font-family:ui-monospace,monospace;"
                  >ETH</text
                >
                <!-- Green checkmarks on working ports -->
                <circle
                  cx="38"
                  cy="62"
                  r="3"
                  fill="none"
                  stroke="rgba(74,222,128,0.5)"
                  stroke-width="0.75"
                />
                <circle
                  cx="76"
                  cy="62"
                  r="3"
                  fill="none"
                  stroke="rgba(74,222,128,0.5)"
                  stroke-width="0.75"
                />
                <circle
                  cx="114"
                  cy="62"
                  r="3"
                  fill="none"
                  stroke="rgba(74,222,128,0.5)"
                  stroke-width="0.75"
                />
              </svg>
            </div>

            <p class="text-xs text-white/50">
              <span class="text-green-400">Fix:</span> Move the Thunderbolt cable
              to any of the three leftmost ports (all support RDMA).
            </p>
          </div>
        </div>
      {/if}
    </div>
  {/if}
{/snippet}

{#snippet clusterWarningsCompact()}
  {#if tbBridgeCycles.length > 0 || macosVersionMismatch || (tb5WithoutRdma && !tb5InfoDismissed) || (macStudioEn2RdmaWarning && !macStudioEn2Dismissed)}
    <div class="absolute top-2 left-2 flex flex-col gap-1">
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
          class="flex items-center gap-1.5 px-2 py-1 rounded border border-yellow-500/50 bg-yellow-500/10 backdrop-blur-sm"
          title="Thunderbolt 5 detected — RDMA not enabled. Enable for faster inter-node communication."
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
            >RDMA NOT ENABLED</span
          >
        </div>
      {/if}
      {#if macStudioEn2RdmaWarning && !macStudioEn2Dismissed}
        <div
          class="flex items-center gap-1.5 px-2 py-1 rounded border border-red-500/50 bg-red-500/10 backdrop-blur-sm"
          title="Mac Studio RDMA incompatible port (en2) — move cable to another TB5 port"
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
          <span class="text-[10px] font-mono text-red-200">BAD RDMA PORT</span>
        </div>
      {/if}
    </div>
  {/if}
{/snippet}

<!-- Global event listeners for slider dragging + onboarding keyboard nav -->
<svelte:window
  onmousemove={handleSliderMouseMove}
  onmouseup={handleSliderMouseUp}
  ontouchmove={handleSliderTouchMove}
  ontouchend={handleSliderTouchEnd}
  onkeydown={(e) => {
    if (!showOnboardingOverlay || stepTransitioning) return;
    if (e.key === "ArrowRight" || e.key === " " || e.key === "Enter") {
      if (onboardingStep >= 1 && onboardingStep <= 4 && showContinueButton) {
        e.preventDefault();
        advanceStep(onboardingStep < 4 ? onboardingStep + 1 : 6);
      }
    }
  }}
/>

<div
  class="relative h-screen w-full flex flex-col bg-exo-dark-gray overflow-hidden"
>
  <!-- Scanline overlay -->
  <!-- Scanline overlay -->
  <div
    class="fixed inset-0 pointer-events-none z-50 scanlines"
    style="transition: opacity 0.5s ease; opacity: {showOnboardingOverlay
      ? 0
      : 0.2};"
  ></div>

  <!-- Shooting Stars Background -->
  <div
    class="shooting-stars"
    style="transition: opacity 0.5s ease; opacity: {showOnboardingOverlay
      ? 0.4
      : 1};"
  >
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

  {#if showOnboardingOverlay}
    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- FULL-SCREEN ONBOARDING WIZARD (overlay)                -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <div
      class="absolute inset-0 flex items-center justify-center z-30 bg-exo-black"
      style="transition: opacity 0.45s cubic-bezier(0.4, 0, 0.2, 1); opacity: {onboardingFadingOut
        ? 0
        : 1};"
    >
      {#if onboardingStep >= 1 && onboardingStep <= 4}
        <!-- Steps 1-4: Cinematic SVG animation story -->
        <div
          class="flex flex-col items-center w-full max-w-3xl px-8"
          style="transition: opacity 0.6s cubic-bezier(0.4, 0, 0.2, 1), transform 0.6s cubic-bezier(0.4, 0, 0.2, 1); opacity: {stepTransitioning
            ? 0
            : 1}; transform: scale({stepTransitioning ? 0.98 : 1});"
        >
          <!-- Logo + Step title -->
          <div class="text-center mb-8">
            <!-- Logo — smoothly shrinks away when leaving step 1 -->
            <div
              style="opacity: {$logoOpacity}; max-height: {$logoOpacity *
                80}px; overflow: hidden; transition: max-height 0.6s cubic-bezier(0.4, 0, 0.2, 1);"
            >
              <img src="/exo-logo.png" alt="exo" class="w-36 mx-auto mb-10" />
            </div>

            <!-- Title — single element, text updates instantly -->
            <h1
              class="text-2xl font-light text-white/90 tracking-wide"
              style="opacity: {$titleOpacity}; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; letter-spacing: 0.02em;"
            >
              {onboardingStep === 1
                ? "EXO connects all your devices into an AI supercomputer."
                : stepTitle}
            </h1>

            <!-- Subtitle — uses tweened opacity, reserves space to prevent layout shift -->
            <p
              class="text-sm mt-2 text-white/40 max-w-md mx-auto"
              style="opacity: {$subtitleOpacity}; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; font-weight: 300; min-height: 1.5em;"
            >
              {#if onboardingStep === 2}
                &nbsp;
              {:else if onboardingStep === 3}
                The model is automatically distributed. Each device handles a
                piece.
              {:else if onboardingStep === 4}
                {stepTitle === "exo self-heals"
                  ? "exo automatically redistributes the model so inference continues without interruption."
                  : "Devices can leave anytime. Laptops close, machines restart."}
              {:else}
                &nbsp;
              {/if}
            </p>
          </div>

          <!-- Device display area -->
          <div class="relative w-full" style="height: 420px;">
            <!-- Device count label — fades in on step 1, fades out on step 2 -->
            <p
              class="absolute left-0 right-0 text-center text-lg text-white/50 font-light tracking-wide z-10"
              style="top: 20px; opacity: {$deviceCountOpacity}; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; pointer-events: none;"
            >
              Your EXO Network
            </p>

            <!-- Step 1: Real topology graph -->
            {#if onboardingStep <= 1 || $topologyOpacity > 0.01}
              <div
                class="absolute inset-0 flex items-center justify-center"
                style="opacity: {$topologyOpacity}; pointer-events: {onboardingStep <=
                1
                  ? 'none'
                  : 'none'};"
              >
                <TopologyGraph class="w-full h-full" />
              </div>
            {/if}

            <!-- Steps 2+: Tweened SVG canvas with device pair -->
            <svg
              viewBox="0 0 700 420"
              class="w-full h-full"
              xmlns="http://www.w3.org/2000/svg"
              style="position: relative;"
            >
              <!-- Device 1 (User's device) -->
              <g
                transform="translate({$device1X}, 210)"
                opacity={$device1Opacity}
                style="transition: opacity 0.6s ease;"
              >
                <DeviceIcon
                  deviceType={userDeviceInfo.deviceType}
                  cx={0}
                  cy={0}
                  size={110}
                  ramPercent={60}
                  uid="onb-d1"
                />
                <text
                  x="0"
                  y="-105"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.9)"
                  style="font-size: 15px; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; font-weight: 500; letter-spacing: 0.01em;"
                >
                  {userDeviceInfo.name}
                </text>
                <text
                  x="0"
                  y="105"
                  text-anchor="middle"
                  style="font-size: 14px; font-family: 'SF Mono', ui-monospace, monospace;"
                >
                  <tspan fill="rgba(255,215,0,0.9)"
                    >{userDeviceInfo.memoryGB}</tspan
                  ><tspan fill="rgba(255,255,255,0.4)">{" "}GB</tspan>
                </text>
              </g>

              <!-- Device 2 (Mac Studio — simulated) -->
              <g
                transform="translate({$device2X}, 210)"
                opacity={$device2Opacity}
                style="transition: opacity 0.6s ease;"
              >
                <!-- Dashed outline to indicate simulated device -->
                <rect
                  x={(-110 * 1.25) / 2 - 6}
                  y={(-110 * 0.85) / 2 - 6}
                  width={110 * 1.25 + 12}
                  height={110 * 0.85 + 12}
                  rx="6"
                  fill="none"
                  stroke="rgba(255,255,255,0.12)"
                  stroke-dasharray="4,4"
                />
                <DeviceIcon
                  deviceType="mac studio"
                  cx={0}
                  cy={0}
                  size={110}
                  ramPercent={80}
                  uid="onb-d2"
                />
                <text
                  x="0"
                  y="-105"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.9)"
                  style="font-size: 15px; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; font-weight: 500; letter-spacing: 0.01em;"
                >
                  Mac Studio
                </text>
                <text
                  x="0"
                  y="105"
                  text-anchor="middle"
                  style="font-size: 14px; font-family: 'SF Mono', ui-monospace, monospace;"
                >
                  <tspan fill="rgba(255,215,0,0.9)">{SIMULATED_STUDIO_GB}</tspan
                  ><tspan fill="rgba(255,255,255,0.4)">{" "}GB</tspan>
                </text>
                <text
                  x="0"
                  y="120"
                  text-anchor="middle"
                  fill="rgba(255,255,255,0.2)"
                  style="font-size: 9px; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; font-style: italic;"
                >
                  (example)
                </text>
              </g>

              <!-- Connection line between devices -->
              <line
                x1={$device1X + 85}
                y1={210}
                x2={$device2X - 85}
                y2={210}
                stroke={$connectionIsRed > 0.5
                  ? "rgba(220,38,38,0.7)"
                  : "rgba(255,255,255,0.15)"}
                stroke-width="1.5"
                stroke-dasharray="6,6"
                opacity={$connectionOpacity}
                class={$connectionIsRed > 0.5
                  ? "onboarding-connection-line-red"
                  : "onboarding-connection-line"}
              />

              <!-- Disconnect X mark -->
              {#if $disconnectXOpacity > 0.01}
                <g
                  transform="translate({($device1X + $device2X) / 2}, 210)"
                  opacity={$disconnectXOpacity}
                >
                  <circle
                    r="18"
                    fill="rgba(220,38,38,0.1)"
                    stroke="rgba(220,38,38,0.6)"
                    stroke-width="1.5"
                  />
                  <line
                    x1="-8"
                    y1="-8"
                    x2="8"
                    y2="8"
                    stroke="rgba(220,38,38,0.8)"
                    stroke-width="2.5"
                    stroke-linecap="round"
                  />
                  <line
                    x1="8"
                    y1="-8"
                    x2="-8"
                    y2="8"
                    stroke="rgba(220,38,38,0.8)"
                    stroke-width="2.5"
                    stroke-linecap="round"
                  />
                </g>
              {/if}

              <!-- Combined memory label -->
              <text
                x={($device1X + $device2X) / 2}
                y={130}
                text-anchor="middle"
                fill="rgba(255,215,0,0.7)"
                style="font-size: 14px; font-family: 'SF Mono', ui-monospace, monospace; font-weight: 500; letter-spacing: 0.02em;"
                opacity={$combinedLabelOpacity}
              >
                {onboardingCombinedGB} GB combined
              </text>

              <!-- Step 2: Models unlocked — staggered slide-up + yellow glow -->
              {#if unlockedModels.length > 0 && $chipPhase > 0.01}
                {@const centerX = ($device1X + $device2X) / 2}
                {@const chipW = 140}
                {@const chipH = 30}
                {@const chipGap = 12}
                {@const totalW =
                  unlockedModels.length * chipW +
                  (unlockedModels.length - 1) * chipGap}
                {@const startX = centerX - totalW / 2}
                <!-- SVG filter for yellow glow -->
                <defs>
                  <filter
                    id="chip-glow"
                    x="-50%"
                    y="-50%"
                    width="200%"
                    height="200%"
                  >
                    <feGaussianBlur
                      in="SourceGraphic"
                      stdDeviation="4"
                      result="blur"
                    />
                    <feColorMatrix
                      in="blur"
                      type="matrix"
                      values="1 0.8 0 0 0  0.8 0.7 0 0 0  0 0 0 0 0  0 0 0 0.4 0"
                      result="glow"
                    />
                    <feMerge>
                      <feMergeNode in="glow" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <!-- Header slides up + fades with yellow tint -->
                {@const headerProgress = Math.min(1, $chipPhase)}
                {@const headerY = 332 + 12 * (1 - headerProgress)}
                {@const yellowR = 234}
                {@const yellowG = 179}
                {@const yellowB = 8}
                <text
                  x={centerX}
                  y={headerY}
                  text-anchor="middle"
                  dominant-baseline="middle"
                  fill="rgba({yellowR},{yellowG},{yellowB},{0.5 *
                    headerProgress})"
                  opacity={headerProgress}
                  style="font-size: 10px; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; font-weight: 500; letter-spacing: 0.1em;"
                >
                  NEW MODELS UNLOCKED
                </text>
                <!-- Model chips — staggered slide-up + scale + yellow highlight -->
                {#each unlockedModels as model, i}
                  {@const stagger = i * 0.6}
                  {@const progress = Math.max(
                    0,
                    Math.min(1, $chipPhase - stagger),
                  )}
                  {@const modelName = (
                    model.name ||
                    model.id.split("/").pop() ||
                    ""
                  ).slice(0, 18)}
                  {@const modelSize = Math.round(getModelSizeGB(model))}
                  {@const slideY = 16 * (1 - progress)}
                  {@const chipScale = 0.85 + 0.15 * progress}
                  <!-- Yellow highlight peaks at ~0.6 progress then settles to subtle -->
                  {@const highlightPeak =
                    progress < 0.6
                      ? progress / 0.6
                      : 1 - ((progress - 0.6) / 0.4) * 0.6}
                  {@const borderYellow = 0.15 + 0.35 * highlightPeak}
                  {@const fillYellow = 0.02 + 0.06 * highlightPeak}
                  {#if progress > 0}
                    <g
                      transform="translate({startX +
                        i * (chipW + chipGap) +
                        chipW / 2}, {358 + slideY}) scale({chipScale})"
                      opacity={progress}
                      filter={highlightPeak > 0.3 ? "url(#chip-glow)" : "none"}
                    >
                      <rect
                        x={-chipW / 2}
                        y={-chipH / 2}
                        width={chipW}
                        height={chipH}
                        rx="15"
                        fill="rgba({yellowR},{yellowG},{yellowB},{fillYellow})"
                        stroke="rgba({yellowR},{yellowG},{yellowB},{borderYellow})"
                        stroke-width="1"
                      />
                      <text
                        x="0"
                        y={modelSize ? -4 : 1}
                        text-anchor="middle"
                        dominant-baseline="middle"
                        fill="rgba(255,255,255,{0.5 + 0.3 * progress})"
                        style="font-size: 10px; font-family: 'SF Mono', ui-monospace, monospace; font-weight: 500;"
                      >
                        {modelName}
                      </text>
                      {#if modelSize}
                        <text
                          x="0"
                          y="8"
                          text-anchor="middle"
                          dominant-baseline="middle"
                          fill="rgba(255,255,255,{0.15 + 0.15 * progress})"
                          style="font-size: 8px; font-family: 'SF Mono', ui-monospace, monospace; font-weight: 400;"
                        >
                          {modelSize} GB
                        </text>
                      {/if}
                    </g>
                  {/if}
                {/each}
              {/if}

              <!-- Model block (unified or split) -->
              {#if $modelBlockOpacity > 0.01}
                {#if $modelSplitProgress < 0.05}
                  <!-- Unified model block — centers on device1 when device2 is hidden -->
                  {@const modelCenterX =
                    $device2Opacity > 0.3
                      ? ($device1X + $device2X) / 2
                      : $device1X}
                  <g
                    transform="translate({modelCenterX}, {$modelBlockY})"
                    opacity={$modelBlockOpacity}
                  >
                    <rect
                      x="-45"
                      y="-13"
                      width="90"
                      height="26"
                      rx="6"
                      fill="rgba(180,140,0,0.08)"
                      stroke="rgba(180,140,0,0.45)"
                      stroke-width="1.5"
                    />
                    <text
                      x="0"
                      y="5"
                      text-anchor="middle"
                      fill="rgba(220,180,40,0.9)"
                      style="font-size: 12px; font-family: -apple-system, system-ui, sans-serif; font-weight: 500;"
                    >
                      LLM
                    </text>
                  </g>
                {:else}
                  <!-- Split model halves flowing down to each device -->
                  {@const splitX =
                    $modelSplitProgress * (($device2X - $device1X) / 2)}
                  {@const centerX = ($device1X + $device2X) / 2}
                  {@const splitY = $modelBlockY + $modelSplitProgress * 80}

                  <!-- Left half -> Device 1 -->
                  <g
                    transform="translate({centerX - splitX}, {splitY})"
                    opacity={$modelBlockOpacity}
                  >
                    <rect
                      x="-45"
                      y="-13"
                      width="90"
                      height="26"
                      rx="6"
                      fill="rgba(180,140,0,0.08)"
                      stroke="rgba(180,140,0,0.35)"
                      stroke-width="1"
                    />
                    <text
                      x="0"
                      y="4"
                      text-anchor="middle"
                      fill="rgba(220,180,40,0.75)"
                      style="font-size: 11px; font-family: -apple-system, system-ui, sans-serif;"
                    >
                      Shard 1/2
                    </text>
                  </g>

                  <!-- Right half -> Device 2 -->
                  <g
                    transform="translate({centerX + splitX}, {splitY})"
                    opacity={$modelBlockOpacity * $device2Opacity}
                  >
                    <rect
                      x="-45"
                      y="-13"
                      width="90"
                      height="26"
                      rx="6"
                      fill="rgba(180,140,0,0.08)"
                      stroke="rgba(180,140,0,0.35)"
                      stroke-width="1"
                    />
                    <text
                      x="0"
                      y="4"
                      text-anchor="middle"
                      fill="rgba(220,180,40,0.75)"
                      style="font-size: 11px; font-family: -apple-system, system-ui, sans-serif;"
                    >
                      Shard 2/2
                    </text>
                  </g>
                {/if}
              {/if}
            </svg>
          </div>

          <!-- Continue button — smooth transition, only for steps 1 and 5 -->
          <div
            style="transition: opacity 0.4s ease, transform 0.4s cubic-bezier(0.4,0,0.2,1); opacity: {showContinueButton
              ? 1
              : 0}; transform: translateY({showContinueButton
              ? '0px'
              : '12px'}); pointer-events: {showContinueButton
              ? 'auto'
              : 'none'}; margin-top: 0.5rem;"
          >
            <button
              type="button"
              onclick={() =>
                advanceStep(onboardingStep < 4 ? onboardingStep + 1 : 6)}
              class="inline-flex items-center gap-2.5 px-10 py-3.5 bg-exo-yellow text-exo-black text-sm font-semibold rounded-full cursor-pointer"
              style="transition: transform 0.2s ease, box-shadow 0.3s ease, filter 0.2s ease; font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif; letter-spacing: 0.02em;"
              onmouseenter={(e) => {
                e.currentTarget.style.filter = "brightness(1.08)";
                e.currentTarget.style.boxShadow =
                  "0 0 30px rgba(255,215,0,0.2)";
              }}
              onmouseleave={(e) => {
                e.currentTarget.style.filter = "brightness(1)";
                e.currentTarget.style.boxShadow = "none";
              }}
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
        </div>
      {:else if onboardingStep === 6}
        <!-- Step 6: Choose a Model -->
        <div
          class="flex flex-col items-center w-full max-w-2xl px-8"
          style="opacity: 0; animation: onb-fade-in 0.5s ease forwards;"
        >
          <div class="text-center mb-8">
            <h1
              class="text-xl font-sans font-light text-white/90 mb-2 tracking-wide"
            >
              Choose a model
            </h1>
            <p class="text-sm font-sans text-white/40">
              Showing recommended models for your devices ({Math.round(
                clusterMemory().total / (1024 * 1024 * 1024),
              )} GB memory available).
            </p>
          </div>

          {#if onboardingError}
            <div
              class="w-full mb-6 px-4 py-3 rounded-lg border border-red-500/30 bg-red-500/10 text-sm font-mono text-red-400"
              in:fade={{ duration: 200 }}
            >
              {onboardingError}
            </div>
          {/if}

          {#if onboardingModels.length === 0}
            <div class="text-center py-8">
              <div class="text-sm text-white/40 font-sans animate-pulse">
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
                    ? 'border-white/10 bg-white/5 hover:border-exo-yellow/50 hover:bg-exo-yellow/5'
                    : 'border-white/10 bg-white/[0.02] hover:border-white/20 opacity-60'}"
                >
                  <div class="flex flex-col items-start gap-1 min-w-0">
                    <div class="flex items-center gap-2">
                      <span
                        class="text-sm font-sans font-medium text-white truncate"
                        >{model.name || model.id}</span
                      >
                      {#each tags as tag}
                        <span
                          class="text-[10px] font-sans font-medium px-1.5 py-0.5 rounded-full bg-exo-yellow/10 text-exo-yellow/80"
                          >{tag}</span
                        >
                      {/each}
                    </div>
                    <span class="text-xs font-mono text-white/40 truncate"
                      >{model.id}</span
                    >
                  </div>
                  <div class="flex items-center gap-3 flex-shrink-0">
                    <span class="text-xs font-mono text-white/50"
                      >{sizeGB >= 1 ? sizeGB.toFixed(0) : sizeGB.toFixed(1)} GB</span
                    >
                    <svg
                      class="w-4 h-4 text-white/40"
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
              modelPickerContext = "dashboard";
              isModelPickerOpen = true;
            }}
            class="text-sm font-sans text-white/40 hover:text-exo-yellow transition-colors cursor-pointer underline underline-offset-4 decoration-white/20 hover:decoration-exo-yellow/50"
          >
            Browse all models
          </button>
        </div>
      {:else if onboardingStep === 7}
        <!-- Step 7: Downloading -->
        <div
          class="text-center max-w-lg px-8"
          style="opacity: 0; animation: onb-fade-in 0.5s ease forwards;"
        >
          <div class="mb-8">
            <h1
              class="text-xl font-sans font-light text-white/90 mb-2 tracking-wide"
            >
              Downloading
            </h1>
            {#if onboardingModelId}
              <p class="text-sm text-white/40 font-sans">
                {onboardingModelId.split("/").pop() ?? onboardingModelId}
              </p>
            {/if}
          </div>

          {#if onboardingDownloadProgress}
            <div class="w-full max-w-md mx-auto space-y-4">
              <div
                class="relative h-2 bg-white/10 rounded-full overflow-hidden"
              >
                <div
                  class="absolute inset-y-0 left-0 bg-gradient-to-r from-exo-yellow to-exo-yellow-darker rounded-full transition-all duration-500"
                  style="width: {onboardingDownloadProgress.percentage}%"
                ></div>
              </div>
              <div class="flex justify-between text-xs font-mono text-white/50">
                <span>{onboardingDownloadProgress.percentage.toFixed(1)}%</span>
                <span
                  >{formatBytes(onboardingDownloadProgress.downloadedBytes)} /
                  {formatBytes(onboardingDownloadProgress.totalBytes)}</span
                >
              </div>
              <div class="flex justify-between text-xs font-mono text-white/40">
                <span>{formatSpeed(onboardingDownloadProgress.speed)}</span>
                <span>ETA: {formatEta(onboardingDownloadProgress.etaMs)}</span>
              </div>
            </div>
          {:else}
            <div class="w-full max-w-md mx-auto">
              <div
                class="relative h-2 bg-white/10 rounded-full overflow-hidden"
              >
                <div
                  class="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-exo-yellow to-exo-yellow-darker rounded-full animate-pulse"
                ></div>
              </div>
              <p class="text-xs font-mono text-white/40 mt-4">
                Preparing download...
              </p>
            </div>
          {/if}

          <p class="text-xs font-sans text-white/40 mt-8">
            This may take a few minutes depending on your connection.
          </p>
        </div>
      {:else if onboardingStep === 8}
        <!-- Step 8: Loading into memory -->
        <div
          class="text-center max-w-lg px-8"
          style="opacity: 0; animation: onb-fade-in 0.5s ease forwards;"
        >
          <div class="mb-6">
            <h1
              class="text-xl font-sans font-light text-white/90 mb-2 tracking-wide"
            >
              Loading into memory
            </h1>
            {#if onboardingModelId}
              <p class="text-sm text-white/40 font-sans">
                {onboardingModelId.split("/").pop() ?? onboardingModelId}
              </p>
            {/if}
          </div>

          <!-- Device icon -->
          <div class="flex justify-center mb-6">
            <svg
              viewBox="0 0 200 200"
              class="w-32 h-32"
              xmlns="http://www.w3.org/2000/svg"
            >
              <DeviceIcon
                deviceType={userDeviceInfo.deviceType}
                cx={100}
                cy={100}
                size={80}
                ramPercent={60}
                uid="onb-loading"
              />
            </svg>
          </div>

          {#if onboardingLoadProgress}
            <div class="w-full max-w-xs mx-auto space-y-3">
              <div
                class="relative h-2 bg-white/10 rounded-full overflow-hidden"
              >
                <div
                  class="absolute inset-y-0 left-0 bg-gradient-to-r from-exo-yellow to-exo-yellow-darker rounded-full transition-all duration-500"
                  style="width: {onboardingLoadProgress.percentage}%"
                ></div>
              </div>
              <p class="text-xs text-white/40 font-mono text-center">
                {onboardingLoadProgress.layersLoaded} / {onboardingLoadProgress.totalLayers}
                layers loaded
              </p>
            </div>
          {:else}
            <div class="flex justify-center mb-4">
              <div
                class="w-8 h-8 border-2 border-exo-yellow/15 border-t-exo-yellow/70 rounded-full animate-spin"
              ></div>
            </div>
            <p class="text-sm text-white/30 font-sans">Loading...</p>
          {/if}
        </div>
      {:else if onboardingStep === 9}
        <!-- Step 9: Ready — centered input with suggestion chips -->
        <!-- Uses onb-fade-opacity (no transform) so fixed-position dropdown in ChatForm works correctly -->
        <div
          class="flex flex-col items-center justify-center w-full max-w-2xl px-8"
          style="opacity: 0; animation: onb-fade-opacity 0.6s ease forwards;"
        >
          <img
            src="/exo-logo.png"
            alt="exo"
            class="w-28 mb-6"
            style="opacity: 0.8;"
          />

          {#if onboardingModelId}
            <p class="text-sm text-white/40 font-mono mb-6">
              {onboardingModelId.split("/").pop() ?? onboardingModelId}
            </p>
          {/if}

          <div class="w-full">
            <ChatForm
              placeholder="Ask anything"
              autofocus={true}
              showHelperText={false}
              showModelSelector={true}
              modelTasks={modelTasks()}
              modelCapabilities={modelCapabilities()}
              onOpenModelPicker={openChatModelPicker}
              onAutoSend={handleChatSend}
            />
          </div>

          <div class="flex flex-wrap justify-center gap-3 mt-6">
            {#each suggestedPrompts as chip}
              <button
                type="button"
                onclick={() => {
                  completeOnboarding();
                  sendMessage(chip);
                }}
                class="px-4 py-2 rounded-full border border-white/10 bg-white/5 text-sm text-white/60 hover:bg-white/10 hover:text-white/80 hover:border-white/20 transition-all duration-200 cursor-pointer"
              >
                {chip}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Replay / Skip — visible on all onboarding steps -->
      <div class="absolute bottom-8 flex items-center gap-6">
        <button
          type="button"
          onclick={() => {
            onboardingStep = 0;
            setTimeout(() => {
              onboardingStep = 1;
            }, 50);
          }}
          class="flex items-center gap-1.5 text-xs font-sans text-white/15 hover:text-white/35 transition-colors duration-300 cursor-pointer"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M2.5 2v5h5" />
            <path d="M2.5 7a6.5 6.5 0 1 1 1.4-2.8" />
          </svg>
          Replay
        </button>
        <button
          type="button"
          onclick={completeOnboarding}
          class="flex items-center gap-1.5 text-xs font-sans text-white/15 hover:text-white/35 transition-colors duration-300 cursor-pointer"
        >
          <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
            <path d="M3 2.5v11L9 8 3 2.5z" />
            <rect x="10.5" y="2.5" width="2.5" height="11" rx="0.5" />
          </svg>
          Skip
        </button>
      </div>
    </div>

    <!-- Model Picker Modal (available during onboarding step 4) -->
    {#if onboardingStep === 6}
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
  {/if}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!-- MAIN DASHBOARD (always rendered, behind onboarding)    -->
  <!-- ═══════════════════════════════════════════════════════ -->
  {#if !topologyOnlyEnabled}
    <HeaderNav
      showHome={true}
      onHome={handleGoHome}
      showSidebarToggle={true}
      {sidebarVisible}
      onToggleSidebar={toggleChatSidebarVisible}
      downloadProgress={activeDownloadSummary}
    />
  {/if}

  <!-- Main Content -->
  <main class="flex-1 flex overflow-hidden relative">
    <!-- Left: Conversation History Sidebar (hidden in topology-only mode, welcome state, or when toggled off) -->
    {#if !topologyOnlyEnabled && sidebarVisible}
      <div
        class="w-80 flex-shrink-0 border-r border-exo-yellow/10"
        role="complementary"
        aria-label="Conversation history"
      >
        <ChatSidebar
          class="h-full"
          onNewChat={handleNewChat}
          onSelectConversation={() => {
            userForcedIdle = false;
          }}
        />
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

          <!-- TB5 RDMA Not Enabled Warning -->
          {#if tb5WithoutRdma && !tb5InfoDismissed}
            <div
              class="absolute left-4 group"
              class:top-16={tbBridgeCycles.length > 0}
              class:top-4={tbBridgeCycles.length === 0}
              role="status"
            >
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
                  RDMA NOT ENABLED
                </span>
                <button
                  type="button"
                  onclick={() => (tb5InfoDismissed = true)}
                  class="ml-1 text-yellow-300/60 hover:text-yellow-200 transition-colors cursor-pointer"
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
                class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-yellow-500/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
              >
                <p class="text-xs text-white/80 mb-2">
                  Thunderbolt 5 hardware detected on multiple nodes. Enable RDMA
                  for significantly faster inter-node communication.
                </p>
                <p class="text-xs text-white/60 mb-1.5">
                  <span class="text-yellow-300">To enable:</span>
                </p>
                <ol
                  class="text-xs text-white/60 list-decimal list-inside space-y-0.5 mb-1.5"
                >
                  <li>Connect nodes with TB5 cables</li>
                  <li>Boot to Recovery (hold power 10s → Options)</li>
                  <li>
                    Run
                    <code class="text-yellow-300 bg-yellow-400/10 px-1 rounded"
                      >rdma_ctl enable</code
                    >
                  </li>
                  <li>Reboot</li>
                </ol>
                <p class="text-xs text-white/40">
                  Requires macOS 26.2+, TB5 cables, and matching OS versions.
                </p>
              </div>
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
        class="flex-1 flex overflow-hidden relative"
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

            {@render clusterWarnings()}

            <!-- TB5 RDMA Not Enabled Warning -->
            {#if tb5WithoutRdma && !tb5InfoDismissed}
              <div
                class="absolute left-4 group"
                class:top-16={tbBridgeCycles.length > 0}
                class:top-4={tbBridgeCycles.length === 0}
                role="status"
              >
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
                    RDMA NOT ENABLED
                  </span>
                  <button
                    type="button"
                    onclick={() => (tb5InfoDismissed = true)}
                    class="ml-1 text-yellow-300/60 hover:text-yellow-200 transition-colors cursor-pointer"
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
                  class="absolute top-full left-0 mt-2 w-80 p-3 rounded border border-yellow-500/30 bg-exo-dark-gray/95 backdrop-blur-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-lg"
                >
                  <p class="text-xs text-white/80 mb-2">
                    Thunderbolt 5 hardware detected on multiple nodes. Enable
                    RDMA for significantly faster inter-node communication.
                  </p>
                  <p class="text-xs text-white/60 mb-1.5">
                    <span class="text-yellow-300">To enable:</span>
                  </p>
                  <ol
                    class="text-xs text-white/60 list-decimal list-inside space-y-0.5 mb-1.5"
                  >
                    <li>Connect nodes with TB5 cables</li>
                    <li>Boot to Recovery (hold power 10s → Options)</li>
                    <li>
                      Run
                      <code
                        class="text-yellow-300 bg-yellow-400/10 px-1 rounded"
                        >rdma_ctl enable</code
                      >
                    </li>
                    <li>Reboot</li>
                  </ol>
                  <p class="text-xs text-white/40">
                    Requires macOS 26.2+, TB5 cables, and matching OS versions.
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

          <!-- Chat Input - Below topology, never overlaps -->
          <div class="px-4 pt-4 pb-6 flex-shrink-0">
            <div class="max-w-3xl mx-auto">
              {#if instanceCount === 0}
                <div class="text-center mb-4">
                  <p class="text-sm text-white/50 font-sans">
                    Select a model to get started.
                  </p>
                </div>
              {/if}
              <ChatForm
                placeholder={instanceCount === 0
                  ? "Choose a model to start chatting"
                  : "Ask anything"}
                showHelperText={false}
                showModelSelector={true}
                modelTasks={modelTasks()}
                modelCapabilities={modelCapabilities()}
                onOpenModelPicker={openChatModelPicker}
                onAutoSend={handleChatSend}
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
                {#each Object.entries(instanceData) as [id, instance]}
                  {@const downloadInfo = getInstanceDownloadStatus(
                    id,
                    instance,
                  )}
                  {@const statusText = downloadInfo.statusText}
                  {@const isDownloading = downloadInfo.isDownloading}
                  {@const isFailed = statusText === "FAILED"}
                  {@const isLoading = statusText === "LOADING"}
                  {@const isWarmingUp =
                    statusText === "WARMING UP" || statusText === "WAITING"}
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
                    transition:slide={{ duration: 250, easing: cubicOut }}
                    onmouseenter={() => (hoveredInstanceId = id)}
                    onmouseleave={() => (hoveredInstanceId = null)}
                    onclick={() => {
                      if (
                        instanceModelId &&
                        instanceModelId !== "Unknown" &&
                        instanceModelId !== "Unknown Model"
                      ) {
                        userForcedIdle = false;
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
                          <div class="mt-2 space-y-1">
                            <div
                              class="text-xs text-blue-400 font-mono tracking-wider"
                            >
                              DOWNLOADING
                            </div>
                            <p
                              class="text-[11px] text-white/50 leading-relaxed"
                            >
                              Downloading model files. Model runs on your
                              devices so needs to be downloaded before you can
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
                              {@const loadStatus =
                                deriveInstanceStatus(instance)}
                              {#if loadStatus.totalLayers && loadStatus.totalLayers > 0}
                                <div class="mt-1 space-y-1">
                                  <div
                                    class="flex justify-between text-xs font-mono"
                                  >
                                    <span class="text-yellow-400"
                                      >{(
                                        ((loadStatus.layersLoaded ?? 0) /
                                          loadStatus.totalLayers) *
                                        100
                                      ).toFixed(0)}%</span
                                    >
                                    <span class="text-exo-light-gray"
                                      >{loadStatus.layersLoaded ?? 0} / {loadStatus.totalLayers}
                                      layers</span
                                    >
                                  </div>
                                  <div
                                    class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                                  >
                                    <div
                                      class="absolute inset-y-0 left-0 bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-300"
                                      style="width: {((loadStatus.layersLoaded ??
                                        0) /
                                        loadStatus.totalLayers) *
                                        100}%"
                                    ></div>
                                  </div>
                                </div>
                              {:else}
                                <p
                                  class="text-[11px] text-white/50 leading-relaxed"
                                >
                                  Loading model into memory...
                                </p>
                              {/if}
                            {:else if isWarmingUp}
                              <p
                                class="text-[11px] text-white/50 leading-relaxed"
                              >
                                Warming up...
                              </p>
                            {:else if isReady || isRunning}
                              <p
                                class="text-[11px] text-green-400/70 leading-relaxed"
                              >
                                Ready to chat!
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

          <!-- Models Panel - Scrollable -->
          <div class="p-4 flex-1 overflow-y-auto">
            <!-- Panel Header -->
            <div class="flex items-center gap-2 mb-3 flex-shrink-0">
              <div class="w-2 h-2 border border-exo-yellow/60 rotate-45"></div>
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
                onclick={() => {
                  modelPickerContext = "dashboard";
                  isModelPickerOpen = true;
                }}
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
                {:else if bestRunningModelId}
                  {@const runModel = models.find(
                    (m) => m.id === bestRunningModelId,
                  )}
                  {#if runModel}
                    {@const sizeGB = getModelSizeGB(runModel)}
                    <span
                      class="flex items-center justify-between gap-2 w-full pr-4"
                    >
                      <span
                        class="flex items-center gap-2 text-exo-light-gray truncate"
                      >
                        <span class="truncate"
                          >{runModel.name || runModel.id}</span
                        >
                      </span>
                      <span class="text-white/50 text-xs flex-shrink-0"
                        >{sizeGB >= 1
                          ? sizeGB.toFixed(0)
                          : sizeGB.toFixed(1)}GB</span
                      >
                    </span>
                  {:else}
                    <span class="text-exo-light-gray">{bestRunningModelId}</span
                    >
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
                            <span class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
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
                            <span class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
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
                            <span class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
                            ></span>
                          {/if}
                        </span>
                        TCP/IP
                      </button>
                      <button
                        onclick={() => {
                          selectedInstanceType = "MlxJaccl";
                          saveLaunchDefaults();
                        }}
                        class="flex items-center gap-2 py-1.5 px-3 text-xs font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType ===
                        'MlxJaccl'
                          ? 'bg-transparent text-exo-yellow border-exo-yellow'
                          : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
                      >
                        <span
                          class="w-3 h-3 rounded-full border-2 flex items-center justify-center {selectedInstanceType ===
                          'MlxJaccl'
                            ? 'border-exo-yellow'
                            : 'border-exo-medium-gray'}"
                        >
                          {#if selectedInstanceType === "MlxJaccl"}
                            <span class="w-1.5 h-1.5 rounded-full bg-exo-yellow"
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
          {#if chatLaunchState !== "idle" && chatLaunchState !== "ready"}
            <!-- Model launching/downloading/loading: show progress -->
            <div class="flex-1 flex items-center justify-center px-8 py-6">
              <div class="flex flex-col items-center gap-6 max-w-md w-full">
                <!-- Model name -->
                {#if pendingChatModelId}
                  <p class="text-sm text-white font-mono tracking-wide">
                    {pendingChatModelId.split("/").pop()?.replace(/-/g, " ") ||
                      pendingChatModelId}
                  </p>
                {/if}

                {#if chatLaunchState === "launching"}
                  <div class="flex flex-col items-center gap-3">
                    <div
                      class="w-8 h-8 border-2 border-exo-yellow/30 border-t-exo-yellow rounded-full animate-spin"
                    ></div>
                    <p
                      class="text-xs text-exo-light-gray font-mono uppercase tracking-wider"
                    >
                      Preparing to launch&hellip;
                    </p>
                  </div>
                {:else if chatLaunchState === "downloading"}
                  <div class="w-full flex flex-col gap-3">
                    <div
                      class="flex items-center justify-between text-xs font-mono"
                    >
                      <span class="text-exo-yellow uppercase tracking-wider"
                        >Downloading</span
                      >
                      {#if chatLaunchDownload}
                        <span class="text-exo-light-gray tabular-nums">
                          {chatLaunchDownload.percentage.toFixed(1)}%
                        </span>
                      {/if}
                    </div>
                    <div
                      class="w-full h-2 bg-exo-dark-gray rounded-full overflow-hidden border border-exo-medium-gray/30"
                    >
                      <div
                        class="h-full bg-gradient-to-r from-exo-yellow/80 to-exo-yellow rounded-full transition-all duration-300"
                        style="width: {chatLaunchDownload?.percentage ?? 0}%"
                      ></div>
                    </div>
                    {#if chatLaunchDownload}
                      <div
                        class="flex justify-between text-[10px] text-exo-light-gray/60 font-mono"
                      >
                        <span
                          >{formatBytes(chatLaunchDownload.downloadedBytes)} / {formatBytes(
                            chatLaunchDownload.totalBytes,
                          )}</span
                        >
                        <span>
                          {#if chatLaunchDownload.speed > 0}
                            {formatBytes(chatLaunchDownload.speed)}/s
                          {/if}
                          {#if chatLaunchDownload.etaMs > 0}
                            &middot; {formatEta(chatLaunchDownload.etaMs)}
                          {/if}
                        </span>
                      </div>
                    {/if}
                  </div>
                {:else if chatLaunchState === "loading"}
                  <div class="w-full flex flex-col gap-3">
                    <div
                      class="flex items-center justify-between text-xs font-mono"
                    >
                      <span class="text-exo-yellow uppercase tracking-wider"
                        >Loading model</span
                      >
                      {#if chatLaunchLoadProgress}
                        <span class="text-exo-light-gray tabular-nums">
                          {chatLaunchLoadProgress.layersLoaded}/{chatLaunchLoadProgress.totalLayers}
                          layers
                        </span>
                      {/if}
                    </div>
                    <div
                      class="w-full h-2 bg-exo-dark-gray rounded-full overflow-hidden border border-exo-medium-gray/30"
                    >
                      <div
                        class="h-full bg-gradient-to-r from-exo-yellow/80 to-exo-yellow rounded-full transition-all duration-300"
                        style="width: {chatLaunchLoadProgress?.percentage ??
                          0}%"
                      ></div>
                    </div>
                  </div>
                {/if}
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
                  onAutoSend={handleChatSend}
                  onOpenModelPicker={openChatModelPicker}
                />
              </div>
            </div>
          {:else if messages().length > 0 || chatLaunchState === "ready"}
            <!-- Normal chat: show messages -->
            <div
              class="flex-1 overflow-y-auto px-8 py-6"
              bind:this={chatScrollRef}
              role="log"
              aria-live="polite"
              aria-label="Chat messages"
            >
              <div class="max-w-7xl mx-auto">
                <ChatMessages scrollParent={chatScrollRef} />
                {#if chatLaunchState === "ready" && selectedChatCategory}
                  {@const prompts =
                    categorySuggestedPrompts[selectedChatCategory] ??
                    categorySuggestedPrompts.auto}
                  <div
                    class="flex flex-col items-center gap-4 mt-12"
                    in:fade={{ duration: 300 }}
                  >
                    <p
                      class="text-xs text-exo-light-gray/60 font-mono uppercase tracking-wider"
                    >
                      Try asking
                    </p>
                    <div class="grid grid-cols-2 gap-2 max-w-lg w-full">
                      {#each prompts as prompt}
                        <button
                          type="button"
                          onclick={() => {
                            chatLaunchState = "idle";
                            selectedChatCategory = null;
                            sendMessage(prompt);
                          }}
                          class="text-left px-3 py-2.5 text-xs text-exo-light-gray hover:text-white font-mono rounded-lg border border-exo-medium-gray/30 hover:border-exo-yellow/30 bg-exo-dark-gray/30 hover:bg-exo-dark-gray/60 transition-all duration-200 cursor-pointer"
                        >
                          {prompt}
                        </button>
                      {/each}
                    </div>
                  </div>
                {/if}
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
                  onAutoSend={handleChatSend}
                  onOpenModelPicker={openChatModelPicker}
                />
              </div>
            </div>
          {:else}
            <!-- No running instance, no messages: show model selector -->
            <div
              class="flex-1 overflow-y-auto flex items-center justify-center px-8 py-6"
            >
              <ChatModelSelector
                models={models.map((m) => ({
                  id: m.id,
                  name: m.name ?? "",
                  base_model: m.base_model ?? "",
                  storage_size_megabytes: m.storage_size_megabytes ?? 0,
                  capabilities: m.capabilities ?? [],
                  family: m.family ?? "",
                  quantization: m.quantization ?? "",
                }))}
                clusterLabel={chatClusterLabel}
                totalMemoryGB={availableMemoryGB()}
                onSelect={handleChatModelSelect}
                onAddModel={handleChatAddModel}
              />
            </div>
            <div
              class="flex-shrink-0 px-8 pb-6 pt-4 bg-gradient-to-t from-exo-black via-exo-black to-transparent"
            >
              <div class="max-w-7xl mx-auto">
                <ChatForm
                  placeholder="Ask anything — we'll pick the best model automatically"
                  showModelSelector={!!bestRunningModelId}
                  modelDisplayOverride={bestRunningModelId ?? undefined}
                  modelTasks={modelTasks()}
                  modelCapabilities={modelCapabilities()}
                  onAutoSend={handleAutoSend}
                  onOpenModelPicker={openChatModelPicker}
                />
              </div>
            </div>
          {/if}
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
                    {@const isLoading = statusText === "LOADING"}
                    {@const isWarmingUp =
                      statusText === "WARMING UP" || statusText === "WAITING"}
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
                          userForcedIdle = false;
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
                                Downloading model files. Model runs on your
                                devices so needs to be downloaded before you can
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
                                {@const loadStatus =
                                  deriveInstanceStatus(instance)}
                                {#if loadStatus.totalLayers && loadStatus.totalLayers > 0}
                                  <div class="mt-1 space-y-1">
                                    <div
                                      class="flex justify-between text-xs font-mono"
                                    >
                                      <span class="text-yellow-400"
                                        >{(
                                          ((loadStatus.layersLoaded ?? 0) /
                                            loadStatus.totalLayers) *
                                          100
                                        ).toFixed(0)}%</span
                                      >
                                      <span class="text-exo-light-gray"
                                        >{loadStatus.layersLoaded ?? 0} / {loadStatus.totalLayers}
                                        layers</span
                                      >
                                    </div>
                                    <div
                                      class="relative h-1.5 bg-exo-black/60 rounded-sm overflow-hidden"
                                    >
                                      <div
                                        class="absolute inset-y-0 left-0 bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-300"
                                        style="width: {((loadStatus.layersLoaded ??
                                          0) /
                                          loadStatus.totalLayers) *
                                          100}%"
                                      ></div>
                                    </div>
                                  </div>
                                {:else}
                                  <p
                                    class="text-[11px] text-white/50 leading-relaxed"
                                  >
                                    Loading model into memory...
                                  </p>
                                {/if}
                              {:else if isWarmingUp}
                                <p
                                  class="text-[11px] text-white/50 leading-relaxed"
                                >
                                  Warming up...
                                </p>
                              {:else if isReady || isRunning}
                                <p
                                  class="text-[11px] text-green-400/70 leading-relaxed"
                                >
                                  Ready to chat!
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
    onSelect={(modelId) => {
      if (modelPickerContext === "chat") {
        handleChatPickerSelect(modelId);
      } else {
        handleModelPickerSelect(modelId);
      }
    }}
    onClose={() => (isModelPickerOpen = false)}
    onToggleFavorite={toggleFavorite}
    onAddModel={addModelFromPicker}
    onDeleteModel={deleteCustomModel}
    totalMemoryGB={clusterMemory().total / (1024 * 1024 * 1024)}
    usedMemoryGB={clusterMemory().used / (1024 * 1024 * 1024)}
    {downloadsData}
    topologyNodes={data?.nodes}
    instanceStatuses={modelInstanceStatuses}
  />
{/if}

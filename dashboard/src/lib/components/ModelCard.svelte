<script lang="ts">
  import type {
    DownloadProgress,
    NodeInfo,
    PlacementPreview,
    TopologyEdge,
  } from "$lib/stores/app.svelte";
  import { debugMode, topologyData } from "$lib/stores/app.svelte";

  interface Props {
    model: { id: string; name?: string; storage_size_megabytes?: number };
    isLaunching?: boolean;
    downloadStatus?: {
      isDownloading: boolean;
      progress: DownloadProgress | null;
      perNode?: Array<{
        nodeId: string;
        nodeName: string;
        progress: DownloadProgress;
      }>;
    } | null;
    nodes?: Record<string, NodeInfo>;
    sharding?: "Pipeline" | "Tensor";
    runtime?: "MlxRing" | "MlxIbv" | "MlxJaccl";
    onLaunch?: () => void;
    tags?: string[];
    apiPreview?: PlacementPreview | null;
    modelIdOverride?: string | null;
  }

  let {
    model,
    isLaunching = false,
    downloadStatus = null,
    nodes = {},
    sharding = "Pipeline",
    runtime = "MlxRing",
    onLaunch,
    tags = [],
    apiPreview = null,
    modelIdOverride = null,
  }: Props = $props();

  // Estimate memory requirements from model name
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

  function formatBytes(bytes: number, decimals = 1): string {
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
    return formatBytes(bps) + "/s";
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

  const isDownloading = $derived(downloadStatus?.isDownloading ?? false);
  const progress = $derived(downloadStatus?.progress);
  const percentage = $derived(progress?.percentage ?? 0);
  let expandedNodes = $state<Set<string>>(new Set());

  function toggleNodeDetails(nodeId: string): void {
    const next = new Set(expandedNodes);
    if (next.has(nodeId)) {
      next.delete(nodeId);
    } else {
      next.add(nodeId);
    }
    expandedNodes = next;
  }

  // Use actual storage_size_megabytes from API if available, otherwise fall back to estimate
  const estimatedMemory = $derived(
    model.storage_size_megabytes
      ? Math.round(model.storage_size_megabytes / 1024)
      : estimateMemoryGB(model.id, model.name),
  );

  function getDeviceType(
    name: string,
  ): "macbook" | "studio" | "mini" | "unknown" {
    const lower = name.toLowerCase();
    if (lower.includes("macbook")) return "macbook";
    if (lower.includes("studio")) return "studio";
    if (lower.includes("mini")) return "mini";
    return "unknown";
  }

  const clampPercent = (value: number): number =>
    Math.min(100, Math.max(0, value));
  const huggingFaceModelId = $derived(modelIdOverride ?? model.id);

  // Get node list in the same order as the topology graph (insertion order of
  // topology nodes), while still ensuring preview nodes render even if the
  // topology payload is missing them. Topology order is preserved exactly so
  // that the mini preview matches the main TopologyGraph layout.
  const nodeList = $derived(() => {
    const nodesFromTopology = Object.keys(nodes).map((id) => {
      const info = nodes[id];
      const totalBytes =
        info.macmon_info?.memory?.ram_total ?? info.system_info?.memory ?? 0;
      const usedBytes = info.macmon_info?.memory?.ram_usage ?? 0;
      const availableBytes = Math.max(totalBytes - usedBytes, 0);
      const totalGB = totalBytes / (1024 * 1024 * 1024);
      const availableGB = availableBytes / (1024 * 1024 * 1024);
      const usedGB = Math.max(totalGB - availableGB, 0);
      const deviceName = info.system_info?.model_id ?? "Unknown";
      const deviceType = getDeviceType(deviceName);

      return {
        id,
        totalGB,
        availableGB,
        usedGB,
        deviceName,
        deviceType,
        usedBytes,
        totalBytes,
      };
    });

    const previewEntries = apiPreview?.memory_delta_by_node ?? null;
    const previewIds = previewEntries ? Object.keys(previewEntries) : [];

    if (previewIds.length === 0) return nodesFromTopology;

    // Append any preview-only nodes (not in topology) at the end
    const topologyIds = new Set(nodesFromTopology.map((n) => n.id));
    const extraPreviewNodes = previewIds
      .filter((id) => !topologyIds.has(id))
      .map((id) => {
        const deltaBytes = previewEntries?.[id] ?? 0;
        const deltaGB = deltaBytes / (1024 * 1024 * 1024);
        const totalGB = Math.max(deltaGB * 1.2, 1);
        const usedGB = Math.max(totalGB - deltaGB, 0);

        return {
          id,
          totalGB,
          availableGB: Math.max(totalGB - usedGB, 0),
          usedGB,
          deviceName: "Unknown",
          deviceType: "unknown" as const,
          usedBytes: usedGB * 1024 * 1024 * 1024,
          totalBytes: totalGB * 1024 * 1024 * 1024,
        };
      });

    return [...nodesFromTopology, ...extraPreviewNodes];
  });

  // Calculate placement preview with all SVG metrics pre-computed
  // Uses API preview data when available, falls back to local estimation
  const placementPreview = $derived(() => {
    const nodeArray = nodeList();
    if (nodeArray.length === 0)
      return {
        nodes: [],
        canFit: false,
        totalAvailable: 0,
        topoWidth: 260,
        topoHeight: 90,
        error: null,
      };

    const numNodes = nodeArray.length;
    const iconSize = numNodes === 1 ? 50 : 36;
    const topoWidth = 260;
    const topoHeight =
      numNodes === 1 ? 90 : numNodes === 2 ? 140 : numNodes * 50 + 20;
    const centerX = topoWidth / 2;
    const centerY = topoHeight / 2;
    const radius =
      numNodes === 1
        ? 0
        : numNodes === 2
          ? 45
          : Math.min(topoWidth, topoHeight) * 0.32;

    // Only use API preview data - no local estimation
    const hasApiPreview =
      apiPreview !== null &&
      apiPreview.error === null &&
      apiPreview.memory_delta_by_node !== null;
    const error = apiPreview?.error ?? null;

    let placementNodes: Array<{
      id: string;
      deviceName: string;
      deviceType: "macbook" | "studio" | "mini" | "unknown";
      totalGB: number;
      currentUsedGB: number;
      modelUsageGB: number;
      currentPercent: number;
      newPercent: number;
      isUsed: boolean;
      x: number;
      y: number;
      iconSize: number;
      screenHeight: number;
      currentFillHeight: number;
      modelFillHeight: number;
    }> = [];

    // Use API placement data directly
    const memoryDelta = apiPreview?.memory_delta_by_node ?? {};
    placementNodes = nodeArray.map((n, i) => {
      const deltaBytes = memoryDelta[n.id] ?? 0;
      const modelUsageGB = deltaBytes / (1024 * 1024 * 1024);
      const isUsed = deltaBytes > 0;
      const angle =
        numNodes === 1 ? 0 : (i / numNodes) * Math.PI * 2 - Math.PI / 2;
      const safeTotal = Math.max(n.totalGB, 0.001);
      const currentPercent = clampPercent((n.usedGB / safeTotal) * 100);
      const newPercent = clampPercent(
        ((n.usedGB + modelUsageGB) / safeTotal) * 100,
      );
      const screenHeight = iconSize * 0.58;

      return {
        id: n.id,
        deviceName: n.deviceName,
        deviceType: n.deviceType,
        totalGB: n.totalGB,
        currentUsedGB: n.usedGB,
        modelUsageGB,
        currentPercent,
        newPercent,
        isUsed,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        iconSize,
        screenHeight,
        currentFillHeight: screenHeight * (currentPercent / 100),
        modelFillHeight: screenHeight * ((newPercent - currentPercent) / 100),
      };
    });

    const totalAvailable = nodeArray.reduce((sum, n) => sum + n.availableGB, 0);
    return {
      nodes: placementNodes,
      canFit: hasApiPreview,
      totalAvailable,
      topoWidth,
      topoHeight,
      error,
    };
  });

  const canFit = $derived(
    apiPreview ? apiPreview.error === null : placementPreview().canFit,
  );
  const placementError = $derived(apiPreview?.error ?? null);
  const nodeCount = $derived(nodeList().length);
  const filterId = $derived(model.id.replace(/[^a-zA-Z0-9]/g, ""));

  // Debug mode state
  const isDebugMode = $derived(debugMode());
  const topology = $derived(topologyData());
  const isRdma = $derived(runtime === "MlxIbv" || runtime === "MlxJaccl");

  // Get interface name for an IP from node data
  function getInterfaceForIp(nodeId: string, ip?: string): string | null {
    if (!ip || !topology?.nodes) return null;

    // Strip port if present
    const cleanIp =
      ip.includes(":") && !ip.includes("[") ? ip.split(":")[0] : ip;

    // Check specified node first
    const node = topology.nodes[nodeId];
    if (node) {
      const match = node.network_interfaces?.find((iface) =>
        (iface.addresses || []).some((addr) => addr === cleanIp || addr === ip),
      );
      if (match?.name) return match.name;

      const mapped =
        node.ip_to_interface?.[cleanIp] || node.ip_to_interface?.[ip];
      if (mapped) return mapped;
    }

    // Fallback: check all nodes
    for (const [, otherNode] of Object.entries(topology.nodes)) {
      if (!otherNode) continue;
      const match = otherNode.network_interfaces?.find((iface) =>
        (iface.addresses || []).some((addr) => addr === cleanIp || addr === ip),
      );
      if (match?.name) return match.name;

      const mapped =
        otherNode.ip_to_interface?.[cleanIp] || otherNode.ip_to_interface?.[ip];
      if (mapped) return mapped;
    }

    return null;
  }

  // Get directional arrow based on node positions
  function getArrow(
    fromNode: { x: number; y: number },
    toNode: { x: number; y: number },
  ): string {
    const dx = toNode.x - fromNode.x;
    const dy = toNode.y - fromNode.y;
    const absX = Math.abs(dx);
    const absY = Math.abs(dy);

    if (absX > absY * 2) {
      return dx > 0 ? "→" : "←";
    } else if (absY > absX * 2) {
      return dy > 0 ? "↓" : "↑";
    } else {
      if (dx > 0 && dy > 0) return "↘";
      if (dx > 0 && dy < 0) return "↗";
      if (dx < 0 && dy > 0) return "↙";
      return "↖";
    }
  }

  // Get connection info for edges between two nodes
  // Returns exactly one connection per direction (A→B and B→A), preferring non-loopback
  function getConnectionInfo(
    nodeId1: string,
    nodeId2: string,
  ): Array<{ ip: string; iface: string | null; from: string; to: string }> {
    if (!topology?.edges) return [];

    // Collect candidates for each direction
    const aToBCandidates: Array<{ ip: string; iface: string | null }> = [];
    const bToACandidates: Array<{ ip: string; iface: string | null }> = [];

    for (const edge of topology.edges) {
      const ip = edge.sendBackIp || "?";
      const iface =
        edge.sendBackInterface || getInterfaceForIp(edge.source, ip);

      if (edge.source === nodeId1 && edge.target === nodeId2) {
        aToBCandidates.push({ ip, iface });
      } else if (edge.source === nodeId2 && edge.target === nodeId1) {
        bToACandidates.push({ ip, iface });
      }
    }

    // Pick best (prefer non-loopback)
    const pickBest = (
      candidates: Array<{ ip: string; iface: string | null }>,
    ) => {
      if (candidates.length === 0) return null;
      return candidates.find((c) => !c.ip.startsWith("127.")) || candidates[0];
    };

    const result: Array<{
      ip: string;
      iface: string | null;
      from: string;
      to: string;
    }> = [];

    const bestAtoB = pickBest(aToBCandidates);
    if (bestAtoB) result.push({ ...bestAtoB, from: nodeId1, to: nodeId2 });

    const bestBtoA = pickBest(bToACandidates);
    if (bestBtoA) result.push({ ...bestBtoA, from: nodeId2, to: nodeId1 });

    return result;
  }
</script>

<div class="relative group">
  <!-- Corner accents -->
  <div
    class="absolute -top-px -left-px w-2 h-2 border-l border-t {canFit
      ? 'border-exo-yellow/30 group-hover:border-exo-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>
  <div
    class="absolute -top-px -right-px w-2 h-2 border-r border-t {canFit
      ? 'border-exo-yellow/30 group-hover:border-exo-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>
  <div
    class="absolute -bottom-px -left-px w-2 h-2 border-l border-b {canFit
      ? 'border-exo-yellow/30 group-hover:border-exo-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>
  <div
    class="absolute -bottom-px -right-px w-2 h-2 border-r border-b {canFit
      ? 'border-exo-yellow/30 group-hover:border-exo-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>

  <div
    class="bg-exo-dark-gray/60 border {canFit
      ? 'border-exo-yellow/20 group-hover:border-exo-yellow/40'
      : 'border-red-500/20'} p-3 transition-all duration-200 group-hover:shadow-[0_0_15px_rgba(255,215,0,0.1)]"
  >
    <!-- Model Name & Memory Required -->
    <div class="flex items-start justify-between gap-2 mb-2">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <div
            class="text-exo-yellow text-xs font-mono tracking-wide truncate"
            title={model.name || model.id}
          >
            {model.name || model.id}
          </div>
          {#if huggingFaceModelId}
            <a
              class="shrink-0 text-white/60 hover:text-exo-yellow transition-colors"
              href={`https://huggingface.co/${huggingFaceModelId}`}
              target="_blank"
              rel="noreferrer noopener"
              aria-label="View model on Hugging Face"
            >
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
          {#if tags.length > 0}
            <div class="flex gap-1 flex-shrink-0">
              {#each tags as tag}
                <span
                  class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase rounded {tag ===
                  'FASTEST'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'}"
                >
                  {tag}
                </span>
              {/each}
            </div>
          {/if}
        </div>
        {#if model.name && model.name !== model.id}
          <div
            class="text-xs text-exo-light-gray font-mono truncate mt-0.5"
            title={model.id}
          >
            {model.id}
          </div>
        {/if}
      </div>
      <div class="flex-shrink-0 text-right">
        <div
          class="text-xs font-mono {canFit
            ? 'text-exo-yellow'
            : 'text-red-400'}"
        >
          {estimatedMemory}GB
        </div>
      </div>
    </div>

    <!-- Configuration Badge -->
    <div class="flex items-center gap-1.5 mb-2">
      <span
        class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/30 text-exo-light-gray border border-exo-medium-gray/40"
      >
        {sharding}
      </span>
      <span
        class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase bg-exo-medium-gray/30 text-exo-light-gray border border-exo-medium-gray/40"
      >
        {runtime === "MlxRing"
          ? "MLX Ring"
          : runtime === "MlxIbv" || runtime === "MlxJaccl"
            ? "MLX RDMA"
            : runtime}
      </span>
    </div>

    <!-- Mini Topology Preview -->
    {#if placementPreview().nodes.length > 0}
      {@const preview = placementPreview()}
      <div
        class="mb-3 bg-exo-black/60 rounded border border-exo-medium-gray/20 p-2 relative overflow-hidden"
      >
        <!-- Scanline effect -->
        <div
          class="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_2px,rgba(255,215,0,0.02)_2px,rgba(255,215,0,0.02)_4px)] pointer-events-none"
        ></div>

        <svg
          width="100%"
          height={preview.topoHeight}
          viewBox="0 0 {preview.topoWidth} {preview.topoHeight}"
          class="overflow-visible"
        >
          <defs>
            <!-- Glow filter for active nodes -->
            <filter
              id="nodeGlow-{filterId}"
              x="-50%"
              y="-50%"
              width="200%"
              height="200%"
            >
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <!-- Strong glow for new memory -->
            <filter
              id="memGlow-{filterId}"
              x="-100%"
              y="-100%"
              width="300%"
              height="300%"
            >
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <!-- Connection lines between nodes (if multiple) -->
          {#if preview.nodes.length > 1}
            {@const usedNodes = preview.nodes.filter((n) => n.isUsed)}
            {@const nodePositions = Object.fromEntries(
              preview.nodes.map((n) => [n.id, { x: n.x, y: n.y }]),
            )}
            {@const allConnections =
              isDebugMode && usedNodes.length > 1
                ? (() => {
                    const conns: Array<{
                      ip: string;
                      iface: string | null;
                      from: string;
                      to: string;
                      midX: number;
                      midY: number;
                      arrow: string;
                    }> = [];
                    for (let i = 0; i < usedNodes.length; i++) {
                      for (let j = i + 1; j < usedNodes.length; j++) {
                        const n1 = usedNodes[i];
                        const n2 = usedNodes[j];
                        const midX = (n1.x + n2.x) / 2;
                        const midY = (n1.y + n2.y) / 2;
                        for (const c of getConnectionInfo(n1.id, n2.id)) {
                          const fromPos = nodePositions[c.from];
                          const toPos = nodePositions[c.to];
                          const arrow =
                            fromPos && toPos ? getArrow(fromPos, toPos) : "→";
                          conns.push({ ...c, midX, midY, arrow });
                        }
                      }
                    }
                    return conns;
                  })()
                : []}
            {#each preview.nodes as node, i}
              {#each preview.nodes.slice(i + 1) as node2}
                <line
                  x1={node.x}
                  y1={node.y}
                  x2={node2.x}
                  y2={node2.y}
                  stroke={node.isUsed && node2.isUsed ? "#FFD700" : "#374151"}
                  stroke-width="1"
                  stroke-dasharray={node.isUsed && node2.isUsed ? "4,2" : "2,4"}
                  opacity={node.isUsed && node2.isUsed ? 0.4 : 0.15}
                />
              {/each}
            {/each}
            <!-- Debug: Show connection IPs/interfaces in corners -->
            {#if isDebugMode && allConnections.length > 0}
              {@const centerX = preview.topoWidth / 2}
              {@const centerY = preview.topoHeight / 2}
              {@const quadrants = {
                topLeft: allConnections.filter(
                  (c) => c.midX < centerX && c.midY < centerY,
                ),
                topRight: allConnections.filter(
                  (c) => c.midX >= centerX && c.midY < centerY,
                ),
                bottomLeft: allConnections.filter(
                  (c) => c.midX < centerX && c.midY >= centerY,
                ),
                bottomRight: allConnections.filter(
                  (c) => c.midX >= centerX && c.midY >= centerY,
                ),
              }}
              {@const padding = 4}
              {@const lineHeight = 8}
              <!-- Top Left -->
              {#each quadrants.topLeft as conn, idx}
                <text
                  x={padding}
                  y={padding + idx * lineHeight}
                  text-anchor="start"
                  dominant-baseline="hanging"
                  font-size="6"
                  font-family="SF Mono, Monaco, monospace"
                  fill={conn.iface
                    ? "rgba(255,255,255,0.85)"
                    : "rgba(248,113,113,0.85)"}
                >
                  {conn.arrow}
                  {isRdma
                    ? conn.iface || "?"
                    : `${conn.ip}${conn.iface ? ` (${conn.iface})` : ""}`}
                </text>
              {/each}
              <!-- Top Right -->
              {#each quadrants.topRight as conn, idx}
                <text
                  x={preview.topoWidth - padding}
                  y={padding + idx * lineHeight}
                  text-anchor="end"
                  dominant-baseline="hanging"
                  font-size="6"
                  font-family="SF Mono, Monaco, monospace"
                  fill={conn.iface
                    ? "rgba(255,255,255,0.85)"
                    : "rgba(248,113,113,0.85)"}
                >
                  {conn.arrow}
                  {isRdma
                    ? conn.iface || "?"
                    : `${conn.ip}${conn.iface ? ` (${conn.iface})` : ""}`}
                </text>
              {/each}
              <!-- Bottom Left -->
              {#each quadrants.bottomLeft as conn, idx}
                <text
                  x={padding}
                  y={preview.topoHeight -
                    padding -
                    (quadrants.bottomLeft.length - 1 - idx) * lineHeight}
                  text-anchor="start"
                  dominant-baseline="auto"
                  font-size="6"
                  font-family="SF Mono, Monaco, monospace"
                  fill={conn.iface
                    ? "rgba(255,255,255,0.85)"
                    : "rgba(248,113,113,0.85)"}
                >
                  {conn.arrow}
                  {isRdma
                    ? conn.iface || "?"
                    : `${conn.ip}${conn.iface ? ` (${conn.iface})` : ""}`}
                </text>
              {/each}
              <!-- Bottom Right -->
              {#each quadrants.bottomRight as conn, idx}
                <text
                  x={preview.topoWidth - padding}
                  y={preview.topoHeight -
                    padding -
                    (quadrants.bottomRight.length - 1 - idx) * lineHeight}
                  text-anchor="end"
                  dominant-baseline="auto"
                  font-size="6"
                  font-family="SF Mono, Monaco, monospace"
                  fill={conn.iface
                    ? "rgba(255,255,255,0.85)"
                    : "rgba(248,113,113,0.85)"}
                >
                  {conn.arrow}
                  {isRdma
                    ? conn.iface || "?"
                    : `${conn.ip}${conn.iface ? ` (${conn.iface})` : ""}`}
                </text>
              {/each}
            {/if}
          {/if}

          {#each preview.nodes as node}
            <g
              transform="translate({node.x}, {node.y})"
              opacity={node.isUsed ? 1 : 0.25}
              filter={node.isUsed ? `url(#nodeGlow-${filterId})` : "none"}
            >
              <!-- Device icon based on type -->
              {#if node.deviceType === "macbook"}
                <!-- MacBook Pro icon with memory fill -->
                <g
                  transform="translate({-node.iconSize / 2}, {-node.iconSize /
                    2})"
                >
                  <!-- Screen bezel -->
                  <rect
                    x="2"
                    y="0"
                    width={node.iconSize - 4}
                    height={node.iconSize * 0.65}
                    rx="2"
                    fill="none"
                    stroke={node.isUsed ? "#FFD700" : "#4B5563"}
                    stroke-width="1.5"
                  />
                  <!-- Screen area (memory fill container) -->
                  <rect
                    x="4"
                    y="2"
                    width={node.iconSize - 8}
                    height={node.screenHeight}
                    fill="#0a0a0a"
                  />
                  <!-- Current memory fill (gray) -->
                  <rect
                    x="4"
                    y={2 + node.screenHeight - node.currentFillHeight}
                    width={node.iconSize - 8}
                    height={node.currentFillHeight}
                    fill="#374151"
                  />
                  <!-- New model memory fill (glowing yellow) -->
                  {#if node.modelUsageGB > 0 && node.isUsed}
                    <rect
                      x="4"
                      y={2 +
                        node.screenHeight -
                        node.currentFillHeight -
                        node.modelFillHeight}
                      width={node.iconSize - 8}
                      height={node.modelFillHeight}
                      fill="#FFD700"
                      filter="url(#memGlow-{filterId})"
                      class="animate-pulse-slow"
                    />
                  {/if}
                  <!-- Base/keyboard -->
                  <path
                    d="M 0 {node.iconSize *
                      0.68} L {node.iconSize} {node.iconSize *
                      0.68} L {node.iconSize - 2} {node.iconSize *
                      0.78} L 2 {node.iconSize * 0.78} Z"
                    fill="none"
                    stroke={node.isUsed ? "#FFD700" : "#4B5563"}
                    stroke-width="1.5"
                  />
                </g>
              {:else if node.deviceType === "studio"}
                <!-- Mac Studio icon -->
                <g
                  transform="translate({-node.iconSize / 2}, {-node.iconSize /
                    2})"
                >
                  <rect
                    x="2"
                    y="2"
                    width={node.iconSize - 4}
                    height={node.iconSize - 4}
                    rx="4"
                    fill="none"
                    stroke={node.isUsed ? "#FFD700" : "#4B5563"}
                    stroke-width="1.5"
                  />
                  <!-- Memory fill background -->
                  <rect
                    x="4"
                    y="4"
                    width={node.iconSize - 8}
                    height={node.iconSize - 8}
                    fill="#0a0a0a"
                  />
                  <!-- Current memory fill -->
                  <rect
                    x="4"
                    y={4 +
                      (node.iconSize - 8) * (1 - node.currentPercent / 100)}
                    width={node.iconSize - 8}
                    height={(node.iconSize - 8) * (node.currentPercent / 100)}
                    fill="#374151"
                  />
                  <!-- New model memory fill -->
                  {#if node.modelUsageGB > 0 && node.isUsed}
                    <rect
                      x="4"
                      y={4 + (node.iconSize - 8) * (1 - node.newPercent / 100)}
                      width={node.iconSize - 8}
                      height={(node.iconSize - 8) *
                        ((node.newPercent - node.currentPercent) / 100)}
                      fill="#FFD700"
                      filter="url(#memGlow-{filterId})"
                      class="animate-pulse-slow"
                    />
                  {/if}
                </g>
              {:else if node.deviceType === "mini"}
                <!-- Mac Mini icon -->
                <g
                  transform="translate({-node.iconSize / 2}, {-node.iconSize /
                    2})"
                >
                  <rect
                    x="2"
                    y={node.iconSize * 0.3}
                    width={node.iconSize - 4}
                    height={node.iconSize * 0.4}
                    rx="3"
                    fill="none"
                    stroke={node.isUsed ? "#FFD700" : "#4B5563"}
                    stroke-width="1.5"
                  />
                  <!-- Memory fill background -->
                  <rect
                    x="4"
                    y={node.iconSize * 0.32}
                    width={node.iconSize - 8}
                    height={node.iconSize * 0.36}
                    fill="#0a0a0a"
                  />
                  <!-- Current memory fill -->
                  <rect
                    x="4"
                    y={node.iconSize * 0.32 +
                      node.iconSize * 0.36 * (1 - node.currentPercent / 100)}
                    width={node.iconSize - 8}
                    height={node.iconSize * 0.36 * (node.currentPercent / 100)}
                    fill="#374151"
                  />
                  <!-- New model memory fill -->
                  {#if node.modelUsageGB > 0 && node.isUsed}
                    <rect
                      x="4"
                      y={node.iconSize * 0.32 +
                        node.iconSize * 0.36 * (1 - node.newPercent / 100)}
                      width={node.iconSize - 8}
                      height={node.iconSize *
                        0.36 *
                        ((node.newPercent - node.currentPercent) / 100)}
                      fill="#FFD700"
                      filter="url(#memGlow-{filterId})"
                      class="animate-pulse-slow"
                    />
                  {/if}
                </g>
              {:else}
                <!-- Unknown device - hexagon -->
                <g
                  transform="translate({-node.iconSize / 2}, {-node.iconSize /
                    2})"
                >
                  <polygon
                    points="{node.iconSize /
                      2},0 {node.iconSize},{node.iconSize *
                      0.25} {node.iconSize},{node.iconSize *
                      0.75} {node.iconSize /
                      2},{node.iconSize} 0,{node.iconSize *
                      0.75} 0,{node.iconSize * 0.25}"
                    fill={node.isUsed ? "rgba(255,215,0,0.1)" : "#0a0a0a"}
                    stroke={node.isUsed ? "#FFD700" : "#4B5563"}
                    stroke-width="1.5"
                  />
                </g>
              {/if}

              <!-- Percentage label -->
              <text
                y={node.iconSize / 2 + 12}
                text-anchor="middle"
                font-size="8"
                font-family="SF Mono, Monaco, monospace"
                fill={node.isUsed
                  ? node.newPercent > 90
                    ? "#f87171"
                    : "#FFD700"
                  : "#4B5563"}
              >
                {node.newPercent.toFixed(0)}%
              </text>
            </g>
          {/each}
        </svg>
      </div>
    {/if}

    <!-- Launch Button -->
    <button
      onclick={onLaunch}
      disabled={isLaunching || !canFit}
      class="w-full py-2 text-sm font-mono tracking-wider uppercase border transition-all duration-200
				{isLaunching
        ? 'bg-transparent text-exo-yellow border-exo-yellow/50 cursor-wait'
        : !canFit
          ? 'bg-red-500/10 text-red-400/70 border-red-500/30 cursor-not-allowed'
          : 'bg-transparent text-exo-light-gray border-exo-light-gray/40 hover:text-exo-yellow hover:border-exo-yellow/50 cursor-pointer'}"
    >
      {#if isLaunching}
        <span class="flex items-center justify-center gap-1.5">
          <span
            class="w-2 h-2 border border-exo-yellow border-t-transparent rounded-full animate-spin"
          ></span>
          LAUNCHING...
        </span>
      {:else if !canFit}
        INSUFFICIENT MEMORY
      {:else}
        ▸ LAUNCH
      {/if}
    </button>
  </div>
</div>

<style>
  @keyframes pulse-slow {
    0%,
    100% {
      opacity: 0.8;
    }
    50% {
      opacity: 1;
    }
  }
  .animate-pulse-slow {
    animation: pulse-slow 1.5s ease-in-out infinite;
  }
</style>

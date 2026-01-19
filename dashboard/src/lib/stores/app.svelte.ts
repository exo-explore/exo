/**
 * AppStore - Central state management for the EXO dashboard
 *
 * Manages:
 * - Chat state (whether a conversation has started)
 * - Topology data from the EXO server
 * - UI state for the topology/chat transition
 */

import { browser } from "$app/environment";

// UUID generation fallback for browsers without crypto.randomUUID
function generateUUID(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  // Fallback implementation
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export interface NodeInfo {
  system_info?: {
    model_id?: string;
    chip?: string;
    memory?: number;
  };
  network_interfaces?: Array<{
    name?: string;
    addresses?: string[];
  }>;
  ip_to_interface?: Record<string, string>;
  macmon_info?: {
    memory?: {
      ram_usage: number;
      ram_total: number;
    };
    temp?: {
      gpu_temp_avg: number;
    };
    gpu_usage?: [number, number];
    sys_power?: number;
  };
  last_macmon_update: number;
  friendly_name?: string;
}

export interface TopologyEdge {
  source: string;
  target: string;
  sendBackIp?: string;
  sendBackInterface?: string;
}

export interface TopologyData {
  nodes: Record<string, NodeInfo>;
  edges: TopologyEdge[];
}

export interface Instance {
  shardAssignments?: {
    modelId?: string;
    runnerToShard?: Record<string, unknown>;
    nodeToRunner?: Record<string, string>;
  };
}

// Granular node state types from the new state structure
interface RawNodeIdentity {
  modelId?: string;
  chipId?: string;
  friendlyName?: string;
}

interface RawMemoryUsage {
  ramTotal?: { inBytes: number };
  ramAvailable?: { inBytes: number };
  swapTotal?: { inBytes: number };
  swapAvailable?: { inBytes: number };
}

interface RawSystemPerformanceProfile {
  gpuUsage?: number;
  temp?: number;
  sysPower?: number;
  pcpuUsage?: number;
  ecpuUsage?: number;
}

interface RawNetworkInterfaceInfo {
  name?: string;
  ipAddress?: string;
  addresses?: Array<{ address?: string } | string>;
  ipv4?: string;
  ipv6?: string;
  ipAddresses?: string[];
  ips?: string[];
}

interface RawNodeNetworkInfo {
  interfaces?: RawNetworkInterfaceInfo[];
}

interface RawSocketConnection {
  sinkMultiaddr?: {
    address?: string;
    ip_address?: string;
    address_type?: string;
    port?: number;
  };
}

interface RawRDMAConnection {
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

type RawConnectionEdge = RawSocketConnection | RawRDMAConnection;

// New nested mapping format: { source: { sink: [edge1, edge2, ...] } }
type RawConnectionsMap = Record<string, Record<string, RawConnectionEdge[]>>;

interface RawTopology {
  nodes: string[];
  connections?: RawConnectionsMap;
}

export interface DownloadProgress {
  totalBytes: number;
  downloadedBytes: number;
  speed: number;
  etaMs: number;
  percentage: number;
  completedFiles: number;
  totalFiles: number;
  files: Array<{
    name: string;
    totalBytes: number;
    downloadedBytes: number;
    speed: number;
    etaMs: number;
    percentage: number;
  }>;
}

export interface ModelDownloadStatus {
  isDownloading: boolean;
  progress: DownloadProgress | null;
  nodeDetails: Array<{
    nodeId: string;
    nodeName: string;
    progress: DownloadProgress;
  }>;
}

// Placement preview from the API
export interface PlacementPreview {
  model_id: string;
  sharding: "Pipeline" | "Tensor";
  instance_meta: "MlxRing" | "MlxIbv" | "MlxJaccl";
  instance: unknown | null;
  memory_delta_by_node: Record<string, number> | null;
  error: string | null;
}

export interface PlacementPreviewResponse {
  previews: PlacementPreview[];
}

interface RawStateResponse {
  topology?: RawTopology;
  instances?: Record<
    string,
    {
      MlxRingInstance?: Instance;
      MlxIbvInstance?: Instance;
      MlxJacclInstance?: Instance;
    }
  >;
  runners?: Record<string, unknown>;
  downloads?: Record<string, unknown[]>;
  // New granular node state fields
  nodeIdentities?: Record<string, RawNodeIdentity>;
  nodeMemory?: Record<string, RawMemoryUsage>;
  nodeSystem?: Record<string, RawSystemPerformanceProfile>;
  nodeNetwork?: Record<string, RawNodeNetworkInfo>;
}

export interface MessageAttachment {
  type: "image" | "text" | "file" | "generated-image";
  name: string;
  content?: string;
  preview?: string;
  mimeType?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  thinking?: string;
  attachments?: MessageAttachment[];
  ttftMs?: number; // Time to first token in ms (for assistant messages)
  tps?: number; // Tokens per second (for assistant messages)
}

export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  modelId: string | null;
  sharding: string | null;
  instanceType: string | null;
}

const STORAGE_KEY = "exo-conversations";
const IMAGE_PARAMS_STORAGE_KEY = "exo-image-generation-params";

// Image generation params interface matching backend API
export interface ImageGenerationParams {
  // Basic params
  size: "512x512" | "768x768" | "1024x1024" | "1024x768" | "768x1024";
  quality: "low" | "medium" | "high";
  outputFormat: "png" | "jpeg";
  // Advanced params
  seed: number | null;
  numInferenceSteps: number | null;
  guidance: number | null;
  negativePrompt: string | null;
  // Edit mode params
  inputFidelity: "low" | "high";
}

// Image being edited
export interface EditingImage {
  imageDataUrl: string;
  sourceMessage: Message;
}

const DEFAULT_IMAGE_PARAMS: ImageGenerationParams = {
  size: "1024x1024",
  quality: "medium",
  outputFormat: "png",
  seed: null,
  numInferenceSteps: null,
  guidance: null,
  negativePrompt: null,
  inputFidelity: "low",
};

interface GranularNodeState {
  nodeIdentities?: Record<string, RawNodeIdentity>;
  nodeMemory?: Record<string, RawMemoryUsage>;
  nodeSystem?: Record<string, RawSystemPerformanceProfile>;
  nodeNetwork?: Record<string, RawNodeNetworkInfo>;
}

function transformNetworkInterface(iface: RawNetworkInterfaceInfo): {
  name?: string;
  addresses: string[];
} {
  const addresses: string[] = [];
  if (iface.ipAddress && typeof iface.ipAddress === "string") {
    addresses.push(iface.ipAddress);
  }
  if (Array.isArray(iface.addresses)) {
    for (const addr of iface.addresses) {
      if (typeof addr === "string") addresses.push(addr);
      else if (addr && typeof addr === "object" && addr.address)
        addresses.push(addr.address);
    }
  }
  if (Array.isArray(iface.ipAddresses)) {
    addresses.push(
      ...iface.ipAddresses.filter((a): a is string => typeof a === "string"),
    );
  }
  if (Array.isArray(iface.ips)) {
    addresses.push(
      ...iface.ips.filter((a): a is string => typeof a === "string"),
    );
  }
  if (iface.ipv4 && typeof iface.ipv4 === "string") addresses.push(iface.ipv4);
  if (iface.ipv6 && typeof iface.ipv6 === "string") addresses.push(iface.ipv6);

  return {
    name: iface.name,
    addresses: Array.from(new Set(addresses)),
  };
}

function transformTopology(
  raw: RawTopology,
  granularState: GranularNodeState,
): TopologyData {
  const nodes: Record<string, NodeInfo> = {};
  const edges: TopologyEdge[] = [];

  for (const nodeId of raw.nodes || []) {
    if (!nodeId) continue;

    // Get data from granular state mappings
    const identity = granularState.nodeIdentities?.[nodeId];
    const memory = granularState.nodeMemory?.[nodeId];
    const system = granularState.nodeSystem?.[nodeId];
    const network = granularState.nodeNetwork?.[nodeId];

    const ramTotal = memory?.ramTotal?.inBytes ?? 0;
    const ramAvailable = memory?.ramAvailable?.inBytes ?? 0;
    const ramUsage = Math.max(ramTotal - ramAvailable, 0);

    const rawInterfaces = network?.interfaces || [];
    const networkInterfaces = rawInterfaces.map(transformNetworkInterface);

    const ipToInterface: Record<string, string> = {};
    for (const iface of networkInterfaces) {
      for (const addr of iface.addresses || []) {
        ipToInterface[addr] = iface.name ?? "";
      }
    }

    nodes[nodeId] = {
      system_info: {
        model_id: identity?.modelId ?? "Unknown",
        chip: identity?.chipId,
        memory: ramTotal,
      },
      network_interfaces: networkInterfaces,
      ip_to_interface: ipToInterface,
      macmon_info: {
        memory: {
          ram_usage: ramUsage,
          ram_total: ramTotal,
        },
        temp:
          system?.temp !== undefined
            ? { gpu_temp_avg: system.temp }
            : undefined,
        gpu_usage:
          system?.gpuUsage !== undefined ? [0, system.gpuUsage] : undefined,
        sys_power: system?.sysPower,
      },
      last_macmon_update: Date.now() / 1000,
      friendly_name: identity?.friendlyName,
    };
  }

  // Handle connections - nested mapping format { source: { sink: [edges] } }
  const connections = raw.connections;
  if (connections && typeof connections === "object") {
    for (const [source, sinks] of Object.entries(connections)) {
      if (!sinks || typeof sinks !== "object") continue;
      for (const [sink, edgeList] of Object.entries(sinks)) {
        if (!Array.isArray(edgeList)) continue;
        for (const edge of edgeList) {
          let sendBackIp: string | undefined;
          if (edge && typeof edge === "object" && "sinkMultiaddr" in edge) {
            const multiaddr = edge.sinkMultiaddr;
            if (multiaddr) {
              sendBackIp =
                multiaddr.ip_address ||
                extractIpFromMultiaddr(multiaddr.address);
            }
          }

          if (nodes[source] && nodes[sink] && source !== sink) {
            edges.push({ source, target: sink, sendBackIp });
          }
        }
      }
    }
  }

  return { nodes, edges };
}

function extractIpFromMultiaddr(ma?: string): string | undefined {
  if (!ma) return undefined;
  const parts = ma.split("/");
  const ip4Idx = parts.indexOf("ip4");
  const ip6Idx = parts.indexOf("ip6");
  const idx = ip4Idx >= 0 ? ip4Idx : ip6Idx;
  if (idx >= 0 && parts.length > idx + 1) {
    return parts[idx + 1];
  }
  return undefined;
}

class AppStore {
  // Conversation state
  conversations = $state<Conversation[]>([]);
  activeConversationId = $state<string | null>(null);

  // Chat state
  hasStartedChat = $state(false);
  messages = $state<Message[]>([]);
  currentResponse = $state("");
  isLoading = $state(false);

  // Performance metrics
  ttftMs = $state<number | null>(null); // Time to first token in ms
  tps = $state<number | null>(null); // Tokens per second
  totalTokens = $state<number>(0); // Total tokens in current response

  // Topology state
  topologyData = $state<TopologyData | null>(null);
  instances = $state<Record<string, unknown>>({});
  runners = $state<Record<string, unknown>>({});
  downloads = $state<Record<string, unknown[]>>({});
  placementPreviews = $state<PlacementPreview[]>([]);
  selectedPreviewModelId = $state<string | null>(null);
  isLoadingPreviews = $state(false);
  lastUpdate = $state<number | null>(null);

  // UI state
  isTopologyMinimized = $state(false);
  isSidebarOpen = $state(false); // Hidden by default, shown when in chat mode
  debugMode = $state(false);
  topologyOnlyMode = $state(false);
  chatSidebarVisible = $state(true); // Shown by default

  // Image generation params
  imageGenerationParams = $state<ImageGenerationParams>({
    ...DEFAULT_IMAGE_PARAMS,
  });

  // Image editing state
  editingImage = $state<EditingImage | null>(null);

  private fetchInterval: ReturnType<typeof setInterval> | null = null;
  private previewsInterval: ReturnType<typeof setInterval> | null = null;
  private lastConversationPersistTs = 0;

  constructor() {
    if (browser) {
      this.startPolling();
      this.loadConversationsFromStorage();
      this.loadDebugModeFromStorage();
      this.loadTopologyOnlyModeFromStorage();
      this.loadChatSidebarVisibleFromStorage();
      this.loadImageGenerationParamsFromStorage();
    }
  }

  /**
   * Load conversations from localStorage
   */
  private loadConversationsFromStorage() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Array<Partial<Conversation>>;
        this.conversations = parsed.map((conversation) => ({
          id: conversation.id ?? generateUUID(),
          name: conversation.name ?? "Chat",
          messages: conversation.messages ?? [],
          createdAt: conversation.createdAt ?? Date.now(),
          updatedAt: conversation.updatedAt ?? Date.now(),
          modelId: conversation.modelId ?? null,
          sharding: conversation.sharding ?? null,
          instanceType: conversation.instanceType ?? null,
        }));
      }
    } catch (error) {
      console.error("Failed to load conversations:", error);
    }
  }

  /**
   * Save conversations to localStorage
   */
  private saveConversationsToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.conversations));
    } catch (error) {
      console.error("Failed to save conversations:", error);
    }
  }

  private loadDebugModeFromStorage() {
    try {
      const stored = localStorage.getItem("exo-debug-mode");
      if (stored !== null) {
        this.debugMode = stored === "true";
      }
    } catch (error) {
      console.error("Failed to load debug mode:", error);
    }
  }

  private saveDebugModeToStorage() {
    try {
      localStorage.setItem("exo-debug-mode", this.debugMode ? "true" : "false");
    } catch (error) {
      console.error("Failed to save debug mode:", error);
    }
  }

  private loadTopologyOnlyModeFromStorage() {
    try {
      const stored = localStorage.getItem("exo-topology-only-mode");
      if (stored !== null) {
        this.topologyOnlyMode = stored === "true";
      }
    } catch (error) {
      console.error("Failed to load topology only mode:", error);
    }
  }

  private saveTopologyOnlyModeToStorage() {
    try {
      localStorage.setItem(
        "exo-topology-only-mode",
        this.topologyOnlyMode ? "true" : "false",
      );
    } catch (error) {
      console.error("Failed to save topology only mode:", error);
    }
  }

  private loadChatSidebarVisibleFromStorage() {
    try {
      const stored = localStorage.getItem("exo-chat-sidebar-visible");
      if (stored !== null) {
        this.chatSidebarVisible = stored === "true";
      }
    } catch (error) {
      console.error("Failed to load chat sidebar visibility:", error);
    }
  }

  private saveChatSidebarVisibleToStorage() {
    try {
      localStorage.setItem(
        "exo-chat-sidebar-visible",
        this.chatSidebarVisible ? "true" : "false",
      );
    } catch (error) {
      console.error("Failed to save chat sidebar visibility:", error);
    }
  }

  private loadImageGenerationParamsFromStorage() {
    try {
      const stored = localStorage.getItem(IMAGE_PARAMS_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Partial<ImageGenerationParams>;
        this.imageGenerationParams = {
          ...DEFAULT_IMAGE_PARAMS,
          ...parsed,
        };
      }
    } catch (error) {
      console.error("Failed to load image generation params:", error);
    }
  }

  private saveImageGenerationParamsToStorage() {
    try {
      localStorage.setItem(
        IMAGE_PARAMS_STORAGE_KEY,
        JSON.stringify(this.imageGenerationParams),
      );
    } catch (error) {
      console.error("Failed to save image generation params:", error);
    }
  }

  getImageGenerationParams(): ImageGenerationParams {
    return this.imageGenerationParams;
  }

  setImageGenerationParams(params: Partial<ImageGenerationParams>) {
    this.imageGenerationParams = {
      ...this.imageGenerationParams,
      ...params,
    };
    this.saveImageGenerationParamsToStorage();
  }

  resetImageGenerationParams() {
    this.imageGenerationParams = { ...DEFAULT_IMAGE_PARAMS };
    this.saveImageGenerationParamsToStorage();
  }

  setEditingImage(imageDataUrl: string, sourceMessage: Message) {
    this.editingImage = { imageDataUrl, sourceMessage };
  }

  clearEditingImage() {
    this.editingImage = null;
  }

  /**
   * Create a new conversation
   */
  createConversation(name?: string): string {
    const id = generateUUID();
    const now = Date.now();

    // Try to derive model and strategy immediately from selected model or running instances
    let derivedModelId = this.selectedChatModel || null;
    let derivedInstanceType: string | null = null;
    let derivedSharding: string | null = null;

    // If no selected model, fall back to the first running instance
    if (!derivedModelId) {
      const firstInstance = Object.values(this.instances)[0];
      if (firstInstance) {
        const candidateModel = this.extractInstanceModelId(firstInstance);
        derivedModelId = candidateModel ?? null;
        const details = this.describeInstance(firstInstance);
        derivedInstanceType = details.instanceType;
        derivedSharding = details.sharding;
      }
    } else {
      // If selected model is set, attempt to get its details from instances
      for (const [, instanceWrapper] of Object.entries(this.instances)) {
        const candidateModelId = this.extractInstanceModelId(instanceWrapper);
        if (candidateModelId === derivedModelId) {
          const details = this.describeInstance(instanceWrapper);
          derivedInstanceType = details.instanceType;
          derivedSharding = details.sharding;
          break;
        }
      }
    }

    const conversation: Conversation = {
      id,
      name:
        name ||
        `Chat ${new Date(now).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`,
      messages: [],
      createdAt: now,
      updatedAt: now,
      modelId: derivedModelId,
      sharding: derivedSharding,
      instanceType: derivedInstanceType,
    };

    this.conversations.unshift(conversation);
    this.activeConversationId = id;
    this.messages = [];
    this.hasStartedChat = true;
    this.isTopologyMinimized = true;
    this.isSidebarOpen = true; // Auto-open sidebar when chatting

    this.saveConversationsToStorage();
    return id;
  }

  /**
   * Load a conversation by ID
   */
  loadConversation(id: string): boolean {
    const conversation = this.conversations.find((c) => c.id === id);
    if (!conversation) return false;

    this.activeConversationId = id;
    this.messages = [...conversation.messages];
    this.hasStartedChat = true;
    this.isTopologyMinimized = true;
    this.isSidebarOpen = true; // Auto-open sidebar when chatting
    this.refreshConversationModelFromInstances();

    return true;
  }

  /**
   * Delete a conversation by ID
   */
  deleteConversation(id: string) {
    this.conversations = this.conversations.filter((c) => c.id !== id);

    if (this.activeConversationId === id) {
      this.activeConversationId = null;
      this.messages = [];
      this.hasStartedChat = false;
      this.isTopologyMinimized = false;
    }

    this.saveConversationsToStorage();
  }

  /**
   * Delete all conversations
   */
  deleteAllConversations() {
    this.conversations = [];
    this.activeConversationId = null;
    this.messages = [];
    this.hasStartedChat = false;
    this.isTopologyMinimized = false;
    this.saveConversationsToStorage();
  }

  /**
   * Rename a conversation
   */
  renameConversation(id: string, newName: string) {
    const conversation = this.conversations.find((c) => c.id === id);
    if (conversation) {
      conversation.name = newName;
      conversation.updatedAt = Date.now();
      this.saveConversationsToStorage();
    }
  }

  private getTaggedValue(obj: unknown): [string | null, unknown] {
    if (!obj || typeof obj !== "object") return [null, null];
    const keys = Object.keys(obj as Record<string, unknown>);
    if (keys.length === 1) {
      return [keys[0], (obj as Record<string, unknown>)[keys[0]]];
    }
    return [null, null];
  }

  private extractInstanceModelId(instanceWrapped: unknown): string | null {
    const [, instance] = this.getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") return null;
    const inst = instance as { shardAssignments?: { modelId?: string } };
    return inst.shardAssignments?.modelId ?? null;
  }

  private describeInstance(instanceWrapped: unknown): {
    sharding: string | null;
    instanceType: string | null;
  } {
    const [instanceTag, instance] = this.getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return { sharding: null, instanceType: null };
    }

    let instanceType: string | null = null;
    if (instanceTag === "MlxRingInstance") instanceType = "MLX Ring";
    else if (
      instanceTag === "MlxIbvInstance" ||
      instanceTag === "MlxJacclInstance"
    )
      instanceType = "MLX RDMA";

    let sharding: string | null = null;
    const inst = instance as {
      shardAssignments?: { runnerToShard?: Record<string, unknown> };
    };
    const runnerToShard = inst.shardAssignments?.runnerToShard || {};
    const firstShardWrapped = Object.values(runnerToShard)[0];
    if (firstShardWrapped) {
      const [shardTag] = this.getTaggedValue(firstShardWrapped);
      if (shardTag === "PipelineShardMetadata") sharding = "Pipeline";
      else if (shardTag === "TensorShardMetadata") sharding = "Tensor";
      else if (shardTag === "PrefillDecodeShardMetadata")
        sharding = "Prefill/Decode";
    }

    return { sharding, instanceType };
  }

  private buildConversationModelInfo(modelId: string): {
    modelId: string;
    sharding: string | null;
    instanceType: string | null;
  } {
    let sharding: string | null = null;
    let instanceType: string | null = null;

    for (const [, instanceWrapper] of Object.entries(this.instances)) {
      const candidateModelId = this.extractInstanceModelId(instanceWrapper);
      if (candidateModelId === modelId) {
        const details = this.describeInstance(instanceWrapper);
        sharding = details.sharding;
        instanceType = details.instanceType;
        break;
      }
    }

    return { modelId, sharding, instanceType };
  }

  private applyConversationModelInfo(info: {
    modelId: string;
    sharding: string | null;
    instanceType: string | null;
  }) {
    if (!this.activeConversationId) return;
    const conversation = this.conversations.find(
      (c) => c.id === this.activeConversationId,
    );
    if (!conversation) return;

    // Keep the first known modelId stable; only backfill if missing
    if (!conversation.modelId) {
      conversation.modelId = info.modelId;
    }
    conversation.sharding = info.sharding;
    conversation.instanceType = info.instanceType;
    this.saveConversationsToStorage();
  }

  private getModelTail(modelId: string): string {
    const parts = modelId.split("/");
    return (parts[parts.length - 1] || modelId).toLowerCase();
  }

  private isBetterModelId(
    currentId: string | null,
    candidateId: string | null,
  ): boolean {
    if (!candidateId) return false;
    if (!currentId) return true;
    const currentTail = this.getModelTail(currentId);
    const candidateTail = this.getModelTail(candidateId);
    return (
      candidateTail.length > currentTail.length &&
      candidateTail.startsWith(currentTail)
    );
  }

  private refreshConversationModelFromInstances() {
    if (!this.activeConversationId) return;
    const conversation = this.conversations.find(
      (c) => c.id === this.activeConversationId,
    );
    if (!conversation) return;

    // Prefer stored model; do not replace it once set. Only backfill when missing.
    let modelId = conversation.modelId;

    // If missing, try the selected model
    if (!modelId && this.selectedChatModel) {
      modelId = this.selectedChatModel;
    }

    // If still missing, fall back to first instance model
    if (!modelId) {
      const firstInstance = Object.values(this.instances)[0];
      if (firstInstance) {
        modelId = this.extractInstanceModelId(firstInstance);
      }
    }

    if (!modelId) return;

    // If a more specific instance modelId is available (e.g., adds "-4bit"), prefer it
    let preferredModelId = modelId;
    for (const [, instanceWrapper] of Object.entries(this.instances)) {
      const candidate = this.extractInstanceModelId(instanceWrapper);
      if (!candidate) continue;
      if (candidate === preferredModelId) {
        break;
      }
      if (this.isBetterModelId(preferredModelId, candidate)) {
        preferredModelId = candidate;
      }
    }

    if (this.isBetterModelId(conversation.modelId, preferredModelId)) {
      conversation.modelId = preferredModelId;
    }

    const info = this.buildConversationModelInfo(preferredModelId);
    const hasNewInfo = Boolean(
      info.sharding || info.instanceType || !conversation.modelId,
    );
    if (hasNewInfo) {
      this.applyConversationModelInfo(info);
    }
  }

  getDebugMode(): boolean {
    return this.debugMode;
  }

  /**
   * Update the active conversation with current messages
   */
  private updateActiveConversation() {
    if (!this.activeConversationId) return;

    const conversation = this.conversations.find(
      (c) => c.id === this.activeConversationId,
    );
    if (conversation) {
      conversation.messages = [...this.messages];
      conversation.updatedAt = Date.now();

      // Auto-generate name from first user message if still has default name
      if (conversation.name.startsWith("Chat ")) {
        const firstUserMsg = conversation.messages.find(
          (m) => m.role === "user" && m.content.trim(),
        );
        if (firstUserMsg) {
          // Clean up the content - remove file context markers and whitespace
          let content = firstUserMsg.content
            .replace(/\[File:.*?\][\s\S]*?```[\s\S]*?```/g, "") // Remove file attachments
            .trim();

          if (content) {
            const preview = content.slice(0, 50);
            conversation.name =
              preview.length < content.length ? preview + "..." : preview;
          }
        }
      }

      this.saveConversationsToStorage();
    }
  }

  private persistActiveConversation(throttleMs = 400) {
    const now = Date.now();
    if (now - this.lastConversationPersistTs < throttleMs) return;
    this.lastConversationPersistTs = now;
    this.updateActiveConversation();
  }

  /**
   * Toggle sidebar visibility
   */
  toggleSidebar() {
    this.isSidebarOpen = !this.isSidebarOpen;
  }

  setDebugMode(enabled: boolean) {
    this.debugMode = enabled;
    this.saveDebugModeToStorage();
  }

  toggleDebugMode() {
    this.debugMode = !this.debugMode;
    this.saveDebugModeToStorage();
  }

  getTopologyOnlyMode(): boolean {
    return this.topologyOnlyMode;
  }

  setTopologyOnlyMode(enabled: boolean) {
    this.topologyOnlyMode = enabled;
    this.saveTopologyOnlyModeToStorage();
  }

  toggleTopologyOnlyMode() {
    this.topologyOnlyMode = !this.topologyOnlyMode;
    this.saveTopologyOnlyModeToStorage();
  }

  getChatSidebarVisible(): boolean {
    return this.chatSidebarVisible;
  }

  setChatSidebarVisible(visible: boolean) {
    this.chatSidebarVisible = visible;
    this.saveChatSidebarVisibleToStorage();
  }

  toggleChatSidebarVisible() {
    this.chatSidebarVisible = !this.chatSidebarVisible;
    this.saveChatSidebarVisibleToStorage();
  }

  startPolling() {
    this.fetchState();
    this.fetchInterval = setInterval(() => this.fetchState(), 1000);
  }

  stopPolling() {
    if (this.fetchInterval) {
      clearInterval(this.fetchInterval);
      this.fetchInterval = null;
    }
    this.stopPreviewsPolling();
  }

  async fetchState() {
    try {
      const response = await fetch("/state");
      if (!response.ok) {
        throw new Error(`Failed to fetch state: ${response.status}`);
      }
      const data: RawStateResponse = await response.json();

      if (data.topology) {
        this.topologyData = transformTopology(data.topology, {
          nodeIdentities: data.nodeIdentities,
          nodeMemory: data.nodeMemory,
          nodeSystem: data.nodeSystem,
          nodeNetwork: data.nodeNetwork,
        });
      }
      if (data.instances) {
        this.instances = data.instances;
        this.refreshConversationModelFromInstances();
      }
      if (data.runners) {
        this.runners = data.runners;
      }
      if (data.downloads) {
        this.downloads = data.downloads;
      }
      this.lastUpdate = Date.now();
    } catch (error) {
      console.error("Error fetching state:", error);
    }
  }

  async fetchPlacementPreviews(modelId: string, showLoading = true) {
    if (!modelId) return;

    if (showLoading) {
      this.isLoadingPreviews = true;
    }
    this.selectedPreviewModelId = modelId;

    try {
      const response = await fetch(
        `/instance/previews?model_id=${encodeURIComponent(modelId)}`,
      );
      if (!response.ok) {
        throw new Error(
          `Failed to fetch placement previews: ${response.status}`,
        );
      }
      const data: PlacementPreviewResponse = await response.json();
      this.placementPreviews = data.previews;
    } catch (error) {
      console.error("Error fetching placement previews:", error);
      this.placementPreviews = [];
    } finally {
      if (showLoading) {
        this.isLoadingPreviews = false;
      }
    }
  }

  startPreviewsPolling(modelId: string) {
    // Stop any existing preview polling
    this.stopPreviewsPolling();

    // Fetch immediately
    this.fetchPlacementPreviews(modelId);

    // Then poll every 15 seconds (don't show loading spinner for subsequent fetches)
    this.previewsInterval = setInterval(() => {
      if (this.selectedPreviewModelId) {
        this.fetchPlacementPreviews(this.selectedPreviewModelId, false);
      }
    }, 15000);
  }

  stopPreviewsPolling() {
    if (this.previewsInterval) {
      clearInterval(this.previewsInterval);
      this.previewsInterval = null;
    }
  }

  selectPreviewModel(modelId: string | null) {
    if (modelId) {
      this.startPreviewsPolling(modelId);
    } else {
      this.stopPreviewsPolling();
      this.selectedPreviewModelId = null;
      this.placementPreviews = [];
    }
  }

  /**
   * Starts a chat conversation - triggers the topology minimization animation
   * Creates a new conversation if none is active
   */
  startChat() {
    if (!this.activeConversationId) {
      this.createConversation();
    } else {
      this.hasStartedChat = true;
      this.isSidebarOpen = true; // Auto-open sidebar when chatting
      // Small delay before minimizing for a nice visual effect
      setTimeout(() => {
        this.isTopologyMinimized = true;
      }, 100);
    }
  }

  /**
   * Add a message to the conversation
   */
  addMessage(role: "user" | "assistant", content: string) {
    const message: Message = {
      id: generateUUID(),
      role,
      content,
      timestamp: Date.now(),
    };
    this.messages.push(message);
    return message;
  }

  /**
   * Delete a message and all subsequent messages
   */
  deleteMessage(messageId: string) {
    const messageIndex = this.messages.findIndex((m) => m.id === messageId);
    if (messageIndex === -1) return;

    // Remove this message and all subsequent messages
    this.messages = this.messages.slice(0, messageIndex);
    this.updateActiveConversation();
  }

  /**
   * Edit a user message content (does not regenerate response)
   */
  editMessage(messageId: string, newContent: string) {
    const message = this.messages.find((m) => m.id === messageId);
    if (!message) return;

    message.content = newContent;
    message.timestamp = Date.now();
    this.updateActiveConversation();
  }

  /**
   * Edit a user message and regenerate the response
   */
  async editAndRegenerate(
    messageId: string,
    newContent: string,
  ): Promise<void> {
    const messageIndex = this.messages.findIndex((m) => m.id === messageId);
    if (messageIndex === -1) return;

    const message = this.messages[messageIndex];
    if (message.role !== "user") return;

    // Update the message content
    message.content = newContent;
    message.timestamp = Date.now();

    // Remove all messages after this one (including the assistant response)
    this.messages = this.messages.slice(0, messageIndex + 1);

    // Regenerate the response
    await this.regenerateLastResponse();
  }

  /**
   * Regenerate the last assistant response
   */
  async regenerateLastResponse(): Promise<void> {
    if (this.isLoading) return;

    // Find the last user message
    let lastUserIndex = -1;
    for (let i = this.messages.length - 1; i >= 0; i--) {
      if (this.messages[i].role === "user") {
        lastUserIndex = i;
        break;
      }
    }

    if (lastUserIndex === -1) return;

    // Remove any messages after the user message
    this.messages = this.messages.slice(0, lastUserIndex + 1);

    // Resend the message to get a new response
    this.isLoading = true;
    this.currentResponse = "";

    // Create placeholder for assistant message
    const assistantMessage = this.addMessage("assistant", "");

    try {
      const systemPrompt = {
        role: "system" as const,
        content:
          "You are a helpful AI assistant. Respond directly and concisely. Do not show your reasoning or thought process.",
      };

      const apiMessages = [
        systemPrompt,
        ...this.messages.slice(0, -1).map((m) => {
          return { role: m.role, content: m.content };
        }),
      ];

      // Determine which model to use
      let modelToUse = this.selectedChatModel;
      if (!modelToUse) {
        const firstInstanceKey = Object.keys(this.instances)[0];
        if (firstInstanceKey) {
          const instance = this.instances[firstInstanceKey] as
            | Record<string, unknown>
            | undefined;
          if (instance) {
            const keys = Object.keys(instance);
            if (keys.length === 1) {
              const inst = instance[keys[0]] as
                | { shardAssignments?: { modelId?: string } }
                | undefined;
              modelToUse = inst?.shardAssignments?.modelId || "";
            }
          }
        }
      }

      if (!modelToUse) {
        const idx = this.messages.findIndex(
          (m) => m.id === assistantMessage.id,
        );
        if (idx !== -1) {
          this.messages[idx].content =
            "Error: No model available. Please launch an instance first.";
        }
        this.isLoading = false;
        this.updateActiveConversation();
        return;
      }

      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelToUse,
          messages: apiMessages,
          stream: true,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        const idx = this.messages.findIndex(
          (m) => m.id === assistantMessage.id,
        );
        if (idx !== -1) {
          this.messages[idx].content =
            `Error: ${response.status} - ${errorText}`;
        }
        this.isLoading = false;
        this.updateActiveConversation();
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        const idx = this.messages.findIndex(
          (m) => m.id === assistantMessage.id,
        );
        if (idx !== -1) {
          this.messages[idx].content = "Error: No response stream available";
        }
        this.isLoading = false;
        this.updateActiveConversation();
        return;
      }

      const decoder = new TextDecoder();
      let fullContent = "";
      let partialLine = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunk).split("\n");
        partialLine = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === "data: [DONE]") continue;

          if (trimmed.startsWith("data: ")) {
            try {
              const json = JSON.parse(trimmed.slice(6));
              const delta = json.choices?.[0]?.delta?.content;
              if (delta) {
                fullContent += delta;
                const { displayContent, thinkingContent } =
                  this.stripThinkingTags(fullContent);
                this.currentResponse = displayContent;

                // Update the assistant message in place (triggers Svelte reactivity)
                const idx = this.messages.findIndex(
                  (m) => m.id === assistantMessage.id,
                );
                if (idx !== -1) {
                  this.messages[idx].content = displayContent;
                  this.messages[idx].thinking = thinkingContent || undefined;
                }
                this.persistActiveConversation();
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }

      // Final cleanup of the message
      const { displayContent, thinkingContent } =
        this.stripThinkingTags(fullContent);
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (idx !== -1) {
        this.messages[idx].content = displayContent;
        this.messages[idx].thinking = thinkingContent || undefined;
      }
      this.persistActiveConversation();
    } catch (error) {
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (idx !== -1) {
        this.messages[idx].content =
          `Error: ${error instanceof Error ? error.message : "Unknown error"}`;
      }
      this.persistActiveConversation();
    } finally {
      this.isLoading = false;
      this.currentResponse = "";
      this.updateActiveConversation();
    }
  }

  /**
   * Selected model for chat (can be set by the UI)
   */
  selectedChatModel = $state("");

  /**
   * Set the model to use for chat
   */
  setSelectedModel(modelId: string) {
    this.selectedChatModel = modelId;
    // Clear stats when model changes
    this.ttftMs = null;
    this.tps = null;
  }

  /**
   * Strip thinking tags from content for display.
   * Handles both complete <think>...</think> blocks and in-progress <think>... blocks during streaming.
   */
  private stripThinkingTags(content: string): {
    displayContent: string;
    thinkingContent: string;
  } {
    const extracted: string[] = [];
    let displayContent = content;

    // Extract complete <think>...</think> blocks
    const completeBlockRegex = /<think>([\s\S]*?)<\/think>/gi;
    let match: RegExpExecArray | null;
    while ((match = completeBlockRegex.exec(content)) !== null) {
      const inner = match[1]?.trim();
      if (inner) extracted.push(inner);
    }
    displayContent = displayContent.replace(completeBlockRegex, "");

    // Handle in-progress thinking block (has <think> but no closing </think> yet)
    const openTagIndex = displayContent.lastIndexOf("<think>");
    if (openTagIndex !== -1) {
      const inProgressThinking = displayContent.slice(openTagIndex + 7).trim();
      if (inProgressThinking) {
        extracted.push(inProgressThinking);
      }
      displayContent = displayContent.slice(0, openTagIndex);
    }

    return {
      displayContent: displayContent.trim(),
      thinkingContent: extracted.join("\n\n"),
    };
  }

  /**
   * Send a message to the LLM and stream the response
   */
  async sendMessage(
    content: string,
    files?: {
      id: string;
      name: string;
      type: string;
      textContent?: string;
      preview?: string;
    }[],
  ): Promise<void> {
    if ((!content.trim() && (!files || files.length === 0)) || this.isLoading)
      return;

    if (!this.hasStartedChat) {
      this.startChat();
    }

    this.isLoading = true;
    this.currentResponse = "";
    this.ttftMs = null;
    this.tps = null;
    this.totalTokens = 0;

    // Build attachments from files
    const attachments: MessageAttachment[] = [];
    let fileContext = "";

    if (files && files.length > 0) {
      for (const file of files) {
        const isImage = file.type.startsWith("image/");

        if (isImage && file.preview) {
          attachments.push({
            type: "image",
            name: file.name,
            preview: file.preview,
            mimeType: file.type,
          });
        } else if (file.textContent) {
          attachments.push({
            type: "text",
            name: file.name,
            content: file.textContent,
            mimeType: file.type,
          });
          // Add text file content to the message context
          fileContext += `\n\n[File: ${file.name}]\n\`\`\`\n${file.textContent}\n\`\`\``;
        } else {
          attachments.push({
            type: "file",
            name: file.name,
            mimeType: file.type,
          });
        }
      }
    }

    // Combine content with file context
    const fullContent = content + fileContext;

    // Add user message with attachments
    const userMessage: Message = {
      id: generateUUID(),
      role: "user",
      content: content, // Store original content for display
      timestamp: Date.now(),
      attachments: attachments.length > 0 ? attachments : undefined,
    };
    this.messages.push(userMessage);

    // Create placeholder for assistant message
    const assistantMessage = this.addMessage("assistant", "");
    this.updateActiveConversation();

    try {
      // Build the messages array for the API with system prompt
      const systemPrompt = {
        role: "system" as const,
        content:
          "You are a helpful AI assistant. Respond directly and concisely. Do not show your reasoning or thought process. When files are shared with you, analyze them and respond helpfully.",
      };

      // Build API messages - include file content for text files
      const apiMessages = [
        systemPrompt,
        ...this.messages.slice(0, -1).map((m) => {
          // Build content including any text file attachments
          let msgContent = m.content;

          // Add text attachments as context
          if (m.attachments) {
            for (const attachment of m.attachments) {
              if (attachment.type === "text" && attachment.content) {
                msgContent += `\n\n[File: ${attachment.name}]\n\`\`\`\n${attachment.content}\n\`\`\``;
              }
            }
          }

          return {
            role: m.role,
            content: msgContent,
          };
        }),
      ];

      // Determine the model to use - prefer selectedChatModel, otherwise try to get from instances
      let modelToUse = this.selectedChatModel;
      if (!modelToUse) {
        // Try to get model from first running instance
        for (const [, instanceWrapper] of Object.entries(this.instances)) {
          if (instanceWrapper && typeof instanceWrapper === "object") {
            const keys = Object.keys(
              instanceWrapper as Record<string, unknown>,
            );
            if (keys.length === 1) {
              const instance = (instanceWrapper as Record<string, unknown>)[
                keys[0]
              ] as { shardAssignments?: { modelId?: string } };
              if (instance?.shardAssignments?.modelId) {
                modelToUse = instance.shardAssignments.modelId;
                break;
              }
            }
          }
        }
      }

      if (!modelToUse) {
        throw new Error(
          "No model selected and no running instances available. Please launch an instance first.",
        );
      }

      const conversationModelInfo = this.buildConversationModelInfo(modelToUse);
      this.applyConversationModelInfo(conversationModelInfo);

      // Start timing for TTFT measurement
      const requestStartTime = performance.now();
      let firstTokenTime: number | null = null;
      let tokenCount = 0;

      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: modelToUse,
          messages: apiMessages,
          temperature: 0.7,
          stream: true,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      const decoder = new TextDecoder();
      let fullContent = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith("data: ")) {
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;

            try {
              const parsed = JSON.parse(data);
              const tokenContent = parsed.choices?.[0]?.delta?.content;
              if (tokenContent) {
                // Track first token for TTFT
                if (firstTokenTime === null) {
                  firstTokenTime = performance.now();
                  this.ttftMs = firstTokenTime - requestStartTime;
                }

                // Count tokens (each SSE chunk is typically one token)
                tokenCount += 1;
                this.totalTokens = tokenCount;

                // Update real-time TPS during streaming
                if (firstTokenTime !== null && tokenCount > 1) {
                  const elapsed = performance.now() - firstTokenTime;
                  this.tps = (tokenCount / elapsed) * 1000;
                }

                fullContent += tokenContent;

                // Strip thinking tags for display and extract thinking content
                const { displayContent, thinkingContent } =
                  this.stripThinkingTags(fullContent);
                this.currentResponse = displayContent;

                // Update the assistant message in place
                const idx = this.messages.findIndex(
                  (m) => m.id === assistantMessage.id,
                );
                if (idx !== -1) {
                  this.messages[idx].content = displayContent;
                  this.messages[idx].thinking = thinkingContent || undefined;
                }
                this.persistActiveConversation();
              }
            } catch {
              // Skip invalid JSON lines
            }
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        const trimmed = buffer.trim();
        if (trimmed.startsWith("data: ") && trimmed.slice(6) !== "[DONE]") {
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            const tokenContent = parsed.choices?.[0]?.delta?.content;
            if (tokenContent) {
              fullContent += tokenContent;
              this.persistActiveConversation();
            }
          } catch {
            // Skip
          }
        }
      }

      // Calculate final TPS
      if (firstTokenTime !== null && tokenCount > 1) {
        const totalGenerationTime = performance.now() - firstTokenTime;
        this.tps = (tokenCount / totalGenerationTime) * 1000; // tokens per second
      }

      // Final cleanup of the message
      const { displayContent, thinkingContent } =
        this.stripThinkingTags(fullContent);
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (idx !== -1) {
        this.messages[idx].content = displayContent;
        this.messages[idx].thinking = thinkingContent || undefined;
        // Store performance metrics on the message
        if (this.ttftMs !== null) {
          this.messages[idx].ttftMs = this.ttftMs;
        }
        if (this.tps !== null) {
          this.messages[idx].tps = this.tps;
        }
      }
      this.persistActiveConversation();
    } catch (error) {
      console.error("Error sending message:", error);
      // Update the assistant message with error
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (idx !== -1) {
        this.messages[idx].content =
          `Error: ${error instanceof Error ? error.message : "Failed to get response"}`;
      }
      this.persistActiveConversation();
    } finally {
      this.isLoading = false;
      this.currentResponse = "";
      this.updateActiveConversation();
    }
  }

  /**
   * Generate an image using the image generation API
   */
  async generateImage(prompt: string, modelId?: string): Promise<void> {
    if (!prompt.trim() || this.isLoading) return;

    if (!this.hasStartedChat) {
      this.startChat();
    }

    this.isLoading = true;
    this.currentResponse = "";

    // Add user message
    const userMessage: Message = {
      id: generateUUID(),
      role: "user",
      content: prompt,
      timestamp: Date.now(),
    };
    this.messages.push(userMessage);

    // Create placeholder for assistant message with generating state
    const assistantMessage = this.addMessage("assistant", "");
    this.messages[this.messages.length - 1].content = "Generating image...";
    this.updateActiveConversation();

    try {
      // Determine the model to use
      let model = modelId || this.selectedChatModel;
      if (!model) {
        throw new Error(
          "No model selected. Please select an image generation model.",
        );
      }

      // Build request body using image generation params
      const params = this.imageGenerationParams;
      const hasAdvancedParams =
        params.seed !== null ||
        params.numInferenceSteps !== null ||
        params.guidance !== null ||
        (params.negativePrompt !== null && params.negativePrompt.trim() !== "");

      const requestBody: Record<string, unknown> = {
        model,
        prompt,
        quality: params.quality,
        size: params.size,
        output_format: params.outputFormat,
        response_format: "b64_json",
        stream: true,
        partial_images: 3,
      };

      if (hasAdvancedParams) {
        requestBody.advanced_params = {
          ...(params.seed !== null && { seed: params.seed }),
          ...(params.numInferenceSteps !== null && {
            num_inference_steps: params.numInferenceSteps,
          }),
          ...(params.guidance !== null && { guidance: params.guidance }),
          ...(params.negativePrompt !== null &&
            params.negativePrompt.trim() !== "" && {
              negative_prompt: params.negativePrompt,
            }),
        };
      }

      const response = await fetch("/v1/images/generations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith("data: ")) {
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;

            try {
              const parsed = JSON.parse(data);
              const imageData = parsed.data?.b64_json;

              if (imageData && idx !== -1) {
                const format = parsed.format || "png";
                const mimeType = `image/${format}`;
                if (parsed.type === "partial") {
                  // Update with partial image and progress
                  const partialNum = (parsed.partial_index ?? 0) + 1;
                  const totalPartials = parsed.total_partials ?? 3;
                  this.messages[idx].content =
                    `Generating... ${partialNum}/${totalPartials}`;
                  this.messages[idx].attachments = [
                    {
                      type: "generated-image",
                      name: `generated-image.${format}`,
                      preview: `data:${mimeType};base64,${imageData}`,
                      mimeType,
                    },
                  ];
                } else if (parsed.type === "final") {
                  // Final image
                  this.messages[idx].content = "";
                  this.messages[idx].attachments = [
                    {
                      type: "generated-image",
                      name: `generated-image.${format}`,
                      preview: `data:${mimeType};base64,${imageData}`,
                      mimeType,
                    },
                  ];
                }
              }
            } catch {
              // Ignore parse errors for incomplete JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Error generating image:", error);
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (idx !== -1) {
        this.messages[idx].content =
          `Error: ${error instanceof Error ? error.message : "Failed to generate image"}`;
      }
    } finally {
      this.isLoading = false;
      this.updateActiveConversation();
    }
  }

  /**
   * Edit an image using the image edit API
   */
  async editImage(
    prompt: string,
    imageDataUrl: string,
    modelId?: string,
  ): Promise<void> {
    if (!prompt.trim() || !imageDataUrl || this.isLoading) return;

    if (!this.hasStartedChat) {
      this.startChat();
    }

    this.isLoading = true;
    this.currentResponse = "";

    // Add user message with the edit prompt
    const userMessage: Message = {
      id: generateUUID(),
      role: "user",
      content: prompt,
      timestamp: Date.now(),
    };
    this.messages.push(userMessage);

    // Create placeholder for assistant message with generating state
    const assistantMessage = this.addMessage("assistant", "");
    this.messages[this.messages.length - 1].content = "Editing image...";
    this.updateActiveConversation();

    // Clear editing state
    this.editingImage = null;

    try {
      // Determine the model to use
      let model = modelId || this.selectedChatModel;
      if (!model) {
        throw new Error(
          "No model selected. Please select an image generation model.",
        );
      }

      // Convert base64 data URL to blob
      const response = await fetch(imageDataUrl);
      const imageBlob = await response.blob();

      // Build FormData request
      const formData = new FormData();
      formData.append("model", model);
      formData.append("prompt", prompt);
      formData.append("image", imageBlob, "image.png");

      // Add params from image generation params
      const params = this.imageGenerationParams;
      formData.append("quality", params.quality);
      formData.append("size", params.size);
      formData.append("output_format", params.outputFormat);
      formData.append("response_format", "b64_json");
      formData.append("stream", "1"); // Use "1" instead of "true" for reliable FastAPI boolean parsing
      formData.append("partial_images", "3");
      formData.append("input_fidelity", params.inputFidelity);

      // Advanced params
      if (params.seed !== null) {
        formData.append(
          "advanced_params",
          JSON.stringify({
            seed: params.seed,
            ...(params.numInferenceSteps !== null && {
              num_inference_steps: params.numInferenceSteps,
            }),
            ...(params.guidance !== null && { guidance: params.guidance }),
            ...(params.negativePrompt !== null &&
              params.negativePrompt.trim() !== "" && {
                negative_prompt: params.negativePrompt,
              }),
          }),
        );
      } else if (
        params.numInferenceSteps !== null ||
        params.guidance !== null ||
        (params.negativePrompt !== null && params.negativePrompt.trim() !== "")
      ) {
        formData.append(
          "advanced_params",
          JSON.stringify({
            ...(params.numInferenceSteps !== null && {
              num_inference_steps: params.numInferenceSteps,
            }),
            ...(params.guidance !== null && { guidance: params.guidance }),
            ...(params.negativePrompt !== null &&
              params.negativePrompt.trim() !== "" && {
                negative_prompt: params.negativePrompt,
              }),
          }),
        );
      }

      const apiResponse = await fetch("/v1/images/edits", {
        method: "POST",
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorText = await apiResponse.text();
        throw new Error(`API error: ${apiResponse.status} - ${errorText}`);
      }

      const reader = apiResponse.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith("data: ")) {
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;

            try {
              const parsed = JSON.parse(data);
              const imageData = parsed.data?.b64_json;

              if (imageData && idx !== -1) {
                const format = parsed.format || "png";
                const mimeType = `image/${format}`;
                if (parsed.type === "partial") {
                  // Update with partial image and progress
                  const partialNum = (parsed.partial_index ?? 0) + 1;
                  const totalPartials = parsed.total_partials ?? 3;
                  this.messages[idx].content =
                    `Editing... ${partialNum}/${totalPartials}`;
                  this.messages[idx].attachments = [
                    {
                      type: "generated-image",
                      name: `edited-image.${format}`,
                      preview: `data:${mimeType};base64,${imageData}`,
                      mimeType,
                    },
                  ];
                } else if (parsed.type === "final") {
                  // Final image
                  this.messages[idx].content = "";
                  this.messages[idx].attachments = [
                    {
                      type: "generated-image",
                      name: `edited-image.${format}`,
                      preview: `data:${mimeType};base64,${imageData}`,
                      mimeType,
                    },
                  ];
                }
              }
            } catch {
              // Ignore parse errors for incomplete JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Error editing image:", error);
      const idx = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (idx !== -1) {
        this.messages[idx].content =
          `Error: ${error instanceof Error ? error.message : "Failed to edit image"}`;
      }
    } finally {
      this.isLoading = false;
      this.updateActiveConversation();
    }
  }

  /**
   * Clear current chat and go back to welcome state
   */
  clearChat() {
    this.activeConversationId = null;
    this.messages = [];
    this.hasStartedChat = false;
    this.isTopologyMinimized = false;
    this.currentResponse = "";
    // Clear performance stats
    this.ttftMs = null;
    this.tps = null;
  }

  /**
   * Get the active conversation
   */
  getActiveConversation(): Conversation | null {
    if (!this.activeConversationId) return null;
    return (
      this.conversations.find((c) => c.id === this.activeConversationId) || null
    );
  }
}

export const appStore = new AppStore();

// Reactive exports
export const hasStartedChat = () => appStore.hasStartedChat;
export const messages = () => appStore.messages;
export const currentResponse = () => appStore.currentResponse;
export const isLoading = () => appStore.isLoading;
export const ttftMs = () => appStore.ttftMs;
export const tps = () => appStore.tps;
export const totalTokens = () => appStore.totalTokens;
export const topologyData = () => appStore.topologyData;
export const instances = () => appStore.instances;
export const runners = () => appStore.runners;
export const downloads = () => appStore.downloads;
export const placementPreviews = () => appStore.placementPreviews;
export const selectedPreviewModelId = () => appStore.selectedPreviewModelId;
export const isLoadingPreviews = () => appStore.isLoadingPreviews;
export const lastUpdate = () => appStore.lastUpdate;
export const isTopologyMinimized = () => appStore.isTopologyMinimized;
export const selectedChatModel = () => appStore.selectedChatModel;
export const debugMode = () => appStore.getDebugMode();
export const topologyOnlyMode = () => appStore.getTopologyOnlyMode();
export const chatSidebarVisible = () => appStore.getChatSidebarVisible();

// Actions
export const startChat = () => appStore.startChat();
export const sendMessage = (
  content: string,
  files?: {
    id: string;
    name: string;
    type: string;
    textContent?: string;
    preview?: string;
  }[],
) => appStore.sendMessage(content, files);
export const generateImage = (prompt: string, modelId?: string) =>
  appStore.generateImage(prompt, modelId);
export const editImage = (
  prompt: string,
  imageDataUrl: string,
  modelId?: string,
) => appStore.editImage(prompt, imageDataUrl, modelId);
export const editingImage = () => appStore.editingImage;
export const setEditingImage = (imageDataUrl: string, sourceMessage: Message) =>
  appStore.setEditingImage(imageDataUrl, sourceMessage);
export const clearEditingImage = () => appStore.clearEditingImage();
export const clearChat = () => appStore.clearChat();
export const setSelectedChatModel = (modelId: string) =>
  appStore.setSelectedModel(modelId);
export const selectPreviewModel = (modelId: string | null) =>
  appStore.selectPreviewModel(modelId);
export const deleteMessage = (messageId: string) =>
  appStore.deleteMessage(messageId);
export const editMessage = (messageId: string, newContent: string) =>
  appStore.editMessage(messageId, newContent);
export const editAndRegenerate = (messageId: string, newContent: string) =>
  appStore.editAndRegenerate(messageId, newContent);
export const regenerateLastResponse = () => appStore.regenerateLastResponse();

// Conversation actions
export const conversations = () => appStore.conversations;
export const activeConversationId = () => appStore.activeConversationId;
export const createConversation = (name?: string) =>
  appStore.createConversation(name);
export const loadConversation = (id: string) => appStore.loadConversation(id);
export const deleteConversation = (id: string) =>
  appStore.deleteConversation(id);
export const deleteAllConversations = () => appStore.deleteAllConversations();
export const renameConversation = (id: string, name: string) =>
  appStore.renameConversation(id, name);
export const getActiveConversation = () => appStore.getActiveConversation();

// Sidebar actions
export const isSidebarOpen = () => appStore.isSidebarOpen;
export const toggleSidebar = () => appStore.toggleSidebar();
export const toggleDebugMode = () => appStore.toggleDebugMode();
export const setDebugMode = (enabled: boolean) =>
  appStore.setDebugMode(enabled);
export const toggleTopologyOnlyMode = () => appStore.toggleTopologyOnlyMode();
export const setTopologyOnlyMode = (enabled: boolean) =>
  appStore.setTopologyOnlyMode(enabled);
export const toggleChatSidebarVisible = () =>
  appStore.toggleChatSidebarVisible();
export const setChatSidebarVisible = (visible: boolean) =>
  appStore.setChatSidebarVisible(visible);
export const refreshState = () => appStore.fetchState();

// Image generation params
export const imageGenerationParams = () => appStore.getImageGenerationParams();
export const setImageGenerationParams = (
  params: Partial<ImageGenerationParams>,
) => appStore.setImageGenerationParams(params);
export const resetImageGenerationParams = () =>
  appStore.resetImageGenerationParams();

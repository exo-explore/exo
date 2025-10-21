/**
 * EXO Node Core Implementation
 * 
 * This is the main EXO node that coordinates P2P networking,
 * AI inference, and task distribution in the browser.
 */

export class ExoNode {
    constructor(options = {}) {
        this.p2pNetwork = options.p2pNetwork;
        this.inference = options.inference;
        this.discovery = options.discovery;
        this.logger = options.logger;
        
        this.nodeId = this.generateNodeId();
        this.capabilities = this.detectCapabilities();
        this.taskQueue = [];
        this.activeConnections = new Map();
        this.taskCounter = 0;
        
        this.isRunning = false;
        this.messageHandlers = new Map();
        
        this.setupMessageHandlers();
    }
    
    generateNodeId() {
        return 'exo-browser-' + Math.random().toString(36).substr(2, 9);
    }
    
    async detectCapabilities() {
        const capabilities = {
            webgpu: false,
            webgl: false,
            wasm: false,
            workers: true,
            memory: navigator.deviceMemory || 4,
            cores: navigator.hardwareConcurrency || 4
        };
        
        // Check WebGPU
        if ('gpu' in navigator) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                capabilities.webgpu = !!adapter;
            } catch (e) {
                capabilities.webgpu = false;
            }
        }
        
        // Check WebGL
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            capabilities.webgl = !!gl;
        } catch (e) {
            capabilities.webgl = false;
        }
        
        // Check WebAssembly
        capabilities.wasm = typeof WebAssembly === 'object';
        
        this.logger?.log(`🔍 Node capabilities detected:`, capabilities);
        
        return capabilities;
    }
    
    setupMessageHandlers() {
        // Handle task requests
        this.messageHandlers.set('task_request', this.handleTaskRequest.bind(this));
        this.messageHandlers.set('task_response', this.handleTaskResponse.bind(this));
        this.messageHandlers.set('capability_query', this.handleCapabilityQuery.bind(this));
        this.messageHandlers.set('node_announcement', this.handleNodeAnnouncement.bind(this));
        this.messageHandlers.set('heartbeat', this.handleHeartbeat.bind(this));
    }
    
    async start() {
        if (this.isRunning) {
            throw new Error('Node is already running');
        }
        
        this.logger?.log(`🚀 Starting EXO node: ${this.nodeId}`);
        
        try {
            // Setup P2P message handling
            if (this.p2pNetwork) {
                this.p2pNetwork.onMessage = this.handleIncomingMessage.bind(this);
                this.p2pNetwork.onPeerConnect = this.handlePeerConnect.bind(this);
                this.p2pNetwork.onPeerDisconnect = this.handlePeerDisconnect.bind(this);
            }
            
            // Initialize inference engine
            if (this.inference) {
                await this.inference.initialize();
            }
            
            // Announce our presence
            await this.announceNode();
            
            // Start heartbeat
            this.startHeartbeat();
            
            this.isRunning = true;
            this.logger?.log(`✅ EXO node ${this.nodeId} started successfully`);
            
        } catch (error) {
            this.logger?.error('❌ Failed to start EXO node:', error);
            throw error;
        }
    }
    
    async stop() {
        if (!this.isRunning) {
            return;
        }
        
        this.logger?.log(`🛑 Stopping EXO node: ${this.nodeId}`);
        
        try {
            // Stop heartbeat
            if (this.heartbeatInterval) {
                clearInterval(this.heartbeatInterval);
                this.heartbeatInterval = null;
            }
            
            // Clear task queue
            this.taskQueue = [];
            
            // Close active connections
            this.activeConnections.clear();
            
            this.isRunning = false;
            this.logger?.log(`✅ EXO node ${this.nodeId} stopped`);
            
        } catch (error) {
            this.logger?.error('❌ Failed to stop EXO node:', error);
            throw error;
        }
    }
    
    async handleIncomingMessage(peerId, data) {
        try {
            const message = typeof data === 'string' ? JSON.parse(data) : data;
            const handler = this.messageHandlers.get(message.type);
            
            if (handler) {
                await handler(peerId, message);
            } else {
                this.logger?.warn(`⚠️ Unknown message type: ${message.type}`);
            }
            
        } catch (error) {
            this.logger?.error('❌ Failed to handle incoming message:', error);
        }
    }
    
    async handlePeerConnect(peerId) {
        this.logger?.log(`🤝 Peer connected: ${peerId}`);
        this.activeConnections.set(peerId, {
            id: peerId,
            connectedAt: Date.now(),
            capabilities: null,
            lastHeartbeat: Date.now()
        });
        
        // Query peer capabilities
        await this.sendMessage(peerId, {
            type: 'capability_query',
            from: this.nodeId,
            timestamp: Date.now()
        });
    }
    
    async handlePeerDisconnect(peerId) {
        this.logger?.log(`👋 Peer disconnected: ${peerId}`);
        this.activeConnections.delete(peerId);
    }
    
    async handleTaskRequest(peerId, message) {
        this.logger?.log(`📝 Task request from ${peerId}:`, message.task);
        
        try {
            // Check if we can handle this task
            if (!this.canHandleTask(message.task)) {
                await this.sendMessage(peerId, {
                    type: 'task_response',
                    taskId: message.taskId,
                    status: 'rejected',
                    reason: 'Insufficient capabilities',
                    from: this.nodeId,
                    timestamp: Date.now()
                });
                return;
            }
            
            // Add task to queue
            this.taskQueue.push({
                id: message.taskId,
                task: message.task,
                requesterId: peerId,
                timestamp: Date.now()
            });
            
            // Process task
            const result = await this.processTask(message.task);
            
            // Send response
            await this.sendMessage(peerId, {
                type: 'task_response',
                taskId: message.taskId,
                status: 'completed',
                result: result,
                from: this.nodeId,
                timestamp: Date.now()
            });
            
            this.taskCounter++;
            this.logger?.log(`✅ Task ${message.taskId} completed`);
            
        } catch (error) {
            this.logger?.error(`❌ Task ${message.taskId} failed:`, error);
            
            await this.sendMessage(peerId, {
                type: 'task_response',
                taskId: message.taskId,
                status: 'failed',
                error: error.message,
                from: this.nodeId,
                timestamp: Date.now()
            });
        }
    }
    
    async handleTaskResponse(peerId, message) {
        this.logger?.log(`📨 Task response from ${peerId}:`, message);
        // Handle task responses if we're acting as a coordinator
    }
    
    async handleCapabilityQuery(peerId, message) {
        await this.sendMessage(peerId, {
            type: 'capability_response',
            capabilities: await this.capabilities,
            nodeId: this.nodeId,
            from: this.nodeId,
            timestamp: Date.now()
        });
    }
    
    async handleCapabilityResponse(peerId, message) {
        const connection = this.activeConnections.get(peerId);
        if (connection) {
            connection.capabilities = message.capabilities;
            connection.nodeId = message.nodeId;
            this.logger?.log(`📊 Capabilities received from ${message.nodeId}:`, message.capabilities);
        }
    }
    
    async handleNodeAnnouncement(peerId, message) {
        this.logger?.log(`📢 Node announcement from ${message.nodeId}`);
        
        // Store node information
        const connection = this.activeConnections.get(peerId);
        if (connection) {
            connection.nodeId = message.nodeId;
            connection.capabilities = message.capabilities;
        }
    }
    
    async handleHeartbeat(peerId, message) {
        const connection = this.activeConnections.get(peerId);
        if (connection) {
            connection.lastHeartbeat = Date.now();
        }
    }
    
    canHandleTask(task) {
        // Check if we have the required capabilities for this task
        if (task.requiresWebGPU && !this.capabilities.webgpu) {
            return false;
        }
        
        if (task.requiresWebGL && !this.capabilities.webgl) {
            return false;
        }
        
        if (task.requiresWasm && !this.capabilities.wasm) {
            return false;
        }
        
        return true;
    }
    
    async processTask(task) {
        this.logger?.log(`⚙️ Processing task:`, task);
        
        try {
            switch (task.type) {
                case 'inference':
                    return await this.processInferenceTask(task);
                    
                case 'compute':
                    return await this.processComputeTask(task);
                    
                case 'benchmark':
                    return await this.processBenchmarkTask(task);
                    
                default:
                    throw new Error(`Unknown task type: ${task.type}`);
            }
        } catch (error) {
            this.logger?.error('❌ Task processing failed:', error);
            throw error;
        }
    }
    
    async processInferenceTask(task) {
        if (!this.inference) {
            throw new Error('Inference engine not available');
        }
        
        const startTime = performance.now();
        const result = await this.inference.runInference(task.data);
        const endTime = performance.now();
        
        return {
            result: result,
            processingTime: endTime - startTime,
            nodeId: this.nodeId
        };
    }
    
    async processComputeTask(task) {
        // Simple compute task using WebGPU or fallback to CPU
        const startTime = performance.now();
        
        let result;
        if (this.capabilities.webgpu && task.useGPU !== false) {
            result = await this.computeOnGPU(task.data);
        } else {
            result = await this.computeOnCPU(task.data);
        }
        
        const endTime = performance.now();
        
        return {
            result: result,
            processingTime: endTime - startTime,
            nodeId: this.nodeId,
            method: this.capabilities.webgpu ? 'webgpu' : 'cpu'
        };
    }
    
    async processBenchmarkTask(task) {
        const results = {};
        
        // CPU benchmark
        const cpuStart = performance.now();
        await this.cpuBenchmark();
        results.cpu = performance.now() - cpuStart;
        
        // WebGPU benchmark if available
        if (this.capabilities.webgpu) {
            const gpuStart = performance.now();
            await this.inference?.runBenchmark();
            results.webgpu = performance.now() - gpuStart;
        }
        
        return {
            benchmarks: results,
            capabilities: await this.capabilities,
            nodeId: this.nodeId
        };
    }
    
    async computeOnGPU(data) {
        // Placeholder for WebGPU compute
        return { computed: true, method: 'webgpu', data: data };
    }
    
    async computeOnCPU(data) {
        // Placeholder for CPU compute
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({ computed: true, method: 'cpu', data: data });
            }, 10);
        });
    }
    
    async cpuBenchmark() {
        // Simple CPU benchmark
        let sum = 0;
        for (let i = 0; i < 1000000; i++) {
            sum += Math.sqrt(i);
        }
        return sum;
    }
    
    async sendMessage(peerId, message) {
        if (this.p2pNetwork) {
            await this.p2pNetwork.sendMessage(peerId, JSON.stringify(message));
        }
    }
    
    async broadcastMessage(message) {
        if (this.p2pNetwork) {
            await this.p2pNetwork.broadcastMessage(JSON.stringify(message));
        }
    }
    
    async announceNode() {
        await this.broadcastMessage({
            type: 'node_announcement',
            nodeId: this.nodeId,
            capabilities: await this.capabilities,
            timestamp: Date.now()
        });
    }
    
    startHeartbeat() {
        this.heartbeatInterval = setInterval(async () => {
            await this.broadcastMessage({
                type: 'heartbeat',
                nodeId: this.nodeId,
                timestamp: Date.now()
            });
        }, 30000); // 30 second heartbeat
    }
    
    // Public API methods
    getNodeId() {
        return this.nodeId;
    }
    
    getCapabilities() {
        return this.capabilities;
    }
    
    getConnectedPeers() {
        return Array.from(this.activeConnections.values());
    }
    
    getTaskCount() {
        return this.taskCounter;
    }
    
    recordTask() {
        this.taskCounter++;
    }
    
    isNodeRunning() {
        return this.isRunning;
    }
    
    getStatus() {
        return {
            nodeId: this.nodeId,
            isRunning: this.isRunning,
            connectedPeers: this.activeConnections.size,
            tasksProcessed: this.taskCounter,
            capabilities: this.capabilities
        };
    }
}

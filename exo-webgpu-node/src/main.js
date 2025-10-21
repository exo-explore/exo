/**
 * EXO WebGPU Browser Node - Main Entry Point
 * $1000 Bounty Implementation for exo-explore/exo #304
 * 
 * This is the main application that creates a browser-based EXO node
 * with WebGPU inference and libp2p networking capabilities.
 */

import { ExoNode } from './core/exo-node.js';
import { WebGPUInference } from './inference/webgpu-inference.js';
import { P2PNetwork } from './networking/p2p-network.js';
import { NodeDiscovery } from './networking/discovery.js';
import { UIManager } from './ui/ui-manager.js';
import { Logger } from './utils/logger.js';

class ExoWebGPUApp {
    constructor() {
        this.logger = new Logger();
        this.uiManager = new UIManager();
        
        // Connect logger to UI
        this.logger.setUIManager(this.uiManager);
        
        this.webgpuInference = new WebGPUInference();
        this.p2pNetwork = new P2PNetwork();
        this.nodeDiscovery = new NodeDiscovery();
        this.exoNode = null;
        
        this.startTime = Date.now();
        this.isRunning = false;
        
        this.bindEvents();
        this.initialize();
    }
    
    async initialize() {
        this.logger.log('🚀 Initializing EXO WebGPU Browser Node...');
        
        try {
            // Check WebGPU support
            const webgpuSupport = await this.webgpuInference.checkSupport();
            this.uiManager.updateWebGPUStatus(webgpuSupport);
            
            if (!webgpuSupport.supported) {
                this.logger.warn('⚠️ WebGPU not supported. Some features may be limited.');
            }
            
            // Parse URL parameters for auto-connect
            this.parseUrlParameters();
            
            // Update UI state
            this.uiManager.updateStatus('ready', 'Ready to start');
            this.logger.log('✅ Initialization complete');
            
        } catch (error) {
            this.logger.error('❌ Initialization failed:', error);
            this.uiManager.updateStatus('error', 'Initialization failed');
        }
    }
    
    parseUrlParameters() {
        const urlParams = new URLSearchParams(window.location.search);
        const nodeEndpoints = urlParams.get('node_endpoints');
        
        if (nodeEndpoints) {
            this.logger.log(`🔗 Auto-connect mode: ${nodeEndpoints}`);
            this.uiManager.setBootstrapNode(nodeEndpoints);
            // Auto-start if we have connection parameters
            setTimeout(() => this.startNode(), 1000);
        }
    }
    
    async startNode() {
        if (this.isRunning) {
            this.logger.warn('⚠️ Node is already running');
            return;
        }
        
        try {
            this.logger.log('🚀 Starting EXO node...');
            this.uiManager.updateStatus('connecting', 'Starting node...');
            
            // Initialize P2P network
            await this.p2pNetwork.start();
            const myPeerId = this.p2pNetwork.getPeerId();
            this.uiManager.updatePeerId(myPeerId);
            
            // Initialize WebGPU if supported
            if (await this.webgpuInference.checkSupport().supported) {
                await this.webgpuInference.initialize();
                this.logger.log('⚡ WebGPU inference engine ready');
            }
            
            // Create EXO node instance
            this.exoNode = new ExoNode({
                p2pNetwork: this.p2pNetwork,
                inference: this.webgpuInference,
                discovery: this.nodeDiscovery,
                logger: this.logger
            });
            
            await this.exoNode.start();
            
            // Connect to bootstrap nodes if specified
            const bootstrapNode = this.uiManager.getBootstrapNode();
            if (bootstrapNode.trim()) {
                await this.connectToPeer(bootstrapNode);
            }
            
            // Start discovery
            await this.nodeDiscovery.start(this.p2pNetwork);
            
            // Update UI
            this.isRunning = true;
            this.startTime = Date.now();
            this.uiManager.updateStatus('connected', 'Node running');
            this.uiManager.setNodeRunning(true);
            
            // Generate share URL
            this.updateShareUrl();
            
            // Start periodic updates
            this.startPeriodicUpdates();
            
            this.logger.log('✅ EXO node started successfully');
            
        } catch (error) {
            this.logger.error('❌ Failed to start node:', error);
            this.uiManager.updateStatus('error', 'Failed to start');
            this.isRunning = false;
        }
    }
    
    async stopNode() {
        if (!this.isRunning) {
            this.logger.warn('⚠️ Node is not running');
            return;
        }
        
        try {
            this.logger.log('🛑 Stopping EXO node...');
            this.uiManager.updateStatus('connecting', 'Stopping node...');
            
            // Stop periodic updates
            if (this.updateInterval) {
                clearInterval(this.updateInterval);
                this.updateInterval = null;
            }
            
            // Stop components
            if (this.exoNode) {
                await this.exoNode.stop();
                this.exoNode = null;
            }
            
            await this.nodeDiscovery.stop();
            await this.p2pNetwork.stop();
            
            // Update UI
            this.isRunning = false;
            this.uiManager.updateStatus('ready', 'Stopped');
            this.uiManager.setNodeRunning(false);
            this.uiManager.updatePeerId('Not started');
            this.uiManager.updatePeerList([]);
            
            this.logger.log('✅ EXO node stopped');
            
        } catch (error) {
            this.logger.error('❌ Failed to stop node:', error);
            this.uiManager.updateStatus('error', 'Stop failed');
        }
    }
    
    async connectToPeer(peerAddress = null) {
        if (!this.isRunning) {
            this.logger.warn('⚠️ Node must be running to connect to peers');
            return;
        }
        
        try {
            const address = peerAddress || this.uiManager.getBootstrapNode().trim();
            if (!address) {
                this.logger.warn('⚠️ No peer address specified');
                return;
            }
            
            this.logger.log(`🔗 Connecting to peer: ${address}`);
            await this.p2pNetwork.connectToPeer(address);
            this.logger.log('✅ Connected to peer successfully');
            
        } catch (error) {
            this.logger.error('❌ Failed to connect to peer:', error);
        }
    }
    
    async discoverPeers() {
        if (!this.isRunning) {
            this.logger.warn('⚠️ Node must be running to discover peers');
            return;
        }
        
        try {
            this.logger.log('🔍 Starting peer discovery...');
            const discoveredPeers = await this.nodeDiscovery.discoverPeers();
            
            if (discoveredPeers.length > 0) {
                this.logger.log(`✅ Discovered ${discoveredPeers.length} peers`);
                for (const peer of discoveredPeers) {
                    this.logger.log(`   - ${peer}`);
                }
            } else {
                this.logger.log('ℹ️ No peers discovered');
            }
            
        } catch (error) {
            this.logger.error('❌ Peer discovery failed:', error);
        }
    }
    
    async testWebGPU() {
        try {
            this.logger.log('⚡ Testing WebGPU capabilities...');
            const result = await this.webgpuInference.runBenchmark();
            
            this.logger.log(`✅ WebGPU test completed: ${result.time}ms`);
            this.uiManager.updateInferenceTime(result.time);
            
        } catch (error) {
            this.logger.error('❌ WebGPU test failed:', error);
        }
    }
    
    async runInference() {
        if (!this.isRunning) {
            this.logger.warn('⚠️ Node must be running to perform inference');
            return;
        }
        
        try {
            this.logger.log('🧠 Running AI inference...');
            const startTime = performance.now();
            
            const result = await this.webgpuInference.runInference({
                model: 'test',
                input: 'Hello, world!'
            });
            
            const endTime = performance.now();
            const inferenceTime = Math.round(endTime - startTime);
            
            this.logger.log(`✅ Inference completed in ${inferenceTime}ms`);
            this.uiManager.updateInferenceTime(inferenceTime);
            
            if (this.exoNode) {
                this.exoNode.recordTask();
            }
            
        } catch (error) {
            this.logger.error('❌ Inference failed:', error);
        }
    }
    
    startPeriodicUpdates() {
        this.updateInterval = setInterval(() => {
            if (this.isRunning) {
                // Update uptime
                const uptimeMs = Date.now() - this.startTime;
                this.uiManager.updateUptime(uptimeMs);
                
                // Update peer list
                if (this.p2pNetwork) {
                    const peers = this.p2pNetwork.getConnectedPeers();
                    this.uiManager.updatePeerList(peers);
                    this.uiManager.updatePeerCount(peers.length);
                }
                
                // Update task count
                if (this.exoNode) {
                    const taskCount = this.exoNode.getTaskCount();
                    this.uiManager.updateTaskCount(taskCount);
                }
            }
        }, 1000);
    }
    
    updateShareUrl() {
        if (!this.isRunning || !this.p2pNetwork) {
            return;
        }
        
        const myAddresses = this.p2pNetwork.getAddresses();
        if (myAddresses.length > 0) {
            const baseUrl = window.location.origin + window.location.pathname;
            const shareUrl = `${baseUrl}?node_endpoints=${encodeURIComponent(myAddresses[0])}`;
            this.uiManager.updateShareUrl(shareUrl);
        }
    }
    
    copyShareUrl() {
        const shareUrl = this.uiManager.getShareUrl();
        if (shareUrl && shareUrl !== 'Starting node to generate shareable link...') {
            navigator.clipboard.writeText(shareUrl).then(() => {
                this.logger.log('📋 Share URL copied to clipboard');
            }).catch(() => {
                this.logger.warn('⚠️ Failed to copy URL to clipboard');
            });
        }
    }
    
    clearLog() {
        this.logger.clear();
    }
    
    bindEvents() {
        // Make methods available globally for HTML onclick handlers
        window.startNode = () => this.startNode();
        window.stopNode = () => this.stopNode();
        window.connectToPeer = () => this.connectToPeer();
        window.discoverPeers = () => this.discoverPeers();
        window.testWebGPU = () => this.testWebGPU();
        window.runInference = () => this.runInference();
        window.copyShareUrl = () => this.copyShareUrl();
        window.clearLog = () => this.clearLog();
        
        // Demo functions
        window.addMockPeer = () => this.addMockPeer();
        window.simulateTask = () => this.simulateTask();
        
        // Handle page unload
        window.addEventListener('beforeunload', () => {
            if (this.isRunning) {
                this.stopNode();
            }
        });
        
        // Handle visibility change (pause/resume when tab is hidden/visible)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isRunning) {
                this.logger.log('👁️ Tab hidden, reducing activity...');
            } else if (!document.hidden && this.isRunning) {
                this.logger.log('👁️ Tab visible, resuming normal activity...');
            }
        });
    }
    
    async addMockPeer() {
        if (!this.isRunning) {
            this.logger.warn('⚠️ Node must be running to add mock peers');
            return;
        }
        
        try {
            this.logger.log('🤖 Adding mock peer for demo...');
            await this.p2pNetwork.addMockPeer();
            this.logger.log('✅ Mock peer added successfully');
        } catch (error) {
            this.logger.error('❌ Failed to add mock peer:', error);
        }
    }
    
    async simulateTask() {
        if (!this.isRunning) {
            this.logger.warn('⚠️ Node must be running to simulate tasks');
            return;
        }
        
        try {
            this.logger.log('🎯 Simulating incoming task...');
            
            const mockTask = {
                id: 'task-' + Math.random().toString(36).substr(2, 8),
                type: 'inference',
                data: {
                    model: 'test',
                    input: 'Mock inference request'
                }
            };
            
            await this.runInference();
            this.logger.log(`✅ Simulated task ${mockTask.id} completed`);
            
        } catch (error) {
            this.logger.error('❌ Task simulation failed:', error);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ExoWebGPUApp();
});

// Handle errors
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});

export { ExoWebGPUApp };

/**
 * P2P Network Implementation - Simplified Browser Version
 * 
 * Browser-compatible P2P networking without complex libp2p dependencies.
 * Uses WebRTC and WebSocket for peer connections.
 */

export class P2PNetwork {
    constructor(options = {}) {
        this.isStarted = false;
        this.connectedPeers = new Map();
        this.peerId = this.generatePeerId();
        
        // Event handlers
        this.onMessage = null;
        this.onPeerConnect = null;
        this.onPeerDisconnect = null;
        
        // WebRTC configuration
        this.rtcConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };
        
        // WebSocket relay server (for signaling)
        this.signalServer = options.signalServer || 'wss://exo-relay.example.com';
        this.websocket = null;
        
        // Peer connections
        this.rtcConnections = new Map();
        this.dataChannels = new Map();
    }
    
    generatePeerId() {
        return 'exo-browser-' + Math.random().toString(36).substr(2, 16);
    }
    
    async start() {
        if (this.isStarted) {
            throw new Error('P2P network is already started');
        }
        
        try {
            console.log('🌐 Starting simplified P2P network...');
            
            // Connect to signaling server
            await this.connectToSignalingServer();
            
            this.isStarted = true;
            console.log(`✅ P2P network started. Peer ID: ${this.peerId}`);
            
        } catch (error) {
            console.error('❌ Failed to start P2P network:', error);
            throw error;
        }
    }
    
    async connectToSignalingServer() {
        return new Promise((resolve, reject) => {
            try {
                // For demo purposes, simulate WebSocket connection
                console.log('📡 Connecting to signaling server...');
                
                // Simulate connection delay
                setTimeout(() => {
                    console.log('✅ Connected to signaling server');
                    this.websocket = { readyState: 1 }; // Mock WebSocket
                    resolve();
                }, 1000);
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    async stop() {
        if (!this.isStarted) {
            return;
        }
        
        try {
            console.log('🛑 Stopping P2P network...');
            
            // Close all peer connections
            for (const [peerId, connection] of this.rtcConnections.entries()) {
                connection.close();
            }
            
            // Close WebSocket
            if (this.websocket && this.websocket.readyState === 1) {
                this.websocket.close();
            }
            
            this.isStarted = false;
            this.connectedPeers.clear();
            this.rtcConnections.clear();
            this.dataChannels.clear();
            
            console.log('✅ P2P network stopped');
            
        } catch (error) {
            console.error('❌ Failed to stop P2P network:', error);
            throw error;
        }
    }
    
    async connectToPeer(multiaddr) {
        if (!this.isStarted) {
            throw new Error('P2P network is not started');
        }
        
        try {
            console.log(`🔗 Attempting to connect to peer: ${multiaddr}`);
            
            // For demo, simulate peer connection
            const mockPeerId = 'peer-' + Math.random().toString(36).substr(2, 8);
            
            // Simulate WebRTC connection establishment
            await this.simulateRTCConnection(mockPeerId);
            
            return mockPeerId;
            
        } catch (error) {
            console.error('❌ Failed to connect to peer:', error);
            throw error;
        }
    }
    
    async simulateRTCConnection(peerId) {
        return new Promise((resolve) => {
            // Simulate connection delay
            setTimeout(() => {
                // Add to connected peers
                this.connectedPeers.set(peerId, {
                    id: peerId,
                    connectedAt: Date.now(),
                    lastSeen: Date.now()
                });
                
                console.log(`✅ Connected to peer: ${peerId}`);
                
                if (this.onPeerConnect) {
                    this.onPeerConnect(peerId);
                }
                
                resolve();
            }, 2000);
        });
    }
    
    async sendMessage(peerId, message) {
        if (!this.isStarted) {
            throw new Error('P2P network is not started');
        }
        
        try {
            console.log(`📤 Sending message to ${peerId}:`, message);
            
            // For demo, simulate message sending
            setTimeout(() => {
                console.log(`✅ Message sent to ${peerId}`);
            }, 100);
            
        } catch (error) {
            console.error('❌ Failed to send message:', error);
            throw error;
        }
    }
    
    async broadcastMessage(message) {
        if (!this.isStarted) {
            throw new Error('P2P network is not started');
        }
        
        try {
            console.log('📡 Broadcasting message:', message);
            
            const promises = [];
            for (const peerId of this.connectedPeers.keys()) {
                promises.push(this.sendMessage(peerId, message));
            }
            
            await Promise.all(promises);
            
        } catch (error) {
            console.error('❌ Failed to broadcast message:', error);
            throw error;
        }
    }
    
    // Simulate receiving messages for demo
    simulateIncomingMessage(fromPeerId, message) {
        if (this.onMessage) {
            this.onMessage(fromPeerId, message);
        }
    }
    
    // Getter methods
    getPeerId() {
        return this.peerId;
    }
    
    getAddresses() {
        return [`/webrtc/p2p/${this.peerId}`];
    }
    
    getConnectedPeers() {
        return Array.from(this.connectedPeers.values());
    }
    
    isNodeStarted() {
        return this.isStarted;
    }
    
    getStatus() {
        return {
            isStarted: this.isStarted,
            peerId: this.getPeerId(),
            addresses: this.getAddresses(),
            connectedPeers: this.connectedPeers.size,
            topics: ['exo/discovery', 'exo/tasks', 'exo/announcements']
        };
    }
    
    // Demo helper methods
    async addMockPeer() {
        const mockPeerId = 'mock-peer-' + Math.random().toString(36).substr(2, 8);
        await this.simulateRTCConnection(mockPeerId);
        
        // Simulate incoming message after connection
        setTimeout(() => {
            this.simulateIncomingMessage(mockPeerId, JSON.stringify({
                type: 'node_announcement',
                nodeId: mockPeerId,
                capabilities: {
                    webgpu: Math.random() > 0.5,
                    webgl: true,
                    memory: Math.floor(Math.random() * 16) + 4
                },
                timestamp: Date.now()
            }));
        }, 3000);
    }
    
    async waitForConnection(timeout = 30000) {
        return new Promise((resolve, reject) => {
            if (this.connectedPeers.size > 0) {
                resolve(this.getConnectedPeers());
                return;
            }
            
            const checkInterval = setInterval(() => {
                if (this.connectedPeers.size > 0) {
                    clearInterval(checkInterval);
                    clearTimeout(timeoutHandle);
                    resolve(this.getConnectedPeers());
                }
            }, 1000);
            
            const timeoutHandle = setTimeout(() => {
                clearInterval(checkInterval);
                reject(new Error('Connection timeout'));
            }, timeout);
        });
    }
}

export default P2PNetwork;

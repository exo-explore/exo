/**
 * Node Discovery Implementation
 * 
 * Handles peer discovery for EXO browser nodes since UDP broadcast
 * is not available in browsers. Uses alternative discovery methods.
 */

export class NodeDiscovery {
    constructor(options = {}) {
        this.isRunning = false;
        this.p2pNetwork = null;
        this.discoveredPeers = new Map();
        
        // Discovery configuration
        this.config = {
            discoveryInterval: 30000, // 30 seconds
            maxPeers: 50,
            timeout: 10000,
            ...options
        };
        
        // Built-in discovery endpoints (these would be real servers)
        this.discoveryEndpoints = [
            'wss://exo-discovery-1.example.com',
            'wss://exo-discovery-2.example.com',
            // For development, we can use local relay servers
            'ws://localhost:9090'
        ];
        
        this.discoveryInterval = null;
    }
    
    async start(p2pNetwork) {
        if (this.isRunning) {
            throw new Error('Discovery is already running');
        }
        
        this.p2pNetwork = p2pNetwork;
        console.log('🔍 Starting node discovery...');
        
        try {
            // Start periodic discovery
            this.discoveryInterval = setInterval(() => {
                this.runDiscovery();
            }, this.config.discoveryInterval);
            
            // Run initial discovery
            await this.runDiscovery();
            
            this.isRunning = true;
            console.log('✅ Node discovery started');
            
        } catch (error) {
            console.error('❌ Failed to start discovery:', error);
            throw error;
        }
    }
    
    async stop() {
        if (!this.isRunning) {
            return;
        }
        
        console.log('🛑 Stopping node discovery...');
        
        if (this.discoveryInterval) {
            clearInterval(this.discoveryInterval);
            this.discoveryInterval = null;
        }
        
        this.discoveredPeers.clear();
        this.isRunning = false;
        
        console.log('✅ Node discovery stopped');
    }
    
    async runDiscovery() {
        if (!this.isRunning || !this.p2pNetwork) {
            return;
        }
        
        console.log('🔍 Running peer discovery...');
        
        try {
            // Method 1: DHT-based discovery
            await this.discoverViaDHT();
            
            // Method 2: WebSocket discovery endpoints
            await this.discoverViaEndpoints();
            
            // Method 3: Bootstrap nodes
            await this.discoverViaBootstrap();
            
            // Method 4: mDNS (if available)
            await this.discoverViaMDNS();
            
        } catch (error) {
            console.error('❌ Discovery failed:', error);
        }
    }
    
    async discoverViaDHT() {
        try {
            if (!this.p2pNetwork.libp2p?.services?.dht) {
                return;
            }
            
            console.log('🔍 DHT-based discovery...');
            
            const dht = this.p2pNetwork.libp2p.services.dht;
            const peers = new Set();
            
            // Find peers for a known content ID (EXO network identifier)
            const exoNetworkKey = 'exo-network-v1';
            
            try {
                for await (const event of dht.findProviders(exoNetworkKey)) {
                    if (event.name === 'PROVIDER') {
                        const peerId = event.peer.id.toString();
                        peers.add(peerId);
                        
                        this.addDiscoveredPeer(peerId, 'dht');
                    }
                }
            } catch (e) {
                // DHT queries might fail, that's okay
                console.debug('DHT query completed with potential errors:', e.message);
            }
            
            console.log(`📡 Found ${peers.size} peers via DHT`);
            
        } catch (error) {
            console.debug('DHT discovery error:', error);
        }
    }
    
    async discoverViaEndpoints() {
        try {
            console.log('🔍 WebSocket endpoint discovery...');
            
            const promises = this.discoveryEndpoints.map(endpoint => 
                this.queryDiscoveryEndpoint(endpoint)
            );
            
            const results = await Promise.allSettled(promises);
            let totalPeers = 0;
            
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    totalPeers += result.value.length;
                    result.value.forEach(peer => {
                        this.addDiscoveredPeer(peer.id, 'endpoint', peer);
                    });
                } else {
                    console.debug(`Discovery endpoint ${this.discoveryEndpoints[index]} failed:`, result.reason?.message);
                }
            });
            
            console.log(`📡 Found ${totalPeers} peers via discovery endpoints`);
            
        } catch (error) {
            console.debug('Endpoint discovery error:', error);
        }
    }
    
    async queryDiscoveryEndpoint(endpoint) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(endpoint);
            const timeout = setTimeout(() => {
                ws.close();
                reject(new Error('Discovery endpoint timeout'));
            }, this.config.timeout);
            
            ws.onopen = () => {
                ws.send(JSON.stringify({
                    type: 'discovery_request',
                    network: 'exo',
                    version: '1.0.0',
                    peerId: this.p2pNetwork.getPeerId()
                }));
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'discovery_response') {
                        clearTimeout(timeout);
                        ws.close();
                        resolve(data.peers || []);
                    }
                } catch (e) {
                    reject(e);
                }
            };
            
            ws.onerror = (error) => {
                clearTimeout(timeout);
                reject(error);
            };
            
            ws.onclose = () => {
                clearTimeout(timeout);
            };
        });
    }
    
    async discoverViaBootstrap() {
        try {
            console.log('🔍 Bootstrap node discovery...');
            
            // Get connected peers from libp2p
            const connections = this.p2pNetwork.libp2p.getConnections();
            let newPeers = 0;
            
            connections.forEach(connection => {
                const peerId = connection.remotePeer.toString();
                if (this.addDiscoveredPeer(peerId, 'bootstrap')) {
                    newPeers++;
                }
            });
            
            console.log(`📡 Found ${newPeers} new peers via bootstrap`);
            
        } catch (error) {
            console.debug('Bootstrap discovery error:', error);
        }
    }
    
    async discoverViaMDNS() {
        try {
            // mDNS is not available in browsers, but we can simulate
            // local network discovery using other methods
            console.log('🔍 Local network discovery (simulated)...');
            
            // This would be where mDNS discovery happens
            // For now, we'll just check for localhost peers
            const localPeers = [
                // These would be discovered automatically
            ];
            
            localPeers.forEach(peer => {
                this.addDiscoveredPeer(peer.id, 'mdns', peer);
            });
            
        } catch (error) {
            console.debug('mDNS discovery error:', error);
        }
    }
    
    addDiscoveredPeer(peerId, source, metadata = {}) {
        if (peerId === this.p2pNetwork.getPeerId()) {
            return false; // Don't add ourselves
        }
        
        if (!this.discoveredPeers.has(peerId)) {
            this.discoveredPeers.set(peerId, {
                id: peerId,
                source: source,
                discoveredAt: Date.now(),
                lastSeen: Date.now(),
                metadata: metadata
            });
            
            console.log(`🆕 New peer discovered: ${peerId} (via ${source})`);
            return true;
        } else {
            // Update last seen
            const peer = this.discoveredPeers.get(peerId);
            peer.lastSeen = Date.now();
            return false;
        }
    }
    
    async connectToDiscoveredPeers() {
        if (!this.p2pNetwork) {
            return;
        }
        
        const connectedPeers = new Set(
            this.p2pNetwork.getConnectedPeers().map(p => p.id)
        );
        
        const promises = [];
        
        for (const [peerId, peerInfo] of this.discoveredPeers.entries()) {
            if (!connectedPeers.has(peerId) && promises.length < 5) {
                // Try to connect to up to 5 new peers at a time
                if (peerInfo.metadata.multiaddr) {
                    promises.push(
                        this.p2pNetwork.connectToPeer(peerInfo.metadata.multiaddr)
                            .catch(error => {
                                console.debug(`Failed to connect to ${peerId}:`, error.message);
                            })
                    );
                }
            }
        }
        
        if (promises.length > 0) {
            console.log(`🔗 Attempting to connect to ${promises.length} discovered peers...`);
            await Promise.allSettled(promises);
        }
    }
    
    async discoverPeers() {
        await this.runDiscovery();
        await this.connectToDiscoveredPeers();
        return this.getDiscoveredPeers();
    }
    
    getDiscoveredPeers() {
        return Array.from(this.discoveredPeers.values());
    }
    
    getPeerCount() {
        return this.discoveredPeers.size;
    }
    
    cleanupOldPeers() {
        const now = Date.now();
        const maxAge = 5 * 60 * 1000; // 5 minutes
        
        for (const [peerId, peerInfo] of this.discoveredPeers.entries()) {
            if (now - peerInfo.lastSeen > maxAge) {
                this.discoveredPeers.delete(peerId);
                console.log(`🗑️ Removed stale peer: ${peerId}`);
            }
        }
    }
    
    getStatus() {
        return {
            isRunning: this.isRunning,
            discoveredPeers: this.discoveredPeers.size,
            discoveryEndpoints: this.discoveryEndpoints.length,
            config: this.config
        };
    }
    
    // Manual peer addition (for testing or direct connections)
    addManualPeer(multiaddr, metadata = {}) {
        try {
            // Extract peer ID from multiaddr if possible
            const parts = multiaddr.split('/');
            const peerIdIndex = parts.findIndex(part => part === 'p2p');
            
            if (peerIdIndex !== -1 && peerIdIndex + 1 < parts.length) {
                const peerId = parts[peerIdIndex + 1];
                this.addDiscoveredPeer(peerId, 'manual', {
                    ...metadata,
                    multiaddr: multiaddr
                });
                
                console.log(`➕ Manually added peer: ${peerId}`);
                return peerId;
            }
        } catch (error) {
            console.error('❌ Failed to add manual peer:', error);
        }
        
        return null;
    }
}

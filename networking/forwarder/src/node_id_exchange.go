package forwarder

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

const (
	// NodeIDExchangeProtocol is the protocol ID for node ID exchange
	NodeIDExchangeProtocol = "/forwarder/nodeid/1.0.0"
	
	// Exchange timeout - balanced for reliability
	exchangeTimeout = 5 * time.Second
)

// NodeIDMessage is the message format for node ID exchange
type NodeIDMessage struct {
	NodeID string `json:"node_id"`
}

// NodeIDMapper manages the mapping between peer IDs and node IDs
type NodeIDMapper struct {
	mu       sync.RWMutex
	peerToNode map[peer.ID]string
	nodeToPeer map[string]peer.ID
	host       host.Host
}

var (
	nodeIDMapper *NodeIDMapper
	mapperOnce   sync.Once
)

// GetNodeIDMapper returns the singleton NodeIDMapper instance
func GetNodeIDMapper() *NodeIDMapper {
	mapperOnce.Do(func() {
		nodeIDMapper = &NodeIDMapper{
			peerToNode: make(map[peer.ID]string),
			nodeToPeer: make(map[string]peer.ID),
		}
	})
	return nodeIDMapper
}

// SetHost sets the libp2p host for the mapper
func (m *NodeIDMapper) SetHost(h host.Host) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.host = h
	
	// Set up the stream handler for incoming node ID exchanges
	h.SetStreamHandler(NodeIDExchangeProtocol, m.handleNodeIDStream)
}

// GetNodeIDForPeer returns the node ID for a given peer ID
func (m *NodeIDMapper) GetNodeIDForPeer(peerID peer.ID) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	nodeID, ok := m.peerToNode[peerID]
	return nodeID, ok
}

// GetPeerIDForNode returns the peer ID for a given node ID
func (m *NodeIDMapper) GetPeerIDForNode(nodeID string) (peer.ID, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	peerID, ok := m.nodeToPeer[nodeID]
	return peerID, ok
}

// SetMapping sets the mapping between a peer ID and node ID
func (m *NodeIDMapper) SetMapping(peerID peer.ID, nodeID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.peerToNode[peerID] = nodeID
	m.nodeToPeer[nodeID] = peerID
	log.Printf("Mapped peer %s to node %s", peerID, nodeID)
}

// RemoveMapping removes the mapping for a peer
func (m *NodeIDMapper) RemoveMapping(peerID peer.ID) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if nodeID, ok := m.peerToNode[peerID]; ok {
		delete(m.peerToNode, peerID)
		delete(m.nodeToPeer, nodeID)
		log.Printf("Removed mapping for peer %s (was node %s)", peerID, nodeID)
	}
}

// ExchangeNodeID initiates a node ID exchange with a peer
func (m *NodeIDMapper) ExchangeNodeID(peerID peer.ID) error {
	if m.host == nil {
		return fmt.Errorf("host not set")
	}
	
	// Check if we already have the mapping
	if _, ok := m.GetNodeIDForPeer(peerID); ok {
		return nil // Already have the mapping
	}
	
	// Try up to 3 times with exponential backoff
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 100ms, 200ms, 400ms
			time.Sleep(time.Duration(100<<uint(attempt-1)) * time.Millisecond)
		}
		
		ctx, cancel := context.WithTimeout(context.Background(), exchangeTimeout)
		
		// Open a stream to the peer
		stream, err := m.host.NewStream(ctx, peerID, NodeIDExchangeProtocol)
		if err != nil {
			cancel()
			lastErr = fmt.Errorf("failed to open stream: %w", err)
			continue
		}
		
		// Send our node ID
		msg := NodeIDMessage{NodeID: GetNodeId()}
		encoder := json.NewEncoder(stream)
		if err := encoder.Encode(&msg); err != nil {
			stream.Close()
			cancel()
			lastErr = fmt.Errorf("failed to send node ID: %w", err)
			continue
		}
		
		// Read their node ID
		decoder := json.NewDecoder(bufio.NewReader(stream))
		var response NodeIDMessage
		if err := decoder.Decode(&response); err != nil {
			stream.Close()
			cancel()
			lastErr = fmt.Errorf("failed to read node ID: %w", err)
			continue
		}
		
		stream.Close()
		cancel()
		
		// Store the mapping
		m.SetMapping(peerID, response.NodeID)
		
		return nil
	}
	
	return lastErr
}

// handleNodeIDStream handles incoming node ID exchange requests
func (m *NodeIDMapper) handleNodeIDStream(stream network.Stream) {
	defer stream.Close()
	
	peerID := stream.Conn().RemotePeer()
	
	// Read their node ID
	decoder := json.NewDecoder(bufio.NewReader(stream))
	var msg NodeIDMessage
	if err := decoder.Decode(&msg); err != nil {
		log.Printf("Failed to read node ID from %s: %v", peerID, err)
		return
	}
	
	// Store the mapping
	m.SetMapping(peerID, msg.NodeID)
	
	// Send our node ID back
	response := NodeIDMessage{NodeID: GetNodeId()}
	encoder := json.NewEncoder(stream)
	if err := encoder.Encode(&response); err != nil {
		log.Printf("Failed to send node ID to %s: %v", peerID, err)
		return
	}
}
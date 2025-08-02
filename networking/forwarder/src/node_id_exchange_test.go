package forwarder

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockNodeIDStreamHandler creates a stream handler that responds with a specific node ID
func mockNodeIDStreamHandler(nodeID string) func(stream network.Stream) {
	return func(stream network.Stream) {
		defer stream.Close()
		
		peerID := stream.Conn().RemotePeer()
		
		// Read their node ID
		decoder := json.NewDecoder(bufio.NewReader(stream))
		var msg NodeIDMessage
		if err := decoder.Decode(&msg); err != nil {
			log.Printf("Failed to read node ID from %s: %v", peerID, err)
			return
		}
		
		// Send our node ID back
		response := NodeIDMessage{NodeID: nodeID}
		encoder := json.NewEncoder(stream)
		if err := encoder.Encode(&response); err != nil {
			log.Printf("Failed to send node ID to %s: %v", peerID, err)
			return
		}
	}
}

func TestNodeIDExchange(t *testing.T) {
	// Create two test hosts
	h1, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	require.NoError(t, err)
	defer h1.Close()

	h2, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	require.NoError(t, err)
	defer h2.Close()

	// Set up node ID for host 1
	SetNodeId("node-1")
	mapper1 := GetNodeIDMapper()
	mapper1.SetHost(h1)

	// Set up host 2 with a mock handler that responds with "node-2"
	h2.SetStreamHandler(NodeIDExchangeProtocol, mockNodeIDStreamHandler("node-2"))

	// Connect the hosts
	h1.Peerstore().AddAddrs(h2.ID(), h2.Addrs(), 3600)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	err = h1.Connect(ctx, peer.AddrInfo{ID: h2.ID(), Addrs: h2.Addrs()})
	require.NoError(t, err)

	// Exchange node IDs
	err = mapper1.ExchangeNodeID(h2.ID())
	require.NoError(t, err)

	// Verify the mapping on host 1
	nodeID, ok := mapper1.GetNodeIDForPeer(h2.ID())
	assert.True(t, ok)
	assert.Equal(t, "node-2", nodeID)
}

func TestNodeIDMapperOperations(t *testing.T) {
	mapper := &NodeIDMapper{
		peerToNode: make(map[peer.ID]string),
		nodeToPeer: make(map[string]peer.ID),
	}

	// Test peer ID (simulated)
	peerID := peer.ID("test-peer-id")
	nodeID := "test-node-id"

	// Set mapping
	mapper.SetMapping(peerID, nodeID)

	// Verify forward mapping
	gotNodeID, ok := mapper.GetNodeIDForPeer(peerID)
	assert.True(t, ok)
	assert.Equal(t, nodeID, gotNodeID)

	// Verify reverse mapping
	gotPeerID, ok := mapper.GetPeerIDForNode(nodeID)
	assert.True(t, ok)
	assert.Equal(t, peerID, gotPeerID)

	// Remove mapping
	mapper.RemoveMapping(peerID)

	// Verify removal
	_, ok = mapper.GetNodeIDForPeer(peerID)
	assert.False(t, ok)

	_, ok = mapper.GetPeerIDForNode(nodeID)
	assert.False(t, ok)
}
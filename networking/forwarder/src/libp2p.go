package forwarder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"log"
	"net"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	"github.com/libp2p/go-libp2p/core/pnet"
	mdns "github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/libp2p/go-libp2p/p2p/security/noise"
	"github.com/multiformats/go-multiaddr"
)

var node host.Host
var ps *pubsub.PubSub
var mdnsSer mdns.Service
var once sync.Once
var mu sync.Mutex
var refCount int
var topicsMap = make(map[string]*pubsub.Topic)

// Connection retry state tracking
type peerConnState struct {
	retryCount  int
	lastAttempt time.Time
}

var peerLastAddrs = make(map[peer.ID][]multiaddr.Multiaddr)
var addrsMu sync.Mutex

var connecting = make(map[peer.ID]bool)
var connMu sync.Mutex
var peerRetryState = make(map[peer.ID]*peerConnState)
var retryMu sync.Mutex

const (
	maxRetries     = 5 // Increased for more tolerance to transient failures
	initialBackoff = 2 * time.Second
	maxBackoff     = 33 * time.Second
	retryResetTime = 1 * time.Minute // Reduced for faster recovery after max retries
)

type discoveryNotifee struct {
	h host.Host
}

// sortAddrs returns a sorted copy of addresses for comparison
func sortAddrs(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
	s := make([]multiaddr.Multiaddr, len(addrs))
	copy(s, addrs)
	sort.Slice(s, func(i, j int) bool {
		return s[i].String() < s[j].String()
	})
	return s
}

// addrsChanged checks if two address sets differ
func addrsChanged(a, b []multiaddr.Multiaddr) bool {
	if len(a) != len(b) {
		return true
	}
	sa := sortAddrs(a)
	sb := sortAddrs(b)
	for i := range sa {
		if !sa[i].Equal(sb[i]) {
			return true
		}
	}
	return false
}

// isAddressValid checks if an address should be used for connections
func isAddressValid(addr multiaddr.Multiaddr) bool {
	// Allow loopback for testing if env var is set
	allowLoopback := os.Getenv("FORWARDER_ALLOW_LOOPBACK") == "true"

	// Check IPv4 addresses
	ipStr, err := addr.ValueForProtocol(multiaddr.P_IP4)
	if err == nil && ipStr != "" {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			return false
		}
		// Filter out loopback, unspecified addresses (unless testing)
		if !allowLoopback && (ip.IsLoopback() || ip.IsUnspecified()) {
			return false
		}
		if ip.IsUnspecified() {
			return false
		}
		// Filter out common VPN ranges (Tailscale uses 100.64.0.0/10)
		if ip.To4() != nil && ip.To4()[0] == 100 && ip.To4()[1] >= 64 && ip.To4()[1] <= 127 {
			return false
		}
	}

	// Check IPv6 addresses
	ipStr, err = addr.ValueForProtocol(multiaddr.P_IP6)
	if err == nil && ipStr != "" {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			return false
		}
		// Filter out loopback, unspecified addresses (unless testing)
		if !allowLoopback && (ip.IsLoopback() || ip.IsUnspecified()) {
			return false
		}
		if ip.IsUnspecified() {
			return false
		}
		// Filter out Tailscale IPv6 (fd7a:115c:a1e0::/48)
		if strings.HasPrefix(strings.ToLower(ipStr), "fd7a:115c:a1e0:") {
			return false
		}
	}

	return true
}

// customInterfaceAddresses returns IPs only from interfaces that are up and running (has link)
func customInterfaceAddresses() ([]net.IP, error) {
	var ips []net.IP
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	for _, ifi := range ifaces {
		if ifi.Flags&net.FlagUp == 0 || ifi.Flags&net.FlagRunning == 0 {
			continue
		}
		addrs, err := ifi.Addrs()
		if err != nil {
			return nil, err
		}
		for _, addr := range addrs {
			if ipnet, ok := addr.(*net.IPNet); ok && ipnet.IP != nil {
				ips = append(ips, ipnet.IP)
			}
		}
	}
	return ips, nil
}

// customAddrsFactory expands wildcard listen addrs to actual IPs on up+running interfaces, then filters
func customAddrsFactory(listenAddrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
	ips, err := customInterfaceAddresses()
	if err != nil {
		log.Printf("Error getting interface IPs: %v", err)
		return nil
	}

	var advAddrs []multiaddr.Multiaddr
	for _, la := range listenAddrs {
		comps := multiaddr.Split(la)
		if len(comps) == 0 {
			continue
		}
		first := comps[0]
		protos := first.Protocols()
		if len(protos) == 0 {
			continue
		}
		code := protos[0].Code
		val, err := first.ValueForProtocol(code)
		var isWildcard bool
		if err == nil && ((code == multiaddr.P_IP4 && val == "0.0.0.0") || (code == multiaddr.P_IP6 && val == "::")) {
			isWildcard = true
		}

		if isWildcard {
			// Expand to each valid IP
			for _, ip := range ips {
				var pcodeStr string
				if ip.To4() != nil {
					pcodeStr = "4"
				} else {
					pcodeStr = "6"
				}
				newIPStr := "/ip" + pcodeStr + "/" + ip.String()
				newIPMA, err := multiaddr.NewMultiaddr(newIPStr)
				if err != nil {
					continue
				}
				var newComps []multiaddr.Multiaddrer
				newComps = append(newComps, newIPMA)
				for _, c := range comps[1:] {
					newComps = append(newComps, c.Multiaddr())
				}
				newa := multiaddr.Join(newComps...)
				if isAddressValid(newa) {
					advAddrs = append(advAddrs, newa)
				}
			}
		} else if isAddressValid(la) {
			advAddrs = append(advAddrs, la)
		}
	}
	return advAddrs
}

func (n *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	log.Printf("mDNS discovered peer %s with %d addresses", pi.ID, len(pi.Addrs))

	// Check if already connected first
	if n.h.Network().Connectedness(pi.ID) == network.Connected {
		log.Printf("Already connected to peer %s", pi.ID)
		return
	}

	// Clear any existing addresses for this peer to ensure we use only fresh ones from mDNS
	ps := n.h.Peerstore()
	ps.ClearAddrs(pi.ID)
	log.Printf("Cleared old addresses for peer %s", pi.ID)

	// During normal operation, only higher ID connects to avoid double connections
	// But if we have retry state for this peer, both sides should attempt
	// Also, if we have no connections at all, both sides should attempt
	retryMu.Lock()
	_, hasRetryState := peerRetryState[pi.ID]
	retryMu.Unlock()

	// Check if we should skip based on ID comparison
	// Skip only if we have a higher ID, no retry state, and we already have connections
	if n.h.ID() >= pi.ID && !hasRetryState && len(n.h.Network().Peers()) > 0 {
		log.Printf("Skipping initial connection to peer %s (lower ID)", pi.ID)
		return
	}

	// Filter addresses before attempting connection
	var filteredAddrs []multiaddr.Multiaddr
	for _, addr := range pi.Addrs {
		if isAddressValid(addr) {
			filteredAddrs = append(filteredAddrs, addr)
			log.Printf("Valid address for %s: %s", pi.ID, addr)
		} else {
			log.Printf("Filtered out address for %s: %s", pi.ID, addr)
		}
	}

	if len(filteredAddrs) == 0 {
		log.Printf("No valid addresses for peer %s after filtering, skipping connection attempt", pi.ID)
		return
	}

	// Check for address changes and reset retries if changed
	addrsMu.Lock()
	lastAddrs := peerLastAddrs[pi.ID]
	addrsMu.Unlock()
	if addrsChanged(lastAddrs, filteredAddrs) {
		log.Printf("Detected address change for peer %s, resetting retry count", pi.ID)
		retryMu.Lock()
		if state, ok := peerRetryState[pi.ID]; ok {
			state.retryCount = 0
		}
		retryMu.Unlock()
		// Update last known addresses
		addrsMu.Lock()
		peerLastAddrs[pi.ID] = append([]multiaddr.Multiaddr(nil), filteredAddrs...) // Copy
		addrsMu.Unlock()
	}

	pi.Addrs = filteredAddrs

	// Add the filtered addresses to the peerstore with a reasonable TTL
	ps.AddAddrs(pi.ID, filteredAddrs, peerstore.TempAddrTTL)

	// Attempt connection with retry logic
	go n.connectWithRetry(pi)
}

func (n *discoveryNotifee) connectWithRetry(pi peer.AddrInfo) {
	// Serialize connection attempts per peer
	connMu.Lock()
	if connecting[pi.ID] {
		connMu.Unlock()
		log.Printf("Already connecting to peer %s, skipping duplicate attempt", pi.ID)
		return
	}
	connecting[pi.ID] = true
	connMu.Unlock()
	defer func() {
		connMu.Lock()
		delete(connecting, pi.ID)
		connMu.Unlock()
	}()

	retryMu.Lock()
	state, exists := peerRetryState[pi.ID]
	if !exists {
		state = &peerConnState{}
		peerRetryState[pi.ID] = state
	}

	// Check if we've exceeded max retries
	if state.retryCount >= maxRetries {
		// Check if enough time has passed to reset retry count
		if time.Since(state.lastAttempt) > retryResetTime {
			state.retryCount = 0
			log.Printf("Reset retry count for peer %s due to time elapsed", pi.ID)
		} else {
			retryMu.Unlock()
			log.Printf("Max retries reached for peer %s, skipping", pi.ID)
			return
		}
	}

	// Calculate backoff duration
	backoffDuration := time.Duration(1<<uint(state.retryCount)) * initialBackoff
	if backoffDuration > maxBackoff {
		backoffDuration = maxBackoff
	}

	// Check if we need to wait before retrying
	if state.retryCount > 0 && time.Since(state.lastAttempt) < backoffDuration {
		retryMu.Unlock()
		log.Printf("Backoff active for peer %s, skipping attempt", pi.ID)
		return
	}

	state.lastAttempt = time.Now()
	retryMu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := n.h.Connect(ctx, pi); err != nil {
		log.Printf("Failed to connect to %s (attempt %d/%d): %v", pi.ID, state.retryCount+1, maxRetries, err)

		retryMu.Lock()
		state.retryCount++
		retryMu.Unlock()

		// Schedule retry if we haven't exceeded max attempts
		if state.retryCount < maxRetries {
			time.AfterFunc(backoffDuration, func() {
				// Check if we're still not connected before retrying
				if n.h.Network().Connectedness(pi.ID) != network.Connected {
					n.connectWithRetry(pi)
				}
			})
		}
	} else {
		log.Printf("Successfully connected to %s", pi.ID)

		// Reset retry state on successful connection
		retryMu.Lock()
		delete(peerRetryState, pi.ID)
		retryMu.Unlock()
		addrsMu.Lock()
		delete(peerLastAddrs, pi.ID)
		addrsMu.Unlock()
		log.Printf("Cleared last addresses for disconnected peer %s", pi.ID)
	}
}

func getPrivKey(nodeId string) (crypto.PrivKey, error) {
	seed := sha256.Sum256([]byte(nodeId))
	priv, _, err := crypto.GenerateEd25519Key(bytes.NewReader(seed[:]))
	if err != nil {
		return nil, err
	}
	return priv, nil
}

func getNode(ctx context.Context) {
	once.Do(func() {
		nodeId := GetNodeId()
		var opts []libp2p.Option
		priv, err := getPrivKey(nodeId)
		if err != nil {
			log.Fatalf("failed to generate key: %v", err)
		}
		opts = append(opts, libp2p.Identity(priv))
		opts = append(opts, libp2p.Security(noise.ID, noise.New))

		pskHash := sha256.Sum256([]byte("forwarder_network"))
		psk := pnet.PSK(pskHash[:])
		opts = append(opts, libp2p.PrivateNetwork(psk))

		// Performance optimizations
		opts = append(opts, libp2p.ConnectionManager(nil)) // No connection limits
		opts = append(opts, libp2p.EnableHolePunching())   // Better NAT traversal
		opts = append(opts, libp2p.EnableRelay())          // Allow relaying

		// Custom address factory to avoid advertising down interfaces
		opts = append(opts, libp2p.AddrsFactory(customAddrsFactory))

		node, err = libp2p.New(opts...)
		if err != nil {
			log.Fatalf("failed to create host: %v", err)
		}

		// Configure GossipSub for better performance
		gossipOpts := []pubsub.Option{
			pubsub.WithMessageSigning(false),              // Disable message signing for speed
			pubsub.WithStrictSignatureVerification(false), // Disable signature verification
			pubsub.WithMaxMessageSize(1024 * 1024),        // 1MB max message size for batches
			pubsub.WithValidateQueueSize(1000),            // Larger validation queue
			pubsub.WithPeerOutboundQueueSize(1000),        // Larger peer queues
		}

		ps, err = pubsub.NewGossipSub(ctx, node, gossipOpts...)
		if err != nil {
			node.Close()
			log.Fatalf("failed to create pubsub: %v", err)
		}

		rendezvous := "forwarder_network"
		notifee := &discoveryNotifee{h: node}
		mdnsSer = mdns.NewMdnsService(node, rendezvous, notifee)
		if err := mdnsSer.Start(); err != nil {
			node.Close()
			log.Fatalf("failed to start mdns service: %v", err)
		}

		// Register disconnect notifiee to clear stale addresses
		node.Network().Notify(&disconnectNotifee{})

		// Register event notifiee to track topology changes
		node.Network().Notify(GetNotifee())
		
		// Set up node ID mapper
		GetNodeIDMapper().SetHost(node)

		// Start a goroutine to periodically trigger mDNS discovery
		go periodicMDNSDiscovery()
	})
}

// periodicMDNSDiscovery ensures mDNS continues to work after network changes
func periodicMDNSDiscovery() {
	// Start with faster checks, then slow down
	fastCheckDuration := 5 * time.Second
	slowCheckDuration := 30 * time.Second
	currentDuration := fastCheckDuration
	noConnectionCount := 0

	ticker := time.NewTicker(currentDuration)
	defer ticker.Stop()

	for range ticker.C {
		if mdnsSer == nil || node == nil {
			return
		}

		// Log current connection status
		peers := node.Network().Peers()
		if len(peers) == 0 {
			noConnectionCount++
			log.Printf("No connected peers (check #%d), mDNS service running: %v", noConnectionCount, mdnsSer != nil)

			// Force mDNS to re-announce when we have no peers
			// This helps recovery after network interface changes
			if noConnectionCount > 1 { // Skip first check to avoid unnecessary restart
				forceRestartMDNS()
			}

			// Keep fast checking when disconnected
			if currentDuration != fastCheckDuration {
				currentDuration = fastCheckDuration
				ticker.Reset(currentDuration)
				log.Printf("Switching to fast mDNS checks (every %v)", currentDuration)
			}
		} else {
			log.Printf("Currently connected to %d peers", len(peers))
			noConnectionCount = 0

			// Switch to slow checking when connected
			if currentDuration != slowCheckDuration {
				currentDuration = slowCheckDuration
				ticker.Reset(currentDuration)
				log.Printf("Switching to slow mDNS checks (every %v)", currentDuration)
			}
		}
	}
}

// forceRestartMDNS restarts the mDNS service to force re-announcement
func forceRestartMDNS() {
	mu.Lock()
	defer mu.Unlock()

	if mdnsSer != nil && node != nil {
		log.Printf("Force restarting mDNS service for re-announcement")
		oldMdns := mdnsSer
		rendezvous := "forwarder_network"
		notifee := &discoveryNotifee{h: node}
		newMdns := mdns.NewMdnsService(node, rendezvous, notifee)

		if err := newMdns.Start(); err != nil {
			log.Printf("Failed to restart mDNS service: %v", err)
		} else {
			oldMdns.Close()
			mdnsSer = newMdns
			log.Printf("Successfully restarted mDNS service")
		}
	}
}

// disconnectNotifee clears stale peer addresses on disconnect
type disconnectNotifee struct{}

func (d *disconnectNotifee) Connected(network.Network, network.Conn) {}
func (d *disconnectNotifee) Disconnected(n network.Network, c network.Conn) {
	p := c.RemotePeer()
	ps := n.Peerstore()

	// Clear all addresses from peerstore to force fresh discovery on reconnect
	ps.ClearAddrs(p)

	// Also clear retry state for this peer
	retryMu.Lock()
	delete(peerRetryState, p)
	retryMu.Unlock()

	log.Printf("Cleared stale addresses and retry state for disconnected peer %s", p)

	// Try to restart mDNS discovery after a short delay to handle network interface changes
	go func() {
		time.Sleep(2 * time.Second)
		log.Printf("Triggering mDNS re-discovery after disconnect")
		forceRestartMDNS()
	}()
}
func (d *disconnectNotifee) OpenedStream(network.Network, network.Stream)     {}
func (d *disconnectNotifee) ClosedStream(network.Network, network.Stream)     {}
func (d *disconnectNotifee) Listen(network.Network, multiaddr.Multiaddr)      {}
func (d *disconnectNotifee) ListenClose(network.Network, multiaddr.Multiaddr) {}

type libP2PConnector struct {
	topic     string
	sub       *pubsub.Subscription
	subResend *pubsub.Subscription
	top       *pubsub.Topic
	topResend *pubsub.Topic
	ctx       context.Context
	cancel    context.CancelFunc

	// Async publishing
	writeChan    chan RecordData
	batchSize    int
	batchTimeout time.Duration
	workerPool   int
}

func newLibP2PConnector(topic string, ctx context.Context, cancel context.CancelFunc) *libP2PConnector {
	getNode(ctx)
	mu.Lock()
	var err error
	t, ok := topicsMap[topic]
	if !ok {
		t, err = ps.Join(topic)
		if err != nil {
			mu.Unlock()
			log.Fatalf("failed to join topic %s: %v", topic, err)
		}
		topicsMap[topic] = t
	}

	t2, okResend := topicsMap[topic+"/resend"]
	if !okResend {
		t2, err = ps.Join(topic + "/resend")
		if err != nil {
			mu.Unlock()
			log.Fatalf("failed to join topic %s: %v", topic+"/resend", err)
		}
		topicsMap[topic+"/resend"] = t2
	}

	refCount++
	mu.Unlock()

	connector := &libP2PConnector{
		topic:        topic,
		top:          t,
		topResend:    t2,
		ctx:          ctx,
		cancel:       cancel,
		writeChan:    make(chan RecordData, 2000),
		batchSize:    100,
		batchTimeout: 10 * time.Millisecond,
		workerPool:   5,
	}

	connector.startAsyncPublishers()

	return connector
}

func (c *libP2PConnector) tail(handler func(record RecordData) error) {
	sub, err := c.top.Subscribe()
	if err != nil {
		log.Fatalf("failed to subscribe to topic %s: %v", c.topic, err)
	}
	c.sub = sub
	go handleRecordSub(c.sub, c.ctx, handler)
}

func (c *libP2PConnector) tailResend(handler func(data ResendRequest) error) {
	sub, err := c.topResend.Subscribe()
	if err != nil {
		log.Fatalf("failed to subscribe to topic %s: %v", c.topic, err)
	}
	c.subResend = sub
	go handleSub(c.subResend, c.ctx, handler)
}

func handleSub[T any](sub *pubsub.Subscription, ctx context.Context, handler func(data T) error) {
	for {
		msg, err := sub.Next(ctx)
		if err != nil {
			if err == context.Canceled {
				return
			}
			log.Printf("subscription error for topic %s: %v", sub.Topic(), err)
			return
		}
		var rec T
		err = json.Unmarshal(msg.Data, &rec)
		if err != nil {
			log.Printf("unmarshal error for topic %s: %v", sub.Topic(), err)
			continue
		}
		if handler != nil {
			if err := handler(rec); err != nil {
				log.Printf("handler error for topic %s: %v", sub.Topic(), err)
			}
		}
	}
}

func handleRecordSub(sub *pubsub.Subscription, ctx context.Context, handler func(record RecordData) error) {
	for {
		msg, err := sub.Next(ctx)
		if err != nil {
			if err == context.Canceled {
				return
			}
			log.Printf("subscription error for topic %s: %v", sub.Topic(), err)
			return
		}

		// Try to unmarshal as batch first
		var batch BatchRecord
		if err := json.Unmarshal(msg.Data, &batch); err == nil && len(batch.Records) > 0 {
			// Handle batched records
			for _, record := range batch.Records {
				if handler != nil {
					if err := handler(record); err != nil {
						log.Printf("handler error for batched record: %v", err)
					}
				}
			}
			continue
		}

		// Try to unmarshal as single record (backwards compatibility)
		var record RecordData
		if err := json.Unmarshal(msg.Data, &record); err == nil {
			if handler != nil {
				if err := handler(record); err != nil {
					log.Printf("handler error for single record: %v", err)
				}
			}
			continue
		}

		log.Printf("failed to unmarshal message as batch or single record for topic %s", sub.Topic())
	}
}

func (c *libP2PConnector) startAsyncPublishers() {
	// Start worker pool for batched async publishing
	for i := 0; i < c.workerPool; i++ {
		go c.publishWorker()
	}
}

func (c *libP2PConnector) publishWorker() {
	batch := make([]RecordData, 0, c.batchSize)
	timer := time.NewTimer(c.batchTimeout)
	timer.Stop()

	for {
		select {
		case <-c.ctx.Done():
			// Flush final batch
			if len(batch) > 0 {
				err := c.publishBatch(batch)
				if err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
			}
			return

		case record := <-c.writeChan:
			batch = append(batch, record)

			// Check if we should flush
			if len(batch) >= c.batchSize {
				err := c.publishBatch(batch)
				if err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
				batch = batch[:0]
				timer.Stop()
			} else if len(batch) == 1 {
				// First record in batch, start timer
				timer.Reset(c.batchTimeout)
			}

		case <-timer.C:
			// Timer expired, flush whatever we have
			if len(batch) > 0 {
				err := c.publishBatch(batch)
				if err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
				batch = batch[:0]
			}
		}
	}
}

func (c *libP2PConnector) publishBatch(records []RecordData) error {
	if len(records) == 0 {
		return nil
	}

	// Create batch record
	batchRecord := BatchRecord{Records: records}

	data, err := json.Marshal(batchRecord)
	if err != nil {
		return err
	}

	// Publish with timeout to prevent blocking
	go func() {
		pubCtx, pubCancel := context.WithTimeout(c.ctx, 100*time.Millisecond)
		defer pubCancel()

		if err := c.top.Publish(pubCtx, data); err != nil {
			if err != context.DeadlineExceeded {
				log.Printf("Error publishing batch of %d records: %v", len(records), err)
			}
		}
	}()
	return nil
}

func (c *libP2PConnector) write(record RecordData) error {
	select {
	case c.writeChan <- record:
		return nil
	case <-c.ctx.Done():
		return c.ctx.Err()
	default:
		// Channel full, try to publish directly
		return c.publishSingle(record)
	}
}

func (c *libP2PConnector) publishSingle(record RecordData) error {
	if c.top == nil {
		return context.Canceled
	}
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	return c.top.Publish(c.ctx, data)
}

func (c *libP2PConnector) writeResend(req ResendRequest) error {
	if c.topResend == nil {
		return context.Canceled
	}
	data, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return c.topResend.Publish(c.ctx, data)
}

func (c *libP2PConnector) close() error {
	mu.Lock()
	refCount--
	closeHost := refCount == 0
	mu.Unlock()

	if c.cancel != nil {
		c.cancel()
	}
	if c.sub != nil {
		c.sub.Cancel()
	}
	if c.subResend != nil {
		c.subResend.Cancel()
	}
	if closeHost {
		// close all topics when shutting down host
		for _, top := range topicsMap {
			_ = top.Close()
		}
		topicsMap = make(map[string]*pubsub.Topic)
	}

	c.top = nil

	if !closeHost {
		return nil
	}

	if mdnsSer != nil {
		_ = mdnsSer.Close()
		mdnsSer = nil
	}

	var err error
	if node != nil {
		err = node.Close()
	}

	node = nil
	ps = nil
	refCount = 0
	once = sync.Once{}

	return err
}

func (c *libP2PConnector) getType() string {
	return "libp2p"
}

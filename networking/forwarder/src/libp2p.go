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
	connmgr "github.com/libp2p/go-libp2p/p2p/net/connmgr"
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

type peerConnState struct {
	retryCount  int
	lastAttempt time.Time
}

type peerAddrKey struct {
	id   peer.ID
	addr string // host+transport key (IP|transport)
}

var (
	peerRetryState = make(map[peerAddrKey]*peerConnState)
	retryMu        sync.Mutex

	connecting = make(map[peerAddrKey]bool)
	connMu     sync.Mutex

	mdnsRestartMu     sync.Mutex
	lastMdnsRestart   time.Time
	restartPending    bool
	minRestartSpacing = 2 * time.Second
)

const (
	connectTimeout   = 25 * time.Second
	mdnsFastInterval = 1 * time.Second
	mdnsSlowInterval = 30 * time.Second
)

func sortAddrs(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
	s := make([]multiaddr.Multiaddr, len(addrs))
	copy(s, addrs)
	sort.Slice(s, func(i, j int) bool {
		return s[i].String() < s[j].String()
	})
	return s
}

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

func canonicalAddr(a multiaddr.Multiaddr) string {
	cs := multiaddr.Split(a)
	out := make([]multiaddr.Multiaddrer, 0, len(cs))
	for _, c := range cs {
		for _, p := range c.Protocols() {
			if p.Code == multiaddr.P_P2P {
				goto NEXT
			}
		}
		out = append(out, c.Multiaddr())
	NEXT:
	}
	return multiaddr.Join(out...).String()
}

func ipString(a multiaddr.Multiaddr) string {
	if v, err := a.ValueForProtocol(multiaddr.P_IP4); err == nil {
		return v
	}
	if v, err := a.ValueForProtocol(multiaddr.P_IP6); err == nil {
		return v
	}
	return ""
}

func hostTransportKey(a multiaddr.Multiaddr) string {
	ip := ipString(a)
	t := "tcp"
	if _, err := a.ValueForProtocol(multiaddr.P_QUIC_V1); err == nil {
		t = "quic"
	}
	if _, err := a.ValueForProtocol(multiaddr.P_WS); err == nil {
		t = "ws"
	}
	return ip + "|" + t
}

func isAddressValid(addr multiaddr.Multiaddr) bool {
	allowLoopback := os.Getenv("FORWARDER_ALLOW_LOOPBACK") == "true"

	if ipStr, err := addr.ValueForProtocol(multiaddr.P_IP4); err == nil && ipStr != "" {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			return false
		}
		if !allowLoopback && (ip.IsLoopback() || ip.IsUnspecified()) {
			return false
		}
		if ip.IsUnspecified() {
			return false
		}
		if b := ip.To4(); b != nil && b[0] == 100 && b[1] >= 64 && b[1] <= 127 {
			return false
		}
	}

	if ipStr, err := addr.ValueForProtocol(multiaddr.P_IP6); err == nil && ipStr != "" {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			return false
		}
		if !allowLoopback && (ip.IsLoopback() || ip.IsUnspecified()) {
			return false
		}
		if ip.IsUnspecified() {
			return false
		}
		if strings.HasPrefix(strings.ToLower(ipStr), "fd7a:115c:a1e0:") {
			return false
		}
	}

	return true
}

func customInterfaceAddresses() ([]net.IP, error) {
	var ips []net.IP
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	for _, ifi := range ifaces {
		if ifi.Flags&net.FlagUp == 0 {
			continue
		}
		addrs, err := ifi.Addrs()
		if err != nil {
			return nil, err
		}
		for _, a := range addrs {
			if ipnet, ok := a.(*net.IPNet); ok && ipnet.IP != nil {
				ips = append(ips, ipnet.IP)
			}
		}
	}
	return ips, nil
}

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
		isWildcard := (err == nil &&
			((code == multiaddr.P_IP4 && val == "0.0.0.0") ||
				(code == multiaddr.P_IP6 && val == "::")))

		if isWildcard {
			for _, ip := range ips {
				var pcode string
				if ip.To4() != nil {
					pcode = "4"
				} else {
					pcode = "6"
				}
				newIPMA, err := multiaddr.NewMultiaddr("/ip" + pcode + "/" + ip.String())
				if err != nil {
					continue
				}
				var newComps []multiaddr.Multiaddrer
				newComps = append(newComps, newIPMA)
				for _, c := range comps[1:] {
					newComps = append(newComps, c.Multiaddr())
				}
				newAddr := multiaddr.Join(newComps...)
				if isAddressValid(newAddr) {
					advAddrs = append(advAddrs, newAddr)
				}
			}
		} else if isAddressValid(la) {
			advAddrs = append(advAddrs, la)
		}
	}
	return advAddrs
}

type discoveryNotifee struct{ h host.Host }

func (n *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	log.Printf("mDNS discovered peer %s with %d addresses", pi.ID, len(pi.Addrs))

	var ipList []string
	for _, a := range pi.Addrs {
		if v := ipString(a); v != "" {
			ipList = append(ipList, v)
		}
	}
	if len(ipList) > 0 {
		log.Printf("mDNS %s IPs: %s", pi.ID, strings.Join(ipList, ", "))
	}

	var filtered []multiaddr.Multiaddr
	var ips []net.IP
	for _, a := range pi.Addrs {
		if isAddressValid(a) {
			filtered = append(filtered, a)

			if ipStr := ipString(a); ipStr != "" {
				if ip := net.ParseIP(ipStr); ip != nil {
					ips = append(ips, ip)
				}
			}
		}
	}
	if len(filtered) == 0 {
		log.Printf("No valid addrs for %s", pi.ID)
		return
	}

	ps := n.h.Peerstore()
	ps.AddAddrs(pi.ID, filtered, peerstore.TempAddrTTL)

	tcpAgent := GetTCPAgent()
	if len(ips) > 0 {
		tcpAgent.UpdateDiscoveredIPs(pi.ID, ips)
	}

	existing := make(map[string]struct{})
	for _, c := range n.h.Network().ConnsToPeer(pi.ID) {
		if cm, ok := c.(network.ConnMultiaddrs); ok {
			existing[hostTransportKey(cm.RemoteMultiaddr())] = struct{}{}
		}
	}

	for _, a := range filtered {
		if _, seen := existing[hostTransportKey(a)]; seen {
			continue
		}
		go n.connectWithRetryToAddr(pi.ID, a)
	}
}

func (n *discoveryNotifee) connectWithRetryToAddr(pid peer.ID, addr multiaddr.Multiaddr) {
	key := peerAddrKey{pid, hostTransportKey(addr)}

	connMu.Lock()
	if connecting[key] {
		connMu.Unlock()
		return
	}
	connecting[key] = true
	connMu.Unlock()
	defer func() {
		connMu.Lock()
		delete(connecting, key)
		connMu.Unlock()
	}()

	retryMu.Lock()
	state, ok := peerRetryState[key]
	if !ok {
		state = &peerConnState{}
		peerRetryState[key] = state
	}
	backoff := time.Duration(1<<uint(state.retryCount)) * initialBackoff
	if backoff > maxBackoff {
		backoff = maxBackoff
	}
	if state.retryCount > 0 && time.Since(state.lastAttempt) < backoff {
		retryMu.Unlock()
		return
	}
	state.lastAttempt = time.Now()
	retryMu.Unlock()

	ai := peer.AddrInfo{ID: pid, Addrs: []multiaddr.Multiaddr{addr}}

	ctx, cancel := context.WithTimeout(network.WithForceDirectDial(context.Background(), "ensure-multipath"), connectTimeout)
	defer cancel()

	n.h.Peerstore().AddAddrs(pid, []multiaddr.Multiaddr{addr}, peerstore.TempAddrTTL)

	if err := n.h.Connect(ctx, ai); err != nil {
		log.Printf("Dial %s@%s failed (attempt %d): %v", pid, addr, state.retryCount+1, err)
		retryMu.Lock()
		state.retryCount++
		retryMu.Unlock()

		time.AfterFunc(backoff, func() {
			pathStillMissing := true
			for _, c := range n.h.Network().ConnsToPeer(pid) {
				if cm, ok := c.(network.ConnMultiaddrs); ok &&
					hostTransportKey(cm.RemoteMultiaddr()) == key.addr {
					pathStillMissing = false
					break
				}
			}
			if pathStillMissing {
				n.connectWithRetryToAddr(pid, addr)
			}
		})
		return
	}

	log.Printf("Connected to %s via %s", pid, addr)
	retryMu.Lock()
	delete(peerRetryState, key)
	retryMu.Unlock()
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

		opts = append(opts, libp2p.EnableHolePunching())
		opts = append(opts, libp2p.EnableRelay())

		opts = append(opts, libp2p.AddrsFactory(customAddrsFactory))

		cm, _ := connmgr.NewConnManager(100, 1000, connmgr.WithGracePeriod(2*time.Minute))
		opts = append(opts, libp2p.ConnectionManager(cm))

		var errNode error
		node, errNode = libp2p.New(opts...)
		if errNode != nil {
			log.Fatalf("failed to create host: %v", errNode)
		}

		gossipOpts := []pubsub.Option{
			pubsub.WithMessageSigning(false),
			pubsub.WithStrictSignatureVerification(false),
			pubsub.WithMaxMessageSize(1024 * 1024),
			pubsub.WithValidateQueueSize(1000),
			pubsub.WithPeerOutboundQueueSize(1000),
		}
		ps, err = pubsub.NewGossipSub(ctx, node, gossipOpts...)
		if err != nil {
			_ = node.Close()
			log.Fatalf("failed to create pubsub: %v", err)
		}

		rendezvous := "forwarder_network"
		notifee := &discoveryNotifee{h: node}
		mdnsSer = mdns.NewMdnsService(node, rendezvous, notifee)
		if err := mdnsSer.Start(); err != nil {
			_ = node.Close()
			log.Fatalf("failed to start mdns service: %v", err)
		}

		node.Network().Notify(&disconnectNotifee{})
		node.Network().Notify(GetNotifee())

		tcpAgent := GetTCPAgent()
		if err := tcpAgent.Start(ctx, node.ID()); err != nil {
			log.Printf("Failed to start  TCP agent: %v", err)
		}

		go periodicMDNSDiscovery()
		go watchInterfacesAndKickMDNS()
	})
}

func periodicMDNSDiscovery() {
	current := mdnsSlowInterval
	t := time.NewTicker(current)
	defer t.Stop()

	lastNoPeerRestart := time.Time{}

	for range t.C {
		if mdnsSer == nil || node == nil {
			return
		}
		n := len(node.Network().Peers())
		if n == 0 {
			if current != mdnsFastInterval {
				current = mdnsFastInterval
				t.Reset(current)
			}
			if time.Since(lastNoPeerRestart) > 5*time.Second {
				forceRestartMDNS("no-peers")
				lastNoPeerRestart = time.Now()
			}
		} else {
			if current != mdnsSlowInterval {
				current = mdnsSlowInterval
				t.Reset(current)
			}
		}
	}
}

func watchInterfacesAndKickMDNS() {
	snap := interfacesSignature()
	t := time.NewTicker(1 * time.Second)
	defer t.Stop()

	for range t.C {
		next := interfacesSignature()
		if next != snap {
			snap = next
			kickMDNSBurst("iface-change")
		}
	}
}

func kickMDNSBurst(reason string) {
	forceRestartMDNS(reason)
	time.AfterFunc(2*time.Second, func() { forceRestartMDNS(reason + "-stabilize-2s") })
	time.AfterFunc(6*time.Second, func() { forceRestartMDNS(reason + "-stabilize-6s") })
}

func interfacesSignature() string {
	ifaces, _ := net.Interfaces()
	var b strings.Builder
	for _, ifi := range ifaces {
		if ifi.Flags&net.FlagUp == 0 {
			continue
		}
		addrs, _ := ifi.Addrs()
		b.WriteString(ifi.Name)
		b.WriteByte('|')
		b.WriteString(ifi.Flags.String())
		for _, a := range addrs {
			b.WriteByte('|')
			b.WriteString(a.String())
		}
		b.WriteByte(';')
	}
	return b.String()
}

func forceRestartMDNS(reason string) {
	mdnsRestartMu.Lock()
	defer mdnsRestartMu.Unlock()

	now := time.Now()
	if restartPending || now.Sub(lastMdnsRestart) < minRestartSpacing {
		if !restartPending {
			restartPending = true
			wait := minRestartSpacing - now.Sub(lastMdnsRestart)
			if wait < 0 {
				wait = minRestartSpacing
			}
			time.AfterFunc(wait, func() {
				forceRestartMDNS("coalesced")
			})
		}
		return
	}
	restartPending = false
	lastMdnsRestart = now

	mu.Lock()
	defer mu.Unlock()

	if mdnsSer != nil && node != nil {
		log.Printf("Restarting mDNS (%s)", reason)
		old := mdnsSer
		rendezvous := "forwarder_network"
		notifee := &discoveryNotifee{h: node}
		newMdns := mdns.NewMdnsService(node, rendezvous, notifee)
		if err := newMdns.Start(); err != nil {
			log.Printf("Failed to restart mDNS: %v", err)
			return
		}
		_ = old.Close()
		mdnsSer = newMdns
		GetTCPAgent().OnInterfaceChange()

		retryMu.Lock()
		peerRetryState = make(map[peerAddrKey]*peerConnState)
		retryMu.Unlock()
	}
}

type disconnectNotifee struct{}

func (d *disconnectNotifee) Connected(network.Network, network.Conn) {}
func (d *disconnectNotifee) Disconnected(n network.Network, c network.Conn) {
	go func() {
		time.Sleep(400 * time.Millisecond)
		forceRestartMDNS("disconnect")
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

	conn := &libP2PConnector{
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
	conn.startAsyncPublishers()
	return conn
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
		if err := json.Unmarshal(msg.Data, &rec); err != nil {
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
		var batch BatchRecord
		if err := json.Unmarshal(msg.Data, &batch); err == nil && len(batch.Records) > 0 {
			for _, r := range batch.Records {
				if handler != nil {
					if err := handler(r); err != nil {
						log.Printf("handler error for batched record: %v", err)
					}
				}
			}
			continue
		}
		var single RecordData
		if err := json.Unmarshal(msg.Data, &single); err == nil {
			if handler != nil {
				if err := handler(single); err != nil {
					log.Printf("handler error for single record: %v", err)
				}
			}
			continue
		}
		log.Printf("failed to unmarshal message for topic %s", sub.Topic())
	}
}

func (c *libP2PConnector) startAsyncPublishers() {
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
			if len(batch) > 0 {
				if err := c.publishBatch(batch); err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
			}
			return

		case r := <-c.writeChan:
			batch = append(batch, r)
			if len(batch) >= c.batchSize {
				if err := c.publishBatch(batch); err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
				batch = batch[:0]
				timer.Stop()
			} else if len(batch) == 1 {
				timer.Reset(c.batchTimeout)
			}

		case <-timer.C:
			if len(batch) > 0 {
				if err := c.publishBatch(batch); err != nil {
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
	data, err := json.Marshal(BatchRecord{Records: records})
	if err != nil {
		return err
	}
	go func() {
		pubCtx, cancel := context.WithTimeout(c.ctx, 100*time.Millisecond)
		defer cancel()
		if err := c.top.Publish(pubCtx, data); err != nil && err != context.DeadlineExceeded {
			log.Printf("Error publishing batch of %d: %v", len(records), err)
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

	tcpAgent := GetTCPAgent()
	if err := tcpAgent.Stop(); err != nil {
		log.Printf("Error stopping  TCP agent: %v", err)
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

func (c *libP2PConnector) getType() string { return "libp2p" }

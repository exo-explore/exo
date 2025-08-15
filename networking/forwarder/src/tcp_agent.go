package forwarder

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

const (
	AgentPort = 7847

	HandshakeTimeout    = 5 * time.Second
	HeartbeatInterval   = 1 * time.Second
	HeartbeatReadGrace  = 4 * time.Second
	HeartbeatWriteGrace = 3 * time.Second

	tbGraceWindow = 90 * time.Second

	dialTimeoutDefault   = 6 * time.Second
	dialTimeoutLinkLocal = 12 * time.Second
	initialBackoff       = 500 * time.Millisecond
	maxBackoff           = 60 * time.Second

	scheduleTick       = 300 * time.Millisecond
	maxConcurrentDials = 32

	ttlDiscovered = 5 * time.Minute
	ttlObserved   = 20 * time.Minute
)

type HandshakeMessage struct {
	NodeID    string   `json:"node_id"`
	AgentVer  string   `json:"agent_version"`
	PeerID    string   `json:"peer_id"`
	IPv4s     []string `json:"ipv4s,omitempty"`
	Timestamp int64    `json:"timestamp"`
}

type Edge struct {
	LocalNodeID  string
	RemoteNodeID string
	LocalIP      string
	RemoteIP     string
	Proto        string
}

func (e Edge) Key() string {
	return fmt.Sprintf("%s|%s|%s|%s|%s", e.LocalNodeID, e.RemoteNodeID, e.LocalIP, e.RemoteIP, e.Proto)
}

type connTrack struct {
	tc      *net.TCPConn
	edge    Edge
	dialer  bool
	closed  chan struct{}
	closeMx sync.Once
}

type ipStamp struct {
	seenAt time.Time
	ttl    time.Duration
}

type dialState struct {
	backoff     time.Duration
	nextAttempt time.Time
	connecting  bool
}

type TCPAgent struct {
	nodeID   string
	myPeerID peer.ID

	listener *net.TCPListener
	ctx      context.Context
	cancel   context.CancelFunc

	edgesMu     sync.RWMutex
	activeEdges map[string]*connTrack

	activeByRemoteIPMu sync.RWMutex
	activeByRemoteIP   map[string]bool

	ipDBMu sync.RWMutex
	ipDB   map[peer.ID]map[string]ipStamp

	dialStatesMu  sync.Mutex
	dialStates    map[string]*dialState
	stopScheduler chan struct{}
	schedulerOnce sync.Once
	schedulerWG   sync.WaitGroup

	dialSem chan struct{}

	ifaceGraceUntilMu sync.RWMutex
	ifaceGraceUntil   time.Time
}

var (
	TCPAgentInstance *TCPAgent
	TCPAgentOnce     sync.Once
)

func GetTCPAgent() *TCPAgent {
	TCPAgentOnce.Do(func() {
		TCPAgentInstance = &TCPAgent{
			nodeID:           GetNodeId(),
			activeEdges:      make(map[string]*connTrack),
			activeByRemoteIP: make(map[string]bool),
			ipDB:             make(map[peer.ID]map[string]ipStamp),
			dialStates:       make(map[string]*dialState),
			stopScheduler:    make(chan struct{}),
			dialSem:          make(chan struct{}, maxConcurrentDials),
		}
	})
	return TCPAgentInstance
}

func (a *TCPAgent) Start(ctx context.Context, myPeerID peer.ID) error {
	a.nodeID = GetNodeId()
	a.myPeerID = myPeerID

	ctx2, cancel := context.WithCancel(ctx)
	a.ctx, a.cancel = ctx2, cancel

	ln, err := net.ListenTCP("tcp", &net.TCPAddr{Port: AgentPort})
	if err != nil {
		return fmt.Errorf("failed to start TCP agent listener: %w", err)
	}
	a.listener = ln
	log.Printf("TCP path agent listening on :%d", AgentPort)

	a.schedulerOnce.Do(func() {
		a.schedulerWG.Add(1)
		go a.dialSchedulerLoop()
	})

	go a.acceptLoop()
	return nil
}

func (a *TCPAgent) Stop() error {
	if a.cancel != nil {
		a.cancel()
	}
	close(a.stopScheduler)
	a.schedulerWG.Wait()

	if a.listener != nil {
		_ = a.listener.Close()
	}

	a.edgesMu.Lock()
	for _, ct := range a.activeEdges {
		a.closeConn(ct, "agent_stop")
	}
	a.activeEdges = make(map[string]*connTrack)
	a.edgesMu.Unlock()
	return nil
}

func (a *TCPAgent) UpdateDiscoveredIPs(peerID peer.ID, ips []net.IP) {
	now := time.Now()
	add := make(map[string]ipStamp)
	for _, ip := range ips {
		if ip == nil {
			continue
		}
		if v4 := ip.To4(); v4 != nil {
			add[v4.String()] = ipStamp{seenAt: now, ttl: ttlDiscovered}
		}
	}
	if len(add) == 0 {
		return
	}

	a.ipDBMu.Lock()
	a.ipDB[peerID] = mergeStamps(a.ipDB[peerID], add)
	a.ipDBMu.Unlock()

	a.dialStatesMu.Lock()
	for ipStr := range add {
		key := peerID.String() + "|" + ipStr
		if _, ok := a.dialStates[key]; !ok {
			a.dialStates[key] = &dialState{backoff: 0, nextAttempt: time.Now()}
		}
	}
	a.dialStatesMu.Unlock()
}

func (a *TCPAgent) OnInterfaceChange() {
	now := time.Now()
	a.ifaceGraceUntilMu.Lock()
	a.ifaceGraceUntil = now.Add(tbGraceWindow)
	a.ifaceGraceUntilMu.Unlock()

	a.dialStatesMu.Lock()
	for _, ds := range a.dialStates {
		ds.backoff = 0
		ds.nextAttempt = now
	}
	a.dialStatesMu.Unlock()
}

func (a *TCPAgent) acceptLoop() {
	for {
		conn, err := a.listener.AcceptTCP()
		if err != nil {
			select {
			case <-a.ctx.Done():
				return
			default:
			}
			log.Printf("TCP accept error: %v", err)
			continue
		}
		a.setTCPOptions(conn)
		go a.handleIncoming(conn)
	}
}

func (a *TCPAgent) dialSchedulerLoop() {
	defer a.schedulerWG.Done()
	t := time.NewTicker(scheduleTick)
	defer t.Stop()

	for {
		select {
		case <-a.stopScheduler:
			return
		case <-a.ctx.Done():
			return
		case <-t.C:
			a.expireIPs(false)

			type want struct {
				pid peer.ID
				ip  string
			}
			var wants []want

			a.ipDBMu.RLock()
			for pid, set := range a.ipDB {
				if a.myPeerID.String() <= pid.String() {
					continue
				}
				for ipStr, stamp := range set {
					if time.Since(stamp.seenAt) > stamp.ttl {
						continue
					}
					if a.hasActiveToRemoteIP(ipStr) {
						continue
					}
					wants = append(wants, want{pid: pid, ip: ipStr})
				}
			}
			a.ipDBMu.RUnlock()

			sort.Slice(wants, func(i, j int) bool {
				if wants[i].pid == wants[j].pid {
					return wants[i].ip < wants[j].ip
				}
				return wants[i].pid.String() < wants[j].pid.String()
			})

			now := time.Now()
			for _, w := range wants {
				key := w.pid.String() + "|" + w.ip
				a.dialStatesMu.Lock()
				ds, ok := a.dialStates[key]
				if !ok {
					ds = &dialState{}
					a.dialStates[key] = ds
				}
				if ds.connecting || now.Before(ds.nextAttempt) {
					a.dialStatesMu.Unlock()
					continue
				}
				ds.connecting = true
				a.dialStatesMu.Unlock()

				select {
				case a.dialSem <- struct{}{}:
				case <-a.ctx.Done():
					return
				}

				go func(pid peer.ID, ip string) {
					defer func() {
						<-a.dialSem
						a.dialStatesMu.Lock()
						if ds := a.dialStates[pid.String()+"|"+ip]; ds != nil {
							ds.connecting = false
						}
						a.dialStatesMu.Unlock()
					}()
					a.dialAndMaintain(pid, ip)
				}(w.pid, w.ip)
			}
		}
	}
}

func (a *TCPAgent) dialAndMaintain(pid peer.ID, remoteIP string) {
	remoteAddr := fmt.Sprintf("%s:%d", remoteIP, AgentPort)
	d := net.Dialer{Timeout: dialTimeoutForIP(remoteIP)}
	rawConn, err := d.DialContext(a.ctx, "tcp", remoteAddr)
	if err != nil {
		a.bumpDialBackoff(pid, remoteIP, err)
		return
	}
	tc := rawConn.(*net.TCPConn)
	a.setTCPOptions(tc)

	remoteNodeID, remotePeerID, observedIPv4s, err := a.performHandshake(tc, true)
	if err != nil {
		_ = tc.Close()
		a.bumpDialBackoff(pid, remoteIP, err)
		return
	}

	finalPID := pid
	if remotePeerID != "" {
		if parsed, perr := peer.Decode(remotePeerID); perr == nil {
			finalPID = parsed
		}
	}

	a.updateObservedIPv4s(finalPID, observedIPv4s)

	localIP := tc.LocalAddr().(*net.TCPAddr).IP.String()
	ct := &connTrack{
		tc:     tc,
		dialer: true,
		edge: Edge{
			LocalNodeID:  a.nodeID,
			RemoteNodeID: remoteNodeID,
			LocalIP:      localIP,
			RemoteIP:     remoteIP,
			Proto:        "tcp",
		},
		closed: make(chan struct{}),
	}

	if !a.registerConn(ct) {
		_ = tc.Close()
		a.bumpDialBackoff(finalPID, remoteIP, errors.New("duplicate edge"))
		return
	}

	a.dialStatesMu.Lock()
	if ds := a.dialStates[finalPID.String()+"|"+remoteIP]; ds != nil {
		ds.backoff = 0
		ds.nextAttempt = time.Now().Add(HeartbeatInterval)
	}
	a.dialStatesMu.Unlock()

	a.runHeartbeatLoops(ct)
}

func (a *TCPAgent) handleIncoming(tc *net.TCPConn) {
	remoteNodeID, remotePeerID, observedIPv4s, err := a.performHandshake(tc, false)
	if err != nil {
		_ = tc.Close()
		return
	}
	if remotePeerID != "" {
		if pid, perr := peer.Decode(remotePeerID); perr == nil {
			a.updateObservedIPv4s(pid, observedIPv4s)
		}
	}

	localIP := tc.LocalAddr().(*net.TCPAddr).IP.String()
	remoteIP := tc.RemoteAddr().(*net.TCPAddr).IP.String()

	ct := &connTrack{
		tc:     tc,
		dialer: false,
		edge: Edge{
			LocalNodeID:  a.nodeID,
			RemoteNodeID: remoteNodeID,
			LocalIP:      localIP,
			RemoteIP:     remoteIP,
			Proto:        "tcp",
		},
		closed: make(chan struct{}),
	}

	if !a.registerConn(ct) {
		_ = tc.Close()
		return
	}
	a.runHeartbeatLoops(ct)
}

func (a *TCPAgent) setTCPOptions(tc *net.TCPConn) {
	_ = tc.SetNoDelay(true)
	_ = tc.SetKeepAlive(true)
	_ = tc.SetKeepAlivePeriod(5 * time.Second)
}

func (a *TCPAgent) performHandshake(tc *net.TCPConn, isDialer bool) (remoteNodeID, remotePeerID string, observedIPv4s []string, err error) {
	_ = tc.SetDeadline(time.Now().Add(HandshakeTimeout))
	defer tc.SetDeadline(time.Time{})

	self := HandshakeMessage{
		NodeID:    a.nodeID,
		AgentVer:  "2.2.0",
		PeerID:    a.myPeerID.String(),
		IPv4s:     currentLocalIPv4s(),
		Timestamp: time.Now().UnixNano(),
	}
	var remote HandshakeMessage

	if isDialer {
		if err = json.NewEncoder(tc).Encode(&self); err != nil {
			return "", "", nil, fmt.Errorf("send handshake: %w", err)
		}
		if err = json.NewDecoder(tc).Decode(&remote); err != nil {
			return "", "", nil, fmt.Errorf("read handshake: %w", err)
		}
	} else {
		if err = json.NewDecoder(tc).Decode(&remote); err != nil {
			return "", "", nil, fmt.Errorf("read handshake: %w", err)
		}
		if err = json.NewEncoder(tc).Encode(&self); err != nil {
			return "", "", nil, fmt.Errorf("send handshake: %w", err)
		}
	}

	if remote.NodeID == "" {
		return "", "", nil, errors.New("empty remote node id")
	}
	for _, ip := range remote.IPv4s {
		if ip != "" && strings.Count(ip, ":") == 0 {
			observedIPv4s = append(observedIPv4s, ip)
		}
	}
	return remote.NodeID, remote.PeerID, observedIPv4s, nil
}

func (a *TCPAgent) registerConn(ct *connTrack) bool {
	key := ct.edge.Key()

	a.edgesMu.Lock()
	if _, exists := a.activeEdges[key]; exists {
		a.edgesMu.Unlock()
		return false
	}
	a.activeEdges[key] = ct

	a.activeByRemoteIPMu.Lock()
	a.activeByRemoteIP[ct.edge.RemoteIP] = true
	a.activeByRemoteIPMu.Unlock()
	a.edgesMu.Unlock()

	WriteEdgeCreatedEvent(ct.edge.LocalNodeID, ct.edge.RemoteNodeID, ct.edge.LocalIP, ct.edge.RemoteIP, ct.edge.Proto)
	return true
}

func (a *TCPAgent) hasActiveToRemoteIP(remoteIP string) bool {
	a.activeByRemoteIPMu.RLock()
	ok := a.activeByRemoteIP[remoteIP]
	a.activeByRemoteIPMu.RUnlock()
	return ok
}

func (a *TCPAgent) recalcRemoteIPActive(remoteIP string) {
	a.edgesMu.RLock()
	active := false
	for _, ct := range a.activeEdges {
		if ct.edge.RemoteIP == remoteIP {
			active = true
			break
		}
	}
	a.edgesMu.RUnlock()

	a.activeByRemoteIPMu.Lock()
	if active {
		a.activeByRemoteIP[remoteIP] = true
	} else {
		delete(a.activeByRemoteIP, remoteIP)
	}
	a.activeByRemoteIPMu.Unlock()
}

func (a *TCPAgent) closeConn(ct *connTrack, _ string) {
	ct.closeMx.Do(func() {
		_ = ct.tc.Close()
		key := ct.edge.Key()

		a.edgesMu.Lock()
		delete(a.activeEdges, key)
		a.edgesMu.Unlock()

		a.recalcRemoteIPActive(ct.edge.RemoteIP)

		WriteEdgeDeletedEvent(ct.edge.LocalNodeID, ct.edge.RemoteNodeID, ct.edge.LocalIP, ct.edge.RemoteIP, ct.edge.Proto)
	})
}

func (a *TCPAgent) runHeartbeatLoops(ct *connTrack) {
	go func() {
		r := bufio.NewReader(ct.tc)
		for {
			_ = ct.tc.SetReadDeadline(time.Now().Add(HeartbeatReadGrace))
			if _, err := r.ReadByte(); err != nil {
				a.closeConn(ct, "read_error")
				return
			}
		}
	}()

	go func() {
		t := time.NewTicker(HeartbeatInterval)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				_ = ct.tc.SetWriteDeadline(time.Now().Add(HeartbeatWriteGrace))
				if _, err := ct.tc.Write([]byte{0x01}); err != nil {
					a.closeConn(ct, "write_error")
					return
				}
			case <-a.ctx.Done():
				a.closeConn(ct, "agent_ctx_done")
				return
			}
		}
	}()
}

func (a *TCPAgent) bumpDialBackoff(pid peer.ID, ip string, err error) {
	key := pid.String() + "|" + ip
	a.dialStatesMu.Lock()
	ds, ok := a.dialStates[key]
	if !ok {
		ds = &dialState{}
		a.dialStates[key] = ds
	}
	if ds.backoff == 0 {
		ds.backoff = initialBackoff
	} else {
		ds.backoff *= 2
		if ds.backoff > maxBackoff {
			ds.backoff = maxBackoff
		}
	}
	ds.nextAttempt = time.Now().Add(ds.backoff)
	a.dialStatesMu.Unlock()

	log.Printf("dial %s@%s failed: %v; next in %s", pid, ip, err, ds.backoff)
}

func mergeStamps(dst map[string]ipStamp, src map[string]ipStamp) map[string]ipStamp {
	if dst == nil {
		dst = make(map[string]ipStamp, len(src))
	}
	for ip, s := range src {
		prev, ok := dst[ip]
		if !ok || s.seenAt.After(prev.seenAt) {
			dst[ip] = s
		}
	}
	return dst
}

func (a *TCPAgent) updateObservedIPv4s(pid peer.ID, ipv4s []string) {
	if len(ipv4s) == 0 {
		return
	}
	now := time.Now()
	add := make(map[string]ipStamp, len(ipv4s))
	for _, ip := range ipv4s {
		if ip != "" && strings.Count(ip, ":") == 0 {
			add[ip] = ipStamp{seenAt: now, ttl: ttlObserved}
		}
	}

	a.ipDBMu.Lock()
	a.ipDB[pid] = mergeStamps(a.ipDB[pid], add)
	a.ipDBMu.Unlock()

	a.dialStatesMu.Lock()
	for ip := range add {
		key := pid.String() + "|" + ip
		if _, ok := a.dialStates[key]; !ok {
			a.dialStates[key] = &dialState{backoff: 0, nextAttempt: time.Now()}
		}
	}
	a.dialStatesMu.Unlock()
}

func (a *TCPAgent) expireIPs(_ bool) {
	a.ifaceGraceUntilMu.RLock()
	graceUntil := a.ifaceGraceUntil
	a.ifaceGraceUntilMu.RUnlock()
	if time.Now().Before(graceUntil) {
		return
	}

	now := time.Now()
	a.ipDBMu.Lock()
	for pid, set := range a.ipDB {
		for ip, stamp := range set {
			if now.Sub(stamp.seenAt) > stamp.ttl {
				delete(set, ip)

				a.dialStatesMu.Lock()
				delete(a.dialStates, pid.String()+"|"+ip)
				a.dialStatesMu.Unlock()

				log.Printf("TCP agent: expired ip %s for %s", ip, pid)
			}
		}
		if len(set) == 0 {
			delete(a.ipDB, pid)
		}
	}
	a.ipDBMu.Unlock()
}

func currentLocalIPv4s() []string {
	var out []string
	ifaces, err := net.Interfaces()
	if err != nil {
		return out
	}
	for _, ifi := range ifaces {
		if ifi.Flags&net.FlagUp == 0 {
			continue
		}
		addrs, _ := ifi.Addrs()
		for _, a := range addrs {
			if ipnet, ok := a.(*net.IPNet); ok && ipnet.IP != nil {
				if v4 := ipnet.IP.To4(); v4 != nil && !v4.IsLoopback() && !v4.IsUnspecified() {
					out = append(out, v4.String())
				}
			}
		}
	}
	sort.Strings(out)
	return dedupeStrings(out)
}

func dedupeStrings(xs []string) []string {
	if len(xs) < 2 {
		return xs
	}
	out := xs[:0]
	last := ""
	for _, s := range xs {
		if s == last {
			continue
		}
		out = append(out, s)
		last = s
	}
	return out
}

func dialTimeoutForIP(ip string) time.Duration {
	if strings.HasPrefix(ip, "169.254.") {
		return dialTimeoutLinkLocal
	}
	return dialTimeoutDefault
}

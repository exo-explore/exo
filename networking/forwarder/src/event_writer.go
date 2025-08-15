package forwarder

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"sync"

	"github.com/google/uuid"
	"github.com/libp2p/go-libp2p/core/network"
	_ "github.com/mattn/go-sqlite3"
	"github.com/multiformats/go-multiaddr"
)

var (
	eventsDBPath string
	eventsDB     *sql.DB
	eventsDBMu   sync.Mutex
)

func SetEventsDBPath(path string) {
	eventsDBMu.Lock()
	defer eventsDBMu.Unlock()
	eventsDBPath = path
}

const (
	EventTypeTopologyEdgeCreated = "TopologyEdgeCreated"
	EventTypeTopologyEdgeDeleted = "TopologyEdgeDeleted"
)

type ConnectionProfile struct {
	Throughput float64 `json:"throughput"`
	Latency    float64 `json:"latency"`
	Jitter     float64 `json:"jitter"`
}

type Multiaddr struct {
	Address     string `json:"address"`
	IPv4Address string `json:"ipv4_address,omitempty"`
	IPv6Address string `json:"ipv6_address,omitempty"`
	Port        int    `json:"port,omitempty"`
	Transport   string `json:"transport,omitempty"` // tcp/quic/ws/etc
}

type Connection struct {
	LocalNodeID       string             `json:"local_node_id"`
	SendBackNodeID    string             `json:"send_back_node_id"`
	LocalMultiaddr    Multiaddr          `json:"local_multiaddr"`
	SendBackMultiaddr Multiaddr          `json:"send_back_multiaddr"`
	ConnectionProfile *ConnectionProfile `json:"connection_profile"`
}

type TopologyEdgeCreated struct {
	EventType string     `json:"event_type"`
	EventID   string     `json:"event_id"`
	Edge      Connection `json:"edge"`
}

type TopologyEdgeDeleted struct {
	EventType string     `json:"event_type"`
	EventID   string     `json:"event_id"`
	Edge      Connection `json:"edge"`
}

func initEventsDB() error {
	eventsDBMu.Lock()
	defer eventsDBMu.Unlock()
	if eventsDB != nil {
		return nil
	}
	if eventsDBPath == "" {
		return nil
	}
	db, err := sql.Open("sqlite3", eventsDBPath)
	if err != nil {
		return fmt.Errorf("failed to open events database: %w", err)
	}
	eventsDB = db

	const schema = `
	CREATE TABLE IF NOT EXISTS events (
		rowid INTEGER PRIMARY KEY AUTOINCREMENT,
		origin TEXT NOT NULL,
		event_type TEXT NOT NULL,
		event_id TEXT NOT NULL,
		event_data TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
	);
	CREATE INDEX IF NOT EXISTS idx_events_origin ON events(origin);
	CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
	CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
	`
	if _, err := eventsDB.Exec(schema); err != nil {
		eventsDB.Close()
		eventsDB = nil
		return fmt.Errorf("failed to create events table: %w", err)
	}
	return nil
}

func writeEvent(eventType string, eventData interface{}) error {
	if eventsDB == nil {
		if err := initEventsDB(); err != nil {
			return err
		}
		if eventsDB == nil {
			return nil
		}
	}
	jsonData, err := json.Marshal(eventData)
	if err != nil {
		return fmt.Errorf("failed to marshal event data: %w", err)
	}
	var eventID string
	switch e := eventData.(type) {
	case *TopologyEdgeCreated:
		eventID = e.EventID
	case *TopologyEdgeDeleted:
		eventID = e.EventID
	default:
		eventID = uuid.New().String()
	}
	const insert = `INSERT INTO events (origin, event_type, event_id, event_data) VALUES (?, ?, ?, ?)`
	_, err = eventsDB.Exec(insert, GetNodeId(), eventType, eventID, string(jsonData))
	return err
}

var WriteEdgeCreatedEvent = func(localNodeID, remoteNodeID, localIP, remoteIP, proto string) {
	event := &TopologyEdgeCreated{
		EventType: EventTypeTopologyEdgeCreated,
		EventID:   uuid.New().String(),
		Edge: Connection{
			LocalNodeID:    localNodeID,
			SendBackNodeID: remoteNodeID,
			LocalMultiaddr: Multiaddr{
				Address:     fmt.Sprintf("/ip4/%s/tcp/7847", localIP),
				IPv4Address: localIP,
				Port:        7847,
				Transport:   proto,
			},
			SendBackMultiaddr: Multiaddr{
				Address:     fmt.Sprintf("/ip4/%s/tcp/7847", remoteIP),
				IPv4Address: remoteIP,
				Port:        7847,
				Transport:   proto,
			},
			ConnectionProfile: nil,
		},
	}
	if err := writeEvent(EventTypeTopologyEdgeCreated, event); err != nil {
		log.Printf("Failed to write edge created event: %v", err)
	} else {
		log.Printf("Wrote TCP edge created event: %s -> %s (%s:%s)", localNodeID, remoteNodeID, remoteIP, proto)
	}
}

var WriteEdgeDeletedEvent = func(localNodeID, remoteNodeID, localIP, remoteIP, proto string) {
	event := &TopologyEdgeDeleted{
		EventType: EventTypeTopologyEdgeDeleted,
		EventID:   uuid.New().String(),
		Edge: Connection{
			LocalNodeID:    localNodeID,
			SendBackNodeID: remoteNodeID,
			LocalMultiaddr: Multiaddr{
				Address:     fmt.Sprintf("/ip4/%s/tcp/7847", localIP),
				IPv4Address: localIP,
				Port:        7847,
				Transport:   proto,
			},
			SendBackMultiaddr: Multiaddr{
				Address:     fmt.Sprintf("/ip4/%s/tcp/7847", remoteIP),
				IPv4Address: remoteIP,
				Port:        7847,
				Transport:   proto,
			},
			ConnectionProfile: nil,
		},
	}
	if err := writeEvent(EventTypeTopologyEdgeDeleted, event); err != nil {
		log.Printf("Failed to write edge deleted event: %v", err)
	} else {
		log.Printf("Wrote TCP edge deleted event: %s -> %s (%s:%s)", localNodeID, remoteNodeID, remoteIP, proto)
	}
}

type NotifeeHandler struct{}

func (n *NotifeeHandler) Listen(net network.Network, ma multiaddr.Multiaddr)      {}
func (n *NotifeeHandler) ListenClose(net network.Network, ma multiaddr.Multiaddr) {}
func (n *NotifeeHandler) Connected(netw network.Network, conn network.Conn) {
	pid := conn.RemotePeer()
	rawR := conn.RemoteMultiaddr()

	if node != nil && node.ConnManager() != nil {
		node.ConnManager().Protect(pid, "multipath-"+hostTransportKey(rawR))
	}

	if ipStr, err := rawR.ValueForProtocol(multiaddr.P_IP4); err == nil && ipStr != "" {
		if ip := net.ParseIP(ipStr); ip != nil {
			GetTCPAgent().UpdateDiscoveredIPs(pid, []net.IP{ip})
		}
	}
}
func (n *NotifeeHandler) Disconnected(net network.Network, conn network.Conn) {
	pid := conn.RemotePeer()
	rawR := conn.RemoteMultiaddr()

	if node != nil && node.ConnManager() != nil {
		tag := "multipath-" + hostTransportKey(rawR)
		node.ConnManager().Unprotect(pid, tag)
	}
}
func (n *NotifeeHandler) OpenedStream(net network.Network, str network.Stream) {}
func (n *NotifeeHandler) ClosedStream(net network.Network, str network.Stream) {}

func GetNotifee() network.Notifiee { return &NotifeeHandler{} }

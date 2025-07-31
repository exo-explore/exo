package forwarder

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
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

// SetEventsDBPath sets the path to the events database
func SetEventsDBPath(path string) {
	eventsDBMu.Lock()
	defer eventsDBMu.Unlock()
	eventsDBPath = path
}

// Event types matching Python's _EventType enum
const (
	EventTypeTopologyEdgeCreated = "TopologyEdgeCreated"
	EventTypeTopologyEdgeDeleted = "TopologyEdgeDeleted"
)

// ConnectionProfile matches Python's ConnectionProfile (optional)
type ConnectionProfile struct {
	Throughput float64 `json:"throughput"`
	Latency    float64 `json:"latency"`
	Jitter     float64 `json:"jitter"`
}

// Multiaddr matches Python's Multiaddr structure
type Multiaddr struct {
	Address     string `json:"address"`
	IPv4Address string `json:"ipv4_address,omitempty"`
	Port        int    `json:"port,omitempty"`
}

// Connection matches Python's Connection model
type Connection struct {
	LocalNodeID         string             `json:"local_node_id"`
	SendBackNodeID      string             `json:"send_back_node_id"`
	LocalMultiaddr      Multiaddr          `json:"local_multiaddr"`
	SendBackMultiaddr   Multiaddr          `json:"send_back_multiaddr"`
	ConnectionProfile   *ConnectionProfile `json:"connection_profile"`
}

// TopologyEdgeCreated matches Python's TopologyEdgeCreated event
type TopologyEdgeCreated struct {
	EventType string     `json:"event_type"`
	EventID   string     `json:"event_id"`
	Edge      Connection `json:"edge"`
}

// TopologyEdgeDeleted matches Python's TopologyEdgeDeleted event
type TopologyEdgeDeleted struct {
	EventType string     `json:"event_type"`
	EventID   string     `json:"event_id"`
	Edge      Connection `json:"edge"`
}

// initEventsDB initializes the events database connection
func initEventsDB() error {
	eventsDBMu.Lock()
	defer eventsDBMu.Unlock()

	if eventsDB != nil {
		return nil // Already initialized
	}

	if eventsDBPath == "" {
		return nil // No events DB configured
	}

	var err error
	eventsDB, err = sql.Open("sqlite3", eventsDBPath)
	if err != nil {
		return fmt.Errorf("failed to open events database: %w", err)
	}

	// Create table if it doesn't exist (matching Python's schema)
	createTableSQL := `
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
	_, err = eventsDB.Exec(createTableSQL)
	if err != nil {
		eventsDB.Close()
		eventsDB = nil
		return fmt.Errorf("failed to create events table: %w", err)
	}

	return nil
}

// writeEvent writes an event to the database
func writeEvent(eventType string, eventData interface{}) error {
	if eventsDB == nil {
		if err := initEventsDB(); err != nil {
			return err
		}
		if eventsDB == nil {
			return nil // No events DB configured
		}
	}

	// Serialize event data to JSON
	jsonData, err := json.Marshal(eventData)
	if err != nil {
		return fmt.Errorf("failed to marshal event data: %w", err)
	}

	// Extract event ID from the event data
	var eventID string
	switch e := eventData.(type) {
	case *TopologyEdgeCreated:
		eventID = e.EventID
	case *TopologyEdgeDeleted:
		eventID = e.EventID
	default:
		eventID = uuid.New().String()
	}

	// Insert event into database
	insertSQL := `INSERT INTO events (origin, event_type, event_id, event_data) VALUES (?, ?, ?, ?)`
	_, err = eventsDB.Exec(insertSQL, GetNodeId(), eventType, eventID, string(jsonData))
	if err != nil {
		return fmt.Errorf("failed to insert event: %w", err)
	}

	return nil
}

// NotifeeHandler implements the libp2p network.Notifiee interface
type NotifeeHandler struct{}

// Listen is called when network starts listening on an addr
func (n *NotifeeHandler) Listen(net network.Network, ma multiaddr.Multiaddr) {}

// ListenClose is called when network stops listening on an addr
func (n *NotifeeHandler) ListenClose(net network.Network, ma multiaddr.Multiaddr) {}

// Connected is called when a connection is opened
func (n *NotifeeHandler) Connected(net network.Network, conn network.Conn) {
	remotePeer := conn.RemotePeer()
	localAddr := conn.LocalMultiaddr()
	remoteAddr := conn.RemoteMultiaddr()

	// Get the actual node IDs (not peer IDs)
	localNodeID := GetNodeId()
	
	// For remote node, we need to extract from peer ID or use a mapping
	// For now, we'll use the peer ID as a placeholder
	// TODO: Implement proper node ID mapping/discovery
	remoteNodeID := remotePeer.String()

	// Create connection event
	event := &TopologyEdgeCreated{
		EventType: EventTypeTopologyEdgeCreated,
		EventID:   uuid.New().String(),
		Edge: Connection{
			LocalNodeID:       localNodeID,
			SendBackNodeID:    remoteNodeID,
			LocalMultiaddr:    parseMultiaddr(localAddr),
			SendBackMultiaddr: parseMultiaddr(remoteAddr),
			ConnectionProfile: nil, // TODO: Add connection profiling if needed
		},
	}

	// Write event to database
	if err := writeEvent(EventTypeTopologyEdgeCreated, event); err != nil {
		log.Printf("Failed to write edge created event: %v", err)
	} else {
		log.Printf("Wrote edge created event: %s -> %s", localNodeID, remoteNodeID)
	}
}

// Disconnected is called when a connection is closed
func (n *NotifeeHandler) Disconnected(net network.Network, conn network.Conn) {
	remotePeer := conn.RemotePeer()
	localAddr := conn.LocalMultiaddr()
	remoteAddr := conn.RemoteMultiaddr()

	// Get the actual node IDs (not peer IDs)
	localNodeID := GetNodeId()
	remoteNodeID := remotePeer.String() // TODO: Implement proper node ID mapping

	// Create disconnection event
	event := &TopologyEdgeDeleted{
		EventType: EventTypeTopologyEdgeDeleted,
		EventID:   uuid.New().String(),
		Edge: Connection{
			LocalNodeID:       localNodeID,
			SendBackNodeID:    remoteNodeID,
			LocalMultiaddr:    parseMultiaddr(localAddr),
			SendBackMultiaddr: parseMultiaddr(remoteAddr),
			ConnectionProfile: nil,
		},
	}

	// Write event to database
	if err := writeEvent(EventTypeTopologyEdgeDeleted, event); err != nil {
		log.Printf("Failed to write edge deleted event: %v", err)
	} else {
		log.Printf("Wrote edge deleted event: %s -> %s", localNodeID, remoteNodeID)
	}
}

// OpenedStream is called when a stream is opened
func (n *NotifeeHandler) OpenedStream(net network.Network, str network.Stream) {}

// ClosedStream is called when a stream is closed
func (n *NotifeeHandler) ClosedStream(net network.Network, str network.Stream) {}

// parseMultiaddr converts a libp2p multiaddr to our Multiaddr struct
func parseMultiaddr(ma multiaddr.Multiaddr) Multiaddr {
	result := Multiaddr{
		Address: ma.String(),
	}
	
	// Extract IPv4 address if present
	if ipStr, err := ma.ValueForProtocol(multiaddr.P_IP4); err == nil {
		result.IPv4Address = ipStr
	}
	
	// Extract port if present
	if portStr, err := ma.ValueForProtocol(multiaddr.P_TCP); err == nil {
		if port, err := strconv.Atoi(portStr); err == nil {
			result.Port = port
		}
	}
	
	return result
}

// GetNotifee returns a singleton instance of the notifee handler
func GetNotifee() network.Notifiee {
	return &NotifeeHandler{}
}
package forwarder

import (
	"os"
	"sync"

	"github.com/google/uuid"
)

var (
	generatedNodeID string
	nodeIDOnce      sync.Once
)

func GetNodeId() string {
	if id := os.Getenv("FORWARDER_NODE_ID"); id != "" {
		return id
	}

	nodeIDOnce.Do(func() {
		generatedNodeID = uuid.New().String()
	})

	return generatedNodeID
}

func SetNodeId(id string) {
	os.Setenv("FORWARDER_NODE_ID", id)
}

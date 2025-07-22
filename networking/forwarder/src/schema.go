package forwarder

import (
	"context"
	"time"
)

type SourceKey struct {
	SourceNodeId string `json:"source_node_id"`
	SourcePath   string `json:"source_path"` // db:table
}

type TrackingData struct {
	SourceKey
	SourceRowID     int64     `json:"source_row_id"`
	SourceTimestamp time.Time `json:"source_timestamp"`
}
type RecordData struct {
	TrackingData
	Data map[string]interface{} `json:"data"`
}

type BatchRecord struct {
	Records []RecordData `json:"records"`
}

type ForwardingPair struct {
	source connection
	sink   connection
}

type connection interface {
	tail(handler func(record RecordData) error)
	write(record RecordData) error
	close() error
	getType() string
}

type LibP2PConnection interface {
	connection
	tailResend(handler func(record ResendRequest) error)
	writeResend(record ResendRequest) error
}

type SQLiteConnection interface {
	connection
	getLatestRowIds() (map[SourceKey]int64, error)
	readRange(start, end int64) ([]RecordData, error)
	getNodeId() string
	getTablePath() string
}

type GapRange struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"`
}
type ResendRequest struct {
	SourceNodeID string     `json:"source_node_id"`
	SourcePath   string     `json:"source_path"`
	Gaps         []GapRange `json:"gaps"`
}

type stateStoreInterface interface {
	onRecord(record RecordData)
	getWriteableMessages() []RecordData
	getResendRequests() []ResendRequest
	getCurrentGaps() map[SourceKey][]gap
}

type Forwarder interface {
	Start(ctx context.Context) error
}

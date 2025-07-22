package forwarder

import (
	"context"
	"fmt"
	"log"
	"time"
)

type libP2PToSqliteForwarder struct {
	source      LibP2PConnection
	sink        SQLiteConnection
	recordStore stateStoreInterface
}

func newLibP2PToSqliteForwarder(source LibP2PConnection, sink SQLiteConnection) (*libP2PToSqliteForwarder, error) {
	latestRowIds, err := sink.getLatestRowIds()
	if err != nil {
		return nil, fmt.Errorf("failed to get latest row IDs: %w", err)
	}
	return &libP2PToSqliteForwarder{
		source:      source,
		sink:        sink,
		recordStore: newStateStore(latestRowIds),
	}, nil
}

func (f *libP2PToSqliteForwarder) Start(ctx context.Context) error {
	f.source.tail(func(record RecordData) error {
		f.recordStore.onRecord(record)
		return nil
	})

	go func() {
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				msgs := f.recordStore.getWriteableMessages()
				for _, msg := range msgs {
					if err := f.sink.write(msg); err != nil {
						log.Printf("Error writing to sink: %v", err)
					}
				}
			}
		}
	}()

	// Resend handler with less frequent checks
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond) // Less frequent than before
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				reqs := f.recordStore.getResendRequests()
				for _, req := range reqs {
					if err := f.source.writeResend(req); err != nil {
						log.Printf("Error writing resend request: %v", err)
					}
				}
			}
		}
	}()

	return nil
}

type sqliteToLibP2PForwarder struct {
	source SQLiteConnection
	sink   LibP2PConnection
}

func newSqliteToLibP2PForwarder(source SQLiteConnection, sink LibP2PConnection) (*sqliteToLibP2PForwarder, error) {
	return &sqliteToLibP2PForwarder{
		source: source,
		sink:   sink,
	}, nil
}

func (f *sqliteToLibP2PForwarder) Start(ctx context.Context) error {
	// Handle resend requests
	f.sink.tailResend(func(req ResendRequest) error {
		if req.SourceNodeID != f.source.getNodeId() {
			return nil
		}
		if req.SourcePath != f.source.getTablePath() {
			return nil
		}

		// Process resends in a separate goroutine to not block
		go func() {
			for _, gap := range req.Gaps {
				records, err := f.source.readRange(gap.Start, gap.End)
				if err != nil {
					log.Printf("Error getting records for resend: %v", err)
					continue
				}
				// Send resend records - libp2p connector will handle batching
				for _, rec := range records {
					if err := f.sink.write(rec); err != nil {
						log.Printf("Error writing resend record: %v", err)
					}
				}
			}
		}()
		return nil
	})

	// Tail new records - libp2p connector handles async batching internally
	f.source.tail(func(record RecordData) error {
		if err := f.sink.write(record); err != nil {
			log.Printf("Error writing record: %v", err)
		}
		return nil
	})

	return nil
}

func NewForwarder(forwardingPair ForwardingPair) (Forwarder, error) {
	if forwardingPair.source.getType() == "libp2p" && forwardingPair.sink.getType() == "sqlite" {
		return newLibP2PToSqliteForwarder(forwardingPair.source.(*libP2PConnector), forwardingPair.sink.(*sqliteConnector))
	} else if forwardingPair.source.getType() == "sqlite" && forwardingPair.sink.getType() == "libp2p" {
		return newSqliteToLibP2PForwarder(forwardingPair.source.(*sqliteConnector), forwardingPair.sink.(*libP2PConnector))
	}
	return nil, fmt.Errorf("unsupported forwarding pair: %v", forwardingPair)
}

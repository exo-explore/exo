package forwarder

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLibP2PConnectorCreation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	conn := newLibP2PConnector("test_topic", ctx, cancel)
	assert.NotNil(t, conn)
	assert.Equal(t, "test_topic", conn.topic)
	assert.NotNil(t, conn.top)
	assert.Nil(t, conn.sub)
	err := conn.close()
	assert.NoError(t, err)
}

func TestLibP2PConnectorGetType(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	conn := newLibP2PConnector("test_topic", ctx, cancel)
	assert.Equal(t, "libp2p", conn.getType())
	err := conn.close()
	assert.NoError(t, err)
}

func TestLibP2PConnectorTailAndWriteSameTopic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	conn := newLibP2PConnector("test_topic_tail_and_write", ctx, cancel)

	received := make(chan RecordData, 1)
	errChan := make(chan error, 1)

	conn.tail(func(rec RecordData) error {
		received <- rec
		return nil
	})

	time.Sleep(100 * time.Millisecond)

	rec := RecordData{
		TrackingData: TrackingData{
			SourceKey: SourceKey{
				SourceNodeId: "test_node_id",
				SourcePath:   "test_path",
			},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: map[string]interface{}{"test_key": "test_value"},
	}
	err := conn.write(rec)
	require.NoError(t, err)

	select {
	case got := <-received:
		assert.Equal(t, rec.SourceKey.SourceNodeId, got.SourceKey.SourceNodeId)
		assert.Equal(t, rec.SourceKey.SourcePath, got.SourceKey.SourcePath)
		assert.Equal(t, rec.SourceRowID, got.SourceRowID)
		assert.Equal(t, rec.Data, got.Data)
		assert.WithinDuration(t, rec.SourceTimestamp, got.SourceTimestamp, time.Second)
	case err := <-errChan:
		t.Fatalf("handler error: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for message")
	}

	err = conn.close()
	assert.NoError(t, err)
}

func TestLibP2PConnectorTailAndWriteDifferentTopic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	conn1 := newLibP2PConnector("test_topic_tail_and_write1", ctx, cancel)
	conn2 := newLibP2PConnector("test_topic_tail_and_write2", ctx, cancel)

	received := make(chan RecordData, 1)

	conn1.tail(func(rec RecordData) error {
		received <- rec
		return nil
	})

	time.Sleep(100 * time.Millisecond)

	rec := RecordData{
		TrackingData: TrackingData{
			SourceKey: SourceKey{
				SourceNodeId: "test_node_id",
				SourcePath:   "test_path",
			},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: map[string]interface{}{"test_key": "test_value"},
	}
	err := conn2.write(rec)
	require.NoError(t, err)

	select {
	case <-received:
		t.Fatal("should not receive message from different topic")
	case <-time.After(500 * time.Millisecond):
	}

	err = conn1.close()
	assert.NoError(t, err)
	err = conn2.close()
	assert.NoError(t, err)
}

func TestLibP2PConnectorMultipleSubscriptionsSameTopic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	conn1 := newLibP2PConnector("test_topic_multiple_subscriptions", ctx, cancel)
	conn2 := newLibP2PConnector("test_topic_multiple_subscriptions", ctx, cancel)

	received1 := make(chan RecordData, 1)
	received2 := make(chan RecordData, 1)

	conn1.tail(func(rec RecordData) error {
		received1 <- rec
		return nil
	})
	conn2.tail(func(rec RecordData) error {
		received2 <- rec
		return nil
	})

	time.Sleep(100 * time.Millisecond)

	rec := RecordData{
		TrackingData: TrackingData{
			SourceKey: SourceKey{
				SourceNodeId: "test_node_id",
				SourcePath:   "test_path",
			},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: map[string]interface{}{"test_key": "test_value"},
	}
	err := conn1.write(rec)
	require.NoError(t, err)

	select {
	case got := <-received1:
		assert.Equal(t, rec.SourceKey.SourceNodeId, got.SourceKey.SourceNodeId)
		assert.Equal(t, rec.SourceKey.SourcePath, got.SourceKey.SourcePath)
		assert.Equal(t, rec.SourceRowID, got.SourceRowID)
		assert.Equal(t, rec.Data, got.Data)
		assert.WithinDuration(t, rec.SourceTimestamp, got.SourceTimestamp, time.Second)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for message on conn1")
	}

	select {
	case got := <-received2:
		assert.Equal(t, rec.SourceKey.SourceNodeId, got.SourceKey.SourceNodeId)
		assert.Equal(t, rec.SourceKey.SourcePath, got.SourceKey.SourcePath)
		assert.Equal(t, rec.SourceRowID, got.SourceRowID)
		assert.Equal(t, rec.Data, got.Data)
		assert.WithinDuration(t, rec.SourceTimestamp, got.SourceTimestamp, time.Second)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for message on conn2")
	}

	err = conn1.close()
	assert.NoError(t, err)
	err = conn2.close()
	assert.NoError(t, err)
}

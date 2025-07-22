package forwarder

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"
)

type mockLibP2PConnector struct {
	tailHandler       func(RecordData) error
	tailResendHandler func(ResendRequest) error
	writtenRecords    []RecordData
	writeErr          error
	resendRequests    []ResendRequest
	writeResendErr    error
}

func (m *mockLibP2PConnector) tail(handler func(record RecordData) error) {
	m.tailHandler = handler
}

func (m *mockLibP2PConnector) tailResend(handler func(req ResendRequest) error) {
	m.tailResendHandler = handler
}

func (m *mockLibP2PConnector) write(record RecordData) error {
	m.writtenRecords = append(m.writtenRecords, record)
	return m.writeErr
}

func (m *mockLibP2PConnector) writeResend(req ResendRequest) error {
	m.resendRequests = append(m.resendRequests, req)
	return m.writeResendErr
}

func (m *mockLibP2PConnector) close() error {
	return nil
}

func (m *mockLibP2PConnector) getType() string {
	return "libp2p"
}

func (m *mockLibP2PConnector) SendRecord(record RecordData) error {
	if m.tailHandler == nil {
		return fmt.Errorf("no tail handler registered")
	}
	return m.tailHandler(record)
}

func (m *mockLibP2PConnector) SendResend(req ResendRequest) error {
	if m.tailResendHandler == nil {
		return fmt.Errorf("no tailResend handler registered")
	}
	return m.tailResendHandler(req)
}

type mockSqliteConnector struct {
	getLatestRowIdsRet map[SourceKey]int64
	getLatestRowIdsErr error
	writtenRecords     []RecordData
	writeErr           error
	readRangeCalls     []struct{ start, end int64 }
	readRangeRet       []RecordData
	readRangeErr       error
	nodeId             string
	tablePath          string
	tailHandler        func(RecordData) error
}

func (m *mockSqliteConnector) getLatestRowIds() (map[SourceKey]int64, error) {
	return m.getLatestRowIdsRet, m.getLatestRowIdsErr
}

func (m *mockSqliteConnector) write(record RecordData) error {
	m.writtenRecords = append(m.writtenRecords, record)
	return m.writeErr
}

func (m *mockSqliteConnector) readRange(start, end int64) ([]RecordData, error) {
	m.readRangeCalls = append(m.readRangeCalls, struct{ start, end int64 }{start, end})
	return m.readRangeRet, m.readRangeErr
}

func (m *mockSqliteConnector) tail(handler func(record RecordData) error) {
	m.tailHandler = handler
}

func (m *mockSqliteConnector) close() error {
	return nil
}

func (m *mockSqliteConnector) getType() string {
	return "sqlite"
}

func (m *mockSqliteConnector) SendRecord(record RecordData) error {
	if m.tailHandler == nil {
		return fmt.Errorf("no tail handler registered")
	}
	return m.tailHandler(record)
}

func (m *mockSqliteConnector) getNodeId() string {
	return m.nodeId
}

func (m *mockSqliteConnector) getTablePath() string {
	return m.tablePath
}

func TestNewLibP2PToSqliteForwarder(t *testing.T) {
	source := &mockLibP2PConnector{}
	sink := &mockSqliteConnector{
		getLatestRowIdsRet: map[SourceKey]int64{},
	}
	f, err := newLibP2PToSqliteForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f == nil {
		t.Fatal("expected non-nil forwarder")
	}
}

func TestLibP2PToSqliteForwarder_Start_InOrderRecords(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &mockLibP2PConnector{}
	sink := &mockSqliteConnector{
		getLatestRowIdsRet: map[SourceKey]int64{},
	}

	f, err := newLibP2PToSqliteForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	key := SourceKey{SourceNodeId: "node1", SourcePath: "path1"}

	rec1 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 1}}
	source.SendRecord(rec1)

	time.Sleep(500 * time.Millisecond)

	if len(sink.writtenRecords) != 1 {
		t.Fatalf("expected 1 written record, got %d", len(sink.writtenRecords))
	}
	if !reflect.DeepEqual(sink.writtenRecords[0], rec1) {
		t.Fatal("written record mismatch")
	}

	rec2 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 2}}
	source.SendRecord(rec2)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 2 {
		t.Fatalf("expected 2 written records, got %d", len(sink.writtenRecords))
	}
	if !reflect.DeepEqual(sink.writtenRecords[1], rec2) {
		t.Fatal("written record mismatch")
	}
}

func TestLibP2PToSqliteForwarder_Start_OutOfOrderRecords(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &mockLibP2PConnector{}
	sink := &mockSqliteConnector{
		getLatestRowIdsRet: map[SourceKey]int64{},
	}

	f, err := newLibP2PToSqliteForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	key := SourceKey{SourceNodeId: "node1", SourcePath: "path1"}

	rec1 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 1}}
	source.SendRecord(rec1)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 1 {
		t.Fatalf("expected 1 written record, got %d", len(sink.writtenRecords))
	}

	rec3 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 3}}
	source.SendRecord(rec3)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 1 {
		t.Fatalf("expected still 1 written record, got %d", len(sink.writtenRecords))
	}

	time.Sleep(5500 * time.Millisecond) // Wait for resend ticker

	if len(source.resendRequests) != 1 {
		t.Fatalf("expected 1 resend request, got %d", len(source.resendRequests))
	}

	req := source.resendRequests[0]
	if req.SourceNodeID != "node1" || req.SourcePath != "path1" {
		t.Fatal("resend request mismatch")
	}
	if len(req.Gaps) != 1 || req.Gaps[0].Start != 2 || req.Gaps[0].End != 2 {
		t.Fatal("gap mismatch")
	}

	rec2 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 2}}
	source.SendRecord(rec2)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 3 {
		t.Fatalf("expected 3 written records, got %d", len(sink.writtenRecords))
	}
	// Check order: rec1, rec2, rec3
	if !reflect.DeepEqual(sink.writtenRecords[1], rec2) || !reflect.DeepEqual(sink.writtenRecords[2], rec3) {
		t.Fatal("written records order mismatch")
	}
}

func TestLibP2PToSqliteForwarder_Start_MultipleSources(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &mockLibP2PConnector{}
	sink := &mockSqliteConnector{
		getLatestRowIdsRet: map[SourceKey]int64{},
	}

	f, err := newLibP2PToSqliteForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	key1 := SourceKey{SourceNodeId: "node1", SourcePath: "path1"}
	key2 := SourceKey{SourceNodeId: "node2", SourcePath: "path2"}

	rec1_1 := RecordData{TrackingData: TrackingData{SourceKey: key1, SourceRowID: 1}}
	source.SendRecord(rec1_1)

	rec2_1 := RecordData{TrackingData: TrackingData{SourceKey: key2, SourceRowID: 1}}
	source.SendRecord(rec2_1)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 2 {
		t.Fatalf("expected 2 written records, got %d", len(sink.writtenRecords))
	}

	rec1_3 := RecordData{TrackingData: TrackingData{SourceKey: key1, SourceRowID: 3}}
	source.SendRecord(rec1_3)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 2 {
		t.Fatalf("expected still 2 written records, got %d", len(sink.writtenRecords))
	}

	time.Sleep(5500 * time.Millisecond)

	if len(source.resendRequests) != 1 {
		t.Fatalf("expected 1 resend request, got %d", len(source.resendRequests))
	}
	if source.resendRequests[0].SourceNodeID != "node1" {
		t.Fatal("resend for wrong source")
	}
}

func TestLibP2PToSqliteForwarder_Start_WithInitialLatest(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	key := SourceKey{SourceNodeId: "node1", SourcePath: "path1"}

	source := &mockLibP2PConnector{}
	sink := &mockSqliteConnector{
		getLatestRowIdsRet: map[SourceKey]int64{key: 5},
	}

	f, err := newLibP2PToSqliteForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec6 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 6}}
	source.SendRecord(rec6)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 1 {
		t.Fatalf("expected 1 written record, got %d", len(sink.writtenRecords))
	}

	rec7 := RecordData{TrackingData: TrackingData{SourceKey: key, SourceRowID: 7}}
	source.SendRecord(rec7)

	time.Sleep(200 * time.Millisecond)

	if len(sink.writtenRecords) != 2 {
		t.Fatalf("expected 2 written records, got %d", len(sink.writtenRecords))
	}
}

func TestNewSqliteToLibP2PForwarder(t *testing.T) {
	source := &mockSqliteConnector{}
	sink := &mockLibP2PConnector{}
	f, err := newSqliteToLibP2PForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f == nil {
		t.Fatal("expected non-nil forwarder")
	}
}

func TestSqliteToLibP2PForwarder_Start_TailRecords(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &mockSqliteConnector{
		nodeId:    "node1",
		tablePath: "path1",
	}
	sink := &mockLibP2PConnector{}

	f, err := newSqliteToLibP2PForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec1 := RecordData{TrackingData: TrackingData{SourceRowID: 1}}
	source.SendRecord(rec1)

	time.Sleep(100 * time.Millisecond)

	if len(sink.writtenRecords) != 1 {
		t.Fatalf("expected 1 written record, got %d", len(sink.writtenRecords))
	}
	if !reflect.DeepEqual(sink.writtenRecords[0], rec1) {
		t.Fatal("written record mismatch")
	}

	rec2 := RecordData{TrackingData: TrackingData{SourceRowID: 2}}
	source.SendRecord(rec2)

	time.Sleep(100 * time.Millisecond)

	if len(sink.writtenRecords) != 2 {
		t.Fatalf("expected 2 written records, got %d", len(sink.writtenRecords))
	}
	if !reflect.DeepEqual(sink.writtenRecords[1], rec2) {
		t.Fatal("written record mismatch")
	}
}

func TestSqliteToLibP2PForwarder_Start_ResendRequest_Matching(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &mockSqliteConnector{
		nodeId:    "node1",
		tablePath: "path1",
		readRangeRet: []RecordData{
			{TrackingData: TrackingData{SourceRowID: 5}},
		},
	}
	sink := &mockLibP2PConnector{}

	f, err := newSqliteToLibP2PForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	req := ResendRequest{
		SourceNodeID: "node1",
		SourcePath:   "path1",
		Gaps:         []GapRange{{Start: 5, End: 6}},
	}
	sink.SendResend(req)

	time.Sleep(100 * time.Millisecond)

	if len(source.readRangeCalls) != 1 {
		t.Fatalf("expected 1 readRange call, got %d", len(source.readRangeCalls))
	}
	if source.readRangeCalls[0].start != 5 || source.readRangeCalls[0].end != 6 {
		t.Fatal("readRange args mismatch")
	}

	if len(sink.writtenRecords) != 1 {
		t.Fatalf("expected 1 written record from resend, got %d", len(sink.writtenRecords))
	}
	if sink.writtenRecords[0].SourceRowID != 5 {
		t.Fatal("resend record mismatch")
	}
}

func TestSqliteToLibP2PForwarder_Start_ResendRequest_NotMatching(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &mockSqliteConnector{
		nodeId:    "node1",
		tablePath: "path1",
	}
	sink := &mockLibP2PConnector{}

	f, err := newSqliteToLibP2PForwarder(source, sink)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = f.Start(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	req := ResendRequest{
		SourceNodeID: "node2",
		SourcePath:   "path2",
		Gaps:         []GapRange{{Start: 5, End: 5}},
	}
	sink.SendResend(req)

	time.Sleep(100 * time.Millisecond)

	if len(source.readRangeCalls) != 0 {
		t.Fatalf("expected 0 readRange calls, got %d", len(source.readRangeCalls))
	}

	if len(sink.writtenRecords) != 0 {
		t.Fatalf("expected 0 written records, got %d", len(sink.writtenRecords))
	}
}

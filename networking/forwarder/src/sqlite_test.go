package forwarder

import (
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"database/sql"

	_ "github.com/mattn/go-sqlite3"
)

func TestNewSQLiteConnectorCreatesTable(t *testing.T) {
	c, err := newSQLiteConnector(":memory:", "test_table")
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer c.close()

	rows, err := c.db.Query(`PRAGMA table_info("test_table")`)
	if err != nil {
		t.Fatalf("failed to query table info: %v", err)
	}
	defer rows.Close()

	expectedCols := map[string]string{
		"source_node_id":   "TEXT",
		"source_path":      "TEXT",
		"source_row_id":    "INTEGER",
		"source_timestamp": "DATETIME",
	}
	foundCols := make(map[string]string)
	for rows.Next() {
		var cid int
		var name, typ string
		var notnull int
		var dflt interface{}
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			t.Fatalf("failed to scan: %v", err)
		}
		foundCols[name] = typ
	}
	if !reflect.DeepEqual(expectedCols, foundCols) {
		t.Errorf("expected columns %v, got %v", expectedCols, foundCols)
	}
}

func TestEnsureTrackingColumnsAddsMissing(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	_, err = db.Exec(`CREATE TABLE test_table (source_node_id TEXT, data TEXT)`)
	if err != nil {
		t.Fatalf("failed to create partial table: %v", err)
	}
	db.Close()

	tempDB := t.TempDir() + "/test.db"
	db, err = sql.Open("sqlite3", tempDB)
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	_, err = db.Exec(`CREATE TABLE test_table (source_node_id TEXT, data TEXT)`)
	if err != nil {
		t.Fatalf("failed to create partial table: %v", err)
	}
	db.Close()

	c, err := newSQLiteConnector(tempDB, "test_table")
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer c.close()

	rows, err := c.db.Query(`PRAGMA table_info("test_table")`)
	if err != nil {
		t.Fatalf("failed to query table info: %v", err)
	}
	defer rows.Close()

	expectedCols := []string{"source_node_id", "data", "source_path", "source_row_id", "source_timestamp"}
	foundCols := []string{}
	for rows.Next() {
		var cid int
		var name string
		var typ string
		var notnull int
		var dflt interface{}
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			t.Fatalf("failed to scan: %v", err)
		}
		foundCols = append(foundCols, name)
	}
	if len(foundCols) != len(expectedCols) {
		t.Errorf("expected %d columns, got %d: %v", len(expectedCols), len(foundCols), foundCols)
	}
}

func TestWriteAndReadRecord(t *testing.T) {
	SetNodeId("node1")
	c, err := newSQLiteConnector("test_write_and_read_db1", "table")
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer func() {
		c.close()
		os.Remove("test_write_and_read_db1")
	}()

	rec := RecordData{
		TrackingData: TrackingData{
			SourceKey: SourceKey{
				SourceNodeId: "node1",
				SourcePath:   "test_write_and_read_db1:table",
			},
			SourceRowID:     42,
			SourceTimestamp: time.Now().UTC(),
		},
		Data: map[string]interface{}{
			"key": "value",
			"num": 123.45,
		},
	}
	if err := c.write(rec); err != nil {
		t.Fatalf("failed to write: %v", err)
	}
	time.Sleep(200 * time.Millisecond) // Wait for flush

	records, err := c.readRange(1, 999)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	got := records[0]
	if got.SourceNodeId != rec.SourceNodeId || got.SourcePath != rec.SourcePath || got.SourceRowID != 1 {
		t.Errorf("tracking data mismatch: got %+v, want %+v", got.TrackingData, rec.TrackingData)
	}
	if !reflect.DeepEqual(got.Data, rec.Data) {
		t.Errorf("data mismatch: got %v, want %v", got.Data, rec.Data)
	}
}

func TestTailDetectsWrites(t *testing.T) {
	SetNodeId("node2")
	db, errDb := sql.Open("sqlite3", "tail_detects_writes_db2")
	if errDb != nil {
		t.Fatalf("failed to open db for alter: %v", errDb)
	}

	_, errExec := db.Exec("CREATE TABLE table2 (test BOOLEAN)")
	if errExec != nil {
		t.Fatalf("failed to create table: %v", errExec)
	}
	db.Close()

	c, err := newSQLiteConnector("tail_detects_writes_db2", "table2")
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer c.close()

	ch := make(chan RecordData, 1)
	c.tail(func(r RecordData) error {
		ch <- r
		return nil
	})
	time.Sleep(100 * time.Millisecond) // Let tail start

	rec := RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node2", SourcePath: "tail_detects_writes_db2:table2"},
			SourceRowID:     100,
			SourceTimestamp: time.Now().UTC(),
		},
		Data: map[string]interface{}{"test": true},
	}
	if err := c.write(rec); err != nil {
		t.Fatalf("failed to write: %v", err)
	}
	time.Sleep(200 * time.Millisecond) // Wait for flush and tail poll

	select {
	case got := <-ch:
		if !reflect.DeepEqual(got.Data, rec.Data) {
			t.Errorf("got %v, want %v", got, rec)
		}
		if got.SourceNodeId != rec.SourceNodeId || got.SourcePath != rec.SourcePath || got.SourceRowID != 1 {
			t.Errorf("tracking data mismatch: got %+v, want %+v", got.TrackingData, rec.TrackingData)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for tail handler")
	}
	os.Remove("tail_detects_writes_db2")
	os.Remove("tail_detects_writes_db2-wal")
	os.Remove("tail_detects_writes_db2-shm")

}

func TestBatchWriteMultipleEdge(t *testing.T) {
	c, err := newSQLiteConnector(":memory:", "test_table")
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer c.close()

	for i := 0; i < 3; i++ {
		rec := RecordData{
			TrackingData: TrackingData{
				SourceKey:       SourceKey{SourceNodeId: fmt.Sprintf("node%d", i), SourcePath: ""},
				SourceRowID:     int64(i),
				SourceTimestamp: time.Time{},
			},
			Data: nil, // Edge: nil Data
		}
		if err := c.write(rec); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}
	time.Sleep(200 * time.Millisecond)

	var count int
	err = c.db.QueryRow(`SELECT COUNT(*) FROM "test_table"`).Scan(&count)
	if err != nil {
		t.Fatalf("failed to count: %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3 rows, got %d", count)
	}
}

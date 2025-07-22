package forwarder

import (
	"database/sql"
	"errors"
	"fmt"
	"log"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type sqliteConnector struct {
	db            *sql.DB
	tableName     string
	stop          chan struct{}
	wg            sync.WaitGroup
	pendingWrites []RecordData
	mu            sync.Mutex
	nodeId        string
	tablePath     string
	// Cache the original columns (non-tracking columns)
	originalColumns []string
	columnTypes     map[string]string
}

func newSQLiteConnector(dbPath, tableName string) (*sqliteConnector, error) {
	if tableName == "" {
		return nil, errors.New("table name cannot be empty")
	}
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA busy_timeout = 500; PRAGMA cache_size = -64000;")
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to apply PRAGMA settings: %w", err)
	}

	// Increase connection pool for better concurrency
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(5 * time.Minute)

	c := &sqliteConnector{
		db:            db,
		tableName:     tableName,
		stop:          make(chan struct{}),
		pendingWrites: []RecordData{},
		nodeId:        GetNodeId(),
		tablePath:     dbPath + ":" + tableName,
		columnTypes:   make(map[string]string),
	}

	// Get the table schema before adding tracking columns
	err = c.loadTableSchema()
	if err != nil && !strings.Contains(err.Error(), "no such table") {
		db.Close()
		return nil, err
	}

	err = c.ensureTrackingColumns()
	if err != nil {
		db.Close()
		return nil, err
	}

	// Reload schema after ensuring tracking columns
	err = c.loadTableSchema()
	if err != nil {
		db.Close()
		return nil, err
	}

	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		c.writerLoop()
	}()
	return c, nil
}

func (c *sqliteConnector) loadTableSchema() error {
	rows, err := c.db.Query(fmt.Sprintf(`PRAGMA table_info("%s")`, c.tableName))
	if err != nil {
		return err
	}
	defer rows.Close()

	trackingCols := make(map[string]bool)
	for _, col := range []string{"source_node_id", "source_path", "source_row_id", "source_timestamp"} {
		trackingCols[col] = true
	}

	c.originalColumns = []string{}
	c.columnTypes = make(map[string]string)

	for rows.Next() {
		var cid int
		var name string
		var typ string
		var notnull int
		var dflt interface{}
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			return err
		}

		c.columnTypes[name] = typ

		// Only include non-tracking columns in originalColumns
		if !trackingCols[name] {
			c.originalColumns = append(c.originalColumns, name)
		}
	}

	return nil
}

func (c *sqliteConnector) getNodeId() string {
	return c.nodeId
}

func (c *sqliteConnector) getTablePath() string {
	return c.tablePath
}

func (c *sqliteConnector) writerLoop() {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			c.mu.Lock()
			batch := c.pendingWrites
			c.pendingWrites = nil
			c.mu.Unlock()
			if len(batch) > 0 {
				if err := c.writeBatch(batch); err != nil {
					log.Printf("Error writing batch: %v", err)
				}
			}
		case <-c.stop:
			c.mu.Lock()
			batch := c.pendingWrites
			c.pendingWrites = nil
			c.mu.Unlock()
			if len(batch) > 0 {
				if err := c.writeBatch(batch); err != nil {
					log.Printf("Error writing final batch: %v", err)
				}
			}
			return
		}
	}
}

func (c *sqliteConnector) writeBatch(records []RecordData) error {
	if len(records) == 0 {
		return nil
	}
	tx, err := c.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Build column list: tracking columns + original columns
	trackingCols := []string{"source_node_id", "source_path", "source_row_id", "source_timestamp"}
	cols := append(trackingCols, c.originalColumns...)
	colStr := strings.Join(cols, `", "`)

	places := make([]string, len(cols))
	for i := range places {
		places[i] = "?"
	}
	singlePlace := "(" + strings.Join(places, ", ") + ")"
	rowPlaces := make([]string, len(records))
	for i := range rowPlaces {
		rowPlaces[i] = singlePlace
	}
	valuesStr := strings.Join(rowPlaces, ", ")

	query := fmt.Sprintf(`INSERT INTO "%s" ("%s") VALUES %s`, c.tableName, colStr, valuesStr)

	vals := make([]interface{}, 0, len(records)*len(cols))
	for _, rec := range records {
		// Add tracking columns
		vals = append(vals, rec.SourceNodeId, rec.SourcePath, rec.SourceRowID, rec.SourceTimestamp)

		// Add original column values from Data map
		for _, col := range c.originalColumns {
			if val, ok := rec.Data[col]; ok {
				vals = append(vals, val)
			} else {
				vals = append(vals, nil)
			}
		}
	}

	_, err = tx.Exec(query, vals...)
	if err != nil {
		return err
	}
	return tx.Commit()
}

func (c *sqliteConnector) ensureTrackingColumns() error {
	// Wrap table creation and alterations in a transaction for atomicity
	tx, err := c.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Check if table exists
	var count int
	err = tx.QueryRow(`SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?`, c.tableName).Scan(&count)
	if err != nil {
		return err
	}
	if count == 0 {
		// Create table with only tracking columns initially
		// The original schema should be defined by the first records written
		typePairs := getJsonTagsWithSqliteTypes(reflect.TypeOf(TrackingData{}))
		colDefs := make([]string, 0, len(typePairs))
		for _, pair := range typePairs {
			colDefs = append(colDefs, fmt.Sprintf("%s %s", pair.name, pair.typeStr))
		}
		createQuery := fmt.Sprintf(`CREATE TABLE "%s" (%s)`, c.tableName, strings.Join(colDefs, ", "))
		_, err := tx.Exec(createQuery)
		if err != nil {
			return err
		}
	} else {
		// Table exists, ensure tracking columns
		existing := make(map[string]bool)
		rows, err := tx.Query(fmt.Sprintf(`PRAGMA table_info("%s")`, c.tableName))
		if err != nil {
			return err
		}
		defer rows.Close()
		for rows.Next() {
			var cid int
			var name string
			var typ string
			var notnull int
			var dflt interface{}
			var pk int
			if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
				return err
			}
			existing[name] = true
		}

		typePairs := getJsonTagsWithSqliteTypes(reflect.TypeOf(TrackingData{}))
		for _, pair := range typePairs {
			if !existing[pair.name] {
				if _, err := tx.Exec(fmt.Sprintf(`ALTER TABLE "%s" ADD COLUMN %s %s`, c.tableName, pair.name, pair.typeStr)); err != nil {
					return err
				}
			}
		}
	}

	return tx.Commit()
}

func (c *sqliteConnector) getLatestRowIds() (map[SourceKey]int64, error) {
	keyCols := getJsonTagNames(reflect.TypeOf(SourceKey{}))
	rowIdField := "SourceRowID"
	rowIDCol := getFieldJsonTag(reflect.TypeOf(TrackingData{}), rowIdField)
	if rowIDCol == "" {
		return nil, fmt.Errorf("could not find field %s in TrackingData struct", rowIdField)
	}

	selectCols := strings.Join(keyCols, ", ")
	query := fmt.Sprintf(`SELECT %s, MAX(%s) FROM "%s" GROUP BY %s`, selectCols, rowIDCol, c.tableName, selectCols)

	rows, err := c.db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	m := make(map[SourceKey]int64)
	for rows.Next() {
		strPtrs := make([]*string, len(keyCols))
		scanArgs := make([]interface{}, 0, len(keyCols)+1)
		for i := range keyCols {
			var s string
			strPtrs[i] = &s
			scanArgs = append(scanArgs, &s)
		}
		var maxPtr int64
		scanArgs = append(scanArgs, &maxPtr)
		if err := rows.Scan(scanArgs...); err != nil {
			return nil, err
		}
		var key SourceKey
		val := reflect.ValueOf(&key).Elem()
		keyType := reflect.TypeOf(key)
		for i, colName := range keyCols {
			// find field with json tag = colName
			for f := 0; f < keyType.NumField(); f++ {
				field := keyType.Field(f)
				tag := strings.Split(field.Tag.Get("json"), ",")[0]
				if tag == "" {
					tag = strings.ToLower(field.Name)
				}
				if tag == colName {
					if strPtrs[i] != nil {
						val.FieldByName(field.Name).SetString(*strPtrs[i])
					}
					break
				}
			}
		}
		m[key] = maxPtr
	}
	return m, nil
}

func (c *sqliteConnector) scanToRecord(rows *sql.Rows) (RecordData, int64, error) {
	// Get column names from the result set
	columns, err := rows.Columns()
	if err != nil {
		return RecordData{}, 0, err
	}

	// Create scan destinations
	scanArgs := make([]interface{}, len(columns))
	values := make([]interface{}, len(columns))
	for i := range values {
		scanArgs[i] = &values[i]
	}

	err = rows.Scan(scanArgs...)
	if err != nil {
		return RecordData{}, 0, err
	}

	var rec RecordData
	rec.Data = make(map[string]interface{})
	var rowID int64

	// Process each column
	for i, col := range columns {
		val := values[i]

		// Handle NULL values
		if val == nil {
			continue
		}

		// Convert []byte to appropriate type
		if b, ok := val.([]byte); ok {
			val = string(b)
		}

		switch col {
		case "source_node_id":
			if s, ok := val.(string); ok {
				rec.SourceNodeId = s
			}
		case "source_path":
			if s, ok := val.(string); ok {
				rec.SourcePath = s
			}
		case "source_row_id":
			switch v := val.(type) {
			case int64:
				rec.SourceRowID = v
			case int:
				rec.SourceRowID = int64(v)
			case string:
				if parsed, err := strconv.ParseInt(v, 10, 64); err == nil {
					rec.SourceRowID = parsed
				}
			}
		case "source_timestamp":
			switch v := val.(type) {
			case time.Time:
				rec.SourceTimestamp = v
			case string:
				if parsed, err := time.Parse(time.RFC3339Nano, v); err == nil {
					rec.SourceTimestamp = parsed
				} else if parsed, err := time.Parse("2006-01-02 15:04:05", v); err == nil {
					rec.SourceTimestamp = parsed
				}
			}
		case "rowid":
			switch v := val.(type) {
			case int64:
				rowID = v
			case int:
				rowID = int64(v)
			}
		default:
			// All other columns go into the Data map
			rec.Data[col] = val
		}
	}

	return rec, rowID, nil
}

func (c *sqliteConnector) readRange(start, end int64) ([]RecordData, error) {
	// Select all columns plus rowid
	query := fmt.Sprintf(`SELECT *, rowid FROM "%s" WHERE rowid >= ? AND rowid <= ? ORDER BY rowid`, c.tableName)
	rows, err := c.db.Query(query, start, end)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []RecordData
	for rows.Next() {
		rec, rowID, err := c.scanToRecord(rows)
		if err != nil {
			return nil, err
		}
		// Override tracking data so that this table is treated as the new source
		rec.SourceNodeId = c.nodeId
		rec.SourcePath = c.tablePath
		rec.SourceRowID = rowID
		rec.SourceTimestamp = time.Now()
		records = append(records, rec)
	}
	return records, nil
}

func (c *sqliteConnector) tail(handler func(record RecordData) error) {
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		var last int64
		err := c.db.QueryRow(fmt.Sprintf(`SELECT IFNULL(MAX(rowid), 0) FROM "%s"`, c.tableName)).Scan(&last)
		if err != nil {
			last = 0
		}
		// Prepare the statement outside the loop for efficiency
		query := fmt.Sprintf(`SELECT *, rowid FROM "%s" WHERE rowid > ? ORDER BY rowid LIMIT ?`, c.tableName)
		stmt, err := c.db.Prepare(query)
		if err != nil {
			log.Printf("Error preparing tail statement: %v", err)
			return
		}
		defer stmt.Close()

		// Adaptive polling: start fast, slow down when idle
		minPollInterval := 1 * time.Millisecond
		maxPollInterval := 50 * time.Millisecond
		currentInterval := minPollInterval
		batchSize := 500 // Process records in larger batches for better throughput

		for {
			select {
			case <-c.stop:
				return
			default:
			}
			rows, err := stmt.Query(last, batchSize)
			if err != nil {
				time.Sleep(currentInterval)
				continue
			}
			hadNew := false
			recordCount := 0
			for rows.Next() {
				rec, rowID, err := c.scanToRecord(rows)
				if err != nil {
					log.Printf("Error scanning record: %v", err)
					break
				}
				// Override tracking data so that this table is treated as the new source
				rec.SourceNodeId = c.nodeId
				rec.SourcePath = c.tablePath
				rec.SourceRowID = rowID
				rec.SourceTimestamp = time.Now()
				last = rowID
				err = handler(rec)
				if err != nil {
					log.Printf("Error handling record: %v", err)
				}
				hadNew = true
				recordCount++
			}
			rows.Close()

			// Adaptive interval adjustment
			if hadNew {
				// Had records, speed up polling
				currentInterval = minPollInterval
				if recordCount == batchSize {
					// Full batch, poll immediately
					continue
				}
			} else {
				// No records, slow down gradually
				currentInterval = time.Duration(float64(currentInterval) * 1.5)
				if currentInterval > maxPollInterval {
					currentInterval = maxPollInterval
				}
			}
			time.Sleep(currentInterval)
		}
	}()
}

func (c *sqliteConnector) write(record RecordData) error {
	// If we don't know the schema yet, try to infer it from the first record
	if len(c.originalColumns) == 0 && len(record.Data) > 0 {
		c.mu.Lock()
		if len(c.originalColumns) == 0 {
			// Infer columns from the data
			for col := range record.Data {
				c.originalColumns = append(c.originalColumns, col)
			}
			// Sort for consistency
			sort.Strings(c.originalColumns)

			// Add columns to table if they don't exist
			tx, err := c.db.Begin()
			if err == nil {
				defer tx.Rollback()
				for col := range record.Data {
					// Infer SQL type from Go type
					sqlType := "TEXT" // default
					switch record.Data[col].(type) {
					case int, int32, int64:
						sqlType = "INTEGER"
					case float32, float64:
						sqlType = "REAL"
					case bool:
						sqlType = "INTEGER"
					}

					// Try to add column (will fail silently if it exists)
					tx.Exec(fmt.Sprintf(`ALTER TABLE "%s" ADD COLUMN "%s" %s`, c.tableName, col, sqlType))
				}
				tx.Commit()
			}
		}
		c.mu.Unlock()
	}

	c.mu.Lock()
	c.pendingWrites = append(c.pendingWrites, record)
	c.mu.Unlock()
	return nil
}

func (c *sqliteConnector) close() error {
	close(c.stop)
	c.wg.Wait()
	return c.db.Close()
}

func (c *sqliteConnector) getType() string {
	return "sqlite"
}

type typedPair struct {
	name    string
	typeStr string
}

func getJsonTagsWithSqliteTypes(t reflect.Type) []typedPair {
	typePairs := []typedPair{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.Anonymous {
			typePairs = append(typePairs, getJsonTagsWithSqliteTypes(f.Type)...)
			continue
		}
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		if tag != "" {
			tag = strings.Split(tag, ",")[0]
		}
		if tag == "" {
			tag = strings.ToLower(f.Name)
		}
		var sqlType string
		switch f.Type.Kind() {
		case reflect.String:
			sqlType = "TEXT"
		case reflect.Int, reflect.Int32, reflect.Int64:
			sqlType = "INTEGER"
		default:
			if f.Type == reflect.TypeOf(time.Time{}) {
				sqlType = "DATETIME"
			} else {
				sqlType = "BLOB"
			}
		}
		typePairs = append(typePairs, typedPair{tag, sqlType})
	}
	return typePairs
}

func getJsonTagNames(t reflect.Type) []string {
	cols := []string{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.Anonymous {
			cols = append(cols, getJsonTagNames(f.Type)...)
			continue
		}
		tag := strings.Split(f.Tag.Get("json"), ",")[0]
		if tag == "-" {
			continue
		}
		if tag == "" {
			tag = strings.ToLower(f.Name)
		}
		cols = append(cols, tag)
	}
	return cols
}

func getFieldJsonTag(t reflect.Type, fieldName string) string {
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.Anonymous {
			if tag := getFieldJsonTag(f.Type, fieldName); tag != "" {
				return tag
			}
			continue
		}
		if f.Name == fieldName {
			tag := strings.Split(f.Tag.Get("json"), ",")[0]
			if tag == "" {
				return strings.ToLower(f.Name)
			}
			return tag
		}
	}
	return ""
}

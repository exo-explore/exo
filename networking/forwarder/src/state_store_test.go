package forwarder

import (
	"testing"
	"time"
)

func TestInOrderMessages_SingleSource(t *testing.T) {
	store := newStateStore(make(map[SourceKey]int64))
	sk := SourceKey{"node1", "path1"}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     2,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     3,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	writeable := store.getWriteableMessages()
	if len(writeable) != 3 || writeable[0].SourceRowID != 1 || writeable[1].SourceRowID != 2 || writeable[2].SourceRowID != 3 {
		t.Errorf("Expected 3 contiguous messages, got %v", writeable)
	}

	gaps := store.getCurrentGaps()[sk]
	if len(gaps) != 0 {
		t.Errorf("Expected no gaps, got %v", gaps)
	}

	if store.lastContiguousRowId[sk] != 3 {
		t.Errorf("Expected lastContiguous=3, got %d", store.lastContiguousRowId[sk])
	}
}

func TestOutOfOrder_CreateGapThenFill(t *testing.T) {
	store := newStateStore(make(map[SourceKey]int64))
	sk := SourceKey{"node1", "path1"}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     3,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	gaps := store.getCurrentGaps()[sk]
	if len(gaps) != 1 || gaps[0].Start != 2 || gaps[0].End != 2 {
		t.Errorf("Expected gap [2,2], got %v", gaps)
	}

	writeable := store.getWriteableMessages()
	if len(writeable) != 1 || writeable[0].SourceRowID != 1 {
		t.Errorf("Expected only 1 written, got %v", writeable)
	}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     2,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	writeable = store.getWriteableMessages()
	if len(writeable) != 2 || writeable[0].SourceRowID != 2 || writeable[1].SourceRowID != 3 {
		t.Errorf("Expected 1 and 2 written, got %v", writeable)
	}

	gaps = store.getCurrentGaps()[sk]
	if len(gaps) != 0 {
		t.Errorf("Expected no gaps after fill, got %v", gaps)
	}

	if store.lastContiguousRowId[sk] != 3 {
		t.Errorf("Expected lastContiguous=3, got %d", store.lastContiguousRowId[sk])
	}
}

func TestFillMiddleOfGap_Split(t *testing.T) {
	store := newStateStore(make(map[SourceKey]int64))
	sk := SourceKey{"node1", "path1"}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     5,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	gaps := store.getCurrentGaps()[sk]
	if len(gaps) != 1 || gaps[0].Start != 2 || gaps[0].End != 4 {
		t.Errorf("Expected gap [1,4], got %v", gaps)
	}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     3,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	gaps = store.getCurrentGaps()[sk]
	if len(gaps) != 2 || gaps[0].Start != 2 || gaps[0].End != 2 || gaps[1].Start != 4 || gaps[1].End != 4 {
		t.Errorf("Expected gaps [1,1] and [3,4], got %v", gaps)
	}

	writeable := store.getWriteableMessages()
	if len(writeable) != 1 || writeable[0].SourceRowID != 1 {
		t.Errorf("Expected only 0 written, got %v", writeable)
	}

	if len(store.pending[sk]) != 2 {
		t.Errorf("Expected 2 pending runs, got %d", len(store.pending[sk]))
	}
}

func TestMultipleRuns_FillConnectingGap_MergeAndPartialAdvance(t *testing.T) {
	store := newStateStore(make(map[SourceKey]int64))
	sk := SourceKey{"node1", "path1"}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     1,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     2,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     4,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     5,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})
	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     7,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	gaps := store.getCurrentGaps()[sk]
	if len(gaps) != 2 || gaps[0].Start != 3 || gaps[0].End != 3 || gaps[1].Start != 6 || gaps[1].End != 6 {
		t.Errorf("Expected gaps [3,3],[6,6], got %v", gaps)
	}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     3,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	writeable := store.getWriteableMessages()
	if len(writeable) != 5 || writeable[4].SourceRowID != 5 {
		t.Errorf("Expected 1-5 written, got %v", writeable)
	}

	gaps = store.getCurrentGaps()[sk]
	if len(gaps) != 1 || gaps[0].Start != 6 || gaps[0].End != 6 {
		t.Errorf("Expected gap [6,6], got %v", gaps)
	}

	if store.lastContiguousRowId[sk] != 5 {
		t.Errorf("Expected lastContiguous=5, got %d", store.lastContiguousRowId[sk])
	}

	if len(store.pending[sk]) != 1 || store.pending[sk][0].start != 7 {
		t.Errorf("Expected pending [7,7], got %v", store.pending[sk])
	}
}

func TestInitialHighRowID_CreateGap_IgnoreDuplicateAndOld(t *testing.T) {
	store := newStateStore(make(map[SourceKey]int64))
	sk := SourceKey{"node1", "path1"}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     3,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	gaps := store.getCurrentGaps()[sk]
	if len(gaps) != 1 || gaps[0].Start != 1 || gaps[0].End != 2 {
		t.Errorf("Expected gap [1,2], got %v", gaps)
	}

	writeable := store.getWriteableMessages()
	if len(writeable) != 0 {
		t.Errorf("Expected no writeable, got %v", writeable)
	}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     3,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	if len(store.pending[sk]) != 1 || len(store.pending[sk][0].records) != 1 {
		t.Errorf("Duplicate added unexpectedly")
	}

	store.onRecord(RecordData{
		TrackingData: TrackingData{
			SourceKey:       SourceKey{SourceNodeId: "node1", SourcePath: "path1"},
			SourceRowID:     -1,
			SourceTimestamp: time.Now(),
		},
		Data: nil,
	})

	if store.lastContiguousRowId[sk] != 0 {
		t.Errorf("Old message affected lastContiguous")
	}
}

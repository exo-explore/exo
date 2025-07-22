package forwarder

import (
	"sort"
	"sync"
	"time"
)

const gracePeriod = 5 * time.Second

type gap struct {
	GapRange
	firstSeen        time.Time
	lastRequestSent  time.Time
	timesRequestSent int
}

type pendingRecordsRange struct {
	start   int64
	end     int64
	records map[int64]RecordData
}

func (g gap) isResendable() bool {
	currentTime := time.Now()
	if currentTime.Before(g.firstSeen.Add(gracePeriod)) {
		return false
	}
	backoff := gracePeriod * (1 << g.timesRequestSent)
	return currentTime.After(g.lastRequestSent.Add(backoff))
}

type stateStore struct {
	mu                  sync.RWMutex
	sourceKeyMu         map[SourceKey]*sync.Mutex
	lastContiguousRowId map[SourceKey]int64
	recordsToWrite      []RecordData
	gaps                map[SourceKey][]gap
	pending             map[SourceKey][]pendingRecordsRange
}

func newStateStore(lastWrittenRowId map[SourceKey]int64) *stateStore {
	return &stateStore{
		lastContiguousRowId: lastWrittenRowId,
		recordsToWrite:      []RecordData{},
		gaps:                make(map[SourceKey][]gap),
		pending:             make(map[SourceKey][]pendingRecordsRange),
		sourceKeyMu:         make(map[SourceKey]*sync.Mutex),
	}
}

func (s *stateStore) onRecord(record RecordData) {
	sk := SourceKey{SourceNodeId: record.SourceNodeId, SourcePath: record.SourcePath}

	s.mu.Lock()
	if _, ok := s.sourceKeyMu[sk]; !ok {
		s.sourceKeyMu[sk] = &sync.Mutex{}
		if _, ok := s.lastContiguousRowId[sk]; !ok {
			s.lastContiguousRowId[sk] = 0
		}
		s.gaps[sk] = []gap{}
		s.pending[sk] = []pendingRecordsRange{}
	}
	s.mu.Unlock()
	s.sourceKeyMu[sk].Lock()
	defer s.sourceKeyMu[sk].Unlock()
	l := s.lastContiguousRowId[sk]
	r := record.SourceRowID
	if r <= l {
		return
	}

	for _, ru := range s.pending[sk] {
		if _, has := ru.records[r]; has {
			return
		}
	}

	currentHighest := l
	for _, ru := range s.pending[sk] {
		if ru.end > currentHighest {
			currentHighest = ru.end
		}
	}

	gaps := s.gaps[sk]
	newGaps := []gap{}
	filled := false
	for _, g := range gaps {
		if g.Start <= r && r <= g.End {
			filled = true
			if g.Start < r {
				newGaps = append(newGaps, gap{GapRange: GapRange{Start: g.Start, End: r - 1}, firstSeen: g.firstSeen, lastRequestSent: g.lastRequestSent, timesRequestSent: g.timesRequestSent})
			}
			if r < g.End {
				newGaps = append(newGaps, gap{GapRange: GapRange{Start: r + 1, End: g.End}, firstSeen: g.firstSeen, lastRequestSent: g.lastRequestSent, timesRequestSent: g.timesRequestSent})
			}
		} else {
			newGaps = append(newGaps, g)
		}
	}
	s.gaps[sk] = mergeGaps(newGaps)

	if !filled && r > currentHighest+1 {
		gr := GapRange{Start: currentHighest + 1, End: r - 1}
		if gr.Start <= gr.End {
			newG := gap{GapRange: gr, firstSeen: time.Now(), lastRequestSent: time.Time{}, timesRequestSent: 0}
			s.gaps[sk] = append(s.gaps[sk], newG)
			s.gaps[sk] = mergeGaps(s.gaps[sk])
		}
	}
	newRun := pendingRecordsRange{start: r, end: r, records: map[int64]RecordData{r: record}}
	s.pending[sk] = addPending(s.pending[sk], newRun)

	var toWrite []RecordData
	runs := s.pending[sk]
	for len(runs) > 0 && runs[0].start == s.lastContiguousRowId[sk]+1 {
		ru := runs[0]
		for id := ru.start; id <= ru.end; id++ {
			toWrite = append(toWrite, ru.records[id])
		}
		s.lastContiguousRowId[sk] = ru.end
		s.pending[sk] = runs[1:]
		runs = s.pending[sk]
	}

	if len(toWrite) > 0 {
		s.mu.Lock()
		s.recordsToWrite = append(s.recordsToWrite, toWrite...)
		s.mu.Unlock()
	}
}

func (s *stateStore) getWriteableMessages() []RecordData {
	s.mu.Lock()
	defer s.mu.Unlock()
	records := s.recordsToWrite[:]
	s.recordsToWrite = []RecordData{}
	return records
}

func (s *stateStore) getResendRequests() []ResendRequest {
	s.mu.RLock()
	keys := make([]SourceKey, 0, len(s.gaps))
	for k := range s.gaps {
		keys = append(keys, k)
	}
	s.mu.RUnlock()

	resendRequests := []ResendRequest{}
	for _, sk := range keys {
		if _, ok := s.sourceKeyMu[sk]; !ok {
			continue
		}
		s.sourceKeyMu[sk].Lock()
		gaps, ok := s.gaps[sk]
		if !ok {
			s.sourceKeyMu[sk].Unlock()
			continue
		}
		gapRanges := []GapRange{}
		for i := range gaps {
			if gaps[i].isResendable() {
				gapRanges = append(gapRanges, gaps[i].GapRange)
				gaps[i].lastRequestSent = time.Now()
				gaps[i].timesRequestSent++
			}
		}
		if len(gapRanges) > 0 {
			resendRequests = append(resendRequests, ResendRequest{
				SourceNodeID: sk.SourceNodeId,
				SourcePath:   sk.SourcePath,
				Gaps:         gapRanges,
			})
		}
		s.sourceKeyMu[sk].Unlock()
	}
	return resendRequests
}

func (s *stateStore) getCurrentGaps() map[SourceKey][]gap {
	s.mu.RLock()
	defer s.mu.RUnlock()
	copied := make(map[SourceKey][]gap, len(s.gaps))
	for k, v := range s.gaps {
		gapCopy := make([]gap, len(v))
		copy(gapCopy, v)
		copied[k] = gapCopy
	}
	return copied
}

func addPending(pending []pendingRecordsRange, newPending pendingRecordsRange) []pendingRecordsRange {
	temp := append(append([]pendingRecordsRange{}, pending...), newPending)
	sort.Slice(temp, func(i, j int) bool { return temp[i].start < temp[j].start })
	merged := []pendingRecordsRange{}
	for _, p := range temp {
		if len(merged) == 0 || merged[len(merged)-1].end+1 < p.start {
			merged = append(merged, p)
			continue
		}
		lastIdx := len(merged) - 1
		if merged[lastIdx].end < p.end {
			merged[lastIdx].end = p.end
		}
		for k, v := range p.records {
			merged[lastIdx].records[k] = v
		}
	}
	return merged
}

func mergeGaps(gaps []gap) []gap {
	if len(gaps) == 0 {
		return gaps
	}
	sort.Slice(gaps, func(i, j int) bool { return gaps[i].Start < gaps[j].Start })
	merged := []gap{gaps[0]}
	for _, g := range gaps[1:] {
		lastIdx := len(merged) - 1
		last := merged[lastIdx]
		if last.End+1 >= g.Start {
			if last.End < g.End {
				merged[lastIdx].End = g.End
			}
			if g.firstSeen.Before(last.firstSeen) {
				merged[lastIdx].firstSeen = g.firstSeen
			}
			if g.lastRequestSent.After(last.lastRequestSent) {
				merged[lastIdx].lastRequestSent = g.lastRequestSent
			}
			if g.timesRequestSent > last.timesRequestSent {
				merged[lastIdx].timesRequestSent = g.timesRequestSent
			}
		} else {
			merged = append(merged, g)
		}
	}
	return merged
}

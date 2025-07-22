package forwarder

import (
	"context"
	"fmt"
	"strings"
)

func ParseForwardingPairs(pairsStr string, ctx context.Context, cancel context.CancelFunc) ([]ForwardingPair, error) {
	if pairsStr == "" {
		return nil, fmt.Errorf("forwarding pairs string is empty")
	}

	pairStrs := strings.Split(pairsStr, ",")
	var connections []ForwardingPair

	for _, pairStr := range pairStrs {
		pairStr = strings.TrimSpace(pairStr)
		if pairStr == "" {
			continue
		}

		parts := strings.Split(pairStr, "|")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid forwarding pair format: %s", pairStr)
		}

		sourceStr := strings.TrimSpace(parts[0])
		sinkStr := strings.TrimSpace(parts[1])

		sourceType := strings.Split(sourceStr, ":")[0]
		sinkType := strings.Split(sinkStr, ":")[0]
		if sinkType == sourceType {
			return nil, fmt.Errorf("source and sink types cannot be the same: %s", pairStr)
		}

		sourceConn, err := parseEndpoint(sourceStr, ctx, cancel)
		if err != nil {
			return nil, fmt.Errorf("invalid source endpoint '%s': %w", sourceStr, err)
		}

		sinkConn, err := parseEndpoint(sinkStr, ctx, cancel)
		if err != nil {
			return nil, fmt.Errorf("invalid sink endpoint '%s': %w", sinkStr, err)
		}

		conn := ForwardingPair{
			source: sourceConn,
			sink:   sinkConn,
		}
		connections = append(connections, conn)
	}
	tables := make(map[string]bool)
	for _, conn := range connections {
		if conn.sink.getType() == "sqlite" {
			tableName := conn.sink.(*sqliteConnector).tableName
			if _, ok := tables[tableName]; ok {
				return nil, fmt.Errorf("sink table '%s' already used in another connection", tableName)
			}
			tables[tableName] = true
		}
	}

	return connections, nil
}

func parseEndpoint(endpointStr string, ctx context.Context, cancel context.CancelFunc) (connection, error) {
	parts := strings.SplitN(endpointStr, ":", 2)
	if len(parts) < 2 || parts[1] == "" {
		return nil, fmt.Errorf("invalid endpoint format: %s", endpointStr)
	}

	endpointType := parts[0]
	endpointArgsStr := parts[1]

	switch endpointType {
	case "sqlite":
		args := strings.SplitN(endpointArgsStr, ":", 2)
		if len(args) != 2 || args[0] == "" || args[1] == "" {
			return nil, fmt.Errorf("invalid sqlite endpoint format: %s. Expected 'sqlite:db_file:table'", endpointStr)
		}
		return newSQLiteConnector(args[0], args[1])
	case "libp2p":
		if strings.Contains(endpointArgsStr, ":") {
			return nil, fmt.Errorf("invalid libp2p topic format: %s. Topic should not contain ':'", endpointStr)
		}
		return newLibP2PConnector(endpointArgsStr, ctx, cancel), nil
	default:
		return nil, fmt.Errorf("unknown endpoint type: %s", endpointType)
	}
}

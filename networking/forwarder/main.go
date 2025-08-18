package main

import (
	"context"
	"flag"
	forwarder "forwarder/src"
	"log"
	"os"
	"os/signal"
	"syscall"
)

var nodeID = flag.String("node-id", "", "Node ID (defaults to FORWARDER_NODE_ID env var or a new UUID)")
var eventsDBPath = flag.String("events-db", "", "Path to the worker events SQLite database")

var SourceHash = "dev"

func main() {
	flag.Parse()

	log.Printf("SourceHash: %s\n", SourceHash)

	os.Setenv("SOURCE_HASH", SourceHash)

	id := *nodeID
	if id != "" {
		forwarder.SetNodeId(id)
	} else {
		id = forwarder.GetNodeId()
	}
	log.Printf("Starting forwarder with node ID: %s", id)

	// Set the events database path if provided
	if *eventsDBPath != "" {
		forwarder.SetEventsDBPath(*eventsDBPath)
		log.Printf("Using events database: %s", *eventsDBPath)
	}

	args := flag.Args()
	if len(args) == 0 {
		log.Fatal("forwarding pairs argument is required as the first positional argument (of the form {source}|{sink}) where source and sink sqlite:db_file:table_name or libp2p:topic")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	forwardingPairs := args[0]
	connections, err := forwarder.ParseForwardingPairs(forwardingPairs, ctx, cancel)
	if err != nil {
		log.Fatalf("Failed to parse forwarding pairs: %v", err)
	}
	for _, conn := range connections {
		log.Printf("Forwarding Pair %v", conn)
	}

	for _, conn := range connections {
		fwd, err := forwarder.NewForwarder(conn)
		if err != nil {
			log.Fatalf("Failed to create forwarder: %v", err)
		}
		fwd.Start(ctx)
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		cancel()
	}()

	<-ctx.Done()
	log.Println("Forwarder is shutting down...")
}

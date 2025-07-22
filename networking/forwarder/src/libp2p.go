package forwarder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"log"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/pnet"
	mdns "github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/libp2p/go-libp2p/p2p/security/noise"
)

var node host.Host
var ps *pubsub.PubSub
var mdnsSer mdns.Service
var once sync.Once
var mu sync.Mutex
var refCount int
var topicsMap = make(map[string]*pubsub.Topic)

type discoveryNotifee struct {
	h host.Host
}

func (n *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	if n.h.ID() >= pi.ID {
		return
	}
	if n.h.Network().Connectedness(pi.ID) == network.Connected {
		return
	}
	ctx := context.Background()
	if err := n.h.Connect(ctx, pi); err != nil {
		log.Printf("Failed to connect to %s: %v", pi.ID.String(), err)
	} else {
		log.Printf("Connected to %s", pi.ID.String())
	}
}

func getPrivKey(nodeId string) (crypto.PrivKey, error) {
	seed := sha256.Sum256([]byte(nodeId))
	priv, _, err := crypto.GenerateEd25519Key(bytes.NewReader(seed[:]))
	if err != nil {
		return nil, err
	}
	return priv, nil
}

func getNode(ctx context.Context) {
	once.Do(func() {
		nodeId := GetNodeId()
		var opts []libp2p.Option
		priv, err := getPrivKey(nodeId)
		if err != nil {
			log.Fatalf("failed to generate key: %v", err)
		}
		opts = append(opts, libp2p.Identity(priv))
		opts = append(opts, libp2p.Security(noise.ID, noise.New))

		pskHash := sha256.Sum256([]byte("forwarder_network"))
		psk := pnet.PSK(pskHash[:])
		opts = append(opts, libp2p.PrivateNetwork(psk))

		// Performance optimizations
		opts = append(opts, libp2p.ConnectionManager(nil)) // No connection limits
		opts = append(opts, libp2p.EnableHolePunching())   // Better NAT traversal
		opts = append(opts, libp2p.EnableRelay())          // Allow relaying

		node, err = libp2p.New(opts...)
		if err != nil {
			log.Fatalf("failed to create host: %v", err)
		}

		// Configure GossipSub for better performance
		gossipOpts := []pubsub.Option{
			pubsub.WithMessageSigning(false),              // Disable message signing for speed
			pubsub.WithStrictSignatureVerification(false), // Disable signature verification
			pubsub.WithMaxMessageSize(1024 * 1024),        // 1MB max message size for batches
			pubsub.WithValidateQueueSize(1000),            // Larger validation queue
			pubsub.WithPeerOutboundQueueSize(1000),        // Larger peer queues
		}

		ps, err = pubsub.NewGossipSub(ctx, node, gossipOpts...)
		if err != nil {
			node.Close()
			log.Fatalf("failed to create pubsub: %v", err)
		}

		rendezvous := "forwarder_network"
		notifee := &discoveryNotifee{h: node}
		mdnsSer = mdns.NewMdnsService(node, rendezvous, notifee)
		if err := mdnsSer.Start(); err != nil {
			node.Close()
			log.Fatalf("failed to start mdns service: %v", err)
		}
	})
}

type libP2PConnector struct {
	topic     string
	sub       *pubsub.Subscription
	subResend *pubsub.Subscription
	top       *pubsub.Topic
	topResend *pubsub.Topic
	ctx       context.Context
	cancel    context.CancelFunc

	// Async publishing
	writeChan    chan RecordData
	batchSize    int
	batchTimeout time.Duration
	workerPool   int
}

func newLibP2PConnector(topic string, ctx context.Context, cancel context.CancelFunc) *libP2PConnector {
	getNode(ctx)
	mu.Lock()
	var err error
	t, ok := topicsMap[topic]
	if !ok {
		t, err = ps.Join(topic)
		if err != nil {
			mu.Unlock()
			log.Fatalf("failed to join topic %s: %v", topic, err)
		}
		topicsMap[topic] = t
	}

	t2, okResend := topicsMap[topic+"/resend"]
	if !okResend {
		t2, err = ps.Join(topic + "/resend")
		if err != nil {
			mu.Unlock()
			log.Fatalf("failed to join topic %s: %v", topic+"/resend", err)
		}
		topicsMap[topic+"/resend"] = t2
	}

	refCount++
	mu.Unlock()

	connector := &libP2PConnector{
		topic:        topic,
		top:          t,
		topResend:    t2,
		ctx:          ctx,
		cancel:       cancel,
		writeChan:    make(chan RecordData, 2000),
		batchSize:    100,
		batchTimeout: 10 * time.Millisecond,
		workerPool:   5,
	}

	connector.startAsyncPublishers()

	return connector
}

func (c *libP2PConnector) tail(handler func(record RecordData) error) {
	sub, err := c.top.Subscribe()
	if err != nil {
		log.Fatalf("failed to subscribe to topic %s: %v", c.topic, err)
	}
	c.sub = sub
	go handleRecordSub(c.sub, c.ctx, handler)
}

func (c *libP2PConnector) tailResend(handler func(data ResendRequest) error) {
	sub, err := c.topResend.Subscribe()
	if err != nil {
		log.Fatalf("failed to subscribe to topic %s: %v", c.topic, err)
	}
	c.subResend = sub
	go handleSub(c.subResend, c.ctx, handler)
}

func handleSub[T any](sub *pubsub.Subscription, ctx context.Context, handler func(data T) error) {
	for {
		msg, err := sub.Next(ctx)
		if err != nil {
			if err == context.Canceled {
				return
			}
			log.Printf("subscription error for topic %s: %v", sub.Topic(), err)
			return
		}
		var rec T
		err = json.Unmarshal(msg.Data, &rec)
		if err != nil {
			log.Printf("unmarshal error for topic %s: %v", sub.Topic(), err)
			continue
		}
		if handler != nil {
			if err := handler(rec); err != nil {
				log.Printf("handler error for topic %s: %v", sub.Topic(), err)
			}
		}
	}
}

func handleRecordSub(sub *pubsub.Subscription, ctx context.Context, handler func(record RecordData) error) {
	for {
		msg, err := sub.Next(ctx)
		if err != nil {
			if err == context.Canceled {
				return
			}
			log.Printf("subscription error for topic %s: %v", sub.Topic(), err)
			return
		}

		// Try to unmarshal as batch first
		var batch BatchRecord
		if err := json.Unmarshal(msg.Data, &batch); err == nil && len(batch.Records) > 0 {
			// Handle batched records
			for _, record := range batch.Records {
				if handler != nil {
					if err := handler(record); err != nil {
						log.Printf("handler error for batched record: %v", err)
					}
				}
			}
			continue
		}

		// Try to unmarshal as single record (backwards compatibility)
		var record RecordData
		if err := json.Unmarshal(msg.Data, &record); err == nil {
			if handler != nil {
				if err := handler(record); err != nil {
					log.Printf("handler error for single record: %v", err)
				}
			}
			continue
		}

		log.Printf("failed to unmarshal message as batch or single record for topic %s", sub.Topic())
	}
}

func (c *libP2PConnector) startAsyncPublishers() {
	// Start worker pool for batched async publishing
	for i := 0; i < c.workerPool; i++ {
		go c.publishWorker()
	}
}

func (c *libP2PConnector) publishWorker() {
	batch := make([]RecordData, 0, c.batchSize)
	timer := time.NewTimer(c.batchTimeout)
	timer.Stop()

	for {
		select {
		case <-c.ctx.Done():
			// Flush final batch
			if len(batch) > 0 {
				err := c.publishBatch(batch)
				if err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
			}
			return

		case record := <-c.writeChan:
			batch = append(batch, record)

			// Check if we should flush
			if len(batch) >= c.batchSize {
				err := c.publishBatch(batch)
				if err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
				batch = batch[:0]
				timer.Stop()
			} else if len(batch) == 1 {
				// First record in batch, start timer
				timer.Reset(c.batchTimeout)
			}

		case <-timer.C:
			// Timer expired, flush whatever we have
			if len(batch) > 0 {
				err := c.publishBatch(batch)
				if err != nil {
					log.Printf("Error publishing batch: %v", err)
				}
				batch = batch[:0]
			}
		}
	}
}

func (c *libP2PConnector) publishBatch(records []RecordData) error {
	if len(records) == 0 {
		return nil
	}

	// Create batch record
	batchRecord := BatchRecord{Records: records}

	data, err := json.Marshal(batchRecord)
	if err != nil {
		return err
	}

	// Publish with timeout to prevent blocking
	go func() {
		pubCtx, pubCancel := context.WithTimeout(c.ctx, 100*time.Millisecond)
		defer pubCancel()

		if err := c.top.Publish(pubCtx, data); err != nil {
			if err != context.DeadlineExceeded {
				log.Printf("Error publishing batch of %d records: %v", len(records), err)
			}
		}
	}()
	return nil
}

func (c *libP2PConnector) write(record RecordData) error {
	select {
	case c.writeChan <- record:
		return nil
	case <-c.ctx.Done():
		return c.ctx.Err()
	default:
		// Channel full, try to publish directly
		return c.publishSingle(record)
	}
}

func (c *libP2PConnector) publishSingle(record RecordData) error {
	if c.top == nil {
		return context.Canceled
	}
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	return c.top.Publish(c.ctx, data)
}

func (c *libP2PConnector) writeResend(req ResendRequest) error {
	if c.topResend == nil {
		return context.Canceled
	}
	data, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return c.topResend.Publish(c.ctx, data)
}

func (c *libP2PConnector) close() error {
	mu.Lock()
	refCount--
	closeHost := refCount == 0
	mu.Unlock()

	if c.cancel != nil {
		c.cancel()
	}
	if c.sub != nil {
		c.sub.Cancel()
	}
	if c.subResend != nil {
		c.subResend.Cancel()
	}
	if closeHost {
		// close all topics when shutting down host
		for _, top := range topicsMap {
			_ = top.Close()
		}
		topicsMap = make(map[string]*pubsub.Topic)
	}

	c.top = nil

	if !closeHost {
		return nil
	}

	if mdnsSer != nil {
		_ = mdnsSer.Close()
		mdnsSer = nil
	}

	var err error
	if node != nil {
		err = node.Close()
	}

	node = nil
	ps = nil
	refCount = 0
	once = sync.Once{}

	return err
}

func (c *libP2PConnector) getType() string {
	return "libp2p"
}

package dm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"log"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pquic "github.com/libp2p/go-libp2p/p2p/transport/quic"

	"github.com/libp2p/go-libp2p"
)

func genPriv(t *testing.T, seed [32]byte) crypto.PrivKey {
	priv, _, err := crypto.GenerateEd25519Key(bytes.NewReader(seed[:]))
	if err != nil {
		t.Fatalf("failed generating key from seed %v: %v", seed, err)
	}
	return priv
}

func createTestHost(t *testing.T, name string, opts ...Option) (host.Host, DirectMessenger) {
	// generate key
	seed := sha256.Sum256([]byte(name))
	id := genPriv(t, seed)

	// create host
	h, err := libp2p.New(
		libp2p.Identity(id),
		libp2p.Transport(libp2pquic.NewTransport),
		libp2p.ListenAddrStrings(
			"/ip4/0.0.0.0/udp/0/quic-v1",
		),
	)
	if err != nil {
		t.Fatalf("failed creating test host '%v': %v", name, err)
	}

	// configure direct messaging
	dmOpts := []Option{WithHandler(&MessageHandlerBundle{
		OnMessageF: func(ctx context.Context, from peer.ID, msg []byte) error {
			log.Printf("[%v]<-[%v]: [%v]%v", name, from, len(msg), msg)
			return nil
		},
	})}
	dmOpts = append(dmOpts, opts...)
	dm, err := NewDirectMessenger(h, dmOpts...)
	if err != nil {
		t.Fatalf("failed creating test DM manager for host '%v': %v", name, err)
	}

	return h, dm
}

func createConnection(t *testing.T, p1, p2 host.Host) {
	ctx := context.Background()
	if err := p1.Connect(ctx, p2.Peerstore().PeerInfo(p2.ID())); err != nil {
		t.Fatalf("failed connecting '%v' to '%v': %v", p1.ID(), p2.ID(), err)
	}
}

func TestJsonEncoder(t *testing.T) {
	peer1, dm1 := createTestHost(t, "peer 1")
	defer dm1.Close()
	defer peer1.Close()

	peer2, dm2 := createTestHost(t, "peer 2")
	defer dm2.Close()
	defer peer2.Close()

	createConnection(t, peer1, peer2)

	if err := dm1.Send(peer2.ID(), make([]byte, 10)); err != nil {
		t.Fatalf("dm1 Send failed: %v", err)
	}

	// big send
	if err := dm2.Send(peer1.ID(), make([]byte, 10_000)); err != nil {
		t.Fatalf("dm2 Send failed: %v", err)
	}
	time.Sleep(500 * time.Millisecond)
	t.Fail()
}

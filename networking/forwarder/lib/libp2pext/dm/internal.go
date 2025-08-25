package dm

import (
	"context"
	"encoding/binary"
	"io"
	"lib"
	"sync"

	logging "github.com/ipfs/go-log/v2"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/proto"
)

const (
	uint64NumBytes = 8
)

var (
	logger = logging.Logger(ServiceName)
)

type directMessenger struct {
	ctx    context.Context
	cancel func()

	h       host.Host
	pid     protocol.ID
	handler MessageHandler
	log     *logging.ZapEventLogger

	scope    network.ResourceScopeSpan
	notifiee network.Notifiee

	mx     sync.Mutex
	closed bool
}

func newDirectMessenger(cfg *Config) (*directMessenger, error) {
	ctx, cancel := context.WithCancel(context.Background())
	dm := &directMessenger{
		ctx:    ctx,
		cancel: cancel,

		h:       cfg.Host,
		pid:     cfg.Protocol,
		handler: cfg.MessageHandler,
		log:     cfg.Logger,
	}

	// get a scope for memory reservations at service level
	err := dm.h.Network().ResourceManager().ViewService(ServiceName,
		func(s network.ServiceScope) error {
			var err error
			dm.scope, err = s.BeginSpan()
			return err
		})
	if err != nil {
		return nil, err
	}

	dm.h.SetStreamHandler(dm.pid, dm.handleStream)
	dm.notifiee = &network.NotifyBundle{} // TODO: add handler funcions in the future if so needed??
	dm.h.Network().Notify(dm.notifiee)

	return dm, nil
}

func (dm *directMessenger) Close() error {
	dm.mx.Lock()
	if !dm.closed {
		dm.closed = true
		dm.mx.Unlock()

		dm.h.RemoveStreamHandler(proto.ProtoIDv2Hop)
		dm.h.Network().StopNotify(dm.notifiee)
		defer dm.scope.Done()
		dm.cancel()
		return nil
	}
	dm.mx.Unlock()
	return nil
}

func (dm *directMessenger) Send(p peer.ID, msg []byte) error {
	dm.log.Infof("outgoing DM stream to: %s", p)

	// create new stream
	s, err := dm.h.NewStream(dm.ctx, p, dm.pid)
	if err != nil {
		return err
	}
	defer s.Close()

	// grab length if byte-buffer and encode it as big-endian
	mLen := len(msg)
	buf := make([]byte, uint64NumBytes, uint64NumBytes+mLen) // allocate enough capacity
	binary.BigEndian.PutUint64(buf, uint64(mLen))
	buf = append(buf, msg...)
	lib.Assert(len(buf) == uint64NumBytes+mLen, "literally what????")

	// write to stream & handle any potential errors
	if _, err := s.Write(buf); err != nil {
		dm.log.Debugf("error writing message to DM service stream: %s", err)
		s.Reset()
		return err
	}

	_ = s.CloseWrite() // signal EOF to caller if half-close is supported
	return nil
}

func (dm *directMessenger) handleStream(s network.Stream) {
	dm.log.Infof("incoming DM stream from: %s", s.Conn().RemotePeer())

	defer s.Close()

	// attach scope to this service (for scoped capacity allocation reasons)
	if err := s.Scope().SetService(ServiceName); err != nil {
		dm.log.Debugf("error attaching stream to DM service: %s", err)
		s.Reset()
		return
	}

	// read big-endian length bytes & decode
	buf := make([]byte, uint64NumBytes)
	if _, err := io.ReadFull(s, buf); err != nil {
		dm.log.Debugf("error reading message length from DM service stream: %s", err)
		s.Reset()
		return
	}
	mLen := binary.BigEndian.Uint64(buf)

	// read rest of message & call OnMessage callback
	buf = make([]byte, mLen)
	if _, err := io.ReadFull(s, buf); err != nil {
		dm.log.Debugf("error reading message body from DM service stream: %s", err)
		s.Reset()
		return
	}
	if err := dm.handler.OnMessage(dm.ctx, s.Conn().RemotePeer(), buf); err != nil {
		dm.log.Debugf("error handling incoming message from DM service stream: %s", err)
		s.Reset()
		return
	}

	_ = s.CloseWrite() // signal EOF to caller if half-close is supported
}

package dm

import (
	"context"
	"errors"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

const (
	ServiceName = "libp2p.ext.dm/v1"
	DmProtocol  = protocol.ID("/dm/1.0.0")
)

var (
	ErrMissingHandler = errors.New("the message handler is missing")
)

type MessageHandler interface {
	OnMessage(ctx context.Context, from peer.ID, msg []byte) error
}

type MessageHandlerBundle struct {
	OnMessageF func(ctx context.Context, from peer.ID, msg []byte) error
}

func (m *MessageHandlerBundle) OnMessage(ctx context.Context, from peer.ID, msg []byte) error {
	return m.OnMessageF(ctx, from, msg)
}

type DirectMessenger interface {
	Send(to peer.ID, msg []byte) error
	Close() error
}

func NewDirectMessenger(h host.Host, opts ...Option) (DirectMessenger, error) {
	cfg := &Config{
		Host:     h,
		Protocol: DmProtocol,
		Logger:   logger,
	}

	// apply all configs
	for _, o := range opts {
		if err := o(cfg); err != nil {
			return nil, err
		}
	}
	if cfg.MessageHandler == nil {
		return nil, ErrMissingHandler
	}

	// create DM from config
	return newDirectMessenger(cfg)
}

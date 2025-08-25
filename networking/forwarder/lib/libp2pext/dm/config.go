package dm

import (
	"context"

	logging "github.com/ipfs/go-log/v2"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

type Config struct {
	Host           host.Host
	Protocol       protocol.ID
	MessageHandler MessageHandler
	Logger         *logging.ZapEventLogger
}

type Option func(c *Config) error // TODO: add more options ??

func WithHandler(h MessageHandler) Option {
	return func(c *Config) error {
		c.MessageHandler = h
		return nil
	}
}
func WithHandlerFunction(onMessage func(ctx context.Context, from peer.ID, msg []byte) error) Option {
	return func(c *Config) error {
		c.MessageHandler = &MessageHandlerBundle{OnMessageF: onMessage}
		return nil
	}
}
func WithLogger(l *logging.ZapEventLogger) Option {
	return func(c *Config) error {
		c.Logger = l
		return nil
	}
}

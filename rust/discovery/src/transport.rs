use crate::alias::AnyResult;
use futures::{AsyncRead, AsyncWrite};
use keccak_const::Sha3_256;
use libp2p::{
    core::{muxing, transport::Boxed}, identity,
    noise,
    pnet, quic, yamux, PeerId, Transport as _,
};
use std::any::Any;

/// Key used for discovery's private network. See [`pnet_upgrade`] for more.
const PNET_PRESHARED_KEY: [u8; 32] = Sha3_256::new().update(b"exo_discovery_network").finalize();

/// Make `discovery` run on a private network, as to not clash with the `forwarder` network.
/// This is implemented as an additional "upgrade" ontop of existing [`libp2p::Transport`] layers.
fn pnet_upgrade<Socket>(
    socket: Socket,
    _ignored: impl Any,
) -> impl Future<Output = Result<pnet::PnetOutput<Socket>, pnet::PnetError>>
where
    Socket: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    pnet::PnetConfig::new(pnet::PreSharedKey::new(PNET_PRESHARED_KEY)).handshake(socket)
}

/// TCP/IP transport layer configuration.
fn tcp_transport(
    keypair: &identity::Keypair,
) -> AnyResult<Boxed<(PeerId, muxing::StreamMuxerBox)>> {
    use libp2p::{
        core::upgrade::Version,
        tcp::{tokio, Config},
    };

    // `TCP_NODELAY` enabled => avoid latency
    let tcp_config = Config::default().nodelay(true);

    // V1 + lazy flushing => 0-RTT negotiation
    let upgrade_version = Version::V1Lazy;

    // Noise is faster than TLS + we don't care much for security
    let noise_config = noise::Config::new(keypair)?;
    //let tls_config = tls::Config::new(keypair)?; // TODO: add this in if needed?? => look into how `.with_tcp` does it...

    // Use default Yamux config for multiplexing
    let yamux_config = yamux::Config::default();

    // Create new Tokio-driven TCP/IP transport layer
    let base_transport = tokio::Transport::new(tcp_config)
        .and_then(pnet_upgrade)
        .upgrade(upgrade_version)
        .authenticate(noise_config)
        .multiplex(yamux_config);

    // Return boxed transport (to flatten complex type)
    Ok(base_transport.boxed())
}

/// QUIC transport layer configuration.
fn quic_transport(keypair: &identity::Keypair) -> Boxed<(PeerId, quic::Connection)> {
    use libp2p::quic::{tokio, Config};

    let quic_config = Config::new(keypair);
    let base_transport = tokio::Transport::new(quic_config).boxed();
    //.and_then(); // As of now, QUIC doesn't support PNet's.., ;( TODO: figure out in future how to do
    unimplemented!("you cannot use this yet !!!");
    base_transport
}

/// Overall composed transport-layer configuration for the `discovery` network.
pub fn discovery_transport(
    keypair: &identity::Keypair,
) -> AnyResult<Boxed<(PeerId, muxing::StreamMuxerBox)>> {
    // TODO: when QUIC is figured out with PNET, re-enable this
    // Ok(tcp_transport(keypair)?
    //     .or_transport(quic_transport(keypair))
    //     .boxed())

    tcp_transport(keypair)
}

//! This crate defines the logic of, and ways to interact with, Exo's **_System Custodian_** daemon.
//!
//! The **_System Custodian_** daemon is supposed to be a long-living process that precedes the
//! launch of the Exo application, and responsible for ensuring the system (configuration, settings,
//! etc.) is in an appropriate state to facilitate the running of Exo application.
//! The **_System Custodian_** daemon shall expose a [D-Bus](https://www.freedesktop.org/wiki/Software/dbus/)
//! service which Exo application use to _control & query_ it.
//!
//! # Lifecycle
//! When the Exo application starts, it will _wake_ the **_System Custodian_** daemon for the
//! duration of its lifetime, and after it has terminated the daemon will go back to sleep. When
//! the daemon wakes up, it will configure the system into a state suitable for the Exo Application;
//! When the daemon goes to sleep, it will revert those changes as much as it can in case they were
//! destructive to the user's pre-existing configurations.
//!
//! # Responsibilities
//! TODO: these are purely on MacOS, but change to be more broad
//! The **_System Custodian_** daemon is responsible for using System Configuration framework to
//!  1. duplicate the current network set
//!  2. modify existing services to turn on IPv6 if not there
//!  3. remove any bridge services & add any missing services that AREN'T bridge
//! TODO: In the future:
//!  1. run a dummy AWDL service to [allow for macOS peer-to-peer wireless networking](https://yggdrasil-network.github.io/2019/08/19/awdl.html)
//!  2. toggle some GPU/memory configurations to speed up GPU (ask Alex what those configurations are)
//!  3. if we ever decide to provide our **own network interfaces** that abstract over some userland
//!     logic, this would be the place to spin that up.
//!
//! Then it will watch the SCDynamicStore for:
//!  1. all __actual__ network interfaces -> collect information on them e.g. their BSD name, MAC
//!     address, MTU, IPv6 addresses, etc. -> and set up watchers/notifiers to inform the DBus
//!     interface of any changes
//!  2. watch for any __undesirable__ changes to configuration and revert it
//!
//! It should somehow (probably through system sockets and/or BSD interface) trigger IPv6 NDP on
//! each of the interfaces & also listen to/query for any changes on the OS routing cache??
//! Basically emulate the `ping6 ff02::1%enX` and `ndp -an` commands BUT BETTER!!!
//!  1. all that info should coalesce back to the overall state colleted -> should be queryable
//!     over D-Bus
//! TODO:
//!  1. we might potentially add to this step a handshake of some kind...? To ensure that we can
//!     ACTUALLY communicate with that machine over that link over e.g. TCP, UDP, etc. Will the
//!     handshake require to know Node ID? Will the handshake require heartbeats? Who knows...
//!  2. if we ever decide to write proprietary L2/L3 protocols for quicker communication,
//!     e.g. [AF_NDRV](https://www.zerotier.com/blog/how-zerotier-eliminated-kernel-extensions-on-macos/)
//!     for raw ethernet frame communication, or even a [custom thunderbolt PCIe driver](https://developer.apple.com/documentation/pcidriverkit/creating-custom-pcie-drivers-for-thunderbolt-devices),
//!     then this would be the place to carry out discovery and propper handshakes with devices
//!     on the other end of the link.
//!

// enable Rust-unstable features for convenience
#![feature(trait_alias)]
#![feature(stmt_expr_attributes)]
#![feature(type_alias_impl_trait)]
#![feature(specialization)]
#![feature(unboxed_closures)]
#![feature(const_trait_impl)]
#![feature(fn_traits)]

pub(crate) mod private {
    // sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {}

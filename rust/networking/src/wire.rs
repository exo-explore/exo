//! Which nodes may connect: the same cluster name, the same wire version and the same build.
//!
//! The wire version is exo's `WIRE_VERSION` (src/exo/routing/wire.py): nodes whose message
//! schemas differ can't read each other. Nodes of different builds can disagree inside a model,
//! between its ranks (MLX, mlx-lm and exo's sharding code all change between builds), and hang
//! part way through it or give wrong tokens. Such nodes must never connect.
//!
//! A discovery Hello carries 8 bytes that say which cluster its sender is in and what it runs:
//! - bytes 0..3: the first 3 bytes of blake3 of the cluster name (the namespace);
//! - bytes 3..5: the wire version, little-endian;
//! - bytes 5..8: the first 3 bytes of blake3 of the build id.
//!
//! Nodes connect only when all three are equal. Two different cluster names, or two different
//! builds, share 3 bytes with probability 2^-24 (1 in 16.8 million) per pair. Before wire
//! versions, exo sent the first 8 bytes of blake3 of the namespace, which is told apart.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

/// What a discovery Hello carries to say which cluster its sender is in.
pub type NamespaceHash = [u8; 8];

/// How long a device on another version stays listed after its last Hello. A node says Hello
/// every second, so this is 30 Hellos missed in a row: the device quit or now runs our version.
/// A busy node that pauses for a few seconds, or a Wi-Fi link dropping a few packets, doesn't
/// make the notice come and go.
pub const OTHER_VERSION_FORGET_AFTER: Duration = Duration::from_secs(30);

/// The most addresses listed at once. A home or office network has a few exo devices on other
/// versions at most, each heard on a few of its addresses (one per network interface). Past
/// this the address heard longest ago is dropped, so Hellos from forged addresses can't grow
/// the list without bound.
pub const MAX_OTHER_VERSION_ADDRESSES: usize = 64;

/// Why a device in our cluster can't join it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PeerVersion {
    /// An exo from before wire versions.
    Unversioned,
    /// Another wire version.
    Wire(u16),
    /// Our wire version, but another build.
    Build,
}

impl fmt::Display for PeerVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unversioned => write!(f, "an exo from before wire versions"),
            Self::Wire(version) => write!(f, "wire version {version}"),
            Self::Build => write!(f, "the same wire version but another build"),
        }
    }
}

/// What a Hello says about its sender, compared with us.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Peer {
    /// Our cluster, our wire version and our build: connect.
    Compatible,
    /// Our cluster, but it can't join.
    OtherVersion(PeerVersion),
    /// Another cluster name (or something that isn't exo).
    OtherCluster,
}

fn prefix<const N: usize>(text: &str) -> [u8; N] {
    blake3::hash(text.as_bytes()).as_bytes()[..N]
        .try_into()
        .expect("N is at most 32")
}

/// This node's cluster and build, as its Hellos say them.
pub struct ClusterKey {
    hash: NamespaceHash,
    /// What exos from before wire versions announce in our cluster
    unversioned: Vec<NamespaceHash>,
    wire_version: u16,
    build_id: String,
}

impl ClusterKey {
    /// `pre_gate_namespace`: another name that exos from before wire versions used for this
    /// cluster (their package version, on the default name), to recognise them by.
    pub fn new(
        namespace: &str,
        wire_version: u16,
        build_id: &str,
        pre_gate_namespace: Option<&str>,
    ) -> Self {
        let mut hash = [0; 8];
        hash[..3].copy_from_slice(&prefix::<3>(namespace));
        hash[3..5].copy_from_slice(&wire_version.to_le_bytes());
        hash[5..].copy_from_slice(&prefix::<3>(build_id));
        let unversioned = std::iter::once(namespace)
            .chain(pre_gate_namespace)
            .map(prefix::<8>)
            .collect();
        Self {
            hash,
            unversioned,
            wire_version,
            build_id: build_id.to_owned(),
        }
    }

    /// What this node announces.
    pub const fn hash(&self) -> NamespaceHash {
        self.hash
    }

    /// This node's wire version and build, as a log line says them.
    pub fn describe(&self) -> String {
        format!(
            "wire version {}, build {}",
            self.wire_version, self.build_id
        )
    }

    pub fn classify(&self, theirs: &NamespaceHash) -> Peer {
        if self.unversioned.contains(theirs) {
            return Peer::OtherVersion(PeerVersion::Unversioned);
        }
        if theirs[..3] != self.hash[..3] {
            return Peer::OtherCluster;
        }
        let wire_version = u16::from_le_bytes([theirs[3], theirs[4]]);
        if wire_version != self.wire_version {
            return Peer::OtherVersion(PeerVersion::Wire(wire_version));
        }
        if theirs[5..] == self.hash[5..] {
            Peer::Compatible
        } else {
            Peer::OtherVersion(PeerVersion::Build)
        }
    }
}

/// A device heard on the network that can't join: every address it was heard on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceOnOtherVersion {
    pub addresses: Vec<IpAddr>,
    pub version: PeerVersion,
}

struct Heard {
    version: PeerVersion,
    /// The last two Hellos heard at this address. A node sends each Hello on all its
    /// interfaces at once, so addresses that heard the same Hello are one device.
    hellos: [[u8; 8]; 2],
    at: Instant,
}

/// The devices discovery heard in our cluster on another version.
#[derive(Clone, Default)]
pub struct OtherVersions(Arc<Mutex<HashMap<IpAddr, Heard>>>);

impl OtherVersions {
    pub fn heard(&self, address: IpAddr, version: PeerVersion, hello: [u8; 8], now: Instant) {
        let mut heard = self.0.lock();
        forget_old(&mut heard, now);
        let previous = heard.get(&address).map_or(hello, |entry| entry.hellos[0]);
        heard.insert(
            address,
            Heard {
                version,
                hellos: [hello, previous],
                at: now,
            },
        );
        while heard.len() > MAX_OTHER_VERSION_ADDRESSES {
            let oldest = heard
                .iter()
                .min_by_key(|(_, entry)| entry.at)
                .map(|(address, _)| *address)
                .expect("not empty");
            heard.remove(&oldest);
        }
    }

    /// The devices heard within `OTHER_VERSION_FORGET_AFTER` of `now`, one per device.
    pub fn current(&self, now: Instant) -> Vec<DeviceOnOtherVersion> {
        let mut heard = self.0.lock();
        forget_old(&mut heard, now);
        let mut addresses: Vec<_> = heard.iter().collect();
        addresses.sort_by_key(|(address, _)| **address);
        // Addresses that heard one of the same Hellos belong to one device
        let mut devices: Vec<(HashSet<[u8; 8]>, DeviceOnOtherVersion)> = Vec::new();
        for (address, entry) in addresses {
            let hellos: HashSet<[u8; 8]> = entry.hellos.into_iter().collect();
            let (same, mut others): (Vec<_>, Vec<_>) = devices
                .into_iter()
                .partition(|(seen, _)| !seen.is_disjoint(&hellos));
            let mut merged = (
                hellos,
                DeviceOnOtherVersion {
                    addresses: vec![*address],
                    version: entry.version,
                },
            );
            for (seen, device) in same {
                merged.0.extend(seen);
                merged.1.addresses.extend(device.addresses);
            }
            merged.1.addresses.sort();
            others.push(merged);
            devices = others;
        }
        let mut devices: Vec<_> = devices.into_iter().map(|(_, device)| device).collect();
        devices.sort_by_key(|device| device.addresses[0]);
        devices
    }
}

fn forget_old(heard: &mut HashMap<IpAddr, Heard>, now: Instant) {
    heard.retain(|_, entry| now.saturating_duration_since(entry.at) < OTHER_VERSION_FORGET_AFTER);
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUILD: &str = "exo 0.3.70, mlx 0.32.4.dev20261030+36969c09, mlx-lm 0.32.3.dev16";

    fn key(namespace: &str, wire_version: u16, build: &str) -> ClusterKey {
        ClusterKey::new(namespace, wire_version, build, None)
    }

    #[test]
    fn the_same_cluster_wire_version_and_build_connect() {
        let ours = key("exo", 1, BUILD);
        assert_eq!(
            ours.classify(&key("exo", 1, BUILD).hash()),
            Peer::Compatible
        );
    }

    #[test]
    fn another_wire_version_is_told_apart_from_another_cluster() {
        let ours = key("studio", 5, BUILD);
        assert_eq!(
            ours.classify(&key("studio", 4, BUILD).hash()),
            Peer::OtherVersion(PeerVersion::Wire(4))
        );
        assert_eq!(
            ours.classify(&key("studio", 6, "another build").hash()),
            Peer::OtherVersion(PeerVersion::Wire(6))
        );
        assert_eq!(
            ours.classify(&key("office", 5, BUILD).hash()),
            Peer::OtherCluster
        );
        assert_eq!(
            ours.classify(&key("office", 4, "another build").hash()),
            Peer::OtherCluster
        );
    }

    #[test]
    fn another_build_does_not_connect() {
        let ours = key("exo", 1, BUILD);
        for other in [
            "exo 0.3.70, mlx 0.32.4.dev20261101+0123abcd, mlx-lm 0.32.3.dev16",
            "exo 0.3.70, mlx 0.32.4, mlx-lm 0.32.3.dev16",
            "exo 0.3.70, mlx none, mlx-lm none",
            "1.0.72 (4f1c2a9)",
        ] {
            assert_eq!(
                ours.classify(&key("exo", 1, other).hash()),
                Peer::OtherVersion(PeerVersion::Build),
                "{other}"
            );
        }
    }

    #[test]
    fn exos_from_before_wire_versions_are_recognised() {
        // What every exo announced until wire versions: the first 8 bytes of blake3
        let unversioned = |namespace: &str| prefix::<8>(namespace);
        // On a name both use
        let ours = key("studio", 1, BUILD);
        assert_eq!(
            ours.classify(&unversioned("studio")),
            Peer::OtherVersion(PeerVersion::Unversioned)
        );
        assert_ne!(ours.hash(), unversioned("studio"));
        // On the default name, whose old form was the package version
        let on_default = ClusterKey::new("exo", 1, BUILD, Some("0.3.70"));
        assert_eq!(
            on_default.classify(&unversioned("0.3.70")),
            Peer::OtherVersion(PeerVersion::Unversioned)
        );
        assert_eq!(
            on_default.classify(&unversioned("0.3.69")),
            Peer::OtherCluster
        );
        assert_eq!(
            key("exo", 1, BUILD).classify(&unversioned("0.3.70")),
            Peer::OtherCluster
        );
    }

    fn address(last: u16) -> IpAddr {
        IpAddr::V6(std::net::Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, last))
    }

    #[test]
    fn forgets_a_device_once_its_hellos_stop() {
        let heard = OtherVersions::default();
        let start = Instant::now();
        heard.heard(address(1), PeerVersion::Wire(2), [1; 8], start);
        assert_eq!(
            heard.current(start + OTHER_VERSION_FORGET_AFTER - Duration::from_millis(1)),
            vec![DeviceOnOtherVersion {
                addresses: vec![address(1)],
                version: PeerVersion::Wire(2)
            }]
        );
        // Each Hello renews it
        let later = start + Duration::from_secs(20);
        heard.heard(address(1), PeerVersion::Wire(2), [2; 8], later);
        assert_eq!(heard.current(start + OTHER_VERSION_FORGET_AFTER).len(), 1);
        assert_eq!(heard.current(later + OTHER_VERSION_FORGET_AFTER), vec![]);
    }

    #[test]
    fn a_device_heard_on_several_addresses_is_one_device() {
        let heard = OtherVersions::default();
        let now = Instant::now();
        // One device's Hello arrives on three of its interfaces, and another device's on one
        for last in 1..=3 {
            heard.heard(address(last), PeerVersion::Wire(2), [7; 8], now);
        }
        heard.heard(address(9), PeerVersion::Build, [8; 8], now);
        // Its next Hello has reached only one address so far: still one device
        heard.heard(address(2), PeerVersion::Wire(2), [9; 8], now);
        assert_eq!(
            heard.current(now),
            vec![
                DeviceOnOtherVersion {
                    addresses: vec![address(1), address(2), address(3)],
                    version: PeerVersion::Wire(2)
                },
                DeviceOnOtherVersion {
                    addresses: vec![address(9)],
                    version: PeerVersion::Build
                },
            ]
        );
    }

    #[test]
    fn the_list_is_bounded_and_pruned_as_it_is_added_to() {
        let heard = OtherVersions::default();
        let start = Instant::now();
        // Forged Hellos from ever more addresses
        for last in 0..(MAX_OTHER_VERSION_ADDRESSES as u16 + 50) {
            let at = start + Duration::from_millis(u64::from(last));
            heard.heard(address(last), PeerVersion::Wire(2), [0; 8], at);
            assert!(heard.0.lock().len() <= MAX_OTHER_VERSION_ADDRESSES);
        }
        // The newest stay
        assert!(
            heard
                .0
                .lock()
                .contains_key(&address(MAX_OTHER_VERSION_ADDRESSES as u16 + 49))
        );
        assert!(!heard.0.lock().contains_key(&address(0)));
        // Old entries go when anything new is heard, without anyone reading the list
        let much_later = start + OTHER_VERSION_FORGET_AFTER + Duration::from_secs(60);
        heard.heard(address(500), PeerVersion::Build, [1; 8], much_later);
        assert_eq!(heard.0.lock().len(), 1);
    }
}

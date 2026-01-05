//! BitTorrent-based download system for model shards using rqbit
//!
//! This crate provides:
//! - Torrent session management via rqbit
//! - Embedded torrent file access
//! - Private tracker announce handling
//! - Selective file download based on shard layer ranges

#![allow(clippy::missing_inline_in_public_items)]

pub mod bencode;
pub mod embedded;
pub mod progress;
pub mod session;
pub mod torrent_files;
pub mod tracker;

pub use bencode::AnnounceParams;
pub use embedded::get_embedded_torrents;
pub use session::{DownloadProgress, TorrentSession};
pub use torrent_files::{get_torrent_file_list, TorrentFileInfo};
pub use tracker::{handle_announce, PeerInfo, TopologyData};

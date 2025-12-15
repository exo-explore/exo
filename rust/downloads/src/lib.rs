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
pub mod tracker;

pub use embedded::get_embedded_torrent;
pub use session::{DownloadProgress, TorrentSession};
pub use tracker::handle_announce;

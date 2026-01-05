//! Torrent file list parsing
//!
//! Provides functionality to extract file information from torrent metadata
//! without adding the torrent to a session.

use anyhow::{Context, Result};
use librqbit::torrent_from_bytes;
use serde::{Deserialize, Serialize};

/// Information about a file in a torrent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorrentFileInfo {
    /// File index (0-based)
    pub index: usize,
    /// File path relative to torrent root
    pub path: String,
    /// File size in bytes
    pub size: u64,
}

/// Get the list of files in a torrent from its raw bytes
///
/// # Arguments
/// * `torrent_data` - Raw .torrent file contents
///
/// # Returns
/// List of file information (index, path, size)
pub fn get_torrent_file_list(torrent_data: &[u8]) -> Result<Vec<TorrentFileInfo>> {
    let torrent_meta = torrent_from_bytes(torrent_data).context("Failed to parse torrent")?;

    // Access the data inside WithRawBytes wrapper
    let info = &torrent_meta.info.data;

    let mut files = Vec::new();

    // Handle both single-file and multi-file torrents
    if let Some(ref file_list) = info.files {
        // Multi-file torrent
        for (index, file) in file_list.iter().enumerate() {
            let path = file
                .path
                .iter()
                .map(|buf| String::from_utf8_lossy(buf.0).to_string())
                .collect::<Vec<_>>()
                .join("/");

            files.push(TorrentFileInfo {
                index,
                path,
                size: file.length,
            });
        }
    } else {
        // Single-file torrent
        let name = match &info.name {
            Some(n) => String::from_utf8_lossy(n.0).to_string(),
            None => String::new(),
        };
        files.push(TorrentFileInfo {
            index: 0,
            path: name,
            size: info.length.unwrap_or(0),
        });
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::get_embedded_torrents;

    #[test]
    fn test_get_torrent_file_list() {
        // Use an embedded torrent for testing
        let torrents = get_embedded_torrents(
            "mlx-community/Qwen3-30B-A3B-4bit",
            "d388dead1515f5e085ef7a0431dd8fadf0886c57",
        );

        assert!(!torrents.is_empty(), "Expected to find embedded torrents");

        for (variant, data) in torrents {
            let files = get_torrent_file_list(&data).expect("Failed to parse torrent");
            assert!(!files.is_empty(), "Expected files in {variant} variant");

            // Verify file info makes sense
            for file in &files {
                assert!(!file.path.is_empty(), "File path should not be empty");
                assert!(file.size > 0, "File size should be positive");
            }

            println!("Variant '{variant}' has {} files", files.len());
            for file in files.iter().take(5) {
                println!("  [{}] {} ({} bytes)", file.index, file.path, file.size);
            }
        }
    }
}

//! Download progress tracking
//!
//! Types for tracking and reporting download progress to Python

use std::collections::HashMap;

/// Progress update for a torrent download
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Total bytes to download
    pub total_bytes: u64,

    /// Bytes downloaded so far
    pub downloaded_bytes: u64,

    /// Number of pieces completed
    pub pieces_completed: usize,

    /// Total number of pieces
    pub total_pieces: usize,

    /// Number of peers connected
    pub peers_connected: usize,

    /// Download speed in bytes/second
    pub speed_bytes_per_sec: f64,

    /// Estimated time remaining in seconds
    pub eta_seconds: Option<f64>,

    /// Per-file progress
    pub files: HashMap<String, FileProgress>,
}

#[derive(Debug, Clone)]
pub struct FileProgress {
    /// Total file size
    pub total_bytes: u64,

    /// Bytes downloaded for this file
    pub downloaded_bytes: u64,

    /// Whether the file is complete
    pub complete: bool,
}

impl DownloadProgress {
    #[inline]
    pub fn new(total_bytes: u64, total_pieces: usize) -> Self {
        Self {
            total_bytes,
            downloaded_bytes: 0,
            pieces_completed: 0,
            total_pieces,
            peers_connected: 0,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
            files: HashMap::new(),
        }
    }

    #[inline]
    pub fn progress_fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let fraction = self.downloaded_bytes as f64 / self.total_bytes as f64;
            fraction
        }
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        self.pieces_completed >= self.total_pieces
    }
}
